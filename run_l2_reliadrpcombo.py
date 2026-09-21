import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reliadrp import ReliaDRPCombo, nig_nll_elem  # noqa: E402

ALPHA = 0.1
TIERS = ["high", "mid", "low"]


def q_conf(s, alpha=ALPHA):
    n = len(s)
    return float(np.quantile(s, (1 - alpha) * (1 + 1.0 / n)))


def picp(y, mu, sig, q, stud):
    lo, hi = (mu - q * sig, mu + q * sig) if stud else (mu - q, mu + q)
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def train(model, tr, va, X, y, w, seed, epochs, batch, env_np=None, n_env=0, eta=0.05,
          beta=1.0):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    dev = next(model.parameters()).device
    best, bad, state = 1e18, 0, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), batch):
            rows = tr[perm[i:i + batch]]
            c, dr, dc = X["c"][rows], X["dr"][rows], X["dc"][rows]
            if env_np is None:
                loss = model.compute_loss(c, dr, dc, y[rows], w[rows], ep, epochs)
            else:
                e = torch.from_numpy(env_np[rows]).to(dev)
                present = torch.unique(e)
                le = torch.stack([model.compute_loss(c[e == k], dr[e == k], dc[e == k],
                                                     y[rows][e == k], w[rows][e == k], ep, epochs)
                                  for k in present])
                loss = le.mean() + beta * le.var(unbiased=False)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            o = model(X["c"][va], X["dr"][va], X["dc"][va])
            rv = float(torch.sqrt(F.mse_loss(o[0], y[va])))
        if rv < best - 1e-5:
            best, bad = rv, 0
            state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    if state is None:
        state = {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    return model


def infer(model, X, idx):
    with torch.no_grad():
        mu, v, a, b = model(X["c"][idx], X["dr"][idx], X["dc"][idx])
        ale = b / (a - 1)
        epi = b / (v * (a - 1))
        return mu.cpu().numpy(), torch.sqrt(ale + epi).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2dir", default=os.path.join(HERE, "data", "processed_ext", "L2"))
    ap.add_argument("--protocols", default="L2_random,L2_LCO,L2_LDO")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--env", action="store_true")
    ap.add_argument("--dro", action="store_true")
    ap.add_argument("--n-env", type=int, default=8)
    ap.add_argument("--dro-eta", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "l2_reliadrpcombo.csv"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = args.l2dir
    resp = pd.read_csv(os.path.join(d, "response_l2.csv"))
    expr = np.load(os.path.join(d, "cells_expr.npy")).astype(np.float32)
    morgan = np.load(os.path.join(d, "drug_morgan.npy")).astype(np.float32)
    ci = resp["cell_idx"].to_numpy(np.int64)
    dri = resp["drow_idx"].to_numpy(np.int64)
    dci = resp["dcol_idx"].to_numpy(np.int64)
    yv = resp["y_css_ri"].to_numpy(np.float32)
    wv = resp["reliability_w"].to_numpy(np.float32)
    tier = resp["reliability_tier"].to_numpy()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    y = torch.from_numpy(yv).to(dev)
    w = torch.from_numpy(wv).to(dev)
    if args.env:
        cntc = np.bincount(ci, minlength=expr.shape[0]).astype(np.float32)
        cntr = np.bincount(dri, minlength=morgan.shape[0]).astype(np.float32)
        cntd = np.bincount(dci, minlength=morgan.shape[0]).astype(np.float32)
        env = (1.0 / np.sqrt(np.maximum(cntc[ci], 1))) * (1.0 / np.sqrt(np.maximum(cntr[dri], 1))) \
            * (1.0 / np.sqrt(np.maximum(cntd[dci], 1)))
        env = (env / env.mean()).astype(np.float32)
        wv = (wv * env).astype(np.float32)
        w = torch.from_numpy(wv).to(dev)
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for P in [p.strip() for p in args.protocols.split(",")]:
        tr = np.load(os.path.join(d, "splits", f"{P}_train.npy"))
        va = np.load(os.path.join(d, "splits", f"{P}_val.npy"))
        te = np.load(os.path.join(d, "splits", f"{P}_test.npy"))
        n_cell = min(384, max(2, len(np.unique(ci[tr])) - 1))
        cpca = PCA(n_cell, random_state=0).fit(expr[np.unique(ci[tr])])
        cd = cpca.transform(expr).astype(np.float32)
        if cd.shape[1] < 384:
            cd = np.hstack([cd, np.zeros((len(cd), 384 - cd.shape[1]), np.float32)])
        dpca = PCA(128, random_state=0).fit(morgan)
        dd = dpca.transform(morgan).astype(np.float32)
        X = {"c": torch.from_numpy(cd[ci]).to(dev),
             "dr": torch.from_numpy(dd[dri]).to(dev),
             "dc": torch.from_numpy(dd[dci]).to(dev)}
        env_np = None
        if args.dro:
            tr_drugs = np.unique(np.concatenate([dri[tr], dci[tr]]))
            km = KMeans(n_clusters=args.n_env, random_state=0, n_init=3).fit(dd[tr_drugs])
            env_all = km.predict(dd).astype(np.int64)
            env_np = env_all[dri]
        for seed in seeds:
            t0 = time.time()
            m = ReliaDRPCombo(d_cell=cd.shape[1], hidden=args.hidden).to(dev)
            m = train(m, tr, va, X, y, w, seed, args.epochs, args.batch,
                      env_np=env_np, n_env=args.n_env, eta=args.dro_eta, beta=args.beta)
            mu_v, sig_v = infer(m, X, va)
            mu_t, sig_t = infer(m, X, te)
            stud = bool(np.std(sig_t) > 1e-8)
            sc = np.abs(yv[va] - mu_v) / (sig_v + 1e-8) if stud else np.abs(yv[va] - mu_v)
            q = q_conf(sc)
            pp, mm = picp(yv[te], mu_t, sig_t, q, stud)
            row = {"protocol": P, "model": "reliadrp", "seed": seed,
                   "rmse": round(float(np.sqrt(np.mean((yv[te] - mu_t) ** 2))), 4),
                   "pearson": round(float(stats.pearsonr(yv[te], mu_t)[0]), 4),
                   "picp": round(pp, 4), "mpiw": round(mm, 4)}
            for t in TIERS:
                mk = tier[te] == t
                if mk.sum() < 10:
                    continue
                p_cp, _ = picp(yv[te][mk], mu_t[mk], sig_t[mk], q, stud)
                mv = tier[va] == t
                qt = q_conf(sc[mv]) if mv.sum() > 5 else q
                p_gc, _ = picp(yv[te][mk], mu_t[mk], sig_t[mk], qt, stud)
                row[f"picp_cp_{t}"] = round(p_cp, 4)
                row[f"picp_gc_{t}"] = round(p_gc, 4)
            rows.append(row)
            print(f"[{P}] reliadrp(combo) s{seed}: rmse={row['rmse']} r={row['pearson']} "
                  f"picp={pp:.3f} cp_high={row.get('picp_cp_high')} gc_high={row.get('picp_gc_high')} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
