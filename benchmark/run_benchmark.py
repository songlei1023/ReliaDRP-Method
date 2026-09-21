import argparse
import inspect
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
sys.path.insert(0, os.path.join(HERE, "core"))  # 库模块在 core/
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")
EXPCODE = os.path.normpath(os.path.join(HERE, "..", "new_data", "exp"))
sys.path.insert(0, EXPCODE)

import model_zoo_2021_2026 as zoo  # noqa: E402

SPL = os.path.join(DATA, "splits")
FEAT = os.path.join(DATA, "feat")
ALPHA = 0.1
TIERS = ["high", "mid", "low"]


def load_multi(P):
    tr = np.load(os.path.join(SPL, f"{P}_train.npy"))
    va = np.load(os.path.join(SPL, f"{P}_val.npy"))
    te = np.load(os.path.join(SPL, f"{P}_test.npy"))
    ci = np.load(os.path.join(SPL, "cell_idx.npy"))
    di = np.load(os.path.join(SPL, "drug_idx.npy"))
    y = np.load(os.path.join(SPL, "y.npy"))
    cells = np.load(os.path.join(FEAT, f"{P}_cells_multi.npy"))
    drugs = np.load(os.path.join(FEAT, f"{P}_drugs.npy"))
    return tr, va, te, ci, di, y, cells, drugs


def load_reliability(path):
    r = pd.read_csv(path, usecols=["reliability_tier", "reliability_w"])
    return r["reliability_tier"].to_numpy(), r["reliability_w"].to_numpy(float)


def call_loss(model, cell, drug, y, w, ctx, ep, epochs):
    if not hasattr(model, "compute_loss"):
        out = model(cell, drug)
        mu = out[0] if isinstance(out, tuple) else out
        loss = F.mse_loss(mu, y, reduction="none")
        return (loss * w).mean() if (w is not None and "w" in ()) else loss.mean()
    sig = inspect.signature(model.compute_loss).parameters
    kw = {}
    if "w" in sig:
        kw["w"] = w
    if "ctx" in sig:
        kw["ctx"] = ctx
    if "ep" in sig:
        kw["ep"] = ep
    if "epochs" in sig:
        kw["epochs"] = epochs
    return model.compute_loss(cell, drug, y, **kw)


def infer(model, ci, di, tc, td, idx):
    with torch.no_grad():
        out = model(tc[ci[idx]], td[di[idx]])
    if isinstance(out, tuple):
        mu, v, a, b = out
        ale = b / (a - 1.0)
        epi = b / (v * (a - 1.0))
        sig = torch.sqrt(ale + epi)
        return mu.cpu().numpy(), sig.cpu().numpy(), ale.cpu().numpy(), epi.cpu().numpy()
    return out.cpu().numpy(), np.ones(len(idx), dtype=np.float32), None, None


def q_conformal(s_cal, alpha=ALPHA):
    n = len(s_cal)
    return float(np.quantile(s_cal, (1 - alpha) * (1 + 1.0 / n)))


def picp_mpiw(y, mu, sig, q, studentized):
    lo, hi = (mu - q * sig, mu + q * sig) if studentized else (mu - q, mu + q)
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def train_one(model, tr, va, ci, di, y, cells, drugs, w, ctx, seed, epochs, batch,
              patience=6):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    dev = next(model.parameters()).device
    tc = torch.from_numpy(cells.astype(np.float32)).to(dev)
    td = torch.from_numpy(drugs.astype(np.float32)).to(dev)
    yt = torch.from_numpy(y.astype(np.float32)).to(dev)
    wt = torch.from_numpy(w.astype(np.float32)).to(dev) if w is not None else None
    ctx_all = torch.from_numpy(ctx.astype(np.int64)).to(dev) if ctx is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, bad = float("inf"), None, 0
    ci_t = torch.from_numpy(ci.astype(np.int64)).to(dev)
    di_t = torch.from_numpy(di.astype(np.int64)).to(dev)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), batch):
            rows = tr[perm[i:i + batch]]
            idx = torch.from_numpy(rows.astype(np.int64)).to(dev)
            c = tc[ci_t[idx]]
            d = td[di_t[idx]]
            yy = yt[idx]
            ww = wt[idx] if wt is not None else None
            cc = ctx_all[idx] if ctx_all is not None else None
            loss = call_loss(model, c, d, yy, ww, cc, ep, epochs)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(tc[ci_t[torch.from_numpy(va.astype(np.int64)).to(dev)]],
                        td[di_t[torch.from_numpy(va.astype(np.int64)).to(dev)]])
            mu_v = out[0] if isinstance(out, tuple) else out
            r = float(torch.sqrt(F.mse_loss(mu_v, yt[torch.from_numpy(va.astype(np.int64)).to(dev)])).item())
        if r < best - 1e-5:
            best, bad = r, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="P0")
    ap.add_argument("--models", default=",".join(zoo.ALL_NAMES))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rel", default=os.path.join(HERE, "reliability", "response_reliability.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "benchmark_results.csv"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    tier_all, w_all = load_reliability(args.rel)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    epochs = 5 if args.quick else args.epochs
    rows = []
    for P in [p.strip() for p in args.protocols.split(",") if p.strip()]:
        tr, va, te, ci, di, y, cells, drugs = load_multi(P)
        if args.quick:
            tr = np.sort(np.random.RandomState(0).choice(tr, min(60000, len(tr)), replace=False))
        ctx = None
        if any(m == "codeae" for m in models):
            from sklearn.cluster import KMeans
            sub = np.random.RandomState(0).choice(tr, min(20000, len(tr)), replace=False)
            km = KMeans(n_clusters=16, random_state=0, n_init=3).fit(cells[ci[sub]])
            ctx = km.predict(cells[ci])[ci].astype(np.int64)
        for name in models:
            for seed in seeds:
                t0 = time.time()
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                model = zoo.build(name).to(dev)
                model = train_one(model, tr, va, ci, di, y, cells, drugs, w_all, ctx,
                                  seed, epochs, args.batch)
                model.eval()
                tc = torch.from_numpy(cells.astype(np.float32)).to(dev)
                td = torch.from_numpy(drugs.astype(np.float32)).to(dev)
                ci_t = torch.from_numpy(ci.astype(np.int64)).to(dev)
                di_t = torch.from_numpy(di.astype(np.int64)).to(dev)
                mu_v, sig_v, _, _ = infer(model, ci_t, di_t, tc, td, va)
                mu_t, sig_t, _, _ = infer(model, ci_t, di_t, tc, td, te)
                stud = bool(np.std(sig_t) > 1e-8)
                s_cal = np.abs(y[va] - mu_v) / (sig_v + 1e-8) if stud else np.abs(y[va] - mu_v)
                q = q_conformal(s_cal)
                picp, mpiw = picp_mpiw(y[te], mu_t, sig_t, q, stud)
                tier_te = tier_all[te]
                row = {"protocol": P, "model": name, "seed": seed,
                       "rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
                       "pearson": round(float(stats.pearsonr(y[te], mu_t)[0]), 4),
                       "r2": round(float(1 - np.sum((y[te] - mu_t) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)), 4),
                       "picp": round(picp, 4), "mpiw": round(mpiw, 4), "q": round(q, 6)}
                if stud:
                    row["spearman_unc"] = round(float(stats.spearmanr(sig_t, np.abs(y[te] - mu_t))[0]), 4)
                for t in TIERS:
                    m = tier_te == t
                    if m.sum() == 0:
                        continue
                    p_cp, _ = picp_mpiw(y[te][m], mu_t[m], sig_t[m], q, stud)
                    qt = q_conformal(s_cal[tier_all[va] == t])
                    p_gc, w_gc = picp_mpiw(y[te][m], mu_t[m], sig_t[m], qt, stud)
                    row[f"picp_cp_{t}"] = round(p_cp, 4)
                    row[f"picp_gc_{t}"] = round(p_gc, 4)
                    row[f"mpiw_gc_{t}"] = round(w_gc, 4)
                rows.append(row)
                print(f"[{P}] {name} seed{seed}: rmse={row['rmse']} r={row['pearson']} "
                      f"picp={picp:.3f} cp_high={row.get('picp_cp_high')} gc_high={row.get('picp_gc_high')} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
