"""Round-2 fast screen for L2 (seed 0, protocols L2_random + L2_LCO).

Selection by val RMSE (same criterion as v2 baseline run). Test Pearson is
printed for information only.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reliadrp import ReliaDRPCombo  # noqa: E402
from reliadrp_v3 import ReliaDRPComboV3, ReliaDRPComboV4  # noqa: E402


def train(model, tr, va, X, y, w, seed, epochs, batch, ema_decay=0.0):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ema = {k: t.detach().clone() for k, t in model.state_dict().items()} \
        if ema_decay > 0 else None
    best, bad, state = 1e18, 0, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), batch):
            rows = tr[perm[i:i + batch]]
            loss = model.compute_loss(X["c"][rows], X["dr"][rows], X["dc"][rows],
                                      y[rows], w[rows], ep, epochs)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    cur = model.state_dict()
                    for k in ema:
                        if cur[k].dtype.is_floating_point:
                            ema[k].mul_(ema_decay).add_(cur[k], alpha=1 - ema_decay)
                        else:
                            ema[k] = cur[k].detach().clone()
        model.eval()
        if ema is not None:
            backup = {k: t.detach().clone() for k, t in model.state_dict().items()}
            model.load_state_dict(ema)
        with torch.no_grad():
            o = model(X["c"][va], X["dr"][va], X["dc"][va])
            rv = float(torch.sqrt(F.mse_loss(o[0], y[va])))
        if ema is not None:
            model.load_state_dict(backup)
        if rv < best - 1e-5:
            best, bad = rv, 0
            src = ema if ema is not None else model.state_dict()
            state = {k: t.detach().clone() for k, t in src.items()}
        else:
            bad += 1
            if bad >= 6:
                break
    if state is not None:
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
    ap.add_argument("--protos", default="L2_random,L2_LCO")
    ap.add_argument("--variants", default="v2,v3mix1,v3mix2,v3mix1-d0.2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(HERE, "_l2_screen.csv"))
    ap.add_argument("--threads", type=int, default=8,
                    help="torch intra-op CPU threads (lower = less desktop lag)")
    ap.add_argument("--low-priority", action="store_true",
                    help="run at below-normal OS priority so the desktop stays responsive")
    ap.add_argument("--skip", default="",
                    help="comma-separated proto:variant pairs already done, e.g. L2_random:v3mix1")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.low_priority:
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print(f"priority=below-normal threads={args.threads}", flush=True)
        except Exception as exc:  # pragma: no cover
            print("priority set failed:", exc, flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    MK = {
        "v2": lambda cd: ReliaDRPCombo(d_cell=cd).to(dev),
        "v3mix1": lambda cd: ReliaDRPComboV3(d_cell=cd, n_mix=1).to(dev),
        "v3mix2": lambda cd: ReliaDRPComboV3(d_cell=cd, n_mix=2).to(dev),
        "v3mix1-d0.2": lambda cd: ReliaDRPComboV3(d_cell=cd, n_mix=1, dropout=0.2).to(dev),
        "v4-h128": lambda cd: ReliaDRPComboV4(d_cell=cd).to(dev),
        "v4-h256": lambda cd: ReliaDRPComboV4(d_cell=cd, hidden=256).to(dev),
        "v4-h256-d0.1": lambda cd: ReliaDRPComboV4(d_cell=cd, hidden=256, dropout=0.1).to(dev),
    }
    names = [v.strip() for v in args.variants.split(",")]

    d = args.l2dir
    resp = pd.read_csv(os.path.join(d, "response_l2.csv"))
    expr = np.load(os.path.join(d, "cells_expr.npy")).astype(np.float32)
    morgan = np.load(os.path.join(d, "drug_morgan.npy")).astype(np.float32)
    ci = resp["cell_idx"].to_numpy(np.int64)
    dri = resp["drow_idx"].to_numpy(np.int64)
    dci = resp["dcol_idx"].to_numpy(np.int64)
    yv = resp["y_css_ri"].to_numpy(np.float32)
    wv = resp["reliability_w"].to_numpy(np.float32)
    y = torch.from_numpy(yv).to(dev)
    w = torch.from_numpy(wv).to(dev)
    rows = []
    if os.path.exists(args.out):
        try:
            rows = pd.read_csv(args.out).to_dict("records")
            print(f"resuming from {args.out}: {len(rows)} rows", flush=True)
        except Exception:
            rows = []
    done = {(r["proto"], r["variant"]) for r in rows}
    done |= {(p.split(":")[0], p.split(":")[1])
             for p in [s.strip() for s in args.skip.split(",") if ":" in s]}
    for P in [p.strip() for p in args.protos.split(",")]:
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
        for vname in names:
            if (P, vname) in done:
                print(f"[{P}] {vname:14s} already done, skipping", flush=True)
                continue
            t0 = time.time()
            m = MK[vname.replace("-ema", "")](cd.shape[1])
            m = train(m, tr, va, X, y, w, 0, args.epochs, args.batch,
                      ema_decay=0.999 if vname.endswith("-ema") else 0.0)
            mu_t, _ = infer(m, X, te)
            mu_v, _ = infer(m, X, va)
            r_te = float(stats.pearsonr(yv[te], mu_t)[0])
            r_va = float(stats.pearsonr(yv[va], mu_v)[0])
            row = {"proto": P, "variant": vname, "val_r": round(r_va, 4),
                   "test_r": round(r_te, 4),
                   "test_rmse": round(float(np.sqrt(np.mean((yv[te] - mu_t) ** 2))), 4),
                   "min": round(float(mu_t.min()), 4), "max": round(float(mu_t.max()), 4),
                   "sec": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"[{P}] {vname:14s} val_r={r_va:.4f} test_r={r_te:.4f} "
                  f"({row['sec']}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
