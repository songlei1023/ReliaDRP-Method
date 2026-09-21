import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reliadrp import ReliaDRPExpr  # noqa: E402

ALPHA = 0.1


def q_conf(s, alpha=ALPHA):
    n = len(s)
    return float(np.quantile(s, (1 - alpha) * (1 + 1.0 / n)))


def train(model, tr, va, X, y, seed, epochs, batch):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, bad, state = 1e18, 0, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), batch):
            rows = tr[perm[i:i + batch]]
            loss = model.compute_loss(X[rows], y[rows], ep, epochs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            mu, _ = model.predict(X[va])
            rv = float(np.sqrt(np.mean((y[va].cpu().numpy() - mu) ** 2)))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3dir", default=os.path.join(HERE, "data", "processed_ext", "L3"))
    ap.add_argument("--protocols", default="L3_random,L3_ldo")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "l3_reliadrp_expr.csv"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = args.l3dir
    cells = np.load(os.path.join(d, "cells_l3.npy")).astype(np.float32)
    yv = np.load(os.path.join(d, "y_l3.npy")).astype(np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    y = torch.from_numpy(yv).to(dev)
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for P in [p.strip() for p in args.protocols.split(",")]:
        tr = np.load(os.path.join(d, "splits", f"{P}_train.npy"))
        va = np.load(os.path.join(d, "splits", f"{P}_val.npy"))
        te = np.load(os.path.join(d, "splits", f"{P}_test.npy"))
        pca = PCA(min(args.dim, len(tr) - 1, cells.shape[1]), random_state=0).fit(cells[tr])
        X = torch.from_numpy(pca.transform(cells).astype(np.float32)).to(dev)
        y_t = torch.from_numpy(yv).to(dev)
        for seed in seeds:
            t0 = time.time()
            m = ReliaDRPExpr(d_in=X.shape[1]).to(dev)
            m = train(m, tr, va, X, y_t, seed, args.epochs, args.batch)
            mu_v, sig_v = m.predict(X[va])
            mu_t, sig_t = m.predict(X[te])
            yte = yv[te]
            yva = yv[va]
            stud = bool(np.std(sig_t) > 1e-8)
            sc = np.abs(yva - mu_v) / (sig_v + 1e-8) if stud else np.abs(yva - mu_v)
            q = q_conf(sc)
            lo, hi = (mu_t - q * sig_t, mu_t + q * sig_t) if stud else (mu_t - q, mu_t + q)
            cov = float(np.mean((yte >= lo) & (yte <= hi)))
            row = {"protocol": P, "model": "reliadrp_expr", "seed": seed,
                   "rmse": round(float(np.sqrt(np.mean((yte - mu_t) ** 2))), 4),
                   "auroc": round(float(roc_auc_score(yte, mu_t)), 4),
                   "acc": round(float(((mu_t >= 0.5) == (yte >= 0.5)).mean()), 4),
                   "picp": round(cov, 4), "mpiw": round(float(np.mean(hi - lo)), 4),
                   "epi_mean": round(float(np.mean(sig_t ** 2)), 5)}
            rows.append(row)
            print(f"[{P}] reliadrp_expr s{seed}: rmse={row['rmse']} auroc={row['auroc']} "
                  f"acc={row['acc']} picp={row['picp']} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
