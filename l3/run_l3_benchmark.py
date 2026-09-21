import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
ALPHA = 0.1


class Clf(nn.Module):
    def __init__(self, d_in, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)[:, 0]


def train_member(X, y, tr, va, seed, epochs=30, batch=512):
    torch.manual_seed(seed)
    m = Clf(X.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt, yt = X[tr], y[tr]
    best, state, bad = 1e18, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = F.binary_cross_entropy_with_logits(m(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(m(X[va]), y[va]).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    m.load_state_dict(state); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3dir", default=os.path.normpath(os.path.join(HERE, "data", "processed_ext", "L3")))
    ap.add_argument("--protocols", default="L3_random,L3_ldo")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "l3_benchmark.csv"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = args.l3dir
    cells = np.load(os.path.join(d, "cells_l3.npy")).astype(np.float32)
    y = np.load(os.path.join(d, "y_l3.npy")).astype(np.float32)
    ds = pd.read_csv(os.path.join(d, "meta_l3.csv"))["dataset"].to_numpy()
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for P in [p.strip() for p in args.protocols.split(",")]:
        tr = np.load(os.path.join(d, "splits", f"{P}_train.npy"))
        va = np.load(os.path.join(d, "splits", f"{P}_val.npy"))
        te = np.load(os.path.join(d, "splits", f"{P}_test.npy"))
        pca = PCA(min(128, len(tr) - 1, cells.shape[1]), random_state=0).fit(cells[tr])
        Xc = torch.from_numpy(pca.transform(cells).astype(np.float32))
        yt = torch.from_numpy(y)
        members = [train_member(Xc, yt, tr, va, s) for s in seeds]
        with torch.no_grad():
            Pv = torch.stack([torch.sigmoid(m(Xc[va])) for m in members]).numpy()
            Pt = torch.stack([torch.sigmoid(m(Xc[te])) for m in members]).numpy()
        pv, pt = Pv.mean(0), Pt.mean(0)
        epi_v, epi_t = Pv.std(0), Pt.std(0)
        acc = float(((pt >= 0.5) == (y[te] >= 0.5)).mean())
        auc = float(roc_auc_score(y[te], pt))
        s_cal = 1.0 - np.where(y[va] > 0.5, pv, 1 - pv)
        n = len(s_cal)
        q = float(np.quantile(s_cal, (1 - ALPHA) * (1 + 1.0 / n)))
        thr = 1.0 - q
        in_set = np.where(y[te] > 0.5, pt, 1 - pt) >= thr
        cov = float(in_set.mean())
        size = float((np.where(pt >= thr, 1, 0) + np.where((1 - pt) >= thr, 1, 0)).mean())
        row = {"protocol": P, "acc": round(acc, 4), "auroc": round(auc, 4),
               "conformal_cov": round(cov, 4), "avg_set_size": round(size, 4),
               "epi_mean": round(float(epi_t.mean()), 5), "n_test": int(len(te))}
        if P.endswith("ldo"):
            seen = ~np.isin(ds[te], np.unique(ds[tr]))
            row["epi_unseen_ds"] = round(float(epi_t[seen].mean()), 5) if seen.sum() else np.nan
            row["epi_seen_ds"] = round(float(epi_t[~seen].mean()), 5) if (~seen).sum() else np.nan
            if seen.sum() and (~seen).sum():
                from sklearn.metrics import roc_auc_score as ra
                lbl = seen.astype(int)
                row["epi_ood_auc"] = round(float(ra(lbl, epi_t)), 4)
                row["conformal_cov_unseen"] = round(float(in_set[seen].mean()), 4)
                row["conformal_cov_seen"] = round(float(in_set[~seen].mean()), 4)
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
