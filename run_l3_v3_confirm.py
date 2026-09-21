"""Round-1 confirmation: 5 seeds for the top L3 variants, both protocols."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_l3_v3_screen import train  # noqa: E402
from reliadrp_v3 import ReliaDRPExprV3  # noqa: E402

VARIANTS = {
    "v3-mix+bce1.0": dict(d_in=384, w_bce=1.0),
    "v3-noSE+bce0.5": dict(d_in=384, w_bce=0.5, use_se=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3dir", default=os.path.join(HERE, "data", "processed_ext", "L3"))
    ap.add_argument("--out", default=os.path.join(HERE, "l3_reliadrp_v3.csv"))
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = args.l3dir
    cells = np.load(os.path.join(d, "cells_l3.npy")).astype(np.float32)
    yv = np.load(os.path.join(d, "y_l3.npy")).astype(np.float32)
    y = torch.from_numpy(yv).to(dev)
    rows = []
    for P in ["L3_random", "L3_ldo"]:
        tr = np.load(os.path.join(d, "splits", f"{P}_train.npy"))
        va = np.load(os.path.join(d, "splits", f"{P}_val.npy"))
        te = np.load(os.path.join(d, "splits", f"{P}_test.npy"))
        pca = PCA(384, random_state=0).fit(cells[tr])
        X = torch.from_numpy(pca.transform(cells).astype(np.float32)).to(dev)
        for vname, kw in VARIANTS.items():
            for seed in range(5):
                m = ReliaDRPExprV3(**kw).to(dev)
                m, vb = train(m, tr, va, X, y, seed, 30, 512, select="auroc")
                m.eval()
                with torch.no_grad():
                    mu_t = m(X[te])[0].cpu().numpy()
                auc = roc_auc_score(yv[te], mu_t)
                acc = float(((mu_t >= 0.5) == (yv[te] >= 0.5)).mean())
                rows.append({"protocol": P, "variant": vname, "seed": seed,
                             "auroc": round(float(auc), 4), "acc": round(acc, 4)})
                print(f"[{P}] {vname} s{seed}: auroc={auc:.4f} acc={acc:.4f}", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
    df = pd.DataFrame(rows)
    print("\n=== mean ± sd ===")
    print(df.groupby(["protocol", "variant"])["auroc"].agg(["mean", "std", "count"]).round(4))


if __name__ == "__main__":
    main()
