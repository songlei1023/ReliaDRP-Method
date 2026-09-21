"""Round-1 fast screen for L3 (seed 0 only, selection by VAL AUROC).

Variants: baseline ReliaDRPExpr (v2) vs ReliaDRPExprV3 flags.
Test AUROC is printed for information only; variant choice uses val AUROC.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
sys.path.insert(0, os.path.join(HERE, "core"))  # 库模块在 core/
sys.path.insert(0, HERE)
from reliadrp import ReliaDRPExpr  # noqa: E402
from reliadrp_v3 import ReliaDRPExprV3  # noqa: E402


def train(model, tr, va, X, y, seed, epochs, batch, select="auroc"):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, bad, state = -1e18 if select == "auroc" else 1e18, 0, None

    def val_score():
        model.eval()
        with torch.no_grad():
            out = model(X[va])
            mu = out[0] if isinstance(out, tuple) else out[0]
        if select == "auroc":
            return roc_auc_score(y[va].cpu().numpy(), mu.cpu().numpy())
        return -float(np.sqrt(np.mean((y[va].cpu().numpy() - mu.cpu().numpy()) ** 2)))

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), batch):
            rows = tr[perm[i:i + batch]]
            loss = model.compute_loss(X[rows], y[rows], ep, epochs)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        s = val_score()
        better = s > best + 1e-5 if select == "auroc" else s < best - 1e-5
        if better:
            best, bad = s, 0
            state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    if state is not None:
        model.load_state_dict(state)
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3dir", default=os.path.join(HERE, "data", "processed_ext", "L3"))
    ap.add_argument("--protos", default="L3_random,L3_ldo")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(HERE, "_l3_screen.csv"))
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = args.l3dir
    cells = np.load(os.path.join(d, "cells_l3.npy")).astype(np.float32)
    yv = np.load(os.path.join(d, "y_l3.npy")).astype(np.float32)
    y = torch.from_numpy(yv).to(dev)

    variants = {
        "v2@256(rmse-sel)": ("v2", dict(d_in=256), 256, "rmse"),
        "v2@384(auc-sel)": ("v2", dict(d_in=384), 384, "auroc"),
        "v3-mix@384": ("v3", dict(d_in=384, w_bce=0.0), 384, "auroc"),
        "v3-mix+bce0.5": ("v3", dict(d_in=384, w_bce=0.5), 384, "auroc"),
        "v3-mix+bce1.0": ("v3", dict(d_in=384, w_bce=1.0), 384, "auroc"),
        "v3-1mix+bce0.5": ("v3", dict(d_in=384, w_bce=0.5, n_mix=1), 384, "auroc"),
        "v3-6tok+bce0.5": ("v3", dict(d_in=384, w_bce=0.5, n_tok=6), 384, "auroc"),
        "v3-drop0.3+bce0.5": ("v3", dict(d_in=384, w_bce=0.5, dropout=0.3), 384, "auroc"),
        "v3-noSE+bce0.5": ("v3", dict(d_in=384, w_bce=0.5, use_se=False), 384, "auroc"),
    }

    rows = []
    for P in [p.strip() for p in args.protos.split(",")]:
        tr = np.load(os.path.join(d, "splits", f"{P}_train.npy"))
        va = np.load(os.path.join(d, "splits", f"{P}_val.npy"))
        te = np.load(os.path.join(d, "splits", f"{P}_test.npy"))
        for vname, (fam, kw, dim, sel) in variants.items():
            t0 = time.time()
            pca = PCA(min(dim, len(tr) - 1, cells.shape[1]), random_state=0).fit(cells[tr])
            X = torch.from_numpy(pca.transform(cells).astype(np.float32)).to(dev)
            if fam == "v2":
                m = ReliaDRPExpr(d_in=X.shape[1]).to(dev)
            else:
                m = ReliaDRPExprV3(**{**kw, "d_in": X.shape[1]}).to(dev)
            m, vb = train(m, tr, va, X, y, 0, args.epochs, args.batch, select=sel)
            m.eval()
            with torch.no_grad():
                mu_t = m(X[te])[0].cpu().numpy()
                mu_v = m(X[va])[0].cpu().numpy()
            row = {"proto": P, "variant": vname, "val": round(vb, 4),
                   "val_auroc": round(roc_auc_score(yv[va], mu_v), 4),
                   "test_auroc": round(roc_auc_score(yv[te], mu_t), 4),
                   "test_rmse": round(float(np.sqrt(np.mean((yv[te] - mu_t) ** 2))), 4),
                   "sec": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"[{P}] {vname:22s} val={row['val']:.4f} valAUROC={row['val_auroc']:.4f} "
                  f"testAUROC={row['test_auroc']:.4f} ({row['sec']}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
