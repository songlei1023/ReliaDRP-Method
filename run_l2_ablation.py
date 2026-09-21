"""L2 ablation: where does MoGraphDRP's L2_random advantage come from?

Runs 1 seed each:
  graph-asis    : MoGraphDRP exactly as the baseline runner feeds it (sanity)
  graph-pairsep : MoGraphDRP but fed our separate dr/dc (mean -> concat)
  combo-noLN    : ReliaDRPCombo with all LayerNorm swapped for plain ReLU
  combo-h256    : ReliaDRPCombo with hidden=256
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_zoo_2021_2026 as zoo  # noqa: E402
from run_benchmark import train_one, infer  # noqa: E402
from reliadrp import ReliaDRPCombo, _block  # noqa: E402
from reliadrp_v3 import TokenMixer, nig_nll_elem  # noqa: E402
from evidrp_v2 import softplus1, softplus_eps  # noqa: E402


class ComboNoLN(ReliaDRPCombo):
    """ReliaDRPCombo without LayerNorm (plain ReLU blocks, mograph-style)."""

    def __init__(self, d_cell=384, d_drug=128, hidden=128, dropout=0.1,
                 w_reg=0.05, w_rel=0.1):
        super().__init__(d_cell, d_drug, hidden, dropout, w_reg, w_rel)
        plain = lambda i, o: nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Dropout(dropout),
                                           nn.Linear(o, o), nn.ReLU())
        self.cell_enc = plain(d_cell, hidden)
        self.drug_enc = plain(d_drug, hidden)
        self.pair_enc = plain(hidden * 4, hidden)
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden * 2), nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden * 2, hidden), nn.ReLU())


class ReliaDRPComboWide(ReliaDRPCombo):
    def __init__(self, **kw):
        kw.setdefault("hidden", 256)
        super().__init__(**kw)


class GraphPairSep(torch.nn.Module):
    """MoGraphDRP trunk, but with separate row/col drug encoders + pair encoder."""

    def __init__(self, hidden=128, dropout=0.3):
        super().__init__()
        import model_zoo_2021_2026 as mz
        self.net = mz.MoGraphDRP(hidden=hidden, dropout=dropout)
        self.drug_enc2 = mz.MoGraphDRP(hidden=hidden, dropout=dropout).drug_proj
        self.pair = _block(hidden * 4, hidden, dropout)
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden * 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden * 2, 1))

    def forward(self, c, dr, dc):
        cellv = self.net(c, (dr + dc) / 2.0)          # scalar! -> replace below
        raise RuntimeError("use forward2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2dir", default=os.path.join(HERE, "data", "processed_ext", "L2"))
    ap.add_argument("--protos", default="L2_random")
    ap.add_argument("--variants", default="graph-asis,combo-noLN,combo-h256")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(HERE, "_l2_ablation.csv"))
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = args.l2dir
    resp = pd.read_csv(os.path.join(d, "response_l2.csv"))
    expr = np.load(os.path.join(d, "cells_expr.npy")).astype(np.float32)
    morgan = np.load(os.path.join(d, "drug_morgan.npy")).astype(np.float32)
    ci = resp["cell_idx"].to_numpy(np.int64)
    dri = resp["drow_idx"].to_numpy(np.int64)
    dci = resp["dcol_idx"].to_numpy(np.int64)
    yv = resp["y_css_ri"].to_numpy(np.float32)
    y = torch.from_numpy(yv).to(dev)
    import numpy as _np
    none_w = None

    rows = []
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
        drugs_mean = ((dd[dri] + dd[dci]) / 2.0).astype(np.float32)
        for vname in [v.strip() for v in args.variants.split(",")]:
            t0 = time.time()
            if vname == "graph-asis":
                # feed exactly like run_layer_baselines: cells, drugs_mean
                m = zoo.build("mograph").to(dev)
                m = train_one(m, tr, va, ci, np.arange(len(resp)), yv, cd, drugs_mean,
                              None, None, 0, args.epochs, args.batch)
                m.eval()
                with torch.no_grad():
                    tc = torch.from_numpy(cd).to(dev)
                    td = torch.from_numpy(drugs_mean).to(dev)
                    ci_t = torch.from_numpy(ci.astype(np.int64)).to(dev)
                    di_t = torch.from_numpy(np.arange(len(resp)).astype(np.int64)).to(dev)
                    mu_v, _, _, _ = infer(m, ci_t, di_t, tc, td, va)
                    mu_t, _, _, _ = infer(m, ci_t, di_t, tc, td, te)
            elif vname == "graph-pairsep":
                sys.exit("graph-pairsep not wired; skipped")
            elif vname == "combo-noLN":
                m = ComboNoLN().to(dev)
                m = train_combo(m, tr, va, cd, dd, ci, dri, dci, yv, dev,
                                args.epochs, args.batch)
                mu_v, mu_t = eval_combo(m, cd, dd, ci, dri, dci, va, te, dev)
            elif vname == "combo-h256":
                m = ReliaDRPComboWide().to(dev)
                m = train_combo(m, tr, va, cd, dd, ci, dri, dci, yv, dev,
                                args.epochs, args.batch)
                mu_v, mu_t = eval_combo(m, cd, dd, ci, dri, dci, va, te, dev)
            row = {"proto": P, "variant": vname,
                   "val_r": round(float(stats.pearsonr(yv[va], mu_v)[0]), 4),
                   "test_r": round(float(stats.pearsonr(yv[te], mu_t)[0]), 4),
                   "sec": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"[{P}] {vname:14s} val_r={row['val_r']:.4f} test_r={row['test_r']:.4f} "
                  f"({row['sec']}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


def train_combo(m, tr, va, cd, dd, ci, dri, dci, yv, dev, epochs, batch):
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    C = torch.from_numpy(cd[ci]).to(dev)
    DR = torch.from_numpy(dd[dri]).to(dev)
    DC = torch.from_numpy(dd[dci]).to(dev)
    Y = torch.from_numpy(yv).to(dev)
    best, bad, state = 1e18, 0, None
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), batch):
            rows = tr[perm[i:i + batch]]
            loss = m.compute_loss(C[rows], DR[rows], DC[rows], Y[rows], None, ep, epochs)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            o = m(C[va], DR[va], DC[va])
            rv = float(torch.sqrt(F.mse_loss(o[0], Y[va])))
        if rv < best - 1e-5:
            best, bad = rv, 0
            state = {k: t.clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    if state is not None:
        m.load_state_dict(state)
    return m


def eval_combo(m, cd, dd, ci, dri, dci, va, te, dev):
    C = torch.from_numpy(cd[ci]).to(dev)
    DR = torch.from_numpy(dd[dri]).to(dev)
    DC = torch.from_numpy(dd[dci]).to(dev)
    with torch.no_grad():
        mu_v = m(C[va], DR[va], DC[va])[0].cpu().numpy()
        mu_t = m(C[te], DR[te], DC[te])[0].cpu().numpy()
    return mu_v, mu_t


if __name__ == "__main__":
    main()
