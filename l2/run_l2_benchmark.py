import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
ALPHA = 0.1
TIERS = ["high", "mid", "low"]


def sp1(x):
    return F.softplus(x) + 1.0


def sp_eps(x):
    return F.softplus(x) + 1e-6


def nig_elem(t, mu, v, a, b):
    import math
    Om = 2.0 * b * (1.0 + v)
    t1 = 0.5 * math.log(math.pi) - 0.5 * torch.log(v)
    t2 = -a * torch.log(Om)
    t3 = (a + 0.5) * torch.log(v * (t - mu) ** 2 + Om)
    t4 = torch.lgamma(a) - torch.lgamma(a + 0.5)
    return t1 + t2 + t3 + t4


class AdditiveNet(nn.Module):
    """Drug-additive baseline: y = b + f(drug_row) + f(drug_col), cell ignored."""

    def __init__(self, d_drug=128):
        super().__init__()
        self.wr = nn.Linear(d_drug, 1)
        self.wc = nn.Linear(d_drug, 1)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, c, dr, dc):
        return (self.wr(dr) + self.wc(dc) + self.b)[:, 0]

    def loss(self, c, dr, dc, y, ep, epochs, w=None, mse_frac=0.5):
        ww = torch.ones_like(y) if w is None else (w / w.mean().clamp(min=1e-8))
        return (F.mse_loss(self(c, dr, dc), y, reduction="none") * ww).mean()


class ComboNet(nn.Module):
    def __init__(self, d_cell=128, d_drug=128, hidden=128, dropout=0.1, evidential=True):
        super().__init__()
        self.evidential = evidential
        self.cell = nn.Sequential(nn.Linear(d_cell, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.drug = nn.Sequential(nn.Linear(d_drug, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout))
        if evidential:
            self.out_mu = nn.Linear(hidden, 1)
            self.out_ev = nn.Linear(hidden, 3)
        else:
            self.out = nn.Linear(hidden, 1)

    def forward(self, c, dr, dc):
        z = self.fuse(torch.cat([self.cell(c), self.drug(dr), self.drug(dc)], -1))
        if not self.evidential:
            return self.out(z)[:, 0]
        mu = self.out_mu(z)[:, 0]
        o = self.out_ev(z)
        return mu, sp1(o[:, 0]), sp1(o[:, 1]), sp_eps(o[:, 2])

    def loss(self, c, dr, dc, y, ep, epochs, w=None, mse_frac=0.5):
        ww = torch.ones_like(y) if w is None else (w / w.mean().clamp(min=1e-8))
        if not self.evidential:
            return (F.mse_loss(self(c, dr, dc), y, reduction="none") * ww).mean()
        mu, v, a, b = self(c, dr, dc)
        if ep < mse_frac * epochs:
            return (F.mse_loss(mu, y, reduction="none") * ww).mean()
        nll = (nig_elem(y, mu, v, a, b) * ww).mean()
        reg = (torch.abs(y - mu) * (2.0 * v + a) * ww).mean() * 0.05
        return nll + reg


def q_conf(s, alpha=ALPHA):
    n = len(s)
    return float(np.quantile(s, (1 - alpha) * (1 + 1.0 / n)))


def picp(y, mu, sig, q, stud):
    lo, hi = (mu - q * sig, mu + q * sig) if stud else (mu - q, mu + q)
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def train(model, tr, va, X, y, w, seed, epochs, batch):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, bad = 1e18, 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), batch):
            r = tr[perm[i:i + batch]]
            loss = model.loss(X["c"][r], X["dr"][r], X["dc"][r], y[r], ep, epochs, w[r])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            o = model(X["c"][va], X["dr"][va], X["dc"][va])
            mu = o[0] if isinstance(o, tuple) else o
            r = float(torch.sqrt(F.mse_loss(mu, y[va])))
        if r < best - 1e-5:
            best, bad = r, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(state)
    return model


def infer(model, X, idx):
    with torch.no_grad():
        o = model(X["c"][idx], X["dr"][idx], X["dc"][idx])
    if isinstance(o, tuple):
        mu, v, a, b = o
        ale = b / (a - 1)
        epi = b / (v * (a - 1))
        return mu.numpy(), torch.sqrt(ale + epi).numpy()
    return o.numpy(), np.ones(len(idx), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2dir", default=os.path.normpath(os.path.join(HERE, "data", "processed_ext", "L2")))
    ap.add_argument("--protocols", default="L2_random,L2_LCO,L2_LDO")
    ap.add_argument("--models", default="evi,mlp")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "l2_benchmark.csv"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = args.l2dir
    resp = pd.read_csv(os.path.join(d, "response_l2.csv"))
    cells = np.load(os.path.join(d, "cells_expr.npy")).astype(np.float32)
    drugs = np.load(os.path.join(d, "drug_morgan.npy")).astype(np.float32)
    yv = resp["y_css_ri"].to_numpy(np.float32)
    wv = resp["reliability_w"].to_numpy(np.float32)
    tier = resp["reliability_tier"].to_numpy()
    ci = resp["cell_idx"].to_numpy(np.int64)
    dri = resp["drow_idx"].to_numpy(np.int64)
    dci = resp["dcol_idx"].to_numpy(np.int64)
    models = [m.strip() for m in args.models.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for P in [p.strip() for p in args.protocols.split(",")]:
        tr = np.load(os.path.join(d, "splits", f"{P}_train.npy"))
        va = np.load(os.path.join(d, "splits", f"{P}_val.npy"))
        te = np.load(os.path.join(d, "splits", f"{P}_test.npy"))
        n_cell = min(128, max(2, len(np.unique(ci[tr])) - 1))
        tcells = PCA(n_cell, random_state=0).fit(cells[np.unique(ci[tr])])
        cd = tcells.transform(cells).astype(np.float32)
        tdrug = PCA(128, random_state=0).fit(drugs)
        dd = tdrug.transform(drugs).astype(np.float32)
        X = {"c": torch.from_numpy(cd[ci]), "dr": torch.from_numpy(dd[dri]),
             "dc": torch.from_numpy(dd[dci])}
        yt = torch.from_numpy(yv)
        wt = torch.from_numpy(wv)
        for name in models:
            for seed in seeds:
                t0 = time.time()
                m = AdditiveNet() if name == "additive" else \
                    ComboNet(d_cell=tcells.n_components_, evidential=(name == "evi"))
                m = train(m, tr, va, X, yt, wt, seed, args.epochs, args.batch)
                mu_v, sig_v = infer(m, X, va)
                mu_t, sig_t = infer(m, X, te)
                stud = bool(np.std(sig_t) > 1e-8)
                s_cal = np.abs(yv[va] - mu_v) / (sig_v + 1e-8) if stud else np.abs(yv[va] - mu_v)
                q = q_conf(s_cal)
                pp, mm = picp(yv[te], mu_t, sig_t, q, stud)
                row = {"protocol": P, "model": name, "seed": seed,
                       "rmse": round(float(np.sqrt(np.mean((yv[te] - mu_t) ** 2))), 4),
                       "pearson": round(float(stats.pearsonr(yv[te], mu_t)[0]), 4),
                       "picp": round(pp, 4), "mpiw": round(mm, 4)}
                for t in TIERS:
                    msk = tier[te] == t
                    if msk.sum() < 10:
                        continue
                    p_cp, _ = picp(yv[te][msk], mu_t[msk], sig_t[msk], q, stud)
                    mv = tier[va] == t
                    qt = q_conf(s_cal[mv]) if mv.sum() > 5 else q
                    p_gc, _ = picp(yv[te][msk], mu_t[msk], sig_t[msk], qt, stud)
                    row[f"picp_cp_{t}"] = round(p_cp, 4)
                    row[f"picp_gc_{t}"] = round(p_gc, 4)
                rows.append(row)
                print(f"[{P}] {name} s{seed}: rmse={row['rmse']} r={row['pearson']} "
                      f"picp={pp:.3f} cp_high={row.get('picp_cp_high')} gc_high={row.get('picp_gc_high')} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
