"""Unified L2 (drug-combination) benchmark — one harness, one recipe.

Why this script exists
----------------------
The L2 table mixed two training paths:

  * the fifteen external baselines came from `run_layer_baselines.py`, which
    trains through `run_benchmark.train_one` for 10 epochs at batch 4096;
  * `reliadrp` (the proposed combination model, `ReliaDRPCombo`) came from
    `run_l2_v3_confirm.py`, whose *own* loop runs 12 epochs, adds gradient
    clipping and uses a different drug representation (two separate drug
    vectors rather than their mean).

The mismatch is smaller than the L3 one, but it is still a mismatch: the
proposed model saw 20% more epochs and a different input layout.  Here every
model goes through the same `train_one` call with the same recipe.

How one harness serves both input layouts
-----------------------------------------
`train_one` hands the model a single drug tensor.  The combination data has a
row-drug and a column-drug, so the drug tensor is stored as the concatenation
`[dr | dc]` (2 x 128 dims) and two thin adapters decide what to do with it:

  * `MeanDrugAdapter`  -> `(dr + dc) / 2`, exactly the representation the
    original baseline table used, so the baselines keep their semantics;
  * `PairDrugAdapter`  -> forwards `(dr, dc)` to `ReliaDRPCombo`, which is the
    only model that exploits the pair structure.

Reliability weights `w` are passed to the two ReliaDRP variants only, matching
the convention already used for L1/L4 (baselines are unweighted).

Everything else — PCA bases, splits, early stopping on validation RMSE,
seed handling — is shared, so the table is like-for-like.
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
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
sys.path.insert(0, os.path.join(HERE, "core"))  # 库模块在 core/
sys.path.insert(0, os.path.join(HERE, "benchmark"))  # 复用 run_benchmark 的 train_one/infer
sys.path.insert(0, os.path.join(HERE, "l2"))

import model_zoo_2021_2026 as zoo  # noqa: E402
from run_benchmark import train_one, infer, q_conformal, picp_mpiw  # noqa: E402
from reliadrp import ReliaDRPCombo  # noqa: E402

EXTR = os.path.join(HERE, "data", "processed_ext")
PROPOSED = "reliadrpcombo"
RELIA_NAMES = {"reliadrp", PROPOSED}


class MeanDrugAdapter(nn.Module):
    """Serves a single-drug baseline from the concatenated [dr|dc] tensor."""

    def __init__(self, inner, d_drug=128):
        super().__init__()
        self.inner = inner
        self.d_drug = d_drug

    def _mean(self, drug):
        return (drug[:, :self.d_drug] + drug[:, self.d_drug:]) / 2.0

    def forward(self, cell, drug):
        return self.inner(cell, self._mean(drug))

    def compute_loss(self, cell, drug, y, w=None, ctx=None, ep=0, epochs=40):
        m = self._mean(drug)
        if hasattr(self.inner, "compute_loss"):
            # keep whatever loss the wrapped model (e.g. the single-drug
            # ReliaDRP ablation, which is reliability-weighted) used before
            import inspect
            sig = inspect.signature(self.inner.compute_loss).parameters
            kw = {}
            if "w" in sig:
                kw["w"] = w
            if "ctx" in sig:
                kw["ctx"] = ctx
            if "ep" in sig:
                kw["ep"] = ep
            if "epochs" in sig:
                kw["epochs"] = epochs
            return self.inner.compute_loss(cell, m, y, **kw)
        out = self.inner(cell, m)
        mu = out[0] if isinstance(out, tuple) else out
        return F.mse_loss(mu, y)


class PairDrugAdapter(nn.Module):
    """Serves ReliaDRPCombo, which exploits the two drug vectors."""

    def __init__(self, inner, d_drug=128):
        super().__init__()
        self.inner = inner
        self.d_drug = d_drug

    def _pair(self, drug):
        return drug[:, :self.d_drug], drug[:, self.d_drug:]

    def forward(self, cell, drug):
        dr, dc = self._pair(drug)
        return self.inner(cell, dr, dc)

    def compute_loss(self, cell, drug, y, w=None, ctx=None, ep=0, epochs=40):
        dr, dc = self._pair(drug)
        return self.inner.compute_loss(cell, dr, dc, y, w=w, ep=ep, epochs=epochs)


def build(name, d_cell, d_drug):
    if name == PROPOSED:
        return PairDrugAdapter(ReliaDRPCombo(d_cell=d_cell, d_drug=d_drug), d_drug)
    return MeanDrugAdapter(zoo.build(name), d_drug)


def prep(d_drug=128, cell_dim=384):
    d = os.path.join(EXTR, "L2")
    resp = pd.read_csv(os.path.join(d, "response_l2.csv"))
    expr = np.load(os.path.join(d, "cells_expr.npy")).astype(np.float32)
    morgan = np.load(os.path.join(d, "drug_morgan.npy")).astype(np.float32)
    ci = resp["cell_idx"].to_numpy(np.int64)
    dri = resp["drow_idx"].to_numpy(np.int64)
    dci = resp["dcol_idx"].to_numpy(np.int64)
    y = resp["y_css_ri"].to_numpy(np.float32)
    w = resp["reliability_w"].to_numpy(np.float32)
    return resp, expr, morgan, ci, dri, dci, y, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=PROPOSED + "," + ",".join(zoo.ALL_NAMES))
    ap.add_argument("--protocols", default="L2_random,L2_LCO,L2_LDO")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--select", default="rmse", choices=["rmse", "auroc"])
    ap.add_argument("--cell-dim", type=int, default=384)
    ap.add_argument("--drug-dim", type=int, default=128)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "l2_unified.csv"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.low_priority:
        try:
            import psutil
            psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("[low-priority] BELOW_NORMAL set", flush=True)
        except Exception as e:  # pragma: no cover
            print("[low-priority] skipped:", e, flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    done = set()
    rows = []
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        if {"protocol", "model", "seed"}.issubset(prev.columns):
            done = {(r["protocol"], r["model"], int(r["seed"]))
                    for _, r in prev.iterrows() if np.isfinite(r.get("pearson", np.nan))}
            rows = prev.to_dict("records")
            print(f"[resume] {len(done)} finished rows", flush=True)

    resp, expr, morgan, ci, dri, dci, y, w = prep(args.drug_dim, args.cell_dim)
    yt = y
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    base = os.path.join(EXTR, "L2", "splits")

    for P in [p.strip() for p in args.protocols.split(",")]:
        tr = np.load(os.path.join(base, f"{P}_train.npy"))
        va = np.load(os.path.join(base, f"{P}_val.npy"))
        te = np.load(os.path.join(base, f"{P}_test.npy"))
        if not any((P, m, s) not in done for m in models for s in seeds):
            continue
        t0 = time.time()
        # cell PCA fit on training cells only (same convention as L1/L3)
        cp = PCA(min(args.cell_dim, len(np.unique(ci[tr])) - 1, expr.shape[1]),
                 random_state=0).fit(expr[np.unique(ci[tr])])
        cd = cp.transform(expr).astype(np.float32)
        if cd.shape[1] < args.cell_dim:
            cd = np.hstack([cd, np.zeros((len(cd), args.cell_dim - cd.shape[1]), np.float32)])
        # drug PCA fit on all drugs (the L1/L2 convention)
        dp = PCA(min(args.drug_dim, morgan.shape[0]), random_state=0).fit(morgan)
        dd = dp.transform(morgan).astype(np.float32)
        # [dr | dc] concatenation -- both layouts read from this tensor
        drugs = np.hstack([dd[dri], dd[dci]]).astype(np.float32)
        tc = torch.from_numpy(cd).to(dev)
        td = torch.from_numpy(drugs).to(dev)
        ci_t = torch.from_numpy(ci).to(dev)
        di_t = torch.from_numpy(np.arange(len(resp)).astype(np.int64)).to(dev)
        print(f"[{P}] train={len(tr)} val={len(va)} test={len(te)} "
              f"cell={cd.shape[1]} drug={drugs.shape[1]} (setup {time.time()-t0:.0f}s)", flush=True)

        for name in models:
            for seed in seeds:
                if (P, name, seed) in done:
                    continue
                t0 = time.time()
                try:
                    model = build(name, cd.shape[1], args.drug_dim).to(dev)
                    ww = w if name in RELIA_NAMES else None
                    model = train_one(model, tr, va, ci, di_t.cpu().numpy(), yt, cd, drugs,
                                      ww, None, seed, args.epochs, args.batch,
                                      patience=args.patience, select=args.select)
                    model.eval()
                    mu_v, sig_v, _, _ = infer(model, ci_t, di_t, tc, td, va)
                    mu_t, sig_t, _, _ = infer(model, ci_t, di_t, tc, td, te)
                    if not np.isfinite(mu_t).all():
                        raise ValueError("non-finite")
                    stud = bool(np.std(sig_t) > 1e-8 and np.isfinite(sig_t).all())
                    sc = np.abs(yt[va] - mu_v) / (sig_v + 1e-8) if stud else np.abs(yt[va] - mu_v)
                    q = q_conformal(sc)
                    pp, mm = picp_mpiw(yt[te], mu_t, sig_t, q, stud)
                    row = {"protocol": P, "model": name, "seed": seed,
                           "rmse": round(float(np.sqrt(np.mean((yt[te] - mu_t) ** 2))), 4),
                           "pearson": round(float(np.corrcoef(yt[te], mu_t)[0, 1]), 4),
                           "picp": round(pp, 4), "mpiw": round(mm, 4),
                           "epochs": args.epochs, "batch": args.batch}
                    print(f"[{P}] {name:14s} s{seed}: r={row['pearson']:.4f} "
                          f"rmse={row['rmse']:.4f} ({time.time()-t0:.0f}s)", flush=True)
                except Exception as e:
                    row = {"protocol": P, "model": name, "seed": seed, "pearson": np.nan,
                           "error": f"{type(e).__name__}: {str(e)[:70]}"}
                    print(f"[{P}] {name} s{seed} FAILED {type(e).__name__}: {str(e)[:90]}",
                          flush=True)
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out, index=False)
                try:
                    del model
                except Exception:
                    pass
                if dev == "cuda":
                    torch.cuda.empty_cache()
        del tc, td, ci_t, di_t, cd, drugs
        if dev == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print("wrote", args.out)
    g = (df.dropna(subset=["pearson"]).groupby(["protocol", "model"])["pearson"]
         .agg(["mean", "std", "count"]))
    for P in [p.strip() for p in args.protocols.split(",")]:
        if P in g.index.get_level_values(0):
            sub = g.loc[P].sort_values("mean", ascending=False)
            r = list(sub.index).index(PROPOSED) + 1 if PROPOSED in sub.index else None
            print(f"\n=== {P} (rank of {PROPOSED}: {r}/{len(sub)}) ===")
            print(sub.round(4).to_string())


if __name__ == "__main__":
    main()
