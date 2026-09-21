"""L2 (drug-combination) hyper-parameter tuning of the v3 model.

Why this script exists
----------------------
The paper is being moved to *one* model generation across all four layers.
L1/L3 already use v3 (`ReliaDRPV3`, `ReliaDRPExprV3`).  For L2 the earlier
screen (`run_l2_v3_screen.py`, single seed, its own training loop) suggested
token mixing alone is *not* the right upgrade there:

    L2_random  v2 0.7548  vs  v3mix1 0.7473
    L2_LCO     v2 0.6631  vs  v3mix1 0.6379
    L2_LDO     v2 0.5225  vs  v3mix1 0.5114

so a real sweep is needed before L2 can be reported on the v3 generation.
The ablations on disk say where to look:

    `_l2_ablation.csv`  combo-noLN   L2_random 0.7763   (v2 with LayerNorm off)
    `_l2_screen_v4.csv` v4-h256      L2_random 0.8129   (+ plain blocks, higher
                                                          capacity, cell x pair
                                                          bilinear)

i.e. LayerNorm and capacity --- not the mixer --- carry most of the L2 gain.
Therefore `ReliaDRPComboV3` now takes a `use_ln` knob and this sweep includes
LayerNorm-free and higher-capacity variants of it.

Fairness rules (identical to the L3 audit and to the L1 sweep)
-------------------------------------------------------------
1. **Same harness, same recipe, same budget as the baselines.**  Every run goes
   through `run_benchmark.train_one` with 12 epochs / batch 4096 /
   validation-RMSE early stopping --- exactly the recipe
   `run_l2_unified.py` uses for the fifteen external models.  Only *our own*
   architecture hyper-parameters are swept.
2. **Same input layout.**  Rows are served through the same `PairDrugAdapter`
   over `[dr | dc]` that `run_l2_unified.py` uses, so the pairing semantics are
   unchanged.
3. **`v2` is always in the grid.**  `ReliaDRPCombo` at 5 seeds is re-run by this
   script under the same process, so the v2-vs-v3 comparison is seed-matched
   rather than borrowed from a different run.
4. **Validation-based selection is measured, not assumed.**  Both validation and
   test Pearson are recorded for every row, so the report can quote the
   val-selected config *and* the setting-free mean.

Usage
-----
    python run_l2_v3_tune.py --seeds 0 --out l2_v3_tune_stage1.csv
    python run_l2_v3_tune.py --configs v2,mix1_noln --seeds 0,1,2,3,4 \
        --out l2_v3_tune_confirm.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_benchmark import train_one, infer, q_conformal, picp_mpiw  # noqa: E402
from reliadrp import ReliaDRPCombo  # noqa: E402
from reliadrp_v3 import ReliaDRPComboV3  # noqa: E402
from run_l2_unified import PairDrugAdapter, prep, EXTR  # noqa: E402

PROTOS = ["L2_random", "L2_LCO", "L2_LDO"]


# --------------------------------------------------------------------------
# `mse_frac` is a `compute_loss` parameter, not a constructor attribute, so it
# cannot be swept from outside without a thin wrapper.  One per base class.
# --------------------------------------------------------------------------
class ComboTuned(ReliaDRPCombo):
    def __init__(self, mse_frac=0.5, **kw):
        super().__init__(**kw)
        self.mse_frac = mse_frac

    def compute_loss(self, c, dr, dc, y, w=None, ep=0, epochs=12):
        return super().compute_loss(c, dr, dc, y, w=w, ep=ep, epochs=epochs,
                                    mse_frac=self.mse_frac)


class ComboV3Tuned(ReliaDRPComboV3):
    def __init__(self, mse_frac=0.5, **kw):
        super().__init__(**kw)
        self.mse_frac = mse_frac

    def compute_loss(self, c, dr, dc, y, w=None, ep=0, epochs=12):
        return super().compute_loss(c, dr, dc, y, w=w, ep=ep, epochs=epochs,
                                    mse_frac=self.mse_frac)


# name -> (family, kwargs).  "v2" is the seed-matched reference the paper row
# currently uses; everything else is a v3 variant.
CONFIGS = {
    # --- v2 reference -----------------------------------------------------
    "v2":             ("v2", {}),
    # --- mixer depth ------------------------------------------------------
    "mix0":           ("v3", dict(n_mix=0)),      # token-mean, no attention
    "mix1":           ("v3", dict(n_mix=1)),      # the screened v3 setting
    "mix2":           ("v3", dict(n_mix=2)),
    # --- normalisation: the L2 ablation says LayerNorm costs ~0.02 Pearson -
    "mix1_noln":      ("v3", dict(n_mix=1, use_ln=False)),
    "mix2_noln":      ("v3", dict(n_mix=2, use_ln=False)),
    # --- capacity ---------------------------------------------------------
    "mix1_noln_h256": ("v3", dict(n_mix=1, use_ln=False, hidden=256)),
    # --- regularisation / loss schedule -----------------------------------
    "mix1_noln_dr02": ("v3", dict(n_mix=1, use_ln=False, dropout=0.2)),
    "mix1_wrel03":    ("v3", dict(n_mix=1, w_rel=0.3)),
    "mix1_mf03":      ("v3", dict(n_mix=1, mse_frac=0.3)),
}


def config_of(name):
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name}; known: {sorted(CONFIGS)}")
    return CONFIGS[name]


def build(name, d_cell, d_drug):
    fam, kw = config_of(name)
    kw = dict(kw)
    kw.setdefault("d_cell", d_cell)
    kw.setdefault("d_drug", d_drug)
    cls = ComboTuned if fam == "v2" else ComboV3Tuned
    return PairDrugAdapter(cls(**kw), d_drug)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--protos", default=",".join(PROTOS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--select", default="rmse", choices=["rmse", "auroc"])
    ap.add_argument("--cell-dim", type=int, default=384)
    ap.add_argument("--drug-dim", type=int, default=128)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "l2_v3_tune.csv"))
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

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    done, rows = set(), []
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        if {"protocol", "config", "seed"}.issubset(prev.columns):
            done = {(r["protocol"], r["config"], int(r["seed"]))
                    for _, r in prev.iterrows() if np.isfinite(r.get("test_r", np.nan))}
            rows = prev.to_dict("records")
            print(f"[resume] {len(done)} finished rows", flush=True)

    resp, expr, morgan, ci, dri, dci, y, w = prep(args.drug_dim, args.cell_dim)
    base = os.path.join(EXTR, "L2", "splits")

    for P in [p.strip() for p in args.protos.split(",") if p.strip()]:
        if not any((P, n, s) not in done for n in names for s in seeds):
            continue
        tr = np.load(os.path.join(base, f"{P}_train.npy"))
        va = np.load(os.path.join(base, f"{P}_val.npy"))
        te = np.load(os.path.join(base, f"{P}_test.npy"))
        # identical representation pipeline to run_l2_unified.py
        cp = PCA(min(args.cell_dim, len(np.unique(ci[tr])) - 1, expr.shape[1]),
                 random_state=0).fit(expr[np.unique(ci[tr])])
        cd = cp.transform(expr).astype(np.float32)
        if cd.shape[1] < args.cell_dim:
            cd = np.hstack([cd, np.zeros((len(cd), args.cell_dim - cd.shape[1]), np.float32)])
        dp = PCA(min(args.drug_dim, morgan.shape[0]), random_state=0).fit(morgan)
        dd = dp.transform(morgan).astype(np.float32)
        drugs = np.hstack([dd[dri], dd[dci]]).astype(np.float32)
        tc = torch.from_numpy(cd).to(dev)
        td = torch.from_numpy(drugs).to(dev)
        ci_t = torch.from_numpy(ci).to(dev)
        di_t = torch.from_numpy(np.arange(len(resp)).astype(np.int64)).to(dev)
        di_np = di_t.cpu().numpy()
        print(f"[{P}] train={len(tr)} val={len(va)} test={len(te)} "
              f"cell={cd.shape[1]} drug={drugs.shape[1]}", flush=True)

        for cname in names:
            for seed in seeds:
                if (P, cname, seed) in done:
                    continue
                t0 = time.time()
                try:
                    model = build(cname, cd.shape[1], args.drug_dim).to(dev)
                    model = train_one(model, tr, va, ci, di_np, y, cd, drugs,
                                      w, None, seed, args.epochs, args.batch,
                                      patience=args.patience, select=args.select)
                    model.eval()
                    mu_v, sig_v, _, _ = infer(model, ci_t, di_t, tc, td, va)
                    mu_t, sig_t, _, _ = infer(model, ci_t, di_t, tc, td, te)
                    if not (np.isfinite(mu_v).all() and np.isfinite(mu_t).all()):
                        raise ValueError("non-finite predictions")
                    stud = bool(np.std(sig_t) > 1e-8 and np.isfinite(sig_t).all())
                    sc = np.abs(y[va] - mu_v) / (sig_v + 1e-8) if stud else np.abs(y[va] - mu_v)
                    q = q_conformal(sc)
                    pp, mm = picp_mpiw(y[te], mu_t, sig_t, q, stud)
                    row = {
                        "protocol": P, "config": cname, "seed": seed,
                        "val_r": round(float(np.corrcoef(y[va], mu_v)[0, 1]), 4),
                        "val_rmse": round(float(np.sqrt(np.mean((y[va] - mu_v) ** 2))), 4),
                        "test_r": round(float(np.corrcoef(y[te], mu_t)[0, 1]), 4),
                        "test_rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
                        "picp": round(pp, 4), "mpiw": round(mm, 4),
                        "epochs": args.epochs, "batch": args.batch,
                        "sec": round(time.time() - t0, 1),
                    }
                    print(f"[{P}] {cname:15s} s{seed}: val_r={row['val_r']:.4f} "
                          f"test_r={row['test_r']:.4f} rmse={row['test_rmse']:.4f} "
                          f"({row['sec']:.0f}s)", flush=True)
                except Exception as e:
                    row = {"protocol": P, "config": cname, "seed": seed,
                           "test_r": np.nan,
                           "error": f"{type(e).__name__}: {str(e)[:70]}"}
                    print(f"[{P}] {cname} s{seed} FAILED "
                          f"{type(e).__name__}: {str(e)[:90]}", flush=True)
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

    print("wrote", args.out)


if __name__ == "__main__":
    main()
