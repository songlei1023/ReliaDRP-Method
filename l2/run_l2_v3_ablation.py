"""L2 mean-drug ablation for the v3 generation (the missing cell in the paper).

Why this script exists
----------------------
The paper's L2 section makes one mechanistic claim:

    "The drug-pair interaction encoder ... is what carries the shift:
     replacing it with the single-drug model fed the mean of the two
     embeddings costs 0.065 drug-blind (0.532 -> 0.467) and 0.077
     cold-cell (0.657 -> 0.580)."

Those two deltas are **v2** numbers.  L1 and L4 already report the v3
generation, and the L2 row is being moved to v3 as well, so the ablation has to
exist on the same generation --- otherwise the section mixes a v3 headline with
a v2 ablation.

What exactly is ablated
-----------------------
The v2 ablation was generated inside `run_l2_unified.py`, which serves every
model through `train_one` with a single drug tensor.  Two adapters decide the
layout:

    MeanDrugAdapter  : (dr + dc) / 2  -> single-drug model   <-- the ablation
    PairDrugAdapter  : (dr, dc)      -> ReliaDRPCombo       <-- the full model

So the ablation is literally "same harness, same recipe, but the drug pair is
collapsed to its mean before it reaches the network".  This script reuses those
exact adapters, so the v3 ablation is the same intervention on the same data.

Fairness rules (identical to `run_l2_v3_tune.py`)
-------------------------------------------------
1. `train_one`, 12 epochs, batch 4096, validation-RMSE early stopping --- the
   recipe used for all fifteen external L2 baselines.
2. Identical representation pipeline (cell PCA on training cells, drug PCA on
   all drugs, `[dr | dc]` tensor).
3. **A capacity-matched combo reference is run in this same process.**  The
   paper's L2 v3 row is `mix1_noln_h256` (LayerNorm off, hidden 256), which is
   *not* capacity-matched to a hidden-128 single-drug model; comparing it to a
   mean-drug model would confound "pair interaction" with "capacity".  So the
   grid includes `v3_combo_mix1` (default hidden 128), whose only difference
   from `v3mean_mix1` is the pair interaction encoder.
4. **Initialisation is seeded before the model is built.**  The third harness
   defect (unseeded init) was fixed for the L4 re-run; applying it here too
   keeps the ablation rows reproducible.

Usage
-----
    python run_l2_v3_ablation.py --seeds 0,1,2,3,4 --threads 4 \
        --out l2_v3_ablation.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
sys.path.insert(0, os.path.join(HERE, "core"))  # 库模块在 core/
sys.path.insert(0, os.path.join(HERE, "benchmark"))  # 复用 run_benchmark 的 train_one/infer
sys.path.insert(0, HERE)

from run_benchmark import train_one, infer, q_conformal, picp_mpiw  # noqa: E402
from reliadrp import ReliaDRP, ReliaDRPCombo  # noqa: E402
from reliadrp_v3 import ReliaDRPV3, ReliaDRPComboV3  # noqa: E402
from run_l2_unified import MeanDrugAdapter, PairDrugAdapter, prep, EXTR  # noqa: E402

PROTOS = ["L2_random", "L2_LCO", "L2_LDO"]


# --------------------------------------------------------------------------
# `mse_frac` is a `compute_loss` parameter, not a constructor attribute, so the
# thin wrappers below promote it.  Mirrors `run_l1_v3_tune.V3Tuned` and
# `run_l2_v3_tune.ComboV3Tuned` exactly.
# --------------------------------------------------------------------------
class SingleV2(ReliaDRP):
    """ReliaDRP with `mse_frac` promoted (v2 single-drug reference)."""

    def __init__(self, mse_frac=0.5, **kw):
        super().__init__(**kw)
        self.mse_frac = mse_frac

    def compute_loss(self, c, d, y, w=None, ctx=None, ep=0, epochs=12):
        return super().compute_loss(c, d, y, w=w, ep=ep, epochs=epochs,
                                    mse_frac=self.mse_frac)


class SingleV3(ReliaDRPV3):
    """ReliaDRPV3 with `mse_frac` promoted (v3 single-drug model)."""

    def __init__(self, mse_frac=0.5, **kw):
        super().__init__(**kw)
        self.mse_frac = mse_frac

    def compute_loss(self, c, d, y, w=None, ctx=None, ep=0, epochs=12):
        return super().compute_loss(c, d, y, w=w, ctx=ctx, ep=ep, epochs=epochs,
                                    mse_frac=self.mse_frac)


class ComboV2Tuned(ReliaDRPCombo):
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


# name -> (layout, factory).  `layout` picks the adapter, so the intervention
# (mean vs pair) is visible in the config table of the report.
CONFIGS = {
    # ---- the full model, both generations -------------------------------
    "v2_combo":        ("pair", lambda d_cell, d_drug: ComboV2Tuned(
        d_cell=d_cell, d_drug=d_drug)),
    "v3_combo_mix1":   ("pair", lambda d_cell, d_drug: ComboV3Tuned(
        d_cell=d_cell, d_drug=d_drug, hidden=128, n_mix=1)),
    # ---- the ablation: same backbone fed the *mean* of the two drugs -----
    "v2_mean":         ("mean", lambda d_cell, d_drug: SingleV2(
        cell_dim=d_cell, drug_dim=d_drug)),
    "v3mean_mix1":     ("mean", lambda d_cell, d_drug: SingleV3(
        cell_dim=d_cell, drug_dim=d_drug, n_mix=1)),
    "v3mean_mix2":     ("mean", lambda d_cell, d_drug: SingleV3(
        cell_dim=d_cell, drug_dim=d_drug, n_mix=2)),
    # the config L1 actually reports (n_mix=2 + the tuned loss schedule)
    "v3mean_mix2_mf03": ("mean", lambda d_cell, d_drug: SingleV3(
        cell_dim=d_cell, drug_dim=d_drug, n_mix=2, mse_frac=0.3)),
    # capacity raised to 256 so it can be paired against the paper's L2 row
    # `mix1_noln_h256`
    "v3mean_mix1_h256": ("mean", lambda d_cell, d_drug: SingleV3(
        cell_dim=d_cell, drug_dim=d_drug, n_mix=1, d_hid=256, d_fus=256)),
}


def build(name, d_cell, d_drug):
    layout, fac = CONFIGS[name]
    inner = fac(d_cell, d_drug)
    if layout == "pair":
        return PairDrugAdapter(inner, d_drug)
    return MeanDrugAdapter(inner, d_drug)


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
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "l2_v3_ablation.csv"))
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
        # identical representation pipeline to run_l2_unified.py / run_l2_v3_tune.py
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
                    # --- seeding fix: init must not depend on process RNG ---
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = build(cname, cd.shape[1], args.drug_dim).to(dev)
                    n_param = sum(p.numel() for p in model.parameters())
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
                        "n_param": n_param,
                        "epochs": args.epochs, "batch": args.batch,
                        "sec": round(time.time() - t0, 1),
                    }
                    print(f"[{P}] {cname:18s} s{seed}: val_r={row['val_r']:.4f} "
                          f"test_r={row['test_r']:.4f} rmse={row['test_rmse']:.4f} "
                          f"({row['sec']:.0f}s, {n_param/1e3:.0f}k params)", flush=True)
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
