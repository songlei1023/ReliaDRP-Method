"""L1 (EXT monotherapy) hyper-parameter tuning of the v3 model.

Context
-------
The paper's L1 row is `ReliaDRPV3` (ReliaDRP + residual TokenMixer over the three
omics-channel tokens).  On L1 the previous v3 setting (`n_mix=2`, everything else
at its default) already ranked 1/16 on all three EXT protocols with
random 0.7562 / LDO 0.3526 / LCO 0.6611 --- but that setting was picked from a
four-variant, single-seed screen.  This script does a proper sweep.

Fairness rules kept from the earlier audits
-------------------------------------------
1. **Same harness, same recipe, same budget as the baselines.**  Every run goes
   through `run_benchmark.train_one` with 10 epochs / batch 4096 / validation-RMSE
   early stopping, which is exactly the recipe `run_extended_benchmark.py` uses
   for the fifteen external models.  Only *our own* architecture
   hyper-parameters are swept, so the budget cannot be tuned in our favour.
2. **No weights.**  The L1-extended panel has no per-sample reliability weight
   (`response_ext_eligible.csv` has no `reliability_w` column), so `w=None` for
   every model --- same as the baseline runs.
3. **Validation-based selection is checked, not assumed.**  The sweep records
   both validation and test Pearson for every config, so we can report the
   val-selected config *and* the setting-free mean, plus the val--test
   correlation that tells us whether selection is usable at all.

Usage
-----
    python run_l1_v3_tune.py --seeds 0 --out l1_v3_tune_stage1.csv
    python run_l1_v3_tune.py --configs mix2,mix2_h256 --seeds 0,1,2,3,4 \
        --out l1_v3_tune_confirm.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
sys.path.insert(0, os.path.join(HERE, "core"))  # 库模块在 core/
sys.path.insert(0, os.path.join(HERE, "benchmark"))  # 复用 run_benchmark 的 train_one/infer
sys.path.insert(0, HERE)

from run_benchmark import train_one, infer  # noqa: E402
from reliadrp_v3 import ReliaDRPV3  # noqa: E402

PROTOS = ["EXT_random", "EXT_LCO", "EXT_LDO"]


class V3Tuned(ReliaDRPV3):
    """ReliaDRPV3 with `mse_frac` promoted to a constructor knob.

    `ReliaDRPV3.compute_loss` takes `mse_frac=0.5` as a parameter rather than an
    attribute, so it cannot be swept from outside without this thin wrapper.
    Everything else is untouched.
    """

    def __init__(self, mse_frac=0.5, **kw):
        super().__init__(**kw)
        self.mse_frac = mse_frac

    def compute_loss(self, cell, drug, y, w=None, ctx=None, ep=0, epochs=10):
        return super().compute_loss(cell, drug, y, w=w, ctx=ctx, ep=ep,
                                    epochs=epochs, mse_frac=self.mse_frac)


# name -> kwargs.  `mix2` is the configuration the paper currently uses, kept as
# the reference point of the whole sweep.
CONFIGS = {
    # --- mixer depth (the mechanism that produced the original gain) --------
    "mix1":          dict(n_mix=1),
    "mix2":          dict(n_mix=2),                     # current paper setting
    "mix3":          dict(n_mix=3),
    "mix4":          dict(n_mix=4),
    # --- regularisation / capacity ----------------------------------------
    "mix2_dr02":     dict(n_mix=2, dropout=0.2),
    "mix2_dr03":     dict(n_mix=2, dropout=0.3),
    "mix2_h256":     dict(n_mix=2, d_hid=256, d_fus=256),
    "mix2_h256dr02": dict(n_mix=2, d_hid=256, d_fus=256, dropout=0.2),
    "mix2_md01":     dict(n_mix=2, mod_drop=0.1),
    "mix2_md02":     dict(n_mix=2, mod_drop=0.2),
    # --- loss schedule / weights ------------------------------------------
    "mix2_mf03":     dict(n_mix=2, mse_frac=0.3),
    "mix2_mf07":     dict(n_mix=2, mse_frac=0.7),
    "mix2_wreg02":   dict(n_mix=2, w_reg=0.2),
    "mix2_wrel03":   dict(n_mix=2, w_rel=0.3),
    # --- ablations (informative, not candidates for selection) -------------
    "mix2_nobn":     dict(n_mix=2, use_bn=False),
    "mix2_nointer":  dict(n_mix=2, interact=False),
    "mix2_ch1lay2":  dict(n_mix=2, ch_layers=2),
}


def config_of(name):
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name}; known: {sorted(CONFIGS)}")
    return CONFIGS[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extdir", default=os.path.join(HERE, "data", "processed_ext"))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--protos", default=",".join(PROTOS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "l1_v3_tune.csv"))
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
    seeds = [int(s) for s in args.seeds.split(",")]

    done, rows = set(), []
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        if {"proto", "config", "seed"}.issubset(prev.columns):
            done = {(r["proto"], r["config"], int(r["seed"]))
                    for _, r in prev.iterrows() if np.isfinite(r.get("test_r", np.nan))}
            rows = prev.to_dict("records")
            print(f"[resume] {len(done)} finished rows", flush=True)

    d = args.extdir
    spl = os.path.join(d, "splits")
    elig = pd.read_csv(os.path.join(d, "response_ext_eligible.csv"))
    ci = np.load(os.path.join(d, "pair_cell_row.npy"))
    di = np.load(os.path.join(d, "pair_drug_row.npy"))
    y = elig["y_auc"].to_numpy(np.float32)

    for P in [p.strip() for p in args.protos.split(",") if p.strip()]:
        if not any((P, n, s) not in done for n in names for s in seeds):
            continue
        tr = np.load(os.path.join(spl, f"{P}_train.npy"))
        va = np.load(os.path.join(spl, f"{P}_val.npy"))
        te = np.load(os.path.join(spl, f"{P}_test.npy"))
        cells = np.load(os.path.join(spl, f"{P}_cells.npy"))
        drugs = np.load(os.path.join(spl, f"{P}_drugs.npy"))
        tc = torch.from_numpy(cells.astype(np.float32)).to(dev)
        td = torch.from_numpy(drugs.astype(np.float32)).to(dev)
        ci_t = torch.from_numpy(ci.astype(np.int64)).to(dev)
        di_t = torch.from_numpy(di.astype(np.int64)).to(dev)
        print(f"[{P}] train={len(tr)} val={len(va)} test={len(te)} "
              f"cell={cells.shape[1]} drug={drugs.shape[1]}", flush=True)

        for cname in names:
            kw = config_of(cname)
            for seed in seeds:
                if (P, cname, seed) in done:
                    continue
                t0 = time.time()
                try:
                    model = V3Tuned(**kw).to(dev)
                    model = train_one(model, tr, va, ci, di, y, cells, drugs,
                                      None, None, seed, args.epochs, args.batch)
                    model.eval()
                    mu_v, _, _, _ = infer(model, ci_t, di_t, tc, td, va)
                    mu_t, _, _, _ = infer(model, ci_t, di_t, tc, td, te)
                    if not (np.isfinite(mu_v).all() and np.isfinite(mu_t).all()):
                        raise ValueError("non-finite predictions")
                    row = {
                        "proto": P, "config": cname, "seed": seed,
                        "val_r": round(float(np.corrcoef(y[va], mu_v)[0, 1]), 4),
                        "val_rmse": round(float(np.sqrt(np.mean((y[va] - mu_v) ** 2))), 4),
                        "test_r": round(float(np.corrcoef(y[te], mu_t)[0, 1]), 4),
                        "test_rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
                        "epochs": args.epochs, "batch": args.batch,
                        "sec": round(time.time() - t0, 1),
                    }
                    print(f"[{P}] {cname:14s} s{seed}: val_r={row['val_r']:.4f} "
                          f"test_r={row['test_r']:.4f} rmse={row['test_rmse']:.4f} "
                          f"({row['sec']}s)", flush=True)
                except Exception as e:
                    row = {"proto": P, "config": cname, "seed": seed,
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
        del tc, td, ci_t, di_t
        if dev == "cuda":
            torch.cuda.empty_cache()

    print("wrote", args.out)


if __name__ == "__main__":
    main()
