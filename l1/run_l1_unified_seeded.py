"""Seeded, single-process rerun of the whole L1 (EXT monotherapy) panel.

Why this script exists
----------------------
The paper's L1 table is assembled from **two different processes**:

  * the fifteen external rows come from `run_extended_benchmark.py`
    (`extended_benchmark_all.csv`), and
  * our own row comes from `run_l1_v3_tune.py` (`l1_v3_tune_confirm.csv`,
    config `mix2_mf03`).

Both used `run_benchmark.train_one`, but *neither* seeded the model
initialisation: `train_one` calls `torch.manual_seed(seed)` **after** the
caller has already built the model, so the initial weights come from whatever
the process RNG happened to hold at import/construction time.  Measured
consequence on L1: the same architecture in two different processes drifts by
up to 0.032 Pearson (EXT_random), while the in-domain margin over the best
baseline is only 0.049.

This script removes that degree of freedom: it runs **every model of the L1
table in one process, with the initialisation seeded**, so the whole board is
reproducible and mutually comparable.

Design notes
------------
1. `torch.manual_seed(seed)` / `np.random.seed(seed)` are called immediately
   *before* model construction --- the actual defect fix.
2. Same harness, same recipe, same budget as the historical table:
   `train_one(..., epochs=10, batch=4096, select="rmse")`.
3. `deepdtf` historically raised `CUDA error: invalid configuration argument`
   on EXT_LDO/EXT_LCO and was therefore run on CPU (see
   `extended_deepdtf_cpu.csv`).  That device policy is replicated here and the
   device used is recorded per row, so the table stays auditable.
4. Resume-friendly: finished (model, proto, seed) triples with a finite
   `test_r` are skipped, so an interrupted run can be restarted safely.

Usage
-----
    python run_l1_unified_seeded.py --seeds 0 --models clclsa,reliadrpv3mix2mf03
    python run_l1_unified_seeded.py --seeds 0,1,2,3,4 --out l1_unified_seeded.csv
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

import model_zoo_2021_2026 as zoo  # noqa: E402
from run_benchmark import (ALPHA, infer, picp_mpiw, q_conformal,  # noqa: E402
                           train_one)
from reliadrp_v3 import ReliaDRPV3  # noqa: E402

PROTOS = ["EXT_random", "EXT_LCO", "EXT_LDO"]

#: The fifteen external baselines of the paper table, in table order.
EXTERNAL = ["clclsa", "mlp", "delfos", "attn", "pancancer", "mgataf", "mmcl",
            "bandrp", "codeae", "crdnn", "fourierdrug", "vaen", "mograph",
            "tgsa", "deepdtf"]
#: Internal controls --- not paper rows, but they keep our own audit notes
#: (`evi2` = non-reliability basis; `reliadrp` = v2 basis) valid under seeding.
CONTROLS = ["reliadrp", "evi2"]
#: Our own paper row: ReliaDRPV3 with the L1-tuned setting.
OURS = "reliadrpv3mix2mf03"


class V3Tuned(ReliaDRPV3):
    """`ReliaDRPV3` with `mse_frac` promoted to a constructor argument.

    Copied verbatim from `run_l1_v3_tune.py` so that the paper row is produced
    by the exact same class, not a re-implementation.
    """

    def __init__(self, mse_frac=0.5, **kw):
        super().__init__(**kw)
        self.mse_frac = mse_frac

    def compute_loss(self, cell, drug, y, w=None, ctx=None, ep=0, epochs=10):
        return super().compute_loss(cell, drug, y, w=w, ctx=ctx, ep=ep,
                                    epochs=epochs, mse_frac=self.mse_frac)


def build(name, dev):
    """Build one model.  Caller must have seeded the RNG already."""
    if name == OURS:
        return V3Tuned(n_mix=2, mse_frac=0.3).to(dev)
    return zoo.build(name).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extdir", default=os.path.join(HERE, "data", "processed_ext"))
    ap.add_argument("--protos", default=",".join(PROTOS))
    ap.add_argument("--models",
                    default=",".join(EXTERNAL + CONTROLS + [OURS]))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cpu-fallback", default="deepdtf",
                    help="comma list of models allowed to retry on CPU after a "
                         "CUDA error (historical handling of deepdtf)")
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "l1_unified_seeded.csv"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.low_priority:
        try:
            import psutil
            psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("[low-priority] BELOW_NORMAL set", flush=True)
        except Exception as e:  # pragma: no cover
            print("[low-priority] skipped:", e, flush=True)

    gpu = "cuda" if torch.cuda.is_available() else "cpu"
    fallback = {m.strip() for m in args.cpu_fallback.split(",") if m.strip()}
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    protos = [p.strip() for p in args.protos.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]

    done, rows = set(), []
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        if {"proto", "model", "seed"}.issubset(prev.columns):
            done = {(r["proto"], r["model"], int(r["seed"]))
                    for _, r in prev.iterrows()
                    if np.isfinite(pd.to_numeric(r.get("test_r"), errors="coerce"))}
            rows = prev.to_dict("records")
            print(f"[resume] {len(done)} finished rows", flush=True)

    d = args.extdir
    spl = os.path.join(d, "splits")
    elig = pd.read_csv(os.path.join(d, "response_ext_eligible.csv"))
    ci = np.load(os.path.join(d, "pair_cell_row.npy"))
    di = np.load(os.path.join(d, "pair_drug_row.npy"))
    y = elig["y_auc"].to_numpy(np.float32)

    total = len(protos) * len(names) * len(seeds)
    k = 0
    t_start = time.time()
    for P in protos:
        tr = np.load(os.path.join(spl, f"{P}_train.npy"))
        va = np.load(os.path.join(spl, f"{P}_val.npy"))
        te = np.load(os.path.join(spl, f"{P}_test.npy"))
        cells = np.load(os.path.join(spl, f"{P}_cells.npy"))
        drugs = np.load(os.path.join(spl, f"{P}_drugs.npy"))

        # Same as the historical table: a 16-cluster cell context for CODE-AE,
        # fitted deterministically from the training split only.
        ctx = None
        if "codeae" in names:
            from sklearn.cluster import KMeans
            sub = np.random.RandomState(0).choice(
                np.unique(ci[tr]), min(2000, len(np.unique(ci[tr]))), replace=False)
            km = KMeans(n_clusters=16, random_state=0, n_init=3).fit(cells[sub])
            ctx = km.predict(cells)[ci].astype(np.int64)

        tc = torch.from_numpy(cells.astype(np.float32)).to(gpu)
        td = torch.from_numpy(drugs.astype(np.float32)).to(gpu)
        ci_t = torch.from_numpy(ci.astype(np.int64)).to(gpu)
        di_t = torch.from_numpy(di.astype(np.int64)).to(gpu)
        print(f"[{P}] train={len(tr)} val={len(va)} test={len(te)} "
              f"cell={cells.shape[1]} drug={drugs.shape[1]}", flush=True)

        for name in names:
            for seed in seeds:
                k += 1
                if (P, name, seed) in done:
                    print(f"  [{k}/{total}] {name:20s} seed{seed} (cached)",
                          flush=True)
                    continue
                t0 = time.time()
                row = {"proto": P, "model": name, "seed": seed,
                       "epochs": args.epochs, "batch": args.batch}
                dev_used = gpu
                try:
                    # ---- the seeding fix: seed BEFORE the weights exist ----
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = build(name, gpu)
                    model = train_one(model, tr, va, ci, di, y, cells, drugs,
                                      None, ctx, seed, args.epochs, args.batch)
                    model.eval()
                    mu_v, sig_v, _, _ = infer(model, ci_t, di_t, tc, td, va)
                    mu_t, sig_t, _, _ = infer(model, ci_t, di_t, tc, td, te)
                    if not (np.isfinite(mu_v).all() and np.isfinite(mu_t).all()):
                        raise ValueError("non-finite predictions")
                    stud = bool(np.std(sig_t) > 1e-8 and np.isfinite(sig_t).all())
                    s_cal = (np.abs(y[va] - mu_v) / (sig_v + 1e-8) if stud
                             else np.abs(y[va] - mu_v))
                    q = q_conformal(s_cal)
                    picp, mpiw = picp_mpiw(y[te], mu_t, sig_t, q, stud)
                    row.update({
                        "val_r": round(float(np.corrcoef(y[va], mu_v)[0, 1]), 4),
                        "val_rmse": round(float(np.sqrt(np.mean((y[va] - mu_v) ** 2))), 4),
                        "test_r": round(float(np.corrcoef(y[te], mu_t)[0, 1]), 4),
                        "test_rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
                        "picp": round(picp, 4), "mpiw": round(mpiw, 4),
                        "device": dev_used,
                        "sec": round(time.time() - t0, 1),
                    })
                except Exception as e:
                    msg = f"{type(e).__name__}: {str(e)[:90]}"
                    cuda_err = "CUDA" in str(e) or "cuda" in str(e)
                    if cuda_err and name in fallback and gpu == "cuda":
                        # Historical policy for deepdtf: retry this cell on CPU.
                        print(f"  [{k}/{total}] {name:20s} seed{seed} "
                              f"CUDA error -> retry on CPU", flush=True)
                        try:
                            torch.cuda.empty_cache()
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            model = build(name, "cpu")
                            model = train_one(model, tr, va, ci, di, y, cells,
                                              drugs, None, ctx, seed,
                                              args.epochs, args.batch)
                            model.eval()
                            tcc = torch.from_numpy(cells.astype(np.float32))
                            tdd = torch.from_numpy(drugs.astype(np.float32))
                            ci_c = torch.from_numpy(ci.astype(np.int64))
                            di_c = torch.from_numpy(di.astype(np.int64))
                            mu_v, sig_v, _, _ = infer(model, ci_c, di_c, tcc, tdd, va)
                            mu_t, sig_t, _, _ = infer(model, ci_c, di_c, tcc, tdd, te)
                            if not (np.isfinite(mu_v).all() and np.isfinite(mu_t).all()):
                                raise ValueError("non-finite predictions")
                            stud = bool(np.std(sig_t) > 1e-8 and np.isfinite(sig_t).all())
                            s_cal = (np.abs(y[va] - mu_v) / (sig_v + 1e-8) if stud
                                     else np.abs(y[va] - mu_v))
                            q = q_conformal(s_cal)
                            picp, mpiw = picp_mpiw(y[te], mu_t, sig_t, q, stud)
                            row.update({
                                "val_r": round(float(np.corrcoef(y[va], mu_v)[0, 1]), 4),
                                "val_rmse": round(float(np.sqrt(np.mean((y[va] - mu_v) ** 2))), 4),
                                "test_r": round(float(np.corrcoef(y[te], mu_t)[0, 1]), 4),
                                "test_rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
                                "picp": round(picp, 4), "mpiw": round(mpiw, 4),
                                "device": "cpu",
                                "sec": round(time.time() - t0, 1),
                            })
                            msg = ""
                        except Exception as e2:
                            msg = f"CPU retry {type(e2).__name__}: {str(e2)[:70]}"
                    if msg:
                        row.update({"test_r": np.nan, "val_r": np.nan,
                                    "error": msg})
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out, index=False)
                if np.isfinite(pd.to_numeric(row.get("test_r"), errors="coerce")):
                    el = time.time() - t_start
                    fin = sum(1 for r in rows
                              if np.isfinite(pd.to_numeric(r.get("test_r"), errors="coerce")))
                    print(f"  [{k}/{total}] {name:20s} seed{seed}: "
                          f"val_r={row['val_r']:.4f} test_r={row['test_r']:.4f} "
                          f"rmse={row['test_rmse']:.4f} dev={row['device']} "
                          f"({row['sec']}s) | {fin} done, elapsed {el/60:.1f} min, "
                          f"eta {el/max(fin,1)*(total-fin)/60:.1f} min", flush=True)
                else:
                    print(f"  [{k}/{total}] {name:20s} seed{seed} FAILED: "
                          f"{row.get('error')}", flush=True)
                try:
                    del model
                except Exception:
                    pass
                torch.cuda.empty_cache()
        del tc, td, ci_t, di_t
        torch.cuda.empty_cache()

    print("wrote", args.out)


if __name__ == "__main__":
    main()
