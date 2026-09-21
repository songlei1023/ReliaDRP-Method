"""Seeded re-run of the L3 (single-cell) unified panel.

Why this script exists
----------------------
`run_l3_unified.py` put all sixteen models on one harness and one recipe, but
it shared the *third* defect with L1 and L4: the model was constructed by the
caller and only afterwards handed to `run_benchmark.train_one`, which is where
`torch.manual_seed(seed)` runs.  The initial weights therefore came from
whatever the process RNG held at import time, so the L3 board --- like the L4
board before its re-run --- was decided partly by lucky draws.

L3 is currently the **only unseeded layer** left in the paper (see the
Reproducibility paragraph), so this script closes the last gap.

What is identical to the historical panel (`l3_unified_panel_e60b1024.csv`)
--------------------------------------------------------------------------
* the same sixteen rows: the fifteen `zoo` baselines plus our v2
  `ReliaDRPExpr` behind the thin `(cell, drug)` adapter, ignoring the drug
  tensor exactly as `run_l3_unified.ExprNet` does;
* the same data path: `run_layer_baselines.prep_l3()` (PCA-384 fit on the
  training split only, zero drug block, identity cell/drug indices);
* the same harness `run_benchmark.train_one` and the same recipe chosen in the
  audit without looking at the test set: 60 epochs / batch 1024 /
  val-AUROC early stopping;
* the same metric, AUROC, driving early stopping for every model, so the
  in-domain/drug-blind comparison stays like-for-like.

What is new
-----------
* `torch.manual_seed(seed)` / `np.random.seed(seed)` are called *immediately
  before* `zoo.build(...)` --- the actual fix;
* five seeds instead of three, matching L1/L2/L4, so every layer now reports a
  multi-seed mean with its seed sd;
* `val_auroc` and the wall-clock `sec` are recorded per row, and a device
  column is written when a CUDA error forces the CPU retry
  (`--cpu-fallback`, historical handling of `deepdtf`);
* resume-friendly: finished `(protocol, model, seed)` with a finite `auroc`
  are skipped, so an interrupted run can be restarted safely.

Usage
-----
    python run_l3_unified_seeded.py --seeds 0 --models reliadrp,delfos
    python run_l3_unified_seeded.py --seeds 0,1,2,3,4 --out l3_unified_seeded.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
sys.path.insert(0, os.path.join(HERE, "core"))  # 库模块在 core/
sys.path.insert(0, os.path.join(HERE, "benchmark"))  # 复用 run_benchmark 的 train_one/infer
sys.path.insert(0, HERE)

import model_zoo_2021_2026 as zoo  # noqa: E402
from run_benchmark import infer, picp_mpiw, q_conformal, train_one  # noqa: E402
from run_layer_baselines import prep_l3  # noqa: E402
from reliadrp import ReliaDRPExpr  # noqa: E402

PROTOS = ["L3_random", "L3_ldo"]

#: The fifteen external baselines, in the order used by the paper's L1 table.
EXTERNAL = ["clclsa", "mlp", "delfos", "attn", "pancancer", "mgataf", "mmcl",
            "bandrp", "codeae", "crdnn", "fourierdrug", "vaen", "mograph",
            "tgsa", "deepdtf"]
#: Our own L3 row: the v2 expression model behind the shared harness.
OURS = "reliadrp"


class ExprNet(nn.Module):
    """Copied verbatim from `run_l3_unified.py` so the paper row is produced by
    the same class, not a re-implementation."""

    def __init__(self, d_in, hidden=256, dropout=0.2, w_reg=0.05, mse_frac=0.5):
        super().__init__()
        self.inner = ReliaDRPExpr(d_in=d_in, hidden=hidden, dropout=dropout, w_reg=w_reg)
        self.mse_frac = mse_frac

    def forward(self, cell, drug):
        return self.inner(cell)

    def compute_loss(self, cell, drug, y, w=None, ctx=None, ep=0, epochs=40, mse_frac=None):
        return self.inner.compute_loss(cell, y, ep=ep, epochs=epochs,
                                       mse_frac=self.mse_frac if mse_frac is None else mse_frac)


def build(name, d_in, dev):
    """Build one model.  The caller must have seeded the RNG already."""
    if name == OURS:
        return ExprNet(d_in=d_in).to(dev)
    return zoo.build(name).to(dev)


def score(model, ci_t, di_t, tc, td, va, te, y):
    """Validation/test predictions -> the same metric block as the panel."""
    model.eval()
    mu_v, sig_v, _, _ = infer(model, ci_t, di_t, tc, td, va)
    mu_t, sig_t, _, _ = infer(model, ci_t, di_t, tc, td, te)
    if not (np.isfinite(mu_v).all() and np.isfinite(mu_t).all()):
        raise ValueError("non-finite predictions")
    stud = bool(np.std(sig_t) > 1e-8 and np.isfinite(sig_t).all())
    s_cal = (np.abs(y[va] - mu_v) / (sig_v + 1e-8) if stud else np.abs(y[va] - mu_v))
    q = q_conformal(s_cal)
    pp, mm = picp_mpiw(y[te], mu_t, sig_t, q, stud)
    return {
        "val_auroc": round(float(roc_auc_score(y[va], mu_v)), 4),
        "rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
        "auroc": round(float(roc_auc_score(y[te], mu_t)), 4),
        "acc": round(float(((mu_t >= 0.5) == (y[te] >= 0.5)).mean()), 4),
        "picp": round(pp, 4), "mpiw": round(mm, 4),
    }


def set_low_priority():
    try:
        import psutil
        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        print("[low-priority] BELOW_NORMAL set", flush=True)
    except Exception as e:  # pragma: no cover
        print("[low-priority] skipped:", e, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3dir", default=os.path.join(HERE, "data", "processed_ext", "L3"))
    ap.add_argument("--protos", default=",".join(PROTOS))
    ap.add_argument("--models", default=",".join(EXTERNAL + [OURS]))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--select", default="auroc", choices=["auroc", "rmse"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cpu-fallback", default="deepdtf")
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "l3_unified_seeded.csv"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.low_priority:
        set_low_priority()

    gpu = "cuda" if torch.cuda.is_available() else "cpu"
    fallback = {m.strip() for m in args.cpu_fallback.split(",") if m.strip()}
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    protos = [p.strip() for p in args.protos.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]

    done, rows = set(), []
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        if {"protocol", "model", "seed"}.issubset(prev.columns):
            done = {(r["protocol"], r["model"], int(r["seed"]))
                    for _, r in prev.iterrows()
                    if np.isfinite(pd.to_numeric(r.get("auroc"), errors="coerce"))}
            rows = prev.to_dict("records")
            print(f"[resume] {len(done)} finished rows in {os.path.basename(args.out)}",
                  flush=True)

    base = os.path.join(args.l3dir, "splits")
    meta, y, tier, make, task, w = prep_l3()
    assert task == "clf", "this script is for the classification layer"

    total = len(protos) * len(names) * len(seeds)
    k = 0
    t_start = time.time()
    for P in protos:
        tr = np.load(os.path.join(base, f"{P}_train.npy"))
        va = np.load(os.path.join(base, f"{P}_val.npy"))
        te = np.load(os.path.join(base, f"{P}_test.npy"))
        cells, drugs, ci, di = make(tr, va, te)
        print(f"[{P}] train={len(tr)} val={len(va)} test={len(te)} "
              f"feat={cells.shape[1]}", flush=True)

        for name in names:
            for seed in seeds:
                k += 1
                if (P, name, seed) in done:
                    print(f"  [{k}/{total}] {name:12s} seed{seed} (cached)", flush=True)
                    continue
                t0 = time.time()
                row = {"protocol": P, "model": name, "seed": seed,
                       "epochs": args.epochs, "batch": args.batch,
                       "select": args.select}
                dev_used = gpu

                def run(device):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = build(name, cells.shape[1], device)
                    model = train_one(model, tr, va, ci, di, y, cells, drugs,
                                      None, None, seed, args.epochs, args.batch,
                                      patience=args.patience, select=args.select)
                    tc = torch.from_numpy(cells.astype(np.float32)).to(device)
                    td = torch.from_numpy(drugs.astype(np.float32)).to(device)
                    ci_t = torch.from_numpy(ci.astype(np.int64)).to(device)
                    di_t = torch.from_numpy(di.astype(np.int64)).to(device)
                    res = score(model, ci_t, di_t, tc, td, va, te, y)
                    del model
                    return res

                try:
                    row.update(run(gpu))
                except Exception as e:
                    msg = f"{type(e).__name__}: {str(e)[:90]}"
                    if ("CUDA" in str(e) or "cuda" in str(e)) and name in fallback and gpu == "cuda":
                        print(f"  [{k}/{total}] {name:12s} seed{seed} "
                              f"CUDA error -> retry on CPU", flush=True)
                        try:
                            torch.cuda.empty_cache()
                            row.update(run("cpu"))
                            dev_used = "cpu"
                            msg = ""
                        except Exception as e2:
                            msg = f"CPU retry {type(e2).__name__}: {str(e2)[:70]}"
                    if msg:
                        row.update({"auroc": np.nan, "val_auroc": np.nan, "error": msg})
                row["device"] = dev_used
                row["sec"] = round(time.time() - t0, 1)
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out, index=False)

                ok = np.isfinite(pd.to_numeric(row.get("auroc"), errors="coerce"))
                if ok:
                    el = time.time() - t_start
                    fin = sum(1 for r in rows
                              if np.isfinite(pd.to_numeric(r.get("auroc"), errors="coerce")))
                    print(f"  [{k}/{total}] {name:12s} seed{seed}: "
                          f"val={row['val_auroc']:.4f} test_auroc={row['auroc']:.4f} "
                          f"acc={row['acc']:.4f} dev={dev_used} ({row['sec']}s) | "
                          f"{fin} done, elapsed {el/60:.1f} min, "
                          f"eta {el/max(fin,1)*(total-fin)/60:.1f} min", flush=True)
                else:
                    print(f"  [{k}/{total}] {name:12s} seed{seed} FAILED: "
                          f"{row.get('error')}", flush=True)
                torch.cuda.empty_cache()
        del cells, drugs, ci, di
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print("wrote", args.out)
    if "auroc" in df.columns:
        g = (df.dropna(subset=["auroc"]).groupby(["protocol", "model"])["auroc"]
             .agg(["mean", "std", "count"]))
        for P in protos:
            if P in g.index.get_level_values(0):
                sub = g.loc[P].sort_values("mean", ascending=False)
                rank = (list(sub.index).index(OURS) + 1) if OURS in sub.index else None
                print(f"\n=== {P} (rank of {OURS}: {rank}/{len(sub)}) ===")
                print(sub.round(4))


if __name__ == "__main__":
    main()
