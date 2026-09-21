"""Seeded re-run of the L3 twelve-configuration hyper-parameter sweep.

Why this script exists
----------------------
The paper's L3 subsection cites two statistics: the unified 16-model panel
(`l3_unified_panel_e60b1024.csv`) and "a twelve-setting sweep averages
0.716 +/- 0.021".  The sweep came from `_l3_v2_confirm.py`, which shares the
third defect --- the model is constructed by the caller and only then handed to
`run_benchmark.train_one`, where `torch.manual_seed(seed)` runs, so the initial
weights come from the process RNG.  Re-seeding only the panel would leave the
two statistics of the *same* subsection on different reproducibility footings,
so the sweep is re-run here under the same fix.

What is identical
-----------------
* the same twelve configurations: `hidden in {256,512}` x
  `dropout in {0.1,0.2,0.3}` x `w_reg = 0.05` x `mse_frac in {0.3,0.5}`;
* the same model (`run_l3_unified.ExprNet`, i.e. the v2 `ReliaDRPExpr`
  behind the shared `(cell, drug)` adapter) and the same PCA-384 fitted on the
  training split only;
* the same harness and recipe: `train_one`, 60 epochs, batch 1024,
  val-AUROC early stopping (patience 8);
* the same discipline: hyper-parameters are selected on validation AUROC only.

What is new
-----------
* `torch.manual_seed(seed)` / `np.random.seed(seed)` immediately **before**
  `ExprNet(...)` --- the fix;
* resume-friendly: a finished row is a (protocol, config, seed) tuple already
  present in --out, so an interrupted sweep restarts safely.

Usage
-----
    python run_l3_seeded_sweep.py --batch 1024 --out l3_sweep_seeded_b1024.csv
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_benchmark import infer, train_one  # noqa: E402
from run_l3_unified_seeded import ExprNet  # noqa: E402

L3DIR = os.path.join(HERE, "data", "processed_ext", "L3")
KEY = ["hidden", "dropout", "w_reg", "mse_frac"]


def load_protocol(P, dim, dev):
    """Splits + train-only PCA, identical to `_l3_v2_confirm.load_protocol`."""
    sp = os.path.join(L3DIR, "splits")
    tr = np.load(os.path.join(sp, f"{P}_train.npy"))
    va = np.load(os.path.join(sp, f"{P}_val.npy"))
    te = np.load(os.path.join(sp, f"{P}_test.npy"))
    y = np.load(os.path.join(L3DIR, "y_l3.npy")).astype(np.float32)
    raw = np.load(os.path.join(L3DIR, "cells_l3.npy")).astype(np.float32)
    n = min(dim, len(tr) - 1, raw.shape[1])
    X = PCA(n, random_state=0).fit(raw[tr]).transform(raw).astype(np.float32)
    drugs = np.zeros((len(raw), 128), np.float32)
    ci = np.arange(len(raw))
    di = np.arange(len(raw))
    return dict(tr=tr, va=va, te=te, y=y, X=X, drugs=drugs, ci=ci, di=di,
                tc=torch.from_numpy(X).to(dev),
                td=torch.from_numpy(drugs).to(dev),
                ci_t=torch.from_numpy(ci.astype(np.int64)).to(dev),
                di_t=torch.from_numpy(di.astype(np.int64)).to(dev),
                d_in=X.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="L3_random,L3_ldo")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--select", default="auroc", choices=["auroc", "rmse"])
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--hidden", default="256,512")
    ap.add_argument("--dropout", default="0.1,0.2,0.3")
    ap.add_argument("--wreg", default="0.05")
    ap.add_argument("--msefrac", default="0.3,0.5")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--low-priority", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "l3_sweep_seeded_b1024.csv"))
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

    grid = list(itertools.product(
        [int(x) for x in args.hidden.split(",")],
        [float(x) for x in args.dropout.split(",")],
        [float(x) for x in args.wreg.split(",")],
        [float(x) for x in args.msefrac.split(",")]))
    seeds = [int(s) for s in args.seeds.split(",")]
    protos = [p.strip() for p in args.protocols.split(",")]
    print(f"{len(grid)} configs x {len(protos)} protocols x {len(seeds)} seeds "
          f"= {len(grid) * len(protos) * len(seeds)} runs "
          f"(epochs={args.epochs} batch={args.batch} select={args.select})", flush=True)

    rows = []
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        rows = prev.to_dict("records")
        print(f"[resume] {len(rows)} rows in {os.path.basename(args.out)}", flush=True)

    def key_of(r):
        return (r["protocol"], float(r["hidden"]), float(r["dropout"]),
                float(r["w_reg"]), float(r["mse_frac"]), int(r["seed"]))

    done = {key_of(r) for r in rows
            if np.isfinite(pd.to_numeric(r.get("test_auroc"), errors="coerce"))}
    total = len(grid) * len(protos) * len(seeds)
    k = len(done)
    t_start = time.time()

    for P in protos:
        t0 = time.time()
        D = load_protocol(P, args.dim, dev)
        print(f"[{P}] train={len(D['tr'])} val={len(D['va'])} test={len(D['te'])} "
              f"feat={D['d_in']} (setup {time.time() - t0:.0f}s)", flush=True)
        for hidden, dropout, w_reg, mse_frac in grid:
            for seed in seeds:
                kk = (P, float(hidden), float(dropout), float(w_reg),
                      float(mse_frac), int(seed))
                if kk in done:
                    continue
                t1 = time.time()
                # ---- the seeding fix: seed BEFORE the weights exist ----
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = ExprNet(d_in=D["d_in"], hidden=hidden, dropout=dropout,
                                w_reg=w_reg, mse_frac=mse_frac).to(dev)
                model = train_one(model, D["tr"], D["va"], D["ci"], D["di"], D["y"],
                                  D["X"], D["drugs"], None, None, seed,
                                  args.epochs, args.batch,
                                  patience=args.patience, select=args.select)
                model.eval()
                mu_v, _, _, _ = infer(model, D["ci_t"], D["di_t"], D["tc"], D["td"], D["va"])
                mu_t, _, _, _ = infer(model, D["ci_t"], D["di_t"], D["tc"], D["td"], D["te"])
                rows.append({"protocol": P, "hidden": hidden, "dropout": dropout,
                             "w_reg": w_reg, "mse_frac": mse_frac, "seed": seed,
                             "epochs": args.epochs, "batch": args.batch,
                             "val_auroc": round(float(roc_auc_score(D["y"][D["va"]], mu_v)), 4),
                             "test_auroc": round(float(roc_auc_score(D["y"][D["te"]], mu_t)), 4),
                             "sec": round(time.time() - t1, 1)})
                k += 1
                pd.DataFrame(rows).to_csv(args.out, index=False)
                el = time.time() - t_start
                print(f"  [{k}/{total}] {P} h{hidden} do{dropout} wr{w_reg} mf{mse_frac} "
                      f"s{seed}: val={rows[-1]['val_auroc']:.4f} "
                      f"test={rows[-1]['test_auroc']:.4f} ({rows[-1]['sec']}s) | "
                      f"elapsed {el/60:.1f} min, eta {el/max(k,1)*(total-k)/60:.1f} min",
                      flush=True)
                del model
                torch.cuda.empty_cache()
        del D
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_picked = args.out.replace(".csv", "_picked.csv")
    print("\n" + "=" * 78)
    print("VALIDATION-AUROC SELECTION (mean over seeds; test read only after picking)")
    print("=" * 78)
    picked = []
    for P in protos:
        sub = df[df.protocol == P]
        g = sub.groupby(KEY)[["val_auroc", "test_auroc"]].mean().sort_values(
            "val_auroc", ascending=False)
        best = g.index[0]
        row = g.iloc[0]
        picked.append({"protocol": P, "hidden": best[0], "dropout": best[1],
                       "w_reg": best[2], "mse_frac": best[3],
                       "val_auroc": round(float(row.val_auroc), 4),
                       "test_auroc": round(float(row.test_auroc), 4)})
        print(f"\n--- {P} ---")
        print(g.round(4).to_string())
        print(f"  SELECTED by val: h{best[0]} do{best[1]} wr{best[2]} mf{best[3]}"
              f"  -> val {row.val_auroc:.4f}  test {row.test_auroc:.4f}")
    pd.DataFrame(picked).to_csv(out_picked, index=False)

    # Setting-free statistic quoted in the paper: mean test AUROC over configs.
    print("\n" + "=" * 78)
    print("SETTING-FREE MEAN over the twelve configurations (test AUROC)")
    print("=" * 78)
    for P in protos:
        sub = df[df.protocol == P]
        cfg = sub.groupby(KEY)["test_auroc"].mean()
        print(f"  {P}: config-mean = {cfg.mean():.4f} (sd over configs "
              f"{cfg.std(ddof=1):.4f}), runs-mean = {sub['test_auroc'].mean():.4f}, "
              f"n_configs={len(cfg)}")
    print("\nwrote", args.out, "and", out_picked)


if __name__ == "__main__":
    main()
