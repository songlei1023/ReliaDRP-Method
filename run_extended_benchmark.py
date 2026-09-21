import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_zoo_2021_2026 as zoo  # noqa: E402
from run_benchmark import train_one, infer, q_conformal, picp_mpiw, ALPHA  # noqa: E402

TIERS = ["high", "mid", "low"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extdir", default=os.path.normpath(os.path.join(HERE, "data", "processed_ext")))
    ap.add_argument("--protocols", default="EXT_random")
    ap.add_argument("--models", default="evi2,mlp,reliadrp")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "extended_benchmark.csv"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = args.extdir
    spl = os.path.join(d, "splits")
    elig = pd.read_csv(os.path.join(d, "response_ext_eligible.csv"))
    ci = np.load(os.path.join(d, "pair_cell_row.npy"))
    di = np.load(os.path.join(d, "pair_drug_row.npy"))
    y = elig["y_auc"].to_numpy(np.float32)
    r2 = elig["r2_mean"].to_numpy(float)
    ok = ~np.isnan(r2)
    q1, q2 = np.nanquantile(r2, 1 / 3), np.nanquantile(r2, 2 / 3)
    tier = np.where(r2 >= q2, "high", np.where(r2 <= q1, "low", "mid"))
    tier[~ok] = "unknown"

    rows = []
    dev = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for P in [p.strip() for p in args.protocols.split(",") if p.strip()]:
        tr = np.load(os.path.join(spl, f"{P}_train.npy"))
        va = np.load(os.path.join(spl, f"{P}_val.npy"))
        te = np.load(os.path.join(spl, f"{P}_test.npy"))
        cells = np.load(os.path.join(spl, f"{P}_cells.npy"))
        drugs = np.load(os.path.join(spl, f"{P}_drugs.npy"))
        ctx = None
        if "codeae" in models:
            from sklearn.cluster import KMeans
            sub = np.random.RandomState(0).choice(np.unique(ci[tr]), min(2000, len(np.unique(ci[tr]))), replace=False)
            km = KMeans(n_clusters=16, random_state=0, n_init=3).fit(cells[sub])
            ctx = km.predict(cells)[ci].astype(np.int64)
        tc = torch.from_numpy(cells.astype(np.float32)).to(dev)
        td = torch.from_numpy(drugs.astype(np.float32)).to(dev)
        ci_t = torch.from_numpy(ci.astype(np.int64)).to(dev)
        di_t = torch.from_numpy(di.astype(np.int64)).to(dev)
        for name in models:
          for seed in seeds:
            t0 = time.time()
            try:
                model = zoo.build(name).to(dev)
                model = train_one(model, tr, va, ci, di, y, cells, drugs, None, ctx, seed,
                                  args.epochs, args.batch)
                model.eval()
                mu_v, sig_v, _, _ = infer(model, ci_t, di_t, tc, td, va)
                mu_t, sig_t, _, _ = infer(model, ci_t, di_t, tc, td, te)
                if not np.isfinite(mu_t).all() or not np.isfinite(mu_v).all():
                    raise ValueError("non-finite predictions")
                stud = bool(np.std(sig_t) > 1e-8 and np.isfinite(sig_t).all())
                s_cal = np.abs(y[va] - mu_v) / (sig_v + 1e-8) if stud else np.abs(y[va] - mu_v)
                q = q_conformal(s_cal)
                picp, mpiw = picp_mpiw(y[te], mu_t, sig_t, q, stud)
                row = {"protocol": P, "model": name, "seed": seed,
                       "rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
                       "pearson": round(float(stats.pearsonr(y[te], mu_t)[0]), 4),
                       "picp": round(picp, 4), "mpiw": round(mpiw, 4)}
                for t in TIERS:
                    m = tier[te] == t
                    if m.sum() < 10:
                        continue
                    p_cp, _ = picp_mpiw(y[te][m], mu_t[m], sig_t[m], q, stud)
                    qt = q_conformal(s_cal[tier[va] == t])
                    p_gc, w_gc = picp_mpiw(y[te][m], mu_t[m], sig_t[m], qt, stud)
                    row[f"picp_cp_{t}"] = round(p_cp, 4)
                    row[f"picp_gc_{t}"] = round(p_gc, 4)
                rows.append(row)
                print(f"[{P}] {name} s{seed}: rmse={row['rmse']} r={row['pearson']} picp={picp:.3f} "
                      f"cp_high={row.get('picp_cp_high')} gc_high={row.get('picp_gc_high')} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                rows.append({"protocol": P, "model": name, "seed": seed,
                             "rmse": np.nan, "pearson": np.nan, "error": f"{type(e).__name__}: {str(e)[:80]}"})
                print(f"[{P}] {name} s{seed} FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
            finally:
                try:
                    del model
                except Exception:
                    pass
                if dev == "cuda":
                    torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
