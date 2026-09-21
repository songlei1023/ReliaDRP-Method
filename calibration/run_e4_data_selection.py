import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
sys.path.insert(0, os.path.join(HERE, "core"))  # 库模块在 core/
sys.path.insert(0, os.path.join(HERE, "benchmark"))  # 复用 run_benchmark 的 train_one/infer
sys.path.insert(0, HERE)

import model_zoo_2021_2026 as zoo  # noqa: E402
from run_benchmark import (load_multi, load_reliability, train_one, infer,
                           q_conformal, picp_mpiw, ALPHA, TIERS)  # noqa: E402

SUBSETS = ["full", "high", "mid", "low"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="P0")
    ap.add_argument("--models", default="reliadrp,evi2,mlp")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rel", default=os.path.join(HERE, "reliability", "response_reliability.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "e4_data_selection.csv"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    P = args.protocol
    tr, va, te, ci, di, y, cells, drugs = load_multi(P)
    tier, w = load_reliability(args.rel)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tc = torch.from_numpy(cells.astype(np.float32)).to(dev)
    td = torch.from_numpy(drugs.astype(np.float32)).to(dev)
    ci_t = torch.from_numpy(ci.astype(np.int64)).to(dev)
    di_t = torch.from_numpy(di.astype(np.int64)).to(dev)
    tier_te, tier_va = tier[te], tier[va]

    rows = []
    for subset in SUBSETS:
        if subset == "full":
            tr_s = tr
        else:
            tr_s = tr[tier[tr] == subset]
        for name in models:
            for seed in seeds:
                model = zoo.build(name).to(dev)
                model = train_one(model, tr_s, va, ci, di, y, cells, drugs, w, None,
                                  seed, args.epochs, args.batch)
                model.eval()
                mu_v, sig_v, _, _ = infer(model, ci_t, di_t, tc, td, va)
                mu_t, sig_t, _, _ = infer(model, ci_t, di_t, tc, td, te)
                stud = bool(np.std(sig_t) > 1e-8)
                s_cal = np.abs(y[va] - mu_v) / (sig_v + 1e-8) if stud else np.abs(y[va] - mu_v)
                q = q_conformal(s_cal)
                row = {"protocol": P, "subset": subset, "model": name, "seed": seed,
                       "n_train": int(len(tr_s)),
                       "rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
                       "pearson": round(float(stats.pearsonr(y[te], mu_t)[0]), 4)}
                for t in TIERS:
                    m = tier_te == t
                    if m.sum() == 0:
                        continue
                    row[f"rmse_{t}"] = round(float(np.sqrt(np.mean((y[te][m] - mu_t[m]) ** 2))), 4)
                    row[f"pearson_{t}"] = round(float(stats.pearsonr(y[te][m], mu_t[m])[0]), 4)
                    p_cp, _ = picp_mpiw(y[te][m], mu_t[m], sig_t[m], q, stud)
                    row[f"picp_cp_{t}"] = round(p_cp, 4)
                    mv = tier_va == t
                    if mv.sum() > 5:
                        qt = q_conformal(s_cal[mv])
                        p_gc, _ = picp_mpiw(y[te][m], mu_t[m], sig_t[m], qt, stud)
                        row[f"picp_gc_{t}"] = round(p_gc, 4)
                rows.append(row)
                print(f"[{P}] train={subset:4s} {name} s{seed}: rmse={row['rmse']} "
                      f"r={row['pearson']} rmse_high={row.get('rmse_high')} "
                      f"cp_high={row.get('picp_cp_high')} gc_high={row.get('picp_gc_high')}", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
