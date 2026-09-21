import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
sys.path.insert(0, os.path.join(HERE, "core"))  # 库模块在 core/
sys.path.insert(0, HERE)
import model_zoo_2021_2026 as zoo  # noqa: E402
from run_benchmark import train_one, infer, q_conformal, picp_mpiw, ALPHA  # noqa: E402

TIERS = ["high", "mid", "low"]
EXTR = os.path.normpath(os.path.join(HERE, "data", "processed_ext"))


def prep_l2():
    d = os.path.join(EXTR, "L2")
    resp = pd.read_csv(os.path.join(d, "response_l2.csv"))
    expr = np.load(os.path.join(d, "cells_expr.npy")).astype(np.float32)
    morgan = np.load(os.path.join(d, "drug_morgan.npy")).astype(np.float32)
    ci = resp["cell_idx"].to_numpy(np.int64)
    dri = resp["drow_idx"].to_numpy(np.int64)
    dci = resp["dcol_idx"].to_numpy(np.int64)
    y = resp["y_css_ri"].to_numpy(np.float32)
    tier = resp["reliability_tier"].to_numpy()
    w = resp["reliability_w"].to_numpy(np.float32)
    r2 = np.full(len(resp), np.nan)

    def make(tr, va, te):
        c = PCA(min(384, len(np.unique(ci[tr])) - 1, expr.shape[1]), random_state=0).fit(expr[np.unique(ci[tr])])
        cd = c.transform(expr).astype(np.float32)
        if cd.shape[1] < 384:
            cd = np.hstack([cd, np.zeros((len(cd), 384 - cd.shape[1]), np.float32)])
        dp = PCA(128, random_state=0).fit(morgan)
        dd = dp.transform(morgan).astype(np.float32)
        drugs = ((dd[dri] + dd[dci]) / 2.0).astype(np.float32)
        return cd, drugs, ci, np.arange(len(resp))
    return resp, y, tier, make, "reg", w


def prep_l3():
    d = os.path.join(EXTR, "L3")
    cells = np.load(os.path.join(d, "cells_l3.npy")).astype(np.float32)
    y = np.load(os.path.join(d, "y_l3.npy")).astype(np.float32)
    meta = pd.read_csv(os.path.join(d, "meta_l3.csv"))
    tier = np.array(["mid"] * len(y))

    def make(tr, va, te):
        c = PCA(min(384, len(tr) - 1, cells.shape[1]), random_state=0).fit(cells[tr])
        cd = c.transform(cells).astype(np.float32)
        if cd.shape[1] < 384:
            cd = np.hstack([cd, np.zeros((len(cd), 384 - cd.shape[1]), np.float32)])
        drugs = np.zeros((len(cells), 128), np.float32)
        return cd, drugs, np.arange(len(cells)), np.arange(len(cells))
    return meta, y, tier, make, "clf", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True, choices=["L2", "L3"])
    ap.add_argument("--models", default=",".join(zoo.ALL_NAMES))
    ap.add_argument("--protocols", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.layer == "L2":
        base = os.path.join(EXTR, "L2", "splits")
        protos = [p.strip() for p in (args.protocols or "L2_random,L2_LCO,L2_LDO").split(",")]
        meta, y, tier, make, task, w = prep_l2()
    else:
        base = os.path.join(EXTR, "L3", "splits")
        protos = [p.strip() for p in (args.protocols or "L3_random,L3_ldo").split(",")]
        meta, y, tier, make, task, w = prep_l3()

    out = args.out or os.path.join(HERE, f"{args.layer.lower()}_baselines.csv")
    models = [m.strip() for m in args.models.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for P in protos:
        tr = np.load(os.path.join(base, f"{P}_train.npy"))
        va = np.load(os.path.join(base, f"{P}_val.npy"))
        te = np.load(os.path.join(base, f"{P}_test.npy"))
        cells, drugs, ci, di = make(tr, va, te)
        tc = torch.from_numpy(cells).to(dev)
        td = torch.from_numpy(drugs).to(dev)
        ci_t = torch.from_numpy(ci.astype(np.int64)).to(dev)
        di_t = torch.from_numpy(di.astype(np.int64)).to(dev)
        for name in models:
            for seed in seeds:
                t0 = time.time()
                try:
                    model = zoo.build(name).to(dev)
                    ww = w if (w is not None and name == "reliadrp") else None
                    model = train_one(model, tr, va, ci, di, y, cells, drugs, ww, None, seed,
                                      args.epochs, args.batch)
                    model.eval()
                    mu_v, sig_v, _, _ = infer(model, ci_t, di_t, tc, td, va)
                    mu_t, sig_t, _, _ = infer(model, ci_t, di_t, tc, td, te)
                    if not np.isfinite(mu_t).all():
                        raise ValueError("non-finite")
                    stud = bool(np.std(sig_t) > 1e-8 and np.isfinite(sig_t).all())
                    sc = np.abs(y[va] - mu_v) / (sig_v + 1e-8) if stud else np.abs(y[va] - mu_v)
                    q = q_conformal(sc)
                    pp, mm = picp_mpiw(y[te], mu_t, sig_t, q, stud)
                    row = {"protocol": P, "model": name, "seed": seed,
                           "rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
                           "pearson": round(float(stats.pearsonr(y[te], mu_t)[0]), 4),
                           "picp": round(pp, 4), "mpiw": round(mm, 4)}
                    if task == "clf":
                        row["auroc"] = round(float(roc_auc_score(y[te], mu_t)), 4)
                        row["acc"] = round(float(((mu_t >= 0.5) == (y[te] >= 0.5)).mean()), 4)
                    rows.append(row)
                    print(f"[{P}] {name} s{seed}: rmse={row['rmse']} r={row.get('pearson')} "
                          f"auroc={row.get('auroc')} picp={pp:.3f} ({time.time()-t0:.0f}s)", flush=True)
                except Exception as e:
                    rows.append({"protocol": P, "model": name, "seed": seed, "rmse": np.nan,
                                 "error": f"{type(e).__name__}: {str(e)[:60]}"})
                    print(f"[{P}] {name} s{seed} FAILED {type(e).__name__}", flush=True)
                finally:
                    try:
                        del model
                    except Exception:
                        pass
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
