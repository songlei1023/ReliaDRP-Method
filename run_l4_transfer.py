import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_zoo_2021_2026 as zoo  # noqa: E402
from run_benchmark import train_one, infer, q_conformal, picp_mpiw, ALPHA  # noqa: E402

TIERS = ["high", "mid", "low"]
MODALITIES = ["expr", "cnv", "mut"]
SEED = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extdir", default=os.path.normpath(os.path.join(HERE, "data", "processed_ext")))
    ap.add_argument("--models", default="evi2,reliadrp,mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--dirs", default="")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "l4_transfer.csv"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = args.extdir
    elig = pd.read_csv(os.path.join(d, "response_ext_eligible.csv"))
    ci = np.load(os.path.join(d, "pair_cell_row.npy"))
    di = np.load(os.path.join(d, "pair_drug_row.npy"))
    y = elig["y_auc"].to_numpy(np.float32)
    r2 = elig["r2_mean"].to_numpy(float)
    wall = np.clip(np.nan_to_num(r2, nan=0.0), 0.0, 1.0).astype(np.float32)
    cells_raw = np.load(os.path.join(d, "cells_multi_raw_ext.npy"))
    drugs_raw = np.load(os.path.join(d, "drug_raw_ext.npy"))
    import json
    with open(os.path.join(d, "ext_features_summary.json"), encoding="utf-8") as f:
        chan = json.load(f)["channel_genes"]
    sizes = [chan[m] for m in MODALITIES]
    offs = np.cumsum([0] + sizes)
    ds = elig["dataset"].to_numpy()
    dsets = set(ds)
    CL = [s for s in ["CCLE", "CTRPv1", "GDSC1"] if s in dsets]
    rng = np.random.RandomState(SEED)
    q1, q2 = np.nanquantile(r2, 1 / 3), np.nanquantile(r2, 2 / 3)
    tier = np.where(r2 >= q2, "high", np.where(r2 <= q1, "low", "mid"))

    def transform_cells(tr_rows):
        out = []
        for k, m in enumerate(MODALITIES):
            sl = cells_raw[:, offs[k]:offs[k + 1]]
            p = PCA(min(128, len(tr_rows) - 1, sl.shape[1]), random_state=0).fit(sl[tr_rows])
            z = p.transform(sl).astype(np.float32)
            if z.shape[1] < 128:
                z = np.hstack([z, np.zeros((len(z), 128 - z.shape[1]), np.float32)])
            out.append(z)
        return np.hstack(out)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    directions = [(CL, [t]) for t in ["BeatAML2", "PDX_Bruna"] if t in dsets]
    directions += [([t], CL) for t in ["BeatAML2", "PDX_Bruna"] if t in dsets]
    for train_ds, test_ds in directions:
        tag = f"{'+'.join(train_ds)}->{'+'.join(test_ds)}"
        if args.dirs and not any(s.strip() in tag for s in args.dirs.split(",")):
            continue
        tr_all = np.where(np.isin(ds, train_ds))[0]
        te = np.where(np.isin(ds, test_ds))[0]
        perm = rng.permutation(len(tr_all))
        nv = int(0.1 * len(tr_all))
        va = tr_all[perm[:nv]]
        tr = tr_all[perm[nv:]]
        cells = transform_cells(np.unique(ci[tr]))
        dtr = np.unique(di[tr])
        dpca = PCA(min(128, len(dtr), drugs_raw.shape[1]), random_state=0).fit(drugs_raw[dtr])
        drugs = dpca.transform(drugs_raw).astype(np.float32)
        if drugs.shape[1] < 128:
            drugs = np.hstack([drugs, np.zeros((len(drugs), 128 - drugs.shape[1]), np.float32)])
        tag = f"{'+'.join(train_ds)}->{'+'.join(test_ds)}"
        tc = torch.from_numpy(cells).to(dev)
        td = torch.from_numpy(drugs).to(dev)
        ci_t = torch.from_numpy(ci.astype(np.int64)).to(dev)
        di_t = torch.from_numpy(di.astype(np.int64)).to(dev)
        _seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]
        for name, seed in [(n, s) for n in [m.strip() for m in args.models.split(",")] for s in _seeds]:
            t0 = time.time()
            model = zoo.build(name).to(dev)
            ww = wall if name == "reliadrp" else None
            model = train_one(model, tr, va, ci, di, y, cells, drugs, ww, None, seed,
                              args.epochs, args.batch)
            model.eval()
            mu_v, sig_v, _, _ = infer(model, ci_t, di_t, tc, td, va)
            mu_t, sig_t, _, _ = infer(model, ci_t, di_t, tc, td, te)
            stud = bool(np.std(sig_t) > 1e-8)
            s_cal = np.abs(y[va] - mu_v) / (sig_v + 1e-8) if stud else np.abs(y[va] - mu_v)
            q = q_conformal(s_cal)
            pp, mm = picp_mpiw(y[te], mu_t, sig_t, q, stud)
            row = {"direction": tag, "model": name, "seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)),
                   "rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
                   "pearson": round(float(stats.pearsonr(y[te], mu_t)[0]), 4),
                   "picp": round(pp, 4), "mpiw": round(mm, 4)}
            for t in TIERS:
                m = tier[te] == t
                if m.sum() < 10:
                    continue
                p_cp, _ = picp_mpiw(y[te][m], mu_t[m], sig_t[m], q, stud)
                qt = q_conformal(s_cal[tier[va] == t]) if (tier[va] == t).sum() > 5 else q
                p_gc, _ = picp_mpiw(y[te][m], mu_t[m], sig_t[m], qt, stud)
                row[f"picp_cp_{t}"] = round(p_cp, 4)
                row[f"picp_gc_{t}"] = round(p_gc, 4)
            rows.append(row)
            print(f"[{tag}] {name}: rmse={row['rmse']} r={row['pearson']} picp={pp:.3f} "
                  f"cp_high={row.get('picp_cp_high')} gc_high={row.get('picp_gc_high')} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
