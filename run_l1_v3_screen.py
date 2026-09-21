"""Round-3 fast screen for L1 (EXT protocols, seed 0). Selection = val RMSE."""

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
from run_benchmark import train_one, infer  # noqa: E402
from reliadrp_v3 import ReliaDRPV3  # noqa: E402


def build_v3(name):
    kw = {"n_mix": 1}
    if "md0.1" in name:
        kw["mod_drop"] = 0.1
    if "md0.2" in name:
        kw["mod_drop"] = 0.2
    if "mix2" in name:
        kw["n_mix"] = 2
    return ReliaDRPV3(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extdir", default=os.path.normpath(os.path.join(HERE, "data", "processed_ext")))
    ap.add_argument("--protos", default="EXT_random,EXT_LCO")
    ap.add_argument("--variants", default="v2,v3mix1,v3mix2,v3mix1-md0.1")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(HERE, "_l1_screen.csv"))
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = args.extdir
    spl = os.path.join(d, "splits")
    elig = pd.read_csv(os.path.join(d, "response_ext_eligible.csv"))
    ci = np.load(os.path.join(d, "pair_cell_row.npy"))
    di = np.load(os.path.join(d, "pair_drug_row.npy"))
    y = elig["y_auc"].to_numpy(np.float32)
    variants = [v.strip() for v in args.variants.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for P in [p.strip() for p in args.protos.split(",")]:
        tr = np.load(os.path.join(spl, f"{P}_train.npy"))
        va = np.load(os.path.join(spl, f"{P}_val.npy"))
        te = np.load(os.path.join(spl, f"{P}_test.npy"))
        cells = np.load(os.path.join(spl, f"{P}_cells.npy"))
        drugs = np.load(os.path.join(spl, f"{P}_drugs.npy"))
        tc = torch.from_numpy(cells.astype(np.float32)).to(dev)
        td = torch.from_numpy(drugs.astype(np.float32)).to(dev)
        ci_t = torch.from_numpy(ci.astype(np.int64)).to(dev)
        di_t = torch.from_numpy(di.astype(np.int64)).to(dev)
        for vname in variants:
            for seed in seeds:
                t0 = time.time()
                if vname == "v2":
                    m = zoo.build("reliadrp").to(dev)
                else:
                    m = build_v3(vname).to(dev)
                m = train_one(m, tr, va, ci, di, y, cells, drugs, None, None,
                              seed, args.epochs, args.batch)
                m.eval()
                mu_v, _, _, _ = infer(m, ci_t, di_t, tc, td, va)
                mu_t, _, _, _ = infer(m, ci_t, di_t, tc, td, te)
                row = {"proto": P, "variant": vname, "seed": seed,
                       "val_r": round(float(stats.pearsonr(y[va], mu_v)[0]), 4),
                       "test_r": round(float(stats.pearsonr(y[te], mu_t)[0]), 4),
                       "test_rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
                       "sec": round(time.time() - t0, 1)}
                rows.append(row)
                print(f"[{P}] {vname:16s} s{seed}: val_r={row['val_r']:.4f} "
                      f"test_r={row['test_r']:.4f} ({row['sec']}s)", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
