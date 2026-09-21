import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

RAW = r"D:\BI\DRP\HDRP\data\_scdrugmap\data_collection_drug_response"
SEED = 42


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--panel", default=os.path.normpath(os.path.join(here, "data", "processed", "gene_names.txt")))
    ap.add_argument("--outdir", default=os.path.normpath(os.path.join(here, "data", "processed_ext", "L3")))
    ap.add_argument("--cap", type=int, default=800)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    panel = [g.strip() for g in open(args.panel, encoding="utf-8") if g.strip()]
    panel_set = set(panel)
    rng = np.random.RandomState(SEED)

    def bkey(s):
        return str(s).replace(".", "-")

    data = []
    ds_ids = []
    labels = []
    used_genes = None
    per_ds = {}
    for annf in sorted(glob.glob(os.path.join(args.raw, "*.annotation.csv"))):
        name = os.path.basename(annf).replace(".annotation.csv", "")
        ann = pd.read_csv(annf, index_col=0)
        ann = ann[ann["condition"].isin(["sensitive", "resistant"])]
        if len(ann) == 0:
            continue
        take = rng.choice(len(ann), min(args.cap, len(ann)), replace=False)
        ann = ann.iloc[np.sort(take)]
        ann_map = {}
        for b in ann.index:
            ann_map.setdefault(bkey(b), b)
        rc = os.path.join(args.raw, f"{name}.rawcount.csv")
        cols = list(pd.read_csv(rc, nrows=0).columns)
        pos, newnames = [0], []
        for i, c in enumerate(cols):
            k = bkey(c)
            if i > 0 and k in ann_map:
                pos.append(i)
                newnames.append(ann_map[k])
        if len(pos) < 50:
            print(f"{name}: SKIP (barcode match {len(pos)-1})", flush=True)
            continue
        X = pd.read_csv(rc, usecols=pos, index_col=0, low_memory=False)
        X.columns = newnames
        X.index = X.index.astype(str)
        X = X[~X.index.duplicated(keep="first")]
        genes = [g for g in panel if g in X.index]
        X = X.loc[genes]
        if used_genes is None:
            used_genes = set(genes)
        else:
            used_genes &= set(genes)
        data.append((name, X, genes))
        per_ds[name] = {"cells": int(X.shape[1]), "genes_on_panel": len(genes)}
        print(f"{name}: cells={X.shape[1]} panel_genes={len(genes)}", flush=True)

    used_genes = [g for g in panel if g in used_genes]
    print("common panel genes:", len(used_genes), flush=True)
    mats, labs, dsid = [], [], []
    for name, X, genes in data:
        X = X.loc[used_genes]
        arr = X.to_numpy(np.float32).T
        lib = arr.sum(1, keepdims=True)
        lib[lib == 0] = 1.0
        arr = np.log1p(arr / lib * 1e4)
        mats.append(arr)
        ann = pd.read_csv(os.path.join(args.raw, f"{name}.annotation.csv"), index_col=0).loc[list(X.columns)]
        labs.append((ann["condition"].to_numpy() == "resistant").astype(np.int64))
        dsid.append(np.array([name] * len(arr)))
        del X
    cells = np.vstack(mats).astype(np.float32)
    y = np.concatenate(labs)
    ds = np.concatenate(dsid)
    np.save(os.path.join(args.outdir, "cells_l3.npy"), cells)
    np.save(os.path.join(args.outdir, "y_l3.npy"), y)
    pd.DataFrame({"dataset": ds, "y": y}).to_csv(os.path.join(args.outdir, "meta_l3.csv"), index=False)
    with open(os.path.join(args.outdir, "genes_l3.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(used_genes))

    rng = np.random.RandomState(SEED)
    n = len(cells)
    perm = rng.permutation(n)
    n_te, n_va = int(0.1 * n), int(0.1 * n)
    os.makedirs(os.path.join(args.outdir, "splits"), exist_ok=True)
    np.save(os.path.join(args.outdir, "splits", "L3_random_train.npy"), np.sort(perm[n_te + n_va:]))
    np.save(os.path.join(args.outdir, "splits", "L3_random_val.npy"), np.sort(perm[:n_va]))
    np.save(os.path.join(args.outdir, "splits", "L3_random_test.npy"), np.sort(perm[n_va:n_te + n_va]))
    hold = set(rng.permutation(np.unique(ds))[:max(1, int(0.2 * len(np.unique(ds))))])
    is_te = np.isin(ds, list(hold))
    rest = np.where(~is_te)[0]
    te = np.where(is_te)[0]
    nv = int(0.1 * len(rest))
    np.save(os.path.join(args.outdir, "splits", "L3_ldo_train.npy"), np.sort(rest[nv:]))
    np.save(os.path.join(args.outdir, "splits", "L3_ldo_val.npy"), np.sort(rest[:nv]))
    np.save(os.path.join(args.outdir, "splits", "L3_ldo_test.npy"), np.sort(te))

    summ = {"n_cells": int(n), "n_genes": len(used_genes), "n_datasets": int(len(np.unique(ds))),
            "label_mean": round(float(y.mean()), 4), "per_dataset": per_ds,
            "held_out_datasets": sorted(hold)}
    with open(os.path.join(args.outdir, "l3_dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_dataset"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
