import argparse
import json
import os
import re

import numpy as np
import pandas as pd

RAW = r"D:\BI\DRP\HDRP\data\_drevalpy"
SOURCES = ["CCLE", "CTRPv1", "GDSC1", "BeatAML2", "PDX_Bruna"]
COLS = ["sample", "cellosaurus_id", "cell_line_name", "pubchem_id", "drug_name",
        "tissue", "AUC", "AUC_curvecurator", "R2", "RMSE", "RelevanceScore",
        "SignalQuality", "Slope", "Regulation", "LN_IC50", "LN_IC50_curvecurator"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def build_source(ds):
    p = os.path.join(RAW, ds, f"{ds}.csv")
    d = pd.read_csv(p, usecols=lambda c: c in COLS, low_memory=False)
    sm = os.path.join(RAW, ds, "drug_smiles.csv")
    smi = pd.read_csv(sm, low_memory=False)
    smi["pc"] = smi["pubchem_id"].astype(str)
    smap = dict(zip(smi["pc"], smi["canonical_smiles"]))
    d["dataset"] = ds
    d["cell_id"] = d["sample"].astype(str)
    d["drug_id"] = d["pubchem_id"].astype(str)
    d["smiles"] = d["drug_id"].map(smap)
    for c in ["AUC", "AUC_curvecurator", "R2", "RMSE", "RelevanceScore", "SignalQuality", "Slope"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby(["dataset", "cell_id", "drug_id"], sort=False).agg(
        cell_line_name=("cell_line_name", "first"),
        drug_name=("drug_name", "first"),
        smiles=("smiles", "first"),
        tissue=("tissue", "first") if "tissue" in d.columns else ("cell_id", "first"),
        y_auc=("AUC", "median"),
        y_auc_curve=("AUC_curvecurator", "median") if "AUC_curvecurator" in d.columns else ("AUC", "median"),
        r2_mean=("R2", "mean") if "R2" in d.columns else ("AUC", "mean"),
        rmse_mean=("RMSE", "mean") if "RMSE" in d.columns else ("AUC", "mean"),
        relevance_mean=("RelevanceScore", "mean") if "RelevanceScore" in d.columns else ("AUC", "mean"),
        n_rep=("AUC", "size"),
    ).reset_index()
    return g


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.normpath(os.path.join(here, "data", "processed_ext")))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    parts = [build_source(ds) for ds in SOURCES]
    ext = pd.concat(parts, ignore_index=True)
    ext["y_auc"] = pd.to_numeric(ext["y_auc"], errors="coerce")
    ext = ext[ext["y_auc"].notna()]
    out = os.path.join(args.outdir, "response_ext.csv")
    ext.to_csv(out, index=False)

    by_ds = {}
    for ds, s in ext.groupby("dataset"):
        by_ds[ds] = {"pairs": int(len(s)), "cells": int(s.cell_id.nunique()),
                     "drugs": int(s.drug_id.nunique()),
                     "y_mean": round(float(s.y_auc.mean()), 4),
                     "smiles_cov": round(float(s.smiles.notna().mean()), 4),
                     "r2_mean": round(float(s.r2_mean.mean()), 4)}
    shared_drugs = ext.groupby("drug_id")["dataset"].nunique()
    shared_cells = ext.groupby("cell_id")["dataset"].nunique()
    summ = {
        "n_pairs": int(len(ext)),
        "n_cells": int(ext.cell_id.nunique()),
        "n_drugs": int(ext.drug_id.nunique()),
        "by_dataset": by_ds,
        "drugs_in_ge2_datasets": int((shared_drugs >= 2).sum()),
        "cells_in_ge2_datasets": int((shared_cells >= 2).sum()),
        "target_y": "AUC (median per cell-drug), source-harmonized",
    }
    with open(os.path.join(args.outdir, "ext_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
