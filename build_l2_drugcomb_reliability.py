import argparse
import json
import os

import numpy as np
import pandas as pd

RAW = r"D:\BI\DRP\HDRP\data\_drugcomb\summary_v_1_5.csv"
SYN = ["synergy_zip", "synergy_loewe", "synergy_hsa", "synergy_bliss"]
COLS = ["drug_row", "drug_col", "cell_line_name", "study_name", "tissue_name",
        "css_ri", "S_mean"] + SYN


def robust_z(s, clip=5.0):
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    iqr = iqr if iqr > 1e-9 else s.std() + 1e-9
    return ((s - med) / iqr).clip(-clip, clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--outdir", default=os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed_ext")))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = pd.read_csv(args.raw, usecols=COLS, low_memory=False)
    for c in ["css_ri", "S_mean"] + SYN:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    print("loaded", d.shape, flush=True)

    z = pd.concat([robust_z(d[c]) for c in SYN], axis=1)
    d["q_model_agree"] = (1.0 - (z.std(axis=1) / 2.0).clip(0, 1)).clip(0, 1)

    key = ["drug_row", "drug_col", "cell_line_name"]
    g = d.groupby(key, sort=False)
    agg = g.agg(
        y_css_ri=("css_ri", "median"),
        y_smean=("S_mean", "median"),
        q_model_agree=("q_model_agree", "mean"),
        n_rows=("S_mean", "size"),
        n_studies=("study_name", "nunique"),
        tissue=("tissue_name", "first"),
    ).reset_index()

    multi = d[d.duplicated(key, keep=False)]
    spread = multi.groupby(key)["S_mean"].agg(
        lambda s: np.nan if len(s) < 2 else float(s.quantile(0.75) - s.quantile(0.25)))
    study_mean = multi.groupby(key + ["study_name"])["S_mean"].median().reset_index()
    study_spread = study_mean.groupby(key)["S_mean"].agg(
        lambda s: np.nan if len(s) < 2 else float(np.nanmax(s) - np.nanmin(s)))
    agg["q_study_agree"] = np.asarray(agg.set_index(key).index.map(
        (1.0 - (study_spread / 40.0).clip(0, 1))))
    agg["q_repeat_agree"] = np.asarray(agg.set_index(key).index.map(
        (1.0 - (spread / 40.0).clip(0, 1))))

    comps = agg[["q_model_agree", "q_study_agree", "q_repeat_agree"]].to_numpy(float)
    wts = np.array([0.5, 0.25, 0.25])
    mask = ~np.isnan(comps)
    wsum = (mask * wts).sum(1)
    agg["reliability_w"] = np.where(wsum > 0, np.nansum(np.where(mask, comps, 0) * wts, 1) / np.where(wsum > 0, wsum, 1), np.nan)
    valid = agg["reliability_w"].dropna()
    t = pd.Series(pd.qcut(valid.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str))
    agg["reliability_tier"] = "unknown"
    agg.loc[valid.index, "reliability_tier"] = t.to_numpy()

    out = os.path.join(args.outdir, "L2_drugcomb_reliability.csv")
    agg.to_csv(out, index=False)

    summary = {
        "n_rows": int(len(d)),
        "n_combinations": int(len(agg)),
        "n_cells": int(d.cell_line_name.nunique()),
        "n_drugs": int(pd.concat([d.drug_row, d.drug_col]).nunique()),
        "n_studies": int(d.study_name.nunique()),
        "combos_in_ge2_studies": int((agg.n_studies >= 2).sum()),
        "q_model_agree_mean": round(float(agg.q_model_agree.mean()), 4),
        "q_study_agree_coverage": round(float(agg.q_study_agree.notna().mean()), 4),
        "reliability_w_mean": round(float(agg.reliability_w.mean()), 4),
        "tier_counts": agg.reliability_tier.value_counts().to_dict(),
        "y_css_ri_mean": round(float(agg.y_css_ri.mean()), 4),
    }
    with open(os.path.join(args.outdir, "L2_drugcomb_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
