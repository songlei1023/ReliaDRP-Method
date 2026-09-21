import argparse
import json
import os
import re

import numpy as np
import pandas as pd

RAW = r"D:\BI\DRP\HDRP\data"
TABLES = [("CTRPv2", "Table5_CTRPTruncatedDrugResponseProfiles.csv"),
          ("GDSCv1", "Table6_GDSCv1TruncatedDrugResponseProfiles.csv"),
          ("GDSCv2", "Table7_GDSCv2TruncatedDrugResponseProfiles.csv"),
          ("PRISM", "Table8_PRISMTruncatedDrugResponseProfiles.csv")]
COLS = ["cell_line", "drug", "replicate", "anchor_decision", "canonical_drug",
        "canonical_cell", "auc_sigmoid_norm", "R^2", "RMSE", "einf", "hillslope", "ec50"]
RMSE_SCALE = 0.5
AUC_STD_SCALE = 0.2


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(here, "data", "raw"))
    ap.add_argument("--processed", default=os.path.join(here, "data", "processed"))
    ap.add_argument("--outdir", default=os.path.join(here, "reliability"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dose_dir = os.path.join(args.raw, "_dose_profiles")

    frames = []
    for ds, f in TABLES:
        d = pd.read_csv(os.path.join(dose_dir, f), usecols=COLS)
        d["dataset"] = ds
        for c in ["auc_sigmoid_norm", "R^2", "RMSE", "einf", "hillslope", "ec50"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["hillslope_abs"] = d["hillslope"].abs()
        d["anchor_on"] = (d["anchor_decision"].astype(str) == "anchor_on").astype(float)
        frames.append(d)
    all_d = pd.concat(frames, ignore_index=True)

    g = all_d.groupby(["dataset", "cell_line", "drug"], sort=False)
    agg = g.agg(n_rep=("replicate", "size"),
                r2_mean=("R^2", "mean"),
                rmse_mean=("RMSE", "mean"),
                hillslope_abs_mean=("hillslope_abs", "mean"),
                einf_mean=("einf", "mean"),
                ec50_med=("ec50", "median"),
                auc_mean=("auc_sigmoid_norm", "mean"),
                auc_std=("auc_sigmoid_norm", "std"),
                frac_anchor_on=("anchor_on", "mean"),
                canonical_cell=("canonical_cell", "first"),
                canonical_drug=("canonical_drug", "first")).reset_index()
    r2c = agg["r2_mean"].clip(0, 1)
    rmc = (1 - agg["rmse_mean"] / RMSE_SCALE).clip(0, 1)
    agg["q_dose"] = (0.6 * r2c + 0.4 * rmc).fillna(0).clip(0, 1)
    qrep = (1 - agg["auc_std"] / AUC_STD_SCALE).clip(0, 1)
    agg["q_dose_rep"] = qrep.where(agg["n_rep"] >= 2, np.nan)
    out = os.path.join(args.outdir, "dose_qc.csv")
    agg.to_csv(out, index=False)

    # coverage against current core response_long
    rl = pd.read_csv(os.path.join(args.processed, "response_long.csv"),
                     dtype={"cell_orig": str, "drug": str, "dataset": str})
    rl["cell_n"] = rl["cell_orig"].map(norm)
    rl["drug_n"] = rl["drug"].map(norm)
    cov = {}
    for ds in ["CTRPv2", "GDSCv1", "GDSCv2", "PRISM"]:
        sub = agg[agg.dataset == ds]
        core = rl[rl.dataset == ds]
        if not len(core):
            cov[ds] = {"in_core": 0, "dose_pairs": int(len(sub)),
                       "matched": 0, "note": "not in current core"}
            continue
        dose_keys = set(zip(sub["cell_line"].map(norm), sub["drug"].map(norm)))
        core_keys = set(zip(core["cell_n"], core["drug_n"]))
        matched = len(core_keys & dose_keys)
        cov[ds] = {"in_core_unique_pairs": len(core_keys), "dose_pairs": int(len(sub)),
                   "matched": matched,
                   "match_rate": round(matched / max(len(core_keys), 1), 4)}

    summary = {
        "n_dose_rows": int(len(all_d)),
        "n_dose_pairs": int(len(agg)),
        "pairs_by_dataset": agg.groupby("dataset").size().to_dict(),
        "q_dose_describe": {k: round(float(v), 4) for k, v in agg["q_dose"].describe().items()},
        "q_dose_by_dataset": {ds: round(float(s["q_dose"].mean()), 4)
                              for ds, s in agg.groupby("dataset")},
        "rep_coverage": round(float(agg["q_dose_rep"].notna().mean()), 4),
        "core_join_coverage": cov,
    }
    with open(os.path.join(args.outdir, "dose_qc_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
