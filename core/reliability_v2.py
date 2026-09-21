import argparse
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
import reliability as r1  # noqa: E402


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=os.path.normpath(os.path.join(HERE, "data", "processed")))
    ap.add_argument("--dose-qc", default=os.path.join(HERE, "reliability", "dose_qc.csv"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "reliability"))
    args = ap.parse_args()

    resp = pd.read_csv(os.path.join(args.processed, "response_long.csv"),
                       dtype={"cell_orig": str, "drug": str, "ach": str, "dataset": str})
    curve = r1.load_curve(args.processed)
    st = r1.pair_stats(curve)
    df = resp.merge(st[["dataset", "cell_orig", "drug", "n_repeat", "aac_std",
                        "hs_med", "einf_med", "ec50_med", "q_fit", "q_repeat"]],
                    on=["dataset", "cell_orig", "drug"], how="left")
    q_fit_v1 = df["q_fit"].copy()
    q_rep_v1 = df["q_repeat"].copy()

    dq = pd.read_csv(args.dose_qc)
    dq["ck"] = dq["cell_line"].map(norm)
    dq["dk"] = dq["drug"].map(norm)
    dq = dq.drop_duplicates(["dataset", "ck", "dk"])
    dose_map = {(r.dataset, r.ck, r.dk): (r.q_dose, r.q_dose_rep, r.n_rep)
                for r in dq.itertuples()}

    keys = list(zip(df["dataset"], df["cell_orig"].map(norm), df["drug"].map(norm)))
    q_dose = np.array([dose_map.get(k, (np.nan, np.nan, 0))[0] for k in keys], float)
    q_dose_rep = np.array([dose_map.get(k, (np.nan, np.nan, 0))[1] for k in keys], float)
    matched = ~np.isnan(q_dose)

    df["q_fit"] = np.where(matched, q_dose, q_fit_v1)
    df["q_repeat"] = np.where(matched & ~np.isnan(q_dose_rep), q_dose_rep, q_rep_v1)
    df["dose_qc_matched"] = matched

    cmap = r1.smiles_map(args.processed)
    agree = r1.agree_quality(resp, cmap)
    agree_map = dict(agree.items())
    df["cs"] = [cmap.get((d, str(dr))) for d, dr in zip(df["dataset"], df["drug"])]
    df["q_agree"] = [agree_map.get((a, c)) for a, c in zip(df["ach"], df["cs"])]

    df["reliability_w"] = r1.combine(df["q_fit"], df["q_repeat"], df["q_agree"])
    df["reliability_tier"] = r1.tier_of(df["reliability_w"])

    cols = ["dataset", "cell_orig", "drug", "ach", "aac", "ic50", "n_repeat", "aac_std",
            "q_fit", "q_repeat", "q_agree", "dose_qc_matched", "reliability_w", "reliability_tier"]
    out = os.path.join(args.outdir, "response_reliability_v2.csv")
    df[cols].to_csv(out, index=False)

    summary = {
        "dose_qc_match_rate": round(float(matched.mean()), 4),
        "match_rate_by_dataset": {ds: round(float(m.mean()), 4) for ds, m in
                                  df.groupby("dataset")["dose_qc_matched"]},
        "w_v2_describe": {k: round(float(v), 4) for k, v in df["reliability_w"].describe().items()},
        "tier_counts_v2": df["reliability_tier"].value_counts().to_dict(),
        "q_fit_mean_v1_vs_v2": [round(float(q_fit_v1.mean()), 4), round(float(df["q_fit"].mean()), 4)],
        "q_repeat_coverage_v1_vs_v2": [round(float(q_rep_v1.notna().mean()), 4),
                                       round(float(df["q_repeat"].notna().mean()), 4)],
        "tier_agreement_with_v1": None,
    }
    v1 = pd.read_csv(os.path.join(args.outdir, "response_reliability.csv"),
                     usecols=["reliability_tier"]).rename(columns={"reliability_tier": "t1"})
    summary["tier_agreement_with_v1"] = round(float((v1["t1"] == df["reliability_tier"]).mean()), 4)
    with open(os.path.join(args.outdir, "reliability_v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
