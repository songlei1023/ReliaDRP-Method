import argparse
import json
import os

import numpy as np
import pandas as pd

CURVE_FILES = {"CTRPv2": "ctrpv2_response.csv", "GDSCv2": "gdscv2_response.csv"}
DRUG_FEAT = {"CTRPv2": "drugs_ctrpv2_feat.csv", "GDSCv2": "drugs_gdscv2_feat.csv"}

HS_ZERO_EPS = 1e-6
EINF_ZERO_EPS = 1e-9
EC50_CENSOR = 1e6
STD_SCALE = 0.2
W_FIT, W_REPEAT, W_AGREE = 0.4, 0.3, 0.3


def load_curve(processed):
    frames = []
    for ds, fname in CURVE_FILES.items():
        d = pd.read_csv(os.path.join(processed, fname), dtype=str)
        d = d.rename(columns={"cellid": "cell_orig", "drugid": "drug"})
        for c in ["aac_recomputed", "ic50_recomputed", "HS", "E_inf", "EC50"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["dataset"] = ds
        frames.append(d[["dataset", "cell_orig", "drug", "aac_recomputed",
                         "ic50_recomputed", "HS", "E_inf", "EC50"]])
    return pd.concat(frames, ignore_index=True)


def pair_stats(curve):
    g = curve.groupby(["dataset", "cell_orig", "drug"], sort=False)
    st = g.agg(
        n_repeat=("aac_recomputed", "size"),
        aac_med=("aac_recomputed", "median"),
        aac_std=("aac_recomputed", "std"),
        hs_med=("HS", "median"),
        einf_med=("E_inf", "median"),
        ec50_med=("EC50", "median"),
    ).reset_index()
    flags = g.agg(
        frac_hs_zero=("HS", lambda s: float((s.abs() < HS_ZERO_EPS).mean())),
        frac_einf_zero=("E_inf", lambda s: float((s.abs() < EINF_ZERO_EPS).mean())),
        frac_ec50_censored=("EC50", lambda s: float((s >= EC50_CENSOR).mean())),
        frac_ic50_missing=("ic50_recomputed", lambda s: float(s.isna().mean())),
    ).reset_index()
    st = st.merge(flags, on=["dataset", "cell_orig", "drug"])
    pen = (0.45 * st["frac_hs_zero"]
           + 0.25 * st["frac_ec50_censored"]
           + 0.20 * st["frac_ic50_missing"]
           + 0.10 * st["frac_einf_zero"])
    st["q_fit"] = (1.0 - pen).clip(0.0, 1.0)
    q_rep = (1.0 - st["aac_std"].fillna(0.0) / STD_SCALE).clip(0.0, 1.0)
    st["q_repeat"] = q_rep.where(st["n_repeat"] >= 2, np.nan)
    return st


def smiles_map(processed):
    cmap = {}
    for ds, fname in DRUG_FEAT.items():
        df = pd.read_csv(os.path.join(processed, fname), dtype=str)
        for k, v in zip(df["drug"], df["canonical_smiles"]):
            cmap[(ds, str(k))] = v
    return cmap


def agree_quality(resp, cmap):
    rl = resp[["dataset", "cell_orig", "drug", "ach", "aac"]].copy()
    rl["cs"] = [cmap.get((d, str(dr))) for d, dr in zip(rl["dataset"], rl["drug"])]
    rl = rl.dropna(subset=["ach", "cs"])
    piv = rl.groupby(["ach", "cs", "dataset"])["aac"].median().unstack("dataset")
    if not set(CURVE_FILES).issubset(piv.columns):
        return pd.Series(dtype=float)
    diff = (piv["CTRPv2"] - piv["GDSCv2"]).abs()
    agree = (1.0 - diff / STD_SCALE).clip(0.0, 1.0)
    return agree.dropna()


def combine(fit, rep, agr):
    arr = np.vstack([fit.to_numpy(float), rep.to_numpy(float), agr.to_numpy(float)])
    wts = np.array([[W_FIT], [W_REPEAT], [W_AGREE]], float)
    mask = ~np.isnan(arr)
    wsum = (mask * wts).sum(0)
    val = np.nansum(np.where(mask, arr, 0.0) * wts, 0)
    return np.where(wsum > 0, val / np.where(wsum > 0, wsum, 1.0), np.nan)


def tier_of(w):
    t = pd.Series(pd.qcut(w.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str),
                  index=w.index)
    t[w.isna()] = "unknown"
    return t


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=os.path.normpath(os.path.join(here, "data", "processed")))
    ap.add_argument("--outdir", default=os.path.join(here, "reliability"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    resp = pd.read_csv(os.path.join(args.processed, "response_long.csv"),
                       dtype={"cell_orig": str, "drug": str, "ach": str, "dataset": str})
    curve = load_curve(args.processed)
    st = pair_stats(curve)

    df = resp.merge(st[["dataset", "cell_orig", "drug", "n_repeat", "aac_std",
                        "hs_med", "einf_med", "ec50_med", "q_fit", "q_repeat"]],
                    on=["dataset", "cell_orig", "drug"], how="left")

    cmap = smiles_map(args.processed)
    agree = agree_quality(resp, cmap)
    agree_map = dict(agree.items())
    df["cs"] = [cmap.get((d, str(dr))) for d, dr in zip(df["dataset"], df["drug"])]
    df["q_agree"] = [agree_map.get((a, c)) for a, c in zip(df["ach"], df["cs"])]

    df["w"] = combine(df["q_fit"], df["q_repeat"], df["q_agree"])
    df["reliability_tier"] = tier_of(df["w"])

    cols = ["dataset", "cell_orig", "drug", "ach", "aac", "ic50", "n_repeat", "aac_std",
            "hs_med", "einf_med", "ec50_med", "q_fit", "q_repeat", "q_agree",
            "reliability_w", "reliability_tier"]
    df = df.rename(columns={"w": "reliability_w"})
    out_csv = os.path.join(args.outdir, "response_reliability.csv")
    df[cols].to_csv(out_csv, index=False)

    summary = {
        "processed": args.processed,
        "n_rows": int(len(df)),
        "n_pairs": int(df.drop_duplicates(["dataset", "cell_orig", "drug"]).shape[0]),
        "curve_join_missing": int(df["q_fit"].isna().sum()),
        "q_agree_coverage": round(float(df["q_agree"].notna().mean()), 4),
        "q_repeat_coverage": round(float(df["q_repeat"].notna().mean()), 4),
        "std_scale": STD_SCALE,
        "weights": {"q_fit": W_FIT, "q_repeat": W_REPEAT, "q_agree": W_AGREE},
        "tier_counts": df["reliability_tier"].value_counts().to_dict(),
        "tier_by_dataset": {ds: sub["reliability_tier"].value_counts().to_dict()
                            for ds, sub in df.groupby("dataset")},
        "w_describe": {k: round(float(v), 6) for k, v in df["reliability_w"].describe().items()},
        "component_means": {c: round(float(df[c].mean()), 6) for c in ["q_fit", "q_repeat", "q_agree"]},
    }
    with open(os.path.join(args.outdir, "reliability_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
