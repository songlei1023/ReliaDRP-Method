import argparse
import json
import os

import numpy as np
import pandas as pd

ALPHA = 0.1


def q_split_conformal(s_cal, alpha=ALPHA):
    n = len(s_cal)
    return float(np.quantile(s_cal, (1 - alpha) * (1 + 1.0 / n)))


def intervals(mu, sig, q, studentized):
    if studentized:
        return mu - q * sig, mu + q * sig
    return mu - q, mu + q


def scores(y, mu, sig, studentized):
    r = np.abs(y - mu)
    return r / (sig + 1e-8) if studentized else r


def measure(y, mu, sig, q, studentized):
    lo, hi = intervals(mu, sig, q, studentized)
    picp = float(np.mean((y >= lo) & (y <= hi)))
    mpiw = float(np.mean(hi - lo))
    return picp, mpiw


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（脚本已移入一级子目录）
    data = os.path.join(here, "data")
    expcode = os.path.normpath(os.path.join(here, "..", "new_data", "exp"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=data)
    ap.add_argument("--rel", default=os.path.join(here, "reliability", "response_reliability.csv"))
    ap.add_argument("--outdir", default=here)
    ap.add_argument("--protocols", default="P0,P1,P2")
    ap.add_argument("--methods", default="evi,evi2,mlp,mc,ens")
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    proc = os.path.join(args.data, "processed")
    preds = os.path.join(args.data, "results", "preds")
    spl = os.path.join(args.data, "splits")

    n_rows = sum(1 for _ in open(os.path.join(proc, "response_long.csv"), encoding="utf-8")) - 1
    rel = pd.read_csv(args.rel, usecols=["dataset", "reliability_tier"])
    if len(rel) != n_rows:
        raise SystemExit(f"row mismatch: reliability {len(rel)} vs response_long {n_rows}")
    tier = rel["reliability_tier"].to_numpy()
    ds = rel["dataset"].to_numpy()

    rows = []
    for P in [p.strip() for p in args.protocols.split(",") if p.strip()]:
        te = np.load(os.path.join(spl, f"{P}_test.npy"))
        rng = np.random.RandomState(args.split_seed)
        perm = rng.permutation(len(te))
        n_cal = len(te) // 2
        cal_pos, ev_pos = perm[:n_cal], perm[n_cal:]
        tier_ev = tier[te[ev_pos]]
        ds_ev = ds[te[ev_pos]]
        for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
            f = os.path.join(preds, f"{P}_{method}.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f, allow_pickle=True)
            mu, sig, y = d["mu"], d["sig"], d["y"]
            if not (len(mu) == len(sig) == len(y) == len(te)):
                print(f"skip {P}/{method}: len mismatch {len(mu)} vs {len(te)}")
                continue
            for stud in [False, True]:
                q = q_split_conformal(scores(y[cal_pos], mu[cal_pos], sig[cal_pos], stud))
                strata = [("all", "all", np.ones(len(ev_pos), bool))]
                for t in ["high", "mid", "low"]:
                    strata.append(("tier", t, tier_ev == t))
                for src in ["CTRPv2", "GDSCv2"]:
                    strata.append(("dataset", src, ds_ev == src))
                for kind, subset, mask in strata:
                    n = int(mask.sum())
                    if n == 0:
                        continue
                    idx = ev_pos[mask]
                    picp, mpiw = measure(y[idx], mu[idx], sig[idx], q, stud)
                    rows.append({"protocol": P, "method": method, "studentized": stud,
                                 "stratifier": kind, "subset": subset, "n": n,
                                 "picp": round(picp, 4), "mpiw": round(mpiw, 4),
                                 "picp_gap": round(picp - (1 - ALPHA), 4), "q": round(q, 6)})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "e1_conditional_coverage.csv")
    df.to_csv(out_csv, index=False)

    marg = df[(df.stratifier == "all")]
    tier_df = df[(df.stratifier == "tier")]
    summary = {
        "n_rows": int(len(df)),
        "marginal_mean_picp": round(float(marg["picp"].mean()), 4),
        "tier_mean_picp": {t: round(float(tier_df[tier_df.subset == t]["picp"].mean()), 4)
                           for t in ["high", "mid", "low"]},
        "worst_tier_gap": round(float(tier_df["picp_gap"].min()), 4),
        "n_negative_tier_gaps": int((tier_df["picp_gap"] < 0).sum()),
        "by_protocol_marginal": {p: round(float(s["picp"].mean()), 4)
                                 for p, s in marg.groupby("protocol")},
    }
    with open(os.path.join(args.outdir, "e1_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", out_csv)
    print(df[df.stratifier != "dataset"].to_string(index=False))


if __name__ == "__main__":
    main()
