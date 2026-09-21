import argparse
import json
import os

import numpy as np
import pandas as pd

ALPHA = 0.1
TIERS = ["high", "mid", "low"]
DSETS = ["CTRPv2", "GDSCv2"]


def scores(y, mu, sig, studentized):
    r = np.abs(y - mu)
    return r / (sig + 1e-8) if studentized else r


def q_plain(s_cal, alpha=ALPHA):
    n = len(s_cal)
    return float(np.quantile(s_cal, (1 - alpha) * (1 + 1.0 / n)))


def q_weighted(s_cal, w_cal, alpha=ALPHA):
    order = np.argsort(s_cal)
    v, w = s_cal[order], w_cal[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return q_plain(s_cal, alpha)
    cw = cw / cw[-1]
    idx = int(np.searchsorted(cw, 1 - alpha))
    return float(v[min(idx, len(v) - 1)])


def measure(y, mu, sig, q, studentized):
    if studentized:
        lo, hi = mu - q * sig, mu + q * sig
    else:
        lo, hi = mu - q, mu + q
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, "data")
    expcode = os.path.normpath(os.path.join(here, "..", "new_data", "exp"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=data)
    ap.add_argument("--rel", default=os.path.join(here, "reliability", "response_reliability.csv"))
    ap.add_argument("--outdir", default=here)
    ap.add_argument("--protocols", default="P0,P1,P2")
    ap.add_argument("--methods", default="evi,mlp,mc,ens")
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    spl = os.path.join(args.data, "splits")
    preds = os.path.join(args.data, "results", "preds")
    rel = pd.read_csv(args.rel, usecols=["dataset", "reliability_tier", "reliability_w"])
    tier = rel["reliability_tier"].to_numpy()
    w_all = rel["reliability_w"].to_numpy()
    ds_all = rel["dataset"].to_numpy()

    rows = []
    for P in [p.strip() for p in args.protocols.split(",") if p.strip()]:
        te = np.load(os.path.join(spl, f"{P}_test.npy"))
        rng = np.random.RandomState(args.split_seed)
        perm = rng.permutation(len(te))
        n_cal = len(te) // 2
        cal_pos, ev_pos = perm[:n_cal], perm[n_cal:]
        for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
            f = os.path.join(preds, f"{P}_{method}.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f, allow_pickle=True)
            mu, sig, y = d["mu"], d["sig"], d["y"]
            if not (len(mu) == len(sig) == len(y) == len(te)):
                continue
            for stud in [False, True]:
                s_cal = scores(y[cal_pos], mu[cal_pos], sig[cal_pos], stud)
                w_cal = w_all[te[cal_pos]]
                q_cp = q_plain(s_cal)
                q_rw = q_weighted(s_cal, w_cal)
                tier_cal = tier[te[cal_pos]]
                q_gc = {t: q_plain(s_cal[tier_cal == t]) for t in TIERS}

                tier_ev = tier[te[ev_pos]]
                ds_ev = ds_all[te[ev_pos]]
                y_ev, mu_ev, sig_ev = y[ev_pos], mu[ev_pos], sig[ev_pos]

                def emit(cp, kind, subset, mask, q):
                    if mask.sum() == 0:
                        return
                    picp, mpiw = measure(y_ev[mask], mu_ev[mask], sig_ev[mask], q, stud)
                    rows.append({"protocol": P, "method": method, "studentized": stud,
                                 "cp": cp, "stratum_kind": kind, "stratum": subset,
                                 "n": int(mask.sum()), "picp": round(picp, 4),
                                 "mpiw": round(mpiw, 4), "picp_gap": round(picp - (1 - ALPHA), 4),
                                 "q": round(float(q), 6)})

                emit("CP", "all", "all", np.ones(len(ev_pos), bool), q_cp)
                for t in TIERS:
                    emit("CP", "tier", t, tier_ev == t, q_cp)
                for t in TIERS:
                    emit("RWCP", "tier", t, tier_ev == t, q_rw)
                for t in TIERS:
                    emit("GCCP", "tier", t, tier_ev == t, q_gc[t])
                for src in DSETS:
                    emit("CP", "dataset", src, ds_ev == src, q_cp)
                    emit("RWCP", "dataset", src, ds_ev == src, q_rw)
                    emit("GCCP", "dataset", src, ds_ev == src, q_gc.get(src, q_cp))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "e2_conformal_recovery.csv"), index=False)

    tier_df = df[df.stratum_kind == "tier"]
    ds_df = df[df.stratum_kind == "dataset"]
    summary = {}
    for cp in ["CP", "RWCP", "GCCP"]:
        sub = tier_df[tier_df.cp == cp]
        if not len(sub):
            continue
        worst = sub.groupby(["protocol", "method", "studentized"])["picp"].min()
        summary[cp] = {
            "mean_tier_picp": round(float(sub["picp"].mean()), 4),
            "min_tier_picp_mean": round(float(worst.mean()), 4),
            "mean_tier_mpiw": round(float(sub["mpiw"].mean()), 4),
            "max_abs_tier_gap": round(float(sub["picp_gap"].abs().max()), 4),
            "n_tier_gaps_gt_002": int((sub["picp_gap"].abs() > 0.02).sum()),
        }
    ds_summary = {cp: round(float(ds_df[ds_df.cp == cp]["picp"].mean()), 4)
                  for cp in ["CP", "RWCP", "GCCP"]}
    out = {"tier_summary": summary, "dataset_mean_picp": ds_summary}
    with open(os.path.join(args.outdir, "e2_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    pivot = tier_df.pivot_table(index=["protocol", "method", "studentized"],
                                columns=["cp", "stratum"], values="picp")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
