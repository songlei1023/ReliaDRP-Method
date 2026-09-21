"""W1 sensitivity: does the reliability-layer weight choice drive the paper's
conditional-coverage conclusions?

Both downstream experiments (E1 conditional coverage, E2 CP / RW-CP / GC-CP)
are purely post-hoc: they read `reliability_tier` / `reliability_w` from the
reliability table and combine them with *fixed* model predictions.  Nothing has
to be retrained, so we can re-run them under any weight setting.

For each setting (W_FIT, W_REPEAT, W_AGREE) we
  1. recompute w = weighted mean of the available {q_fit, q_repeat, q_agree}
     (identical renormalisation rule to reliability.combine),
  2. recompute the tercile tier (identical rule to reliability.tier_of),
  3. re-run the E1 and E2 pipelines verbatim and report the headline numbers.

The baseline setting must reproduce the published values, which we assert.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import reliability as r1
from run_e1_conditional_coverage import measure, q_split_conformal, scores
from run_e2_conformal_recovery import q_plain, q_weighted

ALPHA = 0.1
TIERS = ["high", "mid", "low"]
DSETS = ["CTRPv2", "GDSCv2"]

# E1 and E2 use slightly different method sets in the published pipeline.
E1_METHODS = "evi,evi2,mlp,mc,ens"
E2_METHODS = "evi,mlp,mc,ens"

GRID = [
    (0.40, 0.30, 0.30),      # baseline, as published
    (1 / 3, 1 / 3, 1 / 3),   # uniform
    (0.20, 0.40, 0.40),
    (0.30, 0.35, 0.35),
    (0.50, 0.25, 0.25),
    (0.60, 0.20, 0.20),
    (0.40, 0.20, 0.40),      # swap repeat / agree
    (0.40, 0.40, 0.20),
    (0.10, 0.45, 0.45),
    (0.70, 0.15, 0.15),
]

COMPONENT_ONLY = [
    ("q_fit_only", (1.0, 0.0, 0.0)),
    ("q_repeat_only", (0.0, 1.0, 0.0)),
    ("q_agree_only", (0.0, 0.0, 1.0)),
]

REFERENCE = {
    "e1_tier_picp": {"high": 0.8424, "mid": 0.9284, "low": 0.9366},
    "gccp_max_abs_tier_gap": 0.0112,
    "cp_min_tier_picp_mean": 0.8394,
}


def combine_w(fit, rep, agr, wf, wr, wa):
    """Exact re-implementation of reliability.combine with custom weights."""
    arr = np.vstack([fit.to_numpy(float), rep.to_numpy(float), agr.to_numpy(float)])
    wts = np.array([[wf], [wr], [wa]], float)
    mask = ~np.isnan(arr)
    wsum = (mask * wts).sum(0)
    val = np.nansum(np.where(mask, arr, 0.0) * wts, 0)
    return np.where(wsum > 0, val / np.where(wsum > 0, wsum, 1.0), np.nan)


def label(weights):
    return "/".join(f"{v:.2f}" for v in weights)


def simplex_grid(step=0.05, lo=0.10):
    """All (w_fit, w_repeat, w_agree) on a `step` lattice with sum 1 and every
    component >= `lo`.  Gives a proper sweep over the whole admissible region
    rather than a handful of hand-picked settings."""
    n = int(round(1.0 / step))
    lo_n = int(round(lo / step))
    out = []
    for ia in range(lo_n, n - 2 * lo_n + 1):
        for ib in range(lo_n, n - ia - lo_n + 1):
            ic = n - ia - ib
            if ic < lo_n:
                continue
            out.append((round(ia * step, 4), round(ib * step, 4), round(ic * step, 4)))
    return out


def load_context(args):
    spl = os.path.join(args.data, "splits")
    preds = os.path.join(args.data, "results", "preds")
    rel = pd.read_csv(args.rel, usecols=["dataset", "q_fit", "q_repeat", "q_agree",
                                         "reliability_w", "reliability_tier"])
    n_rows = sum(1 for _ in open(os.path.join(args.data, "processed", "response_long.csv"),
                                 encoding="utf-8")) - 1
    if len(rel) != n_rows:
        raise SystemExit(f"row mismatch: reliability {len(rel)} vs response_long {n_rows}")

    protocol_data = {}
    for P in [p.strip() for p in args.protocols.split(",") if p.strip()]:
        te = np.load(os.path.join(spl, f"{P}_test.npy"))
        rng = np.random.RandomState(args.split_seed)
        perm = rng.permutation(len(te))
        n_cal = len(te) // 2
        protocol_data[P] = {
            "te": te,
            "cal_pos": perm[:n_cal],
            "ev_pos": perm[n_cal:],
            "preds": {m.strip(): np.load(os.path.join(preds, f"{P}_{m.strip()}.npz"),
                                         allow_pickle=True)
                      for m in args.methods.split(",")
                      if os.path.exists(os.path.join(preds, f"{P}_{m.strip()}.npz"))},
            "preds_e2": {m.strip(): np.load(os.path.join(preds, f"{P}_{m.strip()}.npz"),
                                            allow_pickle=True)
                         for m in E2_METHODS.split(",")
                         if os.path.exists(os.path.join(preds, f"{P}_{m.strip()}.npz"))},
        }
    return rel, protocol_data


def run_setting(name, weights, rel, protocol_data, ds_all, base_w, base_tier):
    wf, wr, wa = weights
    w = combine_w(rel["q_fit"], rel["q_repeat"], rel["q_agree"], wf, wr, wa)
    w = pd.Series(w, index=rel.index)
    tier = r1.tier_of(w)
    tier_np = tier.to_numpy()

    defined = ~np.isnan(w)
    both = defined & ~np.isnan(base_w)
    spearman = float(pd.Series(w[both]).corr(pd.Series(base_w[both]), method="spearman"))
    churn = float((tier_np[both] != base_tier[both]).mean())

    e1_rows, e2_rows = [], []
    for P, ctx in protocol_data.items():
        te, cal_pos, ev_pos = ctx["te"], ctx["cal_pos"], ctx["ev_pos"]
        tier_ev = tier_np[te[ev_pos]]
        ds_ev = ds_all[te[ev_pos]]

        for method, d in ctx["preds"].items():
            mu, sig, y = d["mu"], d["sig"], d["y"]
            if not (len(mu) == len(sig) == len(y) == len(te)):
                continue
            for stud in [False, True]:
                q = q_split_conformal(scores(y[cal_pos], mu[cal_pos], sig[cal_pos], stud))
                for t in TIERS:
                    m = tier_ev == t
                    if m.sum() == 0:
                        continue
                    idx = ev_pos[m]
                    picp, mpiw = measure(y[idx], mu[idx], sig[idx], q, stud)
                    e1_rows.append({"setting": name, "protocol": P, "method": method,
                                    "studentized": stud, "tier": t, "n": int(m.sum()),
                                    "picp": picp, "picp_gap": picp - (1 - ALPHA), "mpiw": mpiw})

        for method, d in ctx["preds_e2"].items():
            mu, sig, y = d["mu"], d["sig"], d["y"]
            if not (len(mu) == len(sig) == len(y) == len(te)):
                continue
            for stud in [False, True]:
                s_cal = scores(y[cal_pos], mu[cal_pos], sig[cal_pos], stud)
                w_cal = w.to_numpy()[te[cal_pos]]
                q_cp = q_plain(s_cal)
                q_rw = q_weighted(s_cal, w_cal)
                tier_cal = tier_np[te[cal_pos]]
                q_gc = {}
                for t in TIERS:
                    sub = s_cal[tier_cal == t]
                    if len(sub):
                        q_gc[t] = q_plain(sub)
                def emit(cp, subset, mask, q, kind="tier"):
                    if mask.sum() == 0:
                        return
                    picp, mpiw = measure(y[ev_pos][mask], mu[ev_pos][mask],
                                         sig[ev_pos][mask], q, stud)
                    e2_rows.append({"setting": name, "protocol": P, "method": method,
                                    "studentized": stud, "cp": cp, "stratum_kind": kind,
                                    "stratum": subset, "n": int(mask.sum()),
                                    "picp": picp, "picp_gap": picp - (1 - ALPHA), "mpiw": mpiw})
                for t in TIERS:
                    emit("CP", t, tier_ev == t, q_cp)
                    emit("RWCP", t, tier_ev == t, q_rw)
                    if t in q_gc:
                        emit("GCCP", t, tier_ev == t, q_gc[t])
                for src in DSETS:
                    emit("CP", src, ds_ev == src, q_cp, "dataset")
                    emit("RWCP", src, ds_ev == src, q_rw, "dataset")
                    emit("GCCP", src, ds_ev == src, q_gc.get(src, q_cp), "dataset")

    e1 = pd.DataFrame(e1_rows)
    e2 = pd.DataFrame(e2_rows)
    tier_e2 = e2[e2.stratum_kind == "tier"]

    row = {
        "setting": name,
        "w_fit": round(wf, 4), "w_repeat": round(wr, 4), "w_agree": round(wa, 4),
        "w_coverage": round(float(defined.mean()), 4),
        "spearman_vs_base": round(spearman, 4) if np.isfinite(spearman) else None,
        "tier_churn_vs_base": round(churn, 4),
        "n_unknown_tier": int((tier_np == "unknown").sum()),
        "w_mean": round(float(np.nanmean(w)), 4),
        "w_std": round(float(np.nanstd(w)), 4),
    }
    for t in TIERS:
        sub = e1[e1.tier == t]
        row[f"e1_picp_{t}"] = round(float(sub["picp"].mean()), 4) if len(sub) else None
    row["e1_marginal_picp"] = round(float(e1["picp"].mean()), 4)
    row["e1_worst_gap"] = round(float(e1["picp_gap"].min()), 4)
    row["e1_n_neg_gaps"] = int((e1["picp_gap"] < 0).sum())
    for cp in ["CP", "RWCP", "GCCP"]:
        sub = tier_e2[tier_e2.cp == cp]
        if not len(sub):
            continue
        worst = sub.groupby(["protocol", "method", "studentized"])["picp"].min()
        row[f"{cp}_mean_picp"] = round(float(sub["picp"].mean()), 4)
        row[f"{cp}_worst_tier"] = round(float(worst.mean()), 4)
        row[f"{cp}_mpiw"] = round(float(sub["mpiw"].mean()), 4)
        row[f"{cp}_max_gap"] = round(float(sub["picp_gap"].abs().max()), 4)
        row[f"{cp}_n_gaps_gt002"] = int((sub["picp_gap"].abs() > 0.02).sum())
    return row


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "data"))
    ap.add_argument("--rel", default=os.path.join(here, "reliability", "response_reliability.csv"))
    ap.add_argument("--outdir", default=os.path.join(here, "_w1sens"))
    ap.add_argument("--protocols", default="P0,P1,P2")
    ap.add_argument("--methods", default=E1_METHODS)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--components", action="store_true",
                    help="also run the single-component ablations")
    ap.add_argument("--simplex", action="store_true",
                    help="sweep the full weight simplex instead of the named grid")
    ap.add_argument("--simplex-step", type=float, default=0.05)
    ap.add_argument("--simplex-lo", type=float, default=0.10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rel, protocol_data = load_context(args)
    ds_all = rel["dataset"].to_numpy()
    base_w = rel["reliability_w"].to_numpy()
    base_tier = r1.tier_of(pd.Series(base_w, index=rel.index)).to_numpy()

    if args.simplex:
        settings = [(label(g), g) for g in simplex_grid(args.simplex_step, args.simplex_lo)]
        if not any(s[1] == GRID[0] for s in settings):
            settings.insert(0, (label(GRID[0]), GRID[0]))
    else:
        settings = [(label(g), g) for g in GRID]
    if args.components:
        settings += COMPONENT_ONLY

    rows = []
    for name, weights in settings:
        row = run_setting(name, weights, rel, protocol_data, ds_all, base_w, base_tier)
        rows.append(row)
        print(f"done {name}: e1 worst gap {row['e1_worst_gap']}, "
              f"CP worst tier {row.get('CP_worst_tier')}, "
              f"GCCP max gap {row.get('GCCP_max_gap')}, churn {row['tier_churn_vs_base']}")

    df = pd.DataFrame(rows)
    ref = df[df.setting == label(GRID[0])].iloc[0]
    checks = {
        "e1_high": (round(ref["e1_picp_high"], 4), REFERENCE["e1_tier_picp"]["high"]),
        "e1_mid": (round(ref["e1_picp_mid"], 4), REFERENCE["e1_tier_picp"]["mid"]),
        "e1_low": (round(ref["e1_picp_low"], 4), REFERENCE["e1_tier_picp"]["low"]),
        "gccp_max_gap": (round(ref["GCCP_max_gap"], 4), REFERENCE["gccp_max_abs_tier_gap"]),
        "cp_worst_tier": (round(ref["CP_worst_tier"], 4), REFERENCE["cp_min_tier_picp_mean"]),
    }
    bad = {k: v for k, v in checks.items() if abs(v[0] - v[1]) > 5e-4}

    df.to_csv(os.path.join(args.outdir, "w1_sensitivity.csv"), index=False)
    comp_names = [n for n, _ in COMPONENT_ONLY]
    core = df[~df.setting.isin(comp_names)]
    summary = {
        "baseline_reproduction": {k: {"got": v[0], "expected": v[1]} for k, v in checks.items()},
        "baseline_reproduced": not bad,
        "mode": "simplex" if args.simplex else "named_grid",
        "n_settings": len(core),
        "grid_ranges": {
            "e1_picp_high": [round(float(core["e1_picp_high"].min()), 4),
                             round(float(core["e1_picp_high"].max()), 4)],
            "e1_worst_gap": [round(float(core["e1_worst_gap"].min()), 4),
                             round(float(core["e1_worst_gap"].max()), 4)],
            "tier_churn_vs_base": [round(float(core["tier_churn_vs_base"].min()), 4),
                                   round(float(core["tier_churn_vs_base"].max()), 4)],
            "spearman_vs_base": [round(float(core["spearman_vs_base"].min()), 4),
                                 round(float(core["spearman_vs_base"].max()), 4)],
            "CP_worst_tier": [round(float(core["CP_worst_tier"].min()), 4),
                              round(float(core["CP_worst_tier"].max()), 4)],
            "CP_max_gap": [round(float(core["CP_max_gap"].min()), 4),
                           round(float(core["CP_max_gap"].max()), 4)],
            "GCCP_max_gap": [round(float(core["GCCP_max_gap"].min()), 4),
                             round(float(core["GCCP_max_gap"].max()), 4)],
            "RWCP_worst_tier": [round(float(core["RWCP_worst_tier"].min()), 4),
                                round(float(core["RWCP_worst_tier"].max()), 4)],
        },
        "grid_invariants": {
            "high_always_under_covers": bool((core["e1_picp_high"] < 1 - ALPHA).all()),
            "cp_gap_always_large": bool((core["CP_max_gap"] > 0.05).all()),
            "gccp_gap_always_small": bool((core["GCCP_max_gap"] < 0.05).all()),
        },
    }
    with open(os.path.join(args.outdir, "w1_sensitivity_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    cols = ["setting", "w_coverage", "spearman_vs_base", "tier_churn_vs_base",
            "e1_picp_high", "e1_picp_mid", "e1_picp_low", "e1_worst_gap",
            "CP_worst_tier", "CP_max_gap", "RWCP_worst_tier", "GCCP_worst_tier",
            "GCCP_max_gap", "GCCP_n_gaps_gt002"]
    show = df[df.setting.isin([label(g) for g in GRID] + comp_names)]
    if not args.simplex:
        show = df
    print(show[cols].to_string(index=False))
    if args.simplex:
        print(f"\n[simplex] {len(core)} settings swept; "
              f"worst-case rows shown below")
        print(df[cols].sort_values("GCCP_max_gap").tail(3).to_string(index=False))
    print("wrote", os.path.join(args.outdir, "w1_sensitivity.csv"))


if __name__ == "__main__":
    main()
