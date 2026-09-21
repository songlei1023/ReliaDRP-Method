"""Compute the seeded L3 numbers the paper quotes, old vs new, with ranks.

Reads
-----
* `l3_unified_seeded.csv`        -- the seeded 16-model panel (this re-run)
* `_bak_l3_unified_panel_e60b1024.csv` -- the historical unseeded panel
* `l3_sweep_seeded_b1024.csv`    -- the seeded twelve-configuration sweep

Prints
------
1. per-protocol board (mean AUROC +/- seed sd), rank of our row, spread across
   the sixteen models, and the best baseline, for old and new;
2. per-model delta old -> new, so the "the baselines move, not us" claim can be
   checked;
3. the twelve-configuration setting-free average (config-mean and its sd),
   which the paper's L3 paragraph cites;
4. validation-test AUROC correlation over configurations, the evidence that
   validation cannot select the L3 hyper-parameters.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OURS = "reliadrp"
PROTOS = ["L3_random", "L3_ldo"]
KEY = ["hidden", "dropout", "w_reg", "mse_frac"]


def load(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def board(df, cov="auroc"):
    g = df.dropna(subset=[cov]).groupby(["protocol", "model"])[cov].agg(
        ["mean", "std", "count"])
    out = {}
    for P in PROTOS:
        if P not in g.index.get_level_values(0):
            continue
        s = g.loc[P].sort_values("mean", ascending=False)
        out[P] = s
    return out


def show(tag, b):
    print(f"\n=== {tag} ===")
    for P, s in b.items():
        best = s.index[0]
        print(f"  {P}: n={int(s['count'].iloc[0])} seeds, "
              f"spread {s['mean'].min():.4f}-{s['mean'].max():.4f} "
              f"({s['mean'].max()-s['mean'].min():.4f})")
        if OURS not in s.index:
            print(f"    ours: <not yet in file>  best {best} "
                  f"{s['mean'].iloc[0]:.4f}, models so far {len(s)}")
            continue
        rank = list(s.index).index(OURS) + 1
        print(f"    ours {s.loc[OURS,'mean']:.4f} +/- {s.loc[OURS,'std']:.4f} "
              f"-> rank {rank}/{len(s)}")
        print(f"    best {best} {s['mean'].iloc[0]:.4f}; "
              f"gap {s.loc[OURS,'mean']-s['mean'].iloc[0]:+.4f}")
        if rank > 1:
            nb_ = s.index[rank - 2]
            print(f"    next above: {nb_} {s.loc[nb_,'mean']:.4f}")


def main():
    new = load("l3_unified_seeded.csv")
    old = load("_bak_l3_unified_panel_e60b1024.csv")
    sweep = load("l3_sweep_seeded_b1024.csv")
    sweep512 = load("l3_sweep_seeded_b512.csv")
    oldsweep = load(os.path.join("results_and_paper", "results", "L3_single_cell",
                                 "l3_v2_tuning_e60b1024.csv"))
    oldsweep512 = load(os.path.join("results_and_paper", "results", "L3_single_cell",
                                    "l3_v2_tuning_e60b512.csv"))

    nb, ob = None, None
    if new is not None:
        nb = board(new)
        show("seeded panel (this re-run, 5 seeds)", nb)
    if old is not None:
        ob = board(old)
        show("historical panel (unseeded, 3 seeds)", ob)

    if nb and ob:
        print("\n=== per-model delta old -> new (AUROC) ===")
        for P in PROTOS:
            if P not in nb or P not in ob:
                continue
            a = ob[P]["mean"]
            b = nb[P]["mean"]
            j = pd.concat([a.rename("old"), b.rename("new")], axis=1)
            j["delta"] = j["new"] - j["old"]
            j = j.sort_values("delta")
            print(f"  --- {P} (ours: {j.loc[OURS,'old']:.4f} -> "
                  f"{j.loc[OURS,'new']:.4f}) ---")
            for m, r in j.iterrows():
                mark = " <== ours" if m == OURS else ""
                print(f"    {m:12s} {r['old']:.4f} -> {r['new']:.4f}  "
                      f"{r['delta']:+.4f}{mark}")

    print("\n=== twelve-configuration sweep (setting-free statistic) ===")
    for tag, d, ref in (("seeded b1024", sweep, nb),
                        ("historical b1024", oldsweep, ob),
                        ("seeded b512", sweep512, nb),
                        ("historical b512", oldsweep512, ob)):
        if d is None:
            print(f"  {tag}: <missing>")
            continue
        print(f"  --- {tag} (batch {int(d['batch'].iloc[0])}, "
              f"epochs {int(d['epochs'].iloc[0])}) ---")
        for P in PROTOS:
            sub = d[d.protocol == P]
            if sub.empty:
                continue
            cfg = sub.groupby(KEY)["test_auroc"].mean()
            vcfg = sub.groupby(KEY)["val_auroc"].mean()
            r = np.corrcoef(vcfg.to_numpy(), cfg.to_numpy())[0, 1]
            # place the config-mean on the *matching* sixteen-model board
            rank = None
            if ref and P in ref:
                vals = list(ref[P]["mean"].to_numpy()) + [cfg.mean()]
                rank = sorted(vals, reverse=True).index(cfg.mean()) + 1
            print(f"    {P}: config-mean {cfg.mean():.4f} "
                  f"(sd over configs {cfg.std(ddof=1):.4f}, n={len(cfg)}), "
                  f"runs-mean {sub['test_auroc'].mean():.4f}, "
                  f"rank {rank}/17 on the {tag} board, val-test r = {r:+.3f}")
            best = vcfg.idxmax()
            print(f"      selected by val: h{best[0]} do{best[1]} wr{best[2]} "
                  f"mf{best[3]} -> val {vcfg.max():.4f} "
                  f"test {cfg.loc[best]:.4f}")


if __name__ == "__main__":
    main()
