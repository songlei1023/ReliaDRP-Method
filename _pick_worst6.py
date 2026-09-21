"""Rank the sixteen common models on each of the four layers, then rank models.

The pool is always the fifteen external baselines plus the ReliaDRP row, so
every layer score is comparable.  A model's layer score is its mean leaderboard
position over that layer's protocols/directions (fractional rank on ties); its
overall score is the mean of its four layer scores.  Ours is always kept.
"""
import pandas as pd, numpy as np

BASE = ["clclsa","mlp","delfos","attn","pancancer","mgataf","mmcl","bandrp",
        "codeae","crdnn","fourierdrug","vaen","mograph","tgsa","deepdtf"]
OURS = "ReliaDRP"


def layer_ranks(df, gcol, vcol, groups, ours):
    """ours: {group -> value}.  -> {model: mean rank} over the listed groups."""
    out = {}
    for g in groups:
        s = df[df[gcol] == g].groupby("model")[vcol].mean().dropna()
        s = pd.concat([s, pd.Series({OURS: ours[g]})])
        s = s[s.index.isin(BASE + [OURS])]
        assert len(s) >= 15, (g, len(s))   # L2_LDO: deepdtf is undefined
        for m, rk in s.rank(ascending=False, method="average").items():
            out.setdefault(m, []).append(rk)
    return {m: float(np.mean(v)) for m, v in out.items()}


l1 = pd.read_csv("l1_unified_seeded.csv").rename(columns={"proto": "g"})
o1 = l1[l1.model == "reliadrpv3mix2mf03"].groupby("g")["test_r"].mean()
L1 = layer_ranks(l1[l1.model != "reliadrpv3mix2mf03"], "g", "test_r",
                 ["EXT_random", "EXT_LDO", "EXT_LCO"], o1.to_dict())

l2 = pd.read_csv("l2_unified.csv").rename(columns={"protocol": "g"})
l2 = l2[~l2.model.isin(["reliadrp", "reliadrpcombo"])]
l2t = pd.read_csv("l2_v3_tune_confirm.csv")
o2 = l2t[l2t.config == "mix1_noln_h256"].groupby("protocol")["test_r"].mean()
L2 = layer_ranks(l2, "g", "pearson", ["L2_random", "L2_LDO", "L2_LCO"], o2.to_dict())

l3 = pd.read_csv("l3_unified_seeded.csv").rename(columns={"protocol": "g"})
o3 = l3[l3.model == "reliadrp"].groupby("g")["auroc"].mean()
L3 = layer_ranks(l3[l3.model != "reliadrp"], "g", "auroc", ["L3_random", "L3_ldo"], o3.to_dict())

l4 = pd.read_csv("l4_transfer_unified_seeded.csv").rename(columns={"direction": "g"})
o4 = l4[l4.model == "reliadrpv3mix2md0.1"].groupby("g")["pearson"].mean()
l4 = l4[~l4.model.str.startswith("reliadrp")]
L4 = layer_ranks(l4, "g", "pearson", sorted(l4.g.unique()), o4.to_dict())

rows = []
for m in [OURS] + BASE:
    v = [L1[m], L2[m], L3[m], L4[m]]
    rows.append((m, v, float(np.mean(v))))
rows.sort(key=lambda t: t[2])

print("%-12s %6s %6s %6s %6s | %6s" % ("model", "L1", "L2", "L3", "L4", "mean"))
for m, v, mu in rows:
    print("%-12s %6.2f %6.2f %6.2f %6.2f | %6.2f" % (m, *v, mu))

worst = [m for m, _, _ in rows[-6:]][::-1]
print("\nworst six:", worst)
print("keep    :", [m for m, _, _ in rows[:10]])
