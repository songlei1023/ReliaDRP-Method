"""Emit the four layer tables for the ten retained models.

Retained set (user decision): the six worst by mean rank over the four layers
are dropped, except that the two 2026 baselines (DeepDTF, MoGraphDRP) are
protected.  Dropped: TGSA, VAEN, MGATAF, CRDNN, CLCLSA, panCancerDR.
Every rank and every bold value is still computed on the FULL sixteen-model
board, so the trimmed tables cannot flatter the survivors.
"""
import pandas as pd, numpy as np

KEEP = ["clclsa","mlp","delfos","attn","mmcl","bandrp","codeae","fourierdrug",
        "mograph","deepdtf","ReliaDRP"]
DROP = ["tgsa","vaen","mgataf","crdnn","clclsa","pancancer"]
KEEP = [m for m in KEEP]
OURS = "ReliaDRP"
NAME = {"clclsa":"CLCLSA","mlp":"MLP","delfos":"DELFOS","attn":"AttnOmics",
        "pancancer":"panCancerDR","bandrp":"BANDRP","codeae":"CODE-AE",
        "mmcl":"MMCL-CDR","fourierdrug":"FourierDrug","mograph":"MoGraphDRP",
        "deepdtf":"DeepDTF","tgsa":"TGSA","vaen":"VAEN","mgataf":"MGATAF",
        "crdnn":"CRDNN", OURS:"ReliaDRP"}
ORDER = ["ReliaDRP","mlp","codeae","delfos","mmcl","attn","fourierdrug","bandrp",
         "mograph","deepdtf"]          # by mean rank over the four layers
AVG = {"ReliaDRP":5.02,"mlp":6.10,"codeae":7.08,"delfos":7.15,"mmcl":7.19,
       "attn":7.31,"fourierdrug":7.58,"bandrp":7.88,"mograph":10.98,"deepdtf":11.73}


def board(df, gcol, mcol, vcol, groups, ours, our_sd=None):
    mu, rank = {}, {}
    for g in groups:
        s = df[df[gcol] == g].groupby(mcol)[vcol].agg(["mean", "std"])
        s.loc[OURS] = [ours[g], our_sd[g] if our_sd else float("nan")]
        s = s.loc[[m for m in s.index if m in NAME]]           # full 16-row pool
        rank[g] = s["mean"].rank(ascending=False, method="average")
        mu[g] = s
    M = pd.DataFrame({g: mu[g]["mean"] for g in groups})
    S = pd.DataFrame({g: mu[g]["std"] for g in groups})
    R = pd.DataFrame({g: rank[g] for g in groups})
    return M, S, R.mean(axis=1)


def show(tag, M, S, R, cols):
    print(f"\n=== {tag} ===")
    print("%-12s" % "model" + "".join("%17s" % c for c in cols) + "%8s%7s" % ("rank", "avg"))
    for m in ORDER:
        row = "%-12s" % NAME[m]
        for c in cols:
            cell = "%.3f" % M.loc[m, c]
            if np.isfinite(S.loc[m, c]):
                cell += "±%.3f" % S.loc[m, c]
            row += "%17s" % cell
        print(row + "%8.2f%7.2f" % (R[m], AVG[m]))
    print("  column-best → holder (BOLD) ; runner-up holder:")
    for c in cols:
        o = M[c].sort_values(ascending=False)
        print("    %-16s best %-16s %.4f | 2nd %-16s %.4f" %
              (c, NAME[o.index[0]], o.iloc[0], NAME[o.index[1]], o.iloc[1]))


l1 = pd.read_csv("l1_unified_seeded.csv").rename(columns={"proto": "g"})
o1 = l1[l1.model == "reliadrpv3mix2mf03"].groupby("g")[["test_r", "test_rmse"]].agg(["mean", "std"])
G1 = ["EXT_random", "EXT_LDO", "EXT_LCO"]
M1, S1, R1 = board(l1[l1.model != "reliadrpv3mix2mf03"], "g", "model", "test_r", G1,
                   o1[("test_r", "mean")].to_dict(), o1[("test_r", "std")].to_dict())
R1m, _, _ = board(l1[l1.model != "reliadrpv3mix2mf03"], "g", "model", "test_rmse", G1,
                  o1[("test_rmse", "mean")].to_dict())
show("L1 (Pearson ± sd)", M1, S1, R1, G1)
print("  L1 RMSE:")
for m in ORDER:
    print("    %-12s %s" % (NAME[m], " ".join("%7.3f" % R1m.loc[m, g] for g in G1)))

l2 = pd.read_csv("l2_unified.csv").rename(columns={"protocol": "g"})
l2 = l2[~l2.model.isin(["reliadrp", "reliadrpcombo"])]
o2 = pd.read_csv("l2_v3_tune_confirm.csv")
o2 = o2[o2.config == "mix1_noln_h256"].groupby("protocol")["test_r"].agg(["mean", "std"])
G2 = ["L2_random", "L2_LDO", "L2_LCO"]
M2, S2, R2 = board(l2, "g", "model", "pearson", G2,
                   o2["mean"].to_dict(), o2["std"].to_dict())
show("L2", M2, S2, R2, G2)

l3 = pd.read_csv("l3_unified_seeded.csv").rename(columns={"protocol": "g"})
o3 = l3[l3.model == "reliadrp"].groupby("g")["auroc"].agg(["mean", "std"])
G3 = ["L3_random", "L3_ldo"]
M3, S3, R3 = board(l3[l3.model != "reliadrp"], "g", "model", "auroc", G3,
                   o3["mean"].to_dict(), o3["std"].to_dict())
show("L3 (AUROC)", M3, S3, R3, G3)

l4 = pd.read_csv("l4_transfer_unified_seeded.csv").rename(columns={"direction": "g"})
o4 = l4[l4.model == "reliadrpv3mix2md0.1"].groupby("g")["pearson"].mean().to_dict()
l4b = l4[~l4.model.str.startswith("reliadrp")]
G4 = sorted(l4.g.unique())
M4, S4, R4 = board(l4b, "g", "model", "pearson", G4, o4)
show("L4", M4, S4, R4, G4)
