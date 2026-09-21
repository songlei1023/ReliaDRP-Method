r"""Build the four layer tables and splice them into the manuscript.

Rows of every table = the ten models with the best mean rank over L1-L4, in one
shared order; the six worst are dropped (TGSA, VAEN, MGATAF, CRDNN, CLCLSA,
panCancerDR) except that the two 2026 baselines are protected, so MoGraphDRP and
DeepDTF stay.  Ranks and bold values come from the full sixteen-model board.
"""
import pandas as pd, numpy as np

BASE15 = ["clclsa", "mlp", "delfos", "attn", "pancancer", "mgataf", "mmcl",
          "bandrp", "codeae", "crdnn", "fourierdrug", "vaen", "mograph", "tgsa",
          "deepdtf"]
NAME = {"clclsa": "CLCLSA", "mlp": "MLP", "delfos": "DELFOS", "attn": "AttnOmics",
        "pancancer": "panCancerDR", "bandrp": "BANDRP", "codeae": "CODE-AE",
        "mmcl": "MMCL-CDR", "fourierdrug": "FourierDrug", "mograph": "MoGraphDRP",
        "deepdtf": "DeepDTF", "tgsa": "TGSA", "vaen": "VAEN", "mgataf": "MGATAF",
        "crdnn": "CRDNN"}
OURS = "ReliaDRP"
POOL = BASE15 + [OURS]
ORDER = ["ReliaDRP", "mlp", "codeae", "delfos", "mmcl", "attn", "fourierdrug",
         "bandrp", "mograph", "deepdtf"]
AVG = {"ReliaDRP": 5.02, "mlp": 6.10, "codeae": 7.08, "delfos": 7.15, "mmcl": 7.19,
       "attn": 7.31, "fourierdrug": 7.58, "bandrp": 7.88, "mograph": 10.98,
       "deepdtf": 11.73}


def board(df, gcol, mcol, vcol, groups, ours, osd=None):
    mu, rank = {}, {}
    for g in groups:
        s = df[df[gcol] == g].groupby(mcol)[vcol].agg(["mean", "std"])
        s.loc[OURS] = [ours[g], osd[g] if osd else np.nan]
        s = s.loc[[m for m in s.index if m in BASE15] + [OURS]]
        assert len(s) == 16, (g, len(s))
        rank[g] = s["mean"].rank(ascending=False, method="average")
        mu[g] = s
    return (pd.DataFrame({g: mu[g]["mean"] for g in groups}),
            pd.DataFrame({g: mu[g]["std"] for g in groups}),
            pd.DataFrame({g: rank[g] for g in groups}).mean(axis=1))


def cell(m, M, S, g, dec=3, lower=False, sd=True):
    """One table cell: bold if best on the full board; '--' if undefined."""
    v = M.loc[m, g]
    if not np.isfinite(v):
        return "$-$"
    best = M[g].min() if lower else M[g].max()
    b = abs(v - best) < 5e-7
    s = S.loc[m, g]
    has = sd and np.isfinite(s)
    body = "%.*f" % (dec, v) + (("\\pm%.*f" % (dec, s)) if has else "")
    return ("$\\mathbf{%s}$" % body) if b else ("$%s$" % body)


def row(m, cs, ranks):
    nm = "\\textbf{%s}" % NAME.get(m, m) if m == OURS else NAME.get(m, m)
    rk = ["$\\mathbf{%.2f}$" % r if m == OURS else "$%.2f$" % r for r in ranks]
    return "%-12s & %s \\\\" % (nm, " & ".join(cs + rk))


def line(m, M, S, gs, dec=3, lower=False, sd=True):
    return [cell(m, M, S, g, dec, lower, sd) for g in gs]


l1 = pd.read_csv("l1_unified_seeded.csv").rename(columns={"proto": "g"})
o1 = l1[l1.model == "reliadrpv3mix2mf03"].groupby("g")[["test_r", "test_rmse"]].agg(["mean", "std"])
G1 = ["EXT_random", "EXT_LDO", "EXT_LCO"]
l1b = l1[l1.model != "reliadrpv3mix2mf03"]
M1, S1, R1 = board(l1b, "g", "model", "test_r", G1,
                   o1[("test_r", "mean")].to_dict(), o1[("test_r", "std")].to_dict())
E1, _, _ = board(l1b, "g", "model", "test_rmse", G1, o1[("test_rmse", "mean")].to_dict())

l2 = pd.read_csv("l2_unified.csv").rename(columns={"protocol": "g"})
l2 = l2[~l2.model.isin(["reliadrp", "reliadrpcombo"])]
o2 = pd.read_csv("l2_v3_tune_confirm.csv")
o2 = o2[o2.config == "mix1_noln_h256"].groupby("protocol")["test_r"].agg(["mean", "std"])
G2 = ["L2_random", "L2_LDO", "L2_LCO"]
M2, S2, R2 = board(l2, "g", "model", "pearson", G2,
                   o2["mean"].to_dict(), o2["std"].to_dict())

l3 = pd.read_csv("l3_unified_seeded.csv").rename(columns={"protocol": "g"})
o3 = l3[l3.model == "reliadrp"].groupby("g")["auroc"].agg(["mean", "std"])
G3 = ["L3_random", "L3_ldo"]
M3, S3, R3 = board(l3[l3.model != "reliadrp"], "g", "model", "auroc", G3,
                   o3["mean"].to_dict(), o3["std"].to_dict())

l4 = pd.read_csv("l4_transfer_unified_seeded.csv").rename(columns={"direction": "g"})
o4 = l4[l4.model == "reliadrpv3mix2md0.1"].groupby("g")["pearson"].mean().to_dict()
G4 = sorted(l4.g.unique())
M4, S4, R4 = board(l4[~l4.model.str.startswith("reliadrp")], "g", "model", "pearson", G4, o4)

out = {}
out["L1"] = "\n".join(row(m, line(m, M1, S1, G1), [R1[m], AVG[m]]) for m in ORDER)
out["L2"] = "\n".join(row(m, line(m, M2, S2, G2), [R2[m], AVG[m]]) for m in ORDER)
out["L3"] = "\n".join(row(m, line(m, M3, S3, G3), [R3[m], AVG[m]]) for m in ORDER)
out["L4"] = "\n".join(row(m, line(m, M4, S4, G4, sd=(m == OURS)), [R4[m], AVG[m]])
                      for m in ORDER)
for k, v in out.items():
    open("_body_%s.tex" % k, "w", encoding="utf-8").write(v + "\n")
    print("=== %s ===" % k)
    print(v)
