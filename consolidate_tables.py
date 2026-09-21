import os

import pandas as pd

H = r"D:\BI\DRP\HDRP"


def md(df, sort_col):
    df = df.sort_values(sort_col, ascending=False)
    rows = ["| model | RMSE | Pearson | sd | n |", "|---|---|---|---|---|"]
    for m, r in df.iterrows():
        rows.append(f"| {'**'+m+'**' if m.startswith('reliadrp') else m} | {r.rmse:.4f} | "
                    f"{r.pearson:.4f} | {r.pearson_sd:.4f} | {int(r.n)} |")
    return "\n".join(rows)


out = ["# Main tables: external baselines + ReliaDRP (EviDRP excluded)", "",
       "> All numbers are mean over seeds; sd = seed std of Pearson. ReliaDRP rows in bold.", ""]

l1 = pd.read_csv(os.path.join(H, "extended_benchmark_all.csv"))
l1 = l1[~l1.model.isin(["evi", "evi2"])]
g1 = l1.groupby(["protocol", "model"]).agg(rmse=("rmse", "mean"), pearson=("pearson", "mean"),
                                            pearson_sd=("pearson", "std"), n=("seed", "count")).round(4)
out += ["## Table 1. L1 (harmonised 5-source monotherapy, cell + single drug)", ""]
for P in ["EXT_random", "EXT_LDO", "EXT_LCO"]:
    out += [f"### {P}", md(g1.loc[P], "pearson"), ""]

b = pd.read_csv(os.path.join(H, "l2_baselines.csv"))
if "error" in b.columns:
    b = b[b.error.isna()]
b = b[~b.model.isin(["evi", "evi2"])]
c = pd.read_csv(os.path.join(H, "l2_reliadrpcombo.csv"))
l2 = pd.concat([b[["protocol", "model", "rmse", "pearson", "seed"]],
                c[["protocol", "model", "rmse", "pearson", "seed"]]], ignore_index=True)
g2 = l2.groupby(["protocol", "model"]).agg(rmse=("rmse", "mean"), pearson=("pearson", "mean"),
                                            pearson_sd=("pearson", "std"), n=("seed", "count")).round(4)
out += ["## Table 2. L2 (drug combinations)", ""]
for P in ["L2_random", "L2_LCO", "L2_LDO"]:
    out += [f"### {P}", md(g2.loc[P], "pearson"), ""]

l3 = pd.read_csv(os.path.join(H, "l3_baselines.csv"))
if "error" in l3.columns:
    l3 = l3[l3.error.isna()]
g3 = l3.groupby(["protocol", "model"]).agg(rmse=("rmse", "mean"), pearson=("auroc", "mean"),
                                            pearson_sd=("auroc", "std"), n=("seed", "count")).round(4)
out += ["## Table 3. L3 (single cell; metric column = AUROC)", ""]
for P in ["L3_random", "L3_ldo"]:
    out += [f"### {P}", md(g3.loc[P], "pearson"), ""]

l4 = pd.read_csv(os.path.join(H, "l4_transfer.csv"))
if "error" in l4.columns:
    l4 = l4[l4.error.isna()]
g4 = l4.groupby(["direction", "model"]).agg(rmse=("rmse", "mean"), pearson=("pearson", "mean"),
                                             pearson_sd=("pearson", "std"), n=("seed", "count")).round(4)
out += ["## Table 4. L4 (cell line <-> PDX / patient; partial, missing PDX->CL)", ""]
for D in g4.index.get_level_values(0).unique():
    out += [f"### {D}", md(g4.loc[D], "pearson"), ""]

p = os.path.join(H, "main_tables.md")
open(p, "w", encoding="utf-8").write("\n".join(out))
print("wrote", p, len(out), "lines")
