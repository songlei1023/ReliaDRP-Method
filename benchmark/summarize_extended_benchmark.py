import os
import numpy as np
import pandas as pd

HERE = r"D:\BI\DRP\HDRP"
parts = ["extended_random.csv", "extended_ldo.csv", "extended_lco.csv",
         "extended_deepdtf_cpu.csv"]
frames = []
for f in parts:
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        frames.append(pd.read_csv(p))
df = pd.concat(frames, ignore_index=True)
if "error" in df.columns:
    df = df[df["error"].isna()].drop(columns=["error"])
df.to_csv(os.path.join(HERE, "extended_benchmark_all.csv"), index=False)

g = df.groupby(["protocol", "model"])
agg = g.agg(
    rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
    pearson=("pearson", "mean"), pearson_sd=("pearson", "std"),
    picp=("picp", "mean"), picp_sd=("picp", "std"),
    mpiw=("mpiw", "mean"),
    cp_high=("picp_cp_high", "mean"), gc_high=("picp_gc_high", "mean"),
    n=("seed", "count")).reset_index()
for c in ["rmse", "rmse_sd", "pearson", "pearson_sd", "picp", "picp_sd", "mpiw", "cp_high", "gc_high"]:
    agg[c] = agg[c].round(4)
agg.to_csv(os.path.join(HERE, "extended_benchmark_summary.csv"), index=False)

for P in ["EXT_random", "EXT_LDO", "EXT_LCO"]:
    s = agg[agg.protocol == P].sort_values("pearson", ascending=False)
    print(f"\n=== {P} (sorted by Pearson) ===")
    print(s[["model", "rmse", "rmse_sd", "pearson", "pearson_sd", "picp", "cp_high", "gc_high", "n"]].to_string(index=False))
print("\nrows total:", len(df), "models:", df.model.nunique(), "protocols:", df.protocol.nunique())
