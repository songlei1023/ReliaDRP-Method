# -*- coding: utf-8 -*-
"""L4 (cell-line <-> patient/PDX transfer) —— 统一版评估。

与 run_l4_transfer.py 的区别（也是"统一"的全部内容）：

1. **表征基座四个方向共享**。旧脚本在每个方向内部各自拟合 PCA
   (`PCA(...).fit(sl[tr_rows])`)，于是四个方向各自活在不同的特征空间里，无法横向比较；
   反向方向（训练域只有 487/25 个细胞）的基座更是退化的，投影细胞系细胞时输入被放大
   上千倍（实测 |p99| 668/1104，|max| 2610/3814），导致 RMSE 9~82 的伪结果。
   本脚本对每个模态只在**两域细胞并集**上拟合一次基座，再投影所有细胞。
   （只用特征、不用标签，四个方向与全部模型完全等同。）

2. **药物 PCA 用全部药物拟合**，与 L1/L2 的约定一致
   (`make_extended_splits.py` / `run_l2_benchmark.py` 均为 `.fit(drugs_raw)`)；
   旧 L4 脚本用训练域药物拟合，是相对 L1/L2 的一处偏离。

3. **模型 × 方向 × 种子构成完整网格**：16 模型 × 4 方向 × 相同种子，无缺格。
   旧数据是三套不同实验拼起来的（3 模型×5 种子 / 16 模型×3 种子 / 4 模型×3 种子）。

4. **可断点续跑**：结果逐行落盘，重跑时自动跳过已完成的 (direction, model, seed)。

训练/校准路径完全复用 `run_benchmark.py::train_one`，因此按 CLAUDE.md 的规定
**豁免** shuffled-label 守卫（该函数内部用全局行索引 `tr[perm[...]]`）。

用法：
    python run_l4_transfer_unified.py --models <逗号分隔16个> --seeds 0,1,2,3,4
    python run_l4_transfer_unified.py --dirs 0,1        # 只跑前两个方向
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_zoo_2021_2026 as zoo  # noqa: E402
from run_benchmark import train_one, infer, q_conformal, picp_mpiw  # noqa: E402

TIERS = ["high", "mid", "low"]
MODALITIES = ["expr", "cnv", "mut"]
SPLIT_SEED = 7          # 每个方向的 train/val 划分种子（固定，跨模型跨种子一致）
CL_SOURCES = ["CCLE", "CTRPv1", "GDSC1"]
CLINICAL = ["BeatAML2", "PDX_Bruna"]

# 16 个模型：15 个外部基线 + ReliaDRP
ALL_MODELS = ["reliadrp", "clclsa", "mlp", "delfos", "attn", "pancancer", "mgataf",
              "mmcl", "bandrp", "codeae", "crdnn", "fourierdrug", "vaen",
              "mograph", "tgsa", "deepdtf"]


def build_shared_representation(extdir, chan_dim=128, drug_dim=128):
    """四个方向共用的 (cells, drugs) 表征。基座只在特征上拟合，不使用任何标签。"""
    cells_raw = np.load(os.path.join(extdir, "cells_multi_raw_ext.npy"))
    drugs_raw = np.load(os.path.join(extdir, "drug_raw_ext.npy"))
    with open(os.path.join(extdir, "ext_features_summary.json"), encoding="utf-8") as f:
        chan = json.load(f)["channel_genes"]
    sizes = [chan[m] for m in MODALITIES]
    offs = np.cumsum([0] + sizes)

    # 细胞：基座拟合于全部细胞（两个域的并集），一次，四个方向共用
    ref_cells = np.arange(cells_raw.shape[0])
    blocks, exp_var = [], {}
    for k, m in enumerate(MODALITIES):
        sl = cells_raw[:, offs[k]:offs[k + 1]]
        p = PCA(n_components=min(chan_dim, len(ref_cells) - 1, sl.shape[1]),
                random_state=0).fit(sl[ref_cells])
        z = p.transform(sl).astype(np.float32)
        if z.shape[1] < chan_dim:
            z = np.hstack([z, np.zeros((len(z), chan_dim - z.shape[1]), np.float32)])
        blocks.append(z)
        exp_var[m] = round(float(p.explained_variance_ratio_.sum()), 4)
    cells = np.hstack(blocks)

    # 药物：与 L1/L2 一致，基座拟合于全部药物
    dp = PCA(n_components=min(drug_dim, drugs_raw.shape[0]), random_state=0).fit(drugs_raw)
    drugs = dp.transform(drugs_raw).astype(np.float32)
    if drugs.shape[1] < drug_dim:
        drugs = np.hstack([drugs, np.zeros((len(drugs), drug_dim - drugs.shape[1]), np.float32)])
    exp_var["drug"] = round(float(dp.explained_variance_ratio_.sum()), 4)
    return cells.astype(np.float32), drugs.astype(np.float32), exp_var


def infer_chunked(model, ci_t, di_t, tc, td, idx, chunk=16384):
    """分块推理：逐行独立，数值与一次性前向完全相同，但避免单次前向
    batch 过大触发 CUDA "invalid configuration argument"（deepdtf 在
    36.5 万行测试集上稳定复现）。"""
    mus, sigs = [], []
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            c = idx[s:s + chunk]
            out = model(tc[ci_t[c]], td[di_t[c]])
            if isinstance(out, tuple):
                mu, v, a, b = out
                ale = b / (a - 1.0)
                epi = b / (v * (a - 1.0))
                sig = torch.sqrt(ale + epi)
            else:
                mu = out
                sig = torch.ones(len(c), dtype=mu.dtype, device=mu.device)
            mus.append(mu.cpu().numpy())
            sigs.append(sig.cpu().numpy())
    return np.concatenate(mus), np.concatenate(sigs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extdir", default=os.path.normpath(os.path.join(HERE, "data", "processed_ext")))
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--dirs", default="")           # 支持 0-3 序号 / 精确 tag / 子串
    ap.add_argument("--epochs", type=int, default=40)      # 与 run_benchmark.py (L1) 一致
    ap.add_argument("--batch", type=int, default=2048)     # 与 run_benchmark.py (L1) 一致
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "l4_transfer_unified.csv"))
    ap.add_argument("--desc", default=os.path.join(HERE, "l4_unified_representation.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = args.extdir
    elig = pd.read_csv(os.path.join(d, "response_ext_eligible.csv"))
    ci = np.load(os.path.join(d, "pair_cell_row.npy"))
    di = np.load(os.path.join(d, "pair_drug_row.npy"))
    y = elig["y_auc"].to_numpy(np.float32)
    r2 = elig["r2_mean"].to_numpy(float)
    wall = np.clip(np.nan_to_num(r2, nan=0.0), 0.0, 1.0).astype(np.float32)
    ds = elig["dataset"].to_numpy()
    dsets = set(ds)

    CL = [s for s in CL_SOURCES if s in dsets]
    q1, q2 = np.nanquantile(r2, 1 / 3), np.nanquantile(r2, 2 / 3)
    tier = np.where(r2 >= q2, "high", np.where(r2 <= q1, "low", "mid"))

    # ---- 唯一的表征，四个方向共用 ----------------------------------------
    t0 = time.time()
    cells, drugs, exp_var = build_shared_representation(d)
    print("[repr] shared basis built in %.0fs  cell_dim=%d drug_dim=%d  "
          "explained=%s" % (time.time() - t0, cells.shape[1], drugs.shape[1], exp_var), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # torch 2.x 的 transformer 快速路径在部分 (seq, batch, head) 组合下会触发
    # "CUDA error: invalid configuration argument"（deepdtf 在 BeatAML2->CL 复现崩溃）。
    # 关闭快速路径走等价的 Python 慢路径，数值结果相同。
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        print("[torch] MHA fastpath disabled (slowpath is numerically equivalent)", flush=True)
    except Exception as e:  # 旧版 torch 无此开关
        print("[torch] MHA fastpath switch unavailable: %s" % e, flush=True)
    tc = torch.from_numpy(cells).to(dev)
    td = torch.from_numpy(drugs).to(dev)
    ci_t = torch.from_numpy(ci.astype(np.int64)).to(dev)
    di_t = torch.from_numpy(di.astype(np.int64)).to(dev)

    canonical = []
    for t in CLINICAL:
        if t in dsets:
            canonical.append((CL, [t]))
    for t in CLINICAL:
        if t in dsets:
            canonical.append(([t], CL))

    tokens = [s.strip() for s in args.dirs.split(",") if s.strip()]
    picked = []
    for i, (a, b) in enumerate(canonical):
        tag = "%s->%s" % ("+".join(a), "+".join(b))
        if not tokens or str(i) in tokens or tag in tokens or any(t in tag for t in tokens):
            picked.append((i, tag, a, b))

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    done = set()
    rows = []
    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        done = set(zip(old["direction"], old["model"], old["seed"].astype(int)))
        # 关键：把已有行载入 rows，后续落盘才不会把之前的层覆写掉
        rows = old.to_dict("records")
        print("[resume] %d rows already present in %s" % (len(old), os.path.basename(args.out)), flush=True)

    json.dump({"cell_dim": int(cells.shape[1]), "drug_dim": int(drugs.shape[1]),
               "explained_variance": exp_var,
               "basis": "PCA fit once per modality on ALL cells (union of both domains); "
                        "drug PCA fit on ALL drugs; no labels used",
               "directions": [c[1] for c in canonical],
               "split_seed": SPLIT_SEED, "n_cells": int(cells.shape[0]),
               "n_drugs": int(drugs.shape[0])},
              open(args.desc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    total = len(picked) * len(models) * len(seeds)
    k = 0
    for idx, tag, train_ds, test_ds in picked:
        # 每个方向固定自己的划分种子 -> 与循环顺序、与模型、与种子都无关
        rng = np.random.RandomState(SPLIT_SEED + 100 * idx)
        tr_all = np.where(np.isin(ds, train_ds))[0]
        te = np.where(np.isin(ds, test_ds))[0]
        perm = rng.permutation(len(tr_all))
        nv = int(0.1 * len(tr_all))
        va = tr_all[perm[:nv]]
        tr = tr_all[perm[nv:]]
        print("\n=== [%d] %s  n_train=%d n_val=%d n_test=%d  train_cells=%d test_cells=%d"
              % (idx, tag, len(tr), len(va), len(te),
                 len(np.unique(ci[tr])), len(np.unique(ci[te]))), flush=True)
        for name in models:
            for seed in seeds:
                k += 1
                if (tag, name, seed) in done:
                    print("  [%3d/%3d] %-12s seed%d  skip (done)" % (k, total, name, seed), flush=True)
                    continue
                t1 = time.time()
                try:
                    if name.startswith("reliadrpv3"):
                        from reliadrp_v3 import ReliaDRPV3
                        kw = {"n_mix": 2} if "mix2" in name else {"n_mix": 1}
                        if "md0.1" in name:
                            kw["mod_drop"] = 0.1
                        if "md0.2" in name:
                            kw["mod_drop"] = 0.2
                        model = ReliaDRPV3(**kw).to(dev)
                    else:
                        model = zoo.build(name).to(dev)
                    ww = wall if (name == "reliadrp" or name.startswith("reliadrpv3")) else None
                    model = train_one(model, tr, va, ci, di, y, cells, drugs, ww, None, seed,
                                      args.epochs, args.batch)
                    model.eval()
                    mu_v, sig_v, _, _ = infer(model, ci_t, di_t, tc, td, va)
                    mu_t, sig_t = infer_chunked(model, ci_t, di_t, tc, td, te)
                    stud = bool(np.std(sig_t) > 1e-8)
                    s_cal = np.abs(y[va] - mu_v) / (sig_v + 1e-8) if stud else np.abs(y[va] - mu_v)
                    q = q_conformal(s_cal)
                    pp, mm = picp_mpiw(y[te], mu_t, sig_t, q, stud)
                    row = {"direction": tag, "model": name, "seed": seed,
                           "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                           "rmse": round(float(np.sqrt(np.mean((y[te] - mu_t) ** 2))), 4),
                           "pearson": round(float(stats.pearsonr(y[te], mu_t)[0]), 4),
                           "picp": round(pp, 4), "mpiw": round(mm, 4),
                           "mu_min": round(float(mu_t.min()), 4), "mu_max": round(float(mu_t.max()), 4)}
                    for t in TIERS:
                        m = tier[te] == t
                        if m.sum() < 10:
                            continue
                        p_cp, _ = picp_mpiw(y[te][m], mu_t[m], sig_t[m], q, stud)
                        qt = q_conformal(s_cal[tier[va] == t]) if (tier[va] == t).sum() > 5 else q
                        p_gc, _ = picp_mpiw(y[te][m], mu_t[m], sig_t[m], qt, stud)
                        row["picp_cp_%s" % t] = round(p_cp, 4)
                        row["picp_gc_%s" % t] = round(p_gc, 4)
                except Exception as e:
                    # 单格失败不杀全程：记录并跳过，下次续跑会重试该格
                    print("  [%3d/%3d] %-12s seed%d  FAILED: %s: %s"
                          % (k, total, name, seed, type(e).__name__, e), flush=True)
                    continue
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out, index=False)
                print("  [%3d/%3d] %-12s seed%d  r=%+.4f rmse=%.4f picp=%.3f "
                      "mu[%.2f,%.2f] (%.0fs)"
                      % (k, total, name, seed, row["pearson"], row["rmse"], pp,
                         row["mu_min"], row["mu_max"], time.time() - t1), flush=True)

    allrows = pd.read_csv(args.out) if os.path.exists(args.out) else pd.DataFrame(rows)
    nd, nm, ns = allrows.direction.nunique(), allrows.model.nunique(), allrows.seed.nunique()
    print("\nwrote %s : %d rows = %d directions x %d models x %d seeds (grid %s)"
          % (args.out, len(allrows), nd, nm, ns,
             "COMPLETE" if nd * nm * ns == len(allrows) else "incomplete"))


if __name__ == "__main__":
    main()
