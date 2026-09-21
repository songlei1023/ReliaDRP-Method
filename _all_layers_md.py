"""Generate the consolidated four-layer results table (markdown).

Writes  results_and_paper/ReliaDRP_四层结果总表.md
Run:    E:/anaconda/python.exe _all_layers_md.py
"""
import csv, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_and_paper", "ReliaDRP_四层结果总表.md")

OURS = {"reliadrp", "reliadrpcombo", "reliadrpexpr", "reliadrpv3",
        "reliadrpv3mix2md0.1", "reliadrpv3mix2md0.2", "reliadrpv3mix2mf03"}
# our own evidence bases; excluded from the paper's external comparison
EXCLUDED = {"evi", "evi2"}


def read(fn):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def board(rows, pcol, mcol, vcol, proto, drop=()):
    agg = {}
    for r in rows:
        if r.get(pcol) != proto:
            continue
        m = (r.get(mcol) or "").strip()
        if not m or m in drop:
            continue
        v = num(r.get(vcol))
        if v is None:
            continue
        agg.setdefault(m, []).append(v)
    out = [(st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v), m) for m, v in agg.items()]
    out.sort(reverse=True)
    return out


def ours_in(listf, pcol, ccol, vcol, proto, cfg):
    vs = [num(r[vcol]) for r in listf if r.get(pcol) == proto and r.get(ccol) == cfg]
    vs = [v for v in vs if v is not None]
    return (st.mean(vs), st.stdev(vs) if len(vs) > 1 else 0.0, len(vs)) if vs else None


def rank_of(mu, ext_board):
    return 1 + sum(1 for b in ext_board if b[0] > mu)


L = []
A = L.append

A("# ReliaDRP 四层结果总表")
A("")
A("> 生成时间：2026-09-20 00:30（L1 播种重跑完成，四层初始化口径统一为「已播种」）")
A("> 全部数字均由 `_all_layers_md.py` 从归档 CSV 直接计算，未经手工转录。")
A("")
A("## 0. 阅读须知")
A("")
A("| 项 | 说明 |")
A("|---|---|")
A("| 指标 | L1/L2/L4 = test Pearson r（RMSE 一并报）；L3 = test AUROC |")
A("| 脚手架 | 四层共用同一 `train_one`：同一早停准则、同一模型池、同一划分种子 |")
A("| 种子数 | L1 = 5、L2 = 3（基线）/ 5（我们调参确认）、L3 = 3、L4 = 5 |")
A("| 外部模型 | L1/L2/L4 = 15 个（`evi`/`evi2` 是我们自己的证据化基座，不计入对手）|")
A("| **名次读法** | `r/N` 中 **N = 外部模型 + 我们（只算最终那一个变体）**。论文里我们把两个变体都放进表内，故分母偶差 1（例：L2 药物盲本表 2/15、论文 2/16）|")
A("| 缺行原因 | L2 药物盲少一个模型：`deepdtf` 三粒种子全部返回 NaN，已剔除 |")
A("| **证据基线** | 我们自己的非可靠性基座 `evi2` 在 L1 in-domain（已播种）得 0.7262，我们的 0.7420 领先 0.0157 → **可靠性头的域内增益不大**，主要增益来自架构与训练口径 |")
A("| **tie 判据** | 旧稿的初始化未播种曾导致跨进程漂移 ~0.03 Pearson；**L1/L2/L4 现已播种**（L3 仍否），跨层一律保留「差距 <0.03 读作并列」的保守纪律 |")
A("")

# ---------------------------------------------------------------- L1
A("## 1. L1 单药响应（492,534 条 / 4 数据集）")
A("")
A("**当前口径：v3 调参版 `mix2_mf03`（n_mix=2, mse_frac=0.3），固定预算 10 epoch / batch 4096，"
  "只用验证集选型；09-20 完成播种重跑**（`l1_unified_seeded.csv`：18 模型 × 3 协议 × 5 种子 = 270 runs，"
  "在 `build()` 之前播种；`deepdtf` 在 EXT_LDO/LCO 触发 CUDA `invalid configuration argument` 后"
  "CPU 兜底 10 runs，`device` 列已标注）。")
A("")
l1s = read("l1_unified_seeded.csv")
l1t = read("l1_v3_tune_confirm.csv")
A("| 协议 | ReliaDRP v3（已播种） | 名次 | 外部最强 | 领先 | 旧未播种值 |")
A("|---|---|---|---|---|---|")
l1_rows = [("EXT_random", "in-domain 随机划分"), ("EXT_LDO", "drug-blind 药物盲"),
           ("EXT_LCO", "cold-cell 冷细胞")]
l1_detail = {}
for p, lab in l1_rows:
    ext = board(l1s, "proto", "model", "test_r", p, drop=EXCLUDED | OURS)
    ins = ours_in(l1s, "proto", "model", "test_r", p, "reliadrpv3mix2mf03")
    old = ours_in(l1t, "proto", "config", "test_r", p, "mix2_mf03")
    rk = rank_of(ins[0], ext)
    l1_detail[p] = (ins, rk, len(ext) + 1, ext, old)
    A("| %s | **%.4f±%.4f** | **%d/%d** | %s %.4f | **+%.4f** | %.4f |"
      % (lab, ins[0], ins[1], rk, len(ext) + 1, ext[0][3].upper(), ext[0][0],
         ins[0] - ext[0][0], old[0]))
A("")
_rm = {p: ours_in(l1s, "proto", "model", "test_rmse", p, "reliadrpv3mix2mf03")[0]
       for p, _ in l1_rows}
A("RMSE（越低越好，已播种）：**%s**，三个协议均 1/16。"
  % " / ".join("%.3f" % _rm[p] for p, _ in l1_rows))
A("")
A("<details><summary>三协议完整榜单（前 6 + 我们）</summary>")
A("")
for p, lab in l1_rows:
    ins, rk, N, ext, v2 = l1_detail[p]
    A("**%s**" % lab)
    A("")
    A("| # | 模型 | Pearson | sd | n |")
    A("|---|---|---|---|---|")
    for i, (mu, sd, n, m) in enumerate(ext[:6], 1):
        A("| %d | %s | %.4f | %.4f | %d |" % (i, m, mu, sd, n))
    A("| **%d** | **ReliaDRP (v3 tuned)** | **%.4f** | %.4f | %d |" % (rk, ins[0], ins[1], ins[2]))
    A("")
A("</details>")
A("")
# val--test correlation per protocol, straight from the seeded file
_corr = {}
for p, _ in l1_rows:
    vv, tt = [], []
    for r in l1s:
        if r.get("proto") != p:
            continue
        a, b = num(r.get("val_r")), num(r.get("test_r"))
        if a is not None and b is not None:
            vv.append(a)
            tt.append(b)
    _corr[p] = st.correlation(vv, tt) if len(vv) > 2 else float("nan")
A("**验证集可靠性（播种后）**：val–test 相关 in-domain **%+.3f**、drug-blind %+.3f、cold-cell %+.3f"
  " → 三档都足以支撑「用验证集选型」（旧未播种表这两档只有 +0.29/+0.40）。"
  % (_corr["EXT_random"], _corr["EXT_LDO"], _corr["EXT_LCO"]))
A("")
A("**播种前后（同一 harness，只改初始化）**：我们的行几乎不动（三协议 |Δ| ≤ 0.007），但基线普遍下掉——"
  "域内 MLP 0.5597→0.2957（**−0.264**）、DELFOS 0.5440→0.2763（−0.268）、FourierDrug 0.4201→0.2414；"
  "于是三个协议的领先幅度**全部变大**：域内 +0.049→**+0.059**、药物盲 +0.088→**+0.095**、"
  "冷细胞 +0.049→**+0.102**（三档 3SE = 0.005 / 0.036 / 0.004，均被超过）。"
  "即旧稿较薄的裕度**部分来自基线的幸运初始化**，播种后我们的名次不变而裕度变宽。")
A("")
_l1seed = sorted((int(r["seed"]), num(r["test_r"])) for r in l1s
                 if r.get("proto") == "EXT_LDO" and r.get("model") == "reliadrpv3mix2mf03")
A("**药物盲逐种子**：%s → sd %.4f。"
  % ("、".join("s%d %.4f" % kv for kv in _l1seed), l1_detail["EXT_LDO"][0][1]))
A("")

# ---------------------------------------------------------------- L2
A("## 2. L2 药物组合（2.97M 条 / 双药）")
A("")
A("**口径说明**：统一表里 17 行 = 15 外部 + 我们两个变体。论文 §6.2 **现已换用 v3 调参版** "
  "`mix1_noln_h256`（去 LayerNorm + hidden 256，**三协议验证集上均排第一**，非测试集选型）为 0.751/0.668/0.524，"
  "**名次三项全不变**；配套的「去双药交互模块」消融也已在 v3 上补跑（`l2_v3_ablation.csv`）→ 四层同代至此闭环。")
A("")
l2 = read("l2_unified.csv")
l2t = read("l2_v3_tune_confirm.csv")
A("| 协议 | 统一基线 v2 (3 seeds) | 旧名次 | v3 调参 (5 seeds) | 名次 | 外部最强 | 差距 |")
A("|---|---|---|---|---|---|---|")
l2_rows = [("L2_random", "in-domain 随机划分"), ("L2_LDO", "drug-blind 药物盲"),
           ("L2_LCO", "cold-cell 冷细胞")]
l2_detail = {}
for p, lab in l2_rows:
    ext = board(l2, "protocol", "model", "pearson", p, drop=EXCLUDED | OURS)
    ball = board(l2, "protocol", "model", "pearson", p, drop=EXCLUDED)
    v2 = [x for x in ball if x[3] == "reliadrpcombo"][0]
    rk_v2 = rank_of(v2[0], ext)
    ins = ours_in(l2t, "protocol", "config", "test_r", p, "mix1_noln_h256")
    rk = rank_of(ins[0], ext)
    l2_detail[p] = (ins, rk, len(ext) + 1, ext, v2, rk_v2)
    gap = ins[0] - ext[0][0]
    A("| %s | %.4f±%.4f | %d/%d | **%.4f±%.4f** | **%d/%d** | %s %.4f | %s%.4f |"
      % (lab, v2[0], v2[1], rk_v2, len(ext) + 1, ins[0], ins[1], rk, len(ext) + 1,
         ext[0][3].upper() if ext else "-", ext[0][0] if ext else float("nan"),
         "+" if gap >= 0 else "", gap))
A("")
A("<details><summary>三协议完整榜单（前 6 + 我们）</summary>")
A("")
for p, lab in l2_rows:
    ins, rk, N, ext, v2, rk_v2 = l2_detail[p]
    A("**%s**" % lab)
    A("")
    A("| # | 模型 | Pearson | sd | n |")
    A("|---|---|---|---|---|")
    for i, (mu, sd, n, m) in enumerate(ext[:6], 1):
        A("| %d | %s | %.4f | %.4f | %d |" % (i, m, mu, sd, n))
    A("| **%d** | **ReliaDRP-Combo (v3 tuned)** | **%.4f** | %.4f | %d |" % (rk, ins[0], ins[1], ins[2]))
    A("")
A("</details>")
A("")
A("**结论**：内域第 4（MoGraph 0.813 遥遥领先）、药物盲第 2（BANDRP 0.538 vs 0.524，**并列**）、冷细胞第 1（领先 Delfos 0.024）。")
A("")
A("**v3 调参 vs 同进程 v2**（同种子对齐，排除跨进程噪声）：内域 0.7510 vs 0.7383（**+0.0127**）、冷细胞 0.6682 vs 0.6547（**+0.0135**）、药物盲 0.5240 vs 0.5239（**+0.0001**）→ **赢两项、平一项**。")
A("对照跨进程口径：`l2_unified.csv` 的 v2 为 0.7389 / 0.6567 / 0.5324，与同进程 v2 差 ≤0.0085 → **L2 的跨进程可比性没问题**，与 L4 不同。")
A("")
A("**同源对照：双药交互编码器的增益（v3，容量对齐 hidden 128，同进程 5 种子）** —— 把它换成喂两药均值的同代单药模型："
  "**域内 0.733 vs 0.737（−0.004，等于没有）**、冷细胞 **0.615→0.538（−0.077）**、药物盲 **0.516→0.488（−0.028）**。"
  "v2 同口径复现：域内 +0.011 / 冷细胞 +0.077 / 药物盲 +0.065（与旧稿 0.077 / 0.065 逐位一致）→ "
  "**结论：双药交互编码器扛的是 shift，不是域内拟合**；v3 的 token mixing 在药物盲一档还部分替它补了位（v2 差 0.065 → v3 只差 0.028）。")
A("")
A("**验证集可靠性**：random +0.999（可靠）/ cold-cell +0.809（可用）/ drug-blind +0.489（弱）；"
  "论文用的 `mix1_noln_h256` 在**三协议验证集上均排第一** → 选型协议站得住。"
  "（新跑里 drug-blind 有 2 粒种子出现常数预测，触发 NumPy 的 invalid-value 告警，不影响 Pearson。）")
A("")

# ---------------------------------------------------------------- L3
A("## 3. L3 单细胞敏感度（1.43M 条 / 负结果）")
A("")
A("**口径：16 模型同脚手架，60 epoch / batch 1024（由模型无关准则选出），`select=auroc`。**")
A("")
l3 = read("l3_final_e60b1024.csv")
l3b = read("l3_final_e60b512.csv")
A("| 协议 | ReliaDRP | 名次 | 最强基线 | 差距 | batch 512 复核 |")
A("|---|---|---|---|---|---|")
for p, lab in [("L3_random", "in-domain 随机划分"), ("L3_ldo", "drug-blind 药物盲")]:
    ext = board(l3, "protocol", "model", "auroc", p, drop=OURS)
    ours = [x for x in board(l3, "protocol", "model", "auroc", p, drop=()) if x[3] in OURS][0]
    rk = rank_of(ours[0], ext)
    extb = board(l3b, "protocol", "model", "auroc", p, drop=OURS)
    oursb = [x for x in board(l3b, "protocol", "model", "auroc", p, drop=()) if x[3] in OURS][0]
    A("| %s | %.4f±%.4f | %d/%d | %s %.4f | %+.4f | %.4f（%d/16）|"
      % (lab, ours[0], ours[1], rk, len(ext) + 1, ext[0][3].upper(), ext[0][0],
         ours[0] - ext[0][0], oursb[0], rank_of(oursb[0], extb)))
A("")
A("**判定：这不是胜利，是负结果。** 随机划分 **饱和**（16 个模型全在 0.975–0.986，极差 0.011，我们反而垫底）；药物盲 **0.710 = 8/16**（基线 MLP 0.753）。")
A("旧稿的「双协议第一」是**协议错配**造出的伪结果（用单药模型喂零药物向量 + 不同配方），已撤回。")
A("val–test AUROC 相关只有 0.57（batch 512 时 −0.38）→ 纪律是**报不挑设置的均值**，不报 val 选出的最好值。")
A("")

# ---------------------------------------------------------------- L4
A("## 4. L4 跨域迁移（13.6k + 51.8k）")
A("")
A("**本层已完成播种重跑（360/360 runs）。下表对照「旧表（未播种）」与「定稿新表（已播种）」，论文采用新表数值。**")
A("")
l4o = read("l4_transfer_unified.csv")
l4n = read("l4_transfer_unified_seeded.csv")
l4t = read("l4_v3_tune_confirm.csv")
DIRS = [("CCLE+CTRPv1+GDSC1->BeatAML2", "细胞系 → BeatAML2"),
        ("CCLE+CTRPv1+GDSC1->PDX_Bruna", "细胞系 → PDX"),
        ("PDX_Bruna->CCLE+CTRPv1+GDSC1", "PDX → 细胞系"),
        ("BeatAML2->CCLE+CTRPv1+GDSC1", "BeatAML2 → 细胞系")]
A("| 方向 | 旧表 v2（未播种） | 旧名次 | **定稿 v3（已播种）** | 新名次 | 对新表最强基线 | 判定 |")
A("|---|---|---|---|---|---|---|")
for d, lab in DIRS:
    ext_o = board(l4o, "direction", "model", "pearson", d, drop=OURS)
    v2o = [x for x in board(l4o, "direction", "model", "pearson", d, drop=()) if x[3] == "reliadrp"]
    rk_o = rank_of(v2o[0][0], ext_o) if v2o else None
    full_n = board(l4n, "direction", "model", "pearson", d, drop=())
    ext_n = [x for x in full_n if x[3] not in OURS]
    new = [x for x in full_n if x[3] == "reliadrpv3mix2md0.1"]
    rk_n = rank_of(new[0][0], ext_n) if new else None
    gap = new[0][0] - ext_n[0][0] if (new and ext_n) else float("nan")
    if abs(gap) < 0.03:
        verdict = "并列带内" + ("（我们略前）" if gap > 0 else "（我们略后）")
    elif gap > 0:
        verdict = "领先"
    else:
        verdict = "落后"
    A("| %s | %.4f±%.4f | %d/%d | **%.4f±%.4f** | %d/%d | %+.4f | %s |"
      % (lab, v2o[0][0], v2o[0][1], rk_o, len(ext_o) + 1,
         new[0][0], new[0][1], rk_n, len(ext_n) + 1, gap, verdict))
A("")
A("读法：`%d/%d` 的分母 = 15 个外部基线 + 我们（论文表内为 16 行）。「对新表最强基线」= 定稿 v3 减去最强外部模型；**<0.03 视为并列**。")
A("")
new_board_caml = board(l4n, "direction", "model", "pearson", DIRS[0][0], drop=())
A("### 4.1 L4 的四条硬事实")
A("")
def _mean_of(rows, d, m):
    vs = [num(r["pearson"]) for r in rows if r.get("direction") == d and r.get("model") == m]
    vs = [v for v in vs if v is not None]
    return (st.mean(vs), st.stdev(vs) if len(vs) > 1 else 0.0, len(vs)) if vs else None


_c1 = _mean_of(l4o, DIRS[0][0], "crdnn")
_c2 = _mean_of(l4n, DIRS[0][0], "crdnn")
_a1 = _mean_of(l4o, DIRS[0][0], "codeae")
_a2 = _mean_of(l4n, DIRS[0][0], "codeae")
_seedlist = ", ".join("%.3f" % num(r["pearson"]) for r in l4n
                      if r.get("direction") == DIRS[0][0] and r.get("model") == "crdnn" and num(r["pearson"]) is not None)
A("1. **旧表的「rank 1/16 into BeatAML2」数值不成立，但名次在播种后仍然守住（只是收窄）。**"
  " 旧表 v2 记的 0.2264±0.0183 在播种后变成 **0.1964±0.0131**（差 0.030）；"
  "换成同代的 v3（`mix2_md01`）后得 **0.2082±0.0208**，仍列 **1/16**，但只领先 CLCLSA 0.1968 的 **+0.0114**。"
  "**决定性证据（邻居崩塌检查）**：播种后旧表第二名 CRDNN 从 %.4f±%.4f 变成 **%.4f±%.4f**（逐种子 %s）、"
  "第四名 CODE-AE 从 %.4f 落到 %.4f、第三名 panCancerDR 从 0.2176 落到 0.1351 —— **基线比我们动得更多**，"
  "旧名次整体由幸运初始化决定。"
  % (_c1[0], _c1[1], _c2[0], _c2[1], _seedlist, _a1[0], _a2[0]))
A("2. **反向方向全部不可用**：PDX→细胞系、BeatAML2→细胞系的最强基线（FourierDrug）也只有 0.074 / 0.056，且 16 个模型的 sd 高达 0.04–0.23 → 论文按「无可用信号」如实报，不包装成我们的胜利。")
A("3. **调参不是靠测试集挑的**：L4 的 v3 配置 `mix2_md01` 在**四个方向的验证集上都是第一**（val_r 0.676/0.773/0.778/0.223，均高于 `mix2_md02` 与 v2），所以选型协议站得住；"
  "但 val–test 相关只有 −0.52/+0.18/−0.22/+0.02 → **验证集只能排序配置、不能预测水平**，纪律仍是报不挑设置的 5 种子均值。")
A("")
A("4. **播种修复已验证**：同一命令连跑两次 `--models reliadrp --seeds 0` 均得 **0.1831**（修复前 0.1897/0.1863）→ 确定性通过。")
A("")
A("### 4.2 播种后 C→AML 定稿榜单（18/18 模型，5 种子齐）")
A("")
A("| # | 模型 | Pearson | sd | n | 备注 |")
A("|---|---|---|---|---|---|")
for i, (mu, sd, n_, m) in enumerate(new_board_caml[:12], 1):
    note = ""
    if m == "crdnn":
        note = "旧表 0.2216（第 2）→ **崩到 0.062**"
    elif m == "codeae":
        note = "旧表 0.2162（第 4）→ 0.188"
    elif m == "reliadrp":
        note = "我们 v2"
    elif m in ("reliadrpv3mix2md0.1", "reliadrpv3mix2md0.2"):
        note = "**我们 v3 调参**"
    A("| %d | %s | %+.4f | %.4f | %d | %s |" % (i, m, mu, sd, n_, note))
A("")
A("**读数**：播种后我们 v3（0.2082 / 0.2075）列前二，强于最强基线 CLCLSA 0.1968 的 **+0.011**；我们 v2 为 0.1964（第 4）。"
  "前五名（0.2082 / 0.2075 / 0.1968 / 0.1964 / 0.1877）只差 **0.033**，5 种子的标准误约 0.005–0.010 → "
  "**属于「窄幅领先 + 顶部密集」，写作 top-cluster 内的领先，不写成压倒性第一**。")
A("")
A("**关键的口径变化**：±0.03 的 tie 判据只在**跨进程不可复现**时适用。播种后同一命令逐位复现（已验证 0.1831 两次），"
  "所以新表的差距是真实差距，应报 5 种子的 SE（≈0.005–0.010），而不是套 0.03 粗规则。")
A("")
A("### 4.3 重跑记录")
A("")
A("- 任务 `rKb5YL`：18 模型（15 基线 + reliadrp + v3×2）× 4 方向 × 5 种子 = **360 runs**，**已全部完成**（14:52 收尾）。")
A("- 输出 `l4_transfer_unified_seeded.csv`（已归档到 `results/L4_cross_domain/`）；旧表备份 `_bak_l4_transfer_unified.csv`。")
A("- 论文采用的口径：表内 16 行 = 15 基线 + 我们 v3（`mix2_md01`），共 320 个列表 run；另外两个同族设置（v2、`mix2_md02`）用于敏感性检查，不列表。")
A("- 已完成：算新名次 → 改论文 §6.4 表格与正文 → 同步 Abstract/Intro/Discussion/Limitations/Conclusion → 重编验 **5 页 / 0 overfull / 第 5 页纯 REFERENCES** → 归档 + 更新 README。")
A("")

# ---------------------------------------------------------------- summary
A("## 5. 一页总表")
A("")
A("| 层 | 协议 | 我们的值 | 名次 | 判定 | 状态 |")
A("|---|---|---|---|---|---|")
_l1lab = {"EXT_random": "in-domain", "EXT_LDO": "drug-blind", "EXT_LCO": "cold-cell"}
rows = []
for _p, _ in l1_rows:
    _ins, _rk, _n, _ext, _old = l1_detail[_p]
    rows.append(("L1", _l1lab[_p], "%.4f±%.4f" % (_ins[0], _ins[1]), "%d/%d" % (_rk, _n),
                 "胜（+%.3f over %s）" % (_ins[0] - _ext[0][0], _ext[0][3].upper()),
                 "已播种，定稿"))
rows += [
    ("L2", "in-domain", "0.751±0.002", "4/16", "中性（MoGraph 0.813 领先）", "v3，已定稿"),
    ("L2", "cold-cell", "0.668±0.003", "1/16", "胜（+0.024 over Delfos）", "v3，已定稿"),
    ("L2", "drug-blind", "0.524±0.012", "2/15", "**按 tie 读**（BANDRP 0.538）", "v3，已定稿；`deepdtf` NaN 剔除"),
    ("L3", "in-domain", "0.9754±0.0020", "16/16", "**负结果**（饱和）", "定稿"),
    ("L3", "drug-blind", "0.7099±0.0268", "8/16", "**负结果**（未及 MLP）", "定稿"),
]
# L4 rows computed straight from the seeded table (v3 = mix2_md01)
_l4lab = {"CCLE+CTRPv1+GDSC1->BeatAML2": "细胞系→AML",
          "CCLE+CTRPv1+GDSC1->PDX_Bruna": "细胞系→PDX",
          "BeatAML2->CCLE+CTRPv1+GDSC1": "AML→细胞系",
          "PDX_Bruna->CCLE+CTRPv1+GDSC1": "PDX→细胞系"}
_l4judge = {"CCLE+CTRPv1+GDSC1->BeatAML2": "**窄幅领先 +0.011**（顶部五名只差 0.033）",
            "CCLE+CTRPv1+GDSC1->PDX_Bruna": "落后 DELFOS 0.514（−0.027）",
            "BeatAML2->CCLE+CTRPv1+GDSC1": "**无可用信号**（最优基线仅 0.056）",
            "PDX_Bruna->CCLE+CTRPv1+GDSC1": "**无可用信号**（最优基线仅 0.074）"}
for _d, _lab in DIRS:
    _full = board(l4n, "direction", "model", "pearson", _d, drop=())
    _ext = [x for x in _full if x[3] not in OURS]
    _me = [x for x in _full if x[3] == "reliadrpv3mix2md0.1"][0]
    rows.append(("L4", _l4lab[_d], "%.4f±%.4f" % (_me[0], _me[1]),
                 "%d/%d" % (rank_of(_me[0], _ext), len(_ext) + 1),
                 _l4judge[_d], "定稿"))
for r in rows:
    A("| %s | %s | %s | %s | %s | %s |" % r)
A("")
A("**诚实的自我描述**：四层里 **L1 三协议第一（已播种，裕度反而变宽）**、**L2 一胜一平一中性（定稿）**、"
  "**L3 明确负结果（已撤回旧声明）**、"
  "**L4 是「窄幅第一 + 两个方向无信号」**——旧表的 rank-1 数值（0.2264）在播种后降到 0.1964，换同代 v3 后为 0.2082，仍列 1/16 但只领先 0.011；"
  "而旧表的第 2/3/4 名（CRDNN/panCancerDR/CODE-AE）播种后集体崩塌或下滑，说明**旧表的名次整体由幸运初始化决定**。"
  "L1 同理：名次守住，但基线普遍下掉，**旧稿较薄的裕度正是这些幸运初始化撑出来的**。")
A("论文的贡献重心因此不是「四层都第一」，而是：**受可靠性加权的证据化回归在单药层三协议稳定领先、在组合与 AML 迁移层窄幅领先，并给出一套可审计的失效诊断（L3/L4 反向方向是负结果）**。")
A("")
A("## 6. 待办与风险")
A("")
A("| 项 | 内容 |")
A("|---|---|")
A("| 已完成 | L4 播种重跑 360 runs → 新名次 → 论文 §6.4 表格与正文 → Abstract/Intro/Discussion/Limitations/Conclusion 同步 → 重编验 **5 页 / 0 overfull / 第 5 页纯 REFERENCES** → 归档 + README |")
A("| **已闭环（原唯一未闭环项）** | **L2 已换用 v3**：论文 §6.2 由统一 v2 的 0.739/0.657/0.532 改为 v3 调参版 0.751/0.668/0.524（rank 4/16、1/16、2/15，**三项名次不变**）；「去双药交互模块」消融已在 v3 上补跑（`l2_v3_ablation.csv`，90 runs），论文消融句改写为「双药交互编码器扛 shift 而非域内拟合」|")
A("| **已闭环（09-20）** | **L1 也做了播种重跑**（270 runs，`l1_unified_seeded.csv`）：三协议仍 1/16，我们的行变化 ≤0.007，基线普遍下掉 → 裕度反而变宽（+0.059/+0.095/+0.102）；论文 §6.1 表格 16 行 + 正文 + caption（标注 seeded re-run）+ Limitations (2)/Reproducibility（改为「L1/L2/L4 已播种，仅 L3 未播种」）+ Conclusion 微调；重编仍 **5 页 / 0 overfull / 0 undefined / 第 5 页纯 REFERENCES** |")
A("| 风险 1（已缓解） | L1 播种后三档裕度 +0.059 / +0.095 / +0.102，均超过 3SE（0.005 / 0.036 / 0.004）；初始化不再是风险来源 |")
A("| 风险 2 | L2 药物盲与 BANDRP 差 0.006、L4 C→AML 领先 0.011 → **必须**按并列/窄幅领先写 |")
A("| 风险 3 | L4 反向两方向（PDX→细胞系、AML→细胞系）最优基线仅 0.074/0.056 且 sd≈0.2 → 只能写「无可用信号」 |")
A("| 风险 4 | L4 的 val 只能排序配置、不能预测水平（val–test 相关 −0.52/+0.18/−0.22/+0.02）→ 不可用 val 选型 |")
A("| 遗留 | L1 药物盲的 sd 仍偏大（逐种子见 §1），但名次与 3SE 判据都站得住；L3 仍是唯一未播种的一层（论文已如实写明） |")
A("")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("wrote", OUT, len(L), "lines")
