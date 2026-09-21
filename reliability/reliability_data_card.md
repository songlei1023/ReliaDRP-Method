# 数据卡片：response_reliability（ReliaDRP 可信度分层 v1）

> 生成日期：2026-09-15 ｜ 生成脚本：`../reliability.py` ｜ 依赖：`new_data/processed/`

## 1. 概述

在 `response_long.csv`（492,534 条 (细胞系, 药物) 药敏测量，CTRPv2 + GDSCv2）之上，
为每条测量附加**标签可信度**字段：三个分量质量分 `q_fit / q_repeat / q_agree`、
融合权重 `reliability_w ∈ [0,1]`、以及分层 `reliability_tier ∈ {high, mid, low}`。

用于 ReliaDRP 的条件覆盖诊断（E1）、可靠度加权共形（E2）与不确定性解耦（E3）。

## 2. 来源与拼接

| 来源文件 | 提供字段 | 用途 |
|---|---|---|
| `response_long.csv` | `cell_orig, drug, aac, ic50, ach, dataset` | 基准表 |
| `ctrpv2_response.csv` | `HS, E_inf, EC50, aac_recomputed, ic50_recomputed` | 曲线拟合质量（CTRPv2） |
| `gdscv2_response.csv` | 同上 | 曲线拟合质量（GDSCv2） |
| `drugs_ctrpv2_feat.csv` / `drugs_gdscv2_feat.csv` | `canonical_smiles` | 跨源药物对齐 |

- 拼接键：`(dataset, cell_orig, drug)`；**curve_join_missing = 0**（492,534/492,534 全部匹配）。
- 跨源对齐键：`(ach, canonical_smiles)`。

## 3. 可靠性定义

### q_fit（曲线拟合质量，可用率 100%）
由重复内的**退化标志比例**惩罚：

```
pen = 0.45·frac(HS≈0) + 0.25·frac(EC50≥1e6)
    + 0.20·frac(ic50_recomputed 缺失) + 0.10·frac(E_inf≈0)
q_fit = clip(1 − pen, 0, 1)
```

- `HS≈0`（|HS|<1e-6）：无剂量依赖，拟合退化。
- `EC50≥1e6`：超出测试剂量上界（截断哨兵值）。
- `ic50_recomputed` 缺失：EC50 未落在测试剂量范围内（外推）。
- `E_inf≈0`：无效应（权重最低，因其与真实耐药混淆）。

### q_repeat（同源重复一致性，可用率 20.3%）
```
q_repeat = clip(1 − std(aac)/0.2, 0, 1)   仅当 n_repeat ≥ 2，否则 NaN
```
尺度 0.2 取自观测到的组内 `aac` 标准差分布（中位 ~0.029–0.038，最大 0.65）。

### q_agree（跨源一致性，可用率 6.5%）
对 `(ach, canonical_smiles)` 同时出现在 CTRPv2 与 GDSCv2 的对：
```
q_agree = clip(1 − |aac_CTRPv2 − aac_GDSCv2| / 0.2, 0, 1)
```

### 融合与分层
```
reliability_w = 加权平均(可用分量; w = q_fit:0.4, q_repeat:0.3, q_agree:0.3)
reliability_tier = 按 reliability_w 排名三等分（low / mid / high）
```

## 4. 发布统计（v1，seed 无关）

| 指标 | 值 |
|---|---|
| 行数 / 唯一 (ds,cell,drug) 对 | 492,534 / 422,268 |
| q_fit / q_repeat / q_agree 可用率 | 100% / 20.3% / 6.5% |
| q_fit / q_repeat / q_agree 均值 | 0.850 / 0.748 / 0.593 |
| reliability_w 均值 ± 标准差 | 0.837 ± 0.167（min 0.057，max 1.000） |
| 分层计数 | high 164,178 / mid 164,178 / low 164,178 |
| 分层（CTRPv2） | high 132,898 / mid 111,326 / low 105,796 |
| 分层（GDSCv2） | high 31,280 / mid 52,852 / low 58,382 |

## 5. 已知偏差与局限（诚实声明）

1. **q_fit 与生物学混淆**：低 `E_inf`/高 `EC50` 也可能是**真实耐药**而非坏测量。
   `E_inf≈0` 权重已压低（0.10）；建议 E1 中同时报告"仅 q_agree/q_repeat 分层"的对照。
2. **分量覆盖率低**：`q_agree` 仅 6.5%、`q_repeat` 仅 20.3%，故多数样本 `w ≈ q_fit`。
   跨源复制受限于两数据集共享的 29 个 canonical SMILES 药物。
3. **分层存在并列**：`q_fit` 取离散值导致 `w` 大量并列；三等分使用 `rank(method="first")`，
   同分样本的层归属按行序打破，**同分不同层非实质差异**。
4. **GDSCv2 整体偏低**：主要因 `ic50_recomputed` 缺失率 51.7%（CTRPv2 为 34.4%），
   反映 GDSCv2 剂量范围更窄，而非数据质量更差；跨数据集比较需谨慎。
5. **未使用完整剂量-反应曲线**：当前仅用拟合参数。可升级为 `zenodo 17982056`
   （CTRP/GDSCv1/GDSCv2/PRISM 完整曲线）以改进 `q_fit`。

## 6. 使用方式

```python
import pandas as pd
rel = pd.read_csv("HDRP/reliability/response_reliability.csv")
high = rel[rel.reliability_tier == "high"]          # E4 高可信子集
w    = rel.set_index(["dataset","cell_orig","drug"]).reliability_w  # E2 RW-CP 权重
```

字段：`dataset, cell_orig, drug, ach, aac, ic50, n_repeat, aac_std, hs_med, einf_med, ec50_med, q_fit, q_repeat, q_agree, reliability_w, reliability_tier`

## 7. 验证方法

- 复现：`python HDRP/reliability.py` → 覆盖写 `reliability/response_reliability.csv` 与 `reliability_summary.json`。
- 一致性：`curve_join_missing` 应为 0；三层计数应相等（±并列）。
- 下游：E1 对 `results/preds/{P0,P1,P2}_evidential.npz` 按 `reliability_tier` 分组重算 PICP/MPIW，
  与 `results/conformal.csv` 的边际值对照，检验"低可信层条件覆盖 < 名义 0.90"。

## 8. 许可

本卡片仅引用 `new_data/processed/` 下已下载的公开数据（CTRPv2、GDSCv2、DepMap 24Q2）；
原始数据许可遵循各来源条款。分发本派生表时须同时附来源与许可说明。
