#!/usr/bin/env bash
# L4 统一评估：16 模型 x 4 方向 x 5 种子，训练口径与 L1 (run_benchmark.py) 一致。
#
# 按"种子分层"顺序执行：每跑完一层，所有 (方向 x 模型) 的种子数都相同，
# 因此任何时候中断，已落盘的都是一份**完整的统一网格**，不会出现
# 旧数据那种 5/5/3/3 混种子数的情况。
#
# 可反复重跑：脚本会跳过已完成的 (direction, model, seed)。
#
# 预估耗时（RTX 4060 Laptop 8GB）：正向方向 329k 训练行约 170s/格，
# 反向 BeatAML2->CL 约 25s/格，PDX_Bruna->CL 约 3s/格
# => 单层 64 格约 1.7h，5 层合计约 8~9h。

set -u
cd "$(dirname "$0")/.." || exit 1   # 仓库根

PY="E:/anaconda/python.exe"
OUT="l4_transfer_unified.csv"

for s in 0 1 2 3 4; do
  echo "############ seed layer $s ############"
  "$PY" -u l4/run_l4_transfer_unified.py --seeds "$s" --out "$OUT"
  echo "############ seed layer $s done ############"
done

echo "ALL SEED LAYERS DONE -> $OUT"
