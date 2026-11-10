#!/bin/bash
# 正确性评测：对已启动的 server 跑 OpenCompass。
#
#   bash script/benchmark/bench-correctness.sh gsm8k    # 1319 题，4-shot CoT
#   bash script/benchmark/bench-correctness.sh mrcr     # 长上下文（OpenAI MRCR，需 n_context 足够）
#
# 地址、模型名、tokenizer 在 env.local.sh。并发和长度档在 opencompass/eval_<项>.py。
# 结果：eval-results/<项>/<时间戳>/outputs/**/summary_*.csv，准确率表工具结束时会打。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
[ -f "$HERE/env.local.sh" ] || { echo "!! 缺少 $HERE/env.local.sh" >&2; exit 1; }
set +u
source "$HERE/env.local.sh"
set -u

ITEM="${1:-gsm8k}"
CFG="$HERE/opencompass/eval_$ITEM.py"
[ -f "$CFG" ] || { echo "!! 没有这个测试项：$ITEM（gsm8k / mrcr / quick）" >&2; exit 2; }

# oc_limit（LIMIT 抽样）在 opencompass/ 目录下，让配置里的 import 找得到
export PYTHONPATH="$HERE/opencompass:${PYTHONPATH:-}"

# OpenCompass 会在 work-dir 下再加一层自己的时间戳
RUN_DIR="$REPO_ROOT/eval-results/$ITEM"
mkdir -p "$RUN_DIR"
opencompass "$CFG" --work-dir "$RUN_DIR"

