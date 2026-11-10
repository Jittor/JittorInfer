#!/bin/bash
# 装正确性评测工具 OpenCompass（pip 到 env.local.sh 里的 conda 环境）。
# torch 装 CPU 版，免得被拉 GB 级 nvidia-cuda 库（NPU 机器用不上）。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.local.sh" ] || { echo "!! 缺少 $HERE/env.local.sh" >&2; exit 1; }
set +u
source "$HERE/env.local.sh"
set -u

python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install opencompass==0.5.4
echo "=== 完成 ==="
