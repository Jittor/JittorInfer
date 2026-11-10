#!/bin/bash
# 性能测速：对已启动的 server 跑 inference-benchmarker。
#
#   bash script/benchmark/bench-performance.sh --prefill 32 --decode 96 --vus 4
#   bash script/benchmark/bench-performance.sh --prefill 4096 --decode 1 --vus 1
#
# --prefill / --decode / --vus 必填。
# 地址、模型名、tokenizer 在 env.local.sh；--url / --tokenizer 可临时覆盖。
# --warmup 默认 30s，--duration 默认 120s。
# 指标看工具表格：decode 看 Throughput (tokens/sec)；prefill 看 TTFT (avg)，除以 prompt 长度得 tok/s。
# 前提：server 已启动；inference-benchmarker 已装（没有先跑同目录 install-inference-benchmarker.sh）。
# decode 并发高于 server 的 max_num_request、或 prefill 长于 n_context 时，需要先重启 server。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
[ -f "$HERE/env.local.sh" ] || { echo "!! 缺少 $HERE/env.local.sh" >&2; exit 1; }
set +u
source "$HERE/env.local.sh"
set -u

usage() {
  echo "用法: bash script/benchmark/bench-performance.sh --prefill N --decode N --vus N [--tokenizer HF_REPO] [--url URL] [--warmup 30s] [--duration 120s]" >&2
}

PREFILL_LEN=""
DECODE_LEN=""
VUS=""
TOKENIZER="$TOKENIZER_NAME"
URL="$BENCH_URL"
WARMUP="30s"
DURATION="120s"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefill|--decode|--tokenizer|--vus|--url|--warmup|--duration)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "!! $1 需要参数" >&2
        exit 2
      fi
      case "$1" in
        --prefill)   PREFILL_LEN="$2" ;;
        --decode)    DECODE_LEN="$2" ;;
        --tokenizer) TOKENIZER="$2" ;;
        --vus)       VUS="$2" ;;
        --url)       URL="$2" ;;
        --warmup)    WARMUP="$2" ;;
        --duration)  DURATION="$2" ;;
      esac
      shift 2
      ;;
    *) usage; echo "!! 未知参数：$1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PREFILL_LEN" || -z "$DECODE_LEN" || -z "$VUS" ]]; then
  usage
  echo "!! --prefill、--decode、--vus 必填" >&2
  exit 2
fi
if [[ -z "$TOKENIZER" || -z "$URL" || -z "$MODEL_NAME" ]]; then
  echo "!! 在 script/benchmark/env.local.sh 里设置 BENCH_URL、MODEL_NAME、TOKENIZER_NAME" >&2
  exit 2
fi
if ! [[ "$PREFILL_LEN" =~ ^[1-9][0-9]*$ && "$DECODE_LEN" =~ ^[1-9][0-9]*$ && "$VUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "!! --prefill、--decode、--vus 必须是正整数" >&2
  exit 2
fi

DATASET=hlarcher/inference-benchmarker
DATAFILE=share_gpt_0_turns.json

RUN_DIR="$REPO_ROOT/bench-results/p${PREFILL_LEN}-d${DECODE_LEN}-v${VUS}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR" && cd "$RUN_DIR"
exec "$REPO_ROOT/build/inference-benchmarker/target/release/inference-benchmarker" \
  --url "$URL" --tokenizer-name "$TOKENIZER" --model-name "$MODEL_NAME" \
  --benchmark-kind throughput --duration "$DURATION" --warmup "$WARMUP" \
  --max-vus "$VUS" \
  --dataset "$DATASET" --dataset-file "$DATAFILE" \
  --prompt-options "num_tokens=${PREFILL_LEN},max_tokens=${PREFILL_LEN},min_tokens=${PREFILL_LEN},variance=0" \
  --decode-options "num_tokens=${DECODE_LEN},max_tokens=${DECODE_LEN},min_tokens=${DECODE_LEN},variance=0"
