#!/bin/bash
# Config-driven launcher: 一个 JSON 同时驱动前端与后端。
#   用法: run-server.sh [backend/config/xxx.json]
#
# config 里本脚本消费的键（都在 "server" 段，后端解析时忽略未知键）：
#   served_model_name         — OpenAI API 里的模型名（缺省：模型路径 basename）
#   host                      — OpenAI HTTP 监听地址（缺省 127.0.0.1）
#   port                      — OpenAI HTTP 端口（缺省 8000）
#   engine_ready_timeout_secs — 前端等引擎 READY 的每阶段超时（缺省 1800；
#                               大图编译慢时调大；HELLO/READY 两阶段各自计时）
# 以及 engine.tensor_parallel_size（缺省 1）：>1 时经 script/mpirun-ge.py
# 用 HMPI 启动 N 个 rank，卡号取 DEVICES 或 ASCEND_RT_VISIBLE_DEVICES。
#
# CHAT_TEMPLATE_KWARGS — 传给前端 --default-chat-template-kwargs（缺省关 thinking）
set -e

BACKEND_CONFIG="${1:-backend/config/example_qwen3_server.json}"
HANDSHAKE_PORT=29550   # 后端握手地址硬编码为 tcp://127.0.0.1:29550（server/config.hpp）

config_vars="$(python3 - "$BACKEND_CONFIG" <<'PY'
import json, shlex, sys
cfg = json.load(open(sys.argv[1]))
model  = cfg["model"]
server = cfg.get("server", {})
served = server.get("served_model_name") or model.rstrip("/").rsplit("/", 1)[-1]
print("MODEL=" + shlex.quote(model))
print("SERVED_NAME=" + shlex.quote(served))
print("HOST=" + shlex.quote(server.get("host", "127.0.0.1")))
print("PORT=%d" % int(server.get("port", 8000)))
print("READY_TIMEOUT=%d" % int(server.get("engine_ready_timeout_secs", 1800)))
print("TP=%d" % int(cfg.get("engine", {}).get("tensor_parallel_size", 1)))
PY
)" || {
  echo "failed to parse backend config: $BACKEND_CONFIG" >&2
  exit 1
}
eval "$config_vars"

if [ -z "${CHAT_TEMPLATE_KWARGS:-}" ]; then
  CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
fi

echo "=== config=$BACKEND_CONFIG model=$MODEL served=$SERVED_NAME host=$HOST port=$PORT tp=$TP ==="

echo "=== Starting Rust frontend ==="
cd rust-frontend
./target/release/vllm-rs serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --engine-ready-timeout-secs "$READY_TIMEOUT" \
    --default-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \
    --data-parallel-size 1 \
    --data-parallel-size-local 0 \
    --handshake-port "$HANDSHAKE_PORT" &
FRONTEND_PID=$!
cd ..

sleep 2

echo "=== Starting backend ==="
if [ "$TP" -gt 1 ]; then
  python3 script/mpirun-ge.py "$TP" -- ./build/backend/engine_server "$BACKEND_CONFIG" &
else
  ./build/backend/engine_server "$BACKEND_CONFIG" &
fi
BACKEND_PID=$!

trap "kill $FRONTEND_PID $BACKEND_PID 2>/dev/null; exit" INT TERM
wait
