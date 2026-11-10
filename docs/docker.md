# Docker Quick Start

本文从 GHCR 拉取 finer 镜像，在 Atlas A2 / Ascend 910B
上启动 Qwen3-4B，提供 OpenAI 兼容 HTTP API。容器内由 `script/run-server.sh`
启动 Rust 前端和 C++ 后端。

设备与驱动挂载参考 [vLLM Ascend 的 Docker 使用说明](https://docs.vllm.ai/projects/ascend/en/latest/getting_started/quick_start.html)。
finer 使用自己的启动脚本和 JSON 配置；模型加载、图编译和推理由 C++ 后端完成。

需要 LiteLLM 网关和管理页面时，见 [Compose 部署指南](../docker/litellm/README.md)。
服务划分、配置来源和启动依赖见 [Compose 实现说明](../docker/litellm/implementation.md)。

## 环境要求

- aarch64 Linux、Docker，以及已安装且可用的 Ascend 驱动和固件。
- 本地 Hugging Face 模型文件，包括配置、tokenizer、权重及分片索引。

镜像已包含 CANN 9.1.0-beta.3、HPCKit/HMPI 和前后端二进制。
ENTRYPOINT 自动加载 CANN、NNAL 和 HPCKit 环境，无须激活宿主机的 micromamba 环境。

先确认宿主驱动正常，并选择空闲设备：

```bash
npu-smi info
```

## 拉取镜像

以下命令在宿主机的 **Bash** 中执行，后续步骤复用同一 shell 的变量。
配置文件保存在当前目录下的 `tmp/docker/`。

```bash
IMAGE=ghcr.io/huawei-llm-inference-framework/finer:0.2.0-beta-2026.11.02
docker pull "$IMAGE"
```

## 准备模型和配置

`MODEL_DIR` 必须是直接包含 `config.json` 和 tokenizer 文件的模型目录。
如果下载目录下还有一层 snapshot hash，应指向该子目录。

```bash
MODEL_DIR="$HOME/work/model/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c"
CONFIG_DIR="$PWD/tmp/docker"
mkdir -p "$CONFIG_DIR" "$HOME/.cache"

cat > "$CONFIG_DIR/qwen3-4b.json" <<'JSON'
{
  "model": "/model",
  "engine": {
    "n_context": 8192,
    "max_num_request": 4,
    "max_num_batched_tokens": 32,
    "logits_topk": 128,
    "tensor_parallel_size": 1,
    "optimization": {
      "ge_precision_mode": "auto_bf16",
      "fuse_rope": true,
      "fuse_attention": true
    }
  },
  "generation": {
    "max_new_tokens": 256,
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 50,
    "seed": 42
  },
  "server": {
    "served_model_name": "qwen3-4b",
    "host": "0.0.0.0",
    "port": 8000,
    "engine_ready_timeout_secs": 1800
  }
}
JSON
```

按实际位置修改 `MODEL_DIR`；容器通过只读挂载在 `/model` 访问模型。
`server.host=0.0.0.0` 使容器内服务可经 Docker 端口映射访问。
以下示例只发布到宿主 `127.0.0.1`；需要远程访问时，可将映射中的该地址替换为
宿主对外提供服务的地址。

单卡和 TP2 共用这些参数：

```bash
COMMON_ARGS=(
  --shm-size=1g
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
  -v /usr/local/dcmi:/usr/local/dcmi:ro
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v "$MODEL_DIR:/model:ro"
  -v "$HOME/.cache:/root/.cache"
)
```

宿主驱动库按原路径只读挂载，CANN 工具包使用镜像内版本。
缓存目录单独保存在宿主；该挂载不保证所有 GE 编译产物都被持久化。

## 单卡启动

使用物理 NPU 0：

```bash
docker run --rm --name finer-qwen3-4b \
  "${COMMON_ARGS[@]}" \
  --device /dev/davinci0 \
  -v "$CONFIG_DIR/qwen3-4b.json:/config/qwen3-4b.json:ro" \
  -p 127.0.0.1:18000:8000 \
  -it "$IMAGE" bash
```

进入容器后，在容器的 Bash 中启动服务：

```bash
bash script/run-server.sh /config/qwen3-4b.json
```

首次启动需要加载权重并编译 prefill、decode 图，可能等待数分钟。
日志出现 `READY sent` 和 `starting OpenAI server` 后再请求接口。
`docker ps` 显示容器运行中不代表模型已就绪。

launcher 默认关闭 Qwen3 thinking；需要开启时，在 `docker run` 的镜像名之前加入
`-e CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'`。
上下文和批处理形状由配置决定，修改后需要重启服务。完整字段见[配置参考](configuration.md)。

## 双卡 Tensor Parallel

使用物理 NPU 0、1。若已启动单卡服务，先在其终端按 Ctrl+C 停止服务，再执行
`exit` 退出容器，然后回到宿主 Bash。

若直接使用 TP2，先完成前面的变量、配置和
`COMMON_ARGS` 准备。确认两卡空闲、互联可用且配置了设备 IP：

```bash
npu-smi info
npu-smi info -t topo
/usr/local/Ascend/driver/tools/hccn_tool -i 0 -ip -g
/usr/local/Ascend/driver/tools/hccn_tool -i 1 -ip -g
```

生成 TP2 配置并启动：

```bash
sed 's/"tensor_parallel_size": 1/"tensor_parallel_size": 2/' \
  "$CONFIG_DIR/qwen3-4b.json" > "$CONFIG_DIR/qwen3-4b-tp2.json"

docker run --rm --name finer-qwen3-4b-tp2 \
  "${COMMON_ARGS[@]}" \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  -e DEVICES=0,1 \
  -e OMPI_ALLOW_RUN_AS_ROOT=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/bin/hccn_tool:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v "$CONFIG_DIR/qwen3-4b-tp2.json:/config/qwen3-4b.json:ro" \
  -p 127.0.0.1:18002:8000 \
  -it "$IMAGE" bash
```

进入容器后启动服务：

```bash
bash script/run-server.sh /config/qwen3-4b.json
```

镜像默认以 root 运行，因此设置两个 `OMPI_ALLOW_RUN_AS_ROOT*` 变量供 HMPI 启动。
`run-server.sh` 根据 TP 数调用 `mpirun-ge.py`，通过 `hccn_tool` 查询设备 IP，
自动生成 HCCL rank table 并启动两个 rank。日志中的 `[1,0]`、`[1,1]` 分别对应两个 rank。
每个 rank 单独设置 `ASCEND_RT_VISIBLE_DEVICES`，只看到分配给自己的卡。

当前镜像的 TP launcher 将 `DEVICES` 同时用于 rank table 的物理卡号和运行时可见设备编号。
使用其他卡需要先处理物理/逻辑编号映射，不能只替换 `--device` 和 `DEVICES`。
设备数须与配置中的 TP 数一致，模型维度也须满足切分要求，详见[单机 Tensor Parallel](model-tp.md)。
本例单机 TP2 使用 Docker bridge 网络，无须 privileged、host 网络或 host IPC。

## 验证接口

服务就绪后，另开一个宿主机终端执行以下请求。单卡使用 `18000`，TP2 将 `BASE_URL` 改为 `http://127.0.0.1:18002`：

```bash
BASE_URL=http://127.0.0.1:18000
curl --noproxy '*' -fsS "$BASE_URL/health"
curl --noproxy '*' -fsS "$BASE_URL/v1/models"

curl --noproxy '*' -fsS "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"What is 2 + 3? Answer with just the number."}],"temperature":0,"max_tokens":32}'

curl --noproxy '*' -fsS -N "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Name the capital of France."}],"temperature":0,"max_tokens":32,"stream":true}'
```

`/health` 成功时返回 HTTP 200，响应体可以为空；`/v1/models` 应包含 `qwen3-4b`。
非流式响应包含生成文本，流式响应以 SSE `data:` 输出并以 `[DONE]` 结束。

## 停止与排查

在运行服务的终端按 Ctrl+C，等待服务退出并返回容器 Bash 提示符，再执行：

```bash
exit
```

容器随 Bash 退出，由 `--rm` 自动删除。在宿主机运行 `npu-smi info` 确认设备释放。
再次使用时重新执行 `docker run` 和容器内的启动命令。

常见问题：

- **模型加载失败**：确认 `MODEL_DIR` 指向实际 snapshot，而非其上层目录；配置中的
  `model` 是容器路径 `/model`。不要只挂载权重而遗漏 tokenizer、配置和分片索引。
- **端口暂时不可用**：先查看 `docker logs`，确认计算图编译和 READY 已完成。
  编译超过等待上限时，可在 JSON 中增大 `server.engine_ready_timeout_secs`。
- **TP 启动提示找不到 `hccn_tool` 或查不到设备 IP**：检查工具挂载和对应卡的 IP 查询结果。
- **`docker exec` 找不到运行库**：`docker exec` 不会重新执行镜像 ENTRYPOINT。
  需要运行 Python、后端或 HMPI 工具时，先显式加载环境，例如：

  ```bash
  docker exec "$CONTAINER" bash -c '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
    source /usr/local/Ascend/cann/share/info/ascendnpu-ir/bin/set_env.sh &&
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=${ATB_CXX_ABI:-1} &&
    source /opt/HPCKit/latest/setvars.sh --use-gcc &&
    mpirun --version
  '
  ```
