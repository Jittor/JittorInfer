# 环境准备与运行示例

本文介绍 JittorInfer 的源码安装、模型准备、单卡与单机 TP 推理，以及常用配置。
项目技术与特性概览见 [README](../README.md)。除特别说明外，命令均在仓库根目录的
**Bash** 中执行。

## 选择部署方式

- **源码安装**：按本文操作，使用当前检出的代码构建前后端。
- **独立容器**：按 [Docker 使用指南](docker.md) 拉取镜像，准备模型和设备/驱动挂载，
  启动 Qwen3-4B 单卡或双卡 TP2 服务。
- **网关与管理页面**：按 [LiteLLM Compose 指南](../docker/litellm/README.md) 配置
  模型、设备与凭据，启动 finer、LiteLLM 和 PostgreSQL，再通过 API 或管理页面注册模型路由。

预构建镜像包含 CANN、HPCKit/HMPI 和前后端二进制，ENTRYPOINT 自动加载运行环境，
无需在宿主机编译或激活 micromamba。当前容器文档使用的
`0.2.0-beta-2026.11.02` 镜像不保证包含之后合入的 Qwen MoE、分页 attention 等更新；需要最新功能时，
请使用源码安装。镜像地址、端口、挂载及容器 TP 设备映射限制以 Docker 指南为准。

## 环境要求

部署基线为 **Atlas A2 / Ascend 910B、aarch64 Linux、CANN 9.1.0.beta3**。
宿主机需预先安装匹配的昇腾驱动与固件，并确保当前进程有权访问已分配的 NPU。
其他昇腾型号需更换对应算子包并重新验证，不属于这一部署基线。

以下工具链用于源码安装，容器使用自身工具链：

| 依赖 | 说明 |
| --- | --- |
| CANN Toolkit 与 910B 算子包 | 9.1.0.beta3，由 [环境配方](env/910b.yaml) 安装 |
| CMake / Python | 环境配方使用 CMake 4.3.2 / Python 3.11 |
| GCC / G++ | 支持 C++20；当前构建使用系统 GCC 10.3.1，环境配方不安装编译器 |
| Rust | 单独安装 rustup，前端固定使用 Rust 1.95 |
| HMPI | 按 [HMPI 安装指南](hmpi.md) 安装并加载，编译和运行使用同一套环境 |

**单卡也需要 MPI 依赖**：后端统一通过 MPI 初始化进程组，单卡时为单进程。
预构建镜像已包含该依赖。

## 源码安装

### 1. 获取源码

```bash
git clone https://github.com/Jittor/JittorInfer.git
cd JittorInfer
git -c url."https://github.com/".insteadOf=git@github.com: \
  submodule update --init --recursive
```

子模块固定到主仓库记录的版本。上面的命令在本次拉取时将 GitHub SSH 地址转换为
HTTPS，不修改全局 Git 配置。

### 2. 准备环境

先安装 [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
和 [HMPI](hmpi.md)，然后执行：

```bash
micromamba env create -f docs/env/910b.yaml
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

每次编译、测试或启动服务前，加载环境。`HPCKIT_ROOT` 替换为实际安装位置：

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate forch
source "$CONDA_PREFIX/Ascend/cann/set_env.sh"

export HPCKIT_ROOT="$HOME/software/HPCKit/26.1.RC1"
source "$HPCKIT_ROOT/hmpi/gcc/env/setvars.sh" --mode=release

export PATH="$HOME/.cargo/bin:$PATH"
export PROTOC_INCLUDE="$CONDA_PREFIX/include"
```

`PROTOC_INCLUDE` 用于前端编译时查找 protobuf 的 `.proto` 文件；前后端实际使用
ZMQ + MessagePack 通信。环境配方不是完整锁文件，已有环境应核实 CANN 版本和
依赖。Python protobuf 的兼容说明见 [环境注意事项](env/protobuf.md)。

### 3. 编译

在仓库根目录按以下顺序执行，首次 Rust 构建需要联网下载工具链及依赖：

```bash
CMAKE_BUILD_TYPE=Release bash script/build-third-party.sh
CMAKE_BUILD_TYPE=Release bash script/build-forch.sh
CMAKE_BUILD_TYPE=Release bash script/build-backend.sh
(cd rust-frontend && cargo build --release --bin vllm-rs)
```

默认构建根目录为 `build/`，三个 C++ 构建脚本分别使用其下的 `third_party/`、
`forch/` 和 `backend/` 子目录。`BUILD_DIR` 表示共同的构建根目录，
但启动脚本仍固定使用 `build/backend/engine_server`，本文请保留默认路径。
可设置 `NPROC` 控制 C++ 编译并行数。

## 单卡示例

### 1. 准备模型与配置

以 **Qwen3.5-0.8B** 为例，准备完整的 Hugging Face 格式本地模型目录，包含
`config.json`、tokenizer 与 chat template 相关文件，以及 `model.safetensors`
或带 `model.safetensors.index.json` 索引的分片权重。模型权重需自行获取并遵守其许可证。

```bash
cp backend/config/example_qwen3_5_server.json local.qwen3.5.json
```

将 `local.qwen3.5.json` 中的 `model` 改为模型目录的**实际绝对路径**，例如
`/data/models/Qwen3.5-0.8B`。仓库示例中的 `/beegfs/model/...` 是开发机路径，
不能直接用于其他机器。首次运行可保留其余参数，API 模型名为 `qwen3.5-0.8b`。

### 2. 启动服务

在已加载环境的终端中启动服务，将 `0` 替换为已分配的 NPU 卡号：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 bash script/run-server.sh local.qwen3.5.json
```

脚本先启动 Rust 前端，再启动 C++ 后端。后端在启动时编译并加载计算图，
大模型可能需要数分钟。等待日志出现 `READY sent` 和 `starting OpenAI server`
后再请求接口；进程启动不代表模型已就绪。

### 3. 请求与验证

在另一个终端发送流式聊天请求：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [{"role": "user", "content": "请简要介绍一下昇腾推理。"}],
    "max_tokens": 128,
    "temperature": 0.7,
    "stream": true
  }'
```

OpenAI 兼容客户端使用 `http://127.0.0.1:8000/v1` 作为 `base_url`，模型名与
`server.served_model_name` 保持一致。默认仅监听本机；对外服务可配置
`server.host="0.0.0.0"`，并自行设置鉴权和网络访问控制。

也可以在已加载环境的终端中检查并发请求是否正常返回文本：

```bash
python script/check-serving.py --model qwen3.5-0.8b -n 4
```

### 4. Thinking 与停止

启动脚本默认关闭 thinking。对支持该开关的模型，可在请求中传入
`"chat_template_kwargs": {"enable_thinking": true}`，或启动时统一开启：

```bash
CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}' \
  ASCEND_RT_VISIBLE_DEVICES=0 bash script/run-server.sh local.qwen3.5.json
```

停止服务时，在运行启动脚本的终端按 Ctrl+C。更换配置后需重新启动；不要在同一张卡上
同时运行上面的两种启动示例。

## 单机多卡

以 Qwen3.5 的双卡配置为例，先复制示例并修改其中的模型路径：

```bash
cp backend/config/example_qwen3_5_tp.json local.qwen3.5-tp.json
```

加载与单卡相同的 CANN、HMPI 环境，确认两张卡已分配且互联可用，再启动：

```bash
DEVICES=0,1 bash script/run-server.sh local.qwen3.5-tp.json
```

`engine.tensor_parallel_size` 决定后端进程数，`DEVICES` 指定物理卡号。
请按实际显存容量设置并发与上下文长度；Q/KV head、FFN 与词表的切分约束见
[Tensor Parallel 文档](model-tp.md)。当前启动脚本使用固定握手端口 `29550`，
不要在同一网络命名空间内同时启动多套服务。

## 模型配置示例

| 模型 | 权重精度 | 单卡配置 | 多卡配置 |
| --- | --- | --- | --- |
| Qwen3.5 | BF16，部分 GDN 参数为 FP32 | [0.8B](../backend/config/example_qwen3_5_server.json) | [TP2](../backend/config/example_qwen3_5_tp.json) |
| Qwen3.5 MoE | BF16，部分 GDN 参数为 FP32 | 暂无单卡示例 | [35B-A3B / TP2](../backend/config/example_qwen3_5_moe_tp.json) |
| Qwen3.8 | BF16 | [27B](../backend/config/example_qwen3_8_server.json) | [TP8](../backend/config/example_qwen3_8_tp.json) |
| Qwen3 | BF16 | [0.6B](../backend/config/example_qwen3_server.json) | [TP2](../backend/config/example_qwen3_tp.json) |
| Qwen3 MoE | BF16 | [30B-A3B](../backend/config/example_qwen3_moe_server.json) | [TP2](../backend/config/example_qwen3_moe_tp.json) |
| Llama 2 / 3.x | FP16 | [Llama-2-7B](../backend/config/example_llama.json) | [TP2](../backend/config/example_llama_tp.json) |
| DeepSeek-V2 / V2-Lite | BF16 | [V2-Lite-Chat](../backend/config/example_deepseek_v2_server.json) | [TP2](../backend/config/example_deepseek_v2_tp.json) |

配置是部署起点，不代表所有规模和卡数的组合均可直接运行。未列出的架构、量化或
视觉变体不自动包含在支持范围内。Qwen3.8 示例通过
`model_options.gdn_param_dtype="bf16"` 声明 GDN 参数的存储精度，需与实际权重一致。

## 配置与兼容性

一份 JSON 同时驱动前后端。完整字段见 [配置参考](configuration.md)，
请求生命周期见 [后端架构](architecture.md)。常用部署字段如下：

| 字段 | 用途 |
| --- | --- |
| `model` | 本地 Hugging Face 模型目录 |
| `engine.n_context` | 每个请求的上下文容量，包含 prompt 与已生成 token |
| `engine.max_num_request` | 同时执行的请求槽数；额外请求排队 |
| `engine.max_num_batched_tokens` | 预填充 / 混合步骤的总 token 预算，须不小于槽数 |
| `engine.tensor_parallel_size` | TP 进程数，默认 1 |
| `engine.kv_cache.*` | 分页开关、前缀复用、页大小、页池容量及淘汰策略，详见配置参考 |
| `engine.logits_topk` | NPU 侧候选裁剪上限，必须为正；CPU 在这些候选上采样 |
| `engine.optimization.*` | GE 精度模式及 RoPE、Attention、MoE、GDN 融合选项 |
| `engine.optimization.fuse_moe` | Qwen / DeepSeek MoE 的稀疏专家执行路径；关闭时使用参考路径 |
| `generation.*` | 后端采样默认值；请求参数可覆盖 |
| `server.served_model_name` / `server.host` / `server.port` | API 模型名、HTTP 监听地址与端口 |
| `server.engine_ready_timeout_secs` | 等待引擎就绪的超时，大图编译较慢时可调大 |
| `model_options.gdn_param_dtype` | Qwen3.5 / 3.8 的 GDN 参数存储精度声明 |

- **Qwen3.5 / 3.8 预算**：默认融合 GDN 路径要求
  `max_num_batched_tokens = max_num_request × k`，其中整数 `1 ≤ k ≤ 8`。
  Qwen3.5 dense / MoE 共用这一约束。`k=1` 时预填充与解码共用逐 token 图；
  其他模型可在总预算内灵活分配预填充 token。
- **MoE 并行**：当前是专家中间通道的 TP 切分，不是按专家分布到各卡的 EP。
  `fuse_moe=true` 采用稀疏派发、GroupedMatmul 与结果合并；从对应 MoE 示例开始，
  不要仅替换 dense 配置中的模型路径而忽略容量和优化选项。
- **静态计算图**：修改容量或优化选项需重启；到达 `n_context` 上限的请求以长度限制结束。
- **分页与容量**：融合 attention 路径默认启用分页；关闭融合 attention 时使用 dense 存储。
  当前页大小只支持 128 token。`engine.kv_cache.num_pages` 是每个 rank 的页编号预算，
  各层的数据池独立分配，不自动乘以请求槽数；默认 0 表示刚好容纳一个最大上下文。
  页池不足时会抢占请求并重算，多并发场景应按显存和实际上下文总量调大页池，
  不要将槽数误当成已预留的缓存容量。完整约束见 [配置参考](configuration.md#enginekv_cache)。
- **前缀复用**：分页路径默认复用完整的纯文本前缀块，可以单独关闭而保留分页。
  Qwen3.5 / 3.8（含 Qwen3.5 MoE）因 GDN 状态始终不做跨请求前缀复用，
  其 full-attention 缓存仍可分页。机制说明见 [KV cache 与 attention](kv-cache.md)。
- **当前服务范围**：纯文本生成、单机 TP；尚未实现视觉编码和动态形状执行。
- **API 兼容范围**：不等同于完整 vLLM 功能集。`frequency_penalty`、`presence_penalty`、
  `repetition_penalty` 与 `min_tokens` 尚未实现，非默认设置会被忽略并告警。

## 调试与评测

后端测试位于 `backend/tests/`，通过 `-DBUILD_TESTING=ON` 启用。
其中 `test_model_forward` 可直接驱动模型 Runner，无需启动 HTTP 服务；
主机测试包含前缀缓存与调度抢占检查；NPU 测试包含分页 attention 数值参考和完整模型服务验证。
构建与运行方法见 [测试与验证](testing.md)。
性能测试与 OpenCompass 正确性评测见 [评测工具指南](../script/benchmark/README.md)。

## 从 v0.1.0 迁移

v0.2.0 保留 OpenAI 兼容 API 的使用方式，但不是旧配置的直接替换：

- 使用 Hugging Face safetensors 模型目录，旧 GGUF 文件不能直接加载。
- 使用 `backend/config/` 中的 JSON 配置，替代旧版 `configs/` 中的 YAML。
- 使用 `script/run-server.sh`，替代 `llama-server` / `llama-mpi-server` 启动方式。
- 环境升级到 CANN 9.1.0.beta3，并按当前支持范围重新选择模型、精度与并行配置。
