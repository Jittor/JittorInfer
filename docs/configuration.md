# 配置参考

后端与启动脚本读取同一个 JSON 文件，示例见 `backend/config/`。配置由
`bin::Config` 单点解析，各层只读自己的段；调度上限（`max_model_len`、
`max_num_request`）从 `engine` 段复制，保证与编译出的图一致。
推理图在启动时按这些形状编译一次，任何修改都需要重启。

```json
{
  "model": "/path/to/hf-model",
  "engine": {
    "n_context": 8192,
    "max_num_request": 4,
    "max_num_batched_tokens": 32,
    "logits_topk": 128,
    "tensor_parallel_size": 1,
    "kv_cache": {
      "paged": true,
      "prefix_caching": true,
      "block_size": 128,
      "num_pages": 0,
      "eviction_policy": "lru"
    },
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
    "served_model_name": "qwen3-0.6b",
    "host": "127.0.0.1",
    "port": 8000,
    "engine_ready_timeout_secs": 1800
  },
  "model_options": {}
}
```

## `engine`

| 字段 | 缺省 | 说明 |
|---|---|---|
| `n_context` | 4096 | 单请求最大 token 数（prompt + 生成），即前端看到的 `max_model_len`；游标到达时以 `length` 结束 |
| `max_num_request` | 4 | slot 数，即最大并发 |
| `max_num_batched_tokens` | 2048 | 每步总 token 预算（所有 slot 合计），不得小于 `max_num_request`；启用 fused GDN 的 Qwen3.5/3.8 要求是它的 1–8 倍 |
| `logits_topk` | 2048 | NPU 侧 TopK 裁剪，必须为正；需要全量 logits 时设为模型 vocab size |
| `tensor_parallel_size` | 1 | TP rank 数，须等于 MPI world size，见 [Tensor Parallel](model-tp.md) |
| `n_model_layers` | -1 | 调试用，正值构建前 N 层（不超过模型层数）；非正值为全部 |

### `engine.kv_cache`

| 字段 | 缺省 | 说明 |
|---|---|---|
| `paged` | `true` | 分页存储；仅在 `fuse_attention=true` 时生效，eager 路径使用 dense 存储并告警 |
| `prefix_caching` | `true` | 完整块前缀复用；关闭时仍分页但不查询、不发布 hash。Qwen3.5 始终不复用 |
| `block_size` | 128 | 当前只接受已验证的 128 |
| `num_pages` | 0 | 每个 rank 的共享逻辑页编号预算，各层每个数据 Pool 都分配这么多页；不乘 slot 数。0 表示刚好容纳一个最大上下文，显式值也不得小于它 |
| `eviction_policy` | `lru` | 无引用缓存页的淘汰策略：`lru` / `lfu` / `fifo` |

页池放不下本步计划时，scheduler 抢占最新请求并重算，详见
[KV cache](kv-cache.md)。按显存和实际并发上下文总量选择 `num_pages`；
`max_num_request × ceil(n_context / 128)` 是所有槽均跑满且无共享时的最坏情况容量，
不是必须预留的下限。较小池通过共享与抢占运行，但可能付出重算代价。

### `engine.optimization`

| 字段 | 缺省 | 说明 |
|---|---|---|
| `ge_precision_mode` | `origin` | GE 精度模式：`origin` / `auto_fp16` / `auto_bf16` / `force_fp16` |
| `fuse_rope` | `false` | 使用融合 RoPE 算子 |
| `fuse_attention` | `false` | 使用 FusedInferAttention：Qwen3/Llama/DeepSeek 为 TND prefill + BSND/PagedBSND decode；Qwen3.5/3.8 full 层两张图都用 BSND/PagedBSND。关闭走 eager 参考路径 |
| `fused_gdn_attention` | `true` | Qwen3.5 GDN 使用原生融合递推与 BF16 state；关闭时为 FP32 eager 递推，要求 `ge_precision_mode=origin`，且只编译每槽 1 token 的图 |
| `fuse_moe` | `false` | MoE 使用融合路由与 grouped matmul |

## `generation`

请求未指定时使用的采样默认值：`max_new_tokens`（512）、`temperature`（1.0，
0 为 greedy）、`top_p`（0.9）、`top_k`（50）、`seed`（42）。请求可逐项覆盖。
`frequency/presence/repetition_penalty` 与 `min_tokens` 只解析不实现，非默认值会告警。

## `server`

以下 HTTP 字段由 `script/run-server.sh` 读取并传给前端，后端不直接读取它们：

| 字段 | 缺省 | 说明 |
|---|---|---|
| `served_model_name` | 模型目录名 | OpenAI API 的 `model` 字段 |
| `host` | `127.0.0.1` | HTTP 监听地址；`0.0.0.0` 监听所有 IPv4 接口，需自行做访问控制 |
| `port` | 8000 | HTTP 端口 |
| `engine_ready_timeout_secs` | 1800 | 前端等待后端 READY 的每阶段超时；大图编译慢时调大 |

后端还读取 `server.handshake_address`（默认 `tcp://127.0.0.1:29550`）和
`server.vllm_version`（默认 `finer-engine-0.1`）。启动脚本的前端握手端口仍固定为
29550，所以仅修改 JSON 中的地址会导致两侧不匹配；自定义地址时需同步设置前端启动参数。
环境变量 `CHAT_TEMPLATE_KWARGS` 传给前端，缺省关闭 Qwen thinking。

## `model_options`

不透明的模型私有选项，由各模型的 config 自行解析。当前只有 Qwen3.5 的
`gdn_param_dtype`（`fp32` / `bf16`）：GDN 标量参数在 0.8B/27B 检查点中为 FP32，
在 Qwen3.8 系列中为 BF16。

## 握手中报告的 KV 信息

ReadyResponse 的 `num_gpu_blocks`、`block_size`、`kv_cache_size_tokens` 来自实际
存储（dense 存储按 slot 容量折算），不是配置项；`dtype` 是协议兼容值，不控制精度。
