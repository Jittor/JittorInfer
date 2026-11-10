# 后端架构

JittorInfer 由两个进程组成：Rust 前端（`rust-frontend/`）提供 OpenAI 兼容 HTTP API、
tokenizer 与 chat template；C++ 后端（`backend/`）冒充 vLLM engine core，负责模型、
图编译与批量推理。两者通过 ZMQ + MessagePack 使用 vLLM 的 engine-core 协议通信，
wire 格式以 [`rust-frontend/ENGINE_PROTOCOL.md`](../rust-frontend/ENGINE_PROTOCOL.md)
为准。代码命名空间仍是 `finer::`（项目更名的历史遗留，勿批量改名）。

本文描述后端组件与请求生命周期；KV cache 与 attention 见 [KV cache](kv-cache.md)，
配置字段见 [配置参考](configuration.md)，多卡见 [Tensor Parallel](model-tp.md)。

## 组件与依赖

```
bin/engine_server.cpp            装配：解析配置、MPI、握手、主循环
  │           │
  ▼           ▼
Transport   Scheduler ─────────► ModelRunner（执行接口）          engine/
（wire）    （请求语义）               ▲
                          CacheRunner（公共执行流程）             models/common/runner.hpp
                            Qwen3 / Llama / DeepseekV2 / Qwen35 Runner   models/<name>/
                                 │
                                 ▼
                          SlotMap / AttentionMetadata / 各层 cache   cache/
```

依赖单向：`bin → server → engine`，`models → engine, cache → forch`。
engine 不知道任何具体模型，server/bin 不出现 `forch::Tensor`。

| 组件 | 位置 | 负责 | 不负责 |
|---|---|---|---|
| `Transport` | `server/transport.hpp` | ZMQ DEALER/PUSH、msgpack 编解码、两段握手；TP 时 rank0 收发并广播 | 理解请求含义 |
| `Scheduler` | `server/scheduler.hpp` | 槽状态机、等待队列、选图与 token 分配、抢占与重算、CPU 采样、统计上报 | 张量、图、cache 物理布局 |
| `ModelRunner` | `engine/model-runner.hpp` | scheduler 可见的执行契约：`GraphInfo` 表、`admit_request` / `can_run` / `step` / `reset_slot` / `kv_cache_info` / `reset_prefix_cache` | 采样、请求策略 |
| `CacheRunner` | `models/common/runner.hpp` | GE backend、图编译与执行、共享 cache 寻址、每步输入准备 | 模型数学 |
| 具体 runner | `models/<name>/runner-*.hpp` | 加载权重、建层 cache、声明模型图；特殊设备步骤（如 GDN 槽清零） | 重复实现公共 step 流程 |
| `RunnerLoader` | `models/loader.hpp` | 唯一的 `model_type → runner` 注册表 | |
| `Sampler` | `engine/sampler/` | 温度 → top-k → top-p；使用调用方传入的 RNG | 保存请求 RNG（由 scheduler 持有和播种） |

scheduler 与 runner 只交换纯数据：`GraphInfo`（图名、总 token 预算、每槽上限）、
`SlotStep`（每槽起始位置与本步 token）、`StepOutput`（每槽 top-k `vals` / `indices`）。
scheduler 看不到张量或物理页号。`OutputSink` 抽象输出与 TP 协调，生产实现是
`Transport`，测试使用记录输出的实现。

## 静态形状

每个 runner 在启动时编译固定形状的图，之后不再重建：

- Qwen3、Llama、DeepSeek V2：prefill 总预算为 `max_num_batched_tokens`，decode
  总预算为 `max_num_request`（正常 decode 每槽 1 token）；
- Qwen3.5/3.8：decode 每槽 1 行，fused GDN 的 prefill 每槽 k 行（`1 ≤ k ≤ 8`），
  k=1 时复用 decode 图；eager GDN 只编译每槽 1 行的图。另有设备槽清零图。

每步都填充到图的预算。修改 `n_context`、`max_num_request`、`max_num_batched_tokens`
或 KV 配置都需要重启。

## 请求生命周期

```
前端 Add（两帧：类型字节 + msgpack）
  → Request::from_msgpack（wire 解不通 → Error 结束）
  → Scheduler::handle_add：填采样默认值；拒绝空 prompt 和超出 n_context 的 prompt
  → 空闲槽且未在排空 → admit_to_slot，否则进入 waiting_
      admit_to_slot → runner.admit_request → cursor = cache_hint_len（命中的整块前缀）
  → step() 循环：
      1. 游标到达 n_context 的槽以 length 结束
      2. 未在排空时，waiting_ 依次准入空闲槽
      3. 选图（静态图优先）并在总预算内分配每槽 token
      4. runner.can_run 不通过 → 抢占最新请求、重新规划
      5. runner.step → 推进游标 → 采样 → 流式输出或结束
  → 结束：stop | length | abort | error，reset_slot 释放槽和 cache 引用
  → 每次状态变化后上报 scheduler stats（含最后一个请求结束后的归零值）
```

### 预算分配与 continuous batching

decode 槽每步各占 1 token，其余预算在 prefill 槽之间按顺序分配，单槽不超过剩余
prompt、剩余上下文和图的每槽上限。长 prompt 可以一步占满预算，超出部分自然分多步。
一张图可同时推进 decode 与 prefill 槽。

### 抢占

`can_run` 是无副作用的容量探测。不通过时，scheduler 抢占到达最晚的活跃请求：保留
请求、已生成 token、RNG 与多模态负载，释放其 cache，把 prompt 加已生成 token 作为
重算历史放回等待队列。之后进入**排空**：在所有存活请求结束前不准入新请求，避免
同一请求反复挂接前缀、反复被抢占。重算期间只有历史的最后一行会采样，RNG 不会被
中间行推进。只剩一个请求仍放不下表示配置不变量被破坏，直接报错。

TP 时各 rank 的容量判断取交集，采样结果以 rank0 为准。

### 采样

图内 TopK（`engine.logits_topk`）裁剪 logits，CPU 再做温度和 top-p。`sample_logits`
返回从 `indices` 读出的 vocab token id，不是 `vals` 中的位置。RNG 按请求保存，
不会因批组成或抢占重算额外推进；在候选分数与顺序相同的前提下，相同 seed 产生相同序列。
不同图/批形状可能有浮点误差或 TopK 并列顺序差异，不能承诺任意运行逐 token 相同。

## 多模态（预留）

wire 解码、区间登记和每步 `encode_vision` 派发已经打通，但还没有 runner 实现视觉图；
多模态请求按占位 token 处理，也不参与 token-only 前缀复用。

## 新增模型

1. 在 `backend/models/<name>/` 增加 `config-<name>.hpp`（解析 HF `config.json`）、
   `model-<name>.hpp`（用 `forch::nn` 与 `models/common/` 的 norm、MLP、MoE、RoPE、
   LM head 构图）和 `runner-<name>.hpp`（继承 `CacheRunner`）。
2. 模型 attention 层在投影、norm、RoPE 和 `cache.update` 之后调用
   `full_attention(type, ...)` 或 `mla_attention(type, ...)`，分工见
   [attention 分层](../backend/models/common/attention/README.md)。
3. runner 构造函数：用 `make_owned` 登记 loader 和模型（loader 先登记）；调用
   `create_slot_map`；为每层创建 cache；用 `build_prefill_decode` 或 `build_graph` 建图；
   覆写 `feed_caches`（有设备状态时再覆写 `on_admit`）。
4. 在 `models/loader.hpp` 的 `kRunners` 表加一行。
5. 遵守 `fuse_rope` / `fuse_attention` 和 `n_model_layers`，并保留 eager 参考路径。

## 调试

- `SPDLOG_LEVEL=debug` 打开每请求、每步日志（`[request]`、`[step]`、`[preempt]`）。
- `test_model_forward <config.json> --ids 1,2,3` 不启动前端、直接驱动 runner，
  适合新图冒烟；见 [测试与验证](testing.md)。
- SIGINT 直接 `_Exit(0)`（避免 GE finalize 触发 TBE 子进程错误），SIGTERM 正常退出。
