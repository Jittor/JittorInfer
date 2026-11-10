# KV cache 与 attention

本文描述 `backend/cache/` 与 `models/common/` 的 cache、attention 方案和分页前缀
复用。组件关系见 [后端架构](architecture.md)，配置字段见 [配置参考](configuration.md)。

## 分层

token cache 的寻址状态在每个 runner 中只有一份：所有层的 K/V、MLA latent/RoPE
使用同一组物理行号，但各 Pool 的设备数据仍独立存储。GDN state/conv 则使用独立的
按槽寻址，不参与 token 页表。

| 层次 | 文件 | 负责 |
|---|---|---|
| `SlotMap` | `cache/slot-map.hpp` | 槽游标、逻辑位置到物理行的映射、padding 落点、块表；分页时持有唯一的 `PageManager`、前缀 hash、引用计数和用量 |
| `AttentionMetadata` | `cache/attention-metadata.hpp` | 每个 token 预算一组图占位符；每步生成一次 scatter 写入行、mask、长度和块表，供所有层共享 |
| `GDNMetadata` | `cache/gdn.hpp` | GDN 层共享的 conv 读写行映射与 padding valid |
| `Pool` | `cache/pool.hpp` | 一块设备张量及其占位符，提供 scatter / overwrite / zero_row，不知道寻址 |
| 语义 cache | `cache/full-attention.hpp`、`mla.hpp`、`gdn.hpp` | 每层持有自己的 Pool，按共享写入行写入并返回池视图 |

| 语义 cache | 存储 | 用途 |
|---|---|---|
| `FullAttentionCache` | K、V 两个 Pool（几何可不同） | Qwen3、Llama、Qwen3.5 full 层；DeepSeek eager 参考 |
| `MLACache` | latent、k_pe 两个 Pool | DeepSeek absorbed MLA |
| `GDNCache` | state、conv 两个 Pool，每槽一行 | Qwen3.5 GDN 层 |

`SlotMap` 有两种实现：

- `DenseSlotMap`：每槽一段连续行。`block_size=0`（eager）时 padding 写到池尾的
  dump 行；`block_size>0` 时每槽对齐到块并留 guard 区，块表是图常量。
- `PagedSlotMap`：物理行为 `[dummy 页 | 数据页 | padding 页]`，数据页按需分配，
  块表每步作为运行时输入。

## 一步的顺序

调度器先通过 `runner.can_run` 探测容量，成功后才调用 `CacheRunner::step`：

```
Scheduler: runner.can_run → SlotMap.can_fit（纯探测，不够则抢占并重新规划）
CacheRunner.step: pack → reserve（分配本步页）
                      → feed（写入行、mask、长度、块表、各层池）→ run → commit
```

mask 和块表读取真实物理映射，必须在 reserve 之后生成；`commit` 推进游标，并且只在
图成功写入之后发布完整块的 hash。`can_fit` 不分配、不 pin 页面，成功后必须执行
同一计划。

## Attention 方案

`cache::AttentionType` 是并列的执行方案。布局、mask 与存储寻址在算子内部耦合，
不再拆分：

| 方案 | Query 排布 | KV 读取 | 输入 | 用途 |
|---|---|---|---|---|
| `Eager` | compact 或按槽排布（Qwen3.5/3.8） | 整个池 | FP32 加性 mask `[T, rows]`，与打包方式一致 | 参考路径 |
| `DenseFIA` | compact，视为一条序列 | 整个池 | INT8 mask `[T, rows]` | 消融对照 |
| `TND` | compact + dummy 序列 | 块表 | 累积 Q 长度、KV 长度、causal 模板 | fused prefill |
| `BSND` | 每槽固定 k 行 | 连续槽视图 | 每槽 KV 长度；k>1 时逐 query mask | dense fused decode / Qwen3.5 |
| `PagedBSND` | 每槽固定 k 行 | 块表 | 每槽 KV 长度；k>1 时逐 query mask | 分页 fused decode / Qwen3.5 |

runner 选择方案：`fuse_attention=false` 时全部 Eager；否则 prefill 用 TND，decode
按存储选择 BSND 或 PagedBSND。Qwen3.5 两张图都按槽排布（GDN 需要每槽固定行），
full 层使用 decode 方案。

模型只调用两个分派入口（`models/common/attention/attention.hpp`）：
`full_attention(type, ...)` 和 `mla_attention(type, ...)`。eager GQA 参考实现在公共
`eager-attention.hpp`；DeepSeek 的展开式 eager MLA 依赖权重，留在模型中。
模型与公共 attention 的分工见 [attention 分层](../backend/models/common/attention/README.md)。

### 布局细节

- 池是 token-major `[rows, N*D]`。TND / PagedBSND 把它 reshape 为
  `[blocks, 128, N*D]`；BSND 视为 `[B, stride, N, D]`。不做每步全池重排。
- 每槽逻辑步长 `stride = (n_context / 128 + 1) * 128`，块表列数为 `stride / 128`，
  其中除法为整数除法；分页存储未分配的列指向 dummy 页 0，dense 块表固定映射到槽区间。
- TND 的图尾 padding 拆成最多 32 行的 dummy 序列，使累积 Q 长度等于图宽度；dummy
  只读块 0，写入落 padding 区。
- 空槽在 BSND 图中仍占行，有效可见 KV 长度为 1。分页读 dummy 页 0，dense 读该槽
  第一行；写入走独立 padding/guard 区，输出丢弃，游标不推进。
- MLA decode：分页时使用独立长度、不加 mask；dense 时用统一 scan 长度加逐槽 mask，
  规避 CANN 的 split/merge 路径。
- FP16 TND 使用 `inner_precise=2`；CANN910 默认值在 Llama 上会生成不存在的
  tiling key，编译成功但首次执行失败。

### Qwen3.5 GDN

- fused GDN 的 prefill 总预算 `max_num_batched_tokens = max_num_request × k`，`1 ≤ k ≤ 8`；
  k=1 时只编译 decode 图。
- GDN 每槽作为长度 k 的序列执行，padding 的 beta/g 置零，conv history 只保存到
  最后一个有效 token。
- 持久 state 每槽一块。CANN910 要求算子输入块数不小于 T，prefill 调用前临时 tile
  到 B×k 块，写回时只取前 B 块。
- 写回 scatter 只有注册为图输出才执行，且必须折叠成一个输出（GE 的 OutputCopy
  按输出个数收费）。槽清零图必须在主图之前编译，否则 TBE 编译子进程会崩溃。
- state/conv 在新请求准入时清零，释放槽只更新宿主簿记。
- `fused_gdn_attention=false` 时使用 FP32 eager 递推与 FP32 state，只编译每槽
  1 token 的图。

## 分页与前缀复用

`PageManager`（`cache/page-manager/`）管理物理页元数据、引用计数、无 hash 空闲页
和 hash 索引；LRU、LFU、FIFO 各一个文件。有引用的页不能淘汰，无 hash 的空闲页优先
分配。引用归零的页保留 hash，直到真正被复用才删除索引。

- hash 链：SHA-256（`utils/sha256.hpp`）覆盖版本域、父 digest、块大小、token 数和
  token id，固定小端编码。
- 准入：`attach_prefix` 逐块计算链式 hash 并 acquire，首个 miss 即停止。只复用
  prompt 最后一个 token 之前的整块，保证仍有输入产生 logits。
- 发布：`commit` 收到按槽分组的真实 token，补满一块后才 publish；padding 不参与。
- 多模态请求不做 token-only 前缀复用；Qwen3.5 有递归状态，始终不复用。
- 数据页数 `num_pages` 是每个 rank 的共享页编号预算，默认刚好容纳一个最大上下文；
  每层各数据 Pool 都有相应的物理页，共享的是映射而不是不同层的 K/V 内容。

## 对外报告

- `kv_cache_usage`：被请求引用的页占比（dense 存储按已用 token 折算）。
- `prefix_cache_stats`：自上次上报以来的请求数、查询 token 数和命中 token 数；
  抢占重算计入 `preempted_*`。最后一个请求结束、取消和空闲 reset 之后也会上报。
- `reset_prefix_cache`：分页缓存仍有请求持有页时返回 `false`，无引用时清空所有 hash
  并返回 `true`；dense 路径没有可复用前缀，直接返回 `true`。当前不支持清掉 running
  requests 的变体。HTTP 管理端点需启用 `VLLM_SERVER_DEV_MODE=1`。
- ReadyResponse 的 `num_gpu_blocks` / `block_size` / `kv_cache_size_tokens` 来自
  `SlotMap::info()`。
