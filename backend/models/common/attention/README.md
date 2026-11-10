# Attention 分层

这里是无权重的 attention 计算与算子适配，不是完整的模型 attention 层。
`model-*.hpp` 与本目录的分工以“是否依赖模型参数/架构”和“是否只是算子调用约定”划分，
不能把同一次布局转换或执行路径选择在两边各实现一遍。

## 边界

| 层 | 负责 | 不负责 |
|---|---|---|
| `models/<name>/model-*.hpp` | 参数、TP 分片后的投影、归一化、RoPE、MLA 权重吸收、输出门控/投影；注册 cache 更新子图；调用公共 attention 分派入口 | 重复 scheme 分派、BSND/page 视图、CANN 属性、块表/长度的算子端绑定、设备执行 |
| `models/common/attention/` | 校验准备好的输入，转换成算子视图，绑定 mask/长度/块表、精度和 kernel workaround，返回未输出投影的 attention 结果 | 模型权重/配置、QKV 投影、RoPE 计算、cache 写入/分配/提交、图编译或执行 |
| `cache/` | 持久存储、scatter 更新、全 runner 共享的游标/页分配/块表；根据真实状态生成 mask/长度 | 权重吸收、输出投影、选择模型或执行算子 |
| `models/common/runner.hpp` | 选择执行方案并建图，准备运行时输入，容量检查、执行与提交 | attention 数学、请求抢占策略 |
| `models/<name>/runner-*.hpp` | 加载权重、建层 cache、声明模型图；特殊设备步骤（如 GDN 槽清零） | 重复实现公共 step 流程 |

`AttentionType` 是并列的执行方案（Eager、DenseFIA、TND、BSND、PagedBSND），由 runner
的执行记录传入模型。分派只在本目录发生一次：`full_attention(type, ...)` 和
`mla_attention(type, ...)`（`attention.hpp`），模型不再各自 switch。
公共函数只读取 `cache::AttentionInputs`（写入行、mask、长度、块表、块大小），不接触
SlotMap、页面或 cache 对象。mask/length/块表的**数值生成**在 `cache/attention-metadata.hpp`，
将它们接到算子哪一个端口则在本目录。

## 各模型

| 模型路径 | 调用公共 attention 前 | 公共计算及返回值 | 调用后 |
|---|---|---|---|
| Qwen3 / Qwen3 MoE | QKV 投影、Q/K RMSNorm、RoPE、cache 更新 | full attention；`[T,H,D]` | 合并 heads、`o_proj` |
| Llama | QKV 投影、RoPE、cache 更新 | full attention；`[T,H,D]` | 合并 heads、`o_proj` |
| Qwen3.5 / Qwen3.8 full（含 MoE） | QKV/gate 投影、Q/K norm、partial RoPE、cache 更新 | full attention；`[T,H,D]` | sigmoid gate、`o_proj` |
| DeepSeek V2 absorbed MLA | 低秩 Q/KV 投影、norm、解耦 RoPE、latent/k_pe cache 更新、`Q_nope × W_uk` | MLA；`[T,H,latent]` | `weighted_latent × W_uv`、`o_proj` |
| Qwen3.5 / Qwen3.8 GDN（含 MoE） | 投影、卷积、Q/K L2 norm、带参数的 beta/decay | 递推；attention 输出与新 state | state/conv 写回、RMSNormGated、`out_proj` |

full 与 MLA 入口接收 token-major 三维张量。`T` 是真实执行图的 token 行数，
BSND 路径的 `T=B*Q`，MLA decode 要求 `Q=1`。BSND 四维 query 和物理页 reshape
全部由公共入口处理，模型只处理自己的 head/latent/hidden 维度。

DeepSeek 的 `forward_absorbed_mla` 是完整的模型计算路径，不是另一层 FIA wrapper：

```text
hidden_states
  -> 模型投影 / norm / RoPE / cache.update
  -> 模型 absorb_query（W_uk）
  -> mla_attention(type) -> flash_attention_mla_{tnd,bsnd}
  -> 模型 unabsorb_output（W_uv）/ o_proj
```

`MLACache` 不接收 `kv_b_proj` 权重或 query/value head 维度，权重知识只在模型中。

## Eager 参考

Qwen3/Llama/Qwen3.5 共用 `eager-attention.hpp` 的 GQA 参考实现（分数用模型 dtype，
softmax 用 FP32，共享相同的 GQA 计算流程，保留模型 dtype 与调试命名差异）。DeepSeek 的展开式 eager MLA 物化 K/V 的方式
依赖模型权重，仍留在模型中，但同样调用公共 `eager_attention` 计算分数。
通用、无参数的 GDN eager 递推与 fused GDN 一起放在本目录。

## 文件

| 文件 | 内容 |
|---|---|
| `attention.hpp` | 分派入口 `full_attention` / `mla_attention` |
| `attention-options.hpp` | CANN 属性的命名 enum、输入校验、BSND/分页视图 |
| `flash-attention-full.hpp` | DenseFIA、TND、BSND（`paged` 参数区分连续与分页存储） |
| `flash-attention-mla.hpp` | MLA TND、BSND（dense 统一 scan 长度 workaround 只在此处） |
| `eager-attention.hpp` | GQA eager 参考 |
| `eager-attention-gdn-decode.hpp` / `flash-attention-gdn-bsnd.hpp` | GDN FP32 eager 递推 / 原生融合递推 |

各方案的输入与布局约定见 [KV cache 与 attention](../../../../docs/kv-cache.md)。
