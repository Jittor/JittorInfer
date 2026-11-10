#pragma once

// Qwen3.5 纯文本计算图（decode 路径，plan 见 docs/qwen3.5-decode-plan.md）。
//
// 结构（layer_types 按 3:1 交错，full_attention_interval=4）：
//   linear_attention 层 → Qwen35GatedDeltaNet（GDN decode：conv 单步更新 +
//     RecurrentGatedDeltaRule 融合核 + RMSNormGated）；
//   full_attention 层   → Qwen35Attention（GQA + q/k 零中心 RMSNorm +
//     输出门控 sigmoid(gate) + partial RoPE）。
//
// 与 qwen3 的关键差异：
//   - partial RoPE（rotary_dim = head_dim/4）：kernel 不支持窄表
//     （探针 tests/rope_partial.cpp case3 REJECTED），按 qwen3 双路径
//     结构作用于 head 末轴的前 rotary_dim 维 slice，其余维 cat 透传；
//   - q_proj 输出 2×(heads*head_dim)，每 head [q | gate]，注意力输出乘
//     sigmoid(gate) 后再过 o_proj；
//   - RMSNorm 权重零中心（y = norm(x) * (1 + w)），图内 +1，不动加载器；
//   - kernel 无内置 q/k l2norm（forch R1 测试参考不含 l2norm），模型图
//     内 fp32 预归一后再喂 RecurrentGatedDeltaRule。
//
// 两处 forch 布局陷阱（都踩过，症状都是「静默出错」而不是报错）：
//   - partial RoPE 的 eager 分支（fuse_rope=false）必须把 cos/sin reshape 成
//     [T,1,R] 再广播；直接传 [T,R] 会被 forch 的 broadcast_shape 漏检（它不比较
//     低秩输入的最左维），rows==H 时静默错值、其余 rows 到 GE 编译期才报
//     E80013。
//   - Full attention 的投影、RoPE 和 KV cache 均采用 [T,N,D]。
//     FIA 直接消费 token-major 数据；仅 eager attention 边界转为 [N,T,D]。
//     两种布局不能直接用 reshape 相互替代。
//
// 状态写回的图连接：forch 从图输出反向遍历建图，悬空节点不进图。因此
// GDN 层的 state/conv scatter 写回节点由 GDNCache 暂存，runner 在
// 建图后经 append_write_back 注册为具名图输出，执行期的原地写入才会
// 发生；调用方无须消费这些输出，仅需注册。
//
// 权重命名对齐 HF checkpoint（model.language_model.*）；vision/mtp 键
// 不注册即跳过。各 GDNCache 的 name 必须逐层唯一（占位符按名匹配）。

#include "../../cache/full-attention.hpp"
#include "../../cache/gdn.hpp"
#include "../../engine/config.hpp"
#include "../common/attention/attention.hpp"
#include "../common/attention/eager-attention-gdn-decode.hpp"
#include "../common/attention/flash-attention-gdn-bsnd.hpp"
#include "../common/runner.hpp"

#include "../common/lm-head.hpp"
#include "../common/mlp.hpp"
#include "../common/moe.hpp"
#include "../common/norm.hpp"
#include "../common/rope.hpp"
#include "config-qwen3.5.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <forch/forch.hpp>
#include <forch/nn/embedding.hpp>
#include <forch/nn/linear.hpp>
#include <forch/nn/module.hpp>
#include <forch/nn/parallel_linear.hpp>
#include <forch/nn/parameter.hpp>
#include <forch/ops/batched_mat_mul.hpp>
#include <forch/ops/cast.hpp>
#include <forch/ops/cat.hpp>
#include <forch/ops/const.hpp>
#include <forch/ops/exp.hpp>
#include <forch/ops/mat_mul.hpp>
#include <forch/ops/permute.hpp>
#include <forch/ops/repeat.hpp>
#include <forch/ops/reshape.hpp>
#include <forch/ops/rms_norm.hpp>
#include <forch/ops/rotary_position_embedding.hpp>
#include <forch/ops/rsqrt.hpp>
#include <forch/ops/sigmoid.hpp>
#include <forch/ops/slice.hpp>
#include <forch/ops/softmax.hpp>
#include <forch/ops/softplus.hpp>

namespace finer::models::qwen35 {

using cache::FullAttentionCache;
using cache::GDNCache;
struct Qwen35CacheView {
  cache::FullAttentionCache *kv  = nullptr;
  cache::GDNCache           *gdn = nullptr;
};

// ── 零中心 RMSNorm：y = rms_norm(x) * (1 + w) ────────────────────────────
// Qwen3_5 的 RMSNorm 权重零中心初始化，checkpoint 存 w 本体；在图内 +1，
// 不改加载器。q/k_norm 与 input/post_attention_layernorm 均是这种。
struct RMSNormZeroCentered : public forch::nn::Module {
  forch::nn::Parameter *weight;
  float                 eps;

  RMSNormZeroCentered(int64_t dim, float eps) : eps(eps) {
    weight = register_loadable(
      "weight", forch::nn::Parameter({dim}, forch::DType::BF16));
  }

  forch::Tensor forward(const forch::Tensor &input) const {
    auto dim  = weight->get().shape()[0];
    auto ones = forch::op::load_from_vector(
                  std::vector<float>(static_cast<size_t>(dim), 1.0F), {dim})
      | forch::op::cast(forch::DType::BF16);
    return forch::op::rms_norm(eps)(input, forch::op::add(weight->get(), ones));
  }
};

// ── GDN 输出归一：y = rms_norm(x) * w * silu(gate)（先 norm 后门控）──────
// 权重为普通权重（非零中心），顺序与常见 gated-norm 相反，照抄 HF
// Qwen3_5RMSNormGated。norm.weight 的存储 dtype 随检查点不同（0.8B /
// Qwen3.5-27B 存 F32，Qwen3.8-27B 系存 BF16），由 gdn_param_dtype 显式指定
// 并在注册时对齐（零拷贝）；归一要求 BF16，已是 BF16 直用，F32 则图内 cast。
struct RMSNormGated : public forch::nn::Module {
  forch::nn::Parameter *weight;
  float                 eps;

  RMSNormGated(
    int64_t dim, float eps, forch::DType weight_dtype = forch::DType::FP32)
      : eps(eps) {
    weight =
      register_loadable("weight", forch::nn::Parameter({dim}, weight_dtype));
  }

  forch::Tensor forward(
    const forch::Tensor &x, const forch::Tensor &gate) const {
    auto w  = weight->get();
    auto wb = w.dtype() == forch::DType::BF16
      ? w
      : w | forch::op::cast(forch::DType::BF16);
    auto n  = forch::op::rms_norm(eps)(x, wb);
    return forch::op::mul(n, forch::op::silu(gate));
  }
};

// ── partial RoPE 表：inv_freq[rotary_dim/2]，cos/sin [T, rotary_dim] ────
// 纯文本下 mrope 三行位置相同，交错合成退化为标准 partial rope，无需 3D。
struct Qwen35RotaryEmbedding : public forch::nn::Module {
  forch::Tensor inv_freq;
  int64_t       rotary_dim;

  static forch::Tensor compute_inv_freq(float theta, int64_t rotary_dim) {
    auto               half = static_cast<size_t>(rotary_dim / 2);
    std::vector<float> freqs(half);
    for (size_t i = 0; i < half; i++) {
      auto exp = static_cast<float>(i * 2) / static_cast<float>(rotary_dim);
      freqs[i] = 1.0F / std::pow(theta, exp);
    }
    return forch::op::load_from_vector(
      std::move(freqs), {1, static_cast<int64_t>(half)});
  }

  Qwen35RotaryEmbedding(const Qwen35Config &config)
      : inv_freq(compute_inv_freq(config.rope_theta, config.rotary_dim)),
        rotary_dim(config.rotary_dim) {}

  // position_ids: [T] → cos/sin: [T, rotary_dim] BF16（配对两半共用同一值）
  std::pair<forch::Tensor, forch::Tensor> forward(
    const forch::Tensor &position_ids) const {
    auto p = position_ids | forch::op::cast(forch::DType::FP32)
      | forch::op::reshape({position_ids.shape()[0], 1});
    auto emb = forch::op::mat_mul()(p, inv_freq) | forch::op::repeat({1, 2});
    return {emb | forch::op::cos | forch::op::cast(forch::DType::BF16),
      emb | forch::op::sin | forch::op::cast(forch::DType::BF16)};
  }
};

// ── full-attention 层：GQA + 输出门控 + partial RoPE ─────────────────────
// Model layer owns QKV/gate projections, Q/K norm, partial RoPE and cache
// writes. common/attention returns unprojected attention; sigmoid gating and
// o_proj remain here because they are part of the checkpoint's model
// computation.
struct Qwen35Attention : public forch::nn::Module {
  int64_t head_dim;
  int64_t num_heads;
  int64_t num_key_value_heads;
  int64_t num_key_value_groups;
  int64_t rotary_dim;
  float   scaling;
  bool    attn_output_gate;
  bool    fuse_rope;

  forch::nn::QKVParallelLinear  qkv_proj;
  forch::nn::RowParallelLinear *o_proj;
  RMSNormZeroCentered          *q_norm;
  RMSNormZeroCentered          *k_norm;

  Qwen35Attention(
    const Qwen35Config &config, const engine::EngineConfig &engine_config)
      : head_dim(config.head_dim), rotary_dim(config.rotary_dim),
        scaling(1.0F / std::sqrt(static_cast<float>(config.head_dim))),
        attn_output_gate(config.attn_output_gate),
        qkv_proj(config.hidden_size, config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, engine_config.tp,
          config.attention_bias, forch::DType::BF16,
          config.attn_output_gate ? 2 : 1) {
    num_heads            = qkv_proj.local_query_heads();
    num_key_value_heads  = qkv_proj.local_key_value_heads();
    num_key_value_groups = num_heads / num_key_value_heads;
    o_proj               = register_loadable("o_proj",
                    forch::nn::RowParallelLinear(config.num_attention_heads * config.head_dim,
                      config.hidden_size, engine_config.tp, config.attention_bias,
                      forch::DType::BF16));
    q_norm               = register_loadable(
      "q_norm", RMSNormZeroCentered(config.head_dim, config.rms_norm_eps));
    k_norm = register_loadable(
      "k_norm", RMSNormZeroCentered(config.head_dim, config.rms_norm_eps));

    fuse_rope = engine_config.optimization.fuse_rope;
  }

  // TODO: see Qwen3Attention::load.
  void load(
    forch::nn::Loader *loader, const std::string &prefix = "") override {
    Module::load(loader, prefix);
    qkv_proj.load(loader, prefix);
  }

  // partial RoPE：head 末轴前 rotary_dim 维旋转，其余透传（qwen3 双路径结构）
  forch::Tensor rope_partial(const forch::Tensor &qk, const forch::Tensor &cos,
    const forch::Tensor &sin, int64_t table_rows) const {
    auto qk_rot = qk
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::IntervalFromBegin{rotary_dim}});
    auto qk_pas = qk
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::IntervalTillEnd{rotary_dim}});

    // eager 分支与 fused 分支用同一种广播形状 [rows,1,R]。
    // 若直接传 [rows,R]：forch 的 broadcast_shape 有 off-by-one（比到最左维时
    // `shape().size() > i`
    // 为假，低秩输入的最左维不参与比较），所以它**不会报错**：
    //   rows == 1  → GE 广播成 [1,1,R]，恰好正确（所以长期没暴露）；
    //   rows == H  → GE 把 [rows,R] 当 [1,rows,R]，cos 第 r 行乘到第 r 个
    //   head，
    //                **静默错值**（实测 rows=8=H 时与正确值差 1.25）；
    //   其余 rows  → 到 GE 编译期才报
    //                E80013 "could not be broadcast together"。
    auto cos3 = cos | forch::op::reshape({table_rows, 1, rotary_dim});
    auto sin3 = sin | forch::op::reshape({table_rows, 1, rotary_dim});

    forch::Tensor qk_roped = fuse_rope ? [&] {
      return forch::op::rotary_position_embedding(
        forch::op::RotaryPositionEmbedding::Mode::Half)(qk_rot, cos3, sin3);
    }()
                                       : apply_rope_half(qk_rot, cos3, sin3);
    return forch::op::cat(2)(qk_roped, qk_pas);
  }

  forch::Tensor forward(const forch::Tensor &hidden_states,
    std::pair<forch::Tensor, forch::Tensor>  position_embeddings,
    FullAttentionCache &kv, const ForwardInputs &in) {
    auto rows = hidden_states.shape()[0];

    auto [cos, sin] = std::move(position_embeddings);

    // q_proj 每 head 输出 [q | gate]（HF: view(B,S,Nh,2*Dd) + chunk）
    auto [q_raw, k_raw, v_raw] = qkv_proj.forward(hidden_states);
    auto qg                    = q_raw
      | forch::op::reshape(
        {rows, num_heads, (attn_output_gate ? 2 : 1) * head_dim});
    auto q = qg
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::IntervalFromBegin{head_dim}});
    auto gate = attn_output_gate
      ? qg
        | forch::op::slice({forch::op::Slice::Ellipsis{},
          forch::op::Slice::IntervalTillEnd{head_dim}})
        | forch::op::reshape({rows, num_heads * head_dim})
      : forch::op::load_from_vector(
          std::vector<float>(
            static_cast<size_t>(rows * num_heads * head_dim), 1.0F),
          {rows, num_heads * head_dim})
        | forch::op::cast(forch::DType::BF16);

    auto query_states = q_norm->forward(q);
    auto key_states   = k_norm->forward(
      k_raw | forch::op::reshape({rows, num_key_value_heads, head_dim}));
    auto value_states =
      v_raw | forch::op::reshape({rows, num_key_value_heads, head_dim});

    // partial RoPE（q/k 各切一次）
    query_states = rope_partial(query_states, cos, sin, rows);
    key_states   = rope_partial(key_states, cos, sin, rows);

    std::tie(key_states, value_states) =
      kv.update(key_states, value_states, *in.attention);
    auto attn_output = full_attention(in.attention_type, *in.attention,
      query_states, key_states, value_states, scaling);

    // 输出门控：attn_out * sigmoid(gate)，再 o_proj
    attn_output =
      attn_output | forch::op::reshape({rows, num_heads * head_dim});
    attn_output = forch::op::mul(attn_output, forch::op::sigmoid()(gate));
    return attn_output | *o_proj;
  }
};

// ── Gated DeltaNet 层（decode 单步）───────────────────────────────────────
// per-token 数据流：
//   in_proj_qkv → causal conv1d 单步更新(4-tap) + silu → 切 q/k/v
//   in_proj_b/a → β = sigmoid(b)；g = −exp(A_log)·softplus(a + dt_bias)
//   RecurrentGatedDeltaRule（kernel 隐式 v/k head repeat；q/k 需先 l2norm）
//   → RMSNormGated(out, z) → out_proj
//
// 状态写回：scatter 节点由 GDNCache 暂存、runner 注册为图输出才会
// 被执行（见文件头说明）。全池覆盖 —— kernel 的 ref_state 未触块原样
// 拷贝，conv 全 batch 计算，FREE 槽的垃圾写在自己的块里，槽重用时
// reset_slot 清零。
// Model layer owns projections, convolution, Q/K L2 norm, learned beta/decay,
// state write-back and output gating/projection. common/attention owns only
// the recurrence, padding validity and kernel-specific state layout adaptation.
struct Qwen35GatedDeltaNet : public forch::nn::Module {
  int64_t num_v_heads;
  int64_t num_k_heads;
  int64_t head_k_dim;
  int64_t head_v_dim;
  int64_t key_dim;
  int64_t value_dim;
  int64_t conv_channels;
  int64_t conv_kernel;
  float   eps;

  forch::nn::Linear               *in_proj_qkv;   // [H, 2*key_dim + value_dim]
  forch::nn::ColumnParallelLinear *in_proj_z;     // [H, value_dim]
  forch::nn::ColumnParallelLinear *in_proj_b;     // [H, Nv]
  forch::nn::ColumnParallelLinear *in_proj_a;     // [H, Nv]
  forch::nn::RowParallelLinear    *out_proj;      // [value_dim, H]
  forch::nn::Parameter            *conv1d_weight; // [C, 1, K]，depthwise
  forch::nn::Parameter *A_log; // [Nv] 存储 dtype 随检查点（gdn_param_dtype）
  forch::nn::Parameter *dt_bias; // [Nv]
  RMSNormGated         *norm;    // dim = Dv

  forch::ProcessGroup group;

  // TP splits K and V heads; each rank keeps whole heads of every packed
  // q/k/v segment, so the local recurrence needs no communication.
  Qwen35GatedDeltaNet(
    const Qwen35Config &config, const engine::EngineConfig &engine_config)
      : num_v_heads(
          config.linear_num_value_heads / engine_config.tp.world_size()),
        num_k_heads(
          config.linear_num_key_heads / engine_config.tp.world_size()),
        head_k_dim(config.linear_key_head_dim),
        head_v_dim(config.linear_value_head_dim),
        key_dim(num_k_heads * head_k_dim), value_dim(num_v_heads * head_v_dim),
        conv_channels(2 * key_dim + value_dim),
        conv_kernel(config.linear_conv_kernel_dim), eps(config.rms_norm_eps),
        group(engine_config.tp) {
    if (config.linear_num_key_heads % group.world_size()
      || config.linear_num_value_heads % config.linear_num_key_heads)
      throw std::invalid_argument("Qwen3.5: GDN heads incompatible with TP");
    in_proj_qkv = register_loadable("in_proj_qkv",
      forch::nn::Linear(
        config.hidden_size, conv_channels, false, forch::DType::BF16));
    in_proj_z   = register_loadable("in_proj_z",
        forch::nn::ColumnParallelLinear(config.hidden_size, config.value_dim,
          group, false, forch::DType::BF16));
    in_proj_b   = register_loadable("in_proj_b",
        forch::nn::ColumnParallelLinear(config.hidden_size,
          config.linear_num_value_heads, group, false, forch::DType::BF16));
    in_proj_a   = register_loadable("in_proj_a",
        forch::nn::ColumnParallelLinear(config.hidden_size,
          config.linear_num_value_heads, group, false, forch::DType::BF16));
    out_proj    = register_loadable("out_proj",
         forch::nn::RowParallelLinear(config.value_dim, config.hidden_size, group,
           false, forch::DType::BF16));
    conv1d_weight =
      register_loadable("conv1d.weight", // checkpoint 键带 .weight 后缀
        forch::nn::Parameter(
          {conv_channels, 1, conv_kernel}, forch::DType::BF16));
    A_log = register_loadable(
      "A_log", forch::nn::Parameter({num_v_heads}, config.gdn_param_dtype));
    dt_bias = register_loadable(
      "dt_bias", forch::nn::Parameter({num_v_heads}, forch::DType::BF16));
    norm = register_loadable(
      "norm", RMSNormGated(head_v_dim, eps, config.gdn_param_dtype));
  }

  // Packed q|k|v rows (in_proj_qkv, conv1d) take this rank's slice of each
  // segment; per-head vectors take this rank's heads. ColumnParallel children
  // shard themselves.
  void load(
    forch::nn::Loader *loader, const std::string &prefix = "") override {
    struct Shards final : forch::nn::Loader {
      forch::nn::Loader         *source;
      const Qwen35GatedDeltaNet &layer;
      std::string                prefix;
      Shards(forch::nn::Loader *source, const Qwen35GatedDeltaNet &layer,
        std::string prefix)
          : source(source), layer(layer),
            prefix(prefix.empty() ? "" : prefix + ".") {}
      forch::Tensor load(const std::string &name) override {
        using forch::nn::WeightShard;
        auto       tensor = source->load(name);
        const auto key    = name.substr(prefix.size());
        const auto size   = layer.group.world_size(),
                   rank   = layer.group.rank_id();
        if (key == "A_log" || key == "dt_bias")
          return WeightShard::shard(
            tensor, 0, rank * layer.num_v_heads, layer.num_v_heads);
        if (key != "in_proj_qkv.weight" && key != "conv1d.weight")
          return tensor;
        std::vector<uint8_t> bytes;
        int64_t              start = 0;
        for (auto count : {layer.key_dim, layer.key_dim, layer.value_dim}) {
          auto part =
            WeightShard::shard(tensor, 0, start + rank * count, count);
          const auto &segment =
            static_cast<const WeightShard::ShardConstant *>(part.get_op())
              ->data;
          bytes.insert(bytes.end(), segment.begin(), segment.end());
          start += count * size;
        }
        auto shape = tensor.shape();
        shape[0]   = layer.conv_channels;
        return forch::Tensor({},
          std::make_unique<WeightShard::ShardConstant>(
            std::move(bytes), shape, tensor.dtype()));
      }
    } shards(loader, *this, prefix);
    if (group.world_size() == 1)
      forch::nn::Module::load(loader, prefix);
    else
      forch::nn::Module::load(&shards, prefix);
  }

  // HF l2norm：x * rsqrt(sum(x², -1) + eps)；求和用 [D,1] 全 1 矩乘表达
  // （forch 无 reduce 算子）
  static forch::Tensor l2norm(const forch::Tensor &x, int64_t dim) {
    auto xf   = x | forch::op::cast(forch::DType::FP32);
    auto sq   = forch::op::mul(xf, xf);
    auto ones = forch::op::load_from_vector(
      std::vector<float>(static_cast<size_t>(dim), 1.0F), {dim, 1});
    auto ssq   = forch::op::batched_mat_mul()(sq, ones); // [..., 1]
    auto eps_t = forch::op::load_from_vector(std::vector<float>{1e-6F}, {1});
    auto inv   = forch::op::rsqrt()(forch::op::add(ssq, eps_t));
    return forch::op::mul(xf, inv) | forch::op::cast(forch::DType::BF16);
  }

  forch::Tensor forward(const forch::Tensor &hidden_states, GDNCache &cache,
    const cache::GDNInputs &in, bool fused) {
    auto rows = hidden_states.shape()[0];

    auto qkv_mixed = in_proj_qkv->forward(hidden_states); // [B, C]
    auto z         = in_proj_z->forward(hidden_states)
      | forch::op::reshape({rows, num_v_heads, head_v_dim});
    auto b = in_proj_b->forward(hidden_states); // [B, Nv]
    auto a = in_proj_a->forward(hidden_states); // [B, Nv]

    // Cache owns causal windows and the final per-slot convolution history.
    auto [new_win, conv_next] = cache.convolution_windows(qkv_mixed, in);

    // 深度卷积权重 [C,1,K] → [K,C]，逐 tap 切片广播
    auto w = conv1d_weight->get() | forch::op::permute({2, 0, 1})
      | forch::op::reshape({conv_kernel, conv_channels});
    std::optional<forch::Tensor> acc;
    for (int64_t k = 0; k < conv_kernel; ++k) {
      auto tap = new_win
        | forch::op::slice({forch::op::Slice::IntervalAll{},
          forch::op::Slice::Interval{k, k + 1},
          forch::op::Slice::IntervalAll{}})
        | forch::op::reshape({rows, conv_channels});
      // w 是 [K, C]（行 = tap，列 = channel），第 k 个 tap 的权重是**行** k。
      // 必须切 dim0：切 dim1 得到的是 [K,1]（channel k 在各 tap 上的权重），
      // term 会广播成 [K,C]，而后续 reshape 不校验元素数、会静默取错数据，
      // 导致 q/k/v 错位且幅度缩小千倍、R1 输出≈0、GDN 层退化为恒等层。
      auto wk = w
        | forch::op::slice({forch::op::Slice::Interval{k, k + 1},
          forch::op::Slice::IntervalAll{}}); // [1, C]
      auto term = forch::op::mul(tap, wk);
      acc       = acc.has_value() ? forch::op::add(acc.value(), term) : term;
    }
    auto mixed =
      acc.value() | forch::op::silu; // [B, C]（silu 是 curry 对象，无括号）

    // ── q/k/v 切分 + l2norm ──
    auto q = mixed
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::IntervalFromBegin{key_dim}})
      | forch::op::reshape({rows, num_k_heads, head_k_dim});
    auto k = mixed
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::Interval{key_dim, key_dim * 2}})
      | forch::op::reshape({rows, num_k_heads, head_k_dim});
    auto v = mixed
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::IntervalTillEnd{key_dim * 2}})
      | forch::op::reshape({rows, num_v_heads, head_v_dim});

    q = l2norm(q, head_k_dim);
    k = l2norm(k, head_k_dim);
    // scale_value 已传 R1 attr（kernel 内部缩放），此处不再手动乘

    // ── β 与衰减 g（fp32）───────────────────────────────────────────────
    auto beta = forch::op::sigmoid()(b); // [B, Nv] bf16
    // A_log 存储 dtype 随检查点（gdn_param_dtype）：FP32（0.8B/Qwen3.5-27B）
    // 直用、与旧图逐位一致；BF16（Qwen3.8-27B 系）图内 cast 一次（常量）。
    // exp 的结果必须 FP32 —— R1 的 g 契约。
    auto a_log = A_log->get();
    auto g = forch::op::mul(forch::op::exp()(a_log.dtype() == forch::DType::FP32
                                ? a_log
                                : a_log | forch::op::cast(forch::DType::FP32))
        | forch::op::neg,
      forch::op::softplus()(
        forch::op::add(a | forch::op::cast(forch::DType::FP32),
          dt_bias->get() | forch::op::cast(forch::DType::FP32))));

    auto [gdn_out, state] = fused
      ? flash_attention_gdn_bsnd(
          q, k, v, beta, g, cache.state(), in.lengths, in.indices, in.valid)
      : eager_attention_gdn_decode(q, k, v, beta, g, cache.state(), in.valid);
    cache.update(conv_next, state);

    // ── RMSNormGated + out_proj ──────────────────────────────────────────
    auto o     = gdn_out | forch::op::reshape({rows * num_v_heads, head_v_dim});
    auto z2    = z | forch::op::reshape({rows * num_v_heads, head_v_dim});
    auto gated = norm->forward(o, z2);
    return out_proj->forward(gated | forch::op::reshape({rows, value_dim}));
  }
};

// ── DecoderLayer：按 layer_types 分发 token mixer ────────────────────────
struct Qwen35DecoderLayer : public forch::nn::Module {
  bool                 is_full_attention;
  Qwen35Attention     *self_attn;   // 仅 full 层
  Qwen35GatedDeltaNet *linear_attn; // 仅 linear 层
  MLP                 *mlp;
  RMSNormZeroCentered *input_layernorm;
  RMSNormZeroCentered *post_attention_layernorm;

  Qwen35DecoderLayer(const Qwen35Config &config,
    const engine::EngineConfig &engine_config, int64_t idx)
      : is_full_attention(
          config.layer_is_full_attention.at(static_cast<size_t>(idx))) {
    if (is_full_attention) {
      self_attn =
        register_loadable("self_attn", Qwen35Attention(config, engine_config));
    } else {
      linear_attn = register_loadable(
        "linear_attn", Qwen35GatedDeltaNet(config, engine_config));
    }
    mlp                      = register_loadable("mlp",
                           MLP(config.hidden_size, config.intermediate_size, engine_config.tp,
                             forch::DType::BF16));
    input_layernorm          = register_loadable("input_layernorm",
               RMSNormZeroCentered(config.hidden_size, config.rms_norm_eps));
    post_attention_layernorm = register_loadable("post_attention_layernorm",
      RMSNormZeroCentered(config.hidden_size, config.rms_norm_eps));
  }

  forch::Tensor forward(forch::Tensor hidden, const Qwen35CacheView &caches,
    std::pair<forch::Tensor, forch::Tensor> position_embeddings,
    const ForwardInputs &in, bool fused_gdn) const {
    auto attn_residual = hidden;
    auto h             = input_layernorm->forward(hidden);
    if (is_full_attention)
      h = self_attn->forward(h, std::move(position_embeddings), *caches.kv, in);
    else
      h = linear_attn->forward(h, *caches.gdn, *in.gdn, fused_gdn);
    h = forch::op::add(h, attn_residual);

    auto mlp_residual = h;
    h                 = post_attention_layernorm->forward(h);
    h                 = mlp->forward(h);
    return forch::op::add(h, mlp_residual);
  }
};

// ── 文本塔（HF 权重前缀 model.language_model.*）─────────────────────────
struct Qwen35TextModel : public forch::nn::Module {
  forch::nn::Embedding                      *embed_tokens;
  forch::nn::ModuleList<Qwen35DecoderLayer> *layers;
  RMSNormZeroCentered                       *norm;
  Qwen35RotaryEmbedding                     *rotary_emb;

  Qwen35TextModel(
    const Qwen35Config &config, const engine::EngineConfig &engine_config) {
    embed_tokens = register_loadable("embed_tokens",
      forch::nn::Embedding(
        config.vocab_size, config.hidden_size, forch::DType::BF16));

    auto num_layers    = engine_config.n_model_layers > 0
         ? std::min(engine_config.n_model_layers, config.num_hidden_layers)
         : config.num_hidden_layers;
    auto layers_vector = std::vector<std::unique_ptr<Qwen35DecoderLayer>>();
    layers_vector.reserve(static_cast<size_t>(num_layers));
    for (int64_t i = 0; i < num_layers; i++) {
      layers_vector.push_back(
        std::make_unique<Qwen35DecoderLayer>(config, engine_config, i));
    }
    layers = register_loadable("layers",
      forch::nn::ModuleList<Qwen35DecoderLayer>(std::move(layers_vector)));

    norm = register_loadable(
      "norm", RMSNormZeroCentered(config.hidden_size, config.rms_norm_eps));
    rotary_emb = register_loadable("rotary_emb", Qwen35RotaryEmbedding(config));
  }

  forch::Tensor forward(const ForwardInputs &in,
    const std::vector<Qwen35CacheView> &caches, bool fused_gdn) const {
    if (caches.size() < layers->size())
      throw forch::exception("qwen3.5: missing layer caches");
    auto hidden_states       = embed_tokens->forward(in.input_ids);
    auto position_embeddings = rotary_emb->forward(in.position_ids);
    for (size_t i = 0; i < layers->size(); ++i)
      hidden_states = (*layers)[i]->forward(
        hidden_states, caches[i], position_embeddings, in, fused_gdn);

    return norm->forward(hidden_states);
  }
};

// ── 多模态壳层（权重前缀 model.*，vision/mtp 键不注册即跳过）────────────
struct Qwen35Model : public forch::nn::Module {
  Qwen35TextModel *language_model;

  Qwen35Model(
    const Qwen35Config &config, const engine::EngineConfig &engine_config) {
    language_model = register_loadable(
      "language_model", Qwen35TextModel(config, engine_config));
  }

  forch::Tensor forward(const ForwardInputs &in,
    const std::vector<Qwen35CacheView> &caches, bool fused_gdn) const {
    return language_model->forward(in, caches, fused_gdn);
  }
};

struct Qwen35ForCausalLM : public forch::nn::Module {
  Qwen35Model         *model;
  VocabParallelLMHead *lm_head;

  Qwen35ForCausalLM(
    const Qwen35Config &config, const engine::EngineConfig &engine_config) {
    model   = register_loadable("model", Qwen35Model(config, engine_config));
    lm_head = register_loadable(config.tie_word_embeddings
        ? "model.language_model.embed_tokens"
        : "lm_head",
      VocabParallelLMHead(config.hidden_size, config.vocab_size,
        engine_config.logits_topk, engine_config.tp, forch::DType::BF16));
  }

  forch::op::TopKOutputs forward(const ForwardInputs &in,
    const std::vector<Qwen35CacheView> &caches, bool fused_gdn) const {
    return lm_head->forward(model->forward(in, caches, fused_gdn));
  }
};

} // namespace finer::models::qwen35
