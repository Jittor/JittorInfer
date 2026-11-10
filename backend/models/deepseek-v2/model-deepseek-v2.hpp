#pragma once

/// @file model-deepseek-v2.hpp
/// @brief DeepSeek-V2 (MLA + DeepSeekMoE) translated from the HF reference
/// `backend/models/deepseek-v2/modeling_deepseek.py` (backed up from
/// deepseek-ai/DeepSeek-V2-Lite, which is the same graph as DeepSeek-V2 with
/// q_lora_rank=null and YaRN rope).
///
/// Porting notes:
/// - Eager caches expanded K/V; the absorbed path caches normalized latent
///   and roped k_pe. Weight absorption is model math, not cache addressing
///   or a CANN layout workaround.
/// - RoPE is the *interleaved* pair convention of the reference
///   (`view(d//2, 2).transpose(-2, -1)`), NOT qwen3's half-split.
/// - Routing and shared experts are model-local; routed expert execution
///   and TP reductions use the common moe() graph helper.
/// - Still TODO: dynamic NTK rope, scoring_func != softmax.

#include "../../cache/full-attention.hpp"
#include "../../cache/mla.hpp"
#include "../../engine/config.hpp"
#include "../common/attention/attention.hpp"
#include "../common/runner.hpp"

#include "../common/lm-head.hpp"
#include "../common/mlp.hpp"
#include "../common/moe.hpp"
#include "../common/norm.hpp"
#include "../common/rope.hpp"
#include "config-deepseek-v2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <forch/exception.hpp>
#include <forch/forch.hpp>
#include <forch/nn/embedding.hpp>
#include <forch/nn/linear.hpp>
#include <forch/nn/module.hpp>
#include <forch/nn/parallel_linear.hpp>
#include <forch/ops/batched_mat_mul.hpp>
#include <forch/ops/cast.hpp>
#include <forch/ops/cat.hpp>
#include <forch/ops/const.hpp>
#include <forch/ops/div.hpp>
#include <forch/ops/equal.hpp>
#include <forch/ops/function.hpp>
#include <forch/ops/ge/moe_finalize_routing.hpp>
#include <forch/ops/ge/moe_gating_top_k_softmax.hpp>
#include <forch/ops/ge/moe_init_routing.hpp>
#include <forch/ops/grouped_mat_mul.hpp>
#include <forch/ops/mat_mul.hpp>
#include <forch/ops/one_hot.hpp>
#include <forch/ops/permute.hpp>
#include <forch/ops/reduce.hpp>
#include <forch/ops/repeat.hpp>
#include <forch/ops/reshape.hpp>
#include <forch/ops/rotary_position_embedding.hpp>
#include <forch/ops/scatter_elements.hpp>
#include <forch/ops/select.hpp>
#include <forch/ops/slice.hpp>
#include <forch/ops/softmax.hpp>
#include <forch/ops/topk.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace finer::models::deepseek_v2 {

using cache::FullAttentionCache;
struct DeepseekV2CacheView {
  cache::FullAttentionCache *kv  = nullptr;
  cache::MLACache           *mla = nullptr;
};
using cache::MLACache;
using finer::models::RMSNorm;

// TP keeps every expert on every rank, sliced by intermediate channel; the
// router and low-rank attention latents stay replicated. Routed experts feed
// their sharded matrices straight to grouped matmul and reduce once after
// combining, not once per expert.
using finer::models::MLP;

/**
 * RoPE for the decoupled query/key head (qk_rope_head_dim).
 * Mirrors DeepseekV2RotaryEmbedding / DeepseekV2YarnRotaryEmbedding: cos/sin
 * are computed from position_ids in-graph, with the (possibly yarn-modified)
 * inv_freq table and the yarn mscale folded in as host-side constants.
 * The graph never sees `seq_len` — the reference's cos_cached indexing
 * `cos[position_ids]` and computing `cos(position_ids * inv_freq)` are the
 * same value.
 */
/// CANN RotaryPositionEmbedding(Mode::Interleave) 要求 cos/sin 为相邻重复
/// 布局 [f0,f0,f1,f1,...]; 参考实现是 cat(freqs,freqs) 的 [f0..f31,f0..f31]。
/// 取前半(freqs)后做 [...,32,1] --tile 2--> [...,32,2] --reshape--> [...,64]
/// 完成相邻重复(秩通用; GE 的 repeat 不支持 interleave=true)。
inline forch::Tensor dup_interleave(const forch::Tensor &t, int64_t rope_dim) {
  auto half = t
    | forch::op::slice({forch::op::Slice::Ellipsis{},
      forch::op::Slice::Interval{0, rope_dim / 2}});
  auto                 n = half.shape();
  std::vector<int64_t> unsq(n.begin(), n.end());
  unsq.push_back(1);
  // tile 是“倍率向量”(torch.repeat 语义), 必须全 1、末位 2 ——
  // 若拷入输入维度会得到 in×tile 的爆炸形状
  std::vector<int64_t> tile(unsq.size(), 1);
  tile.back() = 2;
  auto out    = n;
  out.back()  = n.back() * 2;
  return half | forch::op::reshape(unsq) | forch::op::repeat(tile)
    | forch::op::reshape(out);
}

struct DeepseekV2RotaryEmbedding : public forch::nn::Module {
  forch::Tensor inv_freq; // [1, qk_rope_head_dim / 2], FP32
  float         mscale;   // yarn cos/sin multiplier (else 1.0)

  static float yarn_get_mscale(float scale, float mscale) {
    if (scale <= 1.0F)
      return 1.0F;
    return 0.1F * mscale * std::log(scale) + 1.0F;
  }

  static forch::Tensor compute_inv_freq(const DeepseekV2Config &config) {
    constexpr float kPi = 3.14159265358979323846F;
    int64_t         dim = config.qk_rope_head_dim;

    std::vector<float> inv_freq(dim / 2);
    if (config.rope_scaling.has_value()
      && config.rope_scaling->type == "yarn") {
      const auto &rs = *config.rope_scaling;
      // yarn_find_correction_range(beta_fast, beta_slow, ...)
      auto correction_dim = [&](float num_rotations) {
        return static_cast<float>(dim)
          * std::log(static_cast<float>(rs.original_max_position_embeddings)
            / (num_rotations * 2.0F * kPi))
          / (2.0F * std::log(config.rope_theta));
      };
      auto low =
        std::max(static_cast<int64_t>(std::floor(correction_dim(rs.beta_fast))),
          int64_t(0));
      auto high =
        std::min(static_cast<int64_t>(std::ceil(correction_dim(rs.beta_slow))),
          dim / 2 - 1);
      // inv_freq = freq_inter * ramp + freq_extra * (1 - ramp), with
      // ramp = clamp((arange - low) / (high - low), 0, 1)
      for (int64_t i = 0; i < dim / 2; i++) {
        float exp        = 2.0F * i / dim;
        float freq_extra = 1.0F / std::pow(config.rope_theta, exp);
        float freq_inter =
          1.0F / (rs.factor * std::pow(config.rope_theta, exp));
        float ramp  = (high != low)
           ? std::clamp(
              (static_cast<float>(i) - low) / static_cast<float>(high - low),
              0.0F, 1.0F)
           : std::clamp((static_cast<float>(i) - low) / 0.001F, 0.0F, 1.0F);
        inv_freq[i] = freq_inter * ramp + freq_extra * (1.0F - ramp);
      }
    } else {
      // default / linear: linear scaling folds 1/factor into inv_freq
      float factor = 1.0F;
      if (config.rope_scaling.has_value())
        factor = config.rope_scaling->factor;
      for (int64_t i = 0; i < dim / 2; i++) {
        float exp   = 2.0F * i / dim;
        inv_freq[i] = 1.0F / (factor * std::pow(config.rope_theta, exp));
      }
    }
    return forch::op::load_from_vector(std::move(inv_freq), {1, dim / 2});
  }

  DeepseekV2RotaryEmbedding(const DeepseekV2Config &config)
      : inv_freq(compute_inv_freq(config)), mscale(1.0F) {
    if (config.rope_scaling.has_value()) {
      const auto &rs = *config.rope_scaling;
      if (rs.type == "dynamic") {
        // TODO: NTK dynamic rope recomputes the base at runtime from seq_len
        // — a static graph cannot express it; blocked on host-side input.
        throw forch::exception(
          "TODO: rope_scaling type 'dynamic' is unsupported");
      }
      if (rs.type == "yarn") {
        mscale = yarn_get_mscale(rs.factor, rs.mscale)
          / yarn_get_mscale(rs.factor, rs.mscale_all_dim);
      }
    }
  }

  /// output: (cos, sin), each [T, qk_rope_head_dim] (BF16)
  std::pair<forch::Tensor, forch::Tensor> forward(
    const forch::Tensor &position_ids) const {
    auto t                   = position_ids.shape()[0];
    auto position_ids_expand = position_ids
      | forch::op::cast(forch::DType::FP32) | forch::op::reshape({t, 1});

    auto emb = forch::op::mat_mul()(position_ids_expand, inv_freq)
      | forch::op::repeat({1, 2});

    auto scale = forch::op::load_from_vector<float>({mscale});
    // mul is a variadic curry — calling it with one operand would build a
    // Tensor immediately and break the `|` chain, so pass both operands.
    std::pair<forch::Tensor, forch::Tensor> result = {
      forch::op::mul(emb | forch::op::cos, scale)
        | forch::op::cast(forch::DType::BF16),
      forch::op::mul(emb | forch::op::sin, scale)
        | forch::op::cast(forch::DType::BF16)};
    // 输出前适配为 Interleave kernel 的相邻重复布局(rope 唯一路径,
    // 每 forward 一次, 全层共享): cos[j]=freqs[j%32] →
    // cos'[2i]=cos'[2i+1]=freqs[i]
    auto half_dim = result.first.shape().back() / 2;
    result.first  = dup_interleave(result.first, half_dim * 2);
    result.second = dup_interleave(result.second, half_dim * 2);
    return result;
  }
};

/// Multi-head Latent Attention (DeepseekV2Attention in the reference).
/// kv_b_proj is applied to the current step's compressed latent and the
/// expanded K/V is cached per head (reference eager path).
// Model layer owns low-rank projections, norms, RoPE and W_uk/W_uv absorption.
// common/attention sees already absorbed Q and latent/RoPE cache views; it
// handles kernel layouts/options and returns latent-space attention only.
struct DeepseekV2Attention : public forch::nn::Module {
  int64_t num_heads;
  int64_t qk_nope_head_dim;
  int64_t qk_rope_head_dim;
  int64_t kv_lora_rank;
  int64_t v_head_dim;
  int64_t q_head_dim;    // qk_nope + qk_rope
  int64_t num_kv_groups; // num_heads / num_kv_heads (MLA: always 1)
  float   softmax_scale; // q_head_dim^-0.5, * mscale^2 for yarn

  // q_lora_rank == null  -> q_proj direct
  forch::nn::ColumnParallelLinear *q_proj = nullptr;
  // q_lora_rank set     -> q_a_proj -> q_a_layernorm -> q_b_proj
  forch::nn::Linear               *q_a_proj      = nullptr;
  RMSNorm                         *q_a_layernorm = nullptr;
  forch::nn::ColumnParallelLinear *q_b_proj      = nullptr;
  forch::nn::Linear               *kv_a_proj_with_mqa;
  RMSNorm                         *kv_a_layernorm;
  // The absorbed MLA path reads this local head slice without executing it.
  forch::nn::ColumnParallelLinear *kv_b_proj;
  forch::nn::RowParallelLinear    *o_proj;

  DeepseekV2Attention(
    const DeepseekV2Config &config, const engine::EngineConfig &engine_config)
      : num_heads(config.num_attention_heads / engine_config.tp.world_size()),
        qk_nope_head_dim(config.qk_nope_head_dim),
        qk_rope_head_dim(config.qk_rope_head_dim),
        kv_lora_rank(config.kv_lora_rank), v_head_dim(config.v_head_dim),
        q_head_dim(config.q_head_dim()),
        num_kv_groups(config.num_attention_heads / config.num_key_value_heads),
        softmax_scale(
          std::pow(static_cast<float>(config.q_head_dim()), -0.5F)) {
    if (num_kv_groups != 1) {
      // MLA materializes one K/V per head; every released DeepSeek-V2 config
      // has num_key_value_heads == num_attention_heads.
      throw forch::exception(
        "deepseek_v2: MLA requires num_kv_heads == num_heads");
    }
    if (config.num_attention_heads % engine_config.tp.world_size())
      throw forch::exception(
        "deepseek_v2: TP size must divide attention heads");
    if (config.rope_scaling.has_value()
      && config.rope_scaling->mscale_all_dim > 0.0F) {
      auto m = DeepseekV2RotaryEmbedding::yarn_get_mscale(
        config.rope_scaling->factor, config.rope_scaling->mscale_all_dim);
      softmax_scale *= m * m; // reference: softmax_scale * mscale^2
    }

    if (config.q_lora_rank.has_value()) {
      q_a_proj      = register_loadable("q_a_proj",
             forch::nn::Linear(config.hidden_size, *config.q_lora_rank,
               config.attention_bias, forch::DType::BF16));
      q_a_layernorm = register_loadable("q_a_layernorm",
        RMSNorm(*config.q_lora_rank, config.rms_norm_eps, forch::DType::BF16));
      q_b_proj      = register_loadable("q_b_proj",
             forch::nn::ColumnParallelLinear(*config.q_lora_rank,
               config.num_attention_heads * q_head_dim, engine_config.tp, false,
               forch::DType::BF16));
    } else {
      q_proj = register_loadable("q_proj",
        forch::nn::ColumnParallelLinear(config.hidden_size,
          config.num_attention_heads * q_head_dim, engine_config.tp, false,
          forch::DType::BF16));
    }
    kv_a_proj_with_mqa = register_loadable("kv_a_proj_with_mqa",
      forch::nn::Linear(config.hidden_size, kv_lora_rank + qk_rope_head_dim,
        config.attention_bias, forch::DType::BF16));
    kv_a_layernorm     = register_loadable("kv_a_layernorm",
          RMSNorm(kv_lora_rank, config.rms_norm_eps, forch::DType::BF16));
    kv_b_proj          = register_loadable("kv_b_proj",
               forch::nn::ColumnParallelLinear(kv_lora_rank,
                 config.num_attention_heads * (qk_nope_head_dim + v_head_dim),
                 engine_config.tp, false, forch::DType::BF16));
    o_proj             = register_loadable("o_proj",
                  forch::nn::RowParallelLinear(config.num_attention_heads * v_head_dim,
                    config.hidden_size, engine_config.tp, config.attention_bias,
                    forch::DType::BF16));

    // rope 无开关: 恒为 Interleave 融合算子(cos/sin 由 RotaryEmbedding 适配)。
  }

  /// Fused schemes use the absorbed latent cache; the eager reference
  /// materializes per-head K/V in a full-attention cache.
  forch::Tensor forward(const forch::Tensor &hidden_states,
    std::pair<forch::Tensor, forch::Tensor>  position_embeddings, // (cos, sin)
    const DeepseekV2CacheView &caches, const ForwardInputs &in) const {
    if (in.attention_type == cache::AttentionType::Eager) {
      if (!caches.kv)
        throw forch::exception(
          "deepseek: eager attention requires full KV storage");
      return forward_eager_mla(
        hidden_states, std::move(position_embeddings), *caches.kv, in);
    }
    if (!caches.mla)
      throw forch::exception("deepseek: fused MLA requires latent storage");
    return forward_absorbed_mla(
      hidden_states, std::move(position_embeddings), *caches.mla, in);
  }

  // Model-specific eager reference materializes the absorbed K/V projections.
  forch::Tensor forward_eager_mla(const forch::Tensor &hidden_states,
    std::pair<forch::Tensor, forch::Tensor>            position_embeddings,
    FullAttentionCache &pool, const ForwardInputs &in) const {
    auto tq         = hidden_states.shape()[0];
    auto [cos, sin] = std::move(position_embeddings); // [T, qk_rope_head_dim]
    // rotary_position_embedding requires 3-D [T, 1, D] cos/sin; the eager
    // fallback broadcasts the 2-D [T, D] just as well, so reshape once.
    auto cos3 = cos | forch::op::reshape({tq, 1, qk_rope_head_dim});
    auto sin3 = sin | forch::op::reshape({tq, 1, qk_rope_head_dim});

    // ---- query: [T, H * q_head_dim] (low-rank q_a/q_b or direct q_proj) ----
    auto q = (q_proj != nullptr) ? q_proj->forward(hidden_states)
                                 : q_b_proj->forward(q_a_layernorm->forward(
                                     q_a_proj->forward(hidden_states)));
    q      = q | forch::op::reshape({tq, num_heads, q_head_dim})
      | forch::op::permute({1, 0, 2}); // [H, T, q_head_dim]
    auto q_nope = q
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::IntervalFromBegin{qk_nope_head_dim}});
    auto q_pe = q
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::IntervalTillEnd{qk_nope_head_dim}});

    // ---- compressed latent -> k_nope / value / k_pe ----
    auto compressed_kv = kv_a_proj_with_mqa->forward(hidden_states);
    auto kv_nope       = compressed_kv
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::IntervalFromBegin{kv_lora_rank}});
    auto k_pe = compressed_kv
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::IntervalTillEnd{kv_lora_rank}});
    kv_nope = kv_a_layernorm->forward(kv_nope); // [T, kv_lora_rank]

    auto kv = forch::op::mat_mul(false, true)(kv_nope, kv_b_proj->weight())
      | forch::op::reshape({tq, num_heads, qk_nope_head_dim + v_head_dim})
      | forch::op::permute({1, 0, 2}); // [H, T, nope + v]
    auto k_nope = kv
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::IntervalFromBegin{qk_nope_head_dim}});
    auto value_states = kv
      | forch::op::slice({forch::op::Slice::Ellipsis{},
        forch::op::Slice::IntervalTillEnd{qk_nope_head_dim}});

    // ---- decoupled RoPE on the pe parts only (interleave mode) ----
    auto q_pe_t = q_pe | forch::op::permute({1, 0, 2}); // [T, H, rope]
    auto k_pe_t = k_pe | forch::op::reshape({tq, 1, qk_rope_head_dim});
    q_pe_t      = forch::op::rotary_position_embedding(
      forch::op::RotaryPositionEmbedding::Mode::Interleave)(q_pe_t, cos3, sin3);
    k_pe_t = forch::op::rotary_position_embedding(
      forch::op::RotaryPositionEmbedding::Mode::Interleave)(k_pe_t, cos3, sin3);
    auto q_pe_r = q_pe_t | forch::op::permute({1, 0, 2}); // [H, T, rope]
    auto k_pe_r = k_pe_t | forch::op::permute({1, 0, 2}); // [1, T, rope]

    // ---- materialize full K/V and cache it (reference eager path) ----
    auto k_pe_h = k_pe_r | forch::op::repeat({num_heads, 1, 1});
    auto query_states =
      forch::op::cat((int64_t)q_nope.shape().size() - 1)(q_nope, q_pe_r);
    auto key_states =
      forch::op::cat((int64_t)k_nope.shape().size() - 1)(k_nope, k_pe_h);

    // Cache is token-major; the eager reference computes head-major.
    auto head_major = [](const forch::Tensor &x) {
      return x | forch::op::permute({1, 0, 2});
    };
    std::tie(key_states, value_states) = pool.update(
      head_major(key_states), head_major(value_states), *in.attention);
    auto attn_output = eager_attention(query_states, head_major(key_states),
                         head_major(value_states), in.attention->eager_mask,
                         softmax_scale, prefix_name())
      | forch::op::reshape({tq, num_heads * v_head_dim});
    return o_proj->forward(attn_output);
  }

  // Absorbed model path for both prefill and decode; common/attention owns
  // the scheme dispatch and operator layouts.
  forch::Tensor forward_absorbed_mla(const forch::Tensor &hidden_states,
    std::pair<forch::Tensor, forch::Tensor> position_embeddings, MLACache &pool,
    const ForwardInputs &in) const {
    using namespace forch::op;
    auto tq         = hidden_states.shape()[0];
    auto [cos, sin] = std::move(position_embeddings);
    auto cos3       = cos | reshape({tq, 1, qk_rope_head_dim});
    auto sin3       = sin | reshape({tq, 1, qk_rope_head_dim});

    // ---- query (token-major): [T, H, q_head_dim] ----
    auto q      = (q_proj != nullptr) ? q_proj->forward(hidden_states)
                                      : q_b_proj->forward(q_a_layernorm->forward(
                                     q_a_proj->forward(hidden_states)));
    q           = q | reshape({tq, num_heads, q_head_dim});
    auto q_nope = q
      | slice({Slice::Ellipsis{}, Slice::IntervalFromBegin{qk_nope_head_dim}});
    auto q_pe =
      q | slice({Slice::Ellipsis{}, Slice::IntervalTillEnd{qk_nope_head_dim}});

    // ---- compressed latent + shared k_pe ----
    auto compressed_kv = kv_a_proj_with_mqa->forward(hidden_states);
    auto c_kv          = compressed_kv
      | slice({Slice::Ellipsis{}, Slice::IntervalFromBegin{kv_lora_rank}});
    auto k_pe = compressed_kv
      | slice({Slice::Ellipsis{}, Slice::IntervalTillEnd{kv_lora_rank}});
    c_kv = kv_a_layernorm->forward(c_kv); // [T, kv_lora_rank]

    // ---- decoupled rope on the pe parts ----
    // cos/sin 已被模型层适配为相邻重复布局, 单个 Interleave 融合算子
    // 替换 eager 链的 swap+slice+cat+mul/add ~15 个算子。
    auto q_pe_t = rotary_position_embedding(
      RotaryPositionEmbedding::Mode::Interleave)(q_pe, cos3, sin3);
    auto k_pe_t =
      rotary_position_embedding(RotaryPositionEmbedding::Mode::Interleave)(
        k_pe | reshape({tq, 1, qk_rope_head_dim}), cos3, sin3);

    // ---- cache the latent + roped k_pe (token-major, no permutes) ----
    auto latent              = c_kv | reshape({tq, 1, kv_lora_rank});
    auto k_pe_c              = k_pe_t | reshape({tq, 1, qk_rope_head_dim});
    std::tie(latent, k_pe_c) = pool.update(latent, k_pe_c, *in.attention);
    auto qa = absorb_query(q_nope) | reshape({tq, num_heads, kv_lora_rank});
    auto weighted = mla_attention(in.attention_type, *in.attention, qa, q_pe_t,
      latent, k_pe_c, softmax_scale);
    weighted      = weighted | reshape({1, tq, num_heads, kv_lora_rank});
    return o_proj->forward(unabsorb_output(weighted));
  }

  // q_nope [T,H,nope] -> [1,T,H,latent] using this rank's W_uk weights.
  forch::Tensor absorb_query(const forch::Tensor &q_nope) const {
    using namespace forch::op;
    if (q_nope.dim() != 3 || q_nope.shape()[0] <= 0
      || q_nope.shape()[1] != num_heads
      || q_nope.shape()[2] != qk_nope_head_dim)
      throw forch::exception(
        "deepseek: absorb_query requires [T,local_heads,nope]");
    const auto tq        = q_nope.shape()[0];
    auto       w_uk      = absorber(Slice::Interval{0, qk_nope_head_dim});
    auto       q_nope_hm = q_nope | permute({1, 0, 2});
    auto       q_abs     = batched_mat_mul(false, true)(q_nope_hm, w_uk);
    return q_abs | permute({1, 0, 2})
      | reshape({1, tq, num_heads, kv_lora_rank});
  }

  // Latent attention [1,T,H,latent] -> [T,H*v] using this rank's W_uv weights.
  forch::Tensor unabsorb_output(const forch::Tensor &attn) const {
    using namespace forch::op;
    if (attn.dim() != 4 || attn.shape()[0] != 1 || attn.shape()[1] <= 0
      || attn.shape()[2] != num_heads || attn.shape()[3] != kv_lora_rank)
      throw forch::exception(
        "deepseek: unabsorb_output requires [1,T,local_heads,latent]");
    const auto tq   = attn.shape()[1];
    auto       w_uv = absorber(
      Slice::Interval{qk_nope_head_dim, qk_nope_head_dim + v_head_dim});
    auto attn_tm =
      attn | reshape({tq, num_heads, kv_lora_rank}) | permute({1, 0, 2});
    auto out_v = batched_mat_mul(false, false)(attn_tm, w_uv);
    return out_v | permute({1, 0, 2}) | reshape({tq, num_heads * v_head_dim});
  }

private:
  // Checkpoint rows are [H*(nope+v),latent], not an operator storage layout.
  forch::Tensor absorber(forch::op::Slice::Interval rows) const {
    using namespace forch::op;
    return kv_b_proj->weight()
      | reshape({num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank})
      | slice({Slice::IntervalAll{}, std::move(rows), Slice::IntervalAll{}})
      | permute({0, 2, 1});
  }
};

struct DeepseekV2Gate : public forch::nn::Module {
  forch::nn::Parameter *weight; // [n_routed_experts, hidden_size] BF16
  int64_t               top_k;
  int64_t               n_routed_experts;
  float                 routed_scaling_factor;
  bool                  norm_topk_prob;
  int64_t               n_group;
  int64_t               topk_group;
  std::string           topk_method;
  bool fuse_moe_; // greedy 走融合算子(true)还是 eager 通用链(false)

  DeepseekV2Gate(
    const DeepseekV2Config &config, const engine::EngineConfig &engine_config)
      : top_k(config.num_experts_per_tok),
        n_routed_experts(config.n_routed_experts),
        routed_scaling_factor(config.routed_scaling_factor),
        norm_topk_prob(config.norm_topk_prob), n_group(config.n_group),
        topk_group(config.topk_group), topk_method(config.topk_method),
        fuse_moe_(engine_config.optimization.fuse_moe) {
    if (top_k <= 0 || top_k > n_routed_experts)
      throw std::invalid_argument("DeepSeek MoE: invalid top_k");
    if (topk_method == "group_limited_greedy"
      && (n_group <= 0 || n_routed_experts % n_group || topk_group <= 0
        || topk_group > n_group
        || top_k > n_routed_experts / n_group * topk_group))
      throw std::invalid_argument("DeepSeek MoE: invalid routing groups");
    weight = register_loadable("weight",
      forch::nn::Parameter(
        {config.n_routed_experts, config.hidden_size}, forch::DType::BF16));
    if (config.scoring_func != "softmax") {
      throw forch::exception("TODO: scoring_func '" + config.scoring_func
        + "' unsupported (only 'softmax')");
    }
    if (topk_method != "greedy" && topk_method != "group_limited_greedy") {
      throw forch::exception("TODO: topk_method '" + topk_method
        + "' unsupported (only greedy / group_limited_greedy)");
    }
  }

  /// eager routing chain (generic ops only — zero forch::op::ge): gating
  /// matmul -> softmax -> optional group_limited_greedy mask -> TopK
  /// fan-out + norm/scale.
  std::pair<forch::Tensor, forch::Tensor> forward_eager(
    const forch::Tensor &hidden_states) const {
    using namespace forch::op;
    auto t      = hidden_states.shape()[0];
    auto logits = mat_mul(false, true)(hidden_states, weight->get());
    auto scores = logits | softmax(logits.dim() - 1); // [T, E]
    if (topk_method == "greedy") {
      return compute_topk_scores(scores);
    }

    // group_limited_greedy (reference MoEGate):
    //   group_scores = scores.view(T, n_group, -1).max(-1)   [T, n_group]
    //   group_idx = topk(group_scores, topk_group)            [T, topk_group]
    //   group_mask = zeros.scatter_(1, group_idx, 1)          [T, n_group]
    //   score_mask = group_mask.expand(T, n_group, E/n_group)
    //   tmp = scores.masked_fill(~score_mask, 0); topk(tmp)
    auto n_per_group  = n_routed_experts / n_group;
    auto group_scores = scores | reshape({t, n_group, n_per_group})
      | reduce_max({2}, false); // [T, n_group]
    auto group_idx = topk(topk_group, -1, true)(group_scores).indices;
    auto zeros     = load_from_vector<float>(
      std::vector<float>(static_cast<size_t>(t) * n_group, 0.0F), {t, n_group});
    auto ones = load_from_vector<float>(
      std::vector<float>(static_cast<size_t>(t) * topk_group, 1.0F),
      {t, topk_group});
    auto group_mask = scatter_elements(1, "add")(
      zeros, group_idx | cast(forch::DType::INT32), ones); // [T, n_group] 0/1
    auto score_mask = group_mask | reshape({t, n_group, 1})
      | repeat({1, 1, n_per_group}) | reshape({t, n_routed_experts});
    // masked_fill(~score_mask, 0) == select(score_mask==1, scores, 0)
    auto keep =
      equal()(score_mask, load_from_vector<float>({1.0F})); // [T, E] BOOL
    auto tmp_scores = select()(
      keep, scores, load_from_vector<float>({0.0F}) | cast(scores.dtype()));
    return compute_topk_scores(tmp_scores);
  }

  /// TopK fan-out + norm/scale (generic ops). 扇出节点:一次内核同时产出
  /// values/indices(共享同一 node)。
  std::pair<forch::Tensor, forch::Tensor> compute_topk_scores(
    const forch::Tensor &scores) const {
    using namespace forch::op;
    // CANN BF16 TopK requires sorted output; dispatch consumes paired
    // IDs/weights.
    auto tk = topk(top_k, -1, true)(scores);
    auto w  = tk.values;  // [T, k]
    auto i  = tk.indices; // [T, k] int64
    if (norm_topk_prob && top_k > 1) {
      // reference: topk_weight /= topk_weight.sum(-1, keepdim=True)
      auto denom = reduce_sum({1}, true)(w); // [T, 1]
      w          = div()(w, denom);
    } else {
      auto scale =
        load_from_vector<float>({routed_scaling_factor}) | cast(w.dtype());
      w = mul(w, scale);
    }
    return std::pair<forch::Tensor, forch::Tensor>{
      w, i | cast(forch::DType::INT32)};
  }

  // Greedy sparse routing uses the fused softmax/top-k operator.
  std::pair<forch::Tensor, forch::Tensor> forward(
    const forch::Tensor &hidden_states) const {
    using namespace forch::op;
    if (topk_method != "greedy") {
      return forward_eager(hidden_states); // 本就无融合实现,不受开关影响
    }
    if (!fuse_moe_) {
      // eager(generic ops): softmax + TopK fan-out + norm/scale
      return forward_eager(hidden_states);
    }
    auto logits = mat_mul(false, true)(hidden_states, weight->get());
    auto g =
      forch::op::ge::moe_gating_top_k_softmax(top_k, norm_topk_prob)(logits);
    auto w = g.y; // [T, k], 降序
    if (routed_scaling_factor != 1.0F) {
      auto scale_f =
        load_from_vector<float>({routed_scaling_factor}) | cast(w.dtype());
      w = mul(w, scale_f);
    }
    return {w, g.expert_idx}; // idx 已是 INT32
  }
};

/// DeepSeekMoE — routed experts + shared experts.
struct DeepseekV2MoE : public forch::nn::Module {
  forch::nn::ModuleList<MLP> *experts;
  DeepseekV2Gate             *gate;
  MLP                        *shared_experts;
  forch::ProcessGroup         group_;
  bool fuse_moe_; // true: 融合 gate + 稀疏分发;false: 全通用 eager 路径

  DeepseekV2MoE(
    const DeepseekV2Config &config, const engine::EngineConfig &engine_config)
      : group_(engine_config.tp) {
    auto experts_vector = std::vector<std::unique_ptr<MLP>>();
    for (int64_t i = 0; i < config.n_routed_experts; i++) {
      experts_vector.push_back(std::make_unique<MLP>(config.hidden_size,
        config.moe_intermediate_size, group_, forch::DType::BF16));
    }
    fuse_moe_ = engine_config.optimization.fuse_moe;
    experts   = register_loadable(
      "experts", forch::nn::ModuleList<MLP>(std::move(experts_vector)));
    gate = register_loadable("gate", DeepseekV2Gate(config, engine_config));
    if (config.n_shared_experts > 0) {
      shared_experts = register_loadable("shared_experts",
        MLP(config.hidden_size,
          config.moe_intermediate_size * config.n_shared_experts, group_,
          forch::DType::BF16));
    } else {
      shared_experts = nullptr;
    }
  }

  MoEWeights<forch::TensorList> weights() const {
    MoEWeights<forch::TensorList> result{{}, {}, {}};
    for (size_t i = 0; i < experts->size(); ++i) {
      const auto &expert = (*experts)[i];
      result.gate.push_back(expert->gate_proj->weight());
      result.up.push_back(expert->up_proj->weight());
      result.down.push_back(expert->down_proj->weight());
    }
    return result;
  }

  forch::Tensor forward(const forch::Tensor &x) const {
    auto [scores, indices] = gate->forward(x);
    return forward_with(x, scores, indices, fuse_moe_);
  }

  forch::Tensor forward_with(const forch::Tensor &x,
    const forch::Tensor &scores, const forch::Tensor &indices,
    bool sparse) const {
    auto result =
      finer::models::moe(x, weights(), scores, indices, sparse, group_);
    if (shared_experts)
      result = forch::op::add(result, shared_experts->forward(x));
    return result;
  }

  forch::Tensor forward_dense_with(const forch::Tensor &x,
    const forch::Tensor &scores, const forch::Tensor &indices) const {
    return forward_with(x, scores, indices, false);
  }

  forch::Tensor forward_moe_with(const forch::Tensor &x,
    const forch::Tensor &scores, const forch::Tensor &indices) const {
    return forward_with(x, scores, indices, true);
  }
};

struct DeepseekV2DecoderLayer : public forch::nn::Module {
  DeepseekV2Attention *self_attn;
  RMSNorm             *input_layernorm;
  RMSNorm             *post_attention_layernorm;
  // mlp is the dense MLP in the first_k_dense_replace prefix and a
  // routed DeepseekV2MoE afterwards (moe_layer_freq == 1 in all released
  // configs). Exactly one is registered under "mlp".
  MLP           *mlp = nullptr;
  DeepseekV2MoE *moe = nullptr;

  DeepseekV2DecoderLayer(const DeepseekV2Config &config,
    const engine::EngineConfig &engine_config, int64_t layer_idx)
      : self_attn(register_loadable(
          "self_attn", DeepseekV2Attention(config, engine_config))),
        input_layernorm(register_loadable("input_layernorm",
          RMSNorm(
            config.hidden_size, config.rms_norm_eps, forch::DType::BF16))),
        post_attention_layernorm(register_loadable("post_attention_layernorm",
          RMSNorm(
            config.hidden_size, config.rms_norm_eps, forch::DType::BF16))) {
    bool use_moe = config.n_routed_experts > 0
      && layer_idx >= config.first_k_dense_replace
      && layer_idx % config.moe_layer_freq == 0;
    if (use_moe) {
      moe = register_loadable("mlp", DeepseekV2MoE(config, engine_config));
    } else {
      mlp = register_loadable("mlp",
        MLP(config.hidden_size, config.intermediate_size, engine_config.tp,
          forch::DType::BF16));
    }
  }

  forch::Tensor forward(forch::Tensor       hidden_states,
    const DeepseekV2CacheView              &caches,
    std::pair<forch::Tensor, forch::Tensor> position_embeddings,
    const ForwardInputs                    &in) const {
    // attention
    auto attn_residual = hidden_states;
    hidden_states      = input_layernorm->forward(hidden_states);
    hidden_states      = self_attn->forward(
      hidden_states, std::move(position_embeddings), caches, in);
    hidden_states = forch::op::add(hidden_states, attn_residual);

    // mlp
    auto mlp_residual = hidden_states;
    hidden_states     = hidden_states | *post_attention_layernorm;
    hidden_states =
      mlp ? mlp->forward(hidden_states) : moe->forward(hidden_states);
    hidden_states = forch::op::add(hidden_states, mlp_residual);
    return hidden_states;
  }
};

struct DeepseekV2Model : public forch::nn::Module {
  forch::nn::Embedding                          *embed_tokens;
  forch::nn::ModuleList<DeepseekV2DecoderLayer> *layers;
  RMSNorm                                       *norm;
  DeepseekV2RotaryEmbedding                     *rotary_emb;

  DeepseekV2Model(
    const DeepseekV2Config &config, const engine::EngineConfig &engine_config) {
    embed_tokens = register_loadable("embed_tokens",
      forch::nn::Embedding(
        config.vocab_size, config.hidden_size, forch::DType::BF16));

    auto num_layers = engine_config.n_model_layers > 0
      ? std::min(engine_config.n_model_layers, config.num_hidden_layers)
      : config.num_hidden_layers;

    auto layers_vector = std::vector<std::unique_ptr<DeepseekV2DecoderLayer>>();
    for (int64_t i = 0; i < num_layers; i++) {
      layers_vector.push_back(
        std::make_unique<DeepseekV2DecoderLayer>(config, engine_config, i));
    }
    layers = register_loadable("layers",
      forch::nn::ModuleList<DeepseekV2DecoderLayer>(std::move(layers_vector)));

    norm = register_loadable("norm",
      RMSNorm(config.hidden_size, config.rms_norm_eps, forch::DType::BF16));
    rotary_emb =
      register_loadable("rotary_emb", DeepseekV2RotaryEmbedding(config));
  }

  forch::Tensor forward(const ForwardInputs &in,
    const std::vector<DeepseekV2CacheView>  &caches) const {
    if (caches.size() < layers->size())
      throw forch::exception("deepseek: missing layer caches");
    auto hidden_states       = embed_tokens->forward(in.input_ids);
    auto position_embeddings = rotary_emb->forward(in.position_ids);
    for (size_t i = 0; i < layers->size(); ++i)
      hidden_states = (*layers)[i]->forward(
        hidden_states, caches[i], position_embeddings, in);

    hidden_states = norm->forward(hidden_states);
    return hidden_states;
  }
};

struct DeepseekV2ForCausalLM : public forch::nn::Module {
  DeepseekV2Model     *model;
  VocabParallelLMHead *lm_head;

  DeepseekV2ForCausalLM(
    const DeepseekV2Config &config, const engine::EngineConfig &engine_config) {
    model = register_loadable("model", DeepseekV2Model(config, engine_config));
    lm_head = register_loadable(
      config.tie_word_embeddings ? "model.embed_tokens" : "lm_head",
      VocabParallelLMHead(config.hidden_size, config.vocab_size,
        engine_config.logits_topk, engine_config.tp, forch::DType::BF16));
  }

  forch::op::TopKOutputs forward(const ForwardInputs &in,
    const std::vector<DeepseekV2CacheView>           &caches) const {
    return lm_head->forward(model->forward(in, caches));
  }
};

} // namespace finer::models::deepseek_v2
