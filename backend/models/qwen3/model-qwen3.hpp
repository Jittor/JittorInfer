#pragma once

#include "../../cache/full-attention.hpp"
#include "../../engine/config.hpp"
#include "../common/attention/attention.hpp"
#include "../common/runner.hpp"

#include "../common/lm-head.hpp"
#include "../common/mlp.hpp"
#include "../common/moe.hpp"
#include "../common/norm.hpp"
#include "../common/rope.hpp"
#include "config-qwen3.hpp"

#include <cstdint>
#include <forch/forch.hpp>
#include <forch/nn/embedding.hpp>
#include <forch/nn/linear.hpp>
#include <forch/nn/module.hpp>
#include <forch/nn/parallel_linear.hpp>
#include <forch/ops/batched_mat_mul.hpp>
#include <forch/ops/cast.hpp>
#include <forch/ops/cat.hpp>
#include <forch/ops/const.hpp>
#include <forch/ops/function.hpp>
#include <forch/ops/mat_mul.hpp>
#include <forch/ops/permute.hpp>
#include <forch/ops/repeat.hpp>
#include <forch/ops/reshape.hpp>
#include <forch/ops/rotary_position_embedding.hpp>
#include <forch/ops/slice.hpp>
#include <forch/ops/softmax.hpp>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace finer::models::qwen3 {

using cache::FullAttentionCache;

/**
 * 这不是一个高效的实现，而是直接照抄 Qwen3 在 transformer 里的 fallback 实现
 * 主要验证 forch 框架的灵活度和正确性
 */
struct Qwen3RotaryEmbedding : public forch::nn::Module {
  forch::Tensor inv_freq;

  static forch::Tensor compute_default_rope_parameters(
    const Qwen3Config &config) {
    std::vector<float> inv_freq(config.head_dim / 2);
    for (size_t i = 0; i < inv_freq.size(); i++) {
      auto exp =
        static_cast<float>(i * 2) / static_cast<float>(config.head_dim);
      inv_freq[i] = 1.0F / std::pow(config.rope_theta, exp);
    }
    return forch::op::load_from_vector(
      std::move(inv_freq), {1, config.head_dim / 2});
  }

  Qwen3RotaryEmbedding(const Qwen3Config &config)
      : inv_freq(compute_default_rope_parameters(config)) {}

  /**
   * output shape: [seq_len, head_dim]
   */
  std::pair<forch::Tensor, forch::Tensor> forward(
    const forch::Tensor &position_ids) const {
    auto position_ids_expand = position_ids
      | forch::op::cast(forch::DType::FP32)
      | forch::op::reshape({position_ids.shape()[0], 1});

    auto emb = forch::op::mat_mul()(position_ids_expand, inv_freq)
      | forch::op::repeat({1, 2});

    return {emb | forch::op::cos | forch::op::cast(forch::DType::BF16),
      emb | forch::op::sin | forch::op::cast(forch::DType::BF16)};
  }
};

// Model layer: QKV weights, Q/K norm, RoPE, cache update and output projection.
// common/attention consumes prepared Q/K/V and only adapts the selected kernel.
// The eager reference retains this model's arithmetic and debug node names.
struct Qwen3Attention : public forch::nn::Module {
  int64_t head_dim;
  int64_t num_attention_heads;
  int64_t num_key_value_heads;
  int64_t num_key_value_groups;
  float   scaling; // 1 / sqrt(head_dim)
  bool    is_causal = true;
  bool    fuse_rope = false;

  forch::nn::QKVParallelLinear  qkv_proj;
  forch::nn::RowParallelLinear *o_proj;
  RMSNorm                      *q_norm;
  RMSNorm                      *k_norm;

  Qwen3Attention(
    const Qwen3Config &config, const engine::EngineConfig &engine_config)
      : head_dim(config.head_dim),
        scaling(1.0F / std::sqrt(static_cast<float>(config.head_dim))),
        qkv_proj(config.hidden_size, config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, engine_config.tp,
          config.attention_bias, forch::DType::BF16) {
    num_attention_heads  = qkv_proj.local_query_heads();
    num_key_value_heads  = qkv_proj.local_key_value_heads();
    num_key_value_groups = num_attention_heads / num_key_value_heads;
    o_proj               = register_loadable("o_proj",
                    forch::nn::RowParallelLinear(config.num_attention_heads * config.head_dim,
                      config.hidden_size, engine_config.tp, config.attention_bias,
                      forch::DType::BF16));
    q_norm               = register_loadable("q_norm",
                    RMSNorm(config.head_dim, config.rms_norm_eps, forch::DType::BF16));
    k_norm               = register_loadable("k_norm",
                    RMSNorm(config.head_dim, config.rms_norm_eps, forch::DType::BF16));

    fuse_rope = engine_config.optimization.fuse_rope;
  }

  // TODO: register qkv_proj as a child once forch supports an empty child
  // key, and drop this override (also in Llama and Qwen3.5).
  void load(
    forch::nn::Loader *loader, const std::string &prefix = "") override {
    Module::load(loader, prefix);
    // QKV shares the original checkpoint prefix, without a wrapper key.
    qkv_proj.load(loader, prefix);
  }

  forch::Tensor forward(const forch::Tensor &hidden_states,
    std::pair<forch::Tensor, forch::Tensor>  position_embeddings, // (cos, sin)
    FullAttentionCache &kv, const ForwardInputs &in) {
    auto tq = hidden_states.shape()[0];

    auto [cos, sin] = std::move(position_embeddings);
    cos             = cos | forch::op::reshape({tq, 1, head_dim});
    sin             = sin | forch::op::reshape({tq, 1, head_dim});
    std::function<forch::Tensor(const forch::Tensor &)> rope;
    if (fuse_rope) {
      rope = [cos = std::move(cos), sin = std::move(sin)](
               const forch::Tensor &qk) {
        return forch::op::rotary_position_embedding(
          forch::op::RotaryPositionEmbedding::Mode::Half)(qk, cos, sin);
      };
    } else {
      rope = [cos = std::move(cos), sin = std::move(sin)](
               const forch::Tensor &qk) {
        return apply_rope_half(qk, cos, sin);
      };
    }

    auto [q_raw, k_raw, v_raw] = qkv_proj.forward(hidden_states);
    auto query_states          = q_raw                          //
      | forch::op::reshape({tq, num_attention_heads, head_dim}) //
      | *q_norm                                                 //
      | rope;

    auto key_states = k_raw                                     //
      | forch::op::reshape({tq, num_key_value_heads, head_dim}) //
      | *k_norm                                                 //
      | rope;

    auto value_states = v_raw //
      | forch::op::reshape({tq, num_key_value_heads, head_dim});

    std::tie(key_states, value_states) =
      kv.update(key_states, value_states, *in.attention);
    auto attn_output = full_attention(in.attention_type, *in.attention,
      query_states, key_states, value_states, scaling, prefix_name());

    attn_output = attn_output
      | forch::op::reshape({tq, num_attention_heads * head_dim}) | *o_proj;
    return attn_output;
  }
};

struct Qwen3DecoderLayer : public forch::nn::Module {
  Qwen3Attention *self_attn;
  MLP            *mlp;
  RMSNorm        *input_layernorm;
  RMSNorm        *post_attention_layernorm;

  Qwen3DecoderLayer(
    const Qwen3Config &config, const engine::EngineConfig &engine_config)
      : self_attn(register_loadable(
          "self_attn", Qwen3Attention(config, engine_config))),
        mlp(register_loadable("mlp",
          MLP(config.hidden_size, config.intermediate_size, engine_config.tp,
            forch::DType::BF16))),
        input_layernorm(register_loadable("input_layernorm",
          RMSNorm(
            config.hidden_size, config.rms_norm_eps, forch::DType::BF16))),
        post_attention_layernorm(register_loadable("post_attention_layernorm",
          RMSNorm(
            config.hidden_size, config.rms_norm_eps, forch::DType::BF16))) {}

  forch::Tensor forward(forch::Tensor hidden_states, FullAttentionCache &kv,
    std::pair<forch::Tensor, forch::Tensor> position_embeddings,
    const ForwardInputs                    &in) const {
    // attention
    auto attn_residual = hidden_states;
    hidden_states      = input_layernorm->forward(hidden_states);
    hidden_states =
      self_attn->forward(hidden_states, std::move(position_embeddings), kv, in);
    hidden_states = forch::op::add(hidden_states, attn_residual);

    // mlp
    auto mlp_residual = hidden_states;
    hidden_states     = hidden_states | *post_attention_layernorm | *mlp;
    hidden_states     = forch::op::add(hidden_states, mlp_residual);
    return hidden_states;
  }
};

struct Qwen3Model : public forch::nn::Module {
  forch::nn::Embedding                     *embed_tokens;
  forch::nn::ModuleList<Qwen3DecoderLayer> *layers;
  RMSNorm                                  *norm;
  Qwen3RotaryEmbedding                     *rotary_emb;

  Qwen3Model(
    const Qwen3Config &config, const engine::EngineConfig &engine_config) {
    embed_tokens = register_loadable("embed_tokens",
      forch::nn::Embedding(
        config.vocab_size, config.hidden_size, forch::DType::BF16));

    auto num_layers = engine_config.n_model_layers > 0
      ? std::min(engine_config.n_model_layers, config.num_hidden_layers)
      : config.num_hidden_layers;

    auto layers_vector = std::vector<std::unique_ptr<Qwen3DecoderLayer>>();
    for (int64_t i = 0; i < num_layers; i++) {
      layers_vector.push_back(
        std::make_unique<Qwen3DecoderLayer>(config, engine_config));
    }
    layers = register_loadable("layers",
      forch::nn::ModuleList<Qwen3DecoderLayer>(std::move(layers_vector)));

    norm       = register_loadable("norm",
            RMSNorm(config.hidden_size, config.rms_norm_eps, forch::DType::BF16));
    rotary_emb = register_loadable("rotary_emb", Qwen3RotaryEmbedding(config));
  }

  forch::Tensor forward(const ForwardInputs &in,
    const std::vector<FullAttentionCache *> &caches) const {
    if (caches.size() < layers->size())
      throw forch::exception("qwen3: missing layer caches");
    auto hidden_states       = embed_tokens->forward(in.input_ids);
    auto position_embeddings = rotary_emb->forward(in.position_ids);
    for (size_t i = 0; i < layers->size(); ++i)
      hidden_states = (*layers)[i]->forward(
        hidden_states, *caches[i], position_embeddings, in);

    hidden_states = norm->forward(hidden_states);
    return hidden_states;
  }
};

struct Qwen3ForCausalLM : public forch::nn::Module {
  Qwen3Model          *model;
  VocabParallelLMHead *lm_head;

  Qwen3ForCausalLM(
    const Qwen3Config &config, const engine::EngineConfig &engine_config) {
    model   = register_loadable("model", Qwen3Model(config, engine_config));
    lm_head = register_loadable(
      config.tie_word_embeddings ? "model.embed_tokens" : "lm_head",
      VocabParallelLMHead(config.hidden_size, config.vocab_size,
        engine_config.logits_topk, engine_config.tp, forch::DType::BF16));
  }

  forch::op::TopKOutputs forward(const ForwardInputs &in,
    const std::vector<FullAttentionCache *>          &caches) const {
    return lm_head->forward(model->forward(in, caches));
  }
};

} // namespace finer::models::qwen3
