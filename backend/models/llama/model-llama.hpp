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
#include "config-llama.hpp"

#include <forch/forch.hpp>
#include <forch/nn/embedding.hpp>
#include <forch/nn/linear.hpp>
#include <forch/nn/module.hpp>
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>

#include <stdexcept>
#include <utility>
#include <vector>

namespace finer::models::llama {

using cache::FullAttentionCache;

struct LlamaRotaryEmbedding : public forch::nn::Module {
  forch::Tensor inv_freq;

  static forch::Tensor compute_default_rope_parameters(
    const LlamaConfig &config) {
    std::vector<float> inv_freq(config.head_dim / 2);
    for (size_t i = 0; i < inv_freq.size(); i++) {
      const auto exp =
        static_cast<float>(i * 2) / static_cast<float>(config.head_dim);
      inv_freq[i] = 1.0F / std::pow(config.rope_theta, exp);
    }
    return forch::op::load_from_vector(
      std::move(inv_freq), {1, config.head_dim / 2});
  }

  static forch::Tensor compute_llama3_rope_parameters(
    const LlamaConfig &config, const LlamaRopeScaling &rope_scaling) {
    std::vector<float> inv_freq(config.head_dim / 2);
    constexpr float    pi = 3.14159265358979323846F;

    const auto low_freq_wavelen =
      static_cast<float>(rope_scaling.original_max_position_embeddings)
      / rope_scaling.low_freq_factor;
    const auto high_freq_wavelen =
      static_cast<float>(rope_scaling.original_max_position_embeddings)
      / rope_scaling.high_freq_factor;

    for (size_t i = 0; i < inv_freq.size(); i++) {
      const auto exp =
        static_cast<float>(i * 2) / static_cast<float>(config.head_dim);
      const auto default_inv_freq = 1.0F / std::pow(config.rope_theta, exp);
      const auto wavelen          = 2.0F * pi / default_inv_freq;

      if (wavelen > low_freq_wavelen) {
        inv_freq[i] = default_inv_freq / rope_scaling.factor;
      } else if (wavelen < high_freq_wavelen) {
        inv_freq[i] = default_inv_freq;
      } else {
        const auto smooth =
          (static_cast<float>(rope_scaling.original_max_position_embeddings)
              / wavelen
            - rope_scaling.low_freq_factor)
          / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
        inv_freq[i] = (1.0F - smooth) * default_inv_freq / rope_scaling.factor
          + smooth * default_inv_freq;
      }
    }

    return forch::op::load_from_vector(
      std::move(inv_freq), {1, config.head_dim / 2});
  }

  static forch::Tensor compute_rope_parameters(const LlamaConfig &config) {
    if (!config.rope_scaling.has_value()) {
      return compute_default_rope_parameters(config);
    }

    const auto &rope_scaling = config.rope_scaling.value();
    if (rope_scaling.rope_type == "default") {
      return compute_default_rope_parameters(config);
    }
    if (rope_scaling.rope_type == "llama3") {
      return compute_llama3_rope_parameters(config, rope_scaling);
    }

    throw std::runtime_error(
      "Unsupported Llama rope type: " + rope_scaling.rope_type);
  }

  LlamaRotaryEmbedding(const LlamaConfig &config)
      : inv_freq(compute_rope_parameters(config)) {}

  std::pair<forch::Tensor, forch::Tensor> forward(
    const forch::Tensor &position_ids) const {
    auto position_ids_expand = position_ids
      | forch::op::cast(forch::DType::FP32)
      | forch::op::reshape({position_ids.shape()[0], 1});

    auto emb = forch::op::mat_mul()(position_ids_expand, inv_freq)
      | forch::op::repeat({1, 2});

    auto cos = emb | forch::op::cos | forch::op::cast(forch::DType::FP16);
    auto sin = emb | forch::op::sin | forch::op::cast(forch::DType::FP16);
    return {cos, sin};
  }
};

// Model layer: QKV weights, RoPE, cache update and output projection.
// common/attention receives prepared Q/K/V, not hidden states or model weights.
// The FP16 eager reference stays here to preserve its arithmetic/debug names.
struct LlamaAttention : public forch::nn::Module {
  int64_t head_dim;
  int64_t num_attention_heads;
  int64_t num_key_value_heads;
  int64_t num_key_value_groups;
  float   scaling;
  bool    is_causal = true;
  bool    fuse_rope = false;

  forch::nn::QKVParallelLinear  qkv_proj;
  forch::nn::RowParallelLinear *o_proj;

  LlamaAttention(
    const LlamaConfig &config, const engine::EngineConfig &engine_config)
      : head_dim(config.head_dim),
        scaling(1.0F / std::sqrt(static_cast<float>(config.head_dim))),
        qkv_proj(config.hidden_size, config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, engine_config.tp,
          config.attention_bias, forch::DType::FP16) {
    num_attention_heads  = qkv_proj.local_query_heads();
    num_key_value_heads  = qkv_proj.local_key_value_heads();
    num_key_value_groups = num_attention_heads / num_key_value_heads;
    o_proj               = register_loadable("o_proj",
                    forch::nn::RowParallelLinear(config.num_attention_heads * config.head_dim,
                      config.hidden_size, engine_config.tp, config.attention_bias,
                      forch::DType::FP16));

    fuse_rope = engine_config.optimization.fuse_rope;
  }

  // TODO: see Qwen3Attention::load.
  void load(
    forch::nn::Loader *loader, const std::string &prefix = "") override {
    Module::load(loader, prefix);
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
      | rope;

    auto key_states = k_raw                                     //
      | forch::op::reshape({tq, num_key_value_heads, head_dim}) //
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

struct LlamaDecoderLayer : public forch::nn::Module {
  LlamaAttention *self_attn;
  MLP            *mlp;
  RMSNorm        *input_layernorm;
  RMSNorm        *post_attention_layernorm;

  LlamaDecoderLayer(
    const LlamaConfig &config, const engine::EngineConfig &engine_config)
      : self_attn(register_loadable(
          "self_attn", LlamaAttention(config, engine_config))),
        mlp(register_loadable("mlp",
          MLP(config.hidden_size, config.intermediate_size, engine_config.tp,
            forch::DType::FP16, config.mlp_bias))),
        input_layernorm(register_loadable("input_layernorm",
          RMSNorm(
            config.hidden_size, config.rms_norm_eps, forch::DType::FP16))),
        post_attention_layernorm(register_loadable("post_attention_layernorm",
          RMSNorm(
            config.hidden_size, config.rms_norm_eps, forch::DType::FP16))) {}

  forch::Tensor forward(forch::Tensor hidden_states, FullAttentionCache &kv,
    std::pair<forch::Tensor, forch::Tensor> position_embeddings,
    const ForwardInputs                    &in) const {
    auto attn_residual = hidden_states;
    hidden_states      = input_layernorm->forward(hidden_states);
    hidden_states =
      self_attn->forward(hidden_states, std::move(position_embeddings), kv, in);
    hidden_states = forch::op::add(hidden_states, attn_residual);

    auto mlp_residual = hidden_states;
    hidden_states     = hidden_states | *post_attention_layernorm | *mlp;
    hidden_states     = forch::op::add(hidden_states, mlp_residual);
    return hidden_states;
  }
};

struct LlamaModel : public forch::nn::Module {
  forch::nn::Embedding                     *embed_tokens;
  forch::nn::ModuleList<LlamaDecoderLayer> *layers;
  RMSNorm                                  *norm;
  LlamaRotaryEmbedding                     *rotary_emb;

  LlamaModel(
    const LlamaConfig &config, const engine::EngineConfig &engine_config) {
    embed_tokens = register_loadable("embed_tokens",
      forch::nn::Embedding(
        config.vocab_size, config.hidden_size, forch::DType::FP16));

    auto num_layers = engine_config.n_model_layers > 0
      ? std::min(engine_config.n_model_layers, config.num_hidden_layers)
      : config.num_hidden_layers;

    auto layers_vector = std::vector<std::unique_ptr<LlamaDecoderLayer>>();
    for (int64_t i = 0; i < num_layers; i++) {
      layers_vector.push_back(
        std::make_unique<LlamaDecoderLayer>(config, engine_config));
    }
    layers = register_loadable("layers",
      forch::nn::ModuleList<LlamaDecoderLayer>(std::move(layers_vector)));

    norm       = register_loadable("norm",
            RMSNorm(config.hidden_size, config.rms_norm_eps, forch::DType::FP16));
    rotary_emb = register_loadable("rotary_emb", LlamaRotaryEmbedding(config));
  }

  forch::Tensor forward(const ForwardInputs &in,
    const std::vector<FullAttentionCache *> &caches) const {
    if (caches.size() < layers->size())
      throw forch::exception("llama: missing layer caches");
    auto hidden_states       = embed_tokens->forward(in.input_ids);
    auto position_embeddings = rotary_emb->forward(in.position_ids);
    for (size_t i = 0; i < layers->size(); ++i)
      hidden_states = (*layers)[i]->forward(
        hidden_states, *caches[i], position_embeddings, in);

    hidden_states = norm->forward(hidden_states);
    return hidden_states;
  }
};

struct LlamaForCausalLM : public forch::nn::Module {
  LlamaModel          *model;
  VocabParallelLMHead *lm_head;

  LlamaForCausalLM(
    const LlamaConfig &config, const engine::EngineConfig &engine_config) {
    model   = register_loadable("model", LlamaModel(config, engine_config));
    lm_head = register_loadable(
      config.tie_word_embeddings ? "model.embed_tokens" : "lm_head",
      VocabParallelLMHead(config.hidden_size, config.vocab_size,
        engine_config.logits_topk, engine_config.tp, forch::DType::FP16));
  }

  forch::op::TopKOutputs forward(const ForwardInputs &in,
    const std::vector<FullAttentionCache *>          &caches) const {
    return lm_head->forward(model->forward(in, caches));
  }
};

} // namespace finer::models::llama
