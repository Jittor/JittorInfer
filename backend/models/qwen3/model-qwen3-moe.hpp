#pragma once

#include "config-qwen3.hpp"
#include "model-qwen3.hpp"

#include <forch/nn/linear.hpp>
#include <forch/nn/parallel_linear.hpp>
#include <forch/nn/parameter.hpp>
#include <forch/ops/batched_mat_mul.hpp>
#include <forch/ops/cast.hpp>
#include <forch/ops/collective.hpp>
#include <forch/ops/div.hpp>
#include <forch/ops/ge/moe_finalize_routing.hpp>
#include <forch/ops/ge/moe_init_routing.hpp>
#include <forch/ops/grouped_mat_mul.hpp>
#include <forch/ops/one_hot.hpp>
#include <forch/ops/permute.hpp>
#include <forch/ops/reduce.hpp>
#include <forch/ops/repeat.hpp>
#include <forch/ops/reshape.hpp>
#include <forch/ops/softmax.hpp>
#include <forch/ops/topk.hpp>

namespace finer::models::qwen3 {

struct Qwen3MoeExperts : forch::nn::Module {
  forch::nn::Parameter *gate, *up, *down;
  int64_t               experts, hidden, intermediate;
  forch::ProcessGroup   group;

  Qwen3MoeExperts(const Qwen3Config &cfg, const forch::ProcessGroup &parallel)
      : experts(cfg.num_experts), hidden(cfg.hidden_size),
        intermediate(cfg.moe_intermediate_size), group(parallel) {
    if (experts <= 0 || hidden <= 0 || intermediate <= 0
      || intermediate % group.world_size())
      throw std::invalid_argument(
        "MoE dimensions must be positive and width must divide TP size");
    auto local = intermediate / group.world_size();
    gate       = register_loadable("gate",
            forch::nn::Parameter({experts, local, hidden}, forch::DType::BF16));
    up         = register_loadable(
      "up", forch::nn::Parameter({experts, local, hidden}, forch::DType::BF16));
    down = register_loadable("down",
      forch::nn::Parameter({experts, hidden, local}, forch::DType::BF16));
  }

  void load(forch::nn::Loader *loader, const std::string &prefix) override {
    // Stack only this rank's intermediate channels, preserving BF16 bytes.
    // No graph-side concat or full-weight device allocation is needed.
    struct Shards : forch::nn::Loader {
      forch::nn::Loader     *source;
      const Qwen3MoeExperts &owner;
      std::string            prefix;
      Shards(forch::nn::Loader *source, const Qwen3MoeExperts &owner,
        std::string prefix)
          : source(source), owner(owner), prefix(std::move(prefix)) {}
      forch::Tensor load(const std::string &name) override {
        using namespace forch;
        using nn::WeightShard;
        const bool is_down    = name.ends_with(".down");
        const auto projection = is_down
          ? "down_proj"
          : (name.ends_with(".up") ? "up_proj" : "gate_proj");
        const auto local      = owner.intermediate / owner.group.world_size();
        const auto start      = local * owner.group.rank_id();
        const auto bytes_per_expert =
          WeightShard::multiply(WeightShard::multiply(local, owner.hidden), 2);
        std::vector<uint8_t> bytes(
          WeightShard::multiply(owner.experts, bytes_per_expert));
        for (int64_t e = 0; e < owner.experts; ++e) {
          auto key =
            prefix + "." + std::to_string(e) + "." + projection + ".weight";
          auto full     = source->load(key);
          auto expected = is_down ? Shape{owner.hidden, owner.intermediate}
                                  : Shape{owner.intermediate, owner.hidden};
          if (full.shape() != expected || full.dtype() != DType::BF16)
            throw exception(
              "Qwen3 MoE: incompatible expert checkpoint at " + key);
          auto shard = WeightShard::shard(full, is_down ? 1 : 0, start, local);
          const auto *data =
            dynamic_cast<const op::Const *>(shard.get_op())->buffer();
          std::memcpy(
            bytes.data() + e * bytes_per_expert, data, bytes_per_expert);
        }
        auto shape = is_down ? Shape{owner.experts, owner.hidden, local}
                             : Shape{owner.experts, local, owner.hidden};
        return Tensor({},
          std::make_unique<WeightShard::ShardConstant>(
            std::move(bytes), std::move(shape), DType::BF16));
      }
    } shards(loader, *this, prefix);
    Module::load(&shards, prefix);
  }

  MoEWeights<> weights() const {
    return {gate->get(), up->get(), down->get()};
  }

  forch::Tensor forward(const forch::Tensor &x, const forch::Tensor &scores,
    const forch::Tensor &indices, bool sparse) const {
    return finer::models::moe(x, weights(), scores, indices, sparse, group);
  }
};

struct Qwen3MoEGate : forch::nn::Linear {
  int64_t top_k;
  bool    normalize;

  explicit Qwen3MoEGate(const Qwen3Config &config)
      : forch::nn::Linear(
          config.hidden_size, config.num_experts, false, forch::DType::BF16),
        top_k(config.num_experts_per_tok), normalize(config.norm_topk_prob) {}

  std::pair<forch::Tensor, forch::Tensor> forward(const forch::Tensor &x) {
    using namespace forch::op;
    auto selected =
      topk(top_k, -1, true)(forch::nn::Linear::forward(x) | softmax(1));
    auto scores = selected.values;
    if (normalize)
      scores = div()(scores, scores | reduce_sum({-1}, true));
    return {scores, selected.indices | cast(forch::DType::INT32)};
  }
};

struct Qwen3MoE : forch::nn::Module {
  Qwen3MoeExperts *experts;
  Qwen3MoEGate    *gate;
  bool             sparse;
  Qwen3MoE(const Qwen3Config &config, const engine::EngineConfig &ec)
      : sparse(ec.optimization.fuse_moe) {
    experts = register_loadable("experts", Qwen3MoeExperts(config, ec.tp));
    gate    = register_loadable("gate", Qwen3MoEGate(config));
  }
  forch::Tensor forward(const forch::Tensor &x) const {
    auto [scores, indices] = gate->forward(x);
    auto routed            = experts->forward(x, scores, indices, sparse);
    return routed;
  }
};

struct Qwen3MoeDecoderLayer : public forch::nn::Module {
  Qwen3Attention *self_attn;
  MLP            *mlp = nullptr;
  Qwen3MoE       *moe = nullptr;
  RMSNorm        *input_layernorm;
  RMSNorm        *post_attention_layernorm;

  Qwen3MoeDecoderLayer(const Qwen3Config &config,
    const engine::EngineConfig &engine_config, int64_t index)
      : self_attn(register_loadable(
          "self_attn", Qwen3Attention(config, engine_config))),
        input_layernorm(register_loadable("input_layernorm",
          RMSNorm(
            config.hidden_size, config.rms_norm_eps, forch::DType::BF16))),
        post_attention_layernorm(register_loadable("post_attention_layernorm",
          RMSNorm(
            config.hidden_size, config.rms_norm_eps, forch::DType::BF16))) {
    if (config.layer_is_moe(index))
      moe = register_loadable("mlp", Qwen3MoE(config, engine_config));
    else
      mlp = register_loadable("mlp",
        MLP(config.hidden_size, config.intermediate_size, engine_config.tp,
          forch::DType::BF16));
  }

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
    hidden_states     = post_attention_layernorm->forward(hidden_states);
    hidden_states =
      moe ? moe->forward(hidden_states) : mlp->forward(hidden_states);
    hidden_states = forch::op::add(hidden_states, mlp_residual);
    return hidden_states;
  }
};

struct Qwen3MoeModel : public forch::nn::Module {
  forch::nn::Embedding                        *embed_tokens;
  forch::nn::ModuleList<Qwen3MoeDecoderLayer> *layers;
  RMSNorm                                     *norm;
  Qwen3RotaryEmbedding                        *rotary_emb;

  Qwen3MoeModel(
    const Qwen3Config &config, const engine::EngineConfig &engine_config) {
    embed_tokens = register_loadable("embed_tokens",
      forch::nn::Embedding(
        config.vocab_size, config.hidden_size, forch::DType::BF16));

    auto num_layers = engine_config.n_model_layers > 0
      ? std::min(engine_config.n_model_layers, config.num_hidden_layers)
      : config.num_hidden_layers;

    auto layers_vector = std::vector<std::unique_ptr<Qwen3MoeDecoderLayer>>();
    for (int64_t i = 0; i < num_layers; i++) {
      layers_vector.push_back(
        std::make_unique<Qwen3MoeDecoderLayer>(config, engine_config, i));
    }
    layers = register_loadable("layers",
      forch::nn::ModuleList<Qwen3MoeDecoderLayer>(std::move(layers_vector)));

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

struct Qwen3MoeForCausalLM : public forch::nn::Module {
  Qwen3MoeModel       *model;
  VocabParallelLMHead *lm_head;

  Qwen3MoeForCausalLM(
    const Qwen3Config &config, const engine::EngineConfig &engine_config) {
    model   = register_loadable("model", Qwen3MoeModel(config, engine_config));
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
