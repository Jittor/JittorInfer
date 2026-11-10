#pragma once

#include "config-qwen3.5.hpp"
#include "model-qwen3.5.hpp"
#include <forch/nn/linear.hpp>
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

namespace finer::models::qwen35 {

// The checkpoint packs all experts in two tensors, without a .weight suffix.
// Split gate/up and shard the intermediate axis on the host so GE never
// materializes the full checkpoint weights on each TP rank.
struct Qwen35Experts : forch::nn::Module {
  forch::nn::Parameter *gate, *up, *down;
  int64_t               experts, hidden, intermediate;
  forch::ProcessGroup   group;

  Qwen35Experts(const Qwen35Config &cfg, const forch::ProcessGroup &parallel)
      : experts(cfg.num_experts), hidden(cfg.hidden_size),
        intermediate(cfg.moe_intermediate_size), group(parallel) {
    if (cfg.moe_intermediate_size % group.world_size()
      || cfg.shared_expert_intermediate_size % group.world_size())
      throw std::invalid_argument("Qwen3.5 MoE widths must divide TP size");
    auto local = intermediate / group.world_size();
    gate       = register_loadable("gate",
            forch::nn::Parameter({experts, local, hidden}, forch::DType::BF16));
    up         = register_loadable(
      "up", forch::nn::Parameter({experts, local, hidden}, forch::DType::BF16));
    down = register_loadable("down",
      forch::nn::Parameter({experts, hidden, local}, forch::DType::BF16));
  }

  void load(forch::nn::Loader *loader, const std::string &prefix) override {
    struct Shards : forch::nn::Loader {
      forch::Tensor        gate_up, down;
      const Qwen35Experts &owner;
      Shards(forch::nn::Loader *loader, const std::string &prefix,
        const Qwen35Experts &owner)
          : gate_up(loader->load(prefix + ".gate_up_proj")),
            down(loader->load(prefix + ".down_proj")), owner(owner) {
        if (gate_up.shape()
            != forch::Shape{owner.experts, 2 * owner.intermediate, owner.hidden}
          || down.shape()
            != forch::Shape{owner.experts, owner.hidden, owner.intermediate}
          || gate_up.dtype() != forch::DType::BF16
          || down.dtype() != forch::DType::BF16)
          throw forch::exception(
            "Qwen3.5: incompatible packed expert checkpoint at " + prefix);
      }
      forch::Tensor load(const std::string &name) override {
        auto count = owner.intermediate / owner.group.world_size();
        auto start = owner.group.rank_id() * count;
        if (name.ends_with(".down"))
          return forch::nn::WeightShard::shard(down, 2, start, count);
        return forch::nn::WeightShard::shard(gate_up, 1,
          start + (name.ends_with(".up") ? owner.intermediate : 0), count);
      }
    } shards(loader, prefix, *this);
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

struct Qwen35MoEGate : forch::nn::Linear {
  int64_t top_k;
  bool    normalize;

  explicit Qwen35MoEGate(const Qwen35Config &config)
      : forch::nn::Linear(
          config.hidden_size, config.num_experts, false, forch::DType::BF16),
        top_k(config.num_experts_per_tok), normalize(true) {}

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

struct Qwen35MoE : public forch::nn::Module {
  Qwen35Experts     *experts;
  MLP               *shared_expert;
  Qwen35MoEGate     *gate;
  forch::nn::Linear *shared_expert_gate;
  bool               sparse;

  Qwen35MoE(
    const Qwen35Config &config, const engine::EngineConfig &engine_config)
      : sparse(engine_config.optimization.fuse_moe) {
    experts =
      register_loadable("experts", Qwen35Experts(config, engine_config.tp));
    gate               = register_loadable("gate", Qwen35MoEGate(config));
    auto shared_config = config;
    shared_config.intermediate_size = config.shared_expert_intermediate_size;
    shared_expert                   = register_loadable("shared_expert",
                        MLP(shared_config.hidden_size, shared_config.intermediate_size,
                          engine_config.tp, forch::DType::BF16));
    shared_expert_gate              = register_loadable("shared_expert_gate",
                   forch::nn::Linear(config.hidden_size, 1, false, forch::DType::BF16));
  }

  forch::Tensor forward(const forch::Tensor &x) const {
    auto [scores, indices] = gate->forward(x);
    auto routed            = experts->forward(x, scores, indices, sparse);
    auto shared            = forch::op::mul(shared_expert->forward(x),
                 forch::op::sigmoid()(shared_expert_gate->forward(x)));
    return forch::op::add(routed, shared);
  }
};

// ── DecoderLayer：按 layer_types 分发 token mixer ────────────────────────
struct Qwen35MoeDecoderLayer : public forch::nn::Module {
  bool                 is_full_attention;
  Qwen35Attention     *self_attn   = nullptr; // 仅 full 层
  Qwen35GatedDeltaNet *linear_attn = nullptr; // 仅 linear 层
  MLP                 *mlp         = nullptr;
  Qwen35MoE           *moe         = nullptr;
  RMSNormZeroCentered *input_layernorm;
  RMSNormZeroCentered *post_attention_layernorm;

  Qwen35MoeDecoderLayer(const Qwen35Config &config,
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
    if (std::find(
          config.mlp_only_layers.begin(), config.mlp_only_layers.end(), idx)
      == config.mlp_only_layers.end())
      moe = register_loadable("mlp", Qwen35MoE(config, engine_config));
    else
      mlp = register_loadable("mlp",
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
    h                 = moe ? moe->forward(h) : mlp->forward(h);
    return forch::op::add(h, mlp_residual);
  }
};

// ── 文本塔（HF 权重前缀 model.language_model.*）─────────────────────────
struct Qwen35MoeTextModel : public forch::nn::Module {
  forch::nn::Embedding                         *embed_tokens;
  forch::nn::ModuleList<Qwen35MoeDecoderLayer> *layers;
  RMSNormZeroCentered                          *norm;
  Qwen35RotaryEmbedding                        *rotary_emb;

  Qwen35MoeTextModel(
    const Qwen35Config &config, const engine::EngineConfig &engine_config) {
    embed_tokens = register_loadable("embed_tokens",
      forch::nn::Embedding(
        config.vocab_size, config.hidden_size, forch::DType::BF16));

    auto num_layers    = engine_config.n_model_layers > 0
         ? std::min(engine_config.n_model_layers, config.num_hidden_layers)
         : config.num_hidden_layers;
    auto layers_vector = std::vector<std::unique_ptr<Qwen35MoeDecoderLayer>>();
    layers_vector.reserve(static_cast<size_t>(num_layers));
    for (int64_t i = 0; i < num_layers; i++) {
      layers_vector.push_back(
        std::make_unique<Qwen35MoeDecoderLayer>(config, engine_config, i));
    }
    layers = register_loadable("layers",
      forch::nn::ModuleList<Qwen35MoeDecoderLayer>(std::move(layers_vector)));

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
struct Qwen35MoeModel : public forch::nn::Module {
  Qwen35MoeTextModel *language_model;

  Qwen35MoeModel(
    const Qwen35Config &config, const engine::EngineConfig &engine_config) {
    language_model = register_loadable(
      "language_model", Qwen35MoeTextModel(config, engine_config));
  }

  forch::Tensor forward(const ForwardInputs &in,
    const std::vector<Qwen35CacheView> &caches, bool fused_gdn) const {
    return language_model->forward(in, caches, fused_gdn);
  }
};

struct Qwen35MoeForCausalLM : public forch::nn::Module {
  Qwen35MoeModel      *model;
  VocabParallelLMHead *lm_head;

  Qwen35MoeForCausalLM(
    const Qwen35Config &config, const engine::EngineConfig &engine_config) {
    model   = register_loadable("model", Qwen35MoeModel(config, engine_config));
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
