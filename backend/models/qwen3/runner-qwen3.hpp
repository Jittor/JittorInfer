#pragma once

/// @file runner-qwen3.hpp
/// @brief Qwen3Runner: loads Qwen3 / Qwen3-MoE weights, creates one
/// FullAttentionCache per executed layer and compiles prefill/decode graphs.
/// Execution, addressing and metadata live in CacheRunner.

#include <filesystem>
#include <fstream>
#include <memory>
#include <variant>
#include <vector>

#include <forch/huggingface/pretrained.hpp>
#include <nlohmann/json.hpp>

#include "../common/runner.hpp"
#include "config-qwen3.hpp"
#include "model-qwen3-moe.hpp"
#include "model-qwen3.hpp"

namespace finer::models::qwen3 {

class Qwen3Runner : public CacheRunner {
public:
  Qwen3Runner(const engine::EngineConfig &cfg,
    const std::filesystem::path          &model_dir,
    const nlohmann::json & = nlohmann::json::object())
      : CacheRunner(cfg) {
    const Qwen3Config config(
      nlohmann::json::parse(std::ifstream(model_dir / "config.json")));
    // Weight tensors alias the mmapped checkpoint: register it first so it
    // is released after the model.
    auto &loader =
      make_owned<forch::huggingface::Pretrained>(model_dir.string());
    if (config.is_moe)
      model_ = &make_owned<Qwen3MoeForCausalLM>(config, cfg);
    else
      model_ = &make_owned<Qwen3ForCausalLM>(config, cfg);
    std::visit([&](auto *model) { model->load(&loader, ""); }, model_);
    const auto [layers, kv_heads] = std::visit(
      [](auto *model) {
        const auto &layers = *model->model->layers;
        return std::pair{static_cast<int64_t>(layers.size()),
          layers[0]->self_attn->num_key_value_heads};
      },
      model_);

    const bool fused = cfg.optimization.fuse_attention;
    create_slot_map(fused, cache::AttentionVariant::Full);
    for (int64_t i = 0; i < layers; ++i) {
      auto &kv = make_owned<cache::FullAttentionCache>(
        cache::FullAttentionCache::Config{&backend_, storage_rows(),
          {kv_heads, config.head_dim}, {kv_heads, config.head_dim},
          forch::DType::BF16, "L" + std::to_string(i)});
      kv.init();
      caches_.push_back(&kv);
    }
    build_prefill_decode("qwen3", [&](const ForwardInputs &in) {
      return std::visit(
        [&](auto *model) { return model->forward(in, caches_); }, model_);
    });
  }

private:
  void feed_caches(forch::TensorList &inputs) const override {
    for (const auto *kv : caches_)
      kv->feed(inputs);
  }

  std::variant<Qwen3ForCausalLM *, Qwen3MoeForCausalLM *> model_;
  std::vector<cache::FullAttentionCache *>                caches_;
};

} // namespace finer::models::qwen3
