#pragma once

/// @file runner-llama.hpp
/// @brief LlamaRunner: loads Llama weights, creates one FullAttentionCache
/// per executed layer and compiles prefill/decode graphs. Execution,
/// addressing and metadata live in CacheRunner.

#include <filesystem>
#include <fstream>
#include <vector>

#include <forch/huggingface/pretrained.hpp>
#include <nlohmann/json.hpp>

#include "../common/runner.hpp"
#include "config-llama.hpp"
#include "model-llama.hpp"

namespace finer::models::llama {

class LlamaRunner : public CacheRunner {
public:
  LlamaRunner(const engine::EngineConfig &cfg,
    const std::filesystem::path          &model_dir,
    const nlohmann::json & = nlohmann::json::object())
      : CacheRunner(cfg) {
    const LlamaConfig config(
      nlohmann::json::parse(std::ifstream(model_dir / "config.json")));
    // Weight tensors alias the mmapped checkpoint: register it first so it
    // is released after the model.
    auto &loader =
      make_owned<forch::huggingface::Pretrained>(model_dir.string());
    auto &model = make_owned<LlamaForCausalLM>(config, cfg);
    model.load(&loader, "");
    const auto &layers = *model.model->layers;
    // Cache geometry follows this rank's attention heads.
    const auto kv_heads = layers[0]->self_attn->num_key_value_heads;

    create_slot_map(
      cfg.optimization.fuse_attention, cache::AttentionVariant::Full);
    for (size_t i = 0; i < layers.size(); ++i) {
      auto &kv = make_owned<cache::FullAttentionCache>(
        cache::FullAttentionCache::Config{&backend_, storage_rows(),
          {kv_heads, config.head_dim}, {kv_heads, config.head_dim},
          forch::DType::FP16, "L" + std::to_string(i)});
      kv.init();
      caches_.push_back(&kv);
    }
    build_prefill_decode("llama",
      [&](const ForwardInputs &in) { return model.forward(in, caches_); });
  }

private:
  void feed_caches(forch::TensorList &inputs) const override {
    for (const auto *kv : caches_)
      kv->feed(inputs);
  }

  std::vector<cache::FullAttentionCache *> caches_;
};

} // namespace finer::models::llama
