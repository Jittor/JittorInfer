#pragma once

/// @file runner-deepseekv2.hpp
/// @brief DeepseekV2Runner: fused attention stores the absorbed MLA latent
/// and RoPE key (MLACache); the eager reference materializes per-head K/V
/// (FullAttentionCache). Execution, addressing and metadata live in
/// CacheRunner.

#include <filesystem>
#include <fstream>
#include <vector>

#include <forch/huggingface/pretrained.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "../common/runner.hpp"
#include "config-deepseek-v2.hpp"
#include "model-deepseek-v2.hpp"

namespace finer::models::deepseek_v2 {

class DeepseekV2Runner : public CacheRunner {
public:
  DeepseekV2Runner(const engine::EngineConfig &cfg,
    const std::filesystem::path               &model_dir,
    const nlohmann::json & = nlohmann::json::object())
      : CacheRunner(cfg) {
    const DeepseekV2Config config(
      nlohmann::json::parse(std::ifstream(model_dir / "config.json")));
    // Weight tensors alias the mmapped checkpoint: register it first so it
    // is released after the model.
    auto &loader =
      make_owned<forch::huggingface::Pretrained>(model_dir.string());
    auto &model = make_owned<DeepseekV2ForCausalLM>(config, cfg);
    model.load(&loader, "");
    const auto &layers = *model.model->layers;
    const auto  heads  = layers[0]->self_attn->num_heads;

    const bool fused    = cfg.optimization.fuse_attention;
    host_input_indexes_ = false;
    create_slot_map(fused,
      fused ? cache::AttentionVariant::MLA : cache::AttentionVariant::Full);
    for (size_t i = 0; i < layers.size(); ++i) {
      const auto name = "L" + std::to_string(i);
      if (fused) {
        auto &mla = make_owned<cache::MLACache>(cache::MLACache::Config{
          &backend_, storage_rows(), config.kv_lora_rank,
          config.qk_rope_head_dim, forch::DType::BF16, "KV_LATENT_" + name});
        mla.init();
        views_.push_back({.kv = nullptr, .mla = &mla});
      } else {
        // K = nope + rope and V = v_head_dim per head.
        auto &kv = make_owned<cache::FullAttentionCache>(
          cache::FullAttentionCache::Config{&backend_, storage_rows(),
            {heads, config.qk_nope_head_dim + config.qk_rope_head_dim},
            {heads, config.v_head_dim}, forch::DType::BF16, name});
        kv.init();
        views_.push_back({.kv = &kv, .mla = nullptr});
      }
    }
    spdlog::info("DeepseekV2Runner: {} layers of {}", layers.size(),
      fused ? "MLACache (absorbed latent + k_pe)"
            : "FullAttentionCache (eager per-head)");
    build_prefill_decode(
      "deepseek_v2",
      [&](const ForwardInputs &in) { return model.forward(in, views_); },
      /*decode_limit_when_eager=*/1);
  }

private:
  void feed_caches(forch::TensorList &inputs) const override {
    for (const auto &view : views_)
      view.mla ? view.mla->feed(inputs) : view.kv->feed(inputs);
  }

  std::vector<DeepseekV2CacheView> views_;
};

} // namespace finer::models::deepseek_v2
