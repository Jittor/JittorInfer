#pragma once

/// @file runner-qwen35.hpp
/// @brief Qwen35Runner: hybrid stack of full-attention layers
/// (FullAttentionCache) and Gated DeltaNet layers (GDNCache: state + conv
/// pools). One-token decode and up-to-eight-token prefill share per-slot
/// state; prefill capacity = max_num_request * k, k <= 8.
///
/// Recurrent state cannot be reused across requests, so prefix reuse is off
/// even when paging is on; full-attention layers still use paged storage.
/// Device state is zeroed at admission; releasing a slot only resets host
/// bookkeeping, avoiding a second reset graph execution.
///
/// Two hard GE constraints (do not change):
///   1) GDN write-back scatters only run when registered as graph outputs,
///      and must be folded into one output: GE's OutputCopy has a fixed cost
///      per output (36 write-backs ~ 1.2ms/step).
///   2) The reset graph must be built before the main graphs: adding and
///      compiling a graph after the main graph in the same session crashes
///      the TBE compiler subprocess.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <forch/huggingface/pretrained.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "../common/runner.hpp"
#include "config-qwen3.5.hpp"
#include "model-qwen3.5-moe.hpp"
#include "model-qwen3.5.hpp"

namespace finer::models::qwen35 {

class Qwen35Runner : public CacheRunner {
public:
  Qwen35Runner(const engine::EngineConfig &cfg,
    const std::filesystem::path           &model_dir,
    const nlohmann::json &model_options = nlohmann::json::object())
      : CacheRunner(cfg) {
    const bool fused_gdn = cfg.optimization.fused_gdn_attention;
    if (fused_gdn
      && (cfg.max_num_batched_tokens % cfg.max_num_request != 0
        || cfg.max_num_batched_tokens / cfg.max_num_request
          > cache::GDNMetadata::kMaxRowsPerSlot))
      throw std::invalid_argument("Qwen3.5 requires max_num_batched_tokens = "
                                  "max_num_request * k, 1 <= k <= 8");
    if (!fused_gdn
      && cfg.optimization.ge_precision_mode
        != forch::GEBackend::GEBuildOptions::PrecisionMode::Origin)
      throw std::invalid_argument(
        "FP32 eager GDR requires ge_precision_mode=origin");

    // gdn_param_dtype (model_options): A_log / gated-norm weights are F32 in
    // 0.8B/27B checkpoints and BF16 in the Qwen3.8 family.
    const Qwen35Config config(
      nlohmann::json::parse(std::ifstream(model_dir / "config.json")),
      model_options);
    // Weight tensors alias the mmapped checkpoint: register it first so it
    // is released after the model.
    auto &loader =
      make_owned<forch::huggingface::Pretrained>(model_dir.string());
    if (config.is_moe)
      model_ = &make_owned<Qwen35MoeForCausalLM>(config, cfg);
    else
      model_ = &make_owned<Qwen35ForCausalLM>(config, cfg);
    std::visit([&](auto *model) { model->load(&loader, ""); }, model_);
    const auto layers = std::visit(
      [](auto *model) {
        return static_cast<int64_t>(
          model->model->language_model->layers->size());
      },
      model_);

    create_slot_map(cfg.optimization.fuse_attention,
      cache::AttentionVariant::Full, /*prefix_reuse=*/false);
    create_gdn_metadata(config.linear_conv_kernel_dim);
    int64_t full_layers = 0;
    for (int64_t i = 0; i < layers; ++i) {
      const auto label = "L" + std::to_string(i);
      // Cache geometry follows this rank's heads.
      if (config.layer_is_full_attention.at(i)) {
        const auto *attention = std::visit(
          [&](auto *model) {
            return (*model->model->language_model->layers)[i]->self_attn;
          },
          model_);
        const cache::HeadGeometry geometry{
          attention->num_key_value_heads, config.head_dim};
        auto &kv = make_owned<cache::FullAttentionCache>(
          cache::FullAttentionCache::Config{&backend_, storage_rows(), geometry,
            geometry, forch::DType::BF16, label});
        kv.init();
        kv_caches_.push_back(&kv);
        views_.push_back({.kv = &kv});
        ++full_layers;
      } else {
        const auto *linear = std::visit(
          [&](auto *model) {
            return (*model->model->language_model->layers)[i]->linear_attn;
          },
          model_);
        auto &gdn = make_owned<cache::GDNCache>(&backend_,
          cache::GDNGeometry{cfg.max_num_request, linear->num_v_heads,
            config.linear_value_head_dim, config.linear_key_head_dim,
            linear->conv_channels, config.linear_conv_kernel_dim},
          fused_gdn ? forch::DType::BF16 : forch::DType::FP32, "gdn_" + label);
        gdn.init();
        gdn_caches_.push_back(&gdn);
        views_.push_back({.kv = nullptr, .gdn = &gdn});
      }
    }
    spdlog::info("Qwen35Runner: {} layers ({} full-attention / {} GDN), "
                 "gdn_param_dtype={}, {}",
      layers, full_layers, layers - full_layers,
      config.gdn_param_dtype == forch::DType::FP32 ? "fp32" : "bf16",
      fused_gdn ? "native fused GDR with BF16 state"
                : "FP32 eager GDR, single-token graphs, origin precision");

    build_reset_graph(); // Must precede the main graphs; see file header.
    const auto forward = [&, fused_gdn](const ForwardInputs &in) {
      return std::visit(
        [&](auto *model) { return model->forward(in, views_, fused_gdn); },
        model_);
    };
    const auto write_back = [&](forch::TensorList &outputs) {
      forch::TensorList parts;
      for (const auto *gdn : gdn_caches_)
        gdn->append_write_back(parts);
      if (!parts.empty())
        outputs.push_back(fold_trackers(parts, "qwen35_wb"));
    };
    // GDN reads one slot-ordered row block per slot, so both graphs are
    // slot-ordered; the full-attention layers follow the decode scheme.
    build_graph({"qwen35_decode", cfg.max_num_request, 1, decode_attention(),
                  cache::PackType::SlotOrdered},
      forward, write_back);
    if (fused_gdn && cfg.max_num_batched_tokens > cfg.max_num_request)
      build_graph({"qwen35_prefill", cfg.max_num_batched_tokens,
                    cfg.max_num_batched_tokens / cfg.max_num_request,
                    decode_attention(), cache::PackType::SlotOrdered},
        forward, write_back);
  }

private:
  void feed_caches(forch::TensorList &inputs) const override {
    for (const auto &view : views_)
      view.kv ? view.kv->feed(inputs) : view.gdn->feed(inputs);
  }

  // A reused slot still holds the previous request's state; zero it so the
  // first token starts from clean history.
  void on_admit(int64_t slot) override {
    if (!reset_graph_)
      return;
    forch::TensorList inputs;
    for (const auto *gdn : gdn_caches_)
      gdn->feed(inputs);
    inputs.push_back(
      forch::op::load_from_vector(std::vector<int64_t>{slot}, {1})
        .set_name("reset_slot_id"));
    reset_graph_->run(inputs);
  }

  // Fold every GDN pool's single-slot zeroing into one output.
  void build_reset_graph() {
    auto slot = forch::op::create_input({1}, forch::DType::INT64,
      /*forch_ge_enable_auto_H2D_copy=*/true)
                  .set_name("reset_slot_id");
    forch::TensorList parts;
    for (auto *gdn : gdn_caches_)
      gdn->append_reset_updates(slot, parts);
    if (parts.empty())
      return;
    reset_graph_ = std::make_unique<forch::GEBackend::GEGraph>(
      backend_.build(forch::Graph({fold_trackers(parts, "slot_reset_tracker")})
                       .set_name("qwen35_slot_reset"),
        {.precision_mode = cfg_.optimization.ge_precision_mode}));
    reset_graph_->compile();
    reset_graph_->load();
    spdlog::info(
      "Qwen35Runner: slot-reset graph compiled ({} pools)", parts.size());
  }

  // Values nobody reads, only there to force execution: fold to one output.
  static forch::Tensor fold_trackers(
    const forch::TensorList &parts, const std::string &tag) {
    // Trackers may mix FP32 recurrent state and BF16 history.
    forch::Tensor acc = parts[0] | forch::op::cast(forch::DType::FP32);
    for (size_t i = 1; i < parts.size(); ++i)
      acc = forch::op::add(acc, parts[i] | forch::op::cast(forch::DType::FP32));
    return acc.set_name(tag);
  }

  std::variant<Qwen35ForCausalLM *, Qwen35MoeForCausalLM *> model_;
  std::vector<cache::FullAttentionCache *>                  kv_caches_;
  std::vector<cache::GDNCache *>                            gdn_caches_;
  std::vector<Qwen35CacheView>                              views_;
  // Unloaded by this destructor, before the base releases the GDN pools.
  std::unique_ptr<forch::GEBackend::GEGraph> reset_graph_;
};

} // namespace finer::models::qwen35
