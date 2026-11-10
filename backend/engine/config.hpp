#pragma once

#include "../cache/types.hpp"
#include <cstdint>
#include <forch/backend/forch-ge.hpp>
#include <forch/distributed.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace finer::engine {

struct EngineConfig {
  struct Optimization {
    // use fused kernels for Rotary Positional Embeddings (RoPE)
    bool fuse_rope = false;

    // use fused kernels for attention (e.g., FlashAttention)
    bool fuse_attention = false;

    // Native GDR with BF16 state; false selects FP32 eager recurrence/state.
    bool fused_gdn_attention = true;

    // fused MoE pipeline (gate + sparse dispatch); false = generic eager chain
    bool fuse_moe = false;

    forch::GEBackend::GEBuildOptions::PrecisionMode ge_precision_mode =
      forch::GEBackend::GEBuildOptions::PrecisionMode::Origin;

    Optimization() = default;
    Optimization(const nlohmann::json &j) {
      fuse_rope           = j.value("fuse_rope", false);
      fuse_attention      = j.value("fuse_attention", false);
      fused_gdn_attention = j.value("fused_gdn_attention", true);
      fuse_moe            = j.value("fuse_moe", false);

      if (j.contains("ge_precision_mode")) {
        auto mode_str = j["ge_precision_mode"].get<std::string>();
        if (mode_str == "origin") {
          ge_precision_mode =
            forch::GEBackend::GEBuildOptions::PrecisionMode::Origin;
        } else if (mode_str == "auto_fp16") {
          ge_precision_mode =
            forch::GEBackend::GEBuildOptions::PrecisionMode::AutoToFP16;
        } else if (mode_str == "auto_bf16") {
          ge_precision_mode =
            forch::GEBackend::GEBuildOptions::PrecisionMode::AutoToBF16;
        } else if (mode_str == "force_fp16") {
          ge_precision_mode =
            forch::GEBackend::GEBuildOptions::PrecisionMode::ForceFP16;
        } else {
          spdlog::warn(
            "Unknown ge_precision_mode '{}', using default 'origin'", mode_str);
        }
      }
    }
  } optimization;

  struct KVCacheConfig {
    bool    paged      = true;
    int64_t block_size = 128;
    // Physical data pages per pool, independent of concurrent slot count.
    // Zero chooses the minimum capacity for one full context.
    int64_t               num_pages       = 0;
    cache::EvictionPolicy eviction_policy = cache::EvictionPolicy::LRU;

    bool prefix_caching = true;

    KVCacheConfig() = default;
    explicit KVCacheConfig(const nlohmann::json &j) {
      paged          = j.value("paged", true);
      prefix_caching = j.value("prefix_caching", true);
      block_size     = j.value("block_size", int64_t{128});
      num_pages      = j.value("num_pages", int64_t{0});
      if (num_pages < 0)
        throw std::invalid_argument(
          "engine.kv_cache.num_pages must be nonnegative");
      // The production TND + paged MLA paths have been validated at 128.
      // Reject other sizes until their kernel combinations are verified.
      if (block_size != 128)
        throw std::invalid_argument(
          "engine.kv_cache.block_size: currently only 128 is supported");
      const auto policy = j.value("eviction_policy", std::string("lru"));
      if (policy == "lru")
        eviction_policy = cache::EvictionPolicy::LRU;
      else if (policy == "lfu")
        eviction_policy = cache::EvictionPolicy::LFU;
      else if (policy == "fifo")
        eviction_policy = cache::EvictionPolicy::FIFO;
      else
        throw std::invalid_argument(
          "engine.kv_cache.eviction_policy: expected lru, lfu or fifo");
    }
  } kv_cache;

  // overall context size for all batches
  int64_t n_context = 4096;

  // max concurrent requests (slot pool size)
  int64_t max_num_request = 4;

  // NPU-side TopK for logits reduction. Must be > 0; for full logits set it
  // manually to == vocab_size (runners build topk(top_k_, ...)
  // unconditionally).
  int64_t logits_topk = 2048;

  // total token budget per step, shared by all slots (replaces the old
  // max_batch_size * prefill_chunk_size). Must be > 0.
  int64_t max_num_batched_tokens = 2048;

  // only compute a prefix of layers for debugging
  // -1 means all layers
  int64_t n_model_layers = -1;

  // Tensor-parallel group. JSON gives only the size; bin/ fills in the rank
  // after checking it against the launched MPI world.
  forch::ProcessGroup tp;

  // GE must join the same group the model graphs are built for.
  forch::GEBackend::Options ge_options() const {
    auto options  = forch::GEBackend::Options::from_env();
    options.group = tp;
    return options;
  }

  EngineConfig() = default;
  EngineConfig(const nlohmann::json &j) {
    if (j.contains("kv_cache"))
      kv_cache = KVCacheConfig(j["kv_cache"]);
    if (j.contains("optimization")) {
      optimization = Optimization(j["optimization"]);
    }
    n_context = j.value("n_context", 4096);
    if (n_context <= 0)
      throw std::invalid_argument("EngineConfig: n_context must be positive");
    max_num_request = j.value("max_num_request", 4);
    logits_topk     = j.value("logits_topk", 2048);
    if (logits_topk <= 0)
      throw std::invalid_argument(
        "EngineConfig: logits_topk 必须为正；需要全量 logits 时请手动设置为 "
        ">= vocab_size");
    max_num_batched_tokens = j.value("max_num_batched_tokens", 2048);
    if (max_num_request <= 0 || max_num_batched_tokens < max_num_request)
      throw std::invalid_argument(
        "EngineConfig: max_num_request 必须为正且 max_num_batched_tokens 必须 "
        ">= max_num_request");
    n_model_layers = j.value("n_model_layers", -1);
    tp             = forch::ProcessGroup(0, j.value("tensor_parallel_size", 1));
  }
};

} // namespace finer::engine
