#pragma once

/// CacheRunner: the execution lifecycle shared by every GE model runner.
///
/// One SlotMap addresses all token pools of the model; one AttentionMetadata
/// (and, for recurrent layers, one GDNMetadata) supplies per-step inputs to
/// every layer. Concrete runners only load weights, create layer caches,
/// build graphs with build_graph() and, if needed, override the hooks.
///
/// Step order: pack -> can_fit -> reserve -> feed -> run -> commit. Pages are
/// reserved before masks and tables are generated because they read the
/// physical mapping; commit publishes prefix blocks only after the graph has
/// written them.

#include "../../cache/attention-metadata.hpp"
#include "../../cache/gdn.hpp"
#include "../../engine/config.hpp"
#include "../../engine/model-runner.hpp"

#include <forch/backend/forch-ge.hpp>
#include <forch/graph.hpp>
#include <forch/ops/cast.hpp>
#include <forch/ops/function.hpp>
#include <forch/ops/input.hpp>
#include <forch/ops/topk.hpp>
#include <spdlog/spdlog.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace finer::models {

/// Graph-time arguments every model forward receives.
struct ForwardInputs {
  forch::Tensor                 input_ids;    // INT64 [T]
  forch::Tensor                 position_ids; // INT64 [T]
  cache::AttentionType          attention_type;
  const cache::AttentionInputs *attention = nullptr; // Null without KV layers.
  const cache::GDNInputs       *gdn       = nullptr; // Null without GDN layers.
};

class CacheRunner : public engine::ModelRunner {
public:
  bool can_run(int                       graph_index,
    const std::vector<engine::SlotStep> &slots) const override {
    return !slot_map_ || slot_map_->can_fit(pack(graph_index, slots).counts);
  }

  engine::AdmitResult admit_request(int64_t slot,
    const std::vector<int64_t>             &prompt_ids,
    std::vector<mm::MmFeatureSpec>          mm_features) override {
    const bool text_only = mm_features.empty();
    auto       result =
      ModelRunner::admit_request(slot, prompt_ids, std::move(mm_features));
    // Multimodal placeholders share token IDs, so their blocks never match.
    if (slot_map_ && text_only && reuses_prefix_)
      result.cache_hint_len = slot_map_->attach_prefix(slot, prompt_ids);
    on_admit(slot);
    return result;
  }

  engine::StepOutput step(
    int graph_index, const std::vector<engine::SlotStep> &slots) override {
    const auto &graph = graphs_.at(graph_index);
    const auto  batch = pack(graph_index, slots);
    if (!batch.total)
      return engine::empty_step(slots.size());
    auto inputs = engine::token_inputs(batch);
    if (slot_map_) {
      slot_map_->reserve(batch.counts);
      attention_->feed(batch.counts, batch.width, graph.pack_type,
        graph.attention_type, inputs);
    }
    if (gdn_)
      gdn_->feed(batch.counts, batch.width, graph.pack_type, inputs);
    feed_caches(inputs);
    auto outputs = graph.executable->run(inputs);
    auto out = engine::read_topk(outputs, batch.output_rows, cfg_.logits_topk);
    if (slot_map_) {
      cache::SlotTokens tokens(slots.size());
      for (size_t sid = 0; sid < slots.size(); ++sid)
        tokens[sid] = slots[sid].tokens;
      slot_map_->commit(tokens);
    }
    return out;
  }

  void reset_slot(int64_t slot) override {
    if (slot_map_)
      slot_map_->reset_slot(slot);
    clear_request_state(slot);
  }

  cache::KVCacheInfo kv_cache_info() const override {
    return slot_map_ ? slot_map_->info() : cache::KVCacheInfo{};
  }
  bool reset_prefix_cache() override {
    return !slot_map_ || slot_map_->reset_prefix_cache();
  }

protected:
  using Forward = std::function<forch::op::TopKOutputs(const ForwardInputs &)>;

  CacheRunner(const engine::EngineConfig &cfg)
      : ModelRunner(cfg.max_num_request),
        backend_(forch::AscendDevice::from_env(), cfg.ge_options()), cfg_(cfg) {
  }

  /// Shared token addressing for KV/MLA layers. fused selects block-aligned
  /// storage; paging and prefix reuse follow engine.kv_cache. GDN state is
  /// per slot and has no prefix reuse, so recurrent models disable it.
  void create_slot_map(
    bool fused, cache::AttentionVariant variant, bool prefix_reuse = true) {
    const auto &kv = cfg_.kv_cache;
    if (kv.paged && !fused)
      spdlog::warn("kv_cache.paged requires fuse_attention; eager attention "
                   "uses dense storage");
    cache::SlotMapConfig map;
    map.slots           = cfg_.max_num_request;
    map.capacity        = cfg_.n_context;
    map.block_size      = fused ? kv.block_size : 0;
    map.paged           = fused && kv.paged;
    map.num_pages       = kv.num_pages;
    map.padding_tokens  = cfg_.max_num_batched_tokens;
    map.eviction_policy = kv.eviction_policy;
    map.prefix_caching  = kv.prefix_caching;
    slot_map_           = cache::make_slot_map(map);
    attention_ =
      std::make_unique<cache::AttentionMetadata>(*slot_map_, variant);
    reuses_prefix_ = prefix_reuse && slot_map_->prefix_caching();
  }
  void create_gdn_metadata(int64_t conv_kernel) {
    gdn_ =
      std::make_unique<cache::GDNMetadata>(cfg_.max_num_request, conv_kernel);
  }

  int64_t storage_rows() const {
    return slot_map_->storage_rows();
  }

  struct GraphSpec {
    std::string          name;
    int64_t              tokens         = 0;
    int64_t              per_slot_limit = 0; // 0 = unlimited.
    cache::AttentionType attention_type = cache::AttentionType::Eager;
    cache::PackType      pack_type      = cache::PackType::Compact;
  };

  /// Build, compile and register one logits graph. extra_outputs lets a
  /// runner attach side-effect trackers (GDN write-back) after forward.
  void build_graph(const GraphSpec &spec, const Forward &forward,
    const std::function<void(forch::TensorList &)> &extra_outputs = {}) {
    ForwardInputs in{forch::op::create_input({spec.tokens}, forch::DType::INT64)
                       .set_name("input_ids"),
      forch::op::create_input({spec.tokens}, forch::DType::INT64)
        .set_name("position_ids"),
      spec.attention_type,
      attention_ ? &attention_->inputs(spec.tokens) : nullptr,
      gdn_ ? &gdn_->inputs(spec.tokens) : nullptr};
    // Values and indices come from one TopK kernel, so rows correspond.
    auto topk    = forward(in);
    auto vals    = topk.values | forch::op::cast(forch::DType::FP32);
    auto indices = topk.indices;
    vals.set_name("vals");
    indices.set_name("indices");
    forch::TensorList outputs{vals, indices};
    if (extra_outputs)
      extra_outputs(outputs);
    // Hybrid submodels use local input indexes; inline host copies stay off
    // for fused attention until GE remaps root indexes.
    auto exec = std::make_unique<forch::GEBackend::GEGraph>(
      backend_.build(forch::Graph(outputs).set_name(spec.name),
        {.precision_mode = cfg_.optimization.ge_precision_mode,
          .enable_host_input_indexes =
            host_input_indexes_ && !cfg_.optimization.fuse_attention}));
    exec->compile();
    exec->load();
    graph_infos_.push_back({spec.name, spec.tokens, spec.per_slot_limit, true});
    graphs_.push_back({spec.attention_type, spec.pack_type, std::move(exec)});
    spdlog::info("{}: graph compiled (tq={}, per-slot={}, attention={})",
      spec.name, spec.tokens, spec.per_slot_limit,
      static_cast<int>(spec.attention_type));
  }

  /// The two standard graphs: compact prefill over max_num_batched_tokens
  /// and one-token-per-slot decode. Fused attention uses TND prefill and
  /// BSND decode; eager uses explicit masks for both.
  void build_prefill_decode(const std::string &prefix, const Forward &forward,
    int64_t decode_limit_when_eager = 0) {
    const bool fused = cfg_.optimization.fuse_attention;
    build_graph(
      {prefix + "_prefill", cfg_.max_num_batched_tokens, 0,
        fused ? cache::AttentionType::TND : cache::AttentionType::Eager,
        cache::PackType::Compact},
      forward);
    build_graph(
      {prefix + "_decode", cfg_.max_num_request,
        fused ? 1 : decode_limit_when_eager, decode_attention(),
        fused ? cache::PackType::SlotOrdered : cache::PackType::Compact},
      forward);
  }

  /// Slot-ordered decode scheme matching the storage.
  cache::AttentionType decode_attention() const {
    if (!cfg_.optimization.fuse_attention)
      return cache::AttentionType::Eager;
    return slot_map_ && slot_map_->paged() ? cache::AttentionType::PagedBSND
                                           : cache::AttentionType::BSND;
  }

  /// Append every layer's persistent pool tensors to the step inputs.
  virtual void feed_caches(forch::TensorList &inputs) const = 0;
  /// Device work needed before a newly admitted request's first token.
  virtual void on_admit(int64_t) {}

  /// Keep weights, the checkpoint mapping and caches alive for the graphs.
  /// Resources are released in reverse registration order after every graph
  /// has been unloaded, so register the loader before the model it backs.
  template <class T>
  T &own(std::shared_ptr<T> resource) {
    auto &ref = *resource;
    resources_.push_back(std::move(resource));
    return ref;
  }
  template <class T, class... Args>
  T &make_owned(Args &&...args) {
    return own(std::make_shared<T>(std::forward<Args>(args)...));
  }

  ~CacheRunner() override {
    graphs_.clear();
    while (!resources_.empty())
      resources_.pop_back();
  }

  forch::GEBackend            backend_;
  const engine::EngineConfig &cfg_;
  // GE does not remap root host indexes for hybrid submodels (MoE graphs).
  bool host_input_indexes_ = true;

private:
  engine::TokenBatch pack(
    int graph_index, const std::vector<engine::SlotStep> &slots) const {
    return engine::pack_tokens(graph_infos_.at(graph_index), slots,
      cfg_.max_num_request,
      graphs_.at(graph_index).pack_type == cache::PackType::SlotOrdered);
  }

  std::unique_ptr<cache::SlotMap>           slot_map_;
  std::unique_ptr<cache::AttentionMetadata> attention_;
  std::unique_ptr<cache::GDNMetadata>       gdn_;
  bool                                      reuses_prefix_ = false;
  std::vector<std::shared_ptr<void>>        resources_;

  struct ExecutionGraph {
    cache::AttentionType                       attention_type;
    cache::PackType                            pack_type;
    std::unique_ptr<forch::GEBackend::GEGraph> executable;
  };
  std::vector<ExecutionGraph> graphs_;
};

} // namespace finer::models
