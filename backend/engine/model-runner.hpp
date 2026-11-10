#pragma once

/// @file model-runner.hpp
/// @brief ModelRunner — scheduler 与模型之间的执行边界（冻结面）。
///
/// 分界：**scheduler 规划（选图 + 每槽分配），runner 执行**。scheduler 对
/// runner 的全部认知是 graph_infos() 这张编译图元数据表 + 4 个动作入口
/// （admit_request / reset_slot / encode_vision / step）；tensor、图、模型状态
/// 全部在 runner 侧。采样不在这里：step 返回 host 侧 top-k 数组，
/// 由 scheduler 采样（边界按类型划，host 数组不是 tensor）。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <forch/ops/const.hpp>
#include <forch/ops/runtime.hpp>
#include <spdlog/spdlog.h>

#include "../cache/types.hpp"
#include "../utils/mm-feature.hpp"

namespace finer::engine {

/// 一张编译完成的图（纯数据，scheduler 的规划视图）。
struct GraphInfo {
  std::string name;
  /// 总行预算（= 图占位符第一维 = mask 行数）。必填；图是静态形状，
  /// step 按这个宽度跑满。
  int64_t max_total_tokens = 0;
  /// 每槽上限；0 = 图不限制。Qwen3.5 decode 为 1，prefill 最多为 8。
  /// 选图与分配 clamp 由 scheduler 依此字段执行。
  int64_t max_tokens_per_slot = 0;
  /// false = 辅助图（vision / reset），不参与选图。
  bool produces_logits = true;
  /// true = 本图接受**任意**行数（1 … max_total_tokens），由 kernel 按实际
  /// 行数执行（如 chunk-GDR 实现的 prefill 图）；false = 编译期固定的静态形状，
  /// 每步按 max_total_tokens 跑满（尾部 padding）。
  ///
  /// 取舍：**同尺寸下动态图通常慢于静态图**（内核专精程度、dispatch 开销），
  /// scheduler 优先最大化本步有效 token 数；同分时优先静态图，再选较小容量。
  bool dynamic_shape = false;
};

/// 一步里的一个槽。tokens 为空 = 本步无输入，不承诺暂停活跃请求的状态保留。
struct SlotStep {
  /// 该槽已处理的 token 数（= cursor），即本步首 token 的绝对位置。
  int64_t start_position = 0;
  /// 本步该槽的 token。
  std::span<const int64_t> tokens;
};

/// 每槽的 top-k 候选（host 数组），scheduler 据此采样。
/// 与 slots 一一对应；空闲槽为空。indices 是 vocab token id
/// （不是 vals 里的位置——采样取 indices）。
struct StepOutput {
  std::vector<std::vector<float>>   vals;
  std::vector<std::vector<int64_t>> indices;
};

/// 准入结果。
struct AdmitResult {
  /// 已经算好、可以跳过的前缀 token 数（**块对齐**；0 = 无可复用前缀）。
  /// 由 runner 在准入时算一次：它持有 paged 块池（哈希表与 槽→块 映射都在
  /// 缓存内部），所以"能复用多少"是缓存自己的事，scheduler 只需要把它接到
  /// `cursor` 上就够，不必看见块表。
  /// 不变量：`cache_hint_len <= prompt_len - 1`（至少留一个 token 让本步产出
  /// logits）。不支持前缀复用的 runner 返回 0。
  int64_t cache_hint_len = 0;
};

/// Scheduler-facing contract. Concrete runners own graph execution and caches.
class ModelRunner {
public:
  explicit ModelRunner(int64_t slots) : mm_by_slot_(slots) {}
  virtual ~ModelRunner()                      = default;
  ModelRunner(const ModelRunner &)            = delete;
  ModelRunner &operator=(const ModelRunner &) = delete;

  /// Immutable after construction; graph_index is the index in this table.
  const std::vector<GraphInfo> &graph_infos() const {
    return graph_infos_;
  }

  /// Slot lifecycle: admit_request -> step (zero or more times) -> reset_slot.
  /// Admit only a free slot after request validation. Exceptions are fatal:
  /// the runner does not promise rollback or continued execution after failure.
  /// Overrides must register request data (including multimodal features), and
  /// finish any lazy device reset before returning success.
  virtual AdmitResult admit_request(int64_t slot,
    const std::vector<int64_t>             &prompt_ids,
    std::vector<mm::MmFeatureSpec>          mm_features) {
    (void)prompt_ids;
    mm_by_slot_.at(static_cast<size_t>(slot)) = std::move(mm_features);
    return {};
  }

  virtual void encode_vision(const std::vector<mm::VisionJob> &jobs) {
    if (!jobs.empty() && !mm_warned_) {
      mm_warned_ = true;
      spdlog::warn(
        "[mm] 本 runner 未实现 encode_vision：mm 项按占位 token 原样处理");
    }
  }

  /// Release slot bookkeeping and request data. Device state may be reset
  /// lazily when the next request is admitted. Overrides must release request
  /// data (including clear_request_state) and restore host cursor bookkeeping.
  virtual void reset_slot(int64_t slot) = 0;
  /// Pure probe of this exact plan, after admission/prefix attachment. The
  /// scheduler may preempt and replan on false. No state changes may consume
  /// cache capacity between a successful probe and step with the same plan.
  virtual bool can_run(
    int graph_index, const std::vector<SlotStep> &slots) const = 0;

  /// Transfer payloads back to a suspended request before resetting its slot.
  std::vector<mm::MmFeatureSpec> take_request_state(int64_t slot) {
    return std::exchange(mm_by_slot_.at(static_cast<size_t>(slot)), {});
  }
  /// Run one scheduler-selected plan for admitted requests. start_position must
  /// equal the cache cursor, initially cache_hint_len and then advanced by each
  /// consumed token. Empty tokens produce no output or cursor advance; this
  /// interface does not promise active-request pause/resume support. Recurrent
  /// runners require a token for every live request on each executed step:
  /// padding may overwrite device state for slots without input. Admission
  /// must restore new-request state before reuse.
  virtual StepOutput step(
    int graph_index, const std::vector<SlotStep> &slots) = 0;

  /// KV storage capacity and live usage, reported to the frontend.
  virtual cache::KVCacheInfo kv_cache_info() const {
    return {};
  }
  /// Drop reusable prefixes. False while requests still hold cached blocks
  /// (vLLM semantics); runners without prefix reuse succeed trivially.
  virtual bool reset_prefix_cache() {
    return true;
  }

protected:
  const std::vector<mm::MmFeatureSpec> &mm_features(int64_t slot) const {
    return mm_by_slot_.at(static_cast<size_t>(slot));
  }
  void clear_request_state(int64_t slot) {
    mm_by_slot_.at(static_cast<size_t>(slot)).clear();
  }

  std::vector<GraphInfo> graph_infos_;

private:
  std::vector<std::vector<mm::MmFeatureSpec>> mm_by_slot_;
  bool                                        mm_warned_ = false;
};

// Optional packing helpers, not a runner lifecycle. A model may supply any
// other row mapping or execute multiple graphs inside its step().
struct TokenBatch {
  std::vector<int64_t> counts, ids, positions, output_rows;
  int64_t              total = 0, width = 0;
};

inline TokenBatch pack_tokens(const GraphInfo &info,
  const std::vector<SlotStep> &slots, int64_t slot_count, bool slot_ordered) {
  if (slots.size() != static_cast<size_t>(slot_count))
    throw std::runtime_error("runner: slot count mismatch");
  TokenBatch out;
  int64_t    largest = 0;
  for (const auto &slot : slots) {
    auto count = static_cast<int64_t>(slot.tokens.size());
    out.counts.push_back(count);
    out.total += count;
    largest = std::max(largest, count);
  }
  if (out.total > info.max_total_tokens
    || (info.max_tokens_per_slot && largest > info.max_tokens_per_slot))
    throw std::runtime_error(
      "runner: graph token capacity exceeded: " + info.name);
  out.width = info.dynamic_shape ? out.total : info.max_total_tokens;
  if (slot_ordered
    && (out.width % slot_count != 0 || largest > out.width / slot_count))
    throw std::runtime_error(
      "runner: slot rows require equal per-slot capacity");
  out.ids.assign(out.width, 0);
  out.positions.assign(out.width, 0);
  out.output_rows.assign(slot_count, -1);
  int64_t cursor = 0;
  for (size_t sid = 0; sid < slots.size(); ++sid) {
    const auto count = out.counts[sid];
    if (!count)
      continue;
    if (slots[sid].start_position < 0)
      throw std::runtime_error("runner: negative token position");
    const auto offset = slot_ordered
      ? static_cast<int64_t>(sid) * (out.width / slot_count)
      : cursor;
    for (int64_t j = 0; j < count; ++j) {
      out.ids[offset + j]       = slots[sid].tokens[j];
      out.positions[offset + j] = slots[sid].start_position + j;
    }
    out.output_rows[sid] = offset + count - 1;
    cursor += count;
  }
  return out;
}

inline StepOutput empty_step(size_t slots) {
  return {std::vector<std::vector<float>>(slots),
    std::vector<std::vector<int64_t>>(slots)};
}

inline forch::TensorList token_inputs(const TokenBatch &batch) {
  return {
    forch::op::load_from_vector(std::vector<int64_t>(batch.ids), {batch.width})
      .set_name("input_ids"),
    forch::op::load_from_vector(
      std::vector<int64_t>(batch.positions), {batch.width})
      .set_name("position_ids")};
}

template <class Outputs>
StepOutput read_topk(
  const Outputs &outputs, const std::vector<int64_t> &rows, int64_t top_k) {
  const auto &v  = outputs.get("vals");
  const auto &i  = outputs.get("indices");
  const auto *vr = dynamic_cast<const forch::op::Runtime *>(v.get_op());
  const auto *ir = dynamic_cast<const forch::op::Runtime *>(i.get_op());
  if (!vr || !ir || v.dtype() != forch::DType::FP32
    || i.dtype() != forch::DType::INT64 || v.shape().size() != 2
    || i.shape() != v.shape() || v.shape()[1] != top_k)
    throw std::runtime_error("runner: invalid top-k graph outputs");
  auto        out  = empty_step(rows.size());
  const auto *vals = reinterpret_cast<const float *>(vr->get_data());
  const auto *ids  = reinterpret_cast<const int64_t *>(ir->get_data());
  for (size_t sid = 0; sid < rows.size(); ++sid) {
    auto row = rows[sid];
    if (row < 0)
      continue;
    if (row >= v.shape()[0])
      throw std::runtime_error("runner: output row outside graph");
    for (int64_t k = 0; k < top_k; ++k)
      if (!std::isfinite(vals[row * top_k + k]))
        throw std::runtime_error("runner: nonfinite top-k logits");
    out.vals[sid].assign(vals + row * top_k, vals + (row + 1) * top_k);
    out.indices[sid].assign(ids + row * top_k, ids + (row + 1) * top_k);
  }
  return out;
}
} // namespace finer::engine
