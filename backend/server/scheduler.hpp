#pragma once

/// @file scheduler.hpp
/// @brief Slot pool + request scheduler. 调度策略与请求语义的唯一归属地：
/// 谁跑、跑多少、何时停、CPU 采样都在这里；对 runner 的全部认知是
/// graph_infos() 一张表 + admit_request/reset_slot/step 三个入口。

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../engine/model-runner.hpp"
#include "../engine/sampler/default.hpp"
#include "config.hpp"
#include "output-sink.hpp"

namespace finer::server {

// Graph TopK pre-filters; CPU applies Temp+TopP on sorted candidates.
// Returns a vocab token ID (from `indices`), not position in `vals`.
inline int64_t sample_logits(const float *vals, const int64_t *indices,
  int64_t n, const SamplingParams &sp, std::mt19937_64 &rng) {
  // Callers skip slots with empty results, so n > 0 here; token 0 is only a
  // defensive fallback.
  if (n <= 0)
    return 0;
  if (sp.temperature <= 0.0f)
    return indices[0];
  // Graph TopK already sorted — truncate to sp.top_k before Temp+TopP
  int64_t                 k = (sp.top_k > 0) ? std::min(n, sp.top_k) : n;
  std::vector<float>      copy(vals, vals + k);
  sampler::Default::Param p;
  p.temperature = sp.temperature;
  p.top_p       = sp.top_p;
  p.top_k       = 0; // graph already reduced
  auto span     = std::span<float>(copy);
  return sampler::random_sample(sampler::Default(p).sample(sampler::Logits(
                                  span, std::span<const int64_t>(indices, k))),
    rng);
}

/// 调度侧的项视图：只有区间（代价 = length），**没有像素**。scheduler 只知道
/// "本步窗口覆盖到了第几项"；像素、网格、哈希都在 runner 那边（准入时单项 move
/// 过去）。项是原子的：区间就是调度边界，绝不在项内切。
struct MmSpan {
  int64_t offset = 0;
  int64_t length = 0;
};

struct Slot {
  std::unique_ptr<Request> req;
  SlotStatus               status = SlotStatus::FREE;
  int64_t                  cursor = 0;
  /// 采样随机源：**每个请求一个**，准入时按 `sampling.seed` 播种。这样同一
  /// seed + 同一请求必得同一串 token，且与批组成无关（每槽每步恰好抽一次）。
  std::mt19937_64 rng;
  /// 本请求的多模态项区间（准入时登记；纯文本为空）。
  std::vector<MmSpan> mm;
  // Replay history includes already emitted generated tokens. Only its final
  // row may produce a new sample; intermediate replay never advances the RNG.
  std::vector<int64_t> history;
  uint64_t             arrival = 0;
  bool                 resumed = false; // Readmitted after preemption.

  bool is_free() const {
    return status == SlotStatus::FREE;
  }

  void assign(std::unique_ptr<Request> r) {
    req    = std::move(r);
    status = SlotStatus::PREFILLING;
    cursor = 0;
    req->generated_ids.clear();
    req->next_token = 0;
    history         = req->prompt_token_ids;
    // 播种在此（handle_add 已把缺省值填成 cfg.seed，所以这里必然 >= 0）。
    rng.seed(static_cast<uint64_t>(req->sampling.seed));
    mm.clear();
  }

  void reset() {
    req.reset();
    mm.clear();
    status = SlotStatus::FREE;
    history.clear();
  }
};

class Scheduler {
public:
  Scheduler(
    const ServerConfig &cfg, std::unique_ptr<engine::ModelRunner> runner)
      : cfg_(cfg), slots_(cfg.max_num_request), runner_(std::move(runner)) {
    graph_infos_ = runner_->graph_infos();
  }

  void handle_add(std::unique_ptr<Request> req, OutputSink &transport);
  void handle_abort(
    const std::vector<std::string> &request_ids, OutputSink &transport);
  void handle_utility(
    uint64_t call_id, const std::string &method, OutputSink &transport);

  bool step(OutputSink &transport);
  /// Current gauges plus prefix counters accumulated since the last call.
  SchedulerStats take_stats();

  /// Flush changes even when the last request just finished or was aborted.
  /// Idle polls without changes must not resend additive prefix counters.
  void publish_stats(OutputSink &transport) {
    if (!stats_pending_)
      return;
    auto batch            = make_output_batch();
    batch.scheduler_stats = take_stats();
    transport.send_outputs(batch);
  }

  bool has_work() const {
    return !waiting_.empty() || has_active_slots();
  }
  uint64_t preemptions() const {
    return preemptions_;
  }

private:
  bool has_active_slots() const {
    for (const auto &s : slots_)
      if (!s.is_free())
        return true;
    return false;
  }

  int64_t allocate_slot();

  // New and suspended requests enter through the same execution-state setup,
  // but the request's sampling/output state is never reinitialized on resume.
  void admit_to_slot(int64_t sid, Slot pending) {
    stats_pending_ = true;
    slots_[sid]    = std::move(pending);
    auto &slot     = slots_[sid];
    slot.status    = SlotStatus::PREFILLING;
    slot.mm.clear();
    // 调度侧先留一份**区间**视图；重负载（像素/网格）随 mm_features move 给
    // runner —— 准入之后 scheduler 不再持有像素（R2）。
    for (const auto &f : slot.req->mm_features)
      slot.mm.push_back({f.mm_offset, f.mm_length});
    if (!slot.mm.empty())
      spdlog::debug(
        "[mm] request_id={} {} 项", slot.req->request_id, slot.mm.size());
    const auto admit = runner_->admit_request(
      sid, slot.history, std::move(slot.req->mm_features));
    // 跳过 runner 返回的已计算前缀（不支持复用时为 0）：prefill 的
    // 分配按 prompt.size() - cursor 计算，天然只喂后缀。
    slots_[sid].cursor = admit.cache_hint_len;
    auto      &stats   = prefix_stats_;
    const auto queries = static_cast<uint64_t>(slot.history.size());
    const auto hits    = static_cast<uint64_t>(admit.cache_hint_len);
    if (slot.resumed) {
      ++stats.preempted_requests;
      stats.preempted_queries += queries;
      stats.preempted_hits += hits;
    } else {
      ++stats.requests;
      stats.queries += queries;
      stats.hits += hits;
    }
  }

  void preempt(int64_t sid) {
    auto &slot = slots_[sid];
    spdlog::debug(
      "[preempt] request_id={} cursor={}", slot.req->request_id, slot.cursor);
    slot.req->mm_features = runner_->take_request_state(sid);
    slot.history          = slot.req->prompt_token_ids;
    slot.history.insert(slot.history.end(), slot.req->generated_ids.begin(),
      slot.req->generated_ids.end());
    runner_->reset_slot(sid);
    auto pending    = std::move(slot);
    slot            = Slot{};
    pending.resumed = true;
    auto it         = std::find_if(waiting_.begin(), waiting_.end(),
              [&](const Slot &other) { return other.arrival > pending.arrival; });
    waiting_.insert(it, std::move(pending));
    ++preemptions_;
    // Drain survivors before readmitting: otherwise the same request can pin
    // its prefix again every step and cause an endless recomputation cycle.
    draining_ = true;
  }

  // 图的总容量和每槽上限必须一起评估，不能跨图合并约束。
  int64_t graph_score(const engine::GraphInfo &g,
    const std::vector<int64_t> &limits, int64_t num_decode) const {
    if (!g.produces_logits || g.max_total_tokens < num_decode)
      return -1;
    int64_t score = 0;
    for (auto limit : limits) {
      if (g.max_tokens_per_slot > 0)
        limit = std::min(limit, g.max_tokens_per_slot);
      score += std::min(limit, g.max_total_tokens - score);
    }
    return score;
  }

  int select_graph(
    const std::vector<int64_t> &limits, int64_t num_decode) const {
    int     best       = -1;
    int64_t best_score = 0;
    for (size_t i = 0; i < graph_infos_.size(); ++i) {
      const auto &g     = graph_infos_[i];
      const auto  score = graph_score(g, limits, num_decode);
      if (score <= 0)
        continue;
      if (best < 0 || score > best_score
        || (score == best_score
          && (g.dynamic_shape != graph_infos_[best].dynamic_shape
              ? !g.dynamic_shape
              : g.max_total_tokens < graph_infos_[best].max_total_tokens))) {
        best       = static_cast<int>(i);
        best_score = score;
      }
    }
    if (best < 0)
      throw std::runtime_error("Scheduler: no graph can serve active slots");
    return best;
  }

  std::vector<int64_t> build_prefill_tq(
    const engine::GraphInfo &graph, const std::vector<int64_t> &limits);

  /// 把本步涉及的多模态项派给 runner。
  void dispatch_vision(const std::vector<engine::SlotStep> &slot_steps);

  void free_slot(int64_t slot_id) {
    stats_pending_ = true;
    slots_[slot_id].reset();
    runner_->reset_slot(slot_id);
  }
  void send_terminal(const std::string &request_id, FinishReason reason,
    std::optional<int64_t> stop_token, OutputSink &transport,
    std::optional<int64_t> last_token = std::nullopt);
  void process_new_token(
    size_t i, OutputSink &transport, OutputBatch &out_batch);

  ServerConfig                         cfg_;
  std::vector<Slot>                    slots_;
  std::deque<Slot>                     waiting_;
  uint64_t                             next_arrival_ = 0;
  uint64_t                             preemptions_  = 0;
  bool                                 draining_     = false;
  PrefixCacheStats                     prefix_stats_;
  bool                                 stats_pending_ = false;
  std::unique_ptr<engine::ModelRunner> runner_;
  std::vector<engine::GraphInfo> graph_infos_; // 构造后不变的规划视图
};

// Implementation

inline void Scheduler::handle_add(
  std::unique_ptr<Request> req, OutputSink &transport) {
  // wire 层解不通的请求在这里收尾：不占槽、不排队（reason 由 protocol
  // 层给出）。
  if (!req->decode_error.empty()) {
    spdlog::warn(
      "[request] request_id={} 拒绝：{}", req->request_id, req->decode_error);
    send_terminal(
      req->request_id, FinishReason::Error, std::nullopt, transport);
    return;
  }

  // Config defaults for params the client didn't specify
  auto &s = req->sampling;
  if (s.temperature < 0)
    s.temperature = cfg_.temperature;
  if (s.top_p <= 0.0f
    || s.top_p > 1.0f) // 0 会穿过 TopP 的 (0,1] 校验、采样时抛
    s.top_p = cfg_.top_p;
  if (s.top_k < 0)
    s.top_k = cfg_.top_k;
  if (s.max_tokens <= 0)
    s.max_tokens = cfg_.max_new_tokens;
  if (s.seed < 0)
    s.seed = static_cast<int64_t>(cfg_.seed);
  // 惩罚类与 min_tokens
  // 只是解析了、没有实现——非默认值时大声说清楚，别静默忽略。
  if (s.frequency_penalty != 0.0f || s.presence_penalty != 0.0f
    || s.repetition_penalty != 1.0f || s.min_tokens > 0)
    spdlog::warn(
      "[sampling] request_id={} frequency/presence/repetition_penalty "
      "与 min_tokens 尚未实现，忽略",
      req->request_id);
  if (req->prompt_token_ids
        .empty()) { // 空 prompt 会永久挂在 PREFILLING（无 logits 可采）
    spdlog::warn("[request] request_id={} 空 prompt，拒绝", req->request_id);
    send_terminal(
      req->request_id, FinishReason::Error, std::nullopt, transport);
    return;
  }
  if (req->prompt_token_ids.size() > static_cast<size_t>(cfg_.max_model_len)) {
    spdlog::warn(
      "[request] request_id={} prompt exceeds context", req->request_id);
    send_terminal(
      req->request_id, FinishReason::Error, std::nullopt, transport);
    return;
  }
  spdlog::debug("[request] start request_id={} prompt_len={} max_tokens={}",
    req->request_id, req->prompt_token_ids.size(), s.max_tokens);
  Slot pending;
  pending.assign(std::move(req));
  stats_pending_  = true;
  pending.arrival = next_arrival_++;
  auto sid        = allocate_slot();
  if (sid < 0 || draining_ || !waiting_.empty()) {
    waiting_.push_back(std::move(pending));
    return;
  }
  admit_to_slot(sid, std::move(pending));
}

inline void Scheduler::handle_abort(
  const std::vector<std::string> &request_ids, OutputSink &transport) {
  for (const auto &rid : request_ids) {
    bool found = false;
    for (auto it = waiting_.begin(); it != waiting_.end(); ++it) {
      if (it->req->request_id == rid) {
        waiting_.erase(it);
        stats_pending_ = true;
        found          = true;
        break;
      }
    }
    if (found) {
      send_terminal(rid, FinishReason::Abort, std::nullopt, transport);
      continue;
    }
    for (auto &slot : slots_) {
      if (!slot.is_free() && slot.req->request_id == rid) {
        send_terminal(
          slot.req->request_id, FinishReason::Abort, std::nullopt, transport);
        free_slot(&slot - slots_.data());
        break;
      }
    }
  }
}

inline void Scheduler::handle_utility(
  uint64_t call_id, const std::string &method, OutputSink &transport) {
  using nlohmann::json;
  // vLLM-compat: single-wave DP and no sleep mode; prefix reset is real.
  json result;
  if (method == "is_sleeping" || method == "is_scheduler_paused")
    result = false;
  else if (method == "get_supported_tasks")
    result = json::array({"generate"});
  else if (method == "reset_prefix_cache") {
    // Like vLLM, fails while running requests hold cached blocks; the
    // running-request reset argument is not supported. Every TP rank sees
    // the same request stream, so all ranks reach the same answer.
    const bool reset = runner_->reset_prefix_cache();
    prefix_stats_.reset |= reset;
    stats_pending_ |= reset;
    result = reset;
  } else
    result = nullptr;
  transport.send_utility(call_id, result);
}

inline SchedulerStats Scheduler::take_stats() {
  stats_pending_ = false;
  SchedulerStats stats;
  stats.num_waiting_reqs = waiting_.size();
  for (const auto &s : slots_)
    if (!s.is_free())
      stats.num_running_reqs++;
  stats.kv_cache_usage = runner_->kv_cache_info().usage;
  stats.prefix_cache   = std::exchange(prefix_stats_, {});
  return stats;
}

inline int64_t Scheduler::allocate_slot() {
  for (int64_t i = 0; i < static_cast<int64_t>(slots_.size()); ++i)
    if (slots_[i].is_free())
      return i;
  return -1;
}

inline void Scheduler::send_terminal(const std::string &request_id,
  FinishReason reason, std::optional<int64_t> stop_token, OutputSink &transport,
  std::optional<int64_t> last_token) {
  OutputBatch batch = make_output_batch();
  Output      out;
  out.request_id    = request_id;
  out.finish_reason = reason;
  if (stop_token.has_value())
    out.stop_reason_token = *stop_token;
  // last_token: for Length, the final generated-but-unsent token
  // (Stop carries its token via stop_reason_token; Abort carries none).
  if (reason == FinishReason::Length && last_token)
    out.new_token_ids.push_back(*last_token);
  batch.outputs.push_back(std::move(out));
  batch.finished_requests.push_back(request_id);
  // 枚举顺序 = FinishReason 声明顺序；以前非 Stop/Length 一律记成 "abort"，
  // 把 Error 也记成了 abort（wire 值一直是对的，只有日志标签错）。
  static const char *kReasonName[] = {
    "stop", "length", "abort", "error", "repetition"};
  spdlog::debug("[request] finish request_id={} reason={}", request_id,
    kReasonName[static_cast<size_t>(reason)]);
  transport.send_outputs(batch);
}

// Shared by the decode fast path and the mixed path: slot i has just
// produced `next_token` — run stop/max_tokens checks, then emit the
// terminal message or stream the token. Frees the slot on terminal
// conditions; never touches the cursor.
inline void Scheduler::process_new_token(
  size_t i, OutputSink &transport, OutputBatch &out_batch) {
  auto &r = *slots_[i].req;
  if (r.sampling.is_stop_token(r.next_token)) {
    send_terminal(r.request_id, FinishReason::Stop, r.next_token, transport);
    free_slot(i);
    return;
  }
  r.generated_ids.push_back(r.next_token);
  if (r.generated_ids.size() >= static_cast<size_t>(r.sampling.max_tokens)) {
    send_terminal(r.request_id, FinishReason::Length, std::nullopt, transport,
      r.next_token);
    free_slot(i);
    return;
  }
  Output out;
  out.request_id = r.request_id;
  out.new_token_ids.push_back(r.next_token);
  out_batch.outputs.push_back(std::move(out));
}

/// 一步跑图前，把"本步 token 窗口碰到的多模态项"派给 runner，让它保证这些项的
/// embedding 可用（runner 按 identifier 缓存 + 幂等；纯文本 runner 是空操作）。
///
/// 项是**原子**的：一项必须在一次视觉图执行里跑完（patch 间有
/// cross-attention），
/// 所以窗口只要碰到项的区间就要整项带上——重复派发无害（缓存命中即跳过），漏派发
/// 会让 runner 拿不到该项的 embedding（encode_vision 的契约由 runner throw
/// 兜住）。
inline void Scheduler::dispatch_vision(
  const std::vector<engine::SlotStep> &slot_steps) {
  std::vector<mm::VisionJob> jobs;
  for (size_t i = 0; i < slot_steps.size(); ++i) {
    if (slots_[i].mm.empty() || slot_steps[i].tokens.empty())
      continue; // 纯文本槽 / 本步空闲
    const int64_t begin = slot_steps[i].start_position;
    const int64_t end =
      begin + static_cast<int64_t>(slot_steps[i].tokens.size());
    mm::VisionJob job{.slot = static_cast<int64_t>(i)};
    for (size_t j = 0; j < slots_[i].mm.size(); ++j) {
      const auto &sp = slots_[i].mm[j];
      if (sp.offset < end && begin < sp.offset + sp.length) // 区间相交
        job.item_indices.push_back(static_cast<int32_t>(j));
    }
    if (!job.item_indices.empty())
      jobs.push_back(std::move(job));
  }
  if (jobs.empty())
    return; // 纯文本 / 本步没碰到任何项：连调用都不发生
  spdlog::debug("[mm] dispatch encode_vision: {} job(s)", jobs.size());
  runner_->encode_vision(jobs);
}

inline std::vector<int64_t> Scheduler::build_prefill_tq(
  const engine::GraphInfo &graph, const std::vector<int64_t> &limits) {
  // 预算与每槽上限都来自图声明（GraphInfo）：decode 槽各占 1 token，其余按
  // 剩余 prompt 与每槽上限分配给 prefill 槽。RGD-only 的 qwen3.5 每槽上限 = 1
  // → prefill 也逐 token。
  std::vector<int64_t> per_slot_tq(slots_.size(), 0);
  const int64_t        cap    = graph.max_tokens_per_slot; // 0 = 不限
  int64_t              budget = graph.max_total_tokens;

  for (size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].status == SlotStatus::DECODING) {
      per_slot_tq[i] = 1;
      budget -= 1;
    }
  }

  for (size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].status != SlotStatus::PREFILLING)
      continue;
    int64_t give = std::min(budget, limits[i]);
    if (cap > 0)
      give = std::min(give, cap);
    per_slot_tq[i] = std::max<int64_t>(give, 0);
    budget -= per_slot_tq[i];
  }

  return per_slot_tq;
}

inline bool Scheduler::step(OutputSink &transport) {
  // ── 1. Terminate slots at the engine context limit (n_context) ──
  for (size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].status != SlotStatus::PREFILLING
      && slots_[i].status != SlotStatus::DECODING)
      continue;
    if (slots_[i].cursor >= cfg_.max_model_len) {
      spdlog::info("[request] cursor {} reached max_model_len {}, finishing",
        slots_[i].cursor, cfg_.max_model_len);
      send_terminal(slots_[i].req->request_id, FinishReason::Length,
        std::nullopt, transport);
      free_slot(i);
    }
  }

  // ── 2. Slot allocation: waiting → free slots ──
  if (!has_active_slots())
    draining_ = false;
  while (!draining_ && !waiting_.empty()) {
    auto sid = allocate_slot();
    if (sid < 0)
      break;
    admit_to_slot(sid, std::move(waiting_.front()));
    waiting_.pop_front();
  }

  // ── 3. Check for work ──
  int num_prefill = 0, num_decode = 0;
  for (auto &s : slots_) {
    if (s.status == SlotStatus::PREFILLING)
      num_prefill++;
    if (s.status == SlotStatus::DECODING)
      num_decode++;
  }
  if (num_prefill == 0 && num_decode == 0)
    return false;

  auto t0 = std::chrono::high_resolution_clock::now();

  // 每步的规划产物：graph_index + 每槽 token 分配（空闲槽 tokens 为空）。
  std::vector<engine::SlotStep>     slot_steps(slots_.size());
  std::vector<std::vector<int64_t>> chunks(slots_.size());
  std::vector<int64_t>              limits(slots_.size(), 0);
  int                               graph_index = -1;
  for (;;) {
    num_decode = 0;
    std::fill(limits.begin(), limits.end(), 0);
    std::fill(slot_steps.begin(), slot_steps.end(), engine::SlotStep{});
    for (size_t i = 0; i < slots_.size(); ++i) {
      const auto &slot = slots_[i];
      if (slot.status == SlotStatus::DECODING) {
        limits[i] = 1;
        ++num_decode;
      } else if (slot.status == SlotStatus::PREFILLING) {
        const auto remaining =
          static_cast<int64_t>(slot.history.size()) - slot.cursor;
        limits[i] = std::max<int64_t>(
          0, std::min(remaining, cfg_.max_model_len - slot.cursor));
      }
    }
    graph_index = select_graph(limits, num_decode);
    const auto per_slot_tokens =
      build_prefill_tq(graph_infos_[graph_index], limits);
    for (size_t i = 0; i < slots_.size(); ++i) {
      const auto tq = per_slot_tokens[i];
      if (tq == 0)
        continue;
      auto &r = *slots_[i].req;
      if (slots_[i].status == SlotStatus::PREFILLING) {
        chunks[i].assign(slots_[i].history.begin() + slots_[i].cursor,
          slots_[i].history.begin() + slots_[i].cursor + tq);
      } else {
        chunks[i] = {r.next_token};
      }
      slot_steps[i] = {slots_[i].cursor, chunks[i]};
    }
    if (transport.all_have_capacity(runner_->can_run(graph_index, slot_steps)))
      break;
    int64_t victim = -1, active = 0;
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (slots_[i].is_free())
        continue;
      ++active;
      if (victim < 0 || slots_[i].arrival > slots_[victim].arrival)
        victim = static_cast<int64_t>(i);
    }
    if (active <= 1)
      throw std::logic_error(
        "Scheduler: single request cannot fit configured KV pool");
    preempt(victim);
  }

  // 多模态：本步窗口覆盖到的项，先让 runner 保证 embedding 可用（幂等；
  // 纯文本请求恒为空，零开销）。
  dispatch_vision(slot_steps);

  auto t_build   = std::chrono::high_resolution_clock::now();
  auto result    = runner_->step(graph_index, slot_steps);
  stats_pending_ = true;
  auto t_graph   = std::chrono::high_resolution_clock::now();

  // ── 6. Advance cursors (all non-free slots by their allocated tq) ──
  std::vector<int64_t> per_slot_tq(slots_.size(), 0);
  for (size_t i = 0; i < slots_.size(); ++i) {
    per_slot_tq[i] = static_cast<int64_t>(slot_steps[i].tokens.size());
    if (!slots_[i].is_free())
      slots_[i].cursor += per_slot_tq[i];
  }

  // ── 7. Sample + process outputs ──
  auto out_batch = make_output_batch();

  // Each rank samples, then rank0's tokens (or -1 for no token) win.
  std::vector<int64_t> sampled(slots_.size(), -1);
  for (size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].is_free() || result.vals[i].empty())
      continue;
    auto &r = *slots_[i].req;
    if (slots_[i].status == SlotStatus::PREFILLING
      && slots_[i].cursor < static_cast<int64_t>(slots_[i].history.size()))
      continue;
    const auto &v  = result.vals[i];
    const auto &ix = result.indices[i];
    sampled[i] =
      sample_logits(v.data(), ix.data(), v.size(), r.sampling, slots_[i].rng);
  }
  transport.sync_tokens(sampled);

  for (size_t i = 0; i < slots_.size(); ++i) {
    if (sampled[i] < 0)
      continue;
    slots_[i].req->next_token = sampled[i];

    // Transition PREFILLING → DECODING
    if (slots_[i].status == SlotStatus::PREFILLING)
      slots_[i].status = SlotStatus::DECODING;

    process_new_token(i, transport, out_batch);
  }

  if (!out_batch.outputs.empty())
    transport.send_outputs(out_batch);

  auto t_end    = std::chrono::high_resolution_clock::now();
  int  total_tq = 0;
  for (const auto tq : per_slot_tq)
    total_tq += static_cast<int>(tq);
  spdlog::debug(
    "[step] graph={} pf:{} dec:{} tq:{} sel:{:.1f} graph:{:.1f} post:{:.1f} "
    "total:{:.1f}ms",
    graph_infos_[static_cast<size_t>(graph_index)].name, num_prefill,
    num_decode, total_tq,
    std::chrono::duration<double, std::milli>(t_build - t0).count(),
    std::chrono::duration<double, std::milli>(t_graph - t_build).count(),
    std::chrono::duration<double, std::milli>(t_end - t_graph).count(),
    std::chrono::duration<double, std::milli>(t_end - t0).count());
  return true;
}

} // namespace finer::server
