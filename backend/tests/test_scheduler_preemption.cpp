#include "../cache/slot-map.hpp"
#include "../server/scheduler.hpp"
#include "recording-sink.hpp"
#include <iostream>
#include <numeric>

using namespace finer;

static void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

// A numerical host runner backed by the real PagedSlotMap address/refcount
// path.
// Each physical row stores a prefix recurrence, so stale/wrong pages or replay
// positions change output tokens even when cursors happen to agree.
class HostRunner : public engine::ModelRunner {
public:
  HostRunner(bool prefix, cache::EvictionPolicy policy, bool recurrent)
      : ModelRunner(4), map_(config(prefix, policy)),
        rows_(map_.storage_rows()), recurrent_(recurrent) {
    graph_infos_ = {{"prefill", 32, 0, true}, {"decode", 4, 1, true}};
  }
  engine::AdmitResult admit_request(int64_t slot,
    const std::vector<int64_t>             &ids,
    std::vector<mm::MmFeatureSpec>          mm) override {
    ModelRunner::admit_request(slot, ids, std::move(mm));
    return {map_.attach_prefix(slot, ids)};
  }
  bool can_run(
    int graph, const std::vector<engine::SlotStep> &slots) const override {
    auto batch =
      engine::pack_tokens(graph_infos_.at(graph), slots, 4, graph == 1);
    auto       before = map_.positions();
    const auto free   = map_.pages().free_blocks();
    const auto result = map_.can_fit(batch.counts);
    require(before == map_.positions() && free == map_.pages().free_blocks(),
      "capacity probe changed cursor or pages");
    return result;
  }
  engine::StepOutput step(
    int graph, const std::vector<engine::SlotStep> &slots) override {
    require(can_run(graph, slots), "scheduler executed an infeasible plan");
    auto batch =
      engine::pack_tokens(graph_infos_.at(graph), slots, 4, graph == 1);
    map_.reserve(batch.counts);
    const auto table   = map_.block_table(4);
    const auto columns = map_.table_columns();
    const auto ids     = table.data();
    auto       row     = [&](int64_t s, int64_t p) {
      return int64_t(ids[s * columns + p / 128]) * 128 + p % 128;
    };
    auto              out = engine::empty_step(4);
    cache::SlotTokens tokens(4);
    for (size_t sid = 0; sid < slots.size(); ++sid) {
      const auto &slot = slots[sid];
      require(
        slot.tokens.empty() || slot.start_position == map_.positions()[sid],
        "cursor mismatch");
      uint64_t value = recurrent_ ? states_[sid]
        : slot.start_position     ? rows_.at(row(sid, slot.start_position - 1))
                                  : 0;
      for (size_t j = 0; j < slot.tokens.size(); ++j) {
        value =
          (value * 17 + uint64_t(slot.tokens[j]) + slot.start_position + j)
          % 100003;
        rows_.at(row(sid, slot.start_position + j)) = value;
      }
      if (!slot.tokens.empty()) {
        out.vals[sid]    = {1.0f, 0.5f, 0.1f};
        out.indices[sid] = {int64_t(value % 997 + 1),
          int64_t((value + 23) % 997 + 1), int64_t((value + 47) % 997 + 1)};
        states_[sid]     = value;
      }
      tokens[sid] = slot.tokens;
    }
    map_.commit(tokens);
    return out;
  }
  void reset_slot(int64_t slot) override {
    map_.reset_slot(slot);
    states_[slot] = 0;
    clear_request_state(slot);
  }
  cache::KVCacheInfo kv_cache_info() const override {
    return map_.info();
  }
  bool reset_prefix_cache() override {
    return map_.reset_prefix_cache();
  }

private:
  static cache::SlotMapConfig config(
    bool prefix, cache::EvictionPolicy policy) {
    cache::SlotMapConfig cfg;
    cfg.slots           = 4;
    cfg.capacity        = 384;
    cfg.block_size      = 128;
    cfg.paged           = true;
    cfg.num_pages       = 3;
    cfg.padding_tokens  = 32;
    cfg.prefix_caching  = prefix;
    cfg.eviction_policy = policy;
    return cfg;
  }
  cache::PagedSlotMap     map_;
  std::vector<uint64_t>   rows_;
  std::array<uint64_t, 4> states_{};
  bool                    recurrent_;
};

static std::unique_ptr<server::Request> request(int id, int length = 0) {
  auto req        = std::make_unique<server::Request>();
  req->request_id = std::to_string(id);
  req->prompt_token_ids.resize(length ? length : 260 + (id % 4) * 7);
  for (size_t i = 0; i < req->prompt_token_ids.size(); ++i)
    req->prompt_token_ids[i] = (i * 13 + id * 31) % 997 + 1;
  req->sampling.temperature = 0.8;
  req->sampling.top_p       = 1;
  req->sampling.top_k       = 3;
  req->sampling.max_tokens  = 32;
  req->sampling.seed        = 123 + id;
  return req;
}

// Utility responses captured for the reset_prefix_cache contract.
struct UtilitySink : tests::RecordingSink {
  std::vector<nlohmann::json>         utility;
  std::vector<server::SchedulerStats> reports;
  void send_outputs(const server::OutputBatch &batch) override {
    RecordingSink::send_outputs(batch);
    if (batch.scheduler_stats)
      reports.push_back(*batch.scheduler_stats);
  }
  void send_utility(uint64_t, const nlohmann::json &result) override {
    utility.push_back(result);
  }
};

// Exercise the production publication path, not only take_stats() directly:
// completion, abort and idle reset all need a final report with no work left.
static void stats_publication() {
  server::ServerConfig cfg;
  cfg.max_model_len   = 384;
  cfg.max_num_request = 4;
  server::Scheduler scheduler(
    cfg, std::make_unique<HostRunner>(true, cache::EvictionPolicy::LRU, false));
  UtilitySink sink;
  scheduler.publish_stats(sink);
  require(sink.reports.empty(), "idle poll must not emit unchanged stats");
  auto one                 = request(101, 5);
  one->sampling.max_tokens = 1;
  scheduler.handle_add(std::move(one), sink);
  scheduler.step(sink);
  require(!scheduler.has_work(), "single-step request did not finish");
  scheduler.publish_stats(sink);
  require(sink.reports.size() == 1 && sink.reports.back().num_running_reqs == 0
      && sink.reports.back().kv_cache_usage == 0
      && sink.reports.back().prefix_cache.requests == 1
      && sink.reports.back().prefix_cache.queries == 5,
    "single-step completion lost prefix counters or final gauges");
  scheduler.publish_stats(sink);
  require(sink.reports.size() == 1, "idle poll duplicated prefix counters");
  scheduler.handle_add(request(102, 260), sink);
  while (scheduler.has_work()) {
    scheduler.step(sink);
    scheduler.publish_stats(sink);
  }
  require(sink.reports.back().num_running_reqs == 0
      && sink.reports.back().kv_cache_usage == 0,
    "multi-step completion left stale usage");
  uint64_t queries = 0;
  for (const auto &report : sink.reports)
    queries += report.prefix_cache.queries;
  require(queries == 265, "queries must be published exactly once");
  for (int i = 0; i < 5; ++i)
    scheduler.handle_add(request(i, 260), sink);
  scheduler.step(sink);
  scheduler.publish_stats(sink);
  require(sink.reports.back().kv_cache_usage > 0, "expected live usage");
  scheduler.handle_abort({"0", "1", "2", "3", "4"}, sink);
  require(
    !scheduler.has_work(), "abort did not drain running and waiting requests");
  const auto before_abort = sink.reports.size();
  scheduler.publish_stats(sink);
  require(sink.reports.size() == before_abort + 1
      && sink.reports.back().num_running_reqs == 0
      && sink.reports.back().num_waiting_reqs == 0
      && sink.reports.back().kv_cache_usage == 0,
    "abort must publish zero gauges even without another step");
  scheduler.handle_utility(9, "reset_prefix_cache", sink);
  scheduler.publish_stats(sink);
  require(sink.utility.back() == true && sink.reports.back().prefix_cache.reset,
    "idle reset marker was not published");
  const auto count = sink.reports.size();
  scheduler.publish_stats(sink);
  require(sink.reports.size() == count, "idle reset marker published twice");
}

// Stats report real page usage and per-interval prefix counters; reset is
// refused while a request holds pages and drops all reuse once idle.
static void stats_and_reset() {
  server::ServerConfig cfg;
  cfg.max_model_len   = 384;
  cfg.max_num_request = 4;
  server::Scheduler scheduler(
    cfg, std::make_unique<HostRunner>(true, cache::EvictionPolicy::LRU, false));
  UtilitySink sink;
  auto        drain = [&] {
    for (int steps = 0; scheduler.has_work(); ++steps) {
      require(steps < 10000, "stats run did not finish");
      scheduler.step(sink);
    }
  };
  auto first        = request(1, 260);
  auto again        = request(1, 260);
  again->request_id = "again";
  scheduler.handle_add(std::move(first), sink);
  scheduler.step(sink);
  auto live = scheduler.take_stats();
  require(live.num_running_reqs == 1 && live.kv_cache_usage > 0
      && live.prefix_cache.requests == 1 && live.prefix_cache.queries == 260
      && live.prefix_cache.hits == 0,
    "first request reports usage and a miss");
  scheduler.handle_utility(7, "reset_prefix_cache", sink);
  require(sink.utility.back() == false, "reset refused while pages are held");
  drain();
  require(scheduler.take_stats().kv_cache_usage == 0,
    "finished requests release every page");
  scheduler.handle_add(std::move(again), sink);
  drain();
  auto hit = scheduler.take_stats();
  require(hit.prefix_cache.requests == 1 && hit.prefix_cache.hits == 256,
    "identical prompt reuses two completed blocks");
  scheduler.handle_utility(8, "reset_prefix_cache", sink);
  require(sink.utility.back() == true, "idle reset succeeds");
  auto third        = request(1, 260);
  third->request_id = "third";
  scheduler.handle_add(std::move(third), sink);
  drain();
  auto after = scheduler.take_stats();
  require(after.prefix_cache.reset && after.prefix_cache.hits == 0,
    "reset is reported and drops reuse");
}

int main() {
  try {
    stats_publication();
    stats_and_reset();
    for (auto policy : {cache::EvictionPolicy::LRU, cache::EvictionPolicy::LFU,
           cache::EvictionPolicy::FIFO})
      for (bool prefix : {false, true}) {
        server::ServerConfig cfg;
        cfg.max_model_len   = 384;
        cfg.max_num_request = 4;
        auto run            = [&](bool concurrent, bool recurrent) {
          server::Scheduler    scheduler(cfg,
                          std::make_unique<HostRunner>(
              prefix && !recurrent, policy, recurrent));
          tests::RecordingSink sink;
          int                  steps = 0;
          for (int i = 0; i < 12; ++i) {
            scheduler.handle_add(request(i), sink);
            if (!concurrent)
              while (scheduler.has_work()) {
                require(++steps < 10000, "serial no progress");
                scheduler.step(sink);
              }
          }
          while (scheduler.has_work()) {
            require(++steps < 10000, "pressure no progress");
            scheduler.step(sink);
            if (steps == 20 && concurrent)
              scheduler.handle_add(request(12, 129), sink);
          }
          require(sink.finished.size() == size_t(concurrent ? 13 : 12),
                       "requests lost");
          if (concurrent)
            require(scheduler.preemptions() > 0, "pressure never exercised");
          return sink.tokens;
        };
        for (bool recurrent : {false, true}) {
          const auto baseline = run(false, recurrent),
                     pressure = run(true, recurrent);
          for (const auto &[id, tokens] : baseline)
            require(tokens == pressure.at(id),
              "preemption changed generated tokens or RNG");
        }
        // Force a younger short prompt to decode while the oldest long prompt
        // crosses a page boundary. Its already emitted tokens must survive.
        for (bool recurrent : {false, true}) {
          std::map<std::string, std::vector<int64_t>> reference;
          for (bool concurrent : {false, true}) {
            server::Scheduler    replay(cfg,
                 std::make_unique<HostRunner>(
                prefix && !recurrent, policy, recurrent));
            tests::RecordingSink output;
            bool                 saw_decode_preemption = false;
            auto                 tick                  = [&] {
              const auto before = replay.preemptions();
              const bool decoding =
                !output.tokens["1"].empty() && !output.finished.contains("1");
              replay.step(output);
              saw_decode_preemption |=
                decoding && replay.preemptions() > before;
            };
            for (int id = 0; id < 2; ++id) {
              auto req                 = request(id, id == 0 ? 200 : 1);
              req->sampling.max_tokens = 128;
              replay.handle_add(std::move(req), output);
              if (!concurrent)
                while (replay.has_work())
                  tick();
            }
            int iterations = 0;
            while (replay.has_work()) {
              require(++iterations < 10000, "decode replay did not finish");
              tick();
            }
            if (!concurrent)
              reference = output.tokens;
            else {
              require(saw_decode_preemption, "decode victim was not exercised");
              require(reference == output.tokens,
                "decode replay changed RNG/history");
            }
          }
        }
        server::Scheduler scheduler(
          cfg, std::make_unique<HostRunner>(prefix, policy, false));
        tests::RecordingSink sink;
        for (int i = 0; i < 6; ++i)
          scheduler.handle_add(request(i), sink);
        for (int i = 0; i < 8; ++i)
          scheduler.step(sink);
        scheduler.handle_abort({"0", "5"}, sink);
        scheduler.handle_add(request(99, 385), sink);
        int steps = 0;
        while (scheduler.has_work()) {
          require(++steps < 10000, "abort pressure no progress");
          scheduler.step(sink);
        }
        require(sink.finished.at("99") == server::FinishReason::Error,
          "oversized request not rejected");
        require(sink.finished.at("0") == server::FinishReason::Abort
            && sink.finished.at("5") == server::FinishReason::Abort,
          "abort lost");
        for (int i = 20; i < 26; ++i)
          scheduler.handle_add(request(i, 384), sink);
        steps = 0;
        while (scheduler.has_work()) {
          require(++steps < 10000, "full-context requests did not finish");
          scheduler.step(sink);
        }
        for (int i = 20; i < 26; ++i) {
          const auto id = std::to_string(i);
          require(sink.finished.at(id) == server::FinishReason::Length,
            "full context must terminate with length");
          require(sink.tokens.at(id).size() == 1,
            "context boundary output duplicated or lost");
        }
      }
    std::cout
      << "PASS scheduler: constrained pool, all eviction policies, stochastic "
         "replay, recurrent reset, arrival, abort, oversized request\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
