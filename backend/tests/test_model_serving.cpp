// Runs the real scheduler and model without HTTP/tokenization. Serial greedy
// generations are compared with mixed prefill/decode and recompute preemption.
#include "../bin/config.hpp"
#include "../models/loader.hpp"
#include "../server/scheduler.hpp"
#include "recording-sink.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <spdlog/cfg/env.h>

using namespace finer;

struct RankSink : tests::RecordingSink {
  void sync_tokens(std::vector<int64_t> &tokens) override {
    MPI_Bcast(tokens.data(), static_cast<int>(tokens.size()), MPI_INT64_T, 0,
      MPI_COMM_WORLD);
  }
  bool all_have_capacity(bool local) override {
    int value = local, result = 0;
    MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return result != 0;
  }
};

int main(int argc, char **argv) {
  int provided, rank, world;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  try {
    if (argc != 3)
      throw std::invalid_argument(
        "usage: test_model_serving CONFIG REPORT.json");
    spdlog::cfg::load_env_levels();
    auto        json = nlohmann::json::parse(std::ifstream(argv[1]));
    bin::Config cfg(json);
    if (cfg.engine.tp.world_size() != world)
      throw std::runtime_error("TP world mismatch");
    cfg.engine.tp = forch::ProcessGroup(rank, world);
    auto runner = models::load_runner(cfg.model, cfg.engine, cfg.model_options);
    server::Scheduler scheduler(cfg.server, std::move(runner));
    RankSink          sink;
    const auto        test       = json.value("test", nlohmann::json::object());
    const int         count      = test.value("requests", 6),
              generated          = test.value("generated", 12);
    const bool expect_preemption = test.value("expect_preemption", true);
    std::vector<int64_t> seed    = test.value("prompt_ids",
         std::vector<int64_t>{
        1, 15043, 29892, 590, 1024, 338, 263, 2107, 2462, 29889});
    const int            context = cfg.engine.n_context;
    std::vector<int>     lengths = {std::min(33, context / 4),
          std::min(context - 32, context * 3 / 4), std::min(137, context / 3),
          context - 32};
    lengths                      = test.value("lengths", lengths);
    if (seed.empty() || lengths.empty() || count <= 0 || generated <= 0)
      throw std::invalid_argument(
        "test requires nonempty inputs and positive request/generation counts");
    for (auto length : lengths)
      if (length <= 0 || length + generated > context)
        throw std::invalid_argument(
          "test prompt and generation must fit context");
    auto request = [&](int id, const std::string &phase) {
      auto req        = std::make_unique<server::Request>();
      req->request_id = phase + std::to_string(id);
      int length      = lengths[id % lengths.size()];
      for (int i = 0; i < length; ++i)
        req->prompt_token_ids.push_back(seed[i % seed.size()]);
      // Keep requests different before the first block while retaining two
      // repeated prompts to exercise live and released prefix reuse.
      req->prompt_token_ids[0]  = seed[(id / 2) % seed.size()];
      req->sampling.temperature = 0;
      req->sampling.top_k       = 1;
      req->sampling.max_tokens  = generated;
      req->sampling.seed        = 42;
      return req;
    };
    auto drain = [&] {
      int steps = 0;
      while (scheduler.has_work()) {
        if (++steps > 100000)
          throw std::runtime_error("scheduler made no bounded progress");
        scheduler.step(sink);
      }
      return steps;
    };
    nlohmann::json smoke = nlohmann::json::array();
    if (test.contains("smoke_prompts")) {
      int id = 0;
      for (const auto &prompt : test.at("smoke_prompts")) {
        auto       req            = std::make_unique<server::Request>();
        const auto name           = "smoke-" + std::to_string(id++);
        req->request_id           = name;
        req->prompt_token_ids     = prompt.get<std::vector<int64_t>>();
        req->sampling.temperature = 0;
        req->sampling.top_k       = 1;
        req->sampling.max_tokens  = 24;
        req->sampling.seed        = 42;
        scheduler.handle_add(std::move(req), sink);
        drain();
        smoke.push_back(sink.tokens.at(name));
      }
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
      scheduler.handle_add(request(i, "serial-"), sink);
      drain();
    }
    const double serial =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
        .count();
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i)
      scheduler.handle_add(request(i, "mixed-"), sink);
    const auto   steps = drain();
    const double mixed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
        .count();
    nlohmann::json report     = {{"model", cfg.model.string()}, {"tp", world},
          {"context", context}, {"slots", cfg.engine.max_num_request},
          {"pool_pages", cfg.engine.kv_cache.num_pages}, {"serial_seconds", serial},
          {"mixed_seconds", mixed}, {"mixed_steps", steps},
          {"preemptions", scheduler.preemptions()}, {"requests", count},
          {"smoke", smoke}, {"results", nlohmann::json::array()}};
    int            mismatches = 0;
    for (int i = 0; i < count; ++i) {
      const auto &reference = sink.tokens.at("serial-" + std::to_string(i));
      const auto &actual    = sink.tokens.at("mixed-" + std::to_string(i));
      if (reference != actual)
        ++mismatches;
      if (reference.size() != size_t(generated)
        || actual.size() != size_t(generated))
        throw std::runtime_error("missing or duplicated output tokens");
      report["results"].push_back(
        {{"id", i}, {"serial", reference}, {"mixed", actual}});
    }
    report["mismatching_requests"] = mismatches;
    if (rank == 0) {
      std::ofstream(argv[2]) << report.dump(2) << '\n';
      std::cout << "MODEL RESULT " << report.dump() << std::endl;
    }
    if (mismatches)
      throw std::runtime_error(
        "serial/mixed greedy generations diverged; inspect report");
    if (expect_preemption && scheduler.preemptions() == 0)
      throw std::runtime_error("test did not exercise preemption");
    // The test owns the process; skip GE shutdown's asynchronous TBE teardown.
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    std::_Exit(0);
  } catch (const std::exception &e) {
    std::cerr << "rank " << rank << ": " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
