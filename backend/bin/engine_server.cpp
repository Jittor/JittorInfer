/// @file engine_server.cpp
/// @brief Server entry point — composition root: parse config, load runner,
/// handshake + main loop. 不含模型名、不含 forch::Tensor。

#include "../models/loader.hpp"
#include "../server/scheduler.hpp"
#include "../server/transport.hpp"
#include "config.hpp"

#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>
#include <mpi.h>

#include <nlohmann/json.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

namespace {
std::atomic<bool> g_running{true};

// Idle recv poll: long enough to avoid a busy loop, short enough that
// SIGTERM (g_running) is still observed promptly.
constexpr int kIdlePollMs = 100;
} // namespace

using namespace finer::server;

class Server {
public:
  Server(const ServerConfig                    &cfg,
    std::unique_ptr<finer::engine::ModelRunner> runner,
    std::unique_ptr<Transport>                  transport)
      : transport_(std::move(transport)), scheduler_(cfg, std::move(runner)) {}

  void run() {
    // handshake 已在 main 里完成（编译图之前 begin、之后 complete）
    while (g_running.load()) {
      // Busy: drain without blocking. Idle: poll briefly instead of 0 so
      // the loop doesn't spin at 100% CPU.
      handle_input(scheduler_.has_work() ? 0 : kIdlePollMs);
      scheduler_.step(*transport_);
      scheduler_.publish_stats(*transport_);
    }
  }

private:
  void handle_input(int timeout_ms) {
    for (auto &[req_type, payload] : transport_->recv(timeout_ms)) {
      switch (req_type) {
      case RequestType::Add:
        scheduler_.handle_add(
          std::make_unique<Request>(Request::from_msgpack(payload)),
          *transport_);
        break;
      case RequestType::Abort:
        scheduler_.handle_abort(decode_abort_message(payload), *transport_);
        break;
      case RequestType::Utility: {
        if (!payload.is_array() || payload.size() < 3)
          break;
        uint64_t call_id = payload[1].is_number_unsigned()
          ? payload[1].get<uint64_t>()
          : static_cast<uint64_t>(payload[1].get<int64_t>());
        scheduler_.handle_utility(
          call_id, payload[2].get<std::string>(), *transport_);
        break;
      }
      case RequestType::StartDpWave:
        // vLLM-compat stub: single-wave DP, nothing to do.
        break;
      }
    }
  }

  std::unique_ptr<Transport> transport_;
  Scheduler                  scheduler_;
};

int main(int argc, char *argv[]) {
  // run-server.sh 以 SPDLOG_LEVEL=debug 提升 per-request/per-step 日志；
  // 未设置时保持默认 info
  spdlog::cfg::load_env_levels();

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config_path>\n";
    return 1;
  }

  // Always initialize MPI: without mpirun this is a singleton world of one.
  // GE has worker threads, but only this main thread calls MPI.
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto config_json = nlohmann::json::parse(std::ifstream(argv[1]));
  spdlog::info("finer-engine-server: loaded config from '{}'", argv[1]);

  finer::bin::Config bin_cfg(config_json);
  if (bin_cfg.engine.tp.world_size() != world_size)
    throw std::invalid_argument("engine.tensor_parallel_size ("
      + std::to_string(bin_cfg.engine.tp.world_size())
      + ") must equal the MPI world size (" + std::to_string(world_size) + ")");
  bin_cfg.engine.tp = forch::ProcessGroup(rank, world_size);
  spdlog::info("finer-engine-server: handshake={}, engine_index={}, max_bs={}, "
               "max_model_len={}",
    bin_cfg.server.handshake_address, bin_cfg.server.engine_index,
    bin_cfg.server.max_num_request, bin_cfg.server.max_model_len);

  // 先握手注册（HELLO→INIT），再编译图：前端立刻看到引擎注册，大图编译
  // 只消耗前端 READY 阶段的超时预算（两阶段各自独立计时）
  auto transport = std::make_unique<Transport>(bin_cfg.server, rank);
  transport->handshake_begin();

  spdlog::info("finer-engine-server: loading runner from '{}' "
               "(large graphs may take minutes)",
    bin_cfg.model.string());
  auto runner = finer::models::load_runner(
    bin_cfg.model, bin_cfg.engine, bin_cfg.model_options);

  transport->handshake_complete(runner->kv_cache_info());

  {
    Server server(bin_cfg.server, std::move(runner), std::move(transport));

    // re-register after GE init, which may overwrite signal handlers
    // SIGINT: fast exit, avoid TBE subprocess errors during GE finalize
    std::signal(SIGINT, [](int) {
      spdlog::info("finer-engine-server: caught SIGINT, fast exit");
      std::_Exit(0);
    });
    // SIGTERM: graceful shutdown via flag
    std::signal(SIGTERM, [](int) {
      spdlog::info("finer-engine-server: caught SIGTERM, shutting down...");
      g_running.store(false);
    });

    server.run();
    spdlog::info("finer-engine-server: shutting down");
  }
  // Server and GE resources have been destroyed before finalizing MPI.
  MPI_Finalize();
  return 0;
}
