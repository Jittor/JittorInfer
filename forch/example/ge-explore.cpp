#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/mat_mul.hpp"
#include "forch/ops/rms_norm.hpp"

#include "forch/backend/forch-ge.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace forch;

// Global backend reused across all graphs to avoid repeated init costs.
Backend backend = GEBackend(AscendDevice::from_env());

std::vector<float> create_random_buffer(int size, float range = 0.05f) {
  std::vector<float>                    buffer(size);
  std::default_random_engine            generator;
  std::uniform_real_distribution<float> distribution(-range, range);

  // Pre-generate a pool of random values to improve performance for large
  // buffers
  const size_t       n = 1024;
  std::vector<float> rd_pool(n);
  for (auto &x : rd_pool) {
    x = distribution(generator);
  }

  for (int i = 0; i < size; ++i) {
    buffer[i] = rd_pool[i % n];
  }
  return buffer;
}

struct DenseBlock {
  std::vector<float> weight_data;
  std::vector<float> bias_data;
  std::vector<float> gamma_data;
  Tensor             weight;
  Tensor             bias;
  Tensor             gamma;
  std::string        name;

  DenseBlock(int in_features, int out_features, const std::string &name)
      : weight_data(create_random_buffer(in_features * out_features)),
        bias_data(create_random_buffer(out_features)),
        gamma_data(create_random_buffer(out_features, 1.0f)),
        weight(
          op::load_from_buffer(reinterpret_cast<Buffer>(weight_data.data()),
            {in_features, out_features}, DType::FP32)
            .set_name(name + ".weight")),
        bias(op::load_from_buffer(reinterpret_cast<Buffer>(bias_data.data()),
          {1, out_features}, DType::FP32)
               .set_name(name + ".bias")),
        gamma(op::load_from_buffer(reinterpret_cast<Buffer>(gamma_data.data()),
          {out_features}, DType::FP32)
                .set_name(name + ".norm")),
        name(name) {}

  Tensor forward(const Tensor &input) const {
    auto mm   = op::mat_mul()(input, weight).set_name(name + ".mm");
    auto add  = op::add(mm, bias).set_name(name + ".add");
    auto silu = op::silu(add).set_name(name + ".silu");
    auto norm = op::rms_norm(0.00001F)(silu, gamma).set_name(name + ".rmsnorm");
    return norm;
  }
};

struct BigMLP {
  std::vector<DenseBlock> blocks;
  std::string             prefix;

  BigMLP(const std::vector<int> &dims, const std::string &prefix)
      : prefix(prefix) {
    for (size_t i = 1; i < dims.size(); ++i) {
      auto name = prefix + ".layer" + std::to_string(i);
      blocks.emplace_back(dims[i - 1], dims[i], name);
    }
  }

  Tensor forward(Tensor input) {
    Tensor x = std::move(input);
    for (size_t i = 0; i < blocks.size(); ++i) {
      x = blocks[i].forward(x);
    }
    return x;
  }
};

struct RunStats {
  double first_ms   = 0.0;
  double mean_ms    = 0.0;
  double best_ms    = 0.0;
  double worst_ms   = 0.0;
  double compile_s  = 0.0;
  double build_s    = 0.0;
  double load_s     = 0.0;
  size_t iterations = 0;
};

RunStats benchmark_runs(Backend::Graph &graph, const TensorList &inputs,
  size_t repeat, double compile_s, double build_s, double load_s);

RunStats run_single_batch(
  int batch_size, int input_dim, int, const std::vector<int> &dims) {

  auto main_data = op::create_input({batch_size, input_dim}, DType::FP32)
                     .set_name("main.data");

  BigMLP mlp(dims, "main");
  auto   mlp_out = mlp.forward(main_data).set_name("main.output");

  auto         build_start = std::chrono::steady_clock::now();
  forch::Graph main_graph_ir({mlp_out});
  main_graph_ir.set_name("ge-explore-main");
  auto   main_graph = backend.build(main_graph_ir);
  double build_s    = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - build_start)
                     .count();
  auto compile_start = std::chrono::steady_clock::now();
  main_graph.compile();
  double compile_s = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - compile_start)
                       .count();
  auto load_start = std::chrono::steady_clock::now();
  main_graph.load();
  double load_s =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start)
      .count();

  std::vector<float> data_host_data =
    create_random_buffer(input_dim * batch_size, 0.2f);
  auto data_host =
    op::load_from_buffer(reinterpret_cast<Buffer>(data_host_data.data()),
      {batch_size, input_dim}, DType::FP32)
      .set_name("main.data");

  return benchmark_runs(main_graph, {data_host}, 8, compile_s, build_s, load_s);
}

RunStats benchmark_runs(Backend::Graph &graph, const TensorList &inputs,
  size_t repeat, double compile_s, double build_s, double load_s) {
  std::vector<double> durations;
  durations.reserve(repeat);

  for (size_t i = 0; i < repeat; ++i) {
    auto start  = std::chrono::steady_clock::now();
    auto result = graph.run(inputs);
    (void)result;
    auto end = std::chrono::steady_clock::now();
    auto ms  = std::chrono::duration<double, std::milli>(end - start).count();
    durations.push_back(ms);
  }

  RunStats stats;
  stats.iterations = durations.size();
  stats.compile_s  = compile_s;
  stats.build_s    = build_s;
  stats.load_s     = load_s;
  if (!durations.empty()) {
    stats.first_ms = durations.front();
    stats.mean_ms  = std::accumulate(durations.begin(), durations.end(), 0.0)
      / static_cast<double>(durations.size());
    stats.best_ms  = *std::min_element(durations.begin(), durations.end());
    stats.worst_ms = *std::max_element(durations.begin(), durations.end());
  }
  return stats;
}

void print_stats(const std::string &title, const RunStats &stats) {
  std::cout << "\n"
            << title << " (" << stats.iterations
            << " runs, ms): first=" << std::fixed << std::setprecision(3)
            << stats.first_ms << ", mean=" << stats.mean_ms
            << ", best=" << stats.best_ms << ", worst=" << stats.worst_ms
            << ", compile(s)=" << std::setprecision(3) << stats.compile_s
            << ", build(s)=" << stats.build_s << ", load(s)=" << stats.load_s
            << std::endl;
}

int main(int argc, char **argv) {
  int input_dim  = 256;
  int output_dim = 128;
  int batch_size = 4;

  if (argc > 1)
    input_dim = std::atoi(argv[1]);
  if (argc > 2)
    output_dim = std::atoi(argv[2]);
  if (argc > 3)
    batch_size = std::atoi(argv[3]);

  std::cout << "[Setup] Exploring forch-ge backend with a deep MLP.\n"
            << "        Input=" << batch_size << "x" << input_dim
            << ", Output=" << output_dim << std::endl;

  // Seed graph: produces a runtime Tensor used as downstream input.
  auto seed_input = op::create_input({batch_size, input_dim}, DType::FP32)
                      .set_name("seed.input");
  std::vector<float> seed_bias_data = create_random_buffer(input_dim);
  auto               seed_bias =
    op::load_from_buffer(reinterpret_cast<Buffer>(seed_bias_data.data()),
      {1, input_dim}, DType::FP32)
      .set_name("seed.bias");
  auto seed_out =
    op::silu(op::add(seed_input, seed_bias)).set_name("seed.output");
  forch::Graph seed_graph_ir({seed_out});
  seed_graph_ir.set_name("ge-explore-seed");
  auto seed_graph = backend.build(seed_graph_ir);
  seed_graph.compile();
  seed_graph.load();

  // Main graph: accepts Const (host) and Runtime (device) inputs.
  auto main_data = op::create_input({batch_size, input_dim}, DType::FP32)
                     .set_name("main.data");
  auto main_skip = op::create_input({batch_size, input_dim}, DType::FP32)
                     .set_name("main.skip");
  auto fused_in = op::add(main_data, main_skip).set_name("main.fused_in");

  std::vector<int> dims = {
    input_dim, 32768, 32768, 32768, 32768, 32768, output_dim};
  BigMLP mlp(dims, "main");
  auto   mlp_out = mlp.forward(fused_in).set_name("main.output");

  auto         build_start = std::chrono::steady_clock::now();
  forch::Graph main_graph_ir({mlp_out});
  main_graph_ir.set_name("ge-explore-main");
  auto   main_graph   = backend.build(main_graph_ir);
  double main_build_s = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - build_start)
                          .count();
  auto compile_start = std::chrono::steady_clock::now();
  main_graph.compile();
  double main_compile_s = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - compile_start)
                            .count();
  auto load_start = std::chrono::steady_clock::now();
  main_graph.load();
  double main_load_s =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start)
      .count();

  std::cout
    << "\n[Test 1] Mixed input types per run\n"
    << "         - Feed Const Tensor for main.data (host memory).\n"
    << "         - Feed Runtime Tensor for main.skip (output of seed graph).\n"
    << "Purpose  - Validate GE runtime accepts op::Const and op::Runtime\n"
    << "           simultaneously as graph.run inputs.\n";

  std::vector<float> seed_host_data =
    create_random_buffer(input_dim * batch_size, 0.2f);
  auto seed_host =
    op::load_from_buffer(reinterpret_cast<Buffer>(seed_host_data.data()),
      {input_dim, batch_size}, DType::FP32)
      .set_name("seed.input");
  auto seed_outputs = seed_graph.run({seed_host});
  auto runtime_skip = seed_outputs[0].set_name("main.skip");

  std::vector<float> data_host_data =
    create_random_buffer(input_dim * batch_size, 0.2f);
  auto data_host =
    op::load_from_buffer(reinterpret_cast<Buffer>(data_host_data.data()),
      {batch_size, input_dim}, DType::FP32)
      .set_name("main.data");

  auto first_outputs = main_graph.run({data_host, runtime_skip});
  std::cout << "Test 1 output tensor layout: ";
  first_outputs[0].get_op()->dump(std::cout);
  std::cout << std::endl;

  std::cout
    << "\n[Test 2] First-run vs steady-state latency on a loaded graph\n"
    << "Purpose  - Check if the first run after load is slower than later "
       "runs.\n";
  RunStats main_stats = benchmark_runs(main_graph, {data_host, runtime_skip}, 6,
    main_compile_s, main_build_s, main_load_s);
  print_stats("Main graph", main_stats);

  std::cout
    << "\n[Test 3] Alternating runs across two graphs sharing one backend\n"
    << "Purpose  - Observe whether alternating run order introduces overhead\n"
    << "           or state issues.\n";

  const size_t        alternate_iters = 4;
  std::vector<double> alt_seed_ms;
  std::vector<double> alt_main_ms;
  alt_seed_ms.reserve(alternate_iters);
  alt_main_ms.reserve(alternate_iters);

  for (size_t i = 0; i < alternate_iters; ++i) {
    auto seed_start = std::chrono::steady_clock::now();
    auto alt_seed   = seed_graph.run({seed_host});
    auto seed_end   = std::chrono::steady_clock::now();
    alt_seed_ms.push_back(
      std::chrono::duration<double, std::milli>(seed_end - seed_start).count());

    auto alt_skip   = alt_seed[0].set_name("main.skip");
    auto main_start = std::chrono::steady_clock::now();
    auto alt_out    = main_graph.run({data_host, alt_skip});
    auto main_end   = std::chrono::steady_clock::now();
    (void)alt_out;
    alt_main_ms.push_back(
      std::chrono::duration<double, std::milli>(main_end - main_start).count());
  }

  auto mean = [](const std::vector<double> &vals) {
    if (vals.empty())
      return 0.0;
    return std::accumulate(vals.begin(), vals.end(), 0.0)
      / static_cast<double>(vals.size());
  };

  std::cout << "Alternating stats (ms): seed mean=" << std::fixed
            << std::setprecision(3) << mean(alt_seed_ms)
            << ", main mean=" << mean(alt_main_ms) << std::endl;

  std::vector<int> sweep_batches = {1, 16, 64, 256, 1024};
  // 先运行完，最后统一输出结果
  std::vector<std::pair<int, RunStats>> batch_stats;
  for (int b : sweep_batches) {
    RunStats stats = run_single_batch(b, input_dim, output_dim, dims);
    batch_stats.push_back({b, stats});
  }
  std::cout << "\n[Batch Sweep] Throughput vs batch size (8 runs each)\n"
            << "batch | first(ms) | mean(ms) | best(ms) | worst(ms) | "
               "build(s) | compile(s) | load(s)\n";
  for (const auto &pair : batch_stats) {
    const auto &[batch, stats] = pair;
    // clang-format off
    std::cout << std::setw(5) << batch << " | " 
              << std::fixed << std::setprecision(3) 
              << std::setw(9) << stats.first_ms << " | "
              << std::setw(8) << stats.mean_ms << " | " 
              << std::setw(8) << stats.best_ms << " | " 
              << std::setw(9) << stats.worst_ms << " | " 
              << std::setw(8) << stats.build_s << " | " 
              << std::setw(10) <<  stats.compile_s << " | " 
              << std::setw(7) << stats.load_s 
              << std::endl;
    // clang-format on
  }
  std::cout
    << "\nDone. Tests focus on input-type handling and run-time behavior.\n";
  return 0;
}
