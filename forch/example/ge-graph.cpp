#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/mat_mul.hpp"

#include "forch/backend/forch-ge.hpp"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>

int main() {
  using namespace forch;

  const int n = 3;
  const int m = 2;
  const int k = 1;

  float buffer_a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  float buffer_c[] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};

  auto A = op::load_from_buffer(
    reinterpret_cast<Buffer>(buffer_a), {n, m}, DType::FP32)
             .set_name("input.A");
  auto B = op::create_input({m, k}, DType::FP32).set_name("input.B");
  auto C = op::load_from_buffer(
    reinterpret_cast<Buffer>(buffer_c), {n, k}, DType::FP32)
             .set_name("input.C");

  auto X = op::mat_mul()(A, B).set_name("Result.A*B");
  auto Y = op::add(X, C).set_name("Result.A*B+C");

  GEBackend    backend_ge(AscendDevice::from_env());
  forch::Graph graph_ir({Y});
  graph_ir.set_name("example-ge-graph");
  auto ge_graph = backend_ge.build(graph_ir);

  std::cout << "=== Compiling Graph ===" << std::endl;
  ge_graph.compile();

  std::cout << "=== Loading Graph ===" << std::endl;
  ge_graph.load();

  // Prepare input data
  std::vector<float> input_b_buffer = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
  auto               input_b        = op::load_from_buffer(
    reinterpret_cast<Buffer>(input_b_buffer.data()), {m, k}, DType::FP32)
                   .set_name("input.B");

  std::cout << "=== Running Graph ===" << std::endl;
  auto output = ge_graph.run({input_b});

  std::cout << "Output Y: ";
  output[0].get_op()->dump(std::cout);
  std::cout << std::endl;

  auto *output_data = reinterpret_cast<const float *>(output[0].get_data());
  for (size_t i = 0; i < n * k; ++i) {
    std::cout << output_data[i] << " ";
  }
  std::cout << std::endl;

  // Perform computation on CPU for verification
  std::vector<float> cpu_result(n * k);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < m; ++l) {
        sum += buffer_a[i * m + l] * input_b_buffer[l * k + j];
      }
      cpu_result[i * k + j] = sum + buffer_c[i * k + j];
    }
  }
  for (size_t i = 0; i < n * k; ++i) {
    float diff = cpu_result[i] - output_data[i];
    std::cout << "CPU Result: " << cpu_result[i]
              << ", GE Result: " << output_data[i] << ", Diff: " << diff
              << std::endl;
    assert(std::abs(diff) < 1e-5);
  }

  return 0;
}