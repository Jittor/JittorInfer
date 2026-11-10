#include "forch/ops/batched_mat_mul.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace forch;

static bool almost_equal(
  const float *a, const float *b, size_t n, float tol = 1e-5F) {
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

int test_batched_mat_mul(Backend &backend) {
  // batch=2, M=2, K=3, N=2
  Shape a_shape   = {1, 2, 2, 3};
  Shape b_shape   = {2, 1, 3, 2};
  Shape out_shape = {2, 2, 2, 2};

  std::vector<float> a_data = {
    1,
    2,
    3,
    4,
    5,
    6,
    2,
    0,
    1,
    1,
    3,
    2,
  };
  std::vector<float> b_data = {
    1,
    2,
    3,
    4,
    5,
    6,
    2,
    1,
    0,
    3,
    4,
    5,
  };
  std::vector<float> expected(2UL * 2 * 2 * 2);
  for (int b1 = 0; b1 < 2; ++b1) {
    for (int b2 = 0; b2 < 2; ++b2) {
      for (int m = 0; m < 2; ++m) {
        for (int n = 0; n < 2; ++n) {
          float sum = 0.0F;
          for (int k = 0; k < 3; ++k) {
            // a_data[b1, m, k] * b_data[b2, k, n]
            sum += a_data[b1 * 2 * 3 + m * 3 + k]
              * b_data[b2 * 1 * 3 * 2 + k * 2 + n];
          }
          // expected[b2, b1, m, n]
          expected[(((b2 * 2 + b1) * 2) + m) * 2 + n] = sum;
        }
      }
    }
  }

  auto a = op::create_input(a_shape, DType::FP32).set_name("a_bmm");
  auto b = op::create_input(b_shape, DType::FP32).set_name("b_bmm");
  auto y = op::batched_mat_mul()(a, b);
  y.set_name(std::string("y_bmm"));

  if (y.shape() != out_shape) {
    std::cerr << "batched_mat_mul: output shape mismatch" << std::endl;
    return 1;
  }

  forch::Graph graph_ir({y});
  graph_ir.set_name("test_batched_mat_mul");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "batched_mat_mul: compile/load failed: " << e.what()
              << std::endl;
    return 2;
  }

  auto a_tensor =
    op::load_from_vector(std::move(a_data), a_shape).set_name("a_bmm");
  auto b_tensor =
    op::load_from_vector(std::move(b_data), b_shape).set_name("b_bmm");

  auto outputs = graph.run({a_tensor, b_tensor});
  if (outputs.empty()) {
    return 3;
  }

  auto out_tensor = outputs.get(std::string("y_bmm"));
  if (out_tensor.shape() != out_shape) {
    std::cerr << "batched_mat_mul: runtime output shape mismatch" << std::endl;
    return 4;
  }

  const auto *out_ptr = reinterpret_cast<const float *>(out_tensor.get_data());
  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "batched_mat_mul: values differ" << std::endl;
    return 5;
  }

  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  int r = test_batched_mat_mul(backend);
  if (r != 0) {
    return r;
  }

  std::cout << "Batched mat mul test passed." << std::endl;
  return 0;
}
