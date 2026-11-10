#include "forch/ops/div.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace forch;

static bool almost_equal(
  const float *a, const float *b, size_t n, float tol = 1e-4F) {
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

static size_t numel(const Shape &shape) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

static int run_div_case(Backend &backend, const Shape &a_shape,
  const Shape &b_shape, std::vector<float> a, std::vector<float> b,
  const char *case_name) {
  std::cout << "  [" << case_name << "] ... " << std::flush;

  auto in_a = op::create_input(a_shape, DType::FP32)
                .set_name(std::string(case_name) + "_a");
  auto in_b = op::create_input(b_shape, DType::FP32)
                .set_name(std::string(case_name) + "_b");
  auto out = op::div()(in_a, in_b);
  out.set_name(std::string(case_name) + "_out");

  // host reference with broadcasting
  auto b_shape_n = b_shape;
  while (b_shape_n.size() < a_shape.size()) {
    b_shape_n.insert(b_shape_n.begin(), 1);
  }
  size_t             n = numel(a_shape);
  std::vector<float> expected(n);
  for (size_t i = 0; i < n; ++i) {
    // row-major index into b (with leading 1-dims)
    size_t rem = i, bo = 0, stride = 1;
    for (int64_t d = static_cast<int64_t>(a_shape.size()) - 1; d >= 0; --d) {
      size_t ai = rem % static_cast<size_t>(a_shape[d]);
      rem /= static_cast<size_t>(a_shape[d]);
      size_t bi = (b_shape_n[d] == 1) ? 0 : ai;
      bo += bi * stride;
      stride *= static_cast<size_t>(b_shape_n[d]);
    }
    expected[i] = a[i] / b[bo];
  }

  Graph graph_ir({out});
  graph_ir.set_name(case_name);
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << case_name << ": compile/load failed: " << e.what()
              << std::endl;
    return 2;
  }

  auto in_a_t = op::load_from_vector(std::move(a), a_shape)
                  .set_name(std::string(case_name) + "_a");
  auto in_b_t = op::load_from_vector(std::move(b), b_shape)
                  .set_name(std::string(case_name) + "_b");
  auto outputs = graph.run({in_a_t, in_b_t});
  auto out_t   = outputs.get(std::string(case_name) + "_out");

  const float *ptr = reinterpret_cast<const float *>(out_t.get_data());
  if (!almost_equal(ptr, expected.data(), expected.size())) {
    std::cerr << case_name << ": values differ" << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::fabs(ptr[i] - expected[i]) > 1e-4F) {
        std::cerr << "first mismatch at " << i << ": out=" << ptr[i]
                  << ", expected=" << expected[i] << std::endl;
        break;
      }
    }
    return 3;
  }

  std::cout << "ok" << std::endl;
  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  {
    Shape              a_shape = {4};
    Shape              b_shape = {4};
    std::vector<float> a       = {10.0F, 20.0F, 30.0F, 40.0F};
    std::vector<float> b       = {2.0F, 4.0F, 5.0F, 8.0F};
    int r = run_div_case(backend, a_shape, b_shape, a, b, "div_1d");
    if (r != 0)
      return r;
  }
  {
    // scalar divisor broadcasts
    Shape              a_shape = {2, 3};
    Shape              b_shape = {1};
    std::vector<float> a       = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> b       = {2.0F};
    int r = run_div_case(backend, a_shape, b_shape, a, b, "div_scalar_bcast");
    if (r != 0)
      return r;
  }
  {
    // negative numerators/denominators
    Shape              a_shape = {3};
    Shape              b_shape = {3};
    std::vector<float> a       = {-6.0F, 7.0F, -8.0F};
    std::vector<float> b       = {3.0F, -2.0F, -4.0F};
    int r = run_div_case(backend, a_shape, b_shape, a, b, "div_negative");
    if (r != 0)
      return r;
  }
  {
    // 2D over 1D broadcast along the last dim
    Shape              a_shape = {2, 3};
    Shape              b_shape = {3};
    std::vector<float> a       = {2.0F, 4.0F, 6.0F, 8.0F, 10.0F, 12.0F};
    std::vector<float> b       = {1.0F, 2.0F, 3.0F};
    int r = run_div_case(backend, a_shape, b_shape, a, b, "div_2d_over_1d");
    if (r != 0)
      return r;
  }

  std::cout << "Div tests passed." << std::endl;
  return 0;
}
