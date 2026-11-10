#include "forch/ops/select.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/cast.hpp"
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

static size_t numel(const Shape &shape) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

/// Build a BOOL condition Const from an int8 0/1 host vector. The vector must
/// outlive the graph run (load_from_buffer is non-owning). A cast would make
/// this a Function node, which cannot be fed as runtime input data.
static Tensor make_bool_cond(
  std::vector<int8_t> &mask, const Shape &shape, const std::string &name) {
  return op::load_from_buffer(forch::Buffer(mask.data()), shape, DType::BOOL)
    .set_name(name);
}

static int run_select_case(Backend &backend, const Shape &cond_shape,
  const Shape &data_shape, std::vector<int8_t> cond, std::vector<float> then_v,
  std::vector<float> else_v, const char *case_name) {
  std::cout << "  [" << case_name << "] ... " << std::flush;

  auto in_cond = op::create_input(cond_shape, DType::BOOL)
                   .set_name(std::string(case_name) + "_cond");
  auto in_then = op::create_input(data_shape, DType::FP32)
                   .set_name(std::string(case_name) + "_then");
  auto in_else = op::create_input(data_shape, DType::FP32)
                   .set_name(std::string(case_name) + "_else");
  auto out = op::select()(in_cond, in_then, in_else);
  out.set_name(std::string(case_name) + "_out");

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

  // host reference: broadcast cond over data shape (numpy-style)
  auto cond_shape_n = cond_shape;
  while (cond_shape_n.size() < data_shape.size()) {
    cond_shape_n.insert(cond_shape_n.begin(), 1);
  }
  size_t             n = numel(data_shape);
  std::vector<float> expected(n);
  for (size_t i = 0; i < n; ++i) {
    size_t rem = i, co = 0, stride = 1;
    for (int64_t d = static_cast<int64_t>(data_shape.size()) - 1; d >= 0; --d) {
      size_t ai = rem % static_cast<size_t>(data_shape[d]);
      rem /= static_cast<size_t>(data_shape[d]);
      size_t bi = (cond_shape_n[d] == 1) ? 0 : ai;
      co += bi * stride;
      stride *= static_cast<size_t>(cond_shape_n[d]);
    }
    expected[i] = cond[co] ? then_v[i] : else_v[i];
  }

  auto cond_t =
    make_bool_cond(cond, cond_shape, std::string(case_name) + "_cond");
  auto then_t = op::load_from_vector(std::move(then_v), data_shape)
                  .set_name(std::string(case_name) + "_then");
  auto else_t = op::load_from_vector(std::move(else_v), data_shape)
                  .set_name(std::string(case_name) + "_else");
  auto outputs = graph.run({cond_t, then_t, else_t});
  auto out_t   = outputs.get(std::string(case_name) + "_out");

  const float *ptr = reinterpret_cast<const float *>(out_t.get_data());
  if (!almost_equal(ptr, expected.data(), expected.size())) {
    std::cerr << case_name << ": values differ" << std::endl;
    for (size_t i = 0; i < n; ++i) {
      if (std::fabs(ptr[i] - expected[i]) > 1e-5F) {
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
    Shape               cond_shape = {6};
    Shape               data_shape = {6};
    std::vector<int8_t> cond       = {1, 0, 1, 0, 1, 0};
    std::vector<float>  then_v     = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float>  else_v     = {-1.0F, -2.0F, -3.0F, -4.0F, -5.0F, -6.0F};
    int                 r          = run_select_case(
      backend, cond_shape, data_shape, cond, then_v, else_v, "select_1d");
    if (r != 0)
      return r;
  }
  {
    // row-wise condition broadcast over the last dim
    Shape               cond_shape = {2, 1};
    Shape               data_shape = {2, 3};
    std::vector<int8_t> cond       = {1, 0};
    std::vector<float>  then_v     = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float>  else_v     = {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F};
    int r = run_select_case(backend, cond_shape, data_shape, cond, then_v,
      else_v, "select_2d_bcast_cond");
    if (r != 0)
      return r;
  }
  {
    // all-true and all-false
    Shape               cond_shape = {3};
    Shape               data_shape = {3};
    std::vector<int8_t> cond       = {1, 1, 1};
    std::vector<float>  then_v     = {9.0F, 8.0F, 7.0F};
    std::vector<float>  else_v     = {0.0F, 0.0F, 0.0F};
    int                 r          = run_select_case(
      backend, cond_shape, data_shape, cond, then_v, else_v, "select_all_true");
    if (r != 0)
      return r;
  }

  std::cout << "Select tests passed." << std::endl;
  return 0;
}
