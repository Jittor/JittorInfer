#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/reduce.hpp"

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

/// Host reference: reduce `data` over `axes` (already normalized, sorted).
static std::vector<float> reduce_reference(const std::vector<float> &data,
  const Shape &shape, const std::vector<int64_t> &axes, bool keep_dims) {
  std::vector<bool> reduced(shape.size(), false);
  for (auto a : axes) {
    reduced[a] = true;
  }
  Shape                out_shape;
  std::vector<int64_t> dim_map(shape.size(), -1); // input dim -> output dim
  int64_t              oi = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
    if (!reduced[i]) {
      out_shape.push_back(shape[i]);
      dim_map[i] = oi++;
    } else if (keep_dims) {
      out_shape.push_back(1);
      oi++; // keep_dims dims still occupy an output position
    }
  }

  auto out_stride = std::vector<size_t>(out_shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(out_shape.size()) - 2; i >= 0; --i) {
    out_stride[i] = out_stride[i + 1] * static_cast<size_t>(out_shape[i + 1]);
  }

  auto                out = std::vector<float>(numel(out_shape), 0.0F);
  std::vector<size_t> idx(shape.size(), 0);
  size_t              total = numel(shape);
  for (size_t linear = 0; linear < total; ++linear) {
    size_t o = 0;
    for (size_t d = 0; d < shape.size(); ++d) {
      if (dim_map[d] >= 0) {
        o += idx[d] * out_stride[dim_map[d]];
      }
    }
    out[o] += data[linear];

    for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
      if (++idx[d] < static_cast<size_t>(shape[d])) {
        break;
      }
      idx[d] = 0;
    }
  }
  return out;
}

static int run_reduce_sum_case(Backend &backend, const Shape &shape,
  std::vector<float> data, std::vector<int64_t> axes, bool keep_dims,
  const char *case_name) {
  std::cout << "  [" << case_name << "] reduce_sum axes={";
  for (size_t i = 0; i < axes.size(); ++i) {
    std::cout << (i > 0 ? "," : "") << axes[i];
  }
  std::cout << "} keep_dims=" << keep_dims << " ... " << std::flush;

  auto input = op::create_input(shape, DType::FP32)
                 .set_name(std::string(case_name) + "_input");
  auto out = op::reduce_sum(axes, keep_dims)(input);
  out.set_name(std::string(case_name) + "_out");

  // normalize axes for the host reference
  std::vector<int64_t> axes_n;
  for (auto a : axes) {
    axes_n.push_back(a < 0 ? a + static_cast<int64_t>(shape.size()) : a);
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

  auto expected  = reduce_reference(data, shape, axes_n, keep_dims);
  auto in_tensor = op::load_from_vector(std::move(data), shape)
                     .set_name(std::string(case_name) + "_input");
  auto outputs    = graph.run({in_tensor});
  auto out_tensor = outputs.get(std::string(case_name) + "_out");

  if (out_tensor.shape() != out.shape()) {
    std::cerr << case_name << ": shape mismatch at runtime";
    std::cerr << " expected=";
    for (auto d : out.shape())
      std::cerr << d << ",";
    std::cerr << " got=";
    for (auto d : out_tensor.shape())
      std::cerr << d << ",";
    std::cerr << std::endl;
    return 3;
  }
  const float *ptr = reinterpret_cast<const float *>(out_tensor.get_data());
  if (!almost_equal(ptr, expected.data(), expected.size())) {
    std::cerr << case_name << ": values differ" << std::endl;
    size_t n = numel(out.shape());
    for (size_t i = 0; i < n; ++i) {
      if (std::fabs(ptr[i] - expected[i]) > 1e-4F) {
        std::cerr << "first mismatch at " << i << ": out=" << ptr[i]
                  << ", expected=" << expected[i] << std::endl;
        break;
      }
    }
    return 4;
  }

  std::cout << "ok" << std::endl;
  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  {
    Shape              shape = {2, 3};
    std::vector<float> data  = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int                r     = run_reduce_sum_case(
      backend, shape, data, {1}, false, "reduce_sum_2d_dim1");
    if (r != 0)
      return r;
  }
  {
    Shape              shape = {2, 3};
    std::vector<float> data  = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int                r     = run_reduce_sum_case(
      backend, shape, data, {1}, true, "reduce_sum_2d_dim1_keepdims");
    if (r != 0)
      return r;
  }
  {
    Shape              shape = {2, 3};
    std::vector<float> data  = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    int                r     = run_reduce_sum_case(
      backend, shape, data, {0}, false, "reduce_sum_2d_dim0");
    if (r != 0)
      return r;
  }
  {
    Shape              shape = {2, 3};
    std::vector<float> data  = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    // negative axis behaves like dim=1
    int r = run_reduce_sum_case(
      backend, shape, data, {-1}, false, "reduce_sum_neg_dim");
    if (r != 0)
      return r;
  }
  {
    Shape              shape = {2, 3, 4};
    std::vector<float> data(24);
    for (size_t i = 0; i < 24; ++i) {
      data[i] = static_cast<float>(i) - 10.0F;
    }
    int r = run_reduce_sum_case(
      backend, shape, data, {0, 2}, true, "reduce_sum_3d_multi_axis");
    if (r != 0)
      return r;
  }

  std::cout << "ReduceSum tests passed." << std::endl;
  return 0;
}
