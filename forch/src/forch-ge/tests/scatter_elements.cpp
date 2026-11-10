#include "forch/ops/scatter_elements.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace forch;

static size_t numel(const Shape &shape) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

static bool floats_match(
  const float *a, const float *b, size_t n, float tol = 1e-4F) {
  for (size_t i = 0; i < n; ++i) {
    if (a[i] < b[i] - tol || a[i] > b[i] + tol) {
      return false;
    }
  }
  return true;
}

/// Host reference: scatter `updates` into `var` at `indices` along `axis`.
/// reduction: "none" replaces, "add"/"mul" accumulate.
static std::vector<float> scatter_reference(const std::vector<float> &var,
  const Shape &shape, const std::vector<int32_t> &indices, int64_t axis,
  const std::vector<float> &updates, const std::string &reduction) {
  auto out       = var;
  auto in_stride = std::vector<size_t>(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    in_stride[i] = in_stride[i + 1] * static_cast<size_t>(shape[i + 1]);
  }
  size_t u = 0;
  size_t total =
    numel(shape); // same numel for indices/updates (torch scatter_)
  for (size_t linear = 0; linear < total; ++linear) {
    size_t rem = linear, pos = 0;
    for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
      size_t idx = rem % static_cast<size_t>(shape[d]);
      rem /= static_cast<size_t>(shape[d]);
      int64_t ndim =
        axis < 0 ? axis + static_cast<int64_t>(shape.size()) : axis;
      if (d == ndim) {
        pos += static_cast<size_t>(indices[linear]) * in_stride[d];
      } else {
        pos += idx * in_stride[d];
      }
    }
    if (reduction == "add") {
      out[pos] += updates[linear];
    } else if (reduction == "mul") {
      out[pos] *= updates[linear];
    } else {
      out[pos] = updates[linear];
    }
    (void)u;
  }
  return out;
}

static int run_scatter_case(Backend &backend, const Shape &shape,
  std::vector<float> var, std::vector<int32_t> indices, int64_t axis,
  std::vector<float> updates, const std::string &reduction,
  const char *case_name) {
  std::cout << "  [" << case_name << "] reduction=" << reduction
            << " axis=" << axis << " ... " << std::flush;

  auto in_var = op::create_input(shape, DType::FP32)
                  .set_name(std::string(case_name) + "_var");
  auto in_idx = op::create_input(shape, DType::INT32)
                  .set_name(std::string(case_name) + "_indices");
  auto in_upd = op::create_input(shape, DType::FP32)
                  .set_name(std::string(case_name) + "_updates");
  auto out = op::scatter_elements(axis, reduction)(in_var, in_idx, in_upd);
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

  auto expected =
    scatter_reference(var, shape, indices, axis, updates, reduction);
  auto var_t = op::load_from_vector(std::move(var), shape)
                 .set_name(std::string(case_name) + "_var");
  auto idx_t = op::load_from_vector(std::move(indices), shape)
                 .set_name(std::string(case_name) + "_indices");
  auto upd_t = op::load_from_vector(std::move(updates), shape)
                 .set_name(std::string(case_name) + "_updates");
  auto outputs = graph.run({var_t, idx_t, upd_t});
  auto out_t   = outputs.get(std::string(case_name) + "_out");

  const float *ptr = reinterpret_cast<const float *>(out_t.get_data());
  if (!floats_match(ptr, expected.data(), expected.size())) {
    std::cerr << case_name << ": values differ" << std::endl;
    size_t n = numel(shape);
    for (size_t i = 0; i < n; ++i) {
      if (ptr[i] != expected[i]) {
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
    // the MoE pattern: counts[t, expert] += 1 — 2D scatter with add
    Shape                shape = {2, 3};
    std::vector<float>   var   = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    std::vector<int32_t> idx   = {2, 0, 1, 0, 2, 1};
    std::vector<float>   upd   = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    int                  r     = run_scatter_case(
      backend, shape, var, idx, /*axis=*/1, upd, "add", "scatter_add_counts");
    if (r != 0)
      return r;
  }
  {
    // replace semantics ("none") — new_x[idxs] = outs
    Shape                shape = {3, 2};
    std::vector<float>   var   = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int32_t> idx   = {2, 0, 1, 1, 0, 2};
    std::vector<float>   upd   = {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F};
    int r = run_scatter_case(backend, shape, var, idx, /*axis=*/0, upd, "none",
      "scatter_replace_axis0");
    if (r != 0)
      return r;
  }
  {
    // 1D replace
    Shape                shape = {4};
    std::vector<float>   var   = {0.0F, 0.0F, 0.0F, 0.0F};
    std::vector<int32_t> idx   = {3, 1, 1, 2};
    std::vector<float>   upd   = {7.0F, 8.0F, 9.0F, 10.0F};
    int                  r     = run_scatter_case(
      backend, shape, var, idx, /*axis=*/0, upd, "none", "scatter_replace_1d");
    if (r != 0)
      return r;
  }

  std::cout << "ScatterElements tests passed." << std::endl;
  return 0;
}
