#include "forch/ops/one_hot.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <iostream>
#include <vector>

using namespace forch;

static bool ints_match(const int32_t *a, const int32_t *b, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

/// Host reference for one_hot with axis (normalized).
static std::vector<int32_t> one_hot_reference(
  const std::vector<int32_t> &indices, const Shape &indices_shape,
  int64_t depth, int64_t axis) {
  int64_t rank = static_cast<int64_t>(indices_shape.size());
  int64_t ndim = axis < 0 ? axis + rank + 1 : axis;

  Shape out_shape = indices_shape;
  out_shape.insert(out_shape.begin() + ndim, depth);

  auto out = std::vector<int32_t>(1, 1);
  for (auto d : out_shape) {
    out[0] *= static_cast<int32_t>(d);
  }
  out.assign(static_cast<size_t>(out_shape.size() ? out[0] : 0), 0);

  // out_stride for out_shape
  auto stride = std::vector<int64_t>(out_shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(out_shape.size()) - 2; i >= 0; --i) {
    stride[i] = stride[i + 1] * out_shape[i + 1];
  }
  auto in_stride = std::vector<int64_t>(indices_shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(indices_shape.size()) - 2; i >= 0;
       --i) {
    in_stride[i] = in_stride[i + 1] * indices_shape[i + 1];
  }

  size_t total = 1;
  for (auto d : indices_shape) {
    total *= static_cast<size_t>(d);
  }
  for (size_t linear = 0; linear < total; ++linear) {
    size_t rem = linear, in_pos = 0;
    for (int64_t d = static_cast<int64_t>(indices_shape.size()) - 1; d >= 0;
         --d) {
      size_t idx = rem % static_cast<size_t>(indices_shape[d]);
      rem /= static_cast<size_t>(indices_shape[d]);
      in_pos += idx * in_stride[d];
    }
    int64_t on = indices[linear];
    if (on < 0 || on >= depth) {
      continue; // out of range -> all zeros
    }
    size_t out_pos = 0;
    size_t rem2    = linear;
    for (int64_t d = static_cast<int64_t>(indices_shape.size()) - 1; d >= 0;
         --d) {
      size_t idx = rem2 % static_cast<size_t>(indices_shape[d]);
      rem2 /= static_cast<size_t>(indices_shape[d]);
      int64_t od = d < ndim ? d : d + 1;
      out_pos += idx * stride[od];
    }
    out_pos += static_cast<size_t>(on) * stride[ndim];
    out[out_pos] = 1;
  }
  return out;
}

static int run_one_hot_case(Backend &backend, std::vector<int32_t> indices,
  const Shape &indices_shape, int64_t depth, int64_t axis, const char *name) {
  std::cout << "  [" << name << "] depth=" << depth << " axis=" << axis
            << " ... " << std::flush;

  auto input = op::create_input(indices_shape, DType::INT32)
                 .set_name(std::string(name) + "_input");
  auto out = op::one_hot(depth, axis, DType::INT32)(input);
  out.set_name(std::string(name) + "_out");

  // expected shape
  Shape   expected_shape = indices_shape;
  int64_t rank           = static_cast<int64_t>(indices_shape.size());
  int64_t ndim           = axis < 0 ? axis + rank + 1 : axis;
  expected_shape.insert(expected_shape.begin() + ndim, depth);
  if (out.shape() != expected_shape) {
    std::cerr << name << ": shape mismatch before build" << std::endl;
    return 1;
  }

  Graph graph_ir({out});
  graph_ir.set_name(name);
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << name << ": compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  auto expected  = one_hot_reference(indices, indices_shape, depth, axis);
  auto in_tensor = op::load_from_vector(std::move(indices), indices_shape)
                     .set_name(std::string(name) + "_input");
  auto outputs = graph.run({in_tensor});
  auto out_t   = outputs.get(std::string(name) + "_out");

  const int32_t *ptr = reinterpret_cast<const int32_t *>(out_t.get_data());
  if (!ints_match(ptr, expected.data(), expected.size())) {
    std::cerr << name << ": values differ" << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
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
    std::vector<int32_t> indices = {0, 2, 3, 1};
    int                  r       = run_one_hot_case(
      backend, indices, {4}, /*depth=*/4, /*axis=*/-1, "one_hot_1d");
    if (r != 0)
      return r;
  }
  {
    std::vector<int32_t> indices = {0, 2, 3, 1};
    int                  r       = run_one_hot_case(
      backend, indices, {4}, /*depth=*/4, /*axis=*/0, "one_hot_1d_axis0");
    if (r != 0)
      return r;
  }
  {
    std::vector<int32_t> indices = {1, 2, 0, 3, 2, 1};
    int                  r       = run_one_hot_case(
      backend, indices, {2, 3}, /*depth=*/4, /*axis=*/1, "one_hot_2d");
    if (r != 0)
      return r;
  }
  {
    // out-of-range index must not set any bit
    std::vector<int32_t> indices = {0, 7, 2};
    int                  r       = run_one_hot_case(
      backend, indices, {3}, /*depth=*/4, /*axis=*/-1, "one_hot_oob");
    if (r != 0)
      return r;
  }

  std::cout << "OneHot tests passed." << std::endl;
  return 0;
}
