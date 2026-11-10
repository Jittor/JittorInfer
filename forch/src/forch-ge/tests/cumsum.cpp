#include "forch/ops/cumsum.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace forch;

// 沿 axis 的累积和,支持 exclusive/reverse,语义与 numpy.cumsum +
// TF Cumsum 的 exclusive/reverse 组合一致。T 表示元素类型。
template <class T>
static std::vector<T> cumsum_reference(const std::vector<T> &data,
  const Shape &shape, int64_t axis, bool exclusive, bool reverse) {
  const auto rank = static_cast<int64_t>(shape.size());
  if (axis < 0) {
    axis += rank;
  }

  size_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer *= static_cast<size_t>(shape[i]);
  }
  const size_t n     = static_cast<size_t>(shape[axis]);
  size_t       inner = 1;
  for (int64_t i = axis + 1; i < rank; ++i) {
    inner *= static_cast<size_t>(shape[i]);
  }

  std::vector<T> out(data.size());
  for (size_t o = 0; o < outer; ++o) {
    for (size_t in = 0; in < inner; ++in) {
      T running{};
      for (size_t k = 0; k < n; ++k) {
        const size_t idx = reverse ? n - 1 - k : k;
        const size_t pos = o * n * inner + idx * inner + in;
        if (exclusive) {
          out[pos] = running;
          running += data[pos];
        } else {
          running += data[pos];
          out[pos] = running;
        }
      }
    }
  }
  return out;
}

static bool close_enough(
  float actual, float expected, float abs_tol, float rel_tol) {
  // 实际值若为 NaN/inf 直接判失败
  if (!std::isfinite(actual)) {
    return false;
  }
  const auto diff = std::fabs(actual - expected);
  return diff <= abs_tol + rel_tol * std::fabs(expected);
}

static int run_cumsum_case(Backend &backend, const Shape &shape,
  const std::vector<float> &data, int64_t axis, bool exclusive, bool reverse,
  const char *case_name) {
  const auto input_name  = std::string(case_name) + "_input";
  const auto output_name = std::string(case_name) + "_output";

  auto input = op::create_input(shape, DType::FP32).set_name(input_name);
  auto out   = op::cumsum(axis, exclusive, reverse)(input);
  out.set_name(output_name);

  // 声明期:形状与输入一致(负 axis 在 shape() 内归一化校验)
  if (out.shape() != shape) {
    std::cerr << case_name << ": declared shape mismatch" << std::endl;
    return 1;
  }

  forch::Graph graph_ir({out});
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

  const auto expected = cumsum_reference(data, shape, axis, exclusive, reverse);

  auto data_copy = data;
  auto in_tensor =
    op::load_from_vector(std::move(data_copy), shape).set_name(input_name);
  auto outputs    = graph.run({in_tensor});
  auto out_tensor = outputs.get(output_name);
  if (out_tensor.shape() != shape || out_tensor.dtype() != DType::FP32) {
    std::cerr << case_name << ": runtime shape/dtype mismatch" << std::endl;
    return 3;
  }

  const float *ptr = reinterpret_cast<const float *>(out_tensor.get_data());
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!close_enough(ptr[i], expected[i], 1e-4F, 1e-4F)) {
      std::cerr << case_name << ": value mismatch at " << i
                << ": out=" << ptr[i] << ", expected=" << expected[i]
                << std::endl;
      return 4;
    }
  }

  std::cout << "  [" << case_name << "] ok" << std::endl;
  return 0;
}

/// 整数路径的精确对拍(int32 无舍入问题,必须逐位相等)
static int run_cumsum_int_case(Backend &backend, const Shape &shape,
  const std::vector<int32_t> &data, int64_t axis, bool exclusive, bool reverse,
  const char *case_name) {
  const auto input_name  = std::string(case_name) + "_input";
  const auto output_name = std::string(case_name) + "_output";

  auto input = op::create_input(shape, DType::INT32).set_name(input_name);
  auto out   = op::cumsum(axis, exclusive, reverse)(input);
  out.set_name(output_name);
  if (out.dtype() != DType::INT32) {
    std::cerr << case_name << ": declared dtype mismatch" << std::endl;
    return 1;
  }

  forch::Graph graph_ir({out});
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

  const auto expected = cumsum_reference(data, shape, axis, exclusive, reverse);

  auto data_copy = data;
  auto in_tensor =
    op::load_from_vector(std::move(data_copy), shape).set_name(input_name);
  auto outputs    = graph.run({in_tensor});
  auto out_tensor = outputs.get(output_name);
  if (out_tensor.shape() != shape || out_tensor.dtype() != DType::INT32) {
    std::cerr << case_name << ": runtime shape/dtype mismatch" << std::endl;
    return 3;
  }

  const int32_t *ptr = reinterpret_cast<const int32_t *>(out_tensor.get_data());
  for (size_t i = 0; i < expected.size(); ++i) {
    if (ptr[i] != expected[i]) {
      std::cerr << case_name << ": value mismatch at " << i
                << ": out=" << ptr[i] << ", expected=" << expected[i]
                << std::endl;
      return 4;
    }
  }

  std::cout << "  [" << case_name << "] ok" << std::endl;
  return 0;
}

static int test_invalid_axis() {
  // axis 越界(含负轴归一化后越界)必须在图构建期抛 forch::exception
  for (int64_t axis : {int64_t{2}, int64_t{-3}}) {
    auto input = op::create_input(Shape({2, 3}), DType::FP32);
    bool threw = false;
    try {
      auto out = op::cumsum(axis)(input);
      (void)out.shape();
    } catch (const std::exception &) {
      threw = true;
    }
    if (!threw) {
      std::cerr << "cumsum_invalid_axis_" << axis
                << ": expected exception, got none" << std::endl;
      return 1;
    }
  }
  std::cout << "  [cumsum_invalid_axis] ok" << std::endl;
  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  // axis ∈ {0, -1}、exclusive 两种,加 reverse 与组合
  {
    const Shape              shape = {2, 3, 4};
    const std::vector<float> data  = {0.5F, -1.0F, 2.0F, 0.25F, 1.5F, -0.5F,
       3.0F, -2.0F, 0.75F, 1.25F, -1.5F, 2.5F, -0.25F, 0.0F, 1.0F, -3.0F, 2.25F,
       0.5F, -1.25F, 3.5F, 1.75F, -2.5F, 0.125F, 2.0F};
    if (int rc = run_cumsum_case(
          backend, shape, data, 1, false, false, "cumsum_axis1");
        rc != 0) {
      return rc;
    }
    if (int rc = run_cumsum_case(
          backend, shape, data, -1, false, false, "cumsum_axis_neg1");
        rc != 0) {
      return rc;
    }
    if (int rc = run_cumsum_case(
          backend, shape, data, 0, true, false, "cumsum_axis0_exclusive");
        rc != 0) {
      return rc;
    }
    if (int rc = run_cumsum_case(
          backend, shape, data, -1, false, true, "cumsum_axis_neg1_reverse");
        rc != 0) {
      return rc;
    }
    if (int rc = run_cumsum_case(backend, shape, data, 0, true, true,
          "cumsum_axis0_exclusive_reverse");
        rc != 0) {
      return rc;
    }
  }

  // 整数精确对拍
  {
    const Shape                shape = {3, 5};
    const std::vector<int32_t> data  = {
      1, -2, 3, -4, 5, 6, 7, -8, 9, 10, -11, 12, 13, -14, 15};
    if (int rc = run_cumsum_int_case(backend, shape, data, -1, true, false,
          "cumsum_int32_axis_neg1_exclusive");
        rc != 0) {
      return rc;
    }
    if (int rc = run_cumsum_int_case(
          backend, shape, data, 0, false, false, "cumsum_int32_axis0");
        rc != 0) {
      return rc;
    }
  }

  if (int rc = test_invalid_axis(); rc != 0) {
    return rc;
  }

  std::cout << "Cumsum tests passed." << std::endl;
  return 0;
}
