#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/tril.hpp"
#include "forch/ops/triu.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
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

// numpy/torch 语义:作用于最后两维。
// tril: 保留 j - i <= diagonal;i - j >= -diagonal 部分置零
// triu: 保留 j - i >= diagonal
template <class T>
static std::vector<T> trilu_reference(const std::vector<T> &data,
  const Shape &shape, int64_t diagonal, bool upper) {
  const auto rank = static_cast<int64_t>(shape.size());
  const auto rows = shape[rank - 2];
  const auto cols = shape[rank - 1];

  size_t batch = 1;
  for (int64_t i = 0; i < rank - 2; ++i) {
    batch *= static_cast<size_t>(shape[i]);
  }

  std::vector<T> out(data.size(), T{});
  for (size_t b = 0; b < batch; ++b) {
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        const auto   diff = j - i;
        const bool   keep = upper ? diff >= diagonal : diff <= diagonal;
        const size_t pos  = b * static_cast<size_t>(rows * cols)
          + static_cast<size_t>(i * cols + j);
        if (keep) {
          out[pos] = data[pos];
        }
      }
    }
  }
  return out;
}

static int run_trilu_case(Backend &backend, bool upper, const Shape &shape,
  const std::vector<float> &data, int64_t diagonal, const char *case_name) {
  const auto input_name  = std::string(case_name) + "_input";
  const auto output_name = std::string(case_name) + "_output";

  auto input = op::create_input(shape, DType::FP32).set_name(input_name);
  auto out   = upper ? op::triu(diagonal)(input) : op::tril(diagonal)(input);
  out.set_name(output_name);

  // 声明期:形状与输入一致
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

  const auto expected = trilu_reference(data, shape, diagonal, upper);

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
    // 选择性置零不改数值,必须精确相等(含 NaN 输入也无法通过:
    // 这里输入保证有限,只需检查逐位一致)
    if (!std::isfinite(ptr[i]) || ptr[i] != expected[i]) {
      std::cerr << case_name << ": value mismatch at " << i
                << ": out=" << ptr[i] << ", expected=" << expected[i]
                << std::endl;
      return 4;
    }
  }

  std::cout << "  [" << case_name << "] ok" << std::endl;
  return 0;
}

static int run_tril_int_case(Backend &backend, const Shape &shape,
  const std::vector<int32_t> &data, int64_t diagonal, const char *case_name) {
  const auto input_name  = std::string(case_name) + "_input";
  const auto output_name = std::string(case_name) + "_output";

  auto input = op::create_input(shape, DType::INT32).set_name(input_name);
  auto out   = op::tril(diagonal)(input);
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

  const auto expected = trilu_reference(data, shape, diagonal, false);

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

static int test_invalid_rank() {
  // GE Tril/Triu 支持 2-8 维:1 维输入须在构建期报错
  auto input = op::create_input(Shape({5}), DType::FP32);
  bool threw = false;
  try {
    auto out = op::tril(0)(input);
    (void)out.shape();
  } catch (const std::exception &) {
    threw = true;
  }
  if (!threw) {
    std::cerr << "tril_invalid_rank: expected exception, got none" << std::endl;
    return 1;
  }
  threw = false;
  try {
    auto out = op::triu(0)(input);
    (void)out.shape();
  } catch (const std::exception &) {
    threw = true;
  }
  if (!threw) {
    std::cerr << "triu_invalid_rank: expected exception, got none" << std::endl;
    return 1;
  }
  std::cout << "  [trilu_invalid_rank] ok" << std::endl;
  return 0;
}

static std::vector<float> iota_data(const Shape &shape, float base = 0.0F) {
  std::vector<float> data(numel(shape));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = base + static_cast<float>(i);
  }
  return data;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  // 2D 方阵,diagonal ∈ {-1, 0, 1}(需求必测)
  {
    const Shape shape = {4, 4};
    const auto  data  = iota_data(shape, 1.0F);
    for (int64_t diagonal : {-1, 0, 1}) {
      const auto tag = "tril_4x4_diag" + std::to_string(diagonal);
      if (int rc =
            run_trilu_case(backend, false, shape, data, diagonal, tag.c_str());
          rc != 0) {
        return rc;
      }
    }
  }

  // 批维 + 非方阵,tril
  {
    const Shape shape = {2, 3, 5};
    const auto  data  = iota_data(shape);
    if (int rc =
          run_trilu_case(backend, false, shape, data, 0, "tril_b23x35_d0");
        rc != 0) {
      return rc;
    }
  }

  // 批维 + 非方阵,triu
  {
    const Shape shape = {2, 2, 4, 6};
    const auto  data  = iota_data(shape);
    if (int rc = run_trilu_case(backend, true, shape, data, 1, "triu_b2246_d1");
        rc != 0) {
      return rc;
    }
    if (int rc =
          run_trilu_case(backend, true, shape, data, -1, "triu_b2246_dneg1");
        rc != 0) {
      return rc;
    }
  }

  // 整数精确对拍
  {
    const Shape                shape = {3, 4};
    const std::vector<int32_t> data  = {
      1, -2, 3, -4, 5, 6, -7, 8, -9, 10, 11, -12};
    if (int rc =
          run_tril_int_case(backend, shape, data, 1, "tril_int32_3x4_d1");
        rc != 0) {
      return rc;
    }
  }

  if (int rc = test_invalid_rank(); rc != 0) {
    return rc;
  }

  std::cout << "Tril/Triu tests passed." << std::endl;
  return 0;
}
