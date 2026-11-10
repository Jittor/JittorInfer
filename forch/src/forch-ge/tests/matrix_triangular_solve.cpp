#include "forch/ops/matrix_triangular_solve.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <random>
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

/// 把输出批坐标映射到广播输入的线性偏移(in_batch 各维可为 1)
static int64_t broadcast_offset(const std::vector<int64_t> &coords,
  const Shape &out_batch, const Shape &in_batch) {
  const auto out_rank = static_cast<int64_t>(out_batch.size());
  const auto in_rank  = static_cast<int64_t>(in_batch.size());

  std::vector<int64_t> strides(static_cast<size_t>(in_rank), 1);
  for (int64_t i = in_rank - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)]
      * in_batch[static_cast<size_t>(i + 1)];
  }

  int64_t off = 0;
  for (int64_t i = 0; i < in_rank; ++i) {
    auto c = coords[static_cast<size_t>(out_rank - in_rank + i)];
    if (in_batch[static_cast<size_t>(i)] == 1) {
      c = 0;
    }
    off += c * strides[static_cast<size_t>(i)];
  }
  return off;
}

/// 单个 MxM 三角系统的列回代,同 scipy.linalg.solve_triangular 的算法。
/// effective_lower 指示 effective 矩阵(已做 adjoint 转置后)的三角。
static std::vector<float> solve_triangular_ref(const std::vector<float> &a,
  const std::vector<float> &b, int64_t m, int64_t k, bool effective_lower) {
  std::vector<float> x(static_cast<size_t>(m * k), 0.0F);
  for (int64_t col = 0; col < k; ++col) {
    if (effective_lower) {
      // 前代:x[i] = (b[i] - Σ_{j<i} a[i][j] x[j]) / a[i][i]
      for (int64_t i = 0; i < m; ++i) {
        float sum = 0.0F;
        for (int64_t j = 0; j < i; ++j) {
          sum += a[static_cast<size_t>(i * m + j)]
            * x[static_cast<size_t>(j * k + col)];
        }
        x[static_cast<size_t>(i * k + col)] =
          (b[static_cast<size_t>(i * k + col)] - sum)
          / a[static_cast<size_t>(i * m + i)];
      }
    } else {
      // 回代:x[i] = (b[i] - Σ_{j>i} a[i][j] x[j]) / a[i][i]
      for (int64_t i = m - 1; i >= 0; --i) {
        float sum = 0.0F;
        for (int64_t j = i + 1; j < m; ++j) {
          sum += a[static_cast<size_t>(i * m + j)]
            * x[static_cast<size_t>(j * k + col)];
        }
        x[static_cast<size_t>(i * k + col)] =
          (b[static_cast<size_t>(i * k + col)] - sum)
          / a[static_cast<size_t>(i * m + i)];
      }
    }
  }
  return x;
}

/// 批量参考:adjoint 时先转置 matrix,再按 lower xor adjoint 的三角求解。
static std::vector<float> solve_batch_ref(const std::vector<float> &matrix,
  const Shape &matrix_shape, const std::vector<float> &rhs,
  const Shape &rhs_shape, bool lower, bool adjoint) {
  const auto rank = static_cast<int64_t>(matrix_shape.size());
  const auto m    = matrix_shape[rank - 1];
  const auto k    = rhs_shape[rank - 1];
  Shape      out_batch(matrix_shape.begin(), matrix_shape.end() - 2);
  Shape      matrix_batch(matrix_shape.begin(), matrix_shape.end() - 2);
  Shape      rhs_batch(rhs_shape.begin(), rhs_shape.end() - 2);
  for (size_t i = 0; i < out_batch.size(); ++i) {
    out_batch[i] = std::max(matrix_batch[i], rhs_batch[i]);
  }

  const size_t       batch_n = numel(out_batch);
  std::vector<float> out(batch_n * static_cast<size_t>(m * k));

  std::vector<int64_t> coords(out_batch.size(), 0);
  for (size_t b = 0; b < batch_n; ++b) {
    const auto m_off = broadcast_offset(coords, out_batch, matrix_batch);
    const auto r_off = broadcast_offset(coords, out_batch, rhs_batch);

    // 取出 effective 矩阵(实数域 adjoint 即转置)
    std::vector<float> a(static_cast<size_t>(m * m));
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < m; ++j) {
        a[static_cast<size_t>(i * m + j)] = matrix[m_off
          + static_cast<size_t>((adjoint ? j * m + i : i * m + j))];
      }
    }

    std::vector<float> bmat(static_cast<size_t>(m * k));
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < k; ++j) {
        bmat[static_cast<size_t>(i * k + j)] =
          rhs[r_off + static_cast<size_t>(i * k + j)];
      }
    }

    // adjoint 后三角翻转:lower xor adjoint
    const bool effective_lower = lower != adjoint;
    const auto x = solve_triangular_ref(a, bmat, m, k, effective_lower);
    std::copy(x.begin(), x.end(),
      out.begin() + static_cast<int64_t>(b * static_cast<size_t>(m * k)));

    for (int64_t d = static_cast<int64_t>(out_batch.size()) - 1; d >= 0; --d) {
      if (++coords[static_cast<size_t>(d)]
        < out_batch[static_cast<size_t>(d)]) {
        break;
      }
      coords[static_cast<size_t>(d)] = 0;
    }
  }
  return out;
}

static bool close_enough(float actual, float expected) {
  if (!std::isfinite(actual)) {
    return false;
  }
  const auto diff = std::fabs(actual - expected);
  return diff <= 1e-3F + 1e-3F * std::fabs(expected);
}

static int run_solve_case(Backend &backend, const char *case_name,
  const Shape &matrix_shape, const std::vector<float> &matrix,
  const Shape &rhs_shape, const std::vector<float> &rhs, bool lower,
  bool adjoint) {
  const auto matrix_name = std::string(case_name) + "_matrix";
  const auto rhs_name    = std::string(case_name) + "_rhs";
  const auto out_name    = std::string(case_name) + "_out";

  auto matrix_in =
    op::create_input(matrix_shape, DType::FP32).set_name(matrix_name);
  auto rhs_in = op::create_input(rhs_shape, DType::FP32).set_name(rhs_name);
  auto out    = op::matrix_triangular_solve(lower, adjoint)(matrix_in, rhs_in);
  out.set_name(out_name);

  const auto rank = static_cast<int64_t>(matrix_shape.size());
  Shape      expected_shape(matrix_shape.begin(), matrix_shape.end() - 2);
  Shape      rhs_batch(rhs_shape.begin(), rhs_shape.end() - 2);
  for (size_t i = 0; i < expected_shape.size(); ++i) {
    expected_shape[i] = std::max(expected_shape[i], rhs_batch[i]);
  }
  expected_shape.push_back(matrix_shape[rank - 2]);
  expected_shape.push_back(rhs_shape[rank - 1]);

  if (out.shape() != expected_shape || out.dtype() != DType::FP32) {
    std::cerr << case_name << ": declared shape/dtype mismatch" << std::endl;
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

  const auto expected =
    solve_batch_ref(matrix, matrix_shape, rhs, rhs_shape, lower, adjoint);

  auto matrix_copy = matrix;
  auto rhs_copy    = rhs;
  auto matrix_tensor =
    op::load_from_vector(std::move(matrix_copy), matrix_shape)
      .set_name(matrix_name);
  auto rhs_tensor =
    op::load_from_vector(std::move(rhs_copy), rhs_shape).set_name(rhs_name);
  auto outputs    = graph.run({matrix_tensor, rhs_tensor});
  auto out_tensor = outputs.get(out_name);
  if (out_tensor.shape() != expected_shape
    || out_tensor.dtype() != DType::FP32) {
    std::cerr << case_name << ": runtime shape/dtype mismatch" << std::endl;
    return 3;
  }

  const float *ptr = reinterpret_cast<const float *>(out_tensor.get_data());
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!close_enough(ptr[i], expected[i])) {
      std::cerr << case_name << ": value mismatch at " << i
                << ": out=" << ptr[i] << ", expected=" << expected[i]
                << std::endl;
      return 4;
    }
  }

  std::cout << "  [" << case_name << "] ok" << std::endl;
  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  // M=64, K=128(需求指定的规模)
  constexpr int64_t                     m = 64;
  constexpr int64_t                     k = 128;
  std::mt19937                          rng(42);
  std::uniform_real_distribution<float> strict(-0.25F, 0.25F);
  std::normal_distribution<float>       norm(0.0F, 1.0F);

  // 生成严格三角 N(|N| < 1,对角占优),被测系统 A = I − N:
  // 对角恰为 1.0,是"无 unit_diagonal 属性"风险项的目标场景
  auto make_unit_tri = [&](bool lower) {
    Shape              shape = {m, m};
    std::vector<float> a(static_cast<size_t>(m * m), 0.0F);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < m; ++j) {
        const bool in_strict = lower ? j < i : j > i;
        a[static_cast<size_t>(i * m + j)] =
          in_strict ? strict(rng) : (i == j ? 1.0F : 0.0F);
      }
    }
    return std::make_pair(shape, a);
  };

  auto make_rhs = [&](const Shape &shape) {
    std::vector<float> b(numel(shape));
    for (auto &v : b) {
      v = norm(rng);
    }
    return b;
  };

  // 需求必测:lower=true/false 各一(单位下三角/上三角)
  {
    auto [mshape, mat] = make_unit_tri(/*lower=*/true);
    auto rhs           = make_rhs({m, k});
    if (int rc = run_solve_case(backend, "solve_lower_unit", mshape, mat,
          {m, k}, rhs, /*lower=*/true, /*adjoint=*/false);
        rc != 0) {
      return rc;
    }
  }
  {
    auto [mshape, mat] = make_unit_tri(/*lower=*/false);
    auto rhs           = make_rhs({m, k});
    if (int rc = run_solve_case(backend, "solve_upper_unit", mshape, mat,
          {m, k}, rhs, /*lower=*/false, /*adjoint=*/false);
        rc != 0) {
      return rc;
    }
  }

  // adjoint=true(实数域转置):解 L^T X = B
  {
    auto [mshape, mat] = make_unit_tri(/*lower=*/true);
    auto rhs           = make_rhs({m, k});
    if (int rc = run_solve_case(backend, "solve_lower_adjoint", mshape, mat,
          {m, k}, rhs, /*lower=*/true, /*adjoint=*/true);
        rc != 0) {
      return rc;
    }
  }

  // 非单位对角(diag ∈ ±[1.5, 2.5]):捕获 kernel 错误假设单位对角的情况
  {
    Shape                                 shape = {m, m};
    std::vector<float>                    mat(static_cast<size_t>(m * m), 0.0F);
    std::uniform_real_distribution<float> diag(1.5F, 2.5F);
    std::uniform_int_distribution<int>    sign(0, 1);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < m; ++j) {
        mat[static_cast<size_t>(i * m + j)] = j < i
          ? strict(rng)
          : (i == j ? diag(rng) * (sign(rng) == 0 ? 1.0F : -1.0F) : 0.0F);
      }
    }
    auto rhs = make_rhs({m, k});
    if (int rc = run_solve_case(backend, "solve_lower_scaled_diag", shape, mat,
          {m, k}, rhs, /*lower=*/true, /*adjoint=*/false);
        rc != 0) {
      return rc;
    }
  }

  // 风险项:批维 (2,3,4) 广播(rhs 批维 (2,1,4))
  {
    Shape              matrix_shape = {2, 3, 4, m, m};
    std::vector<float> mat(numel(matrix_shape));
    for (size_t b = 0; b < 2 * 3 * 4; ++b) {
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < m; ++j) {
          mat[b * static_cast<size_t>(m * m) + static_cast<size_t>(i * m + j)] =
            j < i ? strict(rng) : (i == j ? 1.0F : 0.0F);
        }
      }
    }
    Shape rhs_shape = {2, 1, 4, m, k};
    auto  rhs       = make_rhs(rhs_shape);
    if (int rc = run_solve_case(backend, "solve_batch_broadcast", matrix_shape,
          mat, rhs_shape, rhs, /*lower=*/true, /*adjoint=*/false);
        rc != 0) {
      return rc;
    }
  }

  // 非方阵 matrix 须在构建期报错
  {
    auto matrix_in = op::create_input(Shape({4, 5}), DType::FP32);
    auto rhs_in    = op::create_input(Shape({4, 3}), DType::FP32);
    bool threw     = false;
    try {
      auto out = op::matrix_triangular_solve()(matrix_in, rhs_in);
      (void)out.shape();
    } catch (const std::exception &) {
      threw = true;
    }
    if (!threw) {
      std::cerr << "solve_invalid_shape: expected exception, got none"
                << std::endl;
      return 1;
    }
    std::cout << "  [solve_invalid_shape] ok" << std::endl;
  }

  std::cout << "MatrixTriangularSolve tests passed." << std::endl;
  return 0;
}
