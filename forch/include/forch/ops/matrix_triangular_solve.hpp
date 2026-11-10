#pragma once

#include "forch/ops/function.hpp"

#include <algorithm>

namespace forch::op {

// 三角回代求解 op(matrix) @ X = rhs:
//   matrix (..., M, M), rhs (..., M, K) → y (..., M, K)
// lower 指示读取 matrix 的哪个三角(与 torch.linalg.solve_triangular 的
// upper/lower 相反,与 TF/scipy 的 lower 一致);adjoint 时求解共轭转置
// (实数域即转置)。
//
// GE IR 只登记 fp32/fp64 且要求 matrix/rhs 同型,无 bf16、无 unit_diagonal
// 属性——调用方解单位下三角系统 A = I − M 时按普通三角解,精度由测试确认。
struct MatrixTriangularSolve : public Function {
  bool lower   = true;
  bool adjoint = false;

  explicit MatrixTriangularSolve(bool lower_ = true, bool adjoint_ = false)
      : lower(lower_), adjoint(adjoint_) {}

  Shape shape(const TensorList &input) const override {
    const auto &matrix = input[0];
    const auto &rhs    = input[1];

    const auto &a = matrix.shape();
    const auto &b = rhs.shape();
    if (a.size() < 2 || b.size() < 2) {
      throw exception("matrix_triangular_solve: matrix/rhs rank must be >= 2");
    }

    const auto m = a[a.size() - 1];
    if (a[a.size() - 2] != m) {
      throw exception(
        "matrix_triangular_solve: innermost matrix must be square MxM");
    }
    if (b[b.size() - 2] != m) {
      throw exception(
        "matrix_triangular_solve: rhs rows must match matrix size M");
    }

    if (matrix.dtype() != rhs.dtype()
      || (matrix.dtype() != DType::FP32 && matrix.dtype() != DType::FP64)) {
      throw exception(
        "matrix_triangular_solve: dtypes must match and be fp32/fp64");
    }

    // 批维 (... 允许 numpy 式广播。GE 是否支持批维广播需运行时验证,
    // 不支持时调用方应退化为循环展开。
    Shape out = broadcast_batch(
      Shape(a.begin(), a.end() - 2), Shape(b.begin(), b.end() - 2));
    out.push_back(m);
    out.push_back(b[b.size() - 1]);
    return out;
  }

private:
  static Shape broadcast_batch(const Shape &x, const Shape &y) {
    const auto rank = std::max(x.size(), y.size());
    Shape      out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
      const auto dx = i < rank - x.size() ? 1 : x[i - (rank - x.size())];
      const auto dy = i < rank - y.size() ? 1 : y[i - (rank - y.size())];
      if (dx != dy && dx != 1 && dy != 1) {
        throw exception(
          "matrix_triangular_solve: batch dims are not broadcastable");
      }
      out[i] = std::max(dx, dy);
    }
    return out;
  }
};

inline auto matrix_triangular_solve(bool lower = true, bool adjoint = false) {
  return TensorBuildCurry<2>{
    std::make_unique<MatrixTriangularSolve>(lower, adjoint)};
}

} // namespace forch::op
