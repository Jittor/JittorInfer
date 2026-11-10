#pragma once

#include "forch/exception.hpp"
#include "forch/ops/function.hpp"

namespace forch::op {

struct MatMul : public Function {
  bool transpose_a;
  bool transpose_b;

  MatMul(bool transpose_a, bool transpose_b)
      : transpose_a(transpose_a), transpose_b(transpose_b) {}

  Shape shape(const TensorList &input) const override {
    // enforce broadcast
    auto shape_a = input[0].shape();
    while (shape_a.size() < 2) {
      shape_a.insert(shape_a.begin(), 1);
    }
    auto shape_b = input[1].shape();
    while (shape_b.size() < 2) {
      shape_b.insert(shape_b.begin(), 1);
    }

    if (shape_a.size() > 2 || shape_b.size() > 2) {
      throw exception("mat_mul: only supports up to 2D tensors");
    }

    if (transpose_a) {
      std::swap(shape_a[0], shape_a[1]);
    }
    if (transpose_b) {
      std::swap(shape_b[0], shape_b[1]);
    }

    if (shape_a[1] != shape_b[0]) {
      throw exception("mat_mul: inner dimensions must match after transpose");
    }

    return {shape_a[0], shape_b[1]};
  }
};

/**
 * matmul without batch, for 1D/2D tensors.
 */
inline auto mat_mul(bool transpose_a = false, bool transpose_b = false) {
  return TensorBuildCurry<2>{
    std::make_unique<MatMul>(transpose_a, transpose_b)};
}

} // namespace forch::op