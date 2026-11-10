#pragma once

#include "forch/exception.hpp"
#include "forch/ops/function.hpp"

#include <algorithm>

namespace forch::op {

struct BatchedMatMul : public Function {
  bool transpose_a;
  bool transpose_b;

  BatchedMatMul(bool transpose_a, bool transpose_b)
      : transpose_a(transpose_a), transpose_b(transpose_b) {}

  Shape shape(const TensorList &input) const override {
    auto shape_a = input[0].shape();
    auto shape_b = input[1].shape();

    if (transpose_a) {
      std::swap(shape_a[shape_a.size() - 2], shape_a[shape_a.size() - 1]);
    }
    if (transpose_b) {
      std::swap(shape_b[shape_b.size() - 2], shape_b[shape_b.size() - 1]);
    }

    if (shape_a.back() != shape_b[shape_b.size() - 2]) {
      throw exception(
        "batched_mat_mul: inner dimensions must match after transpose");
    }

    Shape batch_a(shape_a.begin(), shape_a.end() - 2);
    Shape batch_b(shape_b.begin(), shape_b.end() - 2);
    Shape batch_shape(
      std::max(batch_a.size(), batch_b.size()), static_cast<int64_t>(1));

    const size_t offset_a = batch_shape.size() - batch_a.size();
    const size_t offset_b = batch_shape.size() - batch_b.size();
    for (size_t i = 0; i < batch_shape.size(); ++i) {
      const int64_t dim_a = i < offset_a ? 1 : batch_a[i - offset_a];
      const int64_t dim_b = i < offset_b ? 1 : batch_b[i - offset_b];

      if (dim_a != 1 && dim_b != 1 && dim_a != dim_b) {
        throw exception(
          "batched_mat_mul: batch dimensions are not broadcastable");
      }
      batch_shape[i] = std::max(dim_a, dim_b);
    }

    batch_shape.push_back(shape_a[shape_a.size() - 2]);
    batch_shape.push_back(shape_b.back());
    return batch_shape;
  }
};

inline auto batched_mat_mul(
  bool transpose_a = false, bool transpose_b = false) {
  return TensorBuildCurry<2>{
    std::make_unique<BatchedMatMul>(transpose_a, transpose_b)};
}

} // namespace forch::op
