#pragma once

#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/ops/function.hpp"

namespace forch::op {

struct IndexSelect : public Function {
  int64_t dim;

  IndexSelect(int64_t dim) : dim(dim) {}

  Shape shape(const TensorList &input) const override {
    const auto &input_x       = input[0];
    const auto &input_indices = input[1];

    if (input_indices.dim() != 1) {
      throw exception("forch: index_select op only supports 1D indices");
    }

    auto output_shape = input_x.shape();
    output_shape[dim] = input_indices.shape()[0];
    return output_shape;
  }

  DType dtype(const TensorList &input) const override {
    if (input[1].dtype() != DType::INT64 && input[1].dtype() != DType::INT32) {
      throw exception(
        "forch: index_select op only supports INT64 or INT32 indices");
    }
    return input[0].dtype();
  }
};

inline auto index_select(int64_t dim) {
  return TensorBuildCurry<2>{std::make_unique<IndexSelect>(dim)};
}

} // namespace forch::op
