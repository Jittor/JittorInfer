#pragma once

#include "forch/exception.hpp"
#include "forch/ops/function.hpp"

namespace forch::op {

struct ScatterUpdate : public Function {

  ScatterUpdate() {}

  Shape shape(const TensorList &input) const override {
    if (input.size() < 3) {
      throw exception(
        "scatter_update: requires 3 inputs (var, indices, updates)");
    }

    // result has same shape as `var`
    auto var_shape = input[0].shape();

    // Basic sanity checks (conservative): require updates rank match var rank
    // and non-indexed dimensions match. This can be relaxed if needed.
    auto updates_shape = input[2].shape();
    if (updates_shape.size() != var_shape.size()) {
      throw exception("scatter_update: updates must have same rank as var for "
                      "this implementation");
    }
    for (size_t i = 1; i < var_shape.size(); ++i) {
      if (updates_shape[i] != var_shape[i]) {
        throw exception("scatter_update: updates shape must match var shape on "
                        "non-indexed dimensions");
      }
    }

    return var_shape;
  }

  // Avoid forch's default implicit dtype broadcast across all inputs.
  // Enforce that `var` and `updates` have identical dtype and return that
  // dtype. This prevents the "Incompatible dtypes" exception when indices has a
  // different dtype.
  DType dtype(const TensorList &input) const override {
    if (input.size() < 3) {
      throw exception(
        "scatter_update: requires 3 inputs (var, indices, updates)");
    }
    auto var_dt     = input[0].dtype();
    auto updates_dt = input[2].dtype();
    if (var_dt != updates_dt) {
      throw exception(
        "scatter_update: var and updates must have the same dtype");
    }
    return var_dt;
  }
};

inline auto scatter_update() {
  return TensorBuildCurry<3>{std::make_unique<ScatterUpdate>()};
}

} // namespace forch::op