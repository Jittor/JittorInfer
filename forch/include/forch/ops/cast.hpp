#pragma once

#include "forch/forch.hpp"
#include "forch/ops/function.hpp"

namespace forch::op {

struct Cast : public Function {
  Cast(DType dtype) : _dtype(dtype) {}

  Shape shape(const TensorList &input) const override {
    return input[0].shape();
  }

  DType dtype(const TensorList & /*unused*/ = {}) const override {
    return _dtype;
  }

private:
  DType _dtype;
};

inline auto cast(DType dtype) {
  return TensorBuildCurry<1>{std::make_unique<Cast>(dtype)};
}

} // namespace forch::op