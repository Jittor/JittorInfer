#pragma once

#include "forch/ops/function.hpp"

namespace forch::op {

struct RmsNorm : public Function {
  float epsilon;

  RmsNorm(float epsilon) : epsilon(epsilon) {}

  Shape shape(const TensorList &input) const override {
    const auto &x     = input[0];
    const auto &gamma = input[1];

    if (gamma.dim() != 1) {
      throw exception("forch: RmsNorm gamma must be 1D");
    }

    if (gamma.shape()[0] != x.shape()[x.dim() - 1]) {
      throw exception(
        "forch: RmsNorm gamma size must match the last dimension of input");
    }

    return input[0].shape();
  }
};

inline auto rms_norm(float epsilon) {
  return TensorBuildCurry<2>{std::make_unique<RmsNorm>(epsilon)};
}

} // namespace forch::op
