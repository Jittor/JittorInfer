#pragma once

#include "forch/ops/function.hpp"

namespace forch::op {

struct Softmax : public Function {
  int64_t dim;

  Softmax(int64_t dim) : dim(dim) {}

  Shape shape(const TensorList &input) const override {
    if (dim < 0 || dim >= input[0].dim()) {
      throw exception("softmax: dim out of range");
    }

    return input[0].shape();
  }
};

inline auto softmax(int64_t dim) {
  return TensorBuildCurry<1>{std::make_unique<Softmax>(dim)};
}

} // namespace forch::op