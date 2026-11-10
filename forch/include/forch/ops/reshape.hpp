#pragma once

#include "forch/forch.hpp"
#include "forch/ops/function.hpp"
#include <utility>

namespace forch::op {

struct Reshpae : public Function {
  Shape new_shape;

  Reshpae(Shape shape) : new_shape(std::move(shape)) {}

  Shape shape(const TensorList & /*unused*/) const override {
    // todo: check input shape
    return new_shape;
  }
};

inline auto reshape(Shape shape) {
  return TensorBuildCurry<1>{std::make_unique<Reshpae>(std::move(shape))};
}

} // namespace forch::op