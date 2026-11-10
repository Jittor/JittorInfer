#pragma once

#include "forch/ops/function.hpp"

namespace forch::op {

/// Element-wise division with the same shape broadcasting as add/mul.
/// Both inputs must have the same dtype; the output preserves that dtype.
struct Div : public Function {
  Shape shape(const TensorList &input) const override {
    return broadcast_shape(input);
  }

  DType dtype(const TensorList &input) const override {
    return broadcast_dtype(input);
  }
};

/// div is a factory (not a variable): the curry owns the op instance, which
/// would be consumed on first call otherwise.
inline auto div() {
  return TensorBuildCurry<2>{std::make_unique<Div>()};
}

} // namespace forch::op
