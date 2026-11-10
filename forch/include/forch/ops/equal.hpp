#pragma once

#include "forch/ops/function.hpp"

namespace forch::op {

/// Compare corresponding elements for equality, broadcasting input shapes.
/// Returns a BOOL tensor that can be used as the condition of select().
struct Equal : public Function {
  Shape shape(const TensorList &input) const override {
    return broadcast_shape(input);
  }

  DType dtype(const TensorList & /*unused*/) const override {
    return DType::BOOL;
  }
};

inline auto equal() {
  return TensorBuildCurry<2>{std::make_unique<Equal>()};
}

} // namespace forch::op
