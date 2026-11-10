#pragma once

#include "forch/exception.hpp"
#include "forch/ops/function.hpp"

namespace forch::op {

/// Select then/else elements using a BOOL condition; all three shapes
/// broadcast. The two data inputs must share a dtype, which is also the output
/// dtype.
struct Select : public Function {
  Shape shape(const TensorList &input) const override {
    if (input.size() != 3) {
      throw exception("select: requires 3 inputs (condition, then, else)");
    }
    if (input[0].dtype() != DType::BOOL) {
      throw exception("select: condition must be BOOL");
    }
    return broadcast_shape(input);
  }

  DType dtype(const TensorList &input) const override {
    if (input.size() != 3) {
      throw exception("select: requires 3 inputs (condition, then, else)");
    }
    // The BOOL condition does not participate in data dtype validation.
    TensorList data{input[1], input[2]};
    return broadcast_dtype(data);
  }
};

inline auto select() {
  return TensorBuildCurry<3>{std::make_unique<Select>()};
}

} // namespace forch::op
