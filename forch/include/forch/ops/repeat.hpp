#pragma once

#include "forch/forch.hpp"
#include "forch/ops/function.hpp"
#include <utility>

namespace forch::op {

struct Repeat : public Function {
  Shape repeat;
  bool  interleave;

  Repeat(Shape repeat, bool interleave)
      : repeat(std::move(repeat)), interleave(interleave) {}

  Shape shape(const TensorList &input) const override {
    Shape out_shape = input[0].shape();
    for (size_t i = 0; i < out_shape.size(); i++) {
      out_shape[i] *= repeat[i];
    }
    return out_shape;
  }
};

/**
 * output.shape = input.shape * shape
 * if interleave is false:
 *   output[i][j]... = input[i % input.shape[0]][j % input.shape[1]]...
 * else:
 *   output[i][j]... = input[i // shape[0]][j // shape[1]]...
 */
inline auto repeat(Shape repeat, bool interleave = false) {
  return TensorBuildCurry<1>{
    std::make_unique<Repeat>(std::move(repeat), interleave)};
}

} // namespace forch::op