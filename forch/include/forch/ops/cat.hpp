#pragma once

#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/ops/function.hpp"
#include <cstdint>

namespace forch::op {

struct Cat : public Function {
  int64_t dim;

  Cat(int64_t dim) : dim(dim) {}

  Shape shape(const TensorList &input) const override {
    Shape output_shape = input[0].shape();
    output_shape[dim]  = 0;
    for (const auto &tensor : input) {
      if (tensor.shape().size() != output_shape.size()) {
        throw exception("cat: all inputs must have same rank");
      }
      for (size_t i = 0; i < output_shape.size(); ++i) {
        if (i != static_cast<size_t>(dim)
          && tensor.shape()[i] != output_shape[i]) {
          throw exception(
            "cat: all inputs must have same shape except in cat dimension");
        }
      }
      output_shape[dim] += tensor.shape()[dim];
    }
    return output_shape;
  }
};

inline auto cat(int64_t dim) {
  return TensorBuildCurry<-1>{std::make_unique<Cat>(dim)};
}

} // namespace forch::op