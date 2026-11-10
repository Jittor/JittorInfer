#pragma once

#include "forch/ops/function.hpp"
#include <utility>

namespace forch::op {

struct Permute : public Function {
  std::vector<int64_t> perm;

  Permute(std::vector<int64_t> dims) : perm(std::move(dims)) {}

  Shape shape(const TensorList &input) const override {
    Shape out_shape(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
      out_shape[i] = input[0].shape()[perm[i]];
    }
    return out_shape;
  }
};

inline auto permute(std::vector<int64_t> dims) {
  return TensorBuildCurry<1>{std::make_unique<Permute>(std::move(dims))};
}

} // namespace forch::op
