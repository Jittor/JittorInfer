#pragma once

#include "forch/ops/function.hpp"

namespace forch::op {

// 取最后两维的上三角,diagonal 为对角线偏移(numpy/torch 语义)。
// GE Triu 支持 2-8 维输入。
struct Triu : public Function {
  int64_t diagonal = 0;

  explicit Triu(int64_t diagonal_ = 0) : diagonal(diagonal_) {}

  Shape shape(const TensorList &input) const override {
    auto rank = static_cast<int64_t>(input[0].dim());
    if (rank < 2 || rank > 8) {
      throw exception("triu: input rank must be in [2, 8]");
    }
    return input[0].shape();
  }
};

inline auto triu(int64_t diagonal = 0) {
  return TensorBuildCurry<1>{std::make_unique<Triu>(diagonal)};
}

} // namespace forch::op
