#pragma once

#include "forch/ops/function.hpp"

namespace forch::op {

struct Cumsum : public Function {
  int64_t axis;
  bool    exclusive = false;
  bool    reverse   = false;

  explicit Cumsum(int64_t axis_, bool exclusive_ = false, bool reverse_ = false)
      : axis(axis_), exclusive(exclusive_), reverse(reverse_) {}

  Shape shape(const TensorList &input) const override {
    // GE 语义:axis ∈ [-rank, rank),允许负值
    auto rank = static_cast<int64_t>(input[0].dim());
    auto ndim = axis < 0 ? axis + rank : axis;
    if (ndim < 0 || ndim >= rank) {
      throw exception("cumsum: axis out of range");
    }
    return input[0].shape();
  }
};

// inclusive: y[i] = Σ_{j<=i} x[j]
// exclusive: y[i] = Σ_{j<i}  x[j]
// reverse:   从轴末端向起点累加
inline auto cumsum(int64_t axis, bool exclusive = false, bool reverse = false) {
  return TensorBuildCurry<1>{
    std::make_unique<Cumsum>(axis, exclusive, reverse)};
}

} // namespace forch::op
