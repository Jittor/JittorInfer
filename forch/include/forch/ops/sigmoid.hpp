#pragma once

#include "forch/ops/function.hpp"

namespace forch::op {

// sigmoid(x) = 1 / (1 + e^-x)。dtype 跟随输入,不做隐式升精度,
// 需要 fp32 时调用方自行 cast。

struct Sigmoid : public Function {
  Sigmoid() = default;

  Shape shape(const TensorList &input) const override {
    // 不能依赖 Function::shape() 的默认实现:它按 enum type 派发,子类不占
    // enum 值时 type 为 Dynamic 会抛异常;单输入广播即恒等
    return broadcast_shape(input);
  }
};

inline auto sigmoid() {
  return TensorBuildCurry<1>{std::make_unique<Sigmoid>()};
}

} // namespace forch::op
