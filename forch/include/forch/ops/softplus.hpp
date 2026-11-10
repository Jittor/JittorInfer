#pragma once

#include "forch/ops/function.hpp"

namespace forch::op {

// softplus(x) = log(exp(x) + 1),数值稳定版由 GE kernel 内部处理。
// dtype 跟随输入,不做隐式升精度,需要 fp32 时调用方自行 cast。

struct Softplus : public Function {
  Softplus() = default;

  Shape shape(const TensorList &input) const override {
    // 不能依赖 Function::shape() 的默认实现:它按 enum type 派发,子类不占
    // enum 值时 type 为 Dynamic 会抛异常;单输入广播即恒等
    return broadcast_shape(input);
  }
};

inline auto softplus() {
  return TensorBuildCurry<1>{std::make_unique<Softplus>()};
}

} // namespace forch::op
