#pragma once

#include "forch/ops/function.hpp"

namespace forch::op {

// 自然指数 y = e^x。GE Exp 的 base/scale/shift 属性保持默认组合
// (base = -1 在 GE 语义里表示自然常数 e,scale = 1,shift = 0)。
// dtype 跟随输入,不做隐式升精度,需要 fp32 时调用方自行 cast。

struct Exp : public Function {
  Exp() = default;

  Shape shape(const TensorList &input) const override {
    // 不能依赖 Function::shape() 的默认实现:它按 enum type 派发,子类不占
    // enum 值时 type 为 Dynamic 会抛异常;单输入广播即恒等
    return broadcast_shape(input);
  }
};

inline auto exp() {
  return TensorBuildCurry<1>{std::make_unique<Exp>()};
}

} // namespace forch::op
