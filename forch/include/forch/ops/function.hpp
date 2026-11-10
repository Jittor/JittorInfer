#pragma once

#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/ops.hpp"
#include <memory>
#include <type_traits>
#include <utility>

namespace forch::op {

Shape broadcast_shape(const TensorList &inputs);
DType broadcast_dtype(const TensorList &inputs);

struct Function : public Base {
  // For common ops without extra parameters, directly use `type` to determine
  // the function
  enum class Type { Add, Mul, Silu, Sin, Cos, Dynamic };
  Type type;

  Function(Type type = Type::Dynamic) : type(type) {}

  void accept(Visitor *visitor, const Tensor &self) const override {
    visitor->visit(this, self);
  }

  Shape shape(const TensorList &input) const override {
    if (type == Type::Dynamic) {
      throw exception(std::string("shape() not implemented for Function ")
        + typeid(*this).name());
    } else {
      // Add, Mul, Silu, Sin, Cos all have broadcasting shape rules
      return broadcast_shape(input);
    }
  }

  DType dtype(const TensorList &input) const override {
    // most ops just follow broadcasting dtype rules
    return broadcast_dtype(input);
  }
};

/**
 * Tensor Build Curry, 用于构建 Function 类型的 Tensor。
 * N = -1 时，支持任意个输入，否则只能有 N 个输入。
 */
template <int N = -1, Function::Type T = Function::Type::Dynamic>
struct TensorBuildCurry {
  std::unique_ptr<Function> op;

  TensorBuildCurry() : op(nullptr) {
    static_assert(T != Function::Type::Dynamic, "Type must be specified");
  }
  TensorBuildCurry(std::unique_ptr<Function> op) : op(std::move(op)) {}

  // function that take arbitary number of Tensors
  template <class... Args>
  Tensor operator()(Args... args) {
    static_assert((std::is_constructible_v<Tensor, Args> && ...),
      "All args must be convertible to Tensor");
    static_assert(
      N == -1 || sizeof...(Args) == N, "Number of args must be equal to N");

    if constexpr (T != Function::Type::Dynamic) {
      return Tensor(
        {std::forward<Args>(args)...}, std::make_unique<Function>(T));
    } else {
      if (op == nullptr) {
        throw exception(
          "TensorBuildCurry error: op is nullptr, do not call same "
          "TensorBuildCurry with Dynamic type more than once");
      } else {
        return Tensor(
          {std::forward<Args>(args)...}, std::exchange(op, nullptr));
      }
    }
  }
};

inline TensorBuildCurry<-1, Function::Type::Add> add;
inline TensorBuildCurry<-1, Function::Type::Mul> mul;
inline TensorBuildCurry<1, Function::Type::Silu> silu;
inline TensorBuildCurry<1, Function::Type::Sin>  sin;
inline TensorBuildCurry<1, Function::Type::Cos>  cos;

struct NegBuilder {
  Tensor operator()(const Tensor &input) const;
};
inline NegBuilder neg;

namespace internal {
template <typename T>
inline constexpr bool always_false_v = false;
}

} // namespace forch::op

/**
 * 支持 forch::Tensor | Fn Tensor(Tensor) 的重载，方便链式调用。
 * 我们希望动态引入算子，所以无法写成 tensor.reshape((1,2,3))，只能写成
 * tensor | reshape((1,2,3))
 *
 * 可以是 rhs(lhs) 或者 rhs.forward(lhs)
 */
template <class T>
forch::Tensor operator|(const forch::Tensor &lhs, T &&rhs) {
  using U = std::remove_reference_t<T>;
  if constexpr (std::is_invocable_v<U, const forch::Tensor &>) {
    return std::forward<T>(rhs)(lhs);
  } else if constexpr (std::is_invocable_v<decltype(&U::forward), U,
                         const forch::Tensor &>) {
    return std::forward<T>(rhs).forward(lhs);
  } else {
    static_assert(forch::op::internal::always_false_v<T>,
      "T must be invocable with Tensor or have forward(Tensor)");
  }
}
template <class T>
forch::Tensor operator|(forch::Tensor &&lhs, T &&rhs) {
  using U = std::remove_reference_t<T>;
  if constexpr (std::is_invocable_v<U, forch::Tensor &&>) {
    return std::forward<T>(rhs)(std::forward<forch::Tensor &&>(lhs));
  } else if constexpr (std::is_invocable_v<decltype(&U::forward), U,
                         forch::Tensor &&>) {
    return std::forward<T>(rhs).forward(std::forward<forch::Tensor &&>(lhs));
  } else {
    static_assert(forch::op::internal::always_false_v<T>,
      "T must be invocable with Tensor or have forward(Tensor)");
  }
}