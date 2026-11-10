#pragma once

#include <utility>

#include "forch/forch.hpp"
#include "forch/ops.hpp"

namespace forch::op {

struct Input : public Base {
  Input(Shape shape, DType dtype, bool forch_ge_enable_auto_H2D_copy)
      : _shape(std::move(shape)), _dtype(dtype),
        _forch_ge_enable_auto_H2D_copy(forch_ge_enable_auto_H2D_copy) {}

  void accept(Visitor *visitor, const Tensor &self) const override {
    visitor->visit(this, self);
  }

  Shape shape(const TensorList & /*unused*/ = {}) const override {
    return _shape;
  }

  DType dtype(const TensorList & /*unused*/ = {}) const override {
    return _dtype;
  }

  bool forch_ge_auto_H2D_copy_enabled() const {
    return _forch_ge_enable_auto_H2D_copy;
  }

private:
  Shape _shape;
  DType _dtype;
  bool  _forch_ge_enable_auto_H2D_copy;
};

/**
 * @param forch_ge_enable_auto_H2D_copy If true, the input tensor is REQUIRED to
 * be allocated in host memory. GE will then copy it to device memory on‑the‑fly
 * during execution.
 */
inline Tensor create_input(
  Shape shape, DType dtype, bool forch_ge_enable_auto_H2D_copy = false) {
  return {{},
    std::make_unique<Input>(
      std::move(shape), dtype, forch_ge_enable_auto_H2D_copy)};
}

} // namespace forch::op