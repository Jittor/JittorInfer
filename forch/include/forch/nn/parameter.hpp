#pragma once

#include "forch/exception.hpp"
#include "forch/nn/module.hpp"

#include <optional>

namespace forch::nn {

class Parameter : public Loadable {
public:
  Parameter(Shape shape, DType dtype)
      : _shape(std::move(shape)), _dtype(dtype) {}

  const Tensor &get() const {
    if (!_tensor) {
      throw exception("Parameter is not loaded");
    }
    return *_tensor;
  }

  void load(Loader *loader, const std::string &prefix) override {
    auto tensor = loader->load(prefix);
    if (tensor.shape() != _shape) {
      throw exception(
        std::string("Parameter load shape mismatch for ") + prefix);
    }
    if (tensor.dtype() != _dtype) {
      throw exception(
        std::string("Parameter load dtype mismatch for ") + prefix);
    }
    _tensor = std::move(tensor);
  }

private:
  std::optional<Tensor> _tensor;
  Shape                 _shape;
  DType                 _dtype;
};

} // namespace forch::nn
