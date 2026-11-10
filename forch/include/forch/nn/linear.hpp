#pragma once

#include "forch/forch.hpp"
#include "forch/ops/batched_mat_mul.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/mat_mul.hpp"

#include "forch/nn/module.hpp"
#include "forch/nn/parameter.hpp"

namespace forch::nn {

class Linear : public Module {
public:
  int64_t in_features;
  int64_t out_features;

  Linear(int64_t in_features, int64_t out_features, bool bias = true,
    DType dtype = DType::FP32)
      : in_features(in_features), out_features(out_features), use_bias(bias) {
    weight = register_loadable(
      "weight", Parameter({out_features, in_features}, dtype));

    if (bias) {
      this->bias = register_loadable("bias", Parameter({out_features}, dtype));
    } else {
      this->bias = nullptr;
    }
  }

  Tensor forward(const Tensor &input) {
    Tensor x = (input.shape().size() > 2
        ? op::batched_mat_mul(false, true)
        : op::mat_mul(false, true))(input, weight->get());
    if (use_bias) {
      x = op::add(x, bias->get());
    }
    return x;
  }

private:
  bool       use_bias;
  Parameter *weight;
  Parameter *bias;
};

} // namespace forch::nn
