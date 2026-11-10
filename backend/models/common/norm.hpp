#pragma once
#include <forch/nn/module.hpp>
#include <forch/nn/parameter.hpp>
#include <forch/ops/rms_norm.hpp>

namespace finer::models {
struct RMSNorm : public forch::nn::Module {
  forch::nn::Parameter *weight;
  float                 eps;
  forch::DType          dtype;

  RMSNorm(int64_t hidden_size, float eps = 1e-6,
    forch::DType dtype = forch::DType::BF16)
      : eps(eps), dtype(dtype) {
    weight =
      register_loadable("weight", forch::nn::Parameter({hidden_size}, dtype));
  }

  forch::Tensor forward(const forch::Tensor &input) const {
    return forch::op::rms_norm(eps)(input, weight->get());
  }

  forch::Tensor operator()(const forch::Tensor &input) const {
    return forward(input);
  }
};

} // namespace finer::models
