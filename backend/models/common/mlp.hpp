#pragma once
#include <forch/nn/module.hpp>
#include <forch/nn/parallel_linear.hpp>
#include <forch/ops/function.hpp>

namespace finer::models {
// SiLU-gated MLP. gate/up split intermediate channels across ranks; down
// consumes the local slice and all-reduces (a no-op graph edge at TP1).
struct MLP : public forch::nn::Module {
  forch::nn::ColumnParallelLinear *gate_proj;
  forch::nn::ColumnParallelLinear *up_proj;
  forch::nn::RowParallelLinear    *down_proj;

  MLP(int64_t hidden_size, int64_t intermediate_size,
    const forch::ProcessGroup &group, forch::DType dtype, bool bias = false) {
    gate_proj = register_loadable("gate_proj",
      forch::nn::ColumnParallelLinear(
        hidden_size, intermediate_size, group, bias, dtype));
    up_proj   = register_loadable("up_proj",
        forch::nn::ColumnParallelLinear(
        hidden_size, intermediate_size, group, bias, dtype));
    down_proj = register_loadable("down_proj",
      forch::nn::RowParallelLinear(
        intermediate_size, hidden_size, group, bias, dtype));
  }

  forch::Tensor forward(const forch::Tensor &input) const {
    return down_proj->forward(forch::op::mul(
      forch::op::silu(gate_proj->forward(input)), up_proj->forward(input)));
  }
};

} // namespace finer::models
