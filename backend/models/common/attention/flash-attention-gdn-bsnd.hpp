#pragma once
#include <cmath>
#include <forch/ops/cast.hpp>
#include <forch/ops/function.hpp>
#include <forch/ops/recurrent_gated_delta_rule.hpp>
#include <forch/ops/repeat.hpp>
#include <forch/ops/slice.hpp>

namespace finer::models {
// Native recurrence over prepared token-major Q/K/V, beta and log decay g.
// Models own convolution, L2 normalization and learned gates. This adapter
// masks padding and handles kernel state tiling; it returns output/new state
// without registering cache write-back or applying the output gate/projection.
inline std::pair<forch::Tensor, forch::Tensor> flash_attention_gdn_bsnd(
  forch::Tensor q, forch::Tensor k, forch::Tensor v, forch::Tensor beta,
  forch::Tensor g, forch::Tensor state, forch::Tensor lengths,
  forch::Tensor indices, forch::Tensor valid) {
  if (q.dim() != 3 || state.dim() != 4)
    throw forch::exception("flash GDN requires TND query and rank-4 state");
  const auto rows = q.shape()[0], slots = state.shape()[0];
  if (slots <= 0 || rows < slots || rows % slots != 0)
    throw forch::exception("gdn_attention requires fixed slot packing");
  beta        = forch::op::mul(beta, valid | forch::op::cast(beta.dtype()));
  g           = forch::op::mul(g, valid);
  auto state4 = state;
  auto output = q, final_state = state4;
  {
    // Native prefill still requires BF16 state and BlockNum >= T.
    state4 = state4 | forch::op::cast(forch::DType::BF16);
    if (rows > slots)
      state4 = state4 | forch::op::repeat({rows / slots, 1, 1, 1});
    auto pair = forch::op::recurrent_gated_delta_rule(
      1.0F / std::sqrt(static_cast<float>(q.shape().back())), true)(
      q | forch::op::cast(forch::DType::BF16),
      k | forch::op::cast(forch::DType::BF16),
      v | forch::op::cast(forch::DType::BF16),
      beta | forch::op::cast(forch::DType::BF16), state4, lengths, indices, g);
    output      = pair.out;
    final_state = pair.ref_state;
    if (rows > slots)
      final_state = final_state
        | forch::op::slice({forch::op::Slice::Interval{0, slots},
          forch::op::Slice::IntervalAll{}, forch::op::Slice::IntervalAll{},
          forch::op::Slice::IntervalAll{}});
    final_state = final_state | forch::op::cast(state.dtype());
  }
  return {output | forch::op::cast(v.dtype()), final_state};
}

} // namespace finer::models
