#pragma once
#include <forch/ops/cat.hpp>
#include <forch/ops/function.hpp>
#include <forch/ops/slice.hpp>

namespace finer::models {
/**
 * Rope with mode `half`
 */
inline auto apply_rope_half(
  const forch::Tensor &qk, const forch::Tensor &cos, const forch::Tensor &sin) {
  using namespace forch::op;

  auto rotate_half = [&](const forch::Tensor &x) {
    auto hidden_dim = x.shape().back();
    auto x1 =
      x | slice({Slice::Ellipsis{}, Slice::IntervalFromBegin{hidden_dim / 2}});
    auto x2 =
      x | slice({Slice::Ellipsis{}, Slice::IntervalTillEnd{hidden_dim / 2}});
    return cat((int64_t)x.shape().size() - 1)(x2 | neg, x1);
  };

  return add(mul(qk, cos), mul(rotate_half(qk), sin));
}

} // namespace finer::models
