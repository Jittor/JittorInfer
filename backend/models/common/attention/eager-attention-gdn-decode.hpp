#pragma once
// Stateless FP32 recurrence reference. Learned input/output transformations
// and cache write-back are owned by the model, as in the fused GDN adapter.
#include <cmath>
#include <forch/ops/cast.hpp>
#include <forch/ops/const.hpp>
#include <forch/ops/exp.hpp>
#include <forch/ops/function.hpp>
#include <forch/ops/reduce.hpp>
#include <forch/ops/repeat.hpp>
#include <forch/ops/reshape.hpp>

namespace finer::models {
// One token per slot, state ordered [B, Nv, Dv, Dk]. Keep the recurrence
// (including its persistent output) in FP32; no matmul precision policy or
// BF16 state round-trip is involved. q/k are already L2-normalized.
inline std::pair<forch::Tensor, forch::Tensor> eager_gdr_decode(forch::Tensor q,
  forch::Tensor k, forch::Tensor v, forch::Tensor beta, forch::Tensor g,
  forch::Tensor state) {
  using namespace forch;
  if (q.dim() != 3 || k.shape() != q.shape() || v.dim() != 3
    || state.dim() != 4)
    throw exception("eager GDR: invalid q/k/v/state shape");
  const auto b = q.shape()[0], nk = q.shape()[1], dk = q.shape()[2];
  const auto nv = v.shape()[1], dv = v.shape()[2];
  if (b <= 0 || nk <= 0 || nv <= 0 || dk <= 0 || dv <= 0 || nv % nk
    || v.shape()[0] != b || state.shape() != Shape{b, nv, dv, dk}
    || beta.shape() != Shape{b, nv} || g.shape() != Shape{b, nv})
    throw exception("eager GDR: inconsistent dimensions");
  auto heads = [&](Tensor x) {
    x = x | op::cast(DType::FP32) | op::reshape({b, nk, 1, dk});
    if (nv != nk)
      x = x | op::repeat({1, 1, nv / nk, 1});
    return x | op::reshape({b, nv, 1, dk});
  };
  q     = heads(q);
  k     = heads(k);
  v     = v | op::cast(DType::FP32);
  beta  = beta | op::cast(DType::FP32) | op::reshape({b, nv, 1});
  g     = g | op::cast(DType::FP32) | op::exp() | op::reshape({b, nv, 1, 1});
  state = op::mul(state | op::cast(DType::FP32), g);
  auto memory = op::mul(state, k) | op::reduce_sum({3});
  auto delta =
    op::mul(op::add(v, op::neg(memory)), beta) | op::reshape({b, nv, dv, 1});
  state    = op::add(state, op::mul(delta, k));
  auto out = op::mul(state, q) | op::reduce_sum({3});
  out      = op::mul(out,
         op::load_from_vector(
      std::vector<float>{1.F / std::sqrt(static_cast<float>(dk))}, {}));
  return {out, state};
}

inline std::pair<forch::Tensor, forch::Tensor> eager_attention_gdn_decode(
  forch::Tensor q, forch::Tensor k, forch::Tensor v, forch::Tensor beta,
  forch::Tensor g, forch::Tensor state, forch::Tensor valid) {
  if (q.dim() != 3 || state.dim() != 4 || q.shape()[0] != state.shape()[0])
    throw forch::exception("eager GDN requires one token per slot");
  beta = forch::op::mul(beta, valid | forch::op::cast(beta.dtype()));
  g    = forch::op::mul(g, valid);
  auto [out, next] = eager_gdr_decode(q, k, v, beta, g, state);
  return {out | forch::op::cast(v.dtype()), next};
}

} // namespace finer::models
