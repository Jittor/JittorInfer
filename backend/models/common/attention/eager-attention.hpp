#pragma once

// Eager reference: explicit GQA matmul/softmax with an additive mask over
// the whole physical pool. It is the debugging baseline for every fused
// scheme, so it keeps arithmetic deliberately plain: scores in the model
// dtype, softmax in FP32.

#include <forch/ops/batched_mat_mul.hpp>
#include <forch/ops/cast.hpp>
#include <forch/ops/const.hpp>
#include <forch/ops/function.hpp>
#include <forch/ops/permute.hpp>
#include <forch/ops/repeat.hpp>
#include <forch/ops/reshape.hpp>
#include <forch/ops/softmax.hpp>

#include <optional>
#include <string>

namespace finer::models {

// [KVH, S, D] -> [KVH*n_rep, S, D]
inline forch::Tensor repeat_kv(const forch::Tensor &kv, int64_t n_rep) {
  if (n_rep == 1)
    return kv;
  const auto s = kv.shape();
  return kv | forch::op::reshape({s[0], 1, s[1], s[2]})
    | forch::op::repeat({1, n_rep, 1, 1})
    | forch::op::reshape({s[0] * n_rep, s[1], s[2]});
}

/// q [H, T, D], k/v [KVH, rows, D] head-major; mask [T, rows] FP32 0/-inf.
/// Returns [T, H, Dv]. Intermediate node names keep the debug-dump contract
/// of the original per-model implementations.
inline forch::Tensor eager_attention(const forch::Tensor &q,
  const forch::Tensor &k, const forch::Tensor &v,
  const std::optional<forch::Tensor> &mask, float scale,
  const std::string &debug_prefix = {}) {
  const auto dtype = q.dtype();
  const auto group = q.shape()[0] / k.shape()[0];
  auto       key   = repeat_kv(k, group);
  auto       value = repeat_kv(v, group);
  auto s = forch::op::load_from_vector<float>({scale}) | forch::op::cast(dtype);
  auto weights = forch::op::mul(
    s, forch::op::batched_mat_mul()(q, key | forch::op::permute({0, 2, 1})));
  if (mask)
    weights = forch::op::add(weights, *mask | forch::op::cast(dtype));
  weights = weights | forch::op::cast(forch::DType::FP32)
    | forch::op::softmax(weights.dim() - 1) | forch::op::cast(dtype);
  if (!debug_prefix.empty())
    weights.set_name(debug_prefix + ".attn_weights");
  auto out = forch::op::batched_mat_mul()(weights, value)
    | forch::op::permute({1, 0, 2});
  if (!debug_prefix.empty())
    out.set_name(debug_prefix + ".attn_output");
  return out;
}

} // namespace finer::models
