#pragma once
#include <forch/ops/batched_mat_mul.hpp>
#include <forch/ops/cast.hpp>
#include <forch/ops/cat.hpp>
#include <forch/ops/collective.hpp>
#include <forch/ops/ge/moe_finalize_routing.hpp>
#include <forch/ops/ge/moe_init_routing.hpp>
#include <forch/ops/grouped_mat_mul.hpp>
#include <forch/ops/one_hot.hpp>
#include <forch/ops/permute.hpp>
#include <forch/ops/reduce.hpp>
#include <forch/ops/repeat.hpp>
#include <forch/ops/reshape.hpp>
#include <type_traits>

namespace finer::models {
// Tensor handles only. Models retain checkpoint naming, loading and TP slicing.
// Experts may be a stacked [E,N,K] Tensor or a list of [N,K] Tensors;
// keep that representation unchanged when constructing GroupedMatmul.
template <class Experts = forch::Tensor>
struct MoEWeights {
  Experts gate, up, down;
};

// Router and shared experts belong to the model. Both weight layouts use
// the same expert computation; only the dense reference needs stacked weights.
template <class Experts>
inline forch::Tensor moe(const forch::Tensor &x, const MoEWeights<Experts> &w,
  const forch::Tensor &scores, const forch::Tensor &indices, bool sparse,
  const forch::ProcessGroup &tp) {
  using namespace forch;
  using namespace forch::op;
  auto count = [](const Experts &weights) -> int64_t {
    if constexpr (std::is_same_v<Experts, Tensor>) {
      if (weights.dim() != 3)
        throw exception("moe: stacked expert weights must have rank 3");
      return weights.shape()[0];
    } else {
      return static_cast<int64_t>(weights.size());
    }
  };
  const auto experts = count(w.gate);
  if (experts <= 0 || count(w.up) != experts || count(w.down) != experts)
    throw forch::exception("moe: expected matching nonempty expert weights");
  if (x.dim() != 2 || scores.dim() != 2 || indices.shape() != scores.shape()
    || scores.shape()[0] != x.shape()[0] || indices.shape()[1] <= 0
    || indices.shape()[1] > experts || indices.dtype() != forch::DType::INT32)
    throw forch::exception(
      "moe: expected x [T,H] and scores/INT32 indices [T,K]");
  auto result = [&]() -> Tensor {
    if (sparse) {
      auto routing = ge::moe_init_routing(experts)(x, indices);
      auto g       = grouped_mat_mul(true)(
        routing.expanded_x, w.gate, routing.expert_tokens_count_or_cumsum);
      auto u = grouped_mat_mul(true)(
        routing.expanded_x, w.up, routing.expert_tokens_count_or_cumsum);
      auto y = grouped_mat_mul(true)(
        mul(silu(g), u), w.down, routing.expert_tokens_count_or_cumsum);
      return ge::moe_finalize_routing()(
        y, routing.expanded_row_idx, scores | cast(x.dtype()), indices);
    }
    auto stack = [](const Experts &weights) {
      if constexpr (std::is_same_v<Experts, Tensor>) {
        return weights;
      } else {
        auto out = weights[0]
          | reshape({1, weights[0].shape()[0], weights[0].shape()[1]});
        for (size_t i = 1; i < weights.size(); ++i)
          out = cat(int64_t{0})(out,
            weights[i]
              | reshape({1, weights[i].shape()[0], weights[i].shape()[1]}));
        return out;
      }
    };
    // Dense reference only: evaluate every expert on every token.
    const auto rows = x.shape()[0], width = x.shape()[1],
               k = indices.shape()[1];
    auto mask =
      one_hot(experts, -1, DType::INT32)(indices | reshape({rows * k}))
      | cast(x.dtype());
    auto weights = mul(mask, scores | cast(x.dtype()) | reshape({rows * k, 1}))
      | reshape({rows, k, experts}) | reduce_sum({1}, false) | permute({1, 0});
    auto tiled = x | reshape({1, rows, width}) | repeat({experts, 1, 1});
    auto g     = batched_mat_mul(false, true)(tiled, stack(w.gate));
    auto u     = batched_mat_mul(false, true)(tiled, stack(w.up));
    auto y     = batched_mat_mul(false, true)(mul(silu(g), u), stack(w.down));
    return mul(y, weights | reshape({experts, rows, 1}))
      | reduce_sum({0}, false);
  }();
  return tp.world_size() > 1 ? all_reduce(result, tp) : result;
}

} // namespace finer::models
