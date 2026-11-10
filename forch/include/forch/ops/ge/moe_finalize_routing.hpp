// Weighted scatter-back for sparse MoE dispatch: combine the expanded
// (per-expert-sorted) expert outputs back into the original token rows.
//
// PUBLIC CONTRACT (token-major): row_idx is the TOKEN-MAJOR-keyed rank
// table exactly as `moe_init_routing` emits it: a 1D [T*k] INT32 array
// where position p = t*k+s holds the sorted-position of the routing entry
// (token t, slot s). The caller feeds `routing.expanded_row_idx` straight
// in — the ops connect directly, no caller-side rearrange.
//
//   y(i,j) = sum_s scales(i,s) * expanded_x[ row_idx[i*k+s] ]
//
// Inputs: (expanded_x [T*k, D], row_idx [T*k] INT32, scales [T, k],
// expert_idx [T, k] — required even without bias, verified on 910B).
// Output y = [T, D]. No attrs: T/k derive from the scales shape in
// shape() (which doubles as contract validation at tensor construction).
//
// The CANN kernel layout selection (drop_pad_mode) is pinned by the GE
// mapping — see src/forch-ge/ops/moe_finalize_routing.cpp.
#pragma once

#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/ops/function.hpp"

#include <memory>
#include <utility>

namespace forch::op::ge {

/// Weighted scatter-back. Inputs: (expanded_x [T*k, D], row_idx [T*k]
/// token-major keyed, scales [T, k], expert_idx [T, k]; x1/x2/bias unused:
/// nullptr). Output y = [T, D].
struct MoeFinalizeRouting : public Function {
  Shape shape(const TensorList &input) const override {
    // token-major contract, checked at tensor construction: scales [T, k],
    // row_idx 1D [T*k] (T = scales rows, k = scales cols)
    if (input.size() != 4) {
      throw exception(
        "moe_finalize_routing: requires 4 inputs (expanded_x, row_idx, "
        "scales, expert_idx)");
    }
    const Shape &ss = input[2].shape();
    const Shape &rs = input[1].shape();
    if (ss.size() != 2 || ss[0] <= 0 || ss[1] <= 0) {
      throw exception("moe_finalize_routing: scales must be 2D [T, k]");
    }
    if (rs.size() != 1 || rs[0] != ss[0] * ss[1]) {
      throw exception("moe_finalize_routing: row_idx must be 1D [T*k] "
                      "(token-major keyed, as moe_init_routing emits)");
    }
    if (input[3].shape() != ss) {
      throw exception(
        "moe_finalize_routing: expert_idx must be [T, k] (required even "
        "without bias — the kernel rejects a missing input)");
    }
    return {ss[0], input[0].shape().back()}; // [T, D]
  }
  DType dtype(const TensorList &input) const override {
    return input[0].dtype();
  }
};

inline auto moe_finalize_routing() {
  return TensorBuildCurry<4>{std::make_unique<MoeFinalizeRouting>()};
}

} // namespace forch::op::ge
