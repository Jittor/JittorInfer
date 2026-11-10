// [minimal-impl] KEEP: used by the DeepSeek-V2 gate (group_limited_greedy):
//   group_mask = scatter_elements(1, "add")(
//     zeros [T, n_group], group_idx [T, topk_group], ones [T, topk_group])
// (The MoE combine step is MoeFinalizeRouting — NOT this op.)
#pragma once

#include "forch/exception.hpp"
#include "forch/ops/function.hpp"

#include <string>
#include <utility>

namespace forch::op {

/// Element-wise scatter along an axis with an optional reduction on
/// duplicate writes. Backed by CANN's legacy ScatterElements (V1; the V2
/// AscendC kernel fails to compile under CANN 9.0.0), torch.scatter_-
/// style, verified on 910B for exactly this contract:
///   - indices and updates have identical shapes, same rank as var;
///   - on non-axis dims, indices/updates dims EQUAL var's — partial /
///     broadcast scatter is intentionally not exposed;
///   - indices must be INT32 or INT64;
///   - reduction: "none" (replace), "add", "mul"; with "add"/"mul",
///     duplicate indices accumulate (write order unspecified).
/// It is NOT a 1-D row-index index_add replacement — for row scatter keep
/// indices 2-D and scatter along axis 0.
struct ScatterElements : public Function {
  int64_t     axis;
  std::string reduction;

  ScatterElements(int64_t axis, std::string reduction)
      : axis(axis), reduction(std::move(reduction)) {
    if (this->reduction != "none" && this->reduction != "add"
      && this->reduction != "mul") {
      throw exception("scatter_elements: reduction must be none/add/mul");
    }
  }

  Shape shape(const TensorList &input) const override {
    if (input.size() != 3) {
      throw exception(
        "scatter_elements: requires 3 inputs (var, indices, updates)");
    }
    const auto &var     = input[0];
    const auto &indices = input[1];
    const auto &updates = input[2];

    if (indices.dtype() != DType::INT32 && indices.dtype() != DType::INT64) {
      throw exception("scatter_elements: indices must be INT32 or INT64");
    }
    // torch.scatter_ contract: indices and updates share one shape, same
    // rank as var.
    if (indices.dim() != var.dim() || updates.dim() != var.dim()) {
      throw exception(
        "scatter_elements: indices and updates must have the same rank as var");
    }
    if (indices.shape() != updates.shape()) {
      throw exception(
        "scatter_elements: indices and updates must have identical shapes");
    }
    auto ndim = axis < 0 ? axis + var.dim() : axis;
    if (ndim < 0 || ndim >= var.dim()) {
      throw exception("scatter_elements: axis out of range");
    }
    for (int64_t i = 0; i < var.dim(); ++i) {
      if (i != ndim && updates.shape()[i] != var.shape()[i]) {
        throw exception(
          "scatter_elements: updates must match var on non-axis dims "
          "(partial scatter is not exposed)");
      }
    }
    return var.shape();
  }

  DType dtype(const TensorList &input) const override {
    if (input[0].dtype() != input[2].dtype()) {
      throw exception("scatter_elements: var and updates must share dtype");
    }
    return input[0].dtype();
  }
};

inline auto scatter_elements(
  int64_t axis = 0, const std::string &reduction = "none") {
  return TensorBuildCurry<3>{
    std::make_unique<ScatterElements>(axis, reduction)};
}

} // namespace forch::op
