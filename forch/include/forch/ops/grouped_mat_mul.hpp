#pragma once

#include "forch/exception.hpp"
#include "forch/ops/function.hpp"

#include <utility>

namespace forch::op {

/// Apply a separate matrix multiplication to each contiguous row group of x.
/// Equivalent to per-group mat_mul followed by cat along the row dimension.
//
// x is [M, K]. Weights are stacked [E, K, N] or E separate [K, N] tensors;
// transpose_weight swaps the last two weight dimensions before multiplying.
// group_list is [E] INT64: nonnegative, nondecreasing exclusive end offsets,
// with an implicit initial offset of 0 and a final offset of M. Equal adjacent
// offsets represent empty groups. Output is [M, N], preserving x's row order
// and dtype; x and all weights must have the same dtype.
struct GroupedMatMul : public Function {
  bool transpose_weight;

  enum class WeightForm {
    Stacked, // one [E, K, N] (or [E, N, K]) tensor
    List,    // E separate 2D tensors
  };

  // Resolved from weights by shape() during Tensor construction.
  mutable WeightForm weight_form  = WeightForm::List;
  mutable int64_t    num_groups   = -1;
  mutable int64_t    out_features = -1;

  explicit GroupedMatMul(bool transpose_weight)
      : transpose_weight(transpose_weight) {}

  // inputs: [x, weight, group_list] (stacked) or
  //         [x, weight_0, ..., weight_{E-1}, group_list] (list)
  Shape shape(const TensorList &input) const override {
    if (input.size() < 3) {
      throw exception("grouped_mat_mul: requires x, weight(s) and group_list");
    }
    const Shape &xs = input[0].shape(); // [M, K]
    if (xs.size() != 2) {
      throw exception("grouped_mat_mul: x must be 2D");
    }
    const Shape &gl = input.back().shape();
    if (gl.size() != 1) {
      throw exception("grouped_mat_mul: group_list must be 1D");
    }

    if (input.size() == 3 && input[1].shape().size() == 3) {
      // stacked: [E, K, N] or [E, N, K]
      weight_form     = WeightForm::Stacked;
      const Shape &ws = input[1].shape();
      num_groups      = ws[0];
      out_features    = transpose_weight ? ws[1] : ws[2];
      const int64_t k = transpose_weight ? ws[2] : ws[1];
      if (k != xs[1]) {
        throw exception(
          "grouped_mat_mul: weight K must match x K after transpose");
      }
      if (gl[0] != num_groups) {
        throw exception("grouped_mat_mul: group_list length must equal E");
      }
      return {xs[0], out_features};
    }

    // All groups must share N so their results form a single [M, N] tensor.
    weight_form = WeightForm::List;
    num_groups  = static_cast<int64_t>(input.size()) - 2;
    if (gl[0] != num_groups) {
      throw exception(
        "grouped_mat_mul: group_list must have one entry per weight");
    }
    out_features = -1;
    for (int64_t e = 0; e < num_groups; ++e) {
      const Shape &ws = input[static_cast<size_t>(e) + 1].shape();
      if (ws.size() != 2) {
        throw exception("grouped_mat_mul: each weight must be 2D");
      }
      const int64_t k = transpose_weight ? ws[1] : ws[0];
      const int64_t n = transpose_weight ? ws[0] : ws[1];
      if (k != xs[1]) {
        throw exception(
          "grouped_mat_mul: weight K must match x K after transpose");
      }
      if (out_features == -1) {
        out_features = n;
      } else if (n != out_features) {
        throw exception("grouped_mat_mul: all weights must share the same N");
      }
    }
    return {xs[0], out_features};
  }

  DType dtype(const TensorList &input) const override {
    if (weight_form == WeightForm::Stacked) {
      if (input[1].dtype() != input[0].dtype()) {
        throw exception("grouped_mat_mul: x and weight must share a dtype");
      }
    } else {
      for (int64_t e = 0; e < num_groups; ++e) {
        if (input[static_cast<size_t>(e) + 1].dtype() != input[0].dtype()) {
          throw exception(
            "grouped_mat_mul: x and all weights must share the same dtype");
        }
      }
    }
    return input[0].dtype();
  }

  struct Builder {
    bool transpose_weight;

    /// stacked: one [E, K, N] (transpose_weight=false) or [E, N, K] (true)
    Tensor operator()(Tensor x, Tensor weight, Tensor group_list) const {
      return {
        TensorList{std::move(x), std::move(weight), std::move(group_list)},
        std::make_unique<GroupedMatMul>(transpose_weight)};
    }

    /// list: E separate [K, N] / [N, K] tensors, one per row group.
    Tensor operator()(
      Tensor x, const TensorList &weights, Tensor group_list) const {
      if (weights.empty()) {
        throw exception("grouped_mat_mul: weights must not be empty");
      }
      TensorList inputs;
      inputs.reserve(2 + weights.size());
      inputs.push_back(std::move(x));
      for (const auto &weight : weights) {
        inputs.push_back(weight);
      }
      inputs.push_back(std::move(group_list));
      return {
        std::move(inputs), std::make_unique<GroupedMatMul>(transpose_weight)};
    }
  };
};

inline GroupedMatMul::Builder grouped_mat_mul(bool transpose_weight = false) {
  return {transpose_weight};
}

} // namespace forch::op
