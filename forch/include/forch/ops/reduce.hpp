#pragma once

#include "forch/exception.hpp"
#include "forch/ops/function.hpp"

#include <utility>
#include <vector>

namespace forch::op {

/// Shared shape inference for the reduce family: dims listed in `axes` are
/// dropped (or kept as 1 when keep_dims), negative axes normalized.
inline Shape reduce_output_shape(
  const TensorList &input, const std::vector<int64_t> &axes, bool keep_dims) {
  const auto &x    = input[0];
  const auto  rank = static_cast<int64_t>(x.shape().size());

  if (axes.empty()) {
    throw exception("reduce: axes must not be empty");
  }
  std::vector<bool> reduced(rank, false);
  for (auto a : axes) {
    auto ndim = a < 0 ? a + rank : a;
    if (ndim < 0 || ndim >= rank) {
      throw exception("reduce: axis out of range");
    }
    reduced[ndim] = true;
  }

  Shape out;
  for (int64_t i = 0; i < rank; ++i) {
    if (!reduced[i]) {
      out.push_back(x.shape()[i]);
    } else if (keep_dims) {
      out.push_back(1);
    }
  }
  return out;
}

struct ReduceSum : public Function {
  std::vector<int64_t> axes;
  bool                 keep_dims;

  ReduceSum(std::vector<int64_t> axes, bool keep_dims)
      : axes(std::move(axes)), keep_dims(keep_dims) {}

  Shape shape(const TensorList &input) const override {
    return reduce_output_shape(input, axes, keep_dims);
  }

  DType dtype(const TensorList &input) const override {
    return input[0].dtype();
  }
};

struct ReduceMax : public Function {
  std::vector<int64_t> axes;
  bool                 keep_dims;

  ReduceMax(std::vector<int64_t> axes, bool keep_dims)
      : axes(std::move(axes)), keep_dims(keep_dims) {}

  Shape shape(const TensorList &input) const override {
    return reduce_output_shape(input, axes, keep_dims);
  }

  DType dtype(const TensorList &input) const override {
    return input[0].dtype();
  }
};

/// Sum over a nonempty list of axes, preserving the input dtype.
/// Negative axes are supported; keep_dims retains reduced axes with size 1.
inline auto reduce_sum(std::vector<int64_t> axes, bool keep_dims = false) {
  return TensorBuildCurry<1>{
    std::make_unique<ReduceSum>(std::move(axes), keep_dims)};
}

/// Take the maximum over a nonempty list of axes, preserving the input dtype.
/// Negative axes are supported; keep_dims retains reduced axes with size 1.
inline auto reduce_max(std::vector<int64_t> axes, bool keep_dims = false) {
  return TensorBuildCurry<1>{
    std::make_unique<ReduceMax>(std::move(axes), keep_dims)};
}

} // namespace forch::op
