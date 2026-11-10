// One-hot encoding: for each element of `indices` (INT32/INT64), emit a
// one-hot vector of length `depth` filled with on=1 / off=0 in the output
// dtype, inserted at position `axis` (-1 = append). Output rank is
// rank(indices) + 1.
//
// Usage:
//   one_hot(depth, axis = -1, dtype = DType::INT32)
//   e.g. indices [T*k] INT32, depth=E -> [T*k, E]
//
// Backend notes (supported output dtypes, kernel quirks) live in the GE
// mapping — see src/forch-ge/ops/one_hot.cpp.
#pragma once

#include "forch/exception.hpp"
#include "forch/ops/function.hpp"

#include <limits>
#include <utility>

namespace forch::op {

/// One-hot encoding. `depth` and on/off values (1/0) are folded in as
/// attributes; `axis` is the insertion position in [-1, rank] (-1 =
/// append), normalized in shape(). `depth` must fit int32.
struct OneHot : public Function {
  int64_t depth;
  int64_t axis; // raw insertion position, [-1, rank]
  DType   dtype_;

  // normalized insertion position, resolved in shape() and read by the
  // backend mapping.
  mutable int64_t axis_resolved = -1;

  OneHot(int64_t depth, int64_t axis, DType dtype)
      : depth(depth), axis(axis), dtype_(dtype) {
    if (depth < 0 || depth > std::numeric_limits<int32_t>::max()) {
      throw exception("one_hot: depth must fit in int32");
    }
  }

  Shape shape(const TensorList &input) const override {
    const auto &x    = input[0];
    const auto  rank = static_cast<int64_t>(x.shape().size());
    if (axis < -1 || axis > rank) {
      throw exception("one_hot: axis must be in [-1, rank]");
    }
    // The CANN spelling and the std::vector insert position are separate
    // values — see the GE mapping comment for the spelling rules.
    axis_resolved           = (axis < 0 || axis == rank) ? -1 : axis;
    const int64_t insert_at = axis_resolved < 0 ? rank : axis_resolved;
    if (x.dtype() != DType::INT32 && x.dtype() != DType::INT64) {
      throw exception("one_hot: indices must be INT32 or INT64");
    }
    Shape out = x.shape();
    out.insert(out.begin() + static_cast<size_t>(insert_at), depth);
    return out;
  }

  DType dtype(const TensorList & /*unused*/) const override {
    return dtype_;
  }
};

inline auto one_hot(
  int64_t depth, int64_t axis = -1, DType dtype = DType::INT32) {
  return TensorBuildCurry<1>{std::make_unique<OneHot>(depth, axis, dtype)};
}

} // namespace forch::op
