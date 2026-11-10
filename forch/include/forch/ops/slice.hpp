#pragma once

#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/ops/function.hpp"
#include <variant>

namespace forch::op {

struct Slice : public Function {
  struct IntervalAll {};
  struct IntervalFromBegin {
    int64_t end;
  };
  struct IntervalTillEnd {
    int64_t begin;
  };
  struct Interval {
    int64_t begin;
    int64_t end;
  };
  struct Ellipsis {};
  using ToInterval     = std::variant<IntervalAll, IntervalFromBegin,
        IntervalTillEnd, Interval, Ellipsis>;
  using ToIntervalList = std::vector<ToInterval>;

  using IntervalList = std::vector<Interval>;
  IntervalList slice;

  Slice(IntervalList slice) : slice(std::move(slice)) {}

  Shape shape(const TensorList & /*unused*/) const override {
    Shape output_shape;
    for (const auto &[begin, end] : slice) {
      output_shape.push_back(end - begin);
    }
    return output_shape;
  }

  static IntervalList to_interval_list(
    const ToIntervalList &to_intervals, const Shape &input_shape);

  struct Builder {
    ToIntervalList to_intervals;

    Builder(ToIntervalList to_intervals)
        : to_intervals(std::move(to_intervals)) {}

    Builder(const Shape &begin, const Shape &end) {
      if (begin.size() != end.size()) {
        throw exception("Slice begin and end must have the same dims");
      }
      for (size_t i = 0; i < begin.size(); ++i) {
        to_intervals.emplace_back(Interval{begin[i], end[i]});
      }
    }

    Tensor operator()(Tensor input) const {
      auto intervals = to_interval_list(to_intervals, input.shape());
      return {
        {std::move(input)}, std::make_unique<Slice>(std::move(intervals))};
    }
  };
};

inline Slice::Builder slice(const Slice::ToIntervalList &to_intervals) {
  return {to_intervals};
}

inline Slice::Builder slice(const Shape &begin, const Shape &end) {
  return {begin, end};
}

} // namespace forch::op