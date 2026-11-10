#include "forch/ops/slice.hpp"
#include "forch/exception.hpp"
#include "forch/forch.hpp"

namespace forch::op {

Slice::IntervalList Slice::to_interval_list(
  const ToIntervalList &to_intervals, const Shape &input_shape) {
  auto ellipsis_cnt = std::count_if(to_intervals.begin(), to_intervals.end(),
    [](const ToInterval &x) { return std::holds_alternative<Ellipsis>(x); });

  if (ellipsis_cnt > 1) {
    throw exception("Only one ellipsis is allowed in slice intervals");
  } else if (ellipsis_cnt == 1) {
    if (to_intervals.size() > input_shape.size() + 1) {
      throw exception("Too many slice intervals");
    }
  } else if (ellipsis_cnt == 0) {
    if (to_intervals.size() != input_shape.size()) {
      throw exception("Number of slice intervals must match input shape");
    }
  }

  auto normalize_interval = [](int64_t begin, int64_t end, int64_t size) {
    if (begin < 0) {
      begin += size;
    }
    if (end < 0) {
      end += size;
    }
    if (begin < 0 || begin > size || end < 0 || end > size || begin > end) {
      throw exception("Invalid slice interval");
    }
    return Slice::Interval{begin, end};
  };

  auto res         = IntervalList{};
  auto it_interval = to_intervals.begin();
  auto it_shape    = input_shape.begin();
  for (; it_interval != to_intervals.end(); it_interval++) {
    if (std::holds_alternative<Ellipsis>(*it_interval)) {
      size_t remaining_dims = input_shape.size() - to_intervals.size() + 1;
      for (size_t i = 0; i < remaining_dims; ++i) {
        res.emplace_back(0, *it_shape++);
      }
    } else if (std::holds_alternative<IntervalAll>(*it_interval)) {
      auto begin = 0;
      auto end   = *it_shape;
      res.push_back(normalize_interval(begin, end, *it_shape++));
    } else if (std::holds_alternative<IntervalFromBegin>(*it_interval)) {
      auto begin = 0;
      auto end   = std::get<IntervalFromBegin>(*it_interval).end;
      res.push_back(normalize_interval(begin, end, *it_shape++));
    } else if (std::holds_alternative<IntervalTillEnd>(*it_interval)) {
      auto begin = std::get<IntervalTillEnd>(*it_interval).begin;
      auto end   = *it_shape;
      res.push_back(normalize_interval(begin, end, *it_shape++));
    } else if (std::holds_alternative<Interval>(*it_interval)) {
      auto [begin, end] = std::get<Interval>(*it_interval);
      res.push_back(normalize_interval(begin, end, *it_shape++));
    }
  }
  return res;
}

} // namespace forch::op