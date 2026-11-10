#pragma once

// Header-only translation of public collective attributes to HCCL vocabulary.
#include "forch/ops/collective.hpp"

#include <cstdint>
#include <string>

namespace forch::detail {

// GE creates the default group from the rank table. Nonzero ids only name
// groups; this backend does not yet create subgroups or additional worlds.
inline std::string hccl_group_name(int64_t group_id) {
  return group_id == 0 ? "hccl_world_group" : std::to_string(group_id);
}

inline const char *hccl_reduce_op(op::ReduceOp reduce_op) {
  using op::ReduceOp;
  switch (reduce_op) {
  case ReduceOp::Sum:
    return "sum";
  case ReduceOp::Prod:
    return "prod";
  case ReduceOp::Max:
    return "max";
  case ReduceOp::Min:
    return "min";
  }
  throw exception("forch-ge: unknown ReduceOp");
}

} // namespace forch::detail
