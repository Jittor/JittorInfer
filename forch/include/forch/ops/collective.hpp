#pragma once

#include "forch/distributed.hpp"
#include "forch/exception.hpp"
#include "forch/ops/function.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace forch::op {

/** Reduction applied by the reducing collectives. */
enum class ReduceOp { Sum, Prod, Max, Min };

/**
 * Collective primitives, named after torch.distributed. The participating
 * ranks come in as a `ProcessGroup` argument; which collective library ends up
 * carrying the traffic is the backend's business and does not appear here.
 */

/** Rejects a root that is not a member of the group. */
inline void check_root_rank(
  const char *who, const ProcessGroup &group, int64_t root_rank) {
  if (root_rank < 0 || root_rank >= group.world_size()) {
    throw exception(std::string(who) + ": root_rank "
      + std::to_string(root_rank) + " is outside [0, "
      + std::to_string(group.world_size()) + ")");
  }
}

/** Elementwise reduction across the group; every rank gets the result. */
struct AllReduce : public Function {
  ProcessGroup group;
  ReduceOp     reduce_op;

  AllReduce(ProcessGroup group, ReduceOp reduce_op)
      : group(group), reduce_op(reduce_op) {}

  Shape shape(const TensorList &input) const override {
    return input[0].shape();
  }
};

/** Concatenate every rank's input along dim 0. */
struct AllGather : public Function {
  ProcessGroup group;

  explicit AllGather(ProcessGroup group) : group(group) {}

  Shape shape(const TensorList &input) const override {
    auto x = input[0].shape();
    if (x.empty()) {
      throw exception("all_gather: input must have at least 1 dimension");
    }
    x[0] *= group.world_size();
    return x;
  }
};

/** Reduce, then keep only this rank's block of dim 0. */
struct ReduceScatter : public Function {
  ProcessGroup group;
  ReduceOp     reduce_op;

  ReduceScatter(ProcessGroup group, ReduceOp reduce_op)
      : group(group), reduce_op(reduce_op) {}

  Shape shape(const TensorList &input) const override {
    auto x = input[0].shape();
    if (x.empty()) {
      throw exception("reduce_scatter: input must have at least 1 dimension");
    }
    if (x[0] % group.world_size() != 0) {
      throw exception("reduce_scatter: dim 0 (" + std::to_string(x[0])
        + ") must be divisible by world_size ("
        + std::to_string(group.world_size()) + ")");
    }
    x[0] /= group.world_size();
    return x;
  }
};

/** Broadcast root_rank's input to the whole group. */
struct Broadcast : public Function {
  ProcessGroup group;
  int64_t      root_rank;

  Broadcast(ProcessGroup group, int64_t root_rank)
      : group(group), root_rank(root_rank) {}

  Shape shape(const TensorList &input) const override {
    check_root_rank("broadcast", group, root_rank);
    return input[0].shape();
  }
};

/** Like all_reduce, but only root_rank receives the result. */
struct Reduce : public Function {
  ProcessGroup group;
  int64_t      root_rank;
  ReduceOp     reduce_op;

  Reduce(ProcessGroup group, int64_t root_rank, ReduceOp reduce_op)
      : group(group), root_rank(root_rank), reduce_op(reduce_op) {}

  Shape shape(const TensorList &input) const override {
    check_root_rank("reduce", group, root_rank);
    return input[0].shape();
  }
};

/** Every rank sends block i of dim 0 to rank i. Shape is preserved. */
struct AllToAll : public Function {
  ProcessGroup group;

  explicit AllToAll(ProcessGroup group) : group(group) {}

  Shape shape(const TensorList &input) const override {
    const auto &x = input[0].shape();
    if (x.empty()) {
      throw exception("all_to_all: input must have at least 1 dimension");
    }
    if (x[0] % group.world_size() != 0) {
      throw exception("all_to_all: dim 0 (" + std::to_string(x[0])
        + ") must be divisible by world_size ("
        + std::to_string(group.world_size()) + ")");
    }
    return x;
  }
};

inline Tensor all_reduce(
  const Tensor &x, ProcessGroup group, ReduceOp reduce_op = ReduceOp::Sum) {
  return TensorBuildCurry<1>{std::make_unique<AllReduce>(group, reduce_op)}(x);
}

inline Tensor all_gather(const Tensor &x, ProcessGroup group) {
  return TensorBuildCurry<1>{std::make_unique<AllGather>(group)}(x);
}

inline Tensor reduce_scatter(
  const Tensor &x, ProcessGroup group, ReduceOp reduce_op = ReduceOp::Sum) {
  return TensorBuildCurry<1>{std::make_unique<ReduceScatter>(group, reduce_op)}(
    x);
}

inline Tensor broadcast(
  const Tensor &x, ProcessGroup group, int64_t root_rank = 0) {
  return TensorBuildCurry<1>{std::make_unique<Broadcast>(group, root_rank)}(x);
}

inline Tensor reduce(const Tensor &x, ProcessGroup group, int64_t root_rank = 0,
  ReduceOp reduce_op = ReduceOp::Sum) {
  return TensorBuildCurry<1>{
    std::make_unique<Reduce>(group, root_rank, reduce_op)}(x);
}

inline Tensor all_to_all(const Tensor &x, ProcessGroup group) {
  return TensorBuildCurry<1>{std::make_unique<AllToAll>(group)}(x);
}

} // namespace forch::op
