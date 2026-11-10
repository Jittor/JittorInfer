#include "forch/ops/collective.hpp"
#include "hccl-common.hpp"
#include "op-registry.hpp"

#include "es_HcomAllGather_c.h"
#include "es_HcomAllReduce_c.h"
#include "es_HcomAllToAll_c.h"
#include "es_HcomBroadcast_c.h"
#include "es_HcomReduceScatter_c.h"
#include "es_HcomReduce_c.h"

#include <array>

namespace {

// HCCL limits belong to graph emission, not public tensor metadata. BF16 is
// verified on 910B even for primitives whose proto omits it.
void check_hccl_dtype(const char *who, forch::DType dtype) {
  using forch::DType;
  switch (dtype) {
  case DType::FP32:
  case DType::FP16:
  case DType::BF16:
  case DType::INT32:
  case DType::INT64:
  case DType::INT8:
    return;
  default:
    throw forch::exception(std::string(who)
      + ": HCCL does not move this dtype (FP64 and BOOL are not accepted)");
  }
}

// HcomAllReduce's proto defaults fusion to 1, which routes through
// AutoTuningHcomAllReduceFusion. 0 is the path that is measured working;
// fusion is a later performance question, not a correctness one.
constexpr int64_t kFusion   = 0;
constexpr int64_t kFusionId = -1;

__attribute__((constructor)) void register_collective_ops() {
  using namespace forch::detail;

  GEOpRegistry::register_op<forch::op::AllReduce>(
    [](const GEOpContext &ctx, const forch::op::AllReduce *op) {
      check_hccl_dtype("all_reduce", ctx.input_tensors[0].dtype());
      return EsHcomAllReduce(ctx.inputs[0], hccl_reduce_op(op->reduce_op),
        hccl_group_name(op->group.group_id()).c_str(), kFusion, kFusionId);
    });

  GEOpRegistry::register_op<forch::op::AllGather>(
    [](const GEOpContext &ctx, const forch::op::AllGather *op) {
      check_hccl_dtype("all_gather", ctx.input_tensors[0].dtype());
      // rank_size is the 2nd argument here, the 4th in ReduceScatter
      return EsHcomAllGather(ctx.inputs[0], op->group.world_size(),
        hccl_group_name(op->group.group_id()).c_str(), kFusion, kFusionId);
    });

  GEOpRegistry::register_op<forch::op::ReduceScatter>(
    [](const GEOpContext &ctx, const forch::op::ReduceScatter *op) {
      check_hccl_dtype("reduce_scatter", ctx.input_tensors[0].dtype());
      // rank_size 4th, reduction 2nd; not the AllGather layout
      return EsHcomReduceScatter(ctx.inputs[0], hccl_reduce_op(op->reduce_op),
        hccl_group_name(op->group.group_id()).c_str(), op->group.world_size(),
        kFusion, kFusionId);
    });

  GEOpRegistry::register_op<forch::op::Reduce>(
    [](const GEOpContext &ctx, const forch::op::Reduce *op) {
      check_hccl_dtype("reduce", ctx.input_tensors[0].dtype());
      // root_rank is the 2nd argument
      return EsHcomReduce(ctx.inputs[0], op->root_rank,
        hccl_reduce_op(op->reduce_op),
        hccl_group_name(op->group.group_id()).c_str(), kFusion, kFusionId);
    });

  GEOpRegistry::register_op<forch::op::AllToAll>(
    [](const GEOpContext &ctx, const forch::op::AllToAll *op) {
      check_hccl_dtype("all_to_all", ctx.input_tensors[0].dtype());
      return EsHcomAllToAll(
        ctx.inputs[0], hccl_group_name(op->group.group_id()).c_str());
    });

  GEOpRegistry::register_op<forch::op::Broadcast>(
    [](const GEOpContext &ctx, const forch::op::Broadcast *op) {
      check_hccl_dtype("broadcast", ctx.input_tensors[0].dtype());
      // x is a handle array (plus a count), not a single handle; y is too.
      std::array<EsCTensorHolder *, 1> x_arr{ctx.inputs[0]};
      auto out = EsHcomBroadcast(x_arr.data(), /*x_num=*/1,
        /*y_num=*/1, op->root_rank,
        hccl_group_name(op->group.group_id()).c_str(), kFusion, kFusionId);
      if (out.y == nullptr || out.y_num < 1) {
        throw forch::exception(
          "broadcast: EsHcomBroadcast produced no output handle");
      }
      return out.y[0];
    });
}

} // namespace
