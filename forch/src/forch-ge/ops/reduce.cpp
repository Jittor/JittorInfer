#include "forch/ops/reduce.hpp"
#include "op-registry.hpp"

#include "es_ReduceMax_c.h"
#include "es_ReduceSum_c.h"

namespace {

__attribute__((constructor)) void register_reduce_ops() {
  forch::detail::GEOpRegistry::register_op<forch::op::ReduceSum>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::ReduceSum *op) {
      auto *axes = EsCreateVectorInt64(
        ctx.builder, op->axes.data(), static_cast<int64_t>(op->axes.size()));
      return EsReduceSum(
        ctx.inputs[0], axes, op->keep_dims, /*noop_with_empty_axes=*/false);
    });

  forch::detail::GEOpRegistry::register_op<forch::op::ReduceMax>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::ReduceMax *op) {
      auto *axes = EsCreateVectorInt64(
        ctx.builder, op->axes.data(), static_cast<int64_t>(op->axes.size()));
      return EsReduceMax(
        ctx.inputs[0], axes, op->keep_dims, /*noop_with_empty_axes=*/false);
    });
}

} // namespace
