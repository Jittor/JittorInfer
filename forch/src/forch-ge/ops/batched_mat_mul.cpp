#include "forch/ops/batched_mat_mul.hpp"
#include "op-registry.hpp"

#include "es_BatchMatMulV2_c.h"

// TODO: 检验 BatchMatMul 支持几维，已知 3*3, 3*4 可用。
namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::BatchedMatMul>(
    [](const forch::detail::GEOpContext &ctx,
      const forch::op::BatchedMatMul    *op) {
      return EsBatchMatMulV2(ctx.inputs[0], ctx.inputs[1], nullptr, nullptr,
        op->transpose_a, op->transpose_b, /*offset_x=*/0);
    });
}
} // namespace
