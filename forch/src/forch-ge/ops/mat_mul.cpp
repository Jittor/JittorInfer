#include "forch/ops/mat_mul.hpp"
#include "op-registry.hpp"

#include "es_MatMul_c.h"

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::MatMul>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::MatMul *op) {
      return EsMatMul(ctx.inputs[0], ctx.inputs[1], nullptr, op->transpose_a,
        op->transpose_b);
    });
}
} // namespace
