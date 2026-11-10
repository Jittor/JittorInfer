#include "forch/ops/rms_norm.hpp"
#include "op-registry.hpp"

#include "es_RmsNorm_c.h"

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::RmsNorm>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::RmsNorm *op) {
      return EsRmsNorm(ctx.inputs[0], ctx.inputs[1], op->epsilon).y;
    });
}
} // namespace
