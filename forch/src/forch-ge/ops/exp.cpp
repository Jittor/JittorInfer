#include "forch/ops/exp.hpp"
#include "op-registry.hpp"

#include "es_Exp_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Exp>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Exp *) {
      // base 传 GE 约定的 -1(自然常数 e),scale/shift 恒等,即 y = e^x
      return EsExp(
        ctx.inputs[0], /*base=*/-1.0F, /*scale=*/1.0F, /*shift=*/0.0F);
    });
}

} // namespace
