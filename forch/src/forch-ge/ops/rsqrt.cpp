#include "forch/ops/rsqrt.hpp"
#include "op-registry.hpp"

#include "es_Rsqrt_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Rsqrt>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Rsqrt *) {
      return EsRsqrt(ctx.inputs[0]);
    });
}

} // namespace
