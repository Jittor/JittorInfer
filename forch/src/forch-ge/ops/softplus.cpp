#include "forch/ops/softplus.hpp"
#include "op-registry.hpp"

#include "es_Softplus_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Softplus>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Softplus *) {
      return EsSoftplus(ctx.inputs[0]);
    });
}

} // namespace
