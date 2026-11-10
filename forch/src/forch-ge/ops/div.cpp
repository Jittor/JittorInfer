#include "forch/ops/div.hpp"
#include "op-registry.hpp"

#include "es_Div_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Div>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Div * /*op*/) {
      return EsDiv(ctx.inputs[0], ctx.inputs[1]);
    });
}

} // namespace
