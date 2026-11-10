#include "forch/ops/triu.hpp"
#include "op-registry.hpp"

#include "es_Triu_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Triu>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Triu *op) {
      return EsTriu(ctx.inputs[0], op->diagonal);
    });
}

} // namespace
