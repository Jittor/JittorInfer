#include "forch/ops/tril.hpp"
#include "op-registry.hpp"

#include "es_Tril_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Tril>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Tril *op) {
      return EsTril(ctx.inputs[0], op->diagonal);
    });
}

} // namespace
