#include "forch/ops/scatter_update.hpp"
#include "op-registry.hpp"

#include "es_ScatterUpdate_c.h"

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::ScatterUpdate>(
    [](const forch::detail::GEOpContext &ctx,
      const forch::op::ScatterUpdate * /*op*/) {
      return EsScatterUpdate(
        ctx.inputs[0], ctx.inputs[1], ctx.inputs[2], /*use_locking=*/false);
    });
}
} // namespace
