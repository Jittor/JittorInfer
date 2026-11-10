#include "forch/ops/select.hpp"
#include "op-registry.hpp"

#include "es_SelectV2_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Select>(
    [](
      const forch::detail::GEOpContext &ctx, const forch::op::Select * /*op*/) {
      return EsSelectV2(ctx.inputs[0], ctx.inputs[1], ctx.inputs[2]);
    });
}

} // namespace
