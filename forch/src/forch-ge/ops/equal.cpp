#include "forch/ops/equal.hpp"
#include "op-registry.hpp"

#include "es_Equal_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Equal>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Equal * /*op*/) {
      return EsEqual(ctx.inputs[0], ctx.inputs[1]);
    });
}

} // namespace
