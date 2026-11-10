#include "forch/ops/cast.hpp"
#include "forch-ge-common.hpp"
#include "op-registry.hpp"

#include "es_Cast_c.h"

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Cast>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Cast *op) {
      return EsCast(ctx.inputs[0],
        static_cast<int64_t>(forch::detail::CDtypeDesc(op->dtype()).type));
    });
}
} // namespace
