#include "forch/ops/rotary_position_embedding.hpp"
#include "op-registry.hpp"

#include "es_RotaryPositionEmbedding_c.h"

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::RotaryPositionEmbedding>(
    [](const forch::detail::GEOpContext        &ctx,
      const forch::op::RotaryPositionEmbedding *op) {
      return EsRotaryPositionEmbedding(ctx.inputs[0], ctx.inputs[1],
        ctx.inputs[2],
        /*rotate=*/nullptr, static_cast<int64_t>(op->mode));
    });
}
} // namespace
