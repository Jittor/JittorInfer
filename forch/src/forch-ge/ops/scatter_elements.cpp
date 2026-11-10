#include "forch/ops/scatter_elements.hpp"
#include "op-registry.hpp"

// ScatterElementsV2's AscendC kernel fails to compile in CANN 9.0.0
// ("would lose const qualifier"), so this maps to the legacy TBE
// ScatterElements (V1), which has the same torch.scatter_ semantics.
#include "es_ScatterElements_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::ScatterElements>(
    [](const forch::detail::GEOpContext &ctx,
      const forch::op::ScatterElements  *op) {
      return EsScatterElements(ctx.inputs[0], ctx.inputs[1], ctx.inputs[2],
        op->axis, op->reduction.c_str());
    });
}

} // namespace
