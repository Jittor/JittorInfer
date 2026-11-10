#include "forch/ops/ge/moe_finalize_routing.hpp"
#include "op-registry.hpp"

#include "es_MoeFinalizeRoutingV2_c.h"

namespace {

// MoeFinalizeRouting (CANN MoeFinalizeRoutingV2) GE mapping: weighted
// scatter-back of the expanded expert outputs into the original token
// rows. The facade's public contract is a TOKEN-MAJOR keyed row_idx (=
// moe_init_routing's output format); the kernel's drop_pad_mode attr
// selects which arrangement the incoming expandedRowIdx uses —
//   0: dropless, column-arranged (slot-major keyed, aclnnMoeInitRouting
//      / V1 format)
//   1: drop-pad,  column-arranged (V1 format)
//   2: dropless, ROW-arranged (token-major keyed, aclnnMoeInitRoutingV2
//      format)
//   3: drop-pad,  row-arranged (V2 format)
// so the mapping pins mode 2 (dropless + token-major) and the pair
// connects with no rearrange.
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::ge::MoeFinalizeRouting>(
    [](const forch::detail::GEOpContext &ctx,
      const forch::op::ge::MoeFinalizeRouting * /*op*/) {
      return EsMoeFinalizeRoutingV2(ctx.inputs[0], ctx.inputs[1],
        /*x1=*/nullptr,
        /*x2=*/nullptr, /*bias=*/nullptr, ctx.inputs[2], ctx.inputs[3],
        /*drop_pad_mode=*/2);
    });
}

} // namespace
