#include "forch/ops/reshape.hpp"
#include "op-registry.hpp"

#include "es_Reshape_c.h"

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Reshpae>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Reshpae *op) {
      auto *shape = EsCreateVectorInt64(ctx.builder, op->new_shape.data(),
        static_cast<int64_t>(op->new_shape.size()));
      return EsReshape(ctx.inputs[0], shape, /*axis=*/0, /*num_axes=*/-1);
    });
}
} // namespace
