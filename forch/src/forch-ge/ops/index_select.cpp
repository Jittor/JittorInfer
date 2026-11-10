#include "forch/ops/index_select.hpp"
#include "op-registry.hpp"

#include "es_GatherV2_c.h"

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::IndexSelect>(
    [](
      const forch::detail::GEOpContext &ctx, const forch::op::IndexSelect *op) {
      auto *axis = EsCreateScalarInt64(ctx.builder, op->dim);
      return EsGatherV2(ctx.inputs[0], ctx.inputs[1], axis, /*batch_dims=*/0,
        /*is_preprocessed=*/false, /*negative_index_support=*/false);
    });
}
} // namespace
