#include "forch/ops/cat.hpp"
#include "op-registry.hpp"

#include "es_ConcatV2_c.h"

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Cat>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Cat *op) {
      auto *concat_dim = EsCreateScalarInt64(ctx.builder, op->dim);
      // EsConcatV2 takes EsCTensorHolder** as the input array; cast away the
      // const since the C interface is non-const but does not mutate.
      return EsConcatV2(const_cast<EsCTensorHolder **>(ctx.inputs.data()),
        static_cast<int64_t>(ctx.inputs.size()), concat_dim,
        static_cast<int64_t>(ctx.inputs.size()));
    });
}
} // namespace
