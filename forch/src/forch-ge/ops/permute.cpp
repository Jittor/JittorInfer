#include "forch/ops/permute.hpp"
#include "op-registry.hpp"

#include "es_Transpose_c.h"

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Permute>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Permute *op) {
      auto *perm = EsCreateVectorInt64(
        ctx.builder, op->perm.data(), static_cast<int64_t>(op->perm.size()));
      return EsTranspose(ctx.inputs[0], perm);
    });
}
} // namespace
