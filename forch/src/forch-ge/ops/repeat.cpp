#include "forch/ops/repeat.hpp"
#include "forch/exception.hpp"
#include "op-registry.hpp"

#include "es_Tile_c.h"

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Repeat>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Repeat *op) {
      if (op->interleave) {
        // TODO[low priority]: 可以用 unsqueeze + tile + reshape 实现
        // interleave=true 的功能，或者看看 aclnnReshapeInterleaveWithDim
        // 怎么（什么时候）适配 GE
        throw forch::exception(
          "repeat with interleave=true is not supported in GE backend");
      }
      auto *multiples = EsCreateVectorInt64(ctx.builder, op->repeat.data(),
        static_cast<int64_t>(op->repeat.size()));
      return EsTile(ctx.inputs[0], multiples);
    });
}
} // namespace
