#include "forch/ops/cumsum.hpp"
#include "op-registry.hpp"

#include "es_Cumsum_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Cumsum>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Cumsum *op) {
      // 本算子唯一的实现坑:GE Cumsum 的 axis 是必选 tensor 输入而非标量
      // attr(op proto 要求 shape 为 1 的 int32/int64 张量),须把 attr 包成
      // 1 元素 int64 向量常量。
      auto *axis = EsCreateVectorInt64(ctx.builder, &op->axis, /*dim=*/1);
      return EsCumsum(ctx.inputs[0], axis, op->exclusive, op->reverse);
    });
}

} // namespace
