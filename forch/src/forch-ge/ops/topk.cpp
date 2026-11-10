#include "forch/ops/topk.hpp"
#include "op-registry.hpp"

#include "es_TopK_c.h"

namespace {

__attribute__((constructor)) void register_topk_ops() {
  // TopK 一次内核同时算出 values + indices:注册为多输出节点,图构建期
  // 由 GEGraphBuilder 按 node 指针去重,保证每个逻辑 topk 只有一个 EsTopK。
  forch::detail::GEOpRegistry::register_multi_op<forch::op::TopKNode>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::TopKNode *op) {
      auto *k_tensor = EsCreateScalarInt64(ctx.builder, op->k);
      auto  out =
        EsTopK(ctx.inputs[0], k_tensor, op->sorted, /*largest=*/true, op->dim);
      return std::vector<EsCTensorHolder *>{out.values, out.indices};
    });
}

} // namespace
