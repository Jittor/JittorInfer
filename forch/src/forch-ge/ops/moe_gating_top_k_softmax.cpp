#include "forch/ops/ge/moe_gating_top_k_softmax.hpp"
#include "op-registry.hpp"

#include "es_MoeGatingTopKSoftmaxV2_c.h"

namespace {

// MoeGatingTopKSoftmaxV2: x → (softmax → topk → renorm) → (y, expert_idx)。
// 注册为多输出节点:y/expert_idx 两个端口共享一次内核物化;
// softmaxResult 输出不启用(finished 亦不启用, 部署无 token 裁剪)。
__attribute__((constructor)) void register_ops() {
  forch::detail::GEOpRegistry::register_multi_op<
    forch::op::ge::MoeGatingTopKSoftmaxNode>(
    [](const forch::detail::GEOpContext             &ctx,
      const forch::op::ge::MoeGatingTopKSoftmaxNode *op) {
      auto out = EsMoeGatingTopKSoftmaxV2(ctx.inputs[0], /*finished=*/nullptr,
        op->k, op->renorm ? 1 : 0, /*output_softmax_result=*/false);
      return std::vector<EsCTensorHolder *>{out.y, out.expert_idx};
    });
}

} // namespace
