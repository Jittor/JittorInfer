#include "forch/ops/recurrent_gated_delta_rule.hpp"
#include "op-registry.hpp"

#include "es_RecurrentGatedDeltaRule_c.h"

#include <cstddef>
#include <vector>

namespace {

__attribute__((constructor)) void register_op() {
  // 双输出(out, ref_state):注册为多输出节点,图构建期按 node 指针去重,
  // 每个逻辑算子只发一次 EsRecurrentGatedDeltaRule。
  forch::detail::GEOpRegistry::register_multi_op<
    forch::op::RecurrentGatedDeltaRuleNode>(
    [](const forch::detail::GEOpContext            &ctx,
      const forch::op::RecurrentGatedDeltaRuleNode *op) {
      size_t           cursor = 7;
      EsCTensorHolder *g      = op->with_g ? ctx.inputs[cursor++] : nullptr;
      // gk 本版 CANN 不支持,恒传空 holder(C 接口对 optional 输入收 nullptr)
      EsCTensorHolder *accepted =
        op->with_num_accepted_tokens ? ctx.inputs[cursor++] : nullptr;

      auto out = EsRecurrentGatedDeltaRule(ctx.inputs[0], ctx.inputs[1],
        ctx.inputs[2], ctx.inputs[3], ctx.inputs[4], ctx.inputs[5],
        ctx.inputs[6], g, /*gk=*/nullptr, accepted, op->scale_value);
      // 返回顺序必须与公开端口一致:out 在前,ref_state 在后
      return std::vector<EsCTensorHolder *>{out.out, out.ref_state};
    });
}

} // namespace
