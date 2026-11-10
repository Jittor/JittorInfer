#include "forch/ops/ge/moe_init_routing.hpp"
#include "op-registry.hpp"

#include "es_MoeInitRoutingV3_c.h"

namespace {

// MoeInitRouting (CANN MoeInitRoutingV3) GE mapping: x + expert_idx ->
// (expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum)。注册为
// 多输出节点,一次 EsMoeInitRoutingV3 调用同时拿到三路输出。
//
// 公共契约只暴露 dropless + cumsum,其余 CANN 属性在验证前刻意钉死:
//   drop_pad_mode=0(dropless)、expert_capacity=0(不限容量)、
//   expert_tokens_num_type=0 / expert_tokens_num_flag=true(cumsum)、
//   quant_mode=-1(quant 关闭,expanded_scale 不使用)、row_idx_type=0
//   (gather 索引;1 = scatter 索引,实测 910B 上 0..3 输出完全一致,
//   该属性未生效)。
//
// HISTORY:仅暴露 V3——V1(MoeInitRouting)按 x[sortedRowIdx % T] 选
// token,是 slot-major 语义,在 es_ 静态图输入约束下无法表达 t-major
// 分发,且对拍无性能优势;MoeComputeExpertTokens 冗余(V3 内置 cumsum);
// 两者均已移除。
__attribute__((constructor)) void register_ops() {
  forch::detail::GEOpRegistry::register_multi_op<
    forch::op::ge::MoeInitRoutingNode>(
    [](const forch::detail::GEOpContext       &ctx,
      const forch::op::ge::MoeInitRoutingNode *op) {
      auto out = EsMoeInitRoutingV3(
        /*x*/ ctx.inputs[0],
        /*expert_idx*/ ctx.inputs[1],
        /*scale*/ nullptr,
        /*offset*/ nullptr,
        /*active_num*/ op->active_num,
        /*expert_capacity*/ 0,
        /*expert_num*/ op->expert_num,
        /*drop_pad_mode*/ 0,
        /*expert_tokens_num_type*/ 0,
        /*expert_tokens_num_flag*/ true,
        /*quant_mode*/ -1,
        /*active_expert_range*/ nullptr,
        /*active_expert_range_num*/ 0,
        /*row_idx_type*/ 0);
      return std::vector<EsCTensorHolder *>{out.expanded_x,
        out.expanded_row_idx, out.expert_tokens_count_or_cumsum};
    });
}

} // namespace
