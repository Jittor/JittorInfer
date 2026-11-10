// MoE dispatch routing: sort the (token, slot) routing entries by expert
// and expand the token features in per-expert order.
//
//   expandedXOut[expandedRowIdxOut[i]] = x[i // k]   (i = t*k + s flat)
// i.e. a plain [T, k] expert_idx input selects the right token rows.
// expanded_row_idx is the TOKEN-MAJOR-keyed rank table (position p = t*k+s
// holds the sorted-position of entry (t,s)) — the same contract
// `moe_finalize_routing` consumes, so the two connect directly.
//
// Inputs:
//   x          [T, D]
//   expert_idx [T, k] INT32 (or 1D [T*k])
// Attrs:
//   expert_num = E
//
// Outputs:
//   expanded_x          [T*k, D]
//   expanded_row_idx    [T*k] INT32
//   expert_tokens_count_or_cumsum [E] INT64 (cumsum feeds GroupedMatmul
//   group_list_type=0 directly)
//
// Named `moe_init_routing` without a version suffix; the underlying CANN
// op and the pinned attrs are documented in the GE mapping — see
// src/forch-ge/ops/moe_init_routing.cpp.
#pragma once

#include "forch/ops/multi_output.hpp"

#include <memory>
#include <ostream>

namespace forch::op::ge {

/// CANN MoeInitRoutingV3 多输出节点:一次内核调用同时产出三路输出。共享一个
/// node,图构建期只物化一次底层算子。
struct MoeInitRoutingNode : public MultiOutputBase {
  int64_t expert_num; // E

  // 展开后位置数(T*k),构造期由 shape() 从 expert_idx 的形状推导
  // (GroupedMatMul 的 mutable-resolved-attrs 模式),映射回调直接读取。
  mutable int64_t active_num = -1;

  explicit MoeInitRoutingNode(int64_t expert_num) : expert_num(expert_num) {}

  size_t n_outputs() const override {
    return 3;
  }

  Shape shape(const TensorList &input, size_t index) const override {
    // active_num = T*k = numel(expert_idx),从输入推导(即校验):
    // 同时要求它是 token 数(x 的第一维)的整数倍,否则 k 不是整数。
    const Shape &es = input[1].shape();
    if (es.empty() || input[0].shape().empty()) {
      throw exception("moe_init_routing: x and expert_idx must not be 0-rank");
    }
    int64_t num = 1;
    for (auto d : es)
      num *= d;
    const int64_t tokens = input[0].shape()[0];
    if (tokens <= 0 || num % tokens != 0) {
      throw exception(
        "moe_init_routing: expert_idx numel must be a multiple of the "
        "token count (x.shape()[0])");
    }
    active_num = num;
    if (index == 0) {
      // [active_num, D] — active_num 是展开位置数(T*k)
      return {active_num, input[0].shape().back()};
    }
    if (index == 1) {
      return {active_num}; // expanded_row_idx [T*k]
    }
    return {expert_num}; // expert_tokens_count_or_cumsum [E]
  }

  DType dtype(const TensorList &input, size_t index) const override {
    if (index == 0) {
      return input[0].dtype();
    }
    return index == 1 ? DType::INT32 : DType::INT64;
  }

  void dump(std::ostream &os) const override {
    os << "MoeInitRoutingNode(active_num=" << active_num
       << ", expert_num=" << expert_num << ')';
  }
};

/// 三路输出,字段顺序 = n_outputs 的 index:
/// 0 = expanded_x,1 = expanded_row_idx,2 = expert_tokens_count_or_cumsum。
/// expanded_row_idx 是 token-major keyed 的 rank 表(位置 p=t*k+s 存放条目
/// (t,s) 的排序后位置)——与 moe_finalize_routing 的公共契约一致,可直连
/// (finalize 侧以 drop_pad_mode=2 按行消费,无需任何重排)。
struct MoeInitRoutingOutputs {
  Tensor expanded_x;
  Tensor expanded_row_idx;
  Tensor expert_tokens_count_or_cumsum;
};

/// 三端口主构造器:一次 operator() 调用创建一个共享 node,端口同输入
/// (x, expert_idx)。builder 只存参数——重复调用会各自创建 node(后端
/// 按 node 指针去重,跨调用共享会把第二次的输入静默替换成第一次的)。
struct MoeInitRoutingBuilder {
  int64_t expert_num;

  MoeInitRoutingOutputs operator()(
    const Tensor &x, const Tensor &expert_idx) const {
    auto       node = std::make_shared<MoeInitRoutingNode>(expert_num);
    TensorList inputs{x, expert_idx};
    return {make_port(node, 0, inputs), make_port(node, 1, inputs),
      make_port(node, 2, inputs)};
  }
};

/// 需要"一次内核、三路输出"时使用本接口。active_num(= T*k)从 expert_idx
/// 的形状推导,无需(也不允许)调用者传入。
inline MoeInitRoutingBuilder moe_init_routing(int64_t expert_num) {
  return {expert_num};
}

} // namespace forch::op::ge
