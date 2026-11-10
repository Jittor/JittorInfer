// [minimal-impl] CANN MoeGatingTopKSoftmaxV2 (es_ API) — MoE router 融合算子:
// softmax(x) → topk → (renorm ? y/Σy : 原值) 一次内核完成,替换 gate 的
// SoftmaxV2 + TopK(AI_CPU) + reduce/div 三段(实测 ~0.8ms/step → ~0.1ms)。
//
// 输入:
//   x [T, E] — gating logits(gate matmul 的 FP32 输出)
// 属性:
//   k      = num_experts_per_tok
//   renorm = norm_topk_prob(y 除以 top-k 和)
// 输出:
//   y          [T, k] — 与 x 同 dtype(fp32), 降序
//   expert_idx [T, k] — INT32, 与 y 逐位对应(可直接喂 MoeInitRouting)
#pragma once

#include "forch/exception.hpp"
#include "forch/ops/multi_output.hpp"

#include <memory>
#include <ostream>

namespace forch::op::ge {

struct MoeGatingTopKSoftmaxNode : public MultiOutputBase {
  int64_t k;
  bool    renorm;

  MoeGatingTopKSoftmaxNode(int64_t k, bool renorm) : k(k), renorm(renorm) {}

  size_t n_outputs() const override {
    return 2;
  }

  Shape shape(const TensorList &input, size_t /*index*/) const override {
    // x [T, E]: last dim is the expert-score dim; k must fit inside it.
    const Shape &s = input[0].shape();
    if (s.size() != 2) {
      throw exception("moe_gating_top_k_softmax: x must be 2D [T, E]");
    }
    if (k <= 0 || k > s[1]) {
      throw exception("moe_gating_top_k_softmax: k must be in [1, E]");
    }
    return {s[0], k};
  }

  DType dtype(const TensorList &input, size_t index) const override {
    // y 跟随 x(fp32); expert_idx INT32 — 可直接喂 MoeInitRouting
    return index == 0 ? input[0].dtype() : DType::INT32;
  }

  void dump(std::ostream &os) const override {
    os << "MoeGatingTopKSoftmaxNode(k=" << k << ", renorm=" << renorm << ')';
  }
};

/// 两路输出,字段顺序 = n_outputs 的 index: 0 = y, 1 = expert_idx。
struct MoeGatingTopKSoftmaxOutputs {
  Tensor y;
  Tensor expert_idx;
};

/// 两端口主构造器:一次 operator() 调用创建一个共享 node,两端口同输入。
/// builder 只存参数——重复调用会各自创建 node(后端按 node 指针去重,
/// 跨调用共享会把第二次的输入静默替换成第一次的)。
struct MoeGatingTopKSoftmaxBuilder {
  int64_t k;
  bool    renorm;

  MoeGatingTopKSoftmaxOutputs operator()(const Tensor &x) const {
    auto       node = std::make_shared<MoeGatingTopKSoftmaxNode>(k, renorm);
    TensorList inputs{x};
    return {make_port(node, 0, inputs), make_port(node, 1, inputs)};
  }
};

inline MoeGatingTopKSoftmaxBuilder moe_gating_top_k_softmax(
  int64_t k, bool renorm) {
  return {k, renorm};
}

} // namespace forch::op::ge
