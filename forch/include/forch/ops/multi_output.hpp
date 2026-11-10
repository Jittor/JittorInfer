#pragma once

#include "forch/ops.hpp"

#include <cstddef>
#include <memory>
#include <ostream>
#include <utility>

namespace forch::op {

/**
 * 扇出节点(多输出算子):描述"一次底层计算算出 N 个输出"的参数与输出规则。
 *
 * 它本身不 tensor 化:不能被 Tensor 包裹、不进 Visitor、不进后端 pool。
 * 通过 op::MultiOutputPort 端口张量暴露给图;后端以 typeid(*node) 注册、以 node
 * 指针做物化去重,保证底层调用(如 EsTopK)每图只执行一次。
 */
struct MultiOutputBase {
  virtual ~MultiOutputBase() = default;

  /// 输出路数。
  virtual size_t n_outputs() const = 0;

  /// 第 i 路输出的形状 / 类型。input 为该节点全部输入
  /// Tensor(与端口的输入一致)。
  virtual Shape shape(const TensorList &input, size_t i) const = 0;
  virtual DType dtype(const TensorList &input, size_t i) const = 0;

  virtual void dump(std::ostream &os) const = 0;
};

/**
 * 端口张量:选中多输出节点 node 的第 index 个输出。
 *
 * 对 Tensor 的消费者而言它就是一个普通 op —— 有自己的 shape/dtype,只是
 * 这两个值委托给 node 计算;真正特殊的行为只发生在后端图构建期(如
 * GE GEGraphBuilder::visit(MultiOutputPort):查缓存、按需一次物化、取第 index
 * 路)。
 *
 * 继承 Base 而非 Function:Function 在 forch 里是"注册进后端注册表、物化成
 * 计算节点"的算子类别(TopKValues/MatMul/Softmax…),MultiOutputPort 是零计算的
 * 选择器,不该出现在注册表里。Base::accept 是纯虚,派生必须自己实现,才能
 * 把 this 以 const MultiOutputPort* 的静态类型派发到 Visitor 专属槽位。
 */
struct MultiOutputPort : public Base {
  std::shared_ptr<const MultiOutputBase> node;
  size_t                                 index;

  MultiOutputPort(std::shared_ptr<const MultiOutputBase> node, size_t index)
      : node(std::move(node)), index(index) {
    if (index >= this->node->n_outputs()) {
      throw exception("MultiOutputPort: output index out of range");
    }
  }

  void accept(Visitor *visitor, const Tensor &self) const override {
    visitor->visit(this, self);
  }

  Shape shape(const TensorList &input) const override {
    return node->shape(input, index);
  }

  DType dtype(const TensorList &input) const override {
    return node->dtype(input, index);
  }

  void dump(std::ostream &os) const override {
    os << "MultiOutputPort(index=" << index << ", node=";
    node->dump(os);
    os << ")";
  }
};

/// 便捷构造:把多输出节点第 index 路输出包成 Tensor(节点与输入由此张量持有)。
inline Tensor make_port(std::shared_ptr<const MultiOutputBase> node,
  size_t index, const TensorList &inputs) {
  return Tensor(
    inputs, std::make_unique<MultiOutputPort>(std::move(node), index));
}

} // namespace forch::op
