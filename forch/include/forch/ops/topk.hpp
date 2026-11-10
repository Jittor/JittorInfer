#pragma once

#include "forch/ops/multi_output.hpp"

#include <memory>
#include <ostream>

namespace forch::op {

/// 单路 topk 的形状规则(输出形状与输入相同,只是 dim 维变为 k)。
inline Shape topk_output_shape(
  const Shape &x_shape, int64_t dim, int64_t k, const char *what) {
  auto rank = static_cast<int64_t>(x_shape.size());
  auto ndim = dim < 0 ? dim + rank : dim;

  if (ndim < 0 || ndim >= rank) {
    throw exception(std::string(what) + ": dim out of range");
  }

  auto output_shape  = x_shape;
  output_shape[ndim] = k;
  return output_shape;
}

/**
 * TopK 扇出节点:一次内核同时算出 values(0) 与 indices(1)。
 * 共享同一个 node 的两个端口张量保证在图构建期只物化一个底层 TopK,
 * 从而 values/indices 必然来自同一次计算、逐元素对应。
 */
struct TopKNode : public MultiOutputBase {
  int64_t k;
  int64_t dim;
  bool    sorted;

  TopKNode(int64_t k, int64_t dim, bool sorted)
      : k(k), dim(dim), sorted(sorted) {}

  size_t n_outputs() const override {
    return 2;
  }

  Shape shape(const TensorList &input, size_t /*index*/) const override {
    return topk_output_shape(input[0].shape(), dim, k, "topk");
  }

  DType dtype(const TensorList &input, size_t index) const override {
    // index==0(values) 跟随输入 dtype;index==1(indices) 保持 INT64。
    // 运行时 Tensor 的 dtype 以 GE 实际输出为准(GERuntime::dtype 读真值),
    // 与旧 TopKValues/TopKIndices 拆分版的行为一致。
    return index == 0 ? input[0].dtype() : DType::INT64;
  }

  void dump(std::ostream &os) const override {
    os << "TopKNode(k=" << k << ", dim=" << dim << ", sorted=" << sorted << ')';
  }
};

/// topk 的一对输出,字段顺序 values/indices。
struct TopKOutputs {
  Tensor values;
  Tensor indices;
};

/// 一对端口的主构造器:一次 operator() 调用创建一个共享 node,两个端口
/// 同输入。builder 只存参数——重复调用会各自创建 node(后端按 node 指针
/// 去重,跨调用共享会把第二次的输入静默替换成第一次的)。
struct TopKBuilder {
  int64_t k;
  int64_t dim;
  bool    sorted;

  TopKOutputs operator()(const Tensor &x) const {
    auto       node = std::make_shared<TopKNode>(k, dim, sorted);
    TensorList inputs{x};
    return {make_port(node, 0, inputs), make_port(node, 1, inputs)};
  }
};

/// 多输出构造器:values/indices 两端口同输入。单路输出场景直接取返回
/// 结构的 .values / .indices 字段即可。
inline TopKBuilder topk(int64_t k, int64_t dim = -1, bool sorted = false) {
  return {k, dim, sorted};
}

} // namespace forch::op
