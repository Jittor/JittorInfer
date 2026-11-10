#pragma once

// Header-only dispatch from Forch operation types to GE graph emitters.
#include "forch/exception.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/multi_output.hpp"

#include "esb_funcs.h" // EsCTensorHolder / EsCGraphBuilder forward decls

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

namespace forch::detail {

/**
 * 建图期交给算子 emitter 的上下文：当前 builder（构造标量/向量常量用）、
 * 输入的 EsCTensorHolder，以及它们背后的 forch Tensor（同序）。
 *
 * input_tensors 不是冗余的：EsCTensorHolder 只能写 dtype（EsSetDataType）不能
 * 读，所以需要显式填 y_dtype 之类属性的算子没法从 holder 问出类型，只能用自己
 * 的 shape()/dtype() 规则在 input_tensors 上重算一遍。单输出算子和多输出节点
 * 都能这样推出任意一路输出的 metadata，所以两者共用这一个上下文。
 */
struct GEOpContext {
  EsCGraphBuilder                      *builder;
  const std::vector<EsCTensorHolder *> &inputs;
  const forch::TensorList              &input_tensors;
};

/**
 * 算子注册表：把 forch::op 的具体类型映射到建图回调。
 *
 * 单输出算子（op::Function 子类）和扇出节点（op::MultiOutputBase 子类）走两张
 * 表 —— 两者基类不同、输出路数不同、在 Visitor 里也是两个入口。除此之外一切
 * 统一：同一个 GEOpContext、同一种回调形状 `(const GEOpContext &, const Op
 * *)`、 同一套去重与查表逻辑、同一个 emit 名字（按算子类型重载）。
 */
class GEOpRegistry {
public:
  using Emitter = std::function<EsCTensorHolder *(
    const GEOpContext &, const forch::op::Function *)>;

  using MultiEmitter = std::function<std::vector<EsCTensorHolder *>(
    const GEOpContext &, const forch::op::MultiOutputBase *)>;

  /** 注册单输出算子：回调返回该算子唯一的 holder。 */
  template <class OpType, class F>
  static void register_op(F &&emit) {
    static_assert(std::is_base_of_v<forch::op::Function, OpType>,
      "OpType must be derived from forch::op::Function");

    insert_once<OpType>(instance().emitters,
      [emit = std::forward<F>(emit)](
        const GEOpContext &ctx, const forch::op::Function *op) {
        return emit(ctx, static_cast<const OpType *>(op));
      });
  }

  /**
   * 注册扇出节点：回调返回该节点全部输出的 holders。路数由这里校验 —— 承诺和
   * 检查放在一起，建图侧不必再记得查。
   */
  template <class NodeType, class F>
  static void register_multi_op(F &&emit) {
    static_assert(std::is_base_of_v<forch::op::MultiOutputBase, NodeType>,
      "NodeType must be derived from forch::op::MultiOutputBase");

    insert_once<NodeType>(instance().multi_emitters,
      [emit = std::forward<F>(emit)](
        const GEOpContext &ctx, const forch::op::MultiOutputBase *node) {
        const auto *typed   = static_cast<const NodeType *>(node);
        auto        holders = emit(ctx, typed);
        if (holders.size() != typed->n_outputs()) {
          throw forch::exception(std::string("forch-ge: ")
            + typeid(NodeType).name() + " emitted "
            + std::to_string(holders.size()) + " outputs but declares "
            + std::to_string(typed->n_outputs()));
        }
        return holders;
      });
  }

  static EsCTensorHolder *emit(
    const GEOpContext &ctx, const forch::op::Function *op) {
    return lookup(instance().emitters, typeid(*op))(ctx, op);
  }

  static std::vector<EsCTensorHolder *> emit(
    const GEOpContext &ctx, const forch::op::MultiOutputBase *node) {
    return lookup(instance().multi_emitters, typeid(*node))(ctx, node);
  }

private:
  static GEOpRegistry &instance() {
    static GEOpRegistry instance;
    return instance;
  }

  /// 每个具体算子类型只能注册一次，重复注册是构建期的配置错误。
  template <class OpType, class Map, class Entry>
  static void insert_once(Map &map, Entry entry) {
    const auto key   = std::type_index(typeid(OpType));
    const bool fresh = map.emplace(key, std::move(entry)).second;
    if (!fresh) {
      throw forch::exception(std::string("forch-ge: ") + typeid(OpType).name()
        + " is already registered");
    }
  }

  template <class Map>
  static const typename Map::mapped_type &lookup(
    const Map &map, const std::type_info &type) {
    auto it = map.find(std::type_index(type));
    if (it == map.end()) {
      throw exception(
        std::string("forch-ge: unsupported op type: ") + type.name());
    }
    return it->second;
  }

  std::map<std::type_index, Emitter>      emitters;
  std::map<std::type_index, MultiEmitter> multi_emitters;
};

} // namespace forch::detail
