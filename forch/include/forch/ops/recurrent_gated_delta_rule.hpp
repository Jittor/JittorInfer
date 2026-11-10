#pragma once

#include "forch/exception.hpp"
#include "forch/ops/multi_output.hpp"

#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

namespace forch::op {

/**
 * GDN(gated delta rule)decode 单步算子节点,一次内核同时产出 out 与
 * ref_state 两路输出(端口 0/1)。
 *
 * 输入(前 7 个必选;g 与 num_accepted_tokens 可选,是否出现由
 * with_g / with_num_accepted_tokens 声明):
 *   [0] query               (T, Nk, Dk)            BF16
 *   [1] key                 (T, Nk, Dk)            BF16
 *   [2] value               (T, Nv, Dv)            BF16   Nv >= Nk 且 Nv % Nk
 * == 0 [3] beta                (T, Nv)                BF16 [4] state (BlockNum,
 * Nv, Dv, Dk) BF16 [5] actual_seq_lengths  (B,)                   INT32 [6]
 * ssm_state_indices   (T,)                   INT32  值域 [0, BlockNum) [7] g
 * (T, Nv)                FP32   可选,语义 α = e^g [8] num_accepted_tokens (B,)
 * INT32  可选,缺省每序列 1
 *
 * gk 输入本版 CANN 不支持,不暴露到 facade。
 * scale_value 是 attr,调用方传 1/sqrt(Dk)。
 *
 * 数据约束(依赖运行时值,图构建期不可见,由调用方保证):
 *   T == Σ Lᵢ 且每序列 1 <= Lᵢ <= 8(芯片约束);
 *   ssm_state_indices 逐 token 指向 state block,允许乱序与重复命中。
 */
struct RecurrentGatedDeltaRuleNode : public MultiOutputBase {
  float scale_value;
  bool  with_g                   = false;
  bool  with_num_accepted_tokens = false;

  explicit RecurrentGatedDeltaRuleNode(float scale_value_, bool with_g_ = false,
    bool with_num_accepted_tokens_ = false)
      : scale_value(scale_value_), with_g(with_g_),
        with_num_accepted_tokens(with_num_accepted_tokens_) {}

  size_t n_outputs() const override {
    return 2;
  }

  Shape shape(const TensorList &input, size_t index) const override {
    const auto d = validate(input);
    if (index == 0) {
      return {d.t, d.nv, d.dv};
    }
    return {d.block_num, d.nv, d.dv, d.dk};
  }

  // out 与 ref_state 均为 BF16(GE kernel 的固定输出类型)
  DType dtype(const TensorList & /*input*/, size_t /*index*/) const override {
    return DType::BF16;
  }

  void dump(std::ostream &os) const override {
    os << "RecurrentGatedDeltaRuleNode(scale_value=" << scale_value
       << ", with_g=" << with_g
       << ", with_num_accepted_tokens=" << with_num_accepted_tokens << ')';
  }

private:
  struct Dims {
    int64_t t = 0, b = 0, nk = 0, dk = 0, nv = 0, dv = 0, block_num = 0;
  };

  Dims validate(const TensorList &input) const {
    const size_t want =
      7 + (with_g ? 1 : 0) + (with_num_accepted_tokens ? 1 : 0);
    if (input.size() != want) {
      throw exception("recurrent_gated_delta_rule: expected "
        + std::to_string(want) + " inputs, got "
        + std::to_string(input.size()));
    }

    const auto &query  = input[0];
    const auto &key    = input[1];
    const auto &value  = input[2];
    const auto &beta   = input[3];
    const auto &state  = input[4];
    const auto &seqlen = input[5];
    const auto &index  = input[6];

    if (query.dim() != 3 || key.dim() != 3 || value.dim() != 3
      || beta.dim() != 2 || state.dim() != 4 || seqlen.dim() != 1
      || index.dim() != 1) {
      throw exception("recurrent_gated_delta_rule: unexpected input rank");
    }
    for (const auto t : {&query, &key, &value, &beta, &state}) {
      if (t->dtype() != DType::BF16) {
        throw exception(
          "recurrent_gated_delta_rule: q/k/v/beta/state must be bf16");
      }
    }

    Dims d;
    d.t  = query.shape()[0];
    d.nk = query.shape()[1];
    d.dk = query.shape()[2];

    if (key.shape() != query.shape()) {
      throw exception("recurrent_gated_delta_rule: key shape must equal query");
    }
    if (value.shape()[0] != d.t) {
      throw exception("recurrent_gated_delta_rule: value tokens must equal T");
    }
    d.nv = value.shape()[1];
    d.dv = value.shape()[2];
    if (d.nv < d.nk || d.nv % d.nk != 0) {
      throw exception(
        "recurrent_gated_delta_rule: Nv must be a multiple of Nk (>= Nk)");
    }

    if (beta.shape() != Shape({d.t, d.nv})) {
      throw exception("recurrent_gated_delta_rule: beta must be (T, Nv)");
    }

    d.block_num = state.shape()[0];
    if (state.shape()[1] != d.nv || state.shape()[2] != d.dv
      || state.shape()[3] != d.dk) {
      throw exception(
        "recurrent_gated_delta_rule: state must be (BlockNum, Nv, Dv, Dk)");
    }

    if (seqlen.dtype() != DType::INT32 || index.dtype() != DType::INT32) {
      throw exception(
        "recurrent_gated_delta_rule: seq lengths/indices must be int32");
    }
    d.b = seqlen.shape()[0];
    if (d.b > d.block_num) {
      throw exception("recurrent_gated_delta_rule: BlockNum must be >= B");
    }
    if (index.shape()[0] != d.t) {
      throw exception(
        "recurrent_gated_delta_rule: ssm_state_indices must be (T,)");
    }

    size_t cursor = 7;
    if (with_g) {
      const auto &g = input[cursor++];
      if (g.dtype() != DType::FP32 || g.shape() != Shape({d.t, d.nv})) {
        throw exception("recurrent_gated_delta_rule: g must be (T, Nv) fp32");
      }
    }
    if (with_num_accepted_tokens) {
      const auto &accepted = input[cursor++];
      if (accepted.dtype() != DType::INT32
        || accepted.shape() != Shape({d.b})) {
        throw exception(
          "recurrent_gated_delta_rule: num_accepted_tokens must be (B,) int32");
      }
    }

    return d;
  }
};

/// 双输出,字段顺序即端口顺序:out(0) / ref_state(1)。
struct RecurrentGatedDeltaRuleOutputs {
  Tensor out;
  Tensor ref_state;
};

struct RecurrentGatedDeltaRuleBuilder {
  std::shared_ptr<RecurrentGatedDeltaRuleNode> node;

  template <class... Args>
  RecurrentGatedDeltaRuleOutputs operator()(Args... args) const {
    static_assert((std::is_constructible_v<Tensor, Args> && ...),
      "All args must be convertible to Tensor");

    const size_t argc = sizeof...(Args);
    const size_t want =
      7 + (node->with_g ? 1 : 0) + (node->with_num_accepted_tokens ? 1 : 0);
    if (argc != want) {
      throw exception("recurrent_gated_delta_rule: expected "
        + std::to_string(want) + " inputs, got " + std::to_string(argc));
    }

    TensorList inputs{args...};
    return {make_port(node, 0, inputs), make_port(node, 1, inputs)};
  }
};

/// 一对端口的主构造器:一次调用创建共享 node,out/ref_state 两端口同输入。
/// 与 topk 相同,builder 是单次的,每次逻辑调用使用新的
/// recurrent_gated_delta_rule(...) 表达式。
inline RecurrentGatedDeltaRuleBuilder recurrent_gated_delta_rule(
  float scale_value, bool with_g = false,
  bool with_num_accepted_tokens = false) {
  return {std::make_shared<RecurrentGatedDeltaRuleNode>(
    scale_value, with_g, with_num_accepted_tokens)};
}

} // namespace forch::op
