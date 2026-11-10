#pragma once

#include "forch/exception.hpp"
#include "forch/ops/function.hpp"

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace forch::op {

struct FusedInferAttention : public Function {
  enum class InputLayout {
    BSH,
    BSND,
    BNSD,
    BNSD_BSND,
    TND, // Packed queries with a paged BnBsH KV pool.
  };

  enum class OptionalInput {
    PseShift,
    AttenMask,
    ActualSeqLengths,
    ActualSeqLengthsKv,
    DequantScale1,
    QuantScale1,
    DequantScale2,
    QuantScale2,
    QuantOffset2,
    AntiquantScale,
    AntiquantOffset,
    BlockTable,
    QueryPaddingSize,
    KvPaddingSize,
    KeyAntiquantScale,
    KeyAntiquantOffset,
    ValueAntiquantScale,
    ValueAntiquantOffset,
    KeySharedPrefix,
    ValueSharedPrefix,
    ActualSharedPrefixLen,
    QueryRope,
    KeyRope,
    KeyRopeAntiquantScale,
    DequantScaleQuery,
    LearnableSink,
    QStartIdx,
    KvStartIdx,
    Count,
  };

  static constexpr size_t OptionalInputCount =
    static_cast<size_t>(OptionalInput::Count);
  static constexpr int64_t MaxTokens = INT_MAX;

  using OptionalInputMask = std::array<bool, OptionalInputCount>;

  struct Options {
    explicit Options(int64_t num_heads) : num_heads(num_heads) {}

    int64_t              num_heads;
    float                scale                = 1.0F;
    int64_t              pre_tokens           = MaxTokens;
    int64_t              next_tokens          = MaxTokens;
    InputLayout          input_layout         = InputLayout::BSH;
    int64_t              num_key_value_heads  = 0;
    int64_t              sparse_mode          = 0;
    int64_t              inner_precise        = 1;
    int64_t              block_size           = 0;
    int64_t              antiquant_mode       = 0;
    bool                 softmax_lse_flag     = false;
    int64_t              key_antiquant_mode   = 0;
    int64_t              value_antiquant_mode = 0;
    int64_t              query_quant_mode     = 0;
    int64_t              pse_type             = 0;
    std::optional<DType> output_dtype;
    // Dense BSND MLA decode with uniform KV scan lengths and a slot mask.
    // The GE backend selects the scheduling workaround for this path.
    bool decode_mla = false;
  };

  struct OptionalInputs {
    std::optional<Tensor> pse_shift;
    std::optional<Tensor> atten_mask;
    std::optional<Tensor> actual_seq_lengths;
    std::optional<Tensor> actual_seq_lengths_kv;
    std::optional<Tensor> dequant_scale1;
    std::optional<Tensor> quant_scale1;
    std::optional<Tensor> dequant_scale2;
    std::optional<Tensor> quant_scale2;
    std::optional<Tensor> quant_offset2;
    std::optional<Tensor> antiquant_scale;
    std::optional<Tensor> antiquant_offset;
    std::optional<Tensor> block_table;
    std::optional<Tensor> query_padding_size;
    std::optional<Tensor> kv_padding_size;
    std::optional<Tensor> key_antiquant_scale;
    std::optional<Tensor> key_antiquant_offset;
    std::optional<Tensor> value_antiquant_scale;
    std::optional<Tensor> value_antiquant_offset;
    std::optional<Tensor> key_shared_prefix;
    std::optional<Tensor> value_shared_prefix;
    std::optional<Tensor> actual_shared_prefix_len;
    std::optional<Tensor> query_rope;
    std::optional<Tensor> key_rope;
    std::optional<Tensor> key_rope_antiquant_scale;
    std::optional<Tensor> dequant_scale_query;
    std::optional<Tensor> learnable_sink;
    std::optional<Tensor> q_start_idx;
    std::optional<Tensor> kv_start_idx;

    void append_to(TensorList &inputs, OptionalInputMask &mask) const {
      append(inputs, mask, OptionalInput::PseShift, pse_shift);
      append(inputs, mask, OptionalInput::AttenMask, atten_mask);
      append(inputs, mask, OptionalInput::ActualSeqLengths, actual_seq_lengths);
      append(
        inputs, mask, OptionalInput::ActualSeqLengthsKv, actual_seq_lengths_kv);
      append(inputs, mask, OptionalInput::DequantScale1, dequant_scale1);
      append(inputs, mask, OptionalInput::QuantScale1, quant_scale1);
      append(inputs, mask, OptionalInput::DequantScale2, dequant_scale2);
      append(inputs, mask, OptionalInput::QuantScale2, quant_scale2);
      append(inputs, mask, OptionalInput::QuantOffset2, quant_offset2);
      append(inputs, mask, OptionalInput::AntiquantScale, antiquant_scale);
      append(inputs, mask, OptionalInput::AntiquantOffset, antiquant_offset);
      append(inputs, mask, OptionalInput::BlockTable, block_table);
      append(inputs, mask, OptionalInput::QueryPaddingSize, query_padding_size);
      append(inputs, mask, OptionalInput::KvPaddingSize, kv_padding_size);
      append(
        inputs, mask, OptionalInput::KeyAntiquantScale, key_antiquant_scale);
      append(
        inputs, mask, OptionalInput::KeyAntiquantOffset, key_antiquant_offset);
      append(inputs, mask, OptionalInput::ValueAntiquantScale,
        value_antiquant_scale);
      append(inputs, mask, OptionalInput::ValueAntiquantOffset,
        value_antiquant_offset);
      append(inputs, mask, OptionalInput::KeySharedPrefix, key_shared_prefix);
      append(
        inputs, mask, OptionalInput::ValueSharedPrefix, value_shared_prefix);
      append(inputs, mask, OptionalInput::ActualSharedPrefixLen,
        actual_shared_prefix_len);
      append(inputs, mask, OptionalInput::QueryRope, query_rope);
      append(inputs, mask, OptionalInput::KeyRope, key_rope);
      append(inputs, mask, OptionalInput::KeyRopeAntiquantScale,
        key_rope_antiquant_scale);
      append(
        inputs, mask, OptionalInput::DequantScaleQuery, dequant_scale_query);
      append(inputs, mask, OptionalInput::LearnableSink, learnable_sink);
      append(inputs, mask, OptionalInput::QStartIdx, q_start_idx);
      append(inputs, mask, OptionalInput::KvStartIdx, kv_start_idx);
    }

  private:
    static void append(TensorList &inputs, OptionalInputMask &mask,
      OptionalInput kind, const std::optional<Tensor> &tensor) {
      if (!tensor.has_value()) {
        return;
      }
      mask[static_cast<size_t>(kind)] = true;
      inputs.push_back(*tensor);
    }
  };

  struct Builder {
    Options options;

    Tensor operator()(Tensor query, Tensor key, Tensor value,
      const OptionalInputs &optional_inputs = {}) const {
      return (*this)(std::move(query), TensorList{std::move(key)},
        TensorList{std::move(value)}, optional_inputs);
    }

    Tensor operator()(Tensor query, TensorList key, TensorList value,
      const OptionalInputs &optional_inputs = {}) const {
      if (key.empty() || value.empty()) {
        throw exception(
          "fused_infer_attention: key and value must not be empty");
      }
      if (key.size() != value.size()) {
        throw exception(
          "fused_infer_attention: key and value must have the same length");
      }

      TensorList inputs;
      inputs.reserve(1 + key.size() + value.size() + OptionalInputCount);
      inputs.push_back(std::move(query));
      for (auto &tensor : key) {
        inputs.push_back(std::move(tensor));
      }
      for (auto &tensor : value) {
        inputs.push_back(std::move(tensor));
      }

      OptionalInputMask optional_input_mask{};
      optional_inputs.append_to(inputs, optional_input_mask);

      auto node = std::make_unique<FusedInferAttention>(options,
        static_cast<int64_t>(key.size()), static_cast<int64_t>(value.size()),
        optional_input_mask);
      if (options.decode_mla) {
        const auto &query_shape = inputs[0].shape();
        node->batch_size        = query_shape.empty() ? 0 : query_shape[0];
      }
      return {std::move(inputs), std::move(node)};
    }
  };

  Options           options;
  int64_t           key_num;
  int64_t           value_num;
  OptionalInputMask optional_input_mask;
  // Derived by Builder for decode_mla, alongside key/value count metadata.
  int64_t batch_size = 0;

  FusedInferAttention(Options options, int64_t key_num, int64_t value_num,
    OptionalInputMask optional_input_mask)
      : options(options), key_num(key_num), value_num(value_num),
        optional_input_mask(optional_input_mask) {}

  bool has_optional(OptionalInput input) const {
    return optional_input_mask[static_cast<size_t>(input)];
  }

  Shape shape(const TensorList &input) const override {
    validate_input_count(input);
    validate_options();

    const bool paged_bsnd = options.input_layout == InputLayout::BSND
      && has_optional(OptionalInput::BlockTable);
    if (options.input_layout == InputLayout::TND || paged_bsnd) {
      if (options.decode_mla)
        throw exception("fused_infer_attention: TND MLA scheduling is not "
                        "supported by this wrapper");
      const auto &q = input[0];
      if (q.dtype() != DType::BF16 && q.dtype() != DType::FP16)
        throw exception(
          "fused_infer_attention: paged attention requires BF16 or FP16");
      if (paged_bsnd)
        query_layout_info(q.shape());
      const auto head_dim =
        paged_bsnd ? q.shape()[3] : (q.dim() == 3 ? q.shape()[2] : 0);
      if ((!paged_bsnd && (q.dim() != 3 || q.shape()[1] != options.num_heads))
        || q.shape()[0] <= 0 || head_dim <= 0
        || (paged_bsnd && q.shape()[1] <= 0))
        throw exception(
          "fused_infer_attention: TND query must be [T,num_heads,D]");
      if (key_num != 1 || value_num != 1 || options.block_size <= 0)
        throw exception(
          "fused_infer_attention: paged attention requires one K/V "
          "pool and block_size > 0");
      auto optional_tensor = [&](OptionalInput kind) -> const Tensor & {
        if (!has_optional(kind))
          throw exception(
            "fused_infer_attention: paged attention requires block table and "
            "actual Q/KV lengths");
        size_t index = 1 + key_num + value_num;
        for (size_t i = 0; i < static_cast<size_t>(kind); ++i)
          index += optional_input_mask[i];
        return input.at(index);
      };
      const auto &table = optional_tensor(OptionalInput::BlockTable);
      const auto &ak    = optional_tensor(OptionalInput::ActualSeqLengthsKv);
      if (table.dtype() != DType::INT32 || table.shape().size() != 2
        || table.shape()[0] <= 0 || table.shape()[1] <= 0
        || ak.dtype() != DType::INT64 || ak.shape() != Shape{table.shape()[0]}
        || (paged_bsnd && table.shape()[0] != q.shape()[0]))
        throw exception(
          "fused_infer_attention: invalid paged block table/length metadata");
      if (!paged_bsnd || has_optional(OptionalInput::ActualSeqLengths)) {
        const auto &aq = optional_tensor(OptionalInput::ActualSeqLengths);
        if (aq.dtype() != DType::INT64 || aq.shape() != ak.shape())
          throw exception("fused_infer_attention: invalid paged Q lengths");
      }
      const auto &k = input[1];
      const auto &v = input[2];
      if (k.shape().size() != 3 || k.shape()[0] <= 0
        || k.shape()[1] != options.block_size
        || k.shape()[2] != effective_key_value_heads() * head_dim
        || k.shape() != v.shape() || k.dtype() != q.dtype()
        || v.dtype() != q.dtype())
        throw exception("fused_infer_attention: paged K/V must be "
                        "[blocks,block_size,KVH*D]");
      const bool qr_present = has_optional(OptionalInput::QueryRope);
      const bool kr_present = has_optional(OptionalInput::KeyRope);
      if (qr_present != kr_present)
        throw exception(
          "fused_infer_attention: TND MLA requires both RoPE inputs");
      if (qr_present) {
        const auto &qr          = optional_tensor(OptionalInput::QueryRope);
        const auto &kr          = optional_tensor(OptionalInput::KeyRope);
        const auto  rope_dim    = qr.dim() == q.dim() ? qr.shape().back() : 0;
        auto        expected_qr = q.shape();
        expected_qr.back()      = rope_dim;
        if (rope_dim <= 0 || qr.shape() != expected_qr
          || kr.shape()
            != Shape{k.shape()[0], options.block_size,
              effective_key_value_heads() * rope_dim}
          || qr.dtype() != q.dtype() || kr.dtype() != q.dtype())
          throw exception(
            "fused_infer_attention: invalid paged MLA RoPE metadata");
      }
      return q.shape();
    }
    const auto &query                 = input[0];
    const auto  info                  = query_layout_info(query.shape());
    const auto  kv_heads              = effective_key_value_heads();
    const bool  tensor_list_key_value = key_num > 1 || value_num > 1;

    if (tensor_list_key_value && key_num != info.batch) {
      throw exception("fused_infer_attention: TensorList key/value length must "
                      "equal query batch size");
    }

    for (int64_t i = 0; i < key_num; ++i) {
      const auto &key   = input[static_cast<size_t>(1 + i)];
      const auto &value = input[static_cast<size_t>(1 + key_num + i)];
      if (key.shape() != value.shape()) {
        throw exception(
          "fused_infer_attention: key and value shapes must match");
      }
      validate_key_value_shape(key.shape(), kv_heads, info.head_dim);
      if (tensor_list_key_value && key.shape()[0] != 1) {
        throw exception("fused_infer_attention: TensorList key/value tensors "
                        "must have batch size 1");
      }
    }

    if (options.input_layout == InputLayout::BNSD_BSND) {
      if (info.seq_len <= 1) {
        throw exception("fused_infer_attention: BNSD_BSND requires Q_S > 1");
      }
      return {info.batch, info.seq_len, info.num_heads, info.head_dim};
    }
    return query.shape();
  }

  DType dtype(const TensorList &input) const override {
    if (options.output_dtype.has_value()) {
      return *options.output_dtype;
    }
    return input[0].dtype();
  }

private:
  struct LayoutInfo {
    int64_t batch;
    int64_t seq_len;
    int64_t num_heads;
    int64_t head_dim;
  };

  void validate_input_count(const TensorList &input) const {
    if (key_num <= 0 || value_num <= 0) {
      throw exception("fused_infer_attention: key and value must not be empty");
    }
    if (key_num != value_num) {
      throw exception(
        "fused_infer_attention: key and value must have the same length");
    }

    size_t optional_count = 0;
    for (bool present : optional_input_mask) {
      if (present) {
        ++optional_count;
      }
    }
    const auto expected =
      static_cast<size_t>(1 + key_num + value_num) + optional_count;
    if (input.size() != expected) {
      throw exception(
        "fused_infer_attention: internal input count does not match metadata");
    }
  }

  void validate_options() const {
    if (options.num_heads <= 0) {
      throw exception("fused_infer_attention: num_heads must be positive");
    }

    const auto kv_heads = effective_key_value_heads();
    if (kv_heads <= 0) {
      throw exception(
        "fused_infer_attention: num_key_value_heads must be positive");
    }
    if (options.num_heads % kv_heads != 0) {
      throw exception("fused_infer_attention: num_heads must be divisible by "
                      "num_key_value_heads");
    }
    if (options.num_heads / kv_heads > 64) {
      throw exception(
        "fused_infer_attention: num_heads / num_key_value_heads must be <= 64");
    }
    if (options.output_dtype.has_value() && *options.output_dtype != DType::FP16
      && *options.output_dtype != DType::BF16) {
      throw exception(
        "fused_infer_attention: output_dtype must be FP16 or BF16");
    }
  }

  int64_t effective_key_value_heads() const {
    return options.num_key_value_heads == 0 ? options.num_heads
                                            : options.num_key_value_heads;
  }

  LayoutInfo query_layout_info(const Shape &shape) const {
    switch (options.input_layout) {
    case InputLayout::TND:
      throw exception("fused_infer_attention: TND uses paged validation");
    case InputLayout::BSH:
      if (shape.size() != 3) {
        throw exception("fused_infer_attention: BSH query must be 3D");
      }
      if (shape[2] % options.num_heads != 0) {
        throw exception(
          "fused_infer_attention: BSH hidden size must be divisible by "
          "num_heads");
      }
      return {.batch = shape[0],
        .seq_len     = shape[1],
        .num_heads   = options.num_heads,
        .head_dim    = shape[2] / options.num_heads};

    case InputLayout::BSND:
      if (shape.size() != 4) {
        throw exception("fused_infer_attention: BSND query must be 4D");
      }
      if (shape[2] != options.num_heads) {
        throw exception(
          "fused_infer_attention: BSND query N must equal num_heads");
      }
      return {.batch = shape[0],
        .seq_len     = shape[1],
        .num_heads   = shape[2],
        .head_dim    = shape[3]};

    case InputLayout::BNSD:
    case InputLayout::BNSD_BSND:
      if (shape.size() != 4) {
        throw exception("fused_infer_attention: BNSD query must be 4D");
      }
      if (shape[1] != options.num_heads) {
        throw exception(
          "fused_infer_attention: BNSD query N must equal num_heads");
      }
      return {.batch = shape[0],
        .seq_len     = shape[2],
        .num_heads   = shape[1],
        .head_dim    = shape[3]};
    }

    throw exception("fused_infer_attention: unsupported input layout");
  }

  void validate_key_value_shape(
    const Shape &shape, int64_t kv_heads, int64_t query_head_dim) const {
    switch (options.input_layout) {
    case InputLayout::TND:
      throw exception("fused_infer_attention: TND uses paged validation");
    case InputLayout::BSH:
      if (shape.size() != 3) {
        throw exception("fused_infer_attention: BSH key/value must be 3D");
      }
      if (shape[2] % kv_heads != 0 || shape[2] / kv_heads != query_head_dim) {
        throw exception(
          "fused_infer_attention: BSH key/value hidden size must match "
          "num_key_value_heads * head_dim");
      }
      return;

    case InputLayout::BSND:
      if (shape.size() != 4) {
        throw exception("fused_infer_attention: BSND key/value must be 4D");
      }
      if (shape[2] != kv_heads || shape[3] != query_head_dim) {
        throw exception("fused_infer_attention: BSND key/value N,D must match "
                        "num_key_value_heads and query head_dim");
      }
      return;

    case InputLayout::BNSD:
    case InputLayout::BNSD_BSND:
      if (shape.size() != 4) {
        throw exception("fused_infer_attention: BNSD key/value must be 4D");
      }
      if (shape[1] != kv_heads || shape[3] != query_head_dim) {
        throw exception("fused_infer_attention: BNSD key/value N,D must match "
                        "num_key_value_heads and query head_dim");
      }
      return;
    }

    throw exception("fused_infer_attention: unsupported input layout");
  }
};

inline FusedInferAttention::Builder fused_infer_attention(
  FusedInferAttention::Options options) {
  return {options};
}

inline FusedInferAttention::Builder fused_infer_attention(int64_t num_heads) {
  return fused_infer_attention(FusedInferAttention::Options(num_heads));
}

} // namespace forch::op
