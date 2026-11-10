#include "forch/ops/fused_infer_attention.hpp"
#include "forch-ge-common.hpp"
#include "op-registry.hpp"

#include "acl/acl.h"

#include "es_FusedInferAttentionScore_c.h"

#include "acl/acl_rt.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace {

const char *input_layout_c_str(
  forch::op::FusedInferAttention::InputLayout layout) {
  using InputLayout = forch::op::FusedInferAttention::InputLayout;
  switch (layout) {
  case InputLayout::TND:
    return "TND";
  case InputLayout::BSH:
    return "BSH";
  case InputLayout::BSND:
    return "BSND";
  case InputLayout::BNSD:
    return "BNSD";
  case InputLayout::BNSD_BSND:
    return "BNSD_BSND";
  }
  return "BSH";
}

int64_t output_dtype_attr(const forch::op::FusedInferAttention *op) {
  if (!op->options.output_dtype.has_value()) {
    return 0;
  }
  return static_cast<int64_t>(
    forch::detail::CDtypeDesc(*op->options.output_dtype).type);
}

void set_core_num(EsCTensorHolder *output, const char *name, int64_t count) {
  const auto value = std::to_string(count);
  if (EsSetStringAttrForNode(output, name, value.c_str()) != 0)
    throw forch::exception("Unable to set MLA decode core count");
}

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::FusedInferAttention>(
    [](const forch::detail::GEOpContext    &ctx,
      const forch::op::FusedInferAttention *op) {
      using FusedInferAttention = forch::op::FusedInferAttention;
      using OptionalInput       = FusedInferAttention::OptionalInput;

      const auto key_begin   = static_cast<size_t>(1);
      const auto key_end     = key_begin + static_cast<size_t>(op->key_num);
      const auto value_begin = key_end;
      const auto value_end   = value_begin + static_cast<size_t>(op->value_num);

      std::vector<EsCTensorHolder *> key(
        ctx.inputs.data() + key_begin, ctx.inputs.data() + key_end);
      std::vector<EsCTensorHolder *> value(
        ctx.inputs.data() + value_begin, ctx.inputs.data() + value_end);

      std::array<EsCTensorHolder *, FusedInferAttention::OptionalInputCount>
             optional{};
      size_t cursor = value_end;
      for (size_t i = 0; i < optional.size(); ++i) {
        auto kind = static_cast<OptionalInput>(i);
        if (op->has_optional(kind)) {
          optional[i] = ctx.inputs[cursor++];
        }
      }

      auto pick = [&optional](OptionalInput input) {
        return optional[static_cast<size_t>(input)];
      };

      auto out = EsFusedInferAttentionScore(ctx.inputs[0], key.data(),
        op->key_num, value.data(), op->value_num, pick(OptionalInput::PseShift),
        pick(OptionalInput::AttenMask), pick(OptionalInput::ActualSeqLengths),
        pick(OptionalInput::ActualSeqLengthsKv),
        pick(OptionalInput::DequantScale1), pick(OptionalInput::QuantScale1),
        pick(OptionalInput::DequantScale2), pick(OptionalInput::QuantScale2),
        pick(OptionalInput::QuantOffset2), pick(OptionalInput::AntiquantScale),
        pick(OptionalInput::AntiquantOffset), pick(OptionalInput::BlockTable),
        pick(OptionalInput::QueryPaddingSize),
        pick(OptionalInput::KvPaddingSize),
        pick(OptionalInput::KeyAntiquantScale),
        pick(OptionalInput::KeyAntiquantOffset),
        pick(OptionalInput::ValueAntiquantScale),
        pick(OptionalInput::ValueAntiquantOffset),
        pick(OptionalInput::KeySharedPrefix),
        pick(OptionalInput::ValueSharedPrefix),
        pick(OptionalInput::ActualSharedPrefixLen),
        pick(OptionalInput::QueryRope), pick(OptionalInput::KeyRope),
        pick(OptionalInput::KeyRopeAntiquantScale),
        pick(OptionalInput::DequantScaleQuery),
        pick(OptionalInput::LearnableSink), pick(OptionalInput::QStartIdx),
        pick(OptionalInput::KvStartIdx), op->options.num_heads,
        op->options.scale, op->options.pre_tokens, op->options.next_tokens,
        input_layout_c_str(op->options.input_layout),
        op->options.num_key_value_heads, op->options.sparse_mode,
        op->options.inner_precise, op->options.block_size,
        op->options.antiquant_mode, op->options.softmax_lse_flag,
        op->options.key_antiquant_mode, op->options.value_antiquant_mode,
        op->options.query_quant_mode, op->options.pse_type,
        output_dtype_attr(op));
      if (op->options.decode_mla) {
        // Uniform scan lengths and whole-slot work per Cube avoid this
        // SDK's MLA FlashDecode split/merge path. Keep Vector capacity at
        // twice Cube capacity, including when batch exceeds physical cores.
        const auto batch = op->batch_size;
        if (batch <= 0)
          throw forch::exception("Invalid MLA decode batch size");
        int32_t device     = 0;
        int64_t cube_limit = 0, vector_limit = 0;
        if (aclrtGetDevice(&device) != ACL_SUCCESS
          || aclrtGetDeviceInfo(device, ACL_DEV_ATTR_CUBE_CORE_NUM, &cube_limit)
            != ACL_SUCCESS
          || aclrtGetDeviceInfo(
               device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vector_limit)
            != ACL_SUCCESS)
          throw forch::exception("Cannot query MLA decode device core limits");
        auto cores = std::min({batch, cube_limit, vector_limit / 2});
        if (cores <= 0)
          throw forch::exception("Invalid MLA decode device core limits");
        while (batch % cores != 0)
          --cores;
        set_core_num(out.attention_out, "_op_aicore_num", cores);
        set_core_num(out.attention_out, "_op_vectorcore_num", 2 * cores);
      }
      return out.attention_out;
    });
}

} // namespace
