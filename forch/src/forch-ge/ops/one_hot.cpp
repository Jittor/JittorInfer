#include "forch/ops/one_hot.hpp"
#include "op-registry.hpp"

#include "es_OneHot_c.h"

namespace {

// OneHot GE mapping.
//
// Supported output dtypes (910B): INT32 / INT64 only. The kernel ALWAYS
// writes INT32 data regardless of the declared output dtype — an FP32
// buffer gets the on-value as the raw int bit pattern (0x1 = 1.4e-45f
// instead of 1.0f), and an FP16 buffer only holds half of the INT32
// result. For a float mask use INT32 output + cast(FP32).
//
// on/off scalars (1/0) are created in an accepted CANN scalar type
// matching the output (INT32 or INT64). The axis is the normalized
// insertion position (axis_resolved, resolved in shape()) — CANN rejects
// raw negatives below -1.

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::OneHot>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::OneHot *op) {
      auto *depth =
        EsCreateScalarInt32(ctx.builder, static_cast<int32_t>(op->depth));
      EsCTensorHolder *on  = nullptr;
      EsCTensorHolder *off = nullptr;
      switch (op->dtype_) {
      case forch::DType::INT32:
        on  = EsCreateScalarInt32(ctx.builder, 1);
        off = EsCreateScalarInt32(ctx.builder, 0);
        break;
      case forch::DType::INT64:
        on  = EsCreateScalarInt64(ctx.builder, 1);
        off = EsCreateScalarInt64(ctx.builder, 0);
        break;
      default:
        throw forch::exception(
          "one_hot: output dtype not supported by the GE backend "
          "(INT32/INT64 only — the 910B kernel always writes INT32 data, "
          "FP16/FP32 outputs come out garbled)");
      }
      return EsOneHot(ctx.inputs[0], depth, on, off, op->axis_resolved);
    });
}

} // namespace
