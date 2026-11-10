#include "forch/ops/fused_collective_matmul.hpp"
#include "forch-ge-common.hpp"
#include "hccl-common.hpp"
#include "op-registry.hpp"

#include "es_AllGatherMatmul_c.h"
#include "es_MatmulAllReduceAddRmsNorm_c.h"
#include "es_MatmulAllReduce_c.h"
#include "es_MatmulReduceScatter_c.h"
#include "graph/c_types.h"

namespace {

// Other dtypes require quantization inputs that these emitters do not expose.
void check_fused_matmul_dtype(const char *who, forch::DType dtype) {
  if (dtype != forch::DType::FP16 && dtype != forch::DType::BF16) {
    throw forch::exception(std::string(who)
      + ": the fused matmul accepts FP16/BF16 only; other dtypes would need "
        "the antiquant/dequant-scale inputs, which this backend does not "
        "expose");
  }
}

// The proto only supports comm_turn = 0 in the current version.
constexpr int64_t kCommTurn = 0;

// The proto forbids transposing x1, and the front end exposes no such argument.
constexpr bool kTransA = false;

__attribute__((constructor)) void register_fused_collective_matmul_ops() {
  using namespace forch::detail;

  GEOpRegistry::register_op<forch::op::FusedMatmulAllReduce>(
    [](const GEOpContext &ctx, const forch::op::FusedMatmulAllReduce *op) {
      check_fused_matmul_dtype(
        "fused_matmul_all_reduce", ctx.input_tensors[0].dtype());
      // y_dtype has to be stated. Left at C_DT_UNDEFINED, GE infers DT_FLOAT
      // for y and tiling then rejects it with a message that never mentions
      // y_dtype. An EsCTensorHolder can only be told its dtype, never asked,
      // so run the op's own dtype rule over the input tensors instead of
      // second-guessing it here.
      const auto y_dtype = static_cast<int64_t>(
        forch::detail::CDtypeDesc(op->dtype(ctx.input_tensors)).type);
      // Non-quantized path, so every optional input is nullptr.
      EsCTensorHolder *bias = ctx.inputs.size() >= 3 ? ctx.inputs[2] : nullptr;
      return EsMatmulAllReduce(ctx.inputs[0], ctx.inputs[1], bias,
        /*x3=*/nullptr, /*antiquant_scale=*/nullptr,
        /*antiquant_offset=*/nullptr, /*dequant_scale=*/nullptr,
        /*pertoken_scale=*/nullptr, /*comm_quant_scale_1=*/nullptr,
        /*comm_quant_scale_2=*/nullptr,
        hccl_group_name(op->group.group_id()).c_str(),
        hccl_reduce_op(op->reduce_op), kTransA, op->transpose_b, kCommTurn,
        /*antiquant_group_size=*/0, /*group_size=*/0, y_dtype,
        /*comm_quant_mode=*/0);
    });

  GEOpRegistry::register_op<forch::op::FusedMatmulAllReduceAddRmsNorm>(
    [](const GEOpContext                              &ctx,
      const forch::op::FusedMatmulAllReduceAddRmsNorm *op) {
      check_fused_matmul_dtype(
        "fused_matmul_all_reduce_add_rms_norm", ctx.input_tensors[0].dtype());
      // bias sits before the required residual/gamma: x1, x2, bias, res, gamma.
      // Only y is taken; norm_out is recomputed by the front end.
      auto out = EsMatmulAllReduceAddRmsNorm(ctx.inputs[0], ctx.inputs[1],
        /*bias=*/nullptr, ctx.inputs[2], ctx.inputs[3],
        /*antiquant_scale=*/nullptr,
        /*antiquant_offset=*/nullptr, /*dequant_scale=*/nullptr,
        hccl_group_name(op->group.group_id()).c_str(),
        hccl_reduce_op(op->reduce_op), kTransA, op->transpose_b, kCommTurn,
        /*antiquant_group_size=*/0, op->epsilon);
      if (out.y == nullptr) {
        throw forch::exception("fused_matmul_all_reduce_add_rms_norm: "
                               "EsMatmulAllReduceAddRmsNorm produced no y "
                               "handle");
      }
      return out.y;
    });

  GEOpRegistry::register_op<forch::op::FusedAllGatherMatmul>(
    [](const GEOpContext &ctx, const forch::op::FusedAllGatherMatmul *op) {
      check_fused_matmul_dtype(
        "fused_all_gather_matmul", ctx.input_tensors[0].dtype());
      // gather_out is a big tensor nobody asked for; y is unaffected by it.
      auto out = EsAllGatherMatmul(ctx.inputs[0], ctx.inputs[1],
        /*bias=*/nullptr, hccl_group_name(op->group.group_id()).c_str(),
        kTransA, op->transpose_b, /*gather_index=*/0, kCommTurn,
        op->group.world_size(),
        /*is_gather_out=*/false);
      if (out.y == nullptr) {
        throw forch::exception("fused_all_gather_matmul: EsAllGatherMatmul "
                               "produced no y handle");
      }
      return out.y;
    });

  GEOpRegistry::register_op<forch::op::FusedMatmulReduceScatter>(
    [](const GEOpContext &ctx, const forch::op::FusedMatmulReduceScatter *op) {
      check_fused_matmul_dtype(
        "fused_matmul_reduce_scatter", ctx.input_tensors[0].dtype());
      return EsMatmulReduceScatter(ctx.inputs[0], ctx.inputs[1],
        /*bias=*/nullptr, hccl_group_name(op->group.group_id()).c_str(),
        hccl_reduce_op(op->reduce_op), kTransA, op->transpose_b, kCommTurn,
        op->group.world_size());
    });
}

} // namespace
