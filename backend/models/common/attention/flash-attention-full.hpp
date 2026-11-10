#pragma once

// Fused full-attention schemes. Each entry is one FusedInferAttentionScore
// calling convention; layout, mask and storage addressing are coupled inside
// the kernel, so the schemes stay parallel rather than composed. Inputs are
// token-major projections and cache views; outputs are [T,H,D] before the
// model's output projection.

#include "attention-options.hpp"

namespace finer::models {

// All slots flattened into one sequence with a full [T, rows] mask. Kept as
// the ablation reference for the fused kernels.
inline forch::Tensor flash_attention_full_dense(
  const cache::AttentionInputs &in, const forch::Tensor &q,
  const forch::Tensor &k, const forch::Tensor &v, float scale) {
  using FIA = forch::op::FusedInferAttention;
  validate_flash_full(q, k, v, scale);
  const auto &mask = required(in.fia_mask, "full physical-row mask");
  if (mask.shape() != forch::Shape{q.shape()[0], k.shape()[0]})
    throw forch::exception("flash full dense: mask does not match the pool");
  auto options = flash_options(q.shape()[1], k.shape()[1], scale, q.dtype(),
    FIA::InputLayout::BSND, AttentionPrecision::HighPerformance,
    AttentionMaskMode::Explicit);
  FIA::OptionalInputs extra;
  extra.atten_mask = mask | forch::op::reshape({1, q.shape()[0], k.shape()[0]});
  return forch::op::fused_infer_attention(options)(flash_slot_view(q, 1),
           flash_slot_view(k, 1), flash_slot_view(v, 1), extra)
    | forch::op::reshape(q.shape());
}

// Compact Q [T,H,D] over block-addressed K/V; causal suffix attention with
// cumulative query lengths and dummy padding sequences.
inline forch::Tensor flash_attention_full_tnd(const cache::AttentionInputs &in,
  const forch::Tensor &q, const forch::Tensor &k, const forch::Tensor &v,
  float scale) {
  using FIA = forch::op::FusedInferAttention;
  validate_flash_full(q, k, v, scale);
  auto options = flash_options(q.shape()[1], k.shape()[1], scale, q.dtype(),
    FIA::InputLayout::TND,
    q.dtype() == forch::DType::FP16 ? AttentionPrecision::HighPrecision
                                    : AttentionPrecision::HighPerformance,
    AttentionMaskMode::RightAlignedCausal);
  options.block_size = in.block_size;
  FIA::OptionalInputs extra;
  extra.atten_mask            = required(in.causal_mask, "causal mask");
  extra.actual_seq_lengths    = required(in.query_lengths, "query lengths");
  extra.actual_seq_lengths_kv = required(in.kv_lengths, "KV lengths");
  extra.block_table           = required(in.block_table, "block table");
  return forch::op::fused_infer_attention(options)(q,
    flash_paged_view(k, in.block_size), flash_paged_view(v, in.block_size),
    extra);
}

// Slot-ordered Q [B*Q,H,D]. Contiguous storage reads [B,S,N,D] slot views;
// paged storage reads [blocks, block, N*D] through the slot block table.
// Multi-query chunks also need the per-query causal mask.
inline forch::Tensor flash_attention_full_bsnd(const cache::AttentionInputs &in,
  const forch::Tensor &q, const forch::Tensor &k, const forch::Tensor &v,
  float scale, bool paged) {
  using FIA = forch::op::FusedInferAttention;
  validate_flash_full(q, k, v, scale);
  const auto b = in.slots;
  if (q.shape()[0] % b || (q.shape()[0] > b && !in.decode_mask))
    throw forch::exception(
      "flash full BSND requires equal slot packing and a chunk mask");
  auto options = flash_options(q.shape()[1], k.shape()[1], scale, q.dtype(),
    FIA::InputLayout::BSND, AttentionPrecision::HighPerformance,
    AttentionMaskMode::Explicit);
  FIA::OptionalInputs extra;
  extra.atten_mask = in.decode_mask;
  extra.actual_seq_lengths_kv =
    required(in.decode_kv_lengths, "decode KV lengths");
  // Paged pools are not slot-divisible; only contiguous storage has slot views.
  forch::Tensor key = k, value = v;
  if (paged) {
    options.block_size = in.block_size;
    extra.block_table  = required(in.slot_block_table, "slot block table");
    key                = flash_paged_view(k, in.block_size);
    value              = flash_paged_view(v, in.block_size);
  } else {
    key   = flash_slot_view(k, b);
    value = flash_slot_view(v, b);
  }
  return forch::op::fused_infer_attention(options)(
           flash_slot_view(q, b), key, value, extra)
    | forch::op::reshape(q.shape());
}

} // namespace finer::models
