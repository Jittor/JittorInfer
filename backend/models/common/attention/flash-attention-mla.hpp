#pragma once

// Fused absorbed-MLA schemes: absorbed Q [T,H,latent], separate Q/K RoPE,
// latent used as both K and V. Outputs are [T,H,latent]; weight absorption
// and unabsorption belong to the model.

#include "attention-options.hpp"

namespace finer::models {

inline forch::Tensor flash_attention_mla_tnd(const cache::AttentionInputs &in,
  const forch::Tensor &q, const forch::Tensor &qr, const forch::Tensor &latent,
  const forch::Tensor &kr, float scale) {
  using FIA = forch::op::FusedInferAttention;
  validate_flash_mla(q, qr, latent, kr, scale);
  auto options =
    flash_options(q.shape()[1], 1, scale, q.dtype(), FIA::InputLayout::TND,
      AttentionPrecision::HighPrecision, AttentionMaskMode::RightAlignedCausal);
  options.block_size = in.block_size;
  FIA::OptionalInputs extra;
  extra.atten_mask            = required(in.causal_mask, "causal mask");
  extra.actual_seq_lengths    = required(in.query_lengths, "query lengths");
  extra.actual_seq_lengths_kv = required(in.kv_lengths, "KV lengths");
  extra.block_table           = required(in.block_table, "block table");
  extra.query_rope            = qr;
  extra.key_rope              = flash_paged_view(kr, in.block_size);
  auto kv                     = flash_paged_view(latent, in.block_size);
  return forch::op::fused_infer_attention(options)(q, kv, kv, extra);
}

// One query per slot. Dense storage uses uniform scan lengths plus a mask
// (the metadata supplies both) to avoid CANN's split/merge path; paged
// storage uses independent lengths and the slot block table.
inline forch::Tensor flash_attention_mla_bsnd(const cache::AttentionInputs &in,
  const forch::Tensor &q, const forch::Tensor &qr, const forch::Tensor &latent,
  const forch::Tensor &kr, float scale, bool paged) {
  using FIA = forch::op::FusedInferAttention;
  validate_flash_mla(q, qr, latent, kr, scale);
  const auto b = q.shape()[0];
  if (b != in.slots || (!paged && !in.decode_mask))
    throw forch::exception(
      "flash MLA BSND requires one query per slot (and a mask when dense)");
  auto options =
    flash_options(q.shape()[1], 1, scale, q.dtype(), FIA::InputLayout::BSND,
      AttentionPrecision::HighPrecision, AttentionMaskMode::Explicit);
  FIA::OptionalInputs extra;
  extra.actual_seq_lengths_kv =
    required(in.decode_kv_lengths, "decode KV lengths");
  extra.query_rope = flash_slot_view(qr, b);
  forch::Tensor kv = latent;
  if (paged) {
    options.block_size = in.block_size;
    extra.block_table  = required(in.slot_block_table, "slot block table");
    extra.key_rope     = flash_paged_view(kr, in.block_size);
    kv                 = flash_paged_view(latent, in.block_size);
  } else {
    options.decode_mla = true;
    extra.atten_mask   = in.decode_mask;
    extra.key_rope     = flash_slot_view(kr, b);
    kv                 = flash_slot_view(latent, b);
  }
  return forch::op::fused_infer_attention(options)(
           flash_slot_view(q, b), kv, kv, extra)
    | forch::op::reshape(q.shape());
}

} // namespace finer::models
