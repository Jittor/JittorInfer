#pragma once

// Single dispatch point from an AttentionType to its scheme. Models call
// these after projection/norm/RoPE/cache update and before output
// projection; no model repeats the switch.

#include "../../../cache/full-attention.hpp"
#include "eager-attention.hpp"
#include "flash-attention-full.hpp"
#include "flash-attention-mla.hpp"

namespace finer::models {

/// Full attention over token-major Q [T,H,D] and K/V pool views
/// [rows,KVH,D]. Returns [T,H,D].
inline forch::Tensor full_attention(cache::AttentionType type,
  const cache::AttentionInputs &in, const forch::Tensor &q,
  const forch::Tensor &k, const forch::Tensor &v, float scale,
  const std::string &debug_prefix = {}) {
  using cache::AttentionType;
  switch (type) {
  case AttentionType::Eager: {
    auto heads = [](const forch::Tensor &x) {
      return x | forch::op::permute({1, 0, 2});
    };
    return eager_attention(
      heads(q), heads(k), heads(v), in.eager_mask, scale, debug_prefix);
  }
  case AttentionType::DenseFIA:
    return flash_attention_full_dense(in, q, k, v, scale);
  case AttentionType::TND:
    return flash_attention_full_tnd(in, q, k, v, scale);
  case AttentionType::BSND:
    return flash_attention_full_bsnd(in, q, k, v, scale, false);
  case AttentionType::PagedBSND:
    return flash_attention_full_bsnd(in, q, k, v, scale, true);
  }
  throw forch::exception("full attention: unsupported scheme");
}

/// Absorbed MLA; eager MLA is a model-specific reference.
inline forch::Tensor mla_attention(cache::AttentionType type,
  const cache::AttentionInputs &in, const forch::Tensor &q,
  const forch::Tensor &qr, const forch::Tensor &latent, const forch::Tensor &kr,
  float scale) {
  using cache::AttentionType;
  switch (type) {
  case AttentionType::TND:
    return flash_attention_mla_tnd(in, q, qr, latent, kr, scale);
  case AttentionType::BSND:
    return flash_attention_mla_bsnd(in, q, qr, latent, kr, scale, false);
  case AttentionType::PagedBSND:
    return flash_attention_mla_bsnd(in, q, qr, latent, kr, scale, true);
  default:
    throw forch::exception("MLA attention: unsupported fused scheme");
  }
}

} // namespace finer::models
