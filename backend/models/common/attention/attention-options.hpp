#pragma once
#include "../../../cache/attention-metadata.hpp"
#include <cmath>
#include <forch/ops/fused_infer_attention.hpp>
#include <forch/ops/reshape.hpp>

namespace finer::models {

// CANN FusedInferAttentionScore attribute values, translated only here.
enum class AttentionPrecision : int64_t {
  HighPerformance = 1,
  HighPrecision   = 2
};
enum class AttentionMaskMode : int64_t { Explicit = 0, RightAlignedCausal = 3 };

inline forch::op::FusedInferAttention::Options flash_options(int64_t heads,
  int64_t kv_heads, float scale, forch::DType dtype,
  forch::op::FusedInferAttention::InputLayout layout,
  AttentionPrecision precision, AttentionMaskMode mask_mode) {
  forch::op::FusedInferAttention::Options options(heads);
  options.num_key_value_heads = kv_heads;
  options.scale               = scale;
  options.output_dtype        = dtype;
  options.input_layout        = layout;
  options.inner_precise       = static_cast<int64_t>(precision);
  options.sparse_mode         = static_cast<int64_t>(mask_mode);
  return options;
}

// Fused full-attention entry points accept token-major projections and cache
// views only. Eager layout conversion stays explicit at the model call site.
inline void validate_flash_full(const forch::Tensor &q, const forch::Tensor &k,
  const forch::Tensor &v, float scale) {
  if (q.dim() != 3 || k.dim() != 3 || k.shape() != v.shape()
    || q.shape()[0] <= 0 || q.shape()[1] <= 0 || q.shape()[2] <= 0
    || k.shape()[0] <= 0 || k.shape()[1] <= 0 || k.shape()[2] != q.shape()[2]
    || q.shape()[1] % k.shape()[1] || q.dtype() != k.dtype()
    || q.dtype() != v.dtype()
    || (q.dtype() != forch::DType::FP16 && q.dtype() != forch::DType::BF16)
    || !std::isfinite(scale))
    throw forch::exception(
      "flash full attention requires matching TND FP16/BF16 Q/K/V");
}

inline void validate_flash_mla(const forch::Tensor &q, const forch::Tensor &qr,
  const forch::Tensor &latent, const forch::Tensor &kr, float scale) {
  validate_flash_full(q, latent, latent, scale);
  if (latent.shape()[1] != 1 || qr.dim() != 3 || kr.dim() != 3
    || qr.shape()[0] != q.shape()[0] || qr.shape()[1] != q.shape()[1]
    || qr.shape()[2] <= 0
    || kr.shape() != forch::Shape{latent.shape()[0], 1, qr.shape()[2]}
    || qr.dtype() != q.dtype() || kr.dtype() != q.dtype())
    throw forch::exception(
      "flash MLA requires TND latent and matching RoPE inputs");
}

inline forch::Tensor flash_paged_view(const forch::Tensor &x, int64_t block) {
  if (block <= 0 || x.dim() != 3 || x.shape()[0] % block)
    throw forch::exception("flash attention: invalid paged pool geometry");
  return x
    | forch::op::reshape(
      {x.shape()[0] / block, block, x.shape()[1] * x.shape()[2]});
}

// Slot-major 4-D view [B, rows/B, H, D] of a token-major [rows, H, D] tensor.
inline forch::Tensor flash_slot_view(const forch::Tensor &x, int64_t slots) {
  if (slots <= 0 || x.dim() != 3 || x.shape()[0] % slots)
    throw forch::exception("flash attention: rows must split evenly by slot");
  return x
    | forch::op::reshape(
      {slots, x.shape()[0] / slots, x.shape()[1], x.shape()[2]});
}

template <class T>
const T &required(const std::optional<T> &value, const char *what) {
  if (!value)
    throw forch::exception(std::string("flash attention: missing ") + what);
  return *value;
}

} // namespace finer::models
