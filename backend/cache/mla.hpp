#pragma once

/// MLACache: one layer's absorbed-MLA storage, post-norm latent c_kv
/// [rows, kv_lora_rank] and shared roped k_pe [rows, qk_rope_head_dim], both
/// token-major. Weight absorption belongs to the model; addressing and
/// metadata are shared runner-wide like FullAttentionCache.

#include "attention-metadata.hpp"
#include "pool.hpp"

#include <forch/ops/reshape.hpp>

#include <string>
#include <utility>

namespace cache {

class MLACache {
public:
  struct Config {
    forch::Backend *backend          = nullptr;
    int64_t         rows             = 0; // SlotMap::storage_rows()
    int64_t         kv_lora_rank     = 0;
    int64_t         qk_rope_head_dim = 0;
    forch::DType    dtype            = forch::DType::BF16;
    std::string     name             = "L0";
  };

  explicit MLACache(const Config &cfg)
      : latent_(cfg.backend, cfg.name + "_latent", cfg.rows, cfg.kv_lora_rank,
          cfg.dtype),
        k_pe_(cfg.backend, cfg.name + "_k_pe", cfg.rows, cfg.qk_rope_head_dim,
          cfg.dtype) {}

  void init() {
    latent_.init();
    k_pe_.init();
  }

  /// Inputs are token-major [T,1,*]; returns pool views [rows,1,*].
  std::pair<forch::Tensor, forch::Tensor> update(
    forch::Tensor latent, forch::Tensor k_pe, const AttentionInputs &in) {
    return {
      write(latent_, std::move(latent), in), write(k_pe_, std::move(k_pe), in)};
  }

  void feed(forch::TensorList &inputs) const {
    inputs.push_back(latent_.tensor());
    inputs.push_back(k_pe_.tensor());
  }

private:
  static forch::Tensor write(
    Pool &pool, forch::Tensor input, const AttentionInputs &in) {
    const auto tokens = input.shape()[0];
    if (tokens != in.tokens)
      throw forch::exception("MLACache: token budget mismatch");
    auto updated = pool.scatter(*in.write_rows,
      forch::op::reshape({tokens, pool.width()})(std::move(input)));
    return forch::op::reshape({pool.rows(), 1, pool.width()})(
      std::move(updated));
  }

  Pool latent_, k_pe_;
};

} // namespace cache
