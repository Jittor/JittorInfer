#pragma once

/// FullAttentionCache: one layer's K and V pools. Addressing, cursors and
/// attention metadata are shared runner-wide (SlotMap / AttentionMetadata);
/// this class only scatters the step's K/V into its pools.
///
/// K and V may differ in geometry (DeepSeek's eager MLA reference stores
/// K = nope + rope and V = v_head_dim). Inputs and views are token-major
/// [rows, heads, dim]; head-major eager references permute the view.

#include "attention-metadata.hpp"
#include "pool.hpp"

#include <forch/ops/reshape.hpp>

#include <string>
#include <utility>

namespace cache {

struct HeadGeometry {
  int64_t heads = 0;
  int64_t dim   = 0;
};

class FullAttentionCache {
public:
  struct Config {
    forch::Backend *backend = nullptr;
    int64_t         rows    = 0; // SlotMap::storage_rows()
    HeadGeometry    k, v;
    forch::DType    dtype = forch::DType::BF16;
    std::string     name  = "L0";
  };

  explicit FullAttentionCache(const Config &cfg)
      : cfg_(cfg), k_(cfg.backend, "K_" + cfg.name, cfg.rows,
                     cfg.k.heads * cfg.k.dim, cfg.dtype),
        v_(cfg.backend, "V_" + cfg.name, cfg.rows, cfg.v.heads * cfg.v.dim,
          cfg.dtype) {}

  void init() {
    k_.init();
    v_.init();
  }

  /// Scatter this step's K/V [T, heads, dim] and return full-pool views
  /// [rows, heads, dim], including dummy/guard/padding rows that the
  /// attention metadata always masks.
  std::pair<forch::Tensor, forch::Tensor> update(
    forch::Tensor key, forch::Tensor value, const AttentionInputs &in) {
    return {write(k_, cfg_.k, std::move(key), in),
      write(v_, cfg_.v, std::move(value), in)};
  }

  void feed(forch::TensorList &inputs) const {
    inputs.push_back(k_.tensor());
    inputs.push_back(v_.tensor());
  }

private:
  static forch::Tensor write(Pool &pool, const HeadGeometry &geometry,
    forch::Tensor input, const AttentionInputs &in) {
    const auto tokens = input.shape()[0];
    if (tokens != in.tokens)
      throw forch::exception("FullAttentionCache: token budget mismatch");
    auto updated = pool.scatter(*in.write_rows,
      forch::op::reshape({tokens, pool.width()})(std::move(input)));
    return forch::op::reshape({pool.rows(), geometry.heads, geometry.dim})(
      std::move(updated));
  }

  Config cfg_;
  Pool   k_, v_;
};

} // namespace cache
