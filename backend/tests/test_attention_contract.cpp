// Graph-construction checks for every attention scheme: shapes, required
// inputs, dtype validation, MLA RoPE and kernel attributes. No device needed.
#include "../models/common/attention/attention.hpp"
#include <iostream>

using namespace forch;
using namespace finer::models;
using cache::AttentionType;

static void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
template <class F>
static void rejects(F fn) {
  try {
    fn();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("unsupported attention inputs accepted");
}
static const op::FusedInferAttention *fia(const Tensor &t) {
  // Fused entries may reshape the kernel output back to [T,H,D].
  for (auto node = t;;) {
    if (auto *op = dynamic_cast<const op::FusedInferAttention *>(node.get_op()))
      return op;
    if (node.get_inputs().empty())
      return nullptr;
    node = node.get_inputs()[0];
  }
}

int main() {
  try {
    for (bool paged : {false, true}) {
      cache::SlotMapConfig map_cfg;
      // Three slots make the paged pool rows (dummy + 6 data + 3 padding
      // pages) not divisible by the slot count.
      map_cfg.slots                = 3;
      map_cfg.capacity             = 256;
      map_cfg.block_size           = 128;
      map_cfg.paged                = paged;
      map_cfg.num_pages            = 6;
      auto                     map = cache::make_slot_map(map_cfg);
      cache::AttentionMetadata meta(*map, cache::AttentionVariant::Full);
      const auto               rows = map->storage_rows();
      auto                     q = op::create_input({9, 4, 128}, DType::BF16);
      auto       k   = op::create_input({rows, 2, 128}, DType::BF16);
      const auto in  = meta.inputs(9);
      const auto dec = meta.inputs(3);
      auto       qd  = op::create_input({3, 4, 128}, DType::BF16);
      const auto decode_type =
        paged ? AttentionType::PagedBSND : AttentionType::BSND;
      for (auto type :
        {AttentionType::Eager, AttentionType::DenseFIA, AttentionType::TND})
        require(full_attention(type, in, q, k, k, .1f).shape() == q.shape(),
          "full attention output is token-major");
      require(
        full_attention(decode_type, dec, qd, k, k, .1f).shape() == qd.shape(),
        "decode output is token-major");
      auto *tnd = fia(full_attention(AttentionType::TND, in, q, k, k, .1f));
      require(tnd
          && tnd->options.sparse_mode
            == int64_t(AttentionMaskMode::RightAlignedCausal)
          && tnd->options.input_layout
            == op::FusedInferAttention::InputLayout::TND
          && tnd->options.block_size == 128,
        "TND binds its layout, causal mode and block size");
      auto *bsnd = fia(full_attention(decode_type, dec, qd, k, k, .1f));
      require(bsnd && bsnd->options.block_size == (paged ? 128 : 0),
        "only paged BSND addresses blocks");
      require(
        !dec.decode_mask, "single-query decode needs no per-query chunk mask");
      require(meta.inputs(9).decode_mask->shape()
          == Shape{3, 1, 3, map->slot_stride()},
        "multi-query slot chunks declare a causal mask");
      auto missing = in;
      missing.query_lengths.reset();
      rejects([&] { flash_attention_full_tnd(missing, q, k, k, .1f); });
      missing = in;
      missing.decode_mask.reset();
      rejects([&] { flash_attention_full_bsnd(missing, q, k, k, .1f, paged); });
      missing = in;
      missing.fia_mask.reset();
      rejects([&] { flash_attention_full_dense(missing, q, k, k, .1f); });
      rejects([&] {
        full_attention(AttentionType::TND, in,
          op::create_input({4, 8, 128}, DType::FP32), k, k, .1f);
      });
      rejects([&] {
        flash_attention_full_dense(
          in, q, op::create_input({1, 2, 128}, DType::BF16), k, .1f);
      });

      cache::AttentionMetadata mla_meta(*map, cache::AttentionVariant::MLA);
      auto                     mq = op::create_input({3, 16, 512}, DType::BF16);
      auto                     qr = op::create_input({3, 16, 64}, DType::BF16);
      auto  latent = op::create_input({rows, 1, 512}, DType::BF16);
      auto  kr     = op::create_input({rows, 1, 64}, DType::BF16);
      auto  mi     = mla_meta.inputs(3);
      auto *mla = fia(mla_attention(decode_type, mi, mq, qr, latent, kr, .1f));
      require(mla && mla->options.decode_mla == !paged,
        "only dense MLA decode uses the uniform-scan workaround");
      rejects([&] {
        mla_attention(decode_type, mi, mq,
          op::create_input({3, 15, 64}, DType::BF16), latent, kr, .1f);
      });
      rejects([&] {
        mla_attention(AttentionType::Eager, mi, mq, qr, latent, kr, .1f);
      });
      if (!paged) {
        mi.decode_mask.reset();
        rejects([&] {
          mla_attention(AttentionType::BSND, mi, mq, qr, latent, kr, .1f);
        });
      }
    }
    std::cout << "PASS attention schemes: shapes, required inputs, dtype, MLA "
                 "RoPE, mask mode, dense/paged storage\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
