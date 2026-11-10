#pragma once

/// Lightweight cache vocabulary shared by config parsing, runners and models.
/// Kept free of forch and hashing headers so engine/config.hpp stays cheap.

#include <cstdint>
#include <span>
#include <vector>

namespace cache {

enum class EvictionPolicy { LRU, LFU, FIFO };

// Compact: real tokens by slot, then padding. SlotOrdered: B*k rows, slot sid
// owns rows [sid*k, (sid+1)*k) and unused rows are padding.
enum class PackType { Compact, SlotOrdered };

// Parallel attention schemes. Each one is an operator contract (layout, mask,
// lengths and storage addressing are coupled inside the kernel), so they are
// not decomposed further. DenseFIA is kept as an ablation reference.
enum class AttentionType {
  Eager,     // Explicit matmul/softmax with a full additive mask.
  DenseFIA,  // Fused attention over one flattened sequence with a full mask.
  TND,       // Compact queries, cumulative query lengths and a block table.
  BSND,      // Slot-ordered queries and contiguous per-slot KV storage.
  PagedBSND, // Slot-ordered queries and page-addressed KV storage.
};

// One borrowed token span per slot, excluding padding. Empty means no advance;
// callers keep the underlying token buffers alive until commit returns.
using SlotTokens = std::vector<std::span<const int64_t>>;

// Capacity facts reported to the frontend and scheduler statistics.
struct KVCacheInfo {
  int64_t num_blocks = 0;
  int64_t block_size = 1;
  double  usage      = 0; // Fraction of blocks held by live requests.
};

} // namespace cache
