#pragma once

#include "base.hpp"

namespace cache {

// Successful reuse counts, reset on publication into newly computed storage.
// Equal counts use the physical ID, keeping independent K/V pools
// deterministic.
class LFUPageManager final : public PageManager {
public:
  explicit LFUPageManager(int64_t num_blocks)
      : PageManager(num_blocks), hits_(static_cast<size_t>(num_blocks) + 1, 0) {
  }

private:
  BlockId select_victim() const noexcept override {
    BlockId victim = 0;
    for (BlockId id = 1; id <= num_blocks(); ++id) {
      const auto &entry = block(id);
      if (entry.hash && !entry.ref_count
        && (!victim || hits_[id] < hits_[victim]))
        victim = id;
    }
    return victim;
  }
  void on_publish(BlockId id) noexcept override {
    hits_[id] = 0;
  }
  void on_evict(BlockId id) noexcept override {
    hits_[id] = 0;
  }
  void on_hit(BlockId id) noexcept override {
    if (hits_[id] != std::numeric_limits<uint64_t>::max())
      ++hits_[id];
  }

  std::vector<uint64_t> hits_;
};

} // namespace cache
