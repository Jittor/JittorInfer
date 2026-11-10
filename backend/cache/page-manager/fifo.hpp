#pragma once

#include "base.hpp"

namespace cache {

// First publication order; hits and releases preserve the order.
// Keep pinned entries linked: release must not promote an older entry.
class FIFOPageManager final : public PageManager {
public:
  explicit FIFOPageManager(int64_t num_blocks)
      : PageManager(num_blocks), links_(static_cast<size_t>(num_blocks) + 1) {}

private:
  struct Link {
    BlockId prev = 0;
    BlockId next = 0;
  };

  BlockId select_victim() const noexcept override {
    for (auto id = head_; id; id = links_[id].next)
      if (!block(id).ref_count)
        return id;
    return 0;
  }
  void on_publish(BlockId id) noexcept override {
    append(id);
  }
  void on_evict(BlockId id) noexcept override {
    unlink(id);
  }
  void on_hit(BlockId id) noexcept override {
    // Hits do not change publication order.
  }

  void unlink(BlockId id) noexcept {
    const auto link = links_[id];
    if (link.prev)
      links_[link.prev].next = link.next;
    else
      head_ = link.next;
    if (link.next)
      links_[link.next].prev = link.prev;
    else
      tail_ = link.prev;
    links_[id] = {};
  }
  void append(BlockId id) noexcept {
    links_[id] = {tail_, 0};
    if (tail_)
      links_[tail_].next = id;
    else
      head_ = id;
    tail_ = id;
  }

  std::vector<Link> links_;
  BlockId           head_ = 0;
  BlockId           tail_ = 0;
};

} // namespace cache
