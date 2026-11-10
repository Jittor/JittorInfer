#pragma once

#include "fifo.hpp"
#include "lfu.hpp"
#include "lru.hpp"
#include <memory>

namespace cache {
inline std::unique_ptr<PageManager> make_page_manager(
  int64_t blocks, EvictionPolicy policy) {
  switch (policy) {
  case EvictionPolicy::LRU:
    return std::make_unique<LRUPageManager>(blocks);
  case EvictionPolicy::LFU:
    return std::make_unique<LFUPageManager>(blocks);
  case EvictionPolicy::FIFO:
    return std::make_unique<FIFOPageManager>(blocks);
  }
  throw std::invalid_argument("unknown page eviction policy");
}
} // namespace cache
