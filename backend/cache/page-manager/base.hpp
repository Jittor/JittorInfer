#pragma once

#include "../../utils/sha256.hpp"
#include "../types.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cache {

using BlockId = int32_t;

// Opaque 256-bit content digest supplied by the caller. It must cover the
// parent digest, block tokens and cache namespace; zero is a valid digest.
using BlockHash = std::array<uint64_t, 4>;

// Fixed little-endian encoding keeps identities independent of host byte order
// and distinguishes token boundaries. Each manager belongs to one model pool;
// callers must not publish multimodal embeddings using token IDs alone.
inline BlockHash hash_block(const BlockHash &parent,
  std::span<const int64_t> tokens, int64_t block_size) {
  std::vector<unsigned char> bytes;
  constexpr std::string_view domain = "finer-kv-prefix-sha256-v1";
  bytes.insert(bytes.end(), domain.begin(), domain.end());
  auto append = [&](uint64_t word) {
    for (int i = 0; i < 8; ++i)
      bytes.push_back(static_cast<unsigned char>(word >> (i * 8)));
  };
  for (auto word : parent)
    append(word);
  append(static_cast<uint64_t>(block_size));
  append(tokens.size());
  for (auto token : tokens)
    append(static_cast<uint64_t>(token));
  const auto digest = finer::utils::sha256(bytes);
  BlockHash  result{};
  for (size_t i = 0; i < digest.size(); ++i)
    result[i / 8] |= uint64_t(digest[i]) << ((i % 8) * 8);
  return result;
}

// Metadata lives for the pool lifetime. Callers inspect it through const
// references; PageManager updates ownership and hashes.
struct PagedBlock {
  std::optional<BlockHash> hash;
  BlockId                  id        = 0;
  uint32_t                 ref_count = 0;
};

// Host-only physical allocator. IDs start at one so cache storage can reserve
// page zero as a dummy. Prefix lookup owns no request or device-storage state.
// Operations are serialized by the runner, not internally synchronized.
class PageManager {
public:
  explicit PageManager(int64_t num_blocks) : free_count_(num_blocks) {
    if (num_blocks < 0 || num_blocks >= std::numeric_limits<BlockId>::max())
      throw std::invalid_argument("PageManager: invalid block count");
    blocks_.resize(static_cast<size_t>(num_blocks) + 1);
    unused_.reserve(static_cast<size_t>(num_blocks));
    for (BlockId id = 1; id <= num_blocks; ++id)
      blocks_[id].id = id;
    for (BlockId id = num_blocks; id > 0; --id)
      unused_.push_back(id);
  }
  virtual ~PageManager() = default;

  PageManager(const PageManager &)            = delete;
  PageManager &operator=(const PageManager &) = delete;

  int64_t num_blocks() const {
    return blocks_.size() - 1;
  }
  int64_t free_blocks() const {
    return free_count_;
  }
  int64_t used_blocks() const {
    return num_blocks() - free_count_;
  }
  const PagedBlock &block(BlockId id) const {
    return checked(id);
  }

  std::optional<BlockId> allocate() {
    if (!free_count_)
      return std::nullopt;
    BlockId id;
    if (!unused_.empty()) {
      id = unused_.back();
      unused_.pop_back();
    } else {
      id = select_victim();
    }
    --free_count_;
    auto &entry = blocks_[id];
    if (entry.hash) {
      on_evict(id);
      auto it = cached_.find(*entry.hash);
      it->second.erase(id);
      if (it->second.empty())
        cached_.erase(it);
      entry.hash.reset();
    }
    entry.ref_count = 1;
    return id;
  }

  // Capacity and result storage are checked before consuming any free blocks.
  std::optional<std::vector<BlockId>> allocate(int64_t count) {
    if (count < 0)
      throw std::invalid_argument("PageManager: negative allocation count");
    if (count > free_count_)
      return std::nullopt;
    std::vector<BlockId> result;
    result.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i)
      result.push_back(*allocate());
    return result;
  }

  // Publication is a completion barrier imposed by the caller: only a full,
  // successfully computed, immutable KV block may become visible to lookups.
  // Computing its digest earlier does not make its storage ready for reuse.
  void publish(BlockId id, const BlockHash &hash) {
    auto &entry = checked(id);
    if (!entry.ref_count)
      throw std::logic_error("PageManager: publish of unowned block");
    if (entry.hash) {
      if (*entry.hash != hash)
        throw std::logic_error("PageManager: cannot change a published hash");
      return;
    }
    // Different physical blocks can hold identical completed contents.
    // Keep all copies indexed until each is actually reused.
    auto [it, inserted] = cached_.try_emplace(hash);
    try {
      it->second.insert(id);
    } catch (...) {
      if (inserted)
        cached_.erase(it);
      throw;
    }
    entry.hash = hash;
    on_publish(id);
  }

  // Inspection alone does not pin storage. Serialized callers must acquire
  // before allocating anything that might overwrite the inspected block.
  std::optional<BlockId> find(const BlockHash &hash) const {
    auto it = cached_.find(hash);
    if (it == cached_.end())
      return std::nullopt;
    return *it->second.begin();
  }

  std::optional<BlockId> acquire(const BlockHash &hash) {
    auto id = find(hash);
    if (!id)
      return std::nullopt;
    auto &entry = blocks_[*id];
    if (!entry.ref_count) {
      --free_count_;
    } else if (entry.ref_count == std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("PageManager: reference count overflow");
    ++entry.ref_count;
    on_hit(*id);
    return id;
  }

  void release(BlockId id) {
    auto &entry = checked(id);
    if (!entry.ref_count)
      throw std::logic_error("PageManager: release of free block");
    if (--entry.ref_count == 0) {
      if (!entry.hash)
        unused_.push_back(id);
      ++free_count_;
    }
  }

  // Drop every published digest, as vLLM's reset_prefix_cache does. Refused
  // while any block is referenced: live requests would keep pages whose
  // lookups vanish, and partial resets make hit statistics meaningless.
  bool reset_cache() {
    if (used_blocks())
      return false;
    for (BlockId id = 1; id <= num_blocks(); ++id) {
      auto &entry = blocks_[id];
      if (!entry.hash)
        continue;
      on_evict(id);
      entry.hash.reset();
      unused_.push_back(id);
    }
    cached_.clear();
    return true;
  }

protected:
  // Policies track published blocks, including pinned ones. Only zero-ref
  // blocks may be selected. Release changes eligibility, not access history.
  // Hooks use preallocated metadata so batch operations cannot fail halfway.
  virtual BlockId select_victim() const noexcept  = 0;
  virtual void    on_publish(BlockId id) noexcept = 0;
  virtual void    on_evict(BlockId id) noexcept   = 0;
  virtual void    on_hit(BlockId id) noexcept     = 0;

private:
  const PagedBlock &checked(BlockId id) const {
    if (id <= 0 || static_cast<size_t>(id) >= blocks_.size())
      throw std::out_of_range("PageManager: invalid block ID");
    return blocks_[id];
  }
  PagedBlock &checked(BlockId id) {
    return const_cast<PagedBlock &>(
      static_cast<const PageManager &>(*this).checked(id));
  }

  struct HashDigest {
    size_t operator()(const BlockHash &hash) const noexcept {
      size_t seed = 0;
      for (auto word : hash)
        seed ^= std::hash<uint64_t>{}(word) + size_t(0x9e3779b9) + (seed << 6)
          + (seed >> 2);
      return seed;
    }
  };
  // Ordered IDs make duplicate-content selection deterministic across K/V.
  std::unordered_map<BlockHash, std::set<BlockId>, HashDigest> cached_;
  std::vector<PagedBlock>                                      blocks_;
  std::vector<BlockId>                                         unused_;
  int64_t                                                      free_count_;
};

} // namespace cache
