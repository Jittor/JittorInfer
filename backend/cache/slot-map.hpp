#pragma once

/// SlotMap: logical token positions -> physical pool rows, shared by every
/// pool of a runner (all layers; K and V; MLA latent and RoPE). It owns slot
/// cursors, page allocation, prefix hashes and block tables, so pools hold
/// data only and can never disagree about addressing.
///
/// The runner serializes attach_prefix -> can_fit -> reserve -> execute ->
/// commit. can_fit never allocates or pins pages. reserve allocates this
/// step's pages before write rows, masks and tables are generated. commit runs
/// only after the graph has written the step's KV, because publishing a block
/// makes it visible to later prefix lookups.

#include "page-manager/page-manager.hpp"
#include "types.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace cache {

struct SlotMapConfig {
  int64_t slots    = 0;
  int64_t capacity = 0; // Maximum logical tokens per request.
  // Zero selects unaligned dense rows with one shared tail dump row (eager
  // only); nonzero aligns slots for block tables and BSND views.
  int64_t block_size = 0;
  bool    paged      = false;
  // Paged only. Physical data pages, independent of slot count; zero chooses
  // the minimum that fits one full context.
  int64_t num_pages = 0;
  // Paged only. Rows reserved for padding writes; zero chooses slots*block.
  int64_t        padding_tokens  = 0;
  EvictionPolicy eviction_policy = EvictionPolicy::LRU;
  bool           prefix_caching  = true;
};

inline int64_t ceil_div(int64_t value, int64_t divisor) {
  return value ? 1 + (value - 1) / divisor : 0;
}

class SlotMap {
public:
  SlotMap(int64_t slots, int64_t capacity, int64_t block_size)
      : slots_(slots), capacity_(capacity), block_(block_size),
        positions_(slots > 0 ? slots : 0, 0) {
    if (slots <= 0 || capacity <= 0 || block_size < 0)
      throw std::invalid_argument("SlotMap: invalid slot geometry");
  }
  virtual ~SlotMap()                  = default;
  SlotMap(const SlotMap &)            = delete;
  SlotMap &operator=(const SlotMap &) = delete;

  int64_t slots() const {
    return slots_;
  }
  int64_t capacity() const {
    return capacity_;
  }
  int64_t block_size() const {
    return block_;
  }
  /// Committed cursor of each slot, excluding the step being prepared.
  const std::vector<int64_t> &positions() const {
    return positions_;
  }
  /// Logical per-slot extent of BSND views, masks and block tables: capacity
  /// rounded up past one full guard block.
  int64_t slot_stride() const {
    return block_ ? (capacity_ / block_ + 1) * block_ : capacity_;
  }
  int64_t table_columns() const {
    if (!block_)
      throw std::logic_error("SlotMap: unaligned storage has no block table");
    return slot_stride() / block_;
  }

  virtual bool paged() const = 0;
  virtual bool prefix_caching() const {
    return false;
  }
  /// Physical rows of every pool sharing this map, including dummy, guard
  /// and padding storage.
  virtual int64_t storage_rows() const                               = 0;
  virtual int64_t physical_row(int64_t slot, int64_t position) const = 0;
  /// Scatter destination of the index-th padding row of one step.
  virtual int64_t padding_row(int64_t index) const = 0;
  virtual int64_t padding_capacity() const         = 0;
  /// Attach completed prefix blocks to a free slot and return its cursor.
  /// The last prompt token always stays outside shared storage for logits.
  virtual int64_t attach_prefix(
    int64_t slot, std::span<const int64_t> prompt) = 0;
  /// Whether this exact per-slot advance fits. Never mutates.
  virtual bool can_fit(const std::vector<int64_t> &counts) const = 0;
  /// Allocate pages for this step; must follow a successful can_fit.
  virtual void reserve(const std::vector<int64_t> &counts) = 0;
  virtual void commit(const SlotTokens &tokens)            = 0;
  /// Release the slot's references; shared cached data survives until
  /// eviction.
  virtual void reset_slot(int64_t slot) = 0;
  /// Drop all published prefixes; false while any request holds blocks.
  virtual bool reset_prefix_cache() = 0;
  /// Row-major INT32 [sequences, table_columns()]; rows beyond the slot
  /// count read dummy block zero.
  virtual std::vector<int32_t> block_table(int64_t sequences) const = 0;
  virtual KVCacheInfo          info() const                         = 0;

  /// Validate one step plan and return its real token count.
  int64_t check_plan(
    const std::vector<int64_t> &counts, int64_t tq, PackType pack) const {
    check_counts(counts);
    if (tq <= 0 || (pack == PackType::SlotOrdered && tq % slots_))
      throw std::invalid_argument("SlotMap: invalid packing or token budget");
    int64_t total = 0;
    for (size_t sid = 0; sid < counts.size(); ++sid) {
      if (counts[sid] > capacity_ - positions_[sid])
        throw std::out_of_range("SlotMap: slot capacity exceeded");
      if (counts[sid] > tq - total
        || (pack == PackType::SlotOrdered && counts[sid] > tq / slots_))
        throw std::out_of_range("SlotMap: token budget exceeded");
      total += counts[sid];
    }
    return total;
  }

  /// Scatter rows for one step. Compact packs real tokens slot by slot;
  /// SlotOrdered fixes slot sid at rows [sid*k, (sid+1)*k). Padding rows
  /// never alias data rows.
  std::vector<int64_t> write_rows(
    const std::vector<int64_t> &counts, int64_t tq, PackType pack) const {
    const auto total = check_plan(counts, tq, pack);
    if (tq - total > padding_capacity())
      throw std::out_of_range("SlotMap: padding budget exceeded");
    std::vector<int64_t> rows;
    rows.reserve(tq);
    int64_t padding = 0;
    for (int64_t sid = 0; sid < slots_; ++sid) {
      const auto width =
        pack == PackType::SlotOrdered ? tq / slots_ : counts[sid];
      for (int64_t j = 0; j < width; ++j)
        rows.push_back(j < counts[sid] ? physical_row(sid, positions_[sid] + j)
                                       : padding_row(padding++));
    }
    while (static_cast<int64_t>(rows.size()) < tq)
      rows.push_back(padding_row(padding++));
    return rows;
  }

  /// Calls fn(first_row, count) for each physically contiguous run holding
  /// logical positions [0, length) of a slot.
  template <class F>
  void for_each_run(int64_t slot, int64_t length, F &&fn) const {
    for (int64_t pos = 0; pos < length;) {
      const auto run =
        block_ ? std::min(block_ - pos % block_, length - pos) : length - pos;
      fn(physical_row(slot, pos), run);
      pos += run;
    }
  }

protected:
  void check_counts(const std::vector<int64_t> &counts) const {
    if (counts.size() != static_cast<size_t>(slots_))
      throw std::invalid_argument("SlotMap: invalid slot count");
    for (auto count : counts)
      if (count < 0)
        throw std::invalid_argument("SlotMap: negative token count");
  }
  bool within_capacity(const std::vector<int64_t> &counts) const {
    check_counts(counts);
    for (size_t sid = 0; sid < counts.size(); ++sid)
      if (counts[sid] > capacity_ - positions_[sid])
        return false;
    return true;
  }
  void check_free(int64_t slot, std::span<const int64_t> prompt) const {
    if (slot < 0 || slot >= slots_ || positions_[slot] != 0)
      throw std::logic_error("SlotMap: prefix attach requires a free slot");
    if (prompt.size() > static_cast<size_t>(capacity_))
      throw std::out_of_range("SlotMap: prompt exceeds context");
  }
  void check_slot(int64_t slot) const {
    if (slot < 0 || slot >= slots_)
      throw std::out_of_range("SlotMap: invalid slot");
  }

  int64_t              slots_, capacity_, block_;
  std::vector<int64_t> positions_;
};

/// Contiguous per-slot segments. Unaligned storage sends all padding to one
/// tail dump row; aligned storage keeps a guard region after each slot and
/// cycles padding through those guards.
class DenseSlotMap final : public SlotMap {
public:
  DenseSlotMap(int64_t slots, int64_t capacity, int64_t block_size)
      : SlotMap(slots, capacity, block_size) {}

  bool paged() const override {
    return false;
  }
  int64_t storage_rows() const override {
    return block_ ? slots_ * slot_stride() : slots_ * capacity_ + 1;
  }
  int64_t physical_row(int64_t slot, int64_t position) const override {
    return slot * slot_stride() + position;
  }
  int64_t padding_row(int64_t index) const override {
    if (!block_)
      return slots_ * capacity_;
    const auto guard = slot_stride() - capacity_;
    return ((index / guard) % slots_) * slot_stride() + capacity_
      + index % guard;
  }
  int64_t padding_capacity() const override {
    return std::numeric_limits<int64_t>::max();
  }
  int64_t attach_prefix(
    int64_t slot, std::span<const int64_t> prompt) override {
    check_free(slot, prompt);
    return 0;
  }
  bool can_fit(const std::vector<int64_t> &counts) const override {
    return within_capacity(counts);
  }
  void reserve(const std::vector<int64_t> &counts) override {
    if (!within_capacity(counts))
      throw std::out_of_range("DenseSlotMap: slot capacity exceeded");
  }
  void commit(const SlotTokens &tokens) override {
    std::vector<int64_t> counts;
    for (auto row : tokens)
      counts.push_back(static_cast<int64_t>(row.size()));
    reserve(counts);
    for (int64_t sid = 0; sid < slots_; ++sid)
      positions_[sid] += counts[sid];
  }
  void reset_slot(int64_t slot) override {
    check_slot(slot);
    positions_[slot] = 0;
  }
  bool reset_prefix_cache() override {
    return true;
  }
  std::vector<int32_t> block_table(int64_t sequences) const override {
    const auto columns = table_columns();
    if (sequences < slots_)
      throw std::invalid_argument("DenseSlotMap: invalid block table rows");
    std::vector<int32_t> table(sequences * columns, 0);
    for (int64_t i = 0; i < slots_ * columns; ++i)
      table[i] = static_cast<int32_t>(i);
    return table;
  }
  KVCacheInfo info() const override {
    int64_t used = 0;
    for (auto position : positions_)
      used += position;
    const auto block = block_ ? block_ : 1;
    return {slots_ * ceil_div(capacity_, block), block,
      double(used) / double(slots_ * capacity_)};
  }
};

/// Page-addressed rows: [dummy page 0 | data pages | padding pages]. Pages
/// come from one PageManager per runner, so every layer and pool shares one
/// allocation, one prefix lookup and one block table.
class PagedSlotMap final : public SlotMap {
public:
  explicit PagedSlotMap(
    const SlotMapConfig &config, std::unique_ptr<PageManager> manager = nullptr)
      : SlotMap(config.slots, config.capacity, config.block_size),
        cfg_(normalize(config)),
        manager_(manager
            ? std::move(manager)
            : make_page_manager(cfg_.num_pages, cfg_.eviction_policy)),
        blocks_(cfg_.slots), prefix_(cfg_.slots) {
    if (manager_->num_blocks() != cfg_.num_pages
      || manager_->free_blocks() != cfg_.num_pages)
      throw std::invalid_argument(
        "PagedSlotMap: manager must have the configured free capacity");
    for (BlockId id = 1; id <= manager_->num_blocks(); ++id)
      if (manager_->block(id).hash)
        throw std::invalid_argument(
          "PagedSlotMap: manager contains stale cache hashes");
  }

  bool paged() const override {
    return true;
  }
  bool prefix_caching() const override {
    return cfg_.prefix_caching;
  }
  const PageManager &pages() const {
    return *manager_;
  }
  int64_t storage_rows() const override {
    return (cfg_.num_pages + 1 + ceil_div(cfg_.padding_tokens, block_))
      * block_;
  }
  int64_t physical_row(int64_t slot, int64_t position) const override {
    return int64_t(blocks_.at(slot).at(position / block_)) * block_
      + position % block_;
  }
  int64_t padding_row(int64_t index) const override {
    return (cfg_.num_pages + 1) * block_ + index;
  }
  int64_t padding_capacity() const override {
    return cfg_.padding_tokens;
  }

  int64_t attach_prefix(
    int64_t slot, std::span<const int64_t> prompt) override {
    check_free(slot, prompt);
    if (!blocks_[slot].empty())
      throw std::logic_error("PagedSlotMap: free slot still holds pages");
    prefix_[slot] = {};
    if (!cfg_.prefix_caching)
      return 0;
    const auto limit = prompt.empty() ? 0 : (prompt.size() - 1) / block_;
    std::vector<BlockId> found;
    BlockHash            parent{};
    // Reserve before acquiring references so recording a hit cannot throw.
    found.reserve(limit);
    try {
      for (size_t i = 0; i < limit; ++i) {
        const auto hash =
          hash_block(parent, prompt.subspan(i * block_, block_), block_);
        auto block = manager_->acquire(hash);
        if (!block)
          break;
        found.push_back(*block);
        parent = hash;
      }
    } catch (...) {
      for (auto it = found.rbegin(); it != found.rend(); ++it)
        manager_->release(*it);
      throw;
    }
    blocks_[slot]    = std::move(found);
    prefix_[slot]    = {true, parent, {}};
    positions_[slot] = static_cast<int64_t>(blocks_[slot].size()) * block_;
    return positions_[slot];
  }

  bool can_fit(const std::vector<int64_t> &counts) const override {
    return within_capacity(counts)
      && needed_pages(counts) <= manager_->free_blocks();
  }

  void reserve(const std::vector<int64_t> &counts) override {
    if (!within_capacity(counts))
      throw std::out_of_range("PagedSlotMap: slot capacity exceeded");
    const auto needed = needed_pages(counts);
    if (needed > manager_->free_blocks())
      throw std::logic_error(
        "PagedSlotMap: execute without a successful capacity probe");
    // Reserve host storage before consuming any pages so failure is atomic.
    for (int64_t sid = 0; sid < slots_; ++sid)
      blocks_[sid].reserve(std::max<int64_t>(
        blocks_[sid].size(), ceil_div(positions_[sid] + counts[sid], block_)));
    auto   pages = manager_->allocate(needed);
    size_t next  = 0;
    for (int64_t sid = 0; sid < slots_; ++sid)
      while (static_cast<int64_t>(blocks_[sid].size())
        < ceil_div(positions_[sid] + counts[sid], block_))
        blocks_[sid].push_back((*pages)[next++]);
  }

  void commit(const SlotTokens &tokens) override {
    std::vector<int64_t> counts;
    for (auto row : tokens)
      counts.push_back(static_cast<int64_t>(row.size()));
    reserve(counts);
    for (int64_t sid = 0; sid < slots_; ++sid) {
      auto &state = prefix_[sid];
      for (auto token : tokens[sid]) {
        if (state.enabled) {
          state.pending.push_back(token);
          if (state.pending.size() == static_cast<size_t>(block_)) {
            const auto hash = hash_block(state.parent, state.pending, block_);
            manager_->publish(blocks_[sid].at(positions_[sid] / block_), hash);
            state.parent = hash;
            state.pending.clear();
          }
        }
        ++positions_[sid];
      }
    }
  }

  void reset_slot(int64_t slot) override {
    check_slot(slot);
    auto &blocks = blocks_[slot];
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it)
      manager_->release(*it);
    blocks.clear();
    positions_[slot] = 0;
    prefix_[slot]    = {};
  }

  bool reset_prefix_cache() override {
    return manager_->reset_cache();
  }

  std::vector<int32_t> block_table(int64_t sequences) const override {
    const auto columns = table_columns();
    if (sequences < slots_)
      throw std::invalid_argument("PagedSlotMap: invalid block table rows");
    // Unallocated entries read dummy page zero; padding writes are separate.
    std::vector<int32_t> table(sequences * columns, 0);
    for (int64_t sid = 0; sid < slots_; ++sid)
      std::copy(blocks_[sid].begin(), blocks_[sid].end(),
        table.begin() + sid * columns);
    return table;
  }

  KVCacheInfo info() const override {
    return {cfg_.num_pages, block_,
      double(manager_->used_blocks()) / double(cfg_.num_pages)};
  }

private:
  int64_t needed_pages(const std::vector<int64_t> &counts) const {
    int64_t needed = 0;
    for (int64_t sid = 0; sid < slots_; ++sid)
      needed += std::max<int64_t>(0,
        ceil_div(positions_[sid] + counts[sid], block_)
          - static_cast<int64_t>(blocks_[sid].size()));
    return needed;
  }

  static SlotMapConfig normalize(SlotMapConfig cfg) {
    if (cfg.slots <= 0 || cfg.capacity <= 0 || cfg.block_size <= 0)
      throw std::invalid_argument("PagedSlotMap: dimensions must be positive");
    if (cfg.num_pages < 0 || cfg.padding_tokens < 0)
      throw std::invalid_argument(
        "PagedSlotMap: invalid page or padding budget");
    const auto limit    = std::numeric_limits<int32_t>::max();
    const auto per_slot = ceil_div(cfg.capacity, cfg.block_size);
    if (per_slot >= limit || cfg.slots > limit / (per_slot + 1))
      throw std::invalid_argument("PagedSlotMap: logical table is too large");
    // A single maximal request must make progress, independently of slots.
    if (!cfg.num_pages)
      cfg.num_pages = per_slot;
    if (cfg.num_pages < per_slot)
      throw std::invalid_argument(
        "PagedSlotMap: pool must fit one full context");
    if (!cfg.padding_tokens) {
      if (cfg.block_size > std::numeric_limits<int64_t>::max() / cfg.slots)
        throw std::invalid_argument("PagedSlotMap: padding budget overflows");
      cfg.padding_tokens = cfg.slots * cfg.block_size;
    }
    const auto padding_pages = ceil_div(cfg.padding_tokens, cfg.block_size);
    if (cfg.num_pages >= limit || padding_pages >= limit - cfg.num_pages
      || cfg.block_size > std::numeric_limits<int64_t>::max()
          / (cfg.num_pages + 1 + padding_pages))
      throw std::invalid_argument("PagedSlotMap: physical pool is too large");
    return cfg;
  }

  struct PrefixState {
    bool                 enabled = false;
    BlockHash            parent{};
    std::vector<int64_t> pending;
  };
  SlotMapConfig                     cfg_;
  std::unique_ptr<PageManager>      manager_;
  std::vector<std::vector<BlockId>> blocks_;
  std::vector<PrefixState>          prefix_;
};

inline std::unique_ptr<SlotMap> make_slot_map(const SlotMapConfig &cfg) {
  if (cfg.paged)
    return std::make_unique<PagedSlotMap>(cfg);
  return std::make_unique<DenseSlotMap>(
    cfg.slots, cfg.capacity, cfg.block_size);
}

} // namespace cache
