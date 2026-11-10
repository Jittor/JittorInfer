// Host-only checks for prefix reuse, page accounting and cache reset on the
// runner-wide PagedSlotMap.
#include "../cache/slot-map.hpp"
#include "../engine/config.hpp"
#include <iostream>
#include <stdexcept>

static void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

static cache::SlotMapConfig config(int64_t pages, bool prefix = true) {
  cache::SlotMapConfig cfg;
  cfg.slots          = 2;
  cfg.capacity       = 8;
  cfg.block_size     = 2;
  cfg.paged          = true;
  cfg.num_pages      = pages;
  cfg.prefix_caching = prefix;
  return cfg;
}

int main() {
  try {
    const std::vector<int64_t> pair{1, 2};
    const auto                 hash = cache::hash_block({}, pair, 2);
    require(hash
        == cache::BlockHash{0x441e3c5df316b017ULL, 0xff069f84b8fbe101ULL,
          0x381508d1e8fa6169ULL, 0xa38f68ae61be2fccULL},
      "SHA-256 canonical encoding");
    require(hash != cache::hash_block({1, 0, 0, 0}, pair, 2),
      "parent changes identity");

    cache::PagedSlotMap  map(config(8));
    std::vector<int64_t> prompt{1, 2, 3, 4, 5, 6};
    require(map.attach_prefix(0, prompt) == 0, "initial miss");
    // Reserving pages for a graph is not a successful KV write.
    map.reserve({2, 0});
    map.reset_slot(0);
    require(map.attach_prefix(0, prompt) == 0, "unexecuted graph cannot hit");
    map.commit(cache::SlotTokens{std::span(prompt).first(3), {}});
    map.commit(cache::SlotTokens{std::span(prompt).subspan(3), {}});
    map.reset_slot(0);
    require(map.attach_prefix(1, prompt) == 4,
      "aligned prompt keeps last full block for logits");
    require(map.positions()[1] == 4, "cursor starts at prefix boundary");
    map.commit(cache::SlotTokens{{}, std::span(prompt).subspan(4)});
    map.reset_slot(1);
    std::vector<int64_t> extended{1, 2, 3, 4, 5, 6, 7};
    require(map.attach_prefix(0, extended) == 6,
      "one last token remains when unaligned");
    map.reset_slot(0);
    auto changed = prompt;
    changed[2]   = 9;
    require(map.attach_prefix(0, changed) == 2,
      "first mismatching block ends lookup");
    map.reset_slot(0);
    changed[0] = 9;
    require(map.attach_prefix(0, changed) == 0,
      "different parent cannot reuse suffix");
    map.reset_slot(0);
    std::vector<int64_t> one{1};
    require(map.attach_prefix(0, one) == 0, "single token always executes");
    map.commit(cache::SlotTokens{one, {}});
    map.reset_slot(0);
    require(map.attach_prefix(0, prompt) == 4,
      "partial tail does not corrupt existing cache");
    require(map.block_table(3)[0] == map.block_table(3)[0]
        && map.physical_row(0, 3) % 2 == 1,
      "reused pages address the attached prefix");
    map.reset_slot(0);

    cache::DenseSlotMap dense(2, 8, 2);
    require(dense.attach_prefix(0, prompt) == 0,
      "dense storage never skips prefix computation");

    const finer::engine::EngineConfig disabled_config(nlohmann::json{
      {"kv_cache", {{"paged", true}, {"prefix_caching", false}}}});
    require(disabled_config.kv_cache.paged
        && !disabled_config.kv_cache.prefix_caching,
      "prefix toggle is independent of paging");

    auto                manager  = std::make_unique<cache::LRUPageManager>(8);
    auto               *observer = manager.get();
    cache::PagedSlotMap uncached(config(8, false), std::move(manager));
    for (int repeat = 0; repeat < 3; ++repeat) {
      require(uncached.attach_prefix(0, prompt) == 0,
        "disabled mode never reuses prefixes");
      uncached.commit(cache::SlotTokens{prompt, {}});
      require(uncached.positions()[0] == int64_t(prompt.size()),
        "disabled prefix caching still advances paged storage");
      require(observer->free_blocks() == 5, "disabled mode allocates pages");
      for (cache::BlockId id = 1; id <= 8; ++id)
        require(
          !observer->block(id).hash, "disabled mode never publishes hashes");
      uncached.reset_slot(0);
      require(observer->free_blocks() == 8, "disabled mode releases pages");
    }

    auto  shared_manager  = std::make_unique<cache::LRUPageManager>(4);
    auto *shared_observer = shared_manager.get();
    cache::PagedSlotMap shared(config(4), std::move(shared_manager));
    shared.attach_prefix(0, prompt);
    shared.commit(cache::SlotTokens{prompt, {}});
    shared.reset_slot(0);
    require(shared.attach_prefix(0, extended) == 6
        && shared.attach_prefix(1, extended) == 6,
      "live requests acquire identical completed pages");
    require(shared_observer->free_blocks() == 1,
      "shared references count physical pages once");
    require(shared.info().usage == 0.75 && shared.info().num_blocks == 4
        && shared.info().block_size == 2,
      "usage reports referenced pages");
    require(
      !shared.can_fit({1, 1}), "two private tails exceed single free page");
    require(shared.can_fit({1, 0}), "one private tail fits");
    require(!shared.reset_prefix_cache(),
      "prefix reset is refused while requests hold pages");
    shared.reset_slot(1);
    require(shared_observer->free_blocks() == 1,
      "dropping one shared owner frees no pages");
    shared.reset_slot(0);
    require(shared_observer->free_blocks() == 4,
      "last shared owner makes pages evictable");
    require(shared.reset_prefix_cache(), "idle prefix reset succeeds");
    require(shared.attach_prefix(0, extended) == 0,
      "reset prefix cache no longer hits");
    for (cache::BlockId id = 1; id <= 4; ++id)
      require(!shared_observer->block(id).hash, "reset drops every hash");
    shared.reset_slot(0);

    bool rejected = false;
    try {
      cache::PagedSlotMap tiny(config(3));
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "pool smaller than one context is rejected");
    std::cout << "PASS prefix cache: SHA-256, chunk completion, last token, "
                 "shared pages, usage, reset\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
