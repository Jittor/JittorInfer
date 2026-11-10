// Host-only checks for token packing, shared write rows, attention metadata
// and block tables. Expected values are explicit, not derived from the
// implementation.
#include "../cache/attention-metadata.hpp"
#include "../engine/model-runner.hpp"
#include <cmath>
#include <iostream>

namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
template <class F>
void rejects(F &&fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected, "expected invalid input to be rejected");
}
template <class T>
std::vector<T> values(const forch::Tensor &tensor) {
  auto *host = dynamic_cast<const forch::op::Const *>(tensor.get_op());
  require(host != nullptr, "expected host tensor");
  int64_t count = 1;
  for (auto d : tensor.shape())
    count *= d;
  const auto *data = reinterpret_cast<const T *>(host->buffer());
  return {data, data + count};
}
const forch::Tensor &named(
  const forch::TensorList &list, const std::optional<forch::Tensor> &input) {
  for (const auto &t : list)
    if (t.get_name() == input->get_name())
      return t;
  throw std::runtime_error("metadata input was not fed");
}
// Positions are committed through the public API, as runners do.
void advance(cache::SlotMap &map, std::vector<int64_t> counts) {
  std::vector<std::vector<int64_t>> tokens;
  cache::SlotTokens                 spans;
  for (auto count : counts)
    tokens.emplace_back(count, 7);
  for (auto &row : tokens)
    spans.push_back(row);
  map.reserve(counts);
  map.commit(spans);
}

void packing() {
  using namespace finer;
  std::vector<int64_t>          a{11, 12, 13}, b{21};
  std::vector<engine::SlotStep> slots{{10, a}, {0, {}}, {20, b}};
  auto batch = engine::pack_tokens({"packed", 8, 0, true}, slots, 3, false);
  require(batch.ids == std::vector<int64_t>({11, 12, 13, 21, 0, 0, 0, 0}),
    "packed token order");
  require(batch.output_rows == std::vector<int64_t>({2, -1, 3}),
    "packed last-token rows");
  slots = {{0, {}}, {8, b}, {9, b}};
  batch = engine::pack_tokens({"decode", 3, 1, true}, slots, 3, true);
  require(batch.ids == std::vector<int64_t>({0, 21, 21}), "slot token order");
  require(
    batch.output_rows == std::vector<int64_t>({-1, 1, 2}), "slot output rows");
  rejects([&] { engine::pack_tokens({"bad", 2, 0, true}, slots, 3, true); });
  rejects([&] { engine::pack_tokens({"bad", 1, 0, true}, slots, 3, false); });
  slots = {{10, a}, {0, {}}, {20, b}};
  batch = engine::pack_tokens({"slot_prefill", 24, 8, true}, slots, 3, true);
  require(batch.ids[0] == 11 && batch.ids[2] == 13 && batch.ids[16] == 21
      && batch.ids[8] == 0,
    "prefill retains fixed per-slot rows");
  require(batch.output_rows == std::vector<int64_t>({2, -1, 16}),
    "prefill output rows skip slot padding");
}

// Every real/padding destination of 3 slots x {compact, slot-ordered} x
// active subsets, for unaligned, guard-aligned and exactly aligned slots.
size_t write_rows() {
  size_t checked = 0;
  for (const auto capacity : {int64_t{5}, int64_t{127}, int64_t{128}})
    for (const auto block : {int64_t{0}, int64_t{128}}) {
      cache::DenseSlotMap map(3, capacity, block);
      advance(map, {2, 3, 4});
      const auto                 stride  = map.slot_stride();
      const std::vector<int64_t> padding = !block
        ? std::vector<int64_t>{3 * capacity, 3 * capacity, 3 * capacity}
        : capacity == 127
        ? std::vector<int64_t>{127, 255, 383}
        : std::vector<int64_t>{capacity, capacity + 1, capacity + 2};
      require(map.storage_rows() == (block ? 3 * stride : 3 * capacity + 1),
        "dense storage rows");
      for (int bits = 0; bits < 8; ++bits) {
        const std::vector<int64_t> counts{
          bits & 1, (bits >> 1) & 1, (bits >> 2) & 1};
        const std::vector<int64_t> positions{2, 3, 4};
        std::vector<int64_t>       compact, ordered(3);
        for (int64_t sid = 0; sid < 3; ++sid)
          if (counts[sid])
            compact.push_back(sid * stride + positions[sid]);
        int64_t next = 0;
        for (int64_t sid = 0; sid < 3; ++sid) {
          if (counts[sid]) {
            ordered[sid] = sid * stride + positions[sid];
          } else {
            ordered[sid] = padding[next++];
            compact.push_back(ordered[sid]);
          }
        }
        require(map.write_rows(counts, 3, cache::PackType::Compact) == compact,
          "compact destinations changed");
        require(
          map.write_rows(counts, 3, cache::PackType::SlotOrdered) == ordered,
          "slot-ordered destinations changed");
        checked += 2;
      }
      rejects(
        [&] { map.write_rows({2, 0, 0}, 3, cache::PackType::SlotOrdered); });
      rejects(
        [&] { map.write_rows({1, 0, 0}, 2, cache::PackType::SlotOrdered); });
      rejects(
        [&] { map.write_rows({-1, 0, 0}, 3, cache::PackType::SlotOrdered); });
      rejects([&] { map.write_rows({0, 0}, 3, cache::PackType::Compact); });
      map.reset_slot(0);
      map.reset_slot(1);
      map.reset_slot(2);
      advance(map, {capacity, capacity - 1, 0});
      require(map.write_rows({0, 1, 0}, 3, cache::PackType::SlotOrdered)
          == std::vector<int64_t>{padding[0], stride + capacity - 1,
            padding[1]},
        "near-full guard mapping changed");
      rejects(
        [&] { map.write_rows({1, 0, 0}, 3, cache::PackType::SlotOrdered); });
    }
  return checked;
}

void full_metadata() {
  // Two maps with different capacities and cursors stay independent.
  cache::DenseSlotMap x_map(3, 64, 128), y_map(3, 256, 128);
  advance(x_map, {10, 20, 30});
  advance(y_map, {100, 120, 180});
  cache::AttentionMetadata x(x_map, cache::AttentionVariant::Full, "x"),
    y(y_map, cache::AttentionVariant::Full, "y");
  const auto       &xd = x.inputs(3);
  const auto       &yd = y.inputs(3);
  forch::TensorList xi, yi;
  x.feed(
    {0, 1, 1}, 3, cache::PackType::SlotOrdered, cache::AttentionType::BSND, xi);
  y.feed(
    {0, 1, 1}, 3, cache::PackType::SlotOrdered, cache::AttentionType::BSND, yi);
  require(values<int64_t>(named(xi, xd.decode_kv_lengths))
      == std::vector<int64_t>({1, 21, 31}),
    "short cache lengths");
  require(values<int64_t>(named(yi, yd.decode_kv_lengths))
      == std::vector<int64_t>({1, 121, 181}),
    "long cache lengths");
  require(xd.decode_kv_lengths->get_name() != yd.decode_kv_lengths->get_name(),
    "independent metadata names");
  x_map.reset_slot(1);
  xi.clear();
  x.feed(
    {0, 1, 1}, 3, cache::PackType::SlotOrdered, cache::AttentionType::BSND, xi);
  require(values<int64_t>(named(xi, xd.decode_kv_lengths))
      == std::vector<int64_t>({1, 1, 31}),
    "reset uses map cursor");
  require(values<int64_t>(named(xi, xd.write_rows))
      == std::vector<int64_t>({64, 128, 256 + 30}),
    "write rows follow cursors and guards");

  const auto &xp = x.inputs(8);
  xi.clear();
  x.feed({3, 0, 2}, 8, cache::PackType::Compact, cache::AttentionType::TND, xi);
  require(values<int64_t>(named(xi, xp.query_lengths))
      == std::vector<int64_t>({3, 3, 5, 8}),
    "dummy query budget");
  require(values<int64_t>(named(xi, xp.kv_lengths))
      == std::vector<int64_t>({13, 0, 32, 3}),
    "independent prefill cursor");
  require(xp.block_table->shape() == forch::Shape{4, 1}
      && values<int32_t>(*xp.block_table) == std::vector<int32_t>({0, 1, 2, 0}),
    "dense TND table is a constant with a dummy row");

  const auto &chunk = x.inputs(24);
  xi.clear();
  x.feed({3, 0, 2}, 24, cache::PackType::SlotOrdered,
    cache::AttentionType::BSND, xi);
  require(values<int64_t>(named(xi, chunk.decode_kv_lengths))
      == std::vector<int64_t>({13, 1, 32}),
    "BSND chunk lengths exclude per-slot padding");
  require(chunk.decode_mask->shape() == forch::Shape{3, 1, 8, 128},
    "BSND chunk uses a per-slot causal mask");
  const auto cm = values<int8_t>(named(xi, chunk.decode_mask));
  require(cm[12] == 1 && cm[128 + 11] == 0 && cm[128 + 12] == 1,
    "BSND chunk masks future tokens inside the chunk");

  // Eager/DenseFIA masks cover physical rows of the shared pool.
  const auto &x5 = x.inputs(5);
  const auto &y7 = y.inputs(7);
  xi.clear();
  yi.clear();
  x.feed(
    {3, 0, 2}, 5, cache::PackType::Compact, cache::AttentionType::Eager, xi);
  y.feed(
    {3, 0, 2}, 7, cache::PackType::Compact, cache::AttentionType::Eager, yi);
  require(xi.size() == 2 && yi.size() == 2
      && named(xi, x5.eager_mask).shape()
        == forch::Shape{5, x_map.storage_rows()}
      && named(yi, y7.eager_mask).shape()
        == forch::Shape{7, y_map.storage_rows()},
    "eager feeds write rows plus one mask of graph budget x pool rows");
  forch::TensorList dense;
  x.feed({3, 0, 2}, 5, cache::PackType::Compact, cache::AttentionType::DenseFIA,
    dense);
  require(dense.size() == 2
      && named(dense, x5.fia_mask).dtype() == forch::DType::INT8,
    "DenseFIA feeds its INT8 mask rather than lengths");

  const auto fv = values<float>(
    x.mask({3, 0, 2}, 7, cache::PackType::Compact, forch::DType::FP32));
  const auto iv = values<int8_t>(
    x.mask({3, 0, 2}, 7, cache::PackType::Compact, forch::DType::INT8));
  const auto width = x_map.storage_rows();
  require(fv[10] == 0 && std::isinf(fv[11]) && fv[11] < 0
      && fv[2 * width + 12] == 0 && std::isinf(fv[2 * width + 13]),
    "causal prefix grows within the slot");
  require(fv[3 * width + 256 + 30] == 0 && std::isinf(fv[3 * width + 255])
      && std::isinf(fv[3 * width + 256 + 31]) && std::isinf(fv[64])
      && std::isinf(fv[128]),
    "slot isolation and guards stay masked");
  for (int64_t row = 5; row < 7; ++row)
    for (int64_t col = 0; col < width; ++col)
      require(col == 0 ? fv[row * width + col] == 0
                       : std::isinf(fv[row * width + col]),
        "padding rows attend only position zero");
  for (int64_t i = 0; i < 7 * width; ++i)
    require(iv[i] == (std::isinf(fv[i]) ? 1 : 0),
      "eager and FIA masks have identical visibility");
  rejects([&] {
    x.mask({3, 0, 2}, 4, cache::PackType::Compact, forch::DType::FP32);
  });
  rejects([&] {
    x.mask({65, 0, 0}, 65, cache::PackType::Compact, forch::DType::FP32);
  });
  rejects([&] {
    x.mask({1, 0, 0}, 1, cache::PackType::Compact, forch::DType::BF16);
  });
  require(&xp == &x.inputs(8), "stable memoized tensor group");
  rejects([&] { x.inputs(0); });
  rejects([&] {
    forch::TensorList in;
    x.feed(
      {8, 0, 1}, 8, cache::PackType::Compact, cache::AttentionType::TND, in);
  });
  rejects([&] {
    forch::TensorList in;
    x.feed({1, 1, 1}, 8, cache::PackType::SlotOrdered,
      cache::AttentionType::BSND, in);
  });
  rejects([&] {
    forch::TensorList in;
    x.feed({0, 1, 1}, 3, cache::PackType::SlotOrdered,
      cache::AttentionType::PagedBSND, in);
  });
  cache::DenseSlotMap      eager_map(3, 64, 0);
  cache::AttentionMetadata eager(eager_map, cache::AttentionVariant::Full);
  eager.inputs(3);
  rejects([&] {
    forch::TensorList in;
    eager.feed({1, 1, 1}, 3, cache::PackType::SlotOrdered,
      cache::AttentionType::BSND, in);
  });
}

void paged_metadata() {
  cache::SlotMapConfig cfg;
  cfg.slots      = 2;
  cfg.capacity   = 256;
  cfg.block_size = 128;
  cfg.paged      = true;
  cfg.num_pages  = 4;
  cache::PagedSlotMap      map(cfg);
  cache::AttentionMetadata full(map, cache::AttentionVariant::Full, "full");
  cache::AttentionMetadata mla(map, cache::AttentionVariant::MLA, "mla");
  const auto              &fp = full.inputs(8);
  const auto              &md = mla.inputs(2);
  require(!dynamic_cast<const forch::op::Const *>(fp.block_table->get_op()),
    "paged tables are runtime inputs");
  // Slot 1 takes two pages, slot 0 one: allocation order is deterministic.
  map.reserve({0, 129});
  map.reserve({1, 129});
  forch::TensorList in;
  full.feed({1, 0}, 8, cache::PackType::Compact, cache::AttentionType::TND, in);
  const auto table = values<int32_t>(named(in, fp.block_table));
  require(table.size() == 3 * 3 && table[0] == 3 && table[3] == 1
      && table[4] == 2 && table[6] == 0 && table[8] == 0,
    "paged table maps slots and dummy rows to physical pages");
  const auto rows = values<int64_t>(named(in, fp.write_rows));
  require(rows[0] == 3 * 128 && rows[1] == 5 * 128 && rows[7] == 5 * 128 + 6,
    "paged writes use page rows and a separate padding region");
  require(map.storage_rows() == (1 + 4 + 2) * 128,
    "paged pool = dummy + data + padding pages");
  map.commit({std::span<const int64_t>{}, std::span<const int64_t>{}});
  advance(map, {1, 129});
  in.clear();
  mla.feed({0, 1}, 2, cache::PackType::SlotOrdered,
    cache::AttentionType::PagedBSND, in);
  require(in.size() == 3
      && values<int64_t>(named(in, md.decode_kv_lengths))
        == std::vector<int64_t>({1, 130}),
    "paged MLA decode feeds rows, independent lengths and the slot table");
  require(md.decode_mask->shape() == forch::Shape{2, 1, 384},
    "MLA mask spans one slot stride");
}

void dense_mla_workaround() {
  cache::DenseSlotMap      map(2, 128, 128);
  cache::AttentionMetadata mla(map, cache::AttentionVariant::MLA);
  advance(map, {5, 40});
  const auto       &in = mla.inputs(2);
  forch::TensorList out;
  mla.feed(
    {1, 0}, 2, cache::PackType::SlotOrdered, cache::AttentionType::BSND, out);
  require(values<int64_t>(named(out, in.decode_kv_lengths))
      == std::vector<int64_t>({6, 6}),
    "dense MLA scans one uniform length");
  const auto mask = values<int8_t>(named(out, in.decode_mask));
  require(mask[5] == 0 && mask[6] == 1 && mask[256] == 0 && mask[257] == 1,
    "dense MLA masks each slot's real length");
  rejects([&] {
    forch::TensorList bad;
    mla.inputs(4);
    mla.feed(
      {1, 0}, 4, cache::PackType::SlotOrdered, cache::AttentionType::BSND, bad);
  });
}
} // namespace

int main() {
  try {
    packing();
    const auto rows = write_rows();
    full_metadata();
    paged_metadata();
    dense_mla_workaround();
    std::cout << "PASS runner contract: packing, " << rows
              << " write-row layouts, masks, TND/BSND/MLA metadata, dense and "
                 "paged block tables\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
