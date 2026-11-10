#pragma once

/// AttentionMetadata: everything an attention kernel needs besides Q/K/V,
/// derived once per step from the shared SlotMap and consumed by every layer.
/// Graph-side placeholders are created per token budget; feed() produces only
/// the selected scheme's runtime values. Values come from real cursors and
/// page mappings, so the step's pages must be reserved first.

#include "slot-map.hpp"

#include <forch/exception.hpp>
#include <forch/forch.hpp>
#include <forch/ops/const.hpp>
#include <forch/ops/input.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cache {

// Full attention and absorbed MLA bind different decode masks/lengths.
enum class AttentionVariant { Full, MLA };

/// Graph tensors of one token budget T. Builders declare alternatives; only
/// nodes reachable from graph outputs become executable inputs.
struct AttentionInputs {
  int64_t tokens     = 0;
  int64_t slots      = 0;
  int64_t block_size = 0; // Zero for unaligned dense storage.
  // INT64 [T]: scatter destination of each row, shared by every token pool.
  std::optional<forch::Tensor> write_rows;
  std::optional<forch::Tensor> eager_mask;    // FP32 [T, rows], 0 / -inf.
  std::optional<forch::Tensor> fia_mask;      // INT8 [T, rows], 0 / 1.
  std::optional<forch::Tensor> causal_mask;   // INT8 [2048,2048] template.
  std::optional<forch::Tensor> query_lengths; // INT64 [sequences], cumulative.
  std::optional<forch::Tensor> kv_lengths;    // INT64 [sequences].
  std::optional<forch::Tensor> decode_kv_lengths; // INT64 [B].
  // Full: INT8 [B,1,Q,stride] for multi-query slot chunks only.
  // MLA: INT8 [B,1,stride], the dense decode workaround.
  std::optional<forch::Tensor> decode_mask;
  // INT32 [sequences, columns] (TND) and [B, columns] (paged BSND). Dense
  // tables never change and are constants; paged tables are fed per step.
  std::optional<forch::Tensor> block_table;
  std::optional<forch::Tensor> slot_block_table;
};

class AttentionMetadata {
public:
  // TND padding queries form dummy sequences of at most this many rows. The
  // dummy KV length stays inside block-table entry zero (one block).
  static constexpr int64_t kDummyChunk = 32;

  AttentionMetadata(const SlotMap &map, AttentionVariant variant,
    std::string name = "attention")
      : map_(map), variant_(variant), name_(std::move(name)) {}

  AttentionVariant variant() const {
    return variant_;
  }

  int64_t tnd_sequences(int64_t tq) const {
    const auto chunk = dummy_chunk();
    return map_.slots() + (tq + chunk - 1) / chunk;
  }

  /// Build-time placeholders for token budget tq, memoized per budget.
  const AttentionInputs &inputs(int64_t tq) {
    if (auto it = groups_.find(tq); it != groups_.end())
      return it->second;
    if (tq <= 0)
      throw forch::exception(
        "attention metadata: token budget must be positive");
    const auto      b    = map_.slots();
    const auto      rows = map_.storage_rows();
    const auto      name = name_ + "_t" + std::to_string(tq);
    AttentionInputs in;
    in.tokens     = tq;
    in.slots      = b;
    in.block_size = map_.block_size();
    in.write_rows = forch::op::create_input({tq}, forch::DType::INT64, true)
                      .set_name(name + "_write_rows");
    in.eager_mask = forch::op::create_input({tq, rows}, forch::DType::FP32)
                      .set_name(name + "_eager_mask");
    in.fia_mask = forch::op::create_input({tq, rows}, forch::DType::INT8)
                    .set_name(name + "_fia_mask");
    in.causal_mask       = causal_template();
    const auto sequences = tnd_sequences(tq);
    in.query_lengths =
      forch::op::create_input({sequences}, forch::DType::INT64, true)
        .set_name(name + "_query_lengths");
    in.kv_lengths =
      forch::op::create_input({sequences}, forch::DType::INT64, true)
        .set_name(name + "_kv_lengths");
    in.decode_kv_lengths =
      forch::op::create_input({b}, forch::DType::INT64, true)
        .set_name(name + "_decode_kv_lengths");
    if (map_.block_size()) {
      const auto stride = map_.slot_stride();
      if (variant_ == AttentionVariant::MLA)
        in.decode_mask =
          forch::op::create_input({b, 1, stride}, forch::DType::INT8)
            .set_name(name + "_decode_mask");
      else if (tq > b && tq % b == 0)
        in.decode_mask =
          forch::op::create_input({b, 1, tq / b, stride}, forch::DType::INT8)
            .set_name(name + "_decode_mask");
      in.block_table      = table_input(sequences);
      in.slot_block_table = table_input(b);
    }
    return groups_.emplace(tq, std::move(in)).first->second;
  }

  /// Runtime values for one step plan under the selected scheme.
  void feed(const std::vector<int64_t> &counts, int64_t tq, PackType pack,
    AttentionType type, forch::TensorList &out) const {
    const auto &in = groups_.at(tq);
    out.push_back(
      forch::op::load_from_vector(map_.write_rows(counts, tq, pack), {tq})
        .set_name(in.write_rows->get_name()));
    switch (type) {
    case AttentionType::Eager:
      out.push_back(mask(counts, tq, pack, forch::DType::FP32)
                      .set_name(in.eager_mask->get_name()));
      return;
    case AttentionType::DenseFIA:
      out.push_back(mask(counts, tq, pack, forch::DType::INT8)
                      .set_name(in.fia_mask->get_name()));
      return;
    case AttentionType::TND:
      require_aligned(type);
      feed_tnd(in, counts, tq, out);
      if (map_.paged())
        out.push_back(table(in.block_table->get_name(), tnd_sequences(tq)));
      return;
    case AttentionType::BSND:
    case AttentionType::PagedBSND:
      require_aligned(type);
      if ((type == AttentionType::PagedBSND) != map_.paged())
        throw forch::exception(
          "attention metadata: BSND storage does not match the pool");
      if (variant_ == AttentionVariant::MLA)
        feed_mla_decode(in, counts, tq, type, out);
      else
        feed_bsnd(in, counts, tq, out);
      if (type == AttentionType::PagedBSND)
        out.push_back(table(in.slot_block_table->get_name(), map_.slots()));
      return;
    }
    throw forch::exception("attention metadata: invalid attention type");
  }

  /// Full mask over physical pool rows: query rows follow the packing, slot
  /// rows see logical positions [0, cursor + i]; padding rows see row zero.
  forch::Tensor mask(const std::vector<int64_t> &counts, int64_t tq,
    PackType pack, forch::DType dtype) const {
    map_.check_plan(counts, tq, pack);
    const auto columns = map_.storage_rows();
    const auto b       = map_.slots();
    auto       build   = [&]<typename T>(T blocked) {
      std::vector<T> values(tq * columns, blocked);
      int64_t        row = 0;
      for (int64_t sid = 0; sid < b; ++sid) {
        const auto width = pack == PackType::SlotOrdered ? tq / b : counts[sid];
        for (int64_t i = 0; i < width; ++i, ++row) {
          auto *line = values.data() + row * columns;
          if (i < counts[sid])
            map_.for_each_run(sid, map_.positions()[sid] + i + 1,
                      [&](int64_t first, int64_t n) {
                std::fill_n(line + first, n, T{0});
              });
          else
            line[0] = T{0};
        }
      }
      for (; row < tq; ++row)
        values[row * columns] = T{0};
      return forch::op::load_from_vector(std::move(values), {tq, columns});
    };
    if (dtype == forch::DType::FP32)
      return build(-std::numeric_limits<float>::infinity());
    if (dtype == forch::DType::INT8)
      return build(int8_t{1});
    throw forch::exception("attention metadata: mask dtype must be FP32/INT8");
  }

private:
  int64_t dummy_chunk() const {
    return std::min<int64_t>(kDummyChunk, map_.capacity());
  }
  void require_aligned(AttentionType type) const {
    if (!map_.block_size())
      throw forch::exception("attention metadata: scheme "
        + std::to_string(int(type)) + " requires block-aligned storage");
  }

  forch::Tensor table_input(int64_t sequences) {
    const auto name = name_ + "_blocks_s" + std::to_string(sequences);
    if (!map_.paged())
      return forch::op::load_from_vector(
        map_.block_table(sequences), {sequences, map_.table_columns()})
        .set_name(name);
    return forch::op::create_input(
      {sequences, map_.table_columns()}, forch::DType::INT32)
      .set_name(name);
  }
  forch::Tensor table(const std::string &name, int64_t sequences) const {
    return forch::op::load_from_vector(
      map_.block_table(sequences), {sequences, map_.table_columns()})
      .set_name(name);
  }

  void feed_tnd(const AttentionInputs &in, const std::vector<int64_t> &counts,
    int64_t tq, forch::TensorList &out) const {
    map_.check_plan(counts, tq, PackType::Compact);
    const auto           b = map_.slots(), sequences = tnd_sequences(tq);
    const auto          &positions = map_.positions();
    std::vector<int64_t> aq(sequences), ak(sequences);
    int64_t              cumulative = 0;
    for (int64_t sid = 0; sid < b; ++sid) {
      cumulative += counts[sid];
      aq[sid] = cumulative;
      ak[sid] = counts[sid] ? positions[sid] + counts[sid] : 0;
    }
    for (int64_t sid = b; sid < sequences; ++sid) {
      ak[sid] = std::min(dummy_chunk(), tq - cumulative);
      cumulative += ak[sid];
      aq[sid] = cumulative;
    }
    out.push_back(forch::op::load_from_vector(std::move(aq), {sequences})
                    .set_name(in.query_lengths->get_name()));
    out.push_back(forch::op::load_from_vector(std::move(ak), {sequences})
                    .set_name(in.kv_lengths->get_name()));
  }

  // KV length bounds the whole chunk; a per-query mask also excludes future
  // tokens inside it. Padding queries keep one visible key.
  void feed_bsnd(const AttentionInputs &in, const std::vector<int64_t> &counts,
    int64_t tq, forch::TensorList &out) const {
    map_.check_plan(counts, tq, PackType::SlotOrdered);
    const auto           b = map_.slots(), width = tq / b;
    const auto          &positions = map_.positions();
    std::vector<int64_t> lengths(b, 1);
    for (int64_t sid = 0; sid < b; ++sid)
      if (counts[sid])
        lengths[sid] = positions[sid] + counts[sid];
    if (width > 1) {
      const auto          columns = map_.slot_stride();
      std::vector<int8_t> mask(b * width * columns, 1);
      for (int64_t sid = 0; sid < b; ++sid)
        for (int64_t j = 0; j < width; ++j)
          std::fill_n(mask.data() + (sid * width + j) * columns,
            j < counts[sid] ? positions[sid] + j + 1 : 1, int8_t{0});
      out.push_back(
        forch::op::load_from_vector(std::move(mask), {b, 1, width, columns})
          .set_name(in.decode_mask->get_name()));
    }
    out.push_back(forch::op::load_from_vector(std::move(lengths), {b})
                    .set_name(in.decode_kv_lengths->get_name()));
  }

  // MLA decode has one query per slot. Dense storage scans a uniform length
  // with a mask to avoid CANN's problematic split/merge path; paged storage
  // uses independent lengths and no mask.
  void feed_mla_decode(const AttentionInputs &in,
    const std::vector<int64_t> &counts, int64_t tq, AttentionType type,
    forch::TensorList &out) const {
    const auto b = map_.slots();
    if (tq != b)
      throw forch::exception("attention metadata: MLA decode requires T = B");
    map_.check_plan(counts, tq, PackType::SlotOrdered);
    const auto          &positions = map_.positions();
    std::vector<int64_t> lengths(b, 1);
    for (int64_t sid = 0; sid < b; ++sid)
      if (counts[sid])
        lengths[sid] = positions[sid] + 1;
    if (type == AttentionType::BSND) {
      const auto          stride = map_.slot_stride();
      std::vector<int8_t> values(b * stride, 1);
      for (int64_t sid = 0; sid < b; ++sid)
        std::fill_n(values.data() + sid * stride, lengths[sid], int8_t{0});
      std::fill(lengths.begin(), lengths.end(),
        *std::max_element(lengths.begin(), lengths.end()));
      out.push_back(
        forch::op::load_from_vector(std::move(values), {b, 1, stride})
          .set_name(in.decode_mask->get_name()));
    }
    out.push_back(forch::op::load_from_vector(std::move(lengths), {b})
                    .set_name(in.decode_kv_lengths->get_name()));
  }

  // Shared across instances; graphs that do not reach it never copy it.
  static forch::Tensor causal_template() {
    static const auto mask = [] {
      std::vector<int8_t> values(2048 * 2048, 0);
      for (int64_t i = 0; i < 2048; ++i)
        for (int64_t j = i + 1; j < 2048; ++j)
          values[i * 2048 + j] = 1;
      return forch::op::load_from_vector(std::move(values), {2048, 2048});
    }();
    return mask;
  }

  const SlotMap                     &map_;
  AttentionVariant                   variant_;
  std::string                        name_;
  std::map<int64_t, AttentionInputs> groups_;
};

} // namespace cache
