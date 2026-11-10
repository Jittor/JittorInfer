#pragma once

// GDN keeps BF16 recurrent state (FP32 in the eager reference) and BF16
// convolution history, one row per slot. Each step packs k <= 8 rows per
// slot (SlotOrdered). Padding has beta = g = 0 so it cannot advance the
// recurrent state. CANN910 requires BlockNum >= token capacity only at the
// operator boundary; persistent state stays one block per slot.
//
// Row maps and padding validity depend only on the step plan, so one
// GDNMetadata serves every layer; each GDNCache owns only its two pools.
// Graph outputs, admission-time reset execution and graphs belong to the
// runner.

#include "pool.hpp"
#include "types.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <forch/forch.hpp>
#include <forch/ops/cat.hpp>
#include <forch/ops/index_select.hpp>
#include <forch/ops/reshape.hpp>
#include <forch/ops/slice.hpp>

namespace cache {

struct GDNGeometry {
  int64_t slots         = 0;
  int64_t num_v_heads   = 0; // Nv
  int64_t head_v_dim    = 0; // Dv
  int64_t head_k_dim    = 0; // Dk
  int64_t conv_channels = 0; // C
  int64_t conv_kernel   = 4; // K
};

/// Per-budget graph inputs shared by all GDN layers.
struct GDNInputs {
  int64_t       tokens = 0;
  forch::Tensor indices;    // INT32 [T]: owning slot of each row.
  forch::Tensor lengths;    // INT32 [B]: rows per slot.
  forch::Tensor conv_read;  // INT64 [T*K]: causal window of each row.
  forch::Tensor conv_write; // INT64 [B*K]: final window of each slot.
  forch::Tensor valid;      // FP32 [T,1]: 1 for real rows, 0 for padding.
};

class GDNMetadata {
public:
  static constexpr int64_t kMaxRowsPerSlot = 8;

  GDNMetadata(int64_t slots, int64_t conv_kernel)
      : slots_(slots), kernel_(conv_kernel) {
    if (slots <= 0 || conv_kernel <= 0)
      throw forch::exception("gdn metadata: invalid geometry");
  }

  const GDNInputs &inputs(int64_t tq) {
    if (auto it = groups_.find(tq); it != groups_.end())
      return it->second;
    check_budget(tq);
    const auto           width = tq / slots_, history = slots_ * kernel_;
    std::vector<int32_t> ids(tq);
    std::vector<int64_t> read(tq * kernel_);
    for (int64_t sid = 0; sid < slots_; ++sid)
      for (int64_t j = 0; j < width; ++j) {
        const auto row = sid * width + j;
        ids[row]       = static_cast<int32_t>(sid);
        for (int64_t tap = 0; tap < kernel_; ++tap) {
          const auto relative       = j - (kernel_ - 1) + tap;
          read[row * kernel_ + tap] = relative < 0
            ? sid * kernel_ + kernel_ + relative
            : history + sid * width + relative;
        }
      }
    GDNInputs in{tq, forch::op::load_from_vector(std::move(ids), {tq}),
      forch::op::load_from_vector(
        std::vector<int32_t>(slots_, static_cast<int32_t>(width)), {slots_}),
      forch::op::load_from_vector(std::move(read), {tq * kernel_}),
      forch::op::create_input({slots_ * kernel_}, forch::DType::INT64)
        .set_name(name("conv_write", tq)),
      forch::op::create_input({tq, 1}, forch::DType::FP32)
        .set_name(name("valid", tq))};
    return groups_.emplace(tq, std::move(in)).first->second;
  }

  // Slot sid owns [sid*k, (sid+1)*k); only its first counts[sid] rows are
  // real tokens, so padding never updates persistent history.
  void feed(const std::vector<int64_t> &counts, int64_t tq, PackType pack,
    forch::TensorList &out) const {
    check_budget(tq);
    const auto width = tq / slots_, history = slots_ * kernel_;
    if (pack != PackType::SlotOrdered
      || counts.size() != static_cast<size_t>(slots_))
      throw forch::exception("gdn metadata: requires slot-ordered packing");
    std::vector<int64_t> write(slots_ * kernel_);
    std::vector<float>   valid(tq, 0);
    for (int64_t sid = 0; sid < slots_; ++sid) {
      const auto count = counts[sid];
      if (count < 0 || count > width)
        throw forch::exception("gdn metadata: invalid token counts");
      std::fill_n(valid.begin() + sid * width, count, 1.0F);
      for (int64_t tap = 0; tap < kernel_; ++tap) {
        const auto relative        = count - kernel_ + tap;
        write[sid * kernel_ + tap] = relative < 0
          ? sid * kernel_ + kernel_ + relative
          : history + sid * width + relative;
      }
    }
    out.push_back(forch::op::load_from_vector(std::move(valid), {tq, 1})
                    .set_name(name("valid", tq)));
    out.push_back(
      forch::op::load_from_vector(std::move(write), {slots_ * kernel_})
        .set_name(name("conv_write", tq)));
  }

private:
  void check_budget(int64_t tq) const {
    if (tq <= 0 || tq % slots_ || tq / slots_ > kMaxRowsPerSlot)
      throw forch::exception("gdn metadata: invalid per-slot graph capacity");
  }
  static std::string name(const char *kind, int64_t tq) {
    return std::string("gdn_") + kind + "_t" + std::to_string(tq);
  }

  int64_t                      slots_, kernel_;
  std::map<int64_t, GDNInputs> groups_;
};

class GDNCache {
public:
  GDNCache(forch::Backend *backend, const GDNGeometry &geometry,
    forch::DType state_dtype, std::string name)
      : geo_(geometry), name_(std::move(name)),
        state_(backend, name_ + "_state", geometry.slots,
          geometry.num_v_heads * geometry.head_v_dim * geometry.head_k_dim,
          state_dtype),
        conv_(backend, name_ + "_conv", geometry.slots,
          geometry.conv_kernel * geometry.conv_channels, forch::DType::BF16) {}

  void init() {
    state_.init();
    conv_.init();
  }

  /// Recurrent state view [B, Nv, Dv, Dk].
  forch::Tensor state() {
    return state_.input()
      | forch::op::reshape(
        {geo_.slots, geo_.num_v_heads, geo_.head_v_dim, geo_.head_k_dim});
  }

  // Gather each token's causal window from old history plus this chunk. The
  // second result is the final window of every slot, including unchanged
  // history for slots with no tokens in this step.
  std::pair<forch::Tensor, forch::Tensor> convolution_windows(
    const forch::Tensor &projected, const GDNInputs &in) {
    const auto tq   = projected.shape()[0];
    auto       pool = conv_.input()
      | forch::op::reshape({geo_.slots * geo_.conv_kernel, geo_.conv_channels});
    auto source  = forch::op::cat(0)(pool, projected);
    auto windows = forch::op::index_select(0)(source, in.conv_read)
      | forch::op::reshape({tq, geo_.conv_kernel, geo_.conv_channels});
    auto next = forch::op::index_select(0)(source, in.conv_write)
      | forch::op::reshape({geo_.slots, geo_.conv_kernel * geo_.conv_channels});
    return {windows, next};
  }

  /// Register the in-place write-back. Scatter nodes execute only when they
  /// reach a graph output, but nobody reads their values: exposing whole
  /// pools would copy them back to host every step (~10MB for 18 layers).
  /// A one-element slice keeps the dependency and costs 4 bytes per pool.
  void update(
    const forch::Tensor &conv_next, const forch::Tensor &final_state) {
    state_next_ = tracker(state_.overwrite(final_state), "_state_wb");
    conv_next_  = tracker(conv_.overwrite(conv_next), "_conv_wb");
  }
  void append_write_back(forch::TensorList &outputs) const {
    if (state_next_ && conv_next_) {
      outputs.push_back(*state_next_);
      outputs.push_back(*conv_next_);
    }
  }

  /// Device-side zeroing of one slot; outputs are trackers like update().
  void append_reset_updates(
    const forch::Tensor &slot, forch::TensorList &outputs) {
    outputs.push_back(tracker(state_.zero_row(slot), "_state_z"));
    outputs.push_back(tracker(conv_.zero_row(slot), "_conv_z"));
  }

  void feed(forch::TensorList &inputs) const {
    inputs.push_back(state_.tensor());
    inputs.push_back(conv_.tensor());
  }

private:
  forch::Tensor tracker(const forch::Tensor &pool, const char *tag) const {
    return (pool
      | forch::op::slice(
        {forch::op::Slice::Interval{0, 1}, forch::op::Slice::Interval{0, 1}}))
      .set_name(name_ + tag);
  }

  GDNGeometry                  geo_;
  std::string                  name_;
  Pool                         state_, conv_;
  std::optional<forch::Tensor> state_next_, conv_next_;
};

} // namespace cache
