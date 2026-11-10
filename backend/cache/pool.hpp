#pragma once

/// Pool: one persistent device tensor [rows, width] and its graph
/// placeholder. Pools store data only; row addressing belongs to SlotMap
/// (token caches) or to the owning semantic cache (GDN state).

#include <forch/exception.hpp>
#include <forch/forch.hpp>
#include <forch/graph.hpp>
#include <forch/ops/const.hpp>
#include <forch/ops/input.hpp>
#include <forch/ops/reshape.hpp>
#include <forch/ops/scatter_update.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cache {

class Pool {
public:
  Pool(forch::Backend *backend, std::string name, int64_t rows, int64_t width,
    forch::DType dtype)
      : backend_(backend), name_(std::move(name)), rows_(rows), width_(width),
        dtype_(dtype) {
    if (rows <= 0 || width <= 0)
      throw forch::exception("Pool: dimensions must be positive");
    element_bytes(); // Rejects unsupported dtypes before allocation.
  }
  Pool(const Pool &)            = delete;
  Pool &operator=(const Pool &) = delete;

  const std::string &name() const {
    return name_;
  }
  int64_t rows() const {
    return rows_;
  }
  int64_t width() const {
    return width_;
  }
  forch::DType dtype() const {
    return dtype_;
  }

  /// Allocate zeroed device storage. Padding/guard rows start finite.
  void init() {
    if (data_)
      throw forch::exception("Pool: already initialized");
    std::vector<uint8_t> zeros(
      static_cast<size_t>(rows_ * width_ * element_bytes()), 0);
    auto host = forch::op::load_from_buffer(
      forch::Buffer(zeros.data()), {rows_, width_}, dtype_);
    host.set_name(tensor_name());
    data_ = host | forch::to_device(*backend_);
  }

  /// Graph placeholder, reused by every graph and layer step.
  forch::Tensor input() {
    if (!input_)
      input_ = forch::op::create_input({rows_, width_}, dtype_)
                 .set_name(tensor_name());
    return *input_;
  }
  /// Persistent runtime storage fed to every execution.
  const forch::Tensor &tensor() const {
    if (!data_)
      throw forch::exception("Pool: storage is not initialized");
    return *data_;
  }

  /// Write updates [n, width] into rows [n]; returns the whole updated pool.
  forch::Tensor scatter(const forch::Tensor &rows, forch::Tensor updates) {
    return forch::op::scatter_update()(input(), rows, std::move(updates));
  }

  /// Overwrite every row in order (recurrent state already in slot order).
  forch::Tensor overwrite(const forch::Tensor &updates) {
    std::vector<int64_t> iota(static_cast<size_t>(rows_));
    for (int64_t i = 0; i < rows_; ++i)
      iota[i] = i;
    return scatter(forch::op::load_from_vector(std::move(iota), {rows_}),
      updates | forch::op::reshape({rows_, width_}));
  }

  /// Zero the row selected by INT64 [1] on device. load_from_buffer does not
  /// copy, so the zero row lives as long as the pool, which outlives graphs.
  forch::Tensor zero_row(const forch::Tensor &row) {
    zero_row_.assign(static_cast<size_t>(width_ * element_bytes()), 0);
    auto zeros = forch::op::load_from_buffer(
      forch::Buffer(zero_row_.data()), {1, width_}, dtype_);
    return scatter(row, zeros).set_name(tensor_name() + "_zero");
  }

private:
  std::string tensor_name() const {
    return "pool_" + name_;
  }
  int64_t element_bytes() const {
    switch (dtype_) {
    case forch::DType::FP32:
      return 4;
    case forch::DType::FP16:
    case forch::DType::BF16:
      return 2;
    default:
      throw forch::exception("Pool: unsupported dtype (FP32/FP16/BF16 only)");
    }
  }

  forch::Backend              *backend_;
  std::string                  name_;
  int64_t                      rows_, width_;
  forch::DType                 dtype_;
  std::optional<forch::Tensor> input_;
  std::optional<forch::Tensor> data_;
  std::vector<uint8_t>         zero_row_;
};

} // namespace cache
