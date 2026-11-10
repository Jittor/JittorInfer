#pragma once

#include <cstddef>
#include <memory>

namespace forch {

// Copyable device handle for synchronous byte allocations and transfers.
class Device {
public:
  struct Impl;
  class Buffer;

  Buffer allocate(size_t bytes) const;
  // Both buffers must hold at least bytes. Device addresses must belong to
  // this device. Zero bytes is a no-op, including with null pointers.
  void host_to_device(void *dst, const void *src, size_t bytes) const;
  void device_to_host(void *dst, const void *src, size_t bytes) const;

protected:
  explicit Device(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> _impl;
};

// Owns one allocation and retains the device implementation until after free.
// Buffer is move-only; tensors retain it through std::shared_ptr<Buffer>.
class Device::Buffer {
public:
  Buffer() = default;
  ~Buffer();
  Buffer(const Buffer &)            = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  void *data() const noexcept {
    return _data;
  }
  size_t size() const noexcept {
    return _size;
  }

  Device device() const noexcept {
    return Device(_owner);
  }

private:
  friend class Device;
  Buffer(std::shared_ptr<Impl> owner, void *data, size_t size);
  void reset() noexcept;

  // TODO: Future allocator caches must avoid a strong reference back to Buffer.
  std::shared_ptr<Impl> _owner;
  void                 *_data = nullptr;
  size_t                _size = 0;
};

} // namespace forch
