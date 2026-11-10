#include "device.hpp"
#include "forch/exception.hpp"
#include <utility>

namespace forch {

Device::Device(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}

Device::Buffer Device::allocate(size_t bytes) const {
  return {_impl, bytes == 0 ? nullptr : _impl->allocate(bytes), bytes};
}

void Device::host_to_device(void *dst, const void *src, size_t bytes) const {
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw forch::exception("host_to_device: null pointer with non-zero size");
  }
  _impl->host_to_device(dst, src, bytes);
}

void Device::device_to_host(void *dst, const void *src, size_t bytes) const {
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw forch::exception("device_to_host: null pointer with non-zero size");
  }
  _impl->device_to_host(dst, src, bytes);
}

Device::Buffer::Buffer(std::shared_ptr<Impl> owner, void *data, size_t size)
    : _owner(std::move(owner)), _data(data), _size(size) {}

Device::Buffer::~Buffer() {
  reset();
}

void Device::Buffer::reset() noexcept {
  if (_data != nullptr) {
    _owner->free(_data);
  }
  _data = nullptr;
  _size = 0;
  _owner.reset();
}

Device::Buffer::Buffer(Buffer &&other) noexcept
    : _owner(std::move(other._owner)),
      _data(std::exchange(other._data, nullptr)),
      _size(std::exchange(other._size, 0)) {}

Device::Buffer &Device::Buffer::operator=(Buffer &&other) noexcept {
  if (this != &other) {
    reset();
    _owner = std::move(other._owner);
    _data  = std::exchange(other._data, nullptr);
    _size  = std::exchange(other._size, 0);
  }
  return *this;
}

} // namespace forch
