#pragma once

#include "forch/device/device.hpp"

namespace forch {

struct Device::Impl {
  virtual ~Impl()                              = default;
  virtual void *allocate(size_t bytes) const   = 0;
  virtual void  free(void *ptr) const noexcept = 0;
  virtual void  host_to_device(
     void *dst, const void *src, size_t bytes) const = 0;
  virtual void device_to_host(
    void *dst, const void *src, size_t bytes) const = 0;
};

} // namespace forch
