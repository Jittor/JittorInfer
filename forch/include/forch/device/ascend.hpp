#pragma once

#include "forch/device/device.hpp"
#include <cstdint>

namespace forch {

// Share this handle across backends. The caller uses one Ascend device per
// process and keeps its default context current during serial use and teardown.
// External ACL owners must keep the runtime alive until all buffers are freed.
class AscendDevice : public Device {
public:
  // Logical index within ASCEND_RT_VISIBLE_DEVICES, passed to aclrtSetDevice.
  explicit AscendDevice(int32_t device_id);
  // Read FORCH_GE_DEVICE_ID (logical id, default 0).
  static AscendDevice from_env();
  int32_t             id() const;
};

} // namespace forch
