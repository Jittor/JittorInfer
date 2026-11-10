#include "../device/device.hpp"
#include "../utils.hpp"
#include "forch/device/ascend.hpp"
#include "forch/exception.hpp"
#include "forch/forch.hpp"

#include <acl/acl.h>
#include <string>

namespace forch {
namespace {

void check(aclError error, const char *operation) {
  if (error != ACL_SUCCESS) {
    throw forch::exception(std::string("forch-ascend: ") + operation
      + " failed, ret=" + std::to_string(error));
  }
}

void cleanup(aclError error, const char *operation) noexcept {
  if (error != ACL_SUCCESS) {
    // Cleanup must remain noexcept even if the logger cannot allocate or write.
    logger()->error("forch-ascend: {} failed, ret={}", operation, error);
  }
}

struct AscendImpl final : Device::Impl {
  int32_t id;

  explicit AscendImpl(int32_t device_id) : id(device_id) {
    if (id < 0) {
      throw forch::exception("AscendDevice: device_id must be >= 0");
    }
    const auto ret = aclInit(nullptr);
    if (ret != ACL_ERROR_REPEAT_INITIALIZE) {
      check(ret, "aclInit");
    }
    check(aclrtSetDevice(id), "aclrtSetDevice");
  }

  void *allocate(size_t bytes) const override {
    void *ptr = nullptr;
    check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    return ptr;
  }
  void free(void *ptr) const noexcept override {
    cleanup(aclrtFree(ptr), "aclrtFree");
  }
  void host_to_device(void *dst, const void *src, size_t bytes) const override {
    check(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE),
      "aclrtMemcpy H2D");
  }
  void device_to_host(void *dst, const void *src, size_t bytes) const override {
    check(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
      "aclrtMemcpy D2H");
  }
};

} // namespace

AscendDevice::AscendDevice(int32_t id)
    : Device(std::make_shared<AscendImpl>(id)) {}
AscendDevice AscendDevice::from_env() {
  return AscendDevice(detail::getenv_int<int32_t>("FORCH_GE_DEVICE_ID", 0));
}
int32_t AscendDevice::id() const {
  return static_cast<const AscendImpl *>(_impl.get())->id;
}

} // namespace forch
