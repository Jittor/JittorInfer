#include "../../device/device.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/exception.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"

#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

using namespace forch;
namespace {
void require(
  bool condition, const char *message = "GE device memory contract failed") {
  if (!condition)
    throw forch::exception(message);
}
void rejects(auto &&fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const exception &) {
    rejected = true;
  }
  require(rejected);
}
struct CountingImpl final : Device::Impl {
  explicit CountingImpl(int &frees) : frees(frees) {}

  void *allocate(size_t bytes) const override {
    return new unsigned char[bytes];
  }

  void free(void *ptr) const noexcept override {
    delete[] static_cast<unsigned char *>(ptr);
    ++frees;
  }

  void host_to_device(void *, const void *, size_t) const override {
    throw forch::exception("injected copy failure");
  }
  void device_to_host(void *, const void *, size_t) const override {}

private:
  int &frees;
};

struct CountingDevice : Device {
  explicit CountingDevice(const std::shared_ptr<CountingImpl> &impl)
      : Device(impl) {}
};

void check_buffer_lifetime() {
  int                             frees = 0;
  std::weak_ptr<Device::Impl>     owner;
  std::shared_ptr<Device::Buffer> retained;
  {
    auto impl = std::make_shared<CountingImpl>(frees);
    owner     = impl;
    CountingDevice device(impl);
    int            value = 42;
    rejects([&] {
      auto buffer = device.allocate(sizeof(value));
      device.host_to_device(buffer.data(), &value, sizeof(value));
    });
    require(frees == 1, "failed transfer must release its buffer");
    retained = std::make_shared<Device::Buffer>(device.allocate(1));
  }
  auto alias = retained;
  retained.reset();
  require(!owner.expired() && frees == 1, "buffer released before last owner");
  alias.reset();
  require(owner.expired() && frees == 2, "last owner did not release buffer");
}

} // namespace

int main() {
  try {
    check_buffer_lifetime();
    auto       device = AscendDevice::from_env();
    TensorList retained;
    {
      GEBackend backend(device);
      auto      host =
        op::load_from_vector<float>({1, 2, 3, 4}, {2, 2}).set_name("x");
      auto tensor = host | to_device(backend);
      require(tensor.shape() == host.shape() && tensor.dtype() == host.dtype());
      require(tensor.get_name() == host.get_name());
      retained.push_back(tensor);
      retained.push_back(tensor);
      require(retained[0].get_data() == retained[1].get_data());
      auto scalar = op::load_from_vector<int64_t>({42}, {});
      retained.push_back(scalar | to_device(backend));
      auto empty = op::load_from_vector<float>({}, {0, 2}) | to_device(backend);
      require(empty.shape() == Shape({0, 2}) && empty.get_data() == nullptr);
      float value  = 0;
      auto *buffer = reinterpret_cast<Buffer>(&value);
      rejects([&] {
        op::load_from_buffer(buffer, {-1}, DType::FP32) | to_device(backend);
      });
      rejects([&] {
        op::load_from_buffer(
          buffer, {std::numeric_limits<int64_t>::max(), 2}, DType::FP32)
          | to_device(backend);
      });
      rejects([&] {
        op::load_from_buffer(
          buffer, {std::numeric_limits<int64_t>::max()}, DType::FP64)
          | to_device(backend);
      });
      rejects([&] {
        op::load_from_buffer(nullptr, {1}, DType::FP32) | to_device(backend);
      });
    }
    // Read the retained tensor after GEFinalize and the first alias are gone.
    retained.erase(retained.begin());
    std::vector<float> values(4);
    device.device_to_host(
      values.data(), retained[0].get_data(), 4 * sizeof(float));
    require(values == std::vector<float>({1, 2, 3, 4}));
    int64_t scalar = 0;
    device.device_to_host(&scalar, retained[1].get_data(), sizeof(scalar));
    require(scalar == 42 && retained[1].shape().empty());
    retained.clear();
    std::puts("Device buffer and GE H2D tensor lifetimes: PASS");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
