#include "forch/backend/forch-ge.hpp"
#include "forch/exception.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/softmax.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>

namespace {

void rejects(auto &&fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const forch::exception &) {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("second/invalid backend accepted");
}

void run(forch::GEBackend backend, const char *name) {
  auto input  = forch::op::create_input({2}, forch::DType::FP32).set_name("x");
  auto output = forch::op::softmax(0)(input).set_name("y");
  auto graph  = backend.build(forch::Graph({output}).set_name(name));
  graph.compile();
  graph.load();
  auto feed   = forch::op::load_from_vector<float>({0, 0}, {2}).set_name("x");
  auto result = graph.run({feed}).get("y");
  if (result.shape() != forch::Shape{2} || result.dtype() != forch::DType::FP32)
    throw std::runtime_error("runtime metadata mismatch");
  std::ostringstream dump;
  result.get_op()->dump(dump);
  if (dump.str().find("origin_shape=1d(2)") == std::string::npos
    || dump.str().find("storage_shape=1d(2)") == std::string::npos)
    throw std::runtime_error("runtime dump metadata mismatch");
  const auto *values = static_cast<const float *>(result.get_data());
  for (size_t i = 0; i < 2; ++i) {
    if (!std::isfinite(values[i]) || std::fabs(values[i] - 0.5f) > 1e-6f)
      throw std::runtime_error("backend copy produced incorrect values");
  }
}

} // namespace

int main() {
  try {
    rejects([] { forch::AscendDevice invalid(-1); });
    auto device          = forch::AscendDevice::from_env();
    auto options         = forch::GEBackend::Options::from_env();
    auto invalid         = options;
    invalid.hccl_timeout = std::chrono::seconds(0);
    rejects([&] { forch::GEBackend backend(device, invalid); });
    {
      forch::GEBackend backend(device, options);
      run(backend, "lifecycle_before_rejection");
      rejects([&] { forch::GEBackend second(device, options); });
      auto different = options;
      different.hccl_timeout += std::chrono::seconds(1);
      rejects([&] { forch::GEBackend second(device, different); });
      run(backend, "lifecycle_after_rejection");
    }
    rejects([&] { forch::GEBackend second(device, options); });
    std::puts("backend lifecycle: PASS");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "backend lifecycle: FAIL: %s\n", e.what());
    return 1;
  }
}
