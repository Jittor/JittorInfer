// Host-side contracts: no GE initialization or allocated device required.
#include "../../utils.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/ops/collective.hpp"
#include "forch/ops/fused_collective_matmul.hpp"
#include "forch/ops/input.hpp"
#include "multirank/mr_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void rejects(auto &&fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const forch::exception &) {
    rejected = true;
  }
  require(rejected, "expected forch::exception");
}

void test_group() {
  const forch::ProcessGroup single;
  require(
    single.rank_id() == 0 && single.world_size() == 1 && single.group_id() == 0,
    "default group");
  const forch::ProcessGroup group(1, 2, 3);
  require(
    group.rank_id() == 1 && group.world_size() == 2 && group.group_id() == 3,
    "explicit group");
  rejects([] { (void)forch::ProcessGroup(-1, 2); });
  rejects([] { (void)forch::ProcessGroup(2, 2); });
  rejects([] { (void)forch::ProcessGroup(0, 0); });
  rejects([] { (void)forch::ProcessGroup(0, 2, -1); });
  auto input   = forch::op::create_input({3, 2}, forch::DType::FP32);
  auto reduced = forch::op::all_reduce(
    input, forch::ProcessGroup(0, 2), forch::op::ReduceOp::Max);
  const auto *reduce_op =
    dynamic_cast<const forch::op::AllReduce *>(reduced.get_op());
  require(reduce_op && reduce_op->reduce_op == forch::op::ReduceOp::Max
      && reduced.shape() == input.shape() && reduced.dtype() == input.dtype(),
    "explicit reduction preserves attributes and metadata");
  rejects(
    [&] { (void)forch::op::reduce_scatter(input, forch::ProcessGroup(0, 2)); });
  auto a         = forch::op::create_input({3, 4}, forch::DType::FP16);
  auto good_bias = forch::op::create_input({3}, forch::DType::FP16);
  (void)forch::op::fused_matmul_all_reduce(
    a, a, good_bias, forch::ProcessGroup(0, 2));
  for (auto dtype : {forch::DType::INT8, forch::DType::BF16}) {
    auto bad_bias = forch::op::create_input({3}, dtype);
    rejects([&] {
      (void)forch::op::fused_matmul_all_reduce(
        a, a, bad_bias, forch::ProcessGroup(0, 2));
    });
  }
  require(mr::supports_fused_world(2) && mr::supports_fused_world(4)
      && mr::supports_fused_world(8) && !mr::supports_fused_world(3)
      && !mr::supports_fused_world(1),
    "fused world sizes");
}

void test_env() {
  constexpr auto name = "FORCH_TEST_DECIMAL_INTEGER";
  unsetenv(name);
  require(forch::detail::getenv_int(name, 7) == 7, "unset fallback");
  setenv(name, "", 1);
  require(forch::detail::getenv_int(name, 7) == 7, "empty fallback");
  setenv(name, "-12", 1);
  require(forch::detail::getenv_int(name, 0) == -12, "signed integer");
  for (auto invalid :
    {"1x", " 1", "1 ", "999999999999999999999999", "2147483648"}) {
    setenv(name, invalid, 1);
    rejects([&] { (void)forch::detail::getenv_int<int32_t>(name, 0); });
  }
  setenv(name, "9223372036854775807", 1);
  require(forch::detail::getenv_int<int64_t>(name, 0)
      == std::numeric_limits<int64_t>::max(),
    "int64 max");
  unsetenv(name);
}

void test_half() {
  // Exhaust all binary16 encodings; finite values and signed zero round-trip
  // exactly, while NaNs retain their class rather than a particular payload.
  for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
    float    f    = mr::f16_to_f32(static_cast<uint16_t>(bits));
    uint16_t back = mr::f32_to_f16(f);
    if ((bits & 0x7c00u) == 0x7c00u && (bits & 0x3ffu) != 0) {
      require(
        std::isnan(f) && (back & 0x7c00u) == 0x7c00u && (back & 0x3ffu) != 0,
        "half NaN");
    } else {
      require(back == bits, "half round trip");
    }
  }
  require(mr::f32_to_f16(1.0f + std::ldexp(1.0f, -11)) == 0x3c00,
    "half tie rounds to even");
  require(mr::f32_to_f16(1.0f + 3 * std::ldexp(1.0f, -11)) == 0x3c02,
    "half odd tie rounds up");
  require(mr::f32_to_f16(std::ldexp(1.0f, -25)) == 0, "subnormal tie");
  require(mr::f32_to_f16(65520.0f) == 0x7c00, "half overflow");
  const uint32_t nan_bits = 0x7f800001;
  float          nan;
  std::memcpy(&nan, &nan_bits, sizeof(nan));
  require(
    std::isnan(mr::bf16_to_f32(mr::f32_to_bf16(nan))), "bf16 NaN payload");
  require(!std::isfinite(mr::max_abs_diff({nan}, {0})), "NaN comparison");
  require(
    !std::isfinite(mr::max_abs_diff({INFINITY}, {INFINITY})), "Inf comparison");
}

} // namespace

int main() {
  try {
    test_group();
    test_env();
    test_half();
    std::puts("collective host contracts: PASS");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "collective host contracts: FAIL: %s\n", e.what());
    return 1;
  }
}
