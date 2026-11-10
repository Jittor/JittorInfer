#pragma once

/**
 * Shared bits for the multi-rank tests.
 *
 * Rank, world size, device and rank table are *not* here: the launcher exports
 * them (FORCH_RANK_ID, FORCH_WORLD_SIZE, FORCH_GE_*) and AscendDevice/GEBackend
 * pick them up, so a test only reads its own knobs off argv.
 */

#include "forch/distributed.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mr {

using forch::DType;
using forch::Shape;
using forch::Tensor;

[[noreturn]] inline void fail(const std::string &what) {
  // forch's exceptions already carry the shim and GE diagnostics, so there is
  // nothing to append here.
  throw std::runtime_error("FAIL: " + what);
}

/**
 * every `forch::exception` carries a 32-frame backtrace in its message. That
 * earns its keep for an unexpected failure and is pure noise for one the test
 * asked for, so expected rejections report only the message itself.
 */
inline std::string first_line(const std::string &text) {
  const size_t end = text.find('\n');
  return end == std::string::npos ? text : text.substr(0, end);
}

inline std::string shape_str(const Shape &s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (i != 0)
      out += "x";
    out += std::to_string(s[i]);
  }
  return out;
}

inline size_t numel(const Shape &s) {
  size_t n = 1;
  for (auto d : s) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

// ------------------------------------------------------------------ argv

/**
 * `--key value` only. An unrecognized flag is an error rather than ignored,
 * so a typo in a launcher invocation cannot quietly run the defaults.
 */
class Args {
public:
  Args(int argc, char **argv, std::initializer_list<const char *> known) {
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (!flag.starts_with("--")) {
        fail("unexpected positional argument '" + flag + "'");
      }
      const std::string key = flag.substr(2);
      if (std::ranges::find(known, key) == known.end()) {
        fail("unknown flag '" + flag + "'");
      }
      if (i + 1 >= argc) {
        fail("flag '" + flag + "' needs a value");
      }
      values_.emplace_back(key, argv[++i]);
    }
  }

  std::string get(const char *key, const std::string &fallback) const {
    for (const auto &[k, v] : values_) {
      if (k == key) {
        return v;
      }
    }
    return fallback;
  }

  int64_t get_int(const char *key, int64_t fallback) const {
    const std::string raw = get(key, "");
    if (raw.empty()) {
      return fallback;
    }
    size_t        consumed = 0;
    const int64_t value    = std::stoll(raw, &consumed);
    if (consumed != raw.size()) {
      fail(
        std::string("--") + key + ": expected an integer, got '" + raw + "'");
    }
    return value;
  }

  DType get_dtype(const char *key, DType fallback) const {
    const std::string raw = get(key, "");
    if (raw.empty()) {
      return fallback;
    }
    if (raw == "fp32" || raw == "float")
      return DType::FP32;
    if (raw == "fp16")
      return DType::FP16;
    if (raw == "bf16")
      return DType::BF16;
    fail(std::string("--") + key + ": unknown dtype '" + raw
      + "' (expected fp32/fp16/bf16)");
  }

private:
  std::vector<std::pair<std::string, std::string>> values_;
};

inline const char *dtype_name(DType dt) {
  switch (dt) {
  case DType::FP32:
    return "fp32";
  case DType::FP16:
    return "fp16";
  case DType::BF16:
    return "bf16";
  default:
    return "?";
  }
}

// ------------------------------------------------- half-precision on host

// IEEE binary16 conversion with round-to-nearest-even, independent of host ISA.
// Keep NaNs as NaNs even when their payload lives below the retained bits.

inline uint16_t f32_to_f16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint16_t sign     = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const uint32_t exponent = (bits >> 23) & 0xffu;
  const uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) {
    return sign | (mantissa == 0 ? 0x7c00u : 0x7e00u);
  }
  const int e = static_cast<int>(exponent) - 127;
  if (e > 15)
    return sign | 0x7c00u;
  if (e < -25)
    return sign;
  const uint32_t significand = mantissa | 0x800000u;
  const int      shift       = e < -14 ? -e - 1 : 13;
  uint32_t       rounded     = significand >> shift;
  const uint32_t remainder   = significand & ((1u << shift) - 1);
  const uint32_t halfway     = 1u << (shift - 1);
  rounded += remainder > halfway || (remainder == halfway && (rounded & 1u));
  if (e < -14)
    return sign | static_cast<uint16_t>(rounded);
  return sign | static_cast<uint16_t>(((e + 14) << 10) + rounded);
}

inline float f16_to_f32(uint16_t bits) {
  const uint32_t sign     = static_cast<uint32_t>(bits & 0x8000u) << 16;
  uint32_t       exponent = (bits >> 10) & 0x1fu;
  uint32_t       mantissa = bits & 0x3ffu;
  uint32_t       value;
  if (exponent == 0 && mantissa != 0) {
    int e = -14;
    while ((mantissa & 0x400u) == 0) {
      mantissa <<= 1;
      --e;
    }
    value = sign | (static_cast<uint32_t>(e + 127) << 23)
      | ((mantissa & 0x3ffu) << 13);
  } else {
    exponent = exponent == 0 ? 0 : (exponent == 31 ? 255 : exponent + 112);
    value    = sign | (exponent << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

inline uint16_t f32_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<uint16_t>((x >> 16) | 0x40u);
  }
  const uint32_t lsb = (x >> 16) & 1u;
  x += 0x7fffu + lsb;
  return static_cast<uint16_t>(x >> 16);
}

inline float bf16_to_f32(uint16_t b) {
  const uint32_t x = static_cast<uint32_t>(b) << 16;
  float          f;
  std::memcpy(&f, &x, 4);
  return f;
}

inline uint16_t to_u16_one(float v, DType dt) {
  return (dt == DType::BF16) ? f32_to_bf16(v) : f32_to_f16(v);
}

inline float from_u16_one(uint16_t u, DType dt) {
  return (dt == DType::BF16) ? bf16_to_f32(u) : f16_to_f32(u);
}

inline std::vector<float> read_back(const void *data, size_t n, DType dt) {
  std::vector<float> out(n);
  if (dt == DType::FP32) {
    const auto *p = static_cast<const float *>(data);
    for (size_t i = 0; i < n; ++i) {
      out[i] = p[i];
    }
  } else {
    const auto *p = static_cast<const uint16_t *>(data);
    for (size_t i = 0; i < n; ++i) {
      out[i] = from_u16_one(p[i], dt);
    }
  }
  return out;
}

// Raw bits, for exact equality checks: comparing floats instead would let
// "both are NaN" or +0/-0 pass.
inline std::vector<uint32_t> read_bits(const Tensor &t) {
  const size_t          n  = numel(t.shape());
  const DType           dt = t.dtype();
  std::vector<uint32_t> out(n);
  if (dt == DType::FP32) {
    const auto *p = static_cast<const float *>(t.get_data());
    for (size_t i = 0; i < n; ++i) {
      uint32_t x;
      std::memcpy(&x, &p[i], 4);
      out[i] = x;
    }
  } else {
    const auto *p = static_cast<const uint16_t *>(t.get_data());
    for (size_t i = 0; i < n; ++i) {
      out[i] = p[i];
    }
  }
  return out;
}

/**
 * Owns the half-precision buffers behind the feeds. `load_from_buffer`
 * borrows rather than copies, and `load_from_vector` has no uint16_t dtype
 * mapping, so something has to keep them alive until the graph has run --
 * keep the arena in scope for at least that long.
 */
class FeedArena {
public:
  Tensor feed(const Shape &shape, DType dt, const std::vector<float> &vals,
    const std::string &name) {
    if (numel(shape) != vals.size()) {
      fail("feed '" + name + "': shape " + shape_str(shape) + " does not match "
        + std::to_string(vals.size()) + " values");
    }
    if (dt == DType::FP32) {
      return forch::op::load_from_vector(std::vector<float>(vals), shape)
        .set_name(name);
    }

    auto &held = buffers_.emplace_back(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
      held[i] = to_u16_one(vals[i], dt);
    }
    return forch::op::load_from_buffer(
      reinterpret_cast<forch::Buffer>(held.data()), shape, dt)
      .set_name(name);
  }

private:
  // deque, not vector: emplace_back must not invalidate earlier references.
  std::deque<std::vector<uint16_t>> buffers_;
};

// --------------------------------------------------------- test values

/**
 * A 0/1 value derived from the *global* row and column, so a wrong shard
 * offset shows up as a mismatch instead of being masked by a broadcast
 * pattern. `salt` separates independent tensors.
 *
 * 0/1 keeps every partial sum an exact small integer, which is what makes the
 * comparisons bit-exact rather than tolerance-based.
 */
inline float cell(int64_t row, int64_t col, int64_t salt) {
  uint64_t h = static_cast<uint64_t>(row) * 0x9E3779B97F4A7C15ull
    ^ static_cast<uint64_t>(col) * 0xC2B2AE3D27D4EB4Full
    ^ static_cast<uint64_t>(salt) * 0x165667B19E3779F9ull;
  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 32;
  return static_cast<float>(h & 1ull);
}

// ------------------------------------------------------------- tolerance

// An ulp-scale bound for the dtype, not a fitted tolerance: a wrong collective
// is off by 0.5-2.0, two orders of magnitude further away.
inline double tol_for(DType dt, double magnitude, int64_t world) {
  const double ulp = (dt == DType::FP32) ? magnitude * 1.2e-7
    : (dt == DType::FP16)                ? magnitude * 4.9e-4
                                         : magnitude * 3.9e-3;
  return ulp * (1.0 + static_cast<double>(world)) * 2.0 + 1e-5;
}

// NaN returns infinity: NaN compares false against everything, so a naive
// `d > worst` would report an overflowed result as max_err=0. Hit for real.
inline double max_abs_diff(
  const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size()) {
    fail("max_abs_diff: length mismatch " + std::to_string(a.size()) + " vs "
      + std::to_string(b.size()));
  }
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      return std::numeric_limits<double>::infinity();
    const double d =
      std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (std::isnan(d))
      return std::numeric_limits<double>::infinity();
    if (d > worst)
      worst = d;
  }
  return worst;
}

inline double max_magnitude(const std::vector<float> &v) {
  double m = 0.0;
  for (float x : v) {
    if (!std::isfinite(x))
      fail("non-finite reference magnitude");
    m = std::max(m, static_cast<double>(std::fabs(x)));
  }
  return m;
}

inline bool supports_fused_world(int64_t world) {
  return world == 2 || world == 4 || world == 8;
}

} // namespace mr
