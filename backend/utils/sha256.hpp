#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace finer::utils {

// SHA-256 (FIPS 180-4). The one-shot interface keeps mutable hash state local.
inline std::array<uint8_t, 32> sha256(std::span<const uint8_t> input) {
  if (input.size() > std::numeric_limits<uint64_t>::max() / 8)
    throw std::length_error(
      "SHA-256 input exceeds the 64-bit bit-length field");
  constexpr std::array<uint32_t, 64> constants = {0x428a2f98, 0x71374491,
    0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d,
    0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb,
    0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb,
    0xbef9a3f7, 0xc67178f2};
  std::array<uint32_t, 8> state    = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
       0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  auto                    compress = [&](const uint8_t *block) {
    std::array<uint32_t, 64> words{};
    for (size_t i = 0; i < 16; ++i)
      for (size_t j = 0; j < 4; ++j)
        words[i] = (words[i] << 8) | block[4 * i + j];
    for (size_t i = 16; i < words.size(); ++i) {
      const auto x = words[i - 15], y = words[i - 2];
      const auto s0 = std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3);
      const auto s1 = std::rotr(y, 17) ^ std::rotr(y, 19) ^ (y >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = state;
    for (size_t i = 0; i < words.size(); ++i) {
      const auto s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const auto choice = (e & f) ^ (~e & g);
      const auto t1 = h + s1 + choice + constants[i] + words[i];
      const auto s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto t2       = s0 + majority;
      h                   = g;
      g                   = f;
      f                   = e;
      e                   = d + t1;
      d                   = c;
      c                   = b;
      b                   = a;
      a                   = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  };
  const auto bit_length = uint64_t(input.size()) * 8;
  while (input.size() >= 64) {
    compress(input.data());
    input = input.subspan(64);
  }
  // At most two final blocks: 0x80, zeros, then the big-endian bit length.
  std::array<uint8_t, 128> tail{};
  std::copy(input.begin(), input.end(), tail.begin());
  tail[input.size()]       = 0x80;
  const size_t padded_size = input.size() < 56 ? 64 : 128;
  for (size_t i = 0; i < 8; ++i)
    tail[padded_size - 1 - i] = uint8_t(bit_length >> (8 * i));
  compress(tail.data());
  if (padded_size == 128)
    compress(tail.data() + 64);
  std::array<uint8_t, 32> digest{};
  for (size_t i = 0; i < digest.size(); ++i)
    digest[i] = uint8_t(state[i / 4] >> (24 - 8 * (i % 4)));
  return digest;
}

} // namespace finer::utils
