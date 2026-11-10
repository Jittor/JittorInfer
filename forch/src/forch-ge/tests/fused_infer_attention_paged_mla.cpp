#include "forch/backend/forch-ge.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/fused_infer_attention.hpp"
#include "forch/ops/input.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace forch;
using FIA = op::FusedInferAttention;
static uint16_t bf(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}
static double fp(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16;
  float    result;
  std::memcpy(&result, &bits, 4);
  return result;
}
static float sample(uint32_t index) {
  index ^= index >> 16;
  index *= 0x7feb352dU;
  index ^= index >> 15;
  index *= 0x846ca68bU;
  index ^= index >> 16;
  return float(index & 65535) / 65536 - .5f;
}
int main(int argc, char **argv) {
  try {
    const int64_t     B       = argc > 1 ? std::stoll(argv[1]) : 2;
    const int64_t     H       = argc > 2 ? std::stoll(argv[2]) : 32;
    const int64_t     max_len = argc > 3 ? std::stoll(argv[3]) : 257;
    constexpr int64_t D = 512, R = 64, P = 128;
    if (B <= 0 || H <= 0 || max_len < 257)
      throw std::invalid_argument("need B/H > 0 and max_len >= 257");
    const int64_t C      = (max_len + P - 1) / P;
    const int64_t blocks = B * C + 1;
    auto          q = op::create_input({B, 1, H, D}, DType::BF16).set_name("q");
    auto k  = op::create_input({blocks, P, D}, DType::BF16).set_name("latent");
    auto qr = op::create_input({B, 1, H, R}, DType::BF16).set_name("qr");
    auto kr = op::create_input({blocks, P, R}, DType::BF16).set_name("kr");
    FIA::OptionalInputs extra;
    extra.query_rope  = qr;
    extra.key_rope    = kr;
    extra.block_table = op::create_input({B, C}, DType::INT32).set_name("bt");
    extra.actual_seq_lengths_kv =
      op::create_input({B}, DType::INT64, true).set_name("lengths");
    FIA::Options opt(H);
    opt.input_layout        = FIA::InputLayout::BSND;
    opt.num_key_value_heads = 1;
    opt.block_size          = P;
    opt.scale               = 1.f / std::sqrt(float(D + R));
    opt.inner_precise       = 2;
    auto out = op::fused_infer_attention(opt)(q, k, k, extra).set_name("out");
    if (out.shape() != q.shape() || out.dtype() != DType::BF16)
      throw std::runtime_error("invalid declared output metadata");
    int  rejected = 0;
    auto reject   = [&](auto fn) {
      try {
        fn();
      } catch (const std::exception &) {
        ++rejected;
        return;
      }
      throw std::runtime_error("invalid MLA metadata accepted");
    };
    reject([&] {
      auto x = extra;
      x.key_rope.reset();
      (void)op::fused_infer_attention(opt)(q, k, k, x);
    });
    reject([&] {
      auto x       = extra;
      x.query_rope = op::create_input({B, 1, H - 1, R}, DType::BF16);
      (void)op::fused_infer_attention(opt)(q, k, k, x);
    });
    reject([&] {
      auto x     = extra;
      x.key_rope = op::create_input({blocks, P, R / 2}, DType::BF16);
      (void)op::fused_infer_attention(opt)(q, k, k, x);
    });
    GEBackend backend(AscendDevice::from_env());
    auto      graph =
      backend.build(Graph({out}), {.enable_host_input_indexes = false});
    graph.compile();
    graph.load();
    std::vector<uint16_t> qv(B * H * D), qrv(B * H * R);
    for (size_t i = 0; i < qv.size(); ++i)
      qv[i] = bf(sample(i + 1));
    for (size_t i = 0; i < qrv.size(); ++i)
      qrv[i] = bf(4 * sample(i + 12345));
    double  worst       = 0;
    double  rope_effect = 0;
    int64_t checked     = 0;
    for (int round = 0; round < 3; ++round) {
      std::vector<int32_t>  table(B * C);
      std::vector<int64_t>  lengths(B);
      std::vector<uint16_t> kv(blocks * P * D, bf(32)),
        krv(blocks * P * R, bf(16));
      const int64_t sizes[] = {1, 127, 128, 129, max_len - 1, max_len};
      for (int64_t b = 0; b < B; ++b) {
        lengths[b] = sizes[(b + 2 * round + 3) % 6];
        for (int64_t c = 0; c < C; ++c)
          table[b * C + c] = round % 2 ? 1 + b * C + c : blocks - 1 - b * C - c;
        for (int64_t t = 0; t < lengths[b]; ++t) {
          auto row = table[b * C + t / P] * P + t % P;
          for (int64_t d = 0; d < D; ++d)
            kv[row * D + d] = bf(sample((b * C * P + t) * D + d + 333));
          for (int64_t r = 0; r < R; ++r)
            krv[row * R + r] = bf(4 * sample((b * C * P + t) * R + r + 777));
        }
      }
      auto result =
        graph
          .run({op::load_from_buffer(Buffer(qv.data()), q.shape(), DType::BF16)
                  .set_name("q"),
            op::load_from_buffer(Buffer(kv.data()), k.shape(), DType::BF16)
              .set_name("latent"),
            op::load_from_buffer(Buffer(qrv.data()), qr.shape(), DType::BF16)
              .set_name("qr"),
            op::load_from_buffer(Buffer(krv.data()), kr.shape(), DType::BF16)
              .set_name("kr"),
            op::load_from_vector(std::vector<int32_t>(table), {B, C})
              .set_name("bt"),
            op::load_from_vector(std::vector<int64_t>(lengths), {B})
              .set_name("lengths")})
          .get("out");
      if (result.shape() != q.shape() || result.dtype() != DType::BF16)
        throw std::runtime_error("invalid runtime output metadata");
      const auto *actual = static_cast<const uint16_t *>(result.get_data());
      for (int64_t b = 0; b < B; ++b)
        for (int64_t h = 0; h < H; ++h) {
          std::vector<double> scores(lengths[b]), without_rope(lengths[b]);
          for (int64_t t = 0; t < lengths[b]; ++t) {
            auto row = table[b * C + t / P] * P + t % P;
            for (int64_t d = 0; d < D; ++d)
              scores[t] += fp(qv[(b * H + h) * D + d]) * fp(kv[row * D + d]);
            without_rope[t] = scores[t] * opt.scale;
            for (int64_t r = 0; r < R; ++r)
              scores[t] += fp(qrv[(b * H + h) * R + r]) * fp(krv[row * R + r]);
            scores[t] *= opt.scale;
          }
          auto   max   = *std::max_element(scores.begin(), scores.end());
          double denom = 0;
          for (auto &score : scores) {
            score = std::exp(score - max);
            denom += score;
          }
          const double other_max =
            *std::max_element(without_rope.begin(), without_rope.end());
          double other_denom = 0;
          for (auto &score : without_rope) {
            score = std::exp(score - other_max);
            other_denom += score;
          }
          for (int64_t d = 0; d < D; ++d) {
            double expected = 0, other = 0;
            for (int64_t t = 0; t < lengths[b]; ++t) {
              auto row = table[b * C + t / P] * P + t % P;
              expected += scores[t] / denom * fp(kv[row * D + d]);
              other += without_rope[t] / other_denom * fp(kv[row * D + d]);
            }
            rope_effect  = std::max(rope_effect, std::abs(expected - other));
            double value = fp(actual[(b * H + h) * D + d]);
            double error = std::abs(value - expected);
            worst        = std::max(worst, error);
            if (!std::isfinite(value) || error > .004) {
              std::cerr << "mismatch round=" << round << " b=" << b
                        << " h=" << h << " d=" << d << " actual=" << value
                        << " expected=" << expected << '\n';
              return 1;
            }
            ++checked;
          }
        }
    }
    if (rope_effect < .008)
      throw std::runtime_error("reference is insensitive to RoPE");
    std::cout << "PASS paged MLA B=" << B << " H=" << H
              << " max_len=" << max_len << " rope_effect=" << rope_effect
              << " values=" << checked << " max_abs=" << worst
              << " rejected=" << rejected << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
