#include "forch/backend/forch-ge.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/fused_infer_attention.hpp"
#include "forch/ops/input.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace forch;
using FIA = op::FusedInferAttention;
static uint16_t bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u >> 16;
}
static float fp(uint16_t v) {
  uint32_t u = uint32_t(v) << 16;
  float    f;
  std::memcpy(&f, &u, 4);
  return f;
}
int main() {
  try {
    constexpr int64_t T = 8, H = 4, N = 2, D = 128, P = 128, B = 4, Blocks = 8;
    FIA::Options      options(H);
    options.input_layout        = FIA::InputLayout::TND;
    options.num_key_value_heads = N;
    options.block_size          = P;
    options.sparse_mode         = 3;
    auto q = op::create_input({T, H, D}, DType::BF16).set_name("q");
    auto k = op::create_input({Blocks, P, N * D}, DType::BF16).set_name("k");
    auto v = op::create_input({Blocks, P, N * D}, DType::BF16).set_name("v");
    FIA::OptionalInputs extra;
    extra.actual_seq_lengths =
      op::create_input({B}, DType::INT64, true).set_name("aq");
    extra.actual_seq_lengths_kv =
      op::create_input({B}, DType::INT64, true).set_name("ak");
    std::vector<int32_t> table{7, 6, 5, 4, 3, 2, 1, 0};
    extra.block_table =
      op::load_from_vector(std::vector<int32_t>(table), {B, 2});
    std::vector<int8_t> mask(2048 * 2048, 0);
    for (int i = 0; i < 2048; ++i)
      for (int j = i + 1; j < 2048; ++j)
        mask[i * 2048 + j] = 1;
    extra.atten_mask = op::load_from_vector(std::move(mask), {2048, 2048});
    auto out =
      op::fused_infer_attention(options)(q, k, v, extra).set_name("out");
    if (out.shape() != Shape{T, H, D} || out.dtype() != DType::BF16)
      throw std::runtime_error("metadata");
    int  rejected = 0;
    auto reject   = [&](auto fn) {
      bool caught = false;
      try {
        fn();
      } catch (const std::exception &) {
        caught = true;
      }
      if (!caught)
        throw std::runtime_error("invalid input accepted");
      ++rejected;
    };
    reject([&] {
      auto x       = extra;
      x.query_rope = op::create_input({T, H, 64}, DType::BF16);
      (void)op::fused_infer_attention(options)(q, k, v, x).shape();
    });
    reject([&] {
      auto x       = extra;
      x.query_rope = op::create_input({T, H, 64}, DType::BF16);
      x.key_rope   = op::create_input({Blocks, P, N * 32}, DType::BF16);
      (void)op::fused_infer_attention(options)(q, k, v, x).shape();
    });
    reject([&] {
      auto x = extra;
      x.block_table.reset();
      (void)op::fused_infer_attention(options)(q, k, v, x).shape();
    });
    reject([&] {
      auto x               = extra;
      x.actual_seq_lengths = op::create_input({B}, DType::INT32);
      (void)op::fused_infer_attention(options)(q, k, v, x).shape();
    });
    reject([&] {
      auto x       = options;
      x.block_size = 64;
      (void)op::fused_infer_attention(x)(q, k, v, extra).shape();
    });
    reject([&] {
      (void)op::fused_infer_attention(options)(
        q, k, op::create_input({Blocks, P, N * D}, DType::FP16), extra)
        .shape();
    });
    GEBackend backend(AscendDevice::from_env());
    auto      graph =
      backend.build(Graph({out}), {.enable_host_input_indexes = false});
    graph.compile();
    graph.load();
    std::vector<uint16_t> qd(T * H * D, 0), kd(Blocks * P * N * D, 0),
      vd(kd.size());
    for (int s = 0; s < B; ++s)
      for (int l = 0; l < 256; ++l)
        for (int h = 0; h < N; ++h)
          for (int d = 0; d < D; ++d)
            vd[((table[s * 2 + l / P] * P + l % P) * N + h) * D + d] =
              bf(float(l % 31 + s * 3 + h) / 64);
    double worst   = 0;
    int    checked = 0;
    for (int round = 0; round < 2; ++round) {
      std::vector<int64_t> aq     = round ? std::vector<int64_t>{1, 3, 8, 8}
                                          : std::vector<int64_t>{5, 5, 6, 8};
      std::vector<int64_t> ak     = round ? std::vector<int64_t>{127, 130, 9, 0}
                                          : std::vector<int64_t>{7, 0, 129, 2};
      auto                 result = graph.run({op::load_from_buffer(
                                 Buffer(qd.data()), {T, H, D}, DType::BF16)
                                                 .set_name("q"),
                        op::load_from_buffer(Buffer(kd.data()), {Blocks, P, N * D}, DType::BF16)
                          .set_name("k"),
                        op::load_from_buffer(Buffer(vd.data()), {Blocks, P, N * D}, DType::BF16)
                          .set_name("v"),
                        op::load_from_vector(std::vector<int64_t>(aq), {B}).set_name("aq"),
                        op::load_from_vector(std::vector<int64_t>(ak), {B}).set_name("ak")});
      auto                 tensor = result.get("out");
      if (tensor.shape() != Shape{T, H, D})
        throw std::runtime_error("runtime shape");
      auto    data  = reinterpret_cast<const uint16_t *>(tensor.get_data());
      int64_t start = 0;
      for (int s = 0; s < B; ++s) {
        auto count = aq[s] - start;
        for (int j = 0; j < count; ++j)
          for (int h = 0; h < H; ++h) {
            auto   length   = ak[s] - count + j + 1;
            double expected = 0;
            for (int l = 0; l < length; ++l)
              expected += double(l % 31 + s * 3 + h / (H / N)) / 64 / length;
            for (int d = 0; d < D; ++d) {
              double actual = fp(data[((start + j) * H + h) * D + d]);
              double error  = std::abs(actual - expected);
              if (!std::isfinite(actual) || error > .003)
                throw std::runtime_error("value mismatch");
              worst = std::max(worst, error);
              ++checked;
            }
          }
        start = aq[s];
      }
    }
    std::cout << "PASS values=" << checked << " max_abs=" << worst
              << " rejected=" << rejected << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
