// Regression for the production cache update + TND prefill path. The oracle
// stores logical slots independently and computes causal GQA in double
// precision.
#include "../models/common/attention/attention.hpp"
#include <cmath>
#include <cstring>
#include <forch/backend/forch-ge.hpp>
#include <iostream>
#include <numeric>

using namespace forch;
static uint16_t bf(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
static float fp(uint16_t v) {
  uint32_t u = uint32_t(v) << 16;
  float    f;
  std::memcpy(&f, &u, 4);
  return f;
}
// Independent deterministic samples avoid near-zero averages hiding a bad
// block lookup in long-context tests.
static float sample(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return float(x & 0xffffU) / 16384.0f - 2.0f;
}
int main(int argc, char **argv) {
  try {
    GEBackend  backend(AscendDevice::from_env());
    const bool slot_chunk = argc > 1 && std::string(argv[1]) == "slot";
    const bool prefix     = argc > 1 && std::string(argv[1]) == "prefix";
    const bool paged =
      prefix || (argc > 1 && std::string(argv[argc - 1]) == "paged");
    const bool large =
      argc > 1 && !slot_chunk && !prefix && std::string(argv[1]) != "paged";
    const bool    long_context = large && std::string(argv[1]) == "long";
    const int64_t B            = large ? (long_context ? 16 : 64) : 3;
    const int64_t C            = large ? (long_context ? 4736 : 704) : 129;
    const int64_t T            = slot_chunk ? B * 8
                 : large                    ? (long_context ? 512 : 2048)
                                            : 32;
    const int64_t H = large ? 32 : 4, N = large ? 8 : 2, D = 128;
    // Production path: one SlotMap and metadata, one layer of pools.
    cache::SlotMapConfig map_cfg;
    map_cfg.slots                 = B;
    map_cfg.capacity              = C;
    map_cfg.block_size            = 128;
    map_cfg.paged                 = paged;
    map_cfg.num_pages             = B * ((C + 127) / 128);
    map_cfg.padding_tokens        = T;
    auto                      map = cache::make_slot_map(map_cfg);
    cache::AttentionMetadata  meta(*map, cache::AttentionVariant::Full);
    cache::FullAttentionCache kv(
      {&backend, map->storage_rows(), {N, D}, {N, D}, DType::BF16, "oracle"});
    kv.init();
    if (prefix)
      for (int64_t sid = 0; sid < B; ++sid)
        map->attach_prefix(sid, {});
    auto prefill_inputs = meta.inputs(T);
    // TND must not depend on fields used only by decode.
    if (!slot_chunk) {
      prefill_inputs.decode_kv_lengths.reset();
      prefill_inputs.decode_mask.reset();
      prefill_inputs.slot_block_table.reset();
    }
    const auto &decode_inputs = meta.inputs(B);
    auto        build         = [&](int64_t tq, bool packed) {
      auto        q = op::create_input({tq, H, D}, DType::BF16).set_name("q");
      auto        k = op::create_input({tq, N, D}, DType::BF16).set_name("k");
      auto        v = op::create_input({tq, N, D}, DType::BF16).set_name("v");
      const auto &in  = packed ? prefill_inputs : decode_inputs;
      auto [kc, vc]   = kv.update(k, v, in);
      const auto type = packed && !slot_chunk ? cache::AttentionType::TND
                       : paged ? cache::AttentionType::PagedBSND
                               : cache::AttentionType::BSND;
      auto       output = finer::models::full_attention(
        type, in, q, kc, vc, 1 / std::sqrt(float(D)));
      auto g = backend.build(
        Graph({output.set_name("out")}), {.enable_host_input_indexes = false});
      g.compile();
      g.load();
      return g;
    };
    auto                 prefill = build(T, true), decode = build(B, false);
    std::vector<int64_t> pos(B, 0);
    std::vector<std::vector<int64_t>> histories(B);
    std::vector<uint16_t> logical_k(B * C * N * D), logical_v(B * C * N * D);
    int                   step    = 0;
    double                worst   = 0;
    size_t                checked = 0;
    auto                  run     = [&](
                 std::vector<int64_t> count, bool packed, bool verify = true) {
      const int64_t         tq = packed ? T : B;
      std::vector<uint16_t> q(tq * H * D), k(tq * N * D), v(k.size());
      for (size_t i = 0; i < q.size(); ++i)
        q[i] = bf(large ? sample(uint32_t(i) + uint32_t(step) * 134217757U)
                        : .4f * std::sin(float(i * 7 + step * 13) * .11f));
      for (size_t i = 0; i < k.size(); ++i) {
        k[i] = bf(large
            ? sample(uint32_t(i) ^ 0xabc123U ^ uint32_t(step) * 256715023U)
            : .4f * std::cos(float(i * 3 + step * 19) * .13f));
        v[i] = bf(.7f * std::sin(float(i * 5 + step * 31) * .07f));
      }
      int64_t row = 0;
      for (int64_t s = 0; s < B; ++s)
        for (int64_t j = 0; j < count[s]; ++j, ++row)
          for (int64_t x = 0; x < N * D; ++x) {
            if (large)
              v[(packed ? (slot_chunk ? s * (T / B) + j : row) : s) * N * D
                + x] =
                bf(fp(v[(packed ? (slot_chunk ? s * (T / B) + j : row) : s) * N
                       * D
                     + x])
                  + .25f * s + .5f * (x / D) + .001f * (pos[s] + j));
            auto at = (s * C + pos[s] + j) * N * D + x;
            logical_k[at] =
              k[(packed ? (slot_chunk ? s * (T / B) + j : row) : s) * N * D
                + x];
            logical_v[at] =
              v[(packed ? (slot_chunk ? s * (T / B) + j : row) : s) * N * D
                + x];
          }
      TensorList inputs;
      inputs.push_back(
        op::load_from_buffer(Buffer(q.data()), {tq, H, D}, DType::BF16)
          .set_name("q"));
      inputs.push_back(
        op::load_from_buffer(Buffer(k.data()), {tq, N, D}, DType::BF16)
          .set_name("k"));
      inputs.push_back(
        op::load_from_buffer(Buffer(v.data()), {tq, N, D}, DType::BF16)
          .set_name("v"));
      const auto type = packed && !slot_chunk ? cache::AttentionType::TND
        : paged                               ? cache::AttentionType::PagedBSND
                                              : cache::AttentionType::BSND;
      const auto pack = packed && !slot_chunk ? cache::PackType::Compact
                                              : cache::PackType::SlotOrdered;
      if (!map->can_fit(count))
        throw std::runtime_error("oracle plan unexpectedly exceeds pool");
      map->reserve(count);
      meta.feed(count, tq, pack, type, inputs);
      kv.feed(inputs);
      auto outputs = (packed ? prefill : decode).run(inputs);
      auto out     = outputs.get("out");
      if (out.shape() != Shape{tq, H, D} || out.dtype() != DType::BF16)
        throw std::runtime_error("output metadata");
      auto data = reinterpret_cast<const uint16_t *>(out.get_data());
      // Check every output for finiteness, including discarded padding rows.
      for (int64_t i = 0; i < tq * H * D; ++i)
        if (!std::isfinite(fp(data[i])))
          throw std::runtime_error("nonfinite output");
      row = 0;
      for (int64_t s = 0; s < B; ++s)
        for (int64_t j = 0; j < count[s]; ++j, ++row) {
          if (!verify
            || (large && j != 0 && j != count[s] / 2 && j != count[s] - 1))
            continue;
          int64_t length = pos[s] + j + 1;
          for (int64_t h = 0; h < H; ++h) {
            if (large && h != 0 && h != H / 2 && h != H - 1)
              continue;
            std::vector<double> scores(length);
            double              sum = 0;
            for (int64_t l = 0; l < length; ++l) {
              double dot = 0;
              for (int64_t d = 0; d < D; ++d)
                dot +=
                  double(fp(
                    q[((packed ? (slot_chunk ? s * (T / B) + j : row) : s) * H
                        + h)
                        * D
                      + d]))
                  * fp(logical_k[((s * C + l) * N + h / (H / N)) * D + d]);
              scores[l] = std::exp(dot / std::sqrt(double(D)));
              sum += scores[l];
            }
            for (int64_t d = 0; d < D; ++d) {
              if (large && d % 17 != 0)
                continue;
              double expected = 0;
              for (int64_t l = 0; l < length; ++l)
                expected += scores[l]
                  * fp(logical_v[((s * C + l) * N + h / (H / N)) * D + d])
                  / sum;
              double
                actual = fp(
                  data[((packed ? (slot_chunk ? s * (T / B) + j : row) : s) * H
                         + h)
                      * D
                    + d]),
                err = std::abs(actual - expected);
              if (!std::isfinite(actual)
                || err > .006 + .01 * std::abs(expected))
                throw std::runtime_error("numerical mismatch step="
                  + std::to_string(step) + " err=" + std::to_string(err));
              worst = std::max(worst, err);
              ++checked;
            }
          }
        }
      {
        // Distinct token IDs; the prefix mode reuses them as a prompt.
        std::vector<std::vector<int64_t>> ids(B);
        cache::SlotTokens                 committed(B);
        for (int64_t s = 0; s < B; ++s) {
          for (int64_t j = 0; j < count[s]; ++j)
            ids[s].push_back(s * 1000000000LL + step * 1000000LL + j);
          histories[s].insert(histories[s].end(), ids[s].begin(), ids[s].end());
          committed[s] = ids[s];
        }
        map->commit(committed);
      }
      for (int64_t s = 0; s < B; ++s)
        pos[s] += count[s];
      ++step;
    };
    if (prefix) {
      for (int i = 0; i < 4; ++i)
        run({32, 0, 0}, true);
      auto prompt = histories[0];
      prompt.push_back(123456789);
      auto attach = [&](int sid) {
        if (map->attach_prefix(sid, prompt) != 128)
          throw std::runtime_error("expected a full shared prefix block");
        std::copy_n(
          logical_k.begin(), 128 * N * D, logical_k.begin() + sid * C * N * D);
        std::copy_n(
          logical_v.begin(), 128 * N * D, logical_v.begin() + sid * C * N * D);
        pos[sid] = 128;
      };
      attach(1);
      run({1, 1, 0}, false);
      map->reset_slot(0);
      map->reset_slot(1);
      pos[0] = pos[1] = 0;
      attach(2);
      run({0, 0, 1}, false);
    } else if (large) {
      const int rounds = long_context ? 128 : 3;
      for (int i = 0; i < rounds; ++i)
        run(std::vector<int64_t>(B, 32), true, i < 3 || i == rounds - 1);
      run(std::vector<int64_t>(B, 1), false);
      std::vector<int64_t> skew(B, 1);
      skew[0] = std::min<int64_t>({512, C - pos[0] - 2, T - B + 1});
      run(skew, true);
      run(std::vector<int64_t>(B, 1), false);
      map->reset_slot(0);
      pos[0]  = 0;
      skew[0] = 20;
      run(skew, true);
      std::vector<int64_t> holes(B, 0);
      for (int64_t sid = 1; sid < B; sid += 2)
        holes[sid] = 1;
      run(holes, false);
    } else if (slot_chunk) {
      run({0, 0, 1}, false);
      run({8, 3, 0}, true);
      run({1, 0, 7}, true);
      run({1, 1, 1}, false);
      for (int i = 0; i < 14; ++i)
        run({8, 1, 1}, true);
      run({7, 0, 0}, true); // slot 0 reaches C exactly with a partial chunk
      map->reset_slot(0);
      pos[0] = 0;
      run({2, 0, 8}, true);
      run({1, 0, 1}, false);
    } else {
      run({0, 0, 1}, false); // first decode with holes and empty cache slots
      run({27, 1, 0}, true);
      run({0, 0, 29}, true);
      run({3, 7, 1}, true);
      run({1, 1, 1}, false);
      run({32, 0, 0}, true);
      run({32, 0, 0}, true);
      run({32, 0, 0}, true);
      run({2, 1, 0}, true); // slot 0 reaches C exactly while padding is written
      run({0, 1, 1}, false);
      run({0, 0, 1}, false); // only the last slot is active
      map->reset_slot(0);
      pos[0] = 0;
      run({5, 1, 0}, true);
      run({1, 1, 1}, false);
    }
    bool rejected = false;
    try {
      TensorList           unused;
      std::vector<int64_t> invalid(B, 0);
      invalid[0] = C;
      meta.feed(invalid, T, cache::PackType::Compact, cache::AttentionType::TND,
        unused);
    } catch (const std::exception &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("overflow accepted");
    std::cout << "PASS steps=" << step << " values=" << checked
              << " max_abs=" << worst << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
