// 探针测试：fused RotaryPositionEmbedding 能否支撑 Qwen3.5 的 partial rotary
// (rotary_dim=64 < head_dim=256，等比缩小为 kRot=4 < kD=8)。
//
// 背景：全宽 D 的 half 配对 (i, i+D/2) 与 partial 的 (i, i+kRot/2) 结构不同，
// "cos/sin 补零到全宽"在数学上不可行——partial 只有两条路：
//   a. 组合：slice 前 kRot 维 → 全宽 kRot 的 fused rope → cat 透传（生产兜底）
//   b. kernel 原生接受窄 cos/sin（facade 已放宽为 cos D <= x
//   D，本测试探针判定）
//
// 用例：
//   1. baseline  —— 全宽 Half 模式 vs CPU 参考（环境 sanity，必过）
//   2. 组合兜底  —— slice → rope(kRot) → cat vs partial 参考（必过，生产兜底）
//   3. kernel 原生 —— 经正式注册路径喂窄表 [S,1,kRot]（信息性探针：
//                     REJECTED / MISMATCH 都不算失败，只打印结论；
//                     若 REJECTED，facade 的 <= D 放宽应收回到 == D）
//
// 纯公共 API，不触碰 forch-ge 内部头。
//
// 运行：cd build && ASCEND_RT_VISIBLE_DEVICES=0 ASCEND_WORK_PATH=ascend_work \
//        ./src/forch-ge/tests/forch-ge-test-rope_partial

#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/cat.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/rotary_position_embedding.hpp"
#include "forch/ops/slice.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace forch;

// 与生产 256/64 同比例的缩小尺寸
constexpr int64_t kS = 4; // seq
constexpr int64_t kN = 2; // heads
constexpr int64_t kD = 8; // head_dim
constexpr int64_t kRot = 4; // rotary_dim（前 kRot 维旋转，half 配对 (i, i+2)）

static bool almost_equal(
  const float *a, const float *b, size_t n, float tol = 2e-2F) {
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) > tol * (std::fabs(b[i]) + 1.0F)) {
      std::cerr << "  first mismatch at index " << i << ": got=" << a[i]
                << ", expected=" << b[i] << std::endl;
      return false;
    }
  }
  return true;
}

static std::vector<float> make_x() {
  std::vector<float> x(static_cast<size_t>(kS * kN * kD));
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>(i % 17) * 0.25F - 2.0F;
  }
  return x;
}

// 旋转区频率表：每步 half_rot 个非常规值（含负、非对称），避免对拍退化
static std::vector<float> make_unique_table(float seed) {
  constexpr size_t   kHalfRot = static_cast<size_t>(kRot / 2);
  std::vector<float> t(static_cast<size_t>(kS) * kHalfRot);
  for (size_t s = 0; s < static_cast<size_t>(kS); ++s) {
    for (size_t j = 0; j < kHalfRot; ++j) {
      t[s * kHalfRot + j] = std::cos(
        seed + static_cast<float>(s) * 0.7F + static_cast<float>(j) * 0.9F);
    }
  }
  return t;
}

// 把 unique 表扩成宽 W 的 rope 表（Half 语义：配对两半共用同一值，i % half
// 索引）
static std::vector<float> expand_table(
  const std::vector<float> &unique, int64_t width) {
  const size_t       half = static_cast<size_t>(width) / 2;
  std::vector<float> out(static_cast<size_t>(kS) * static_cast<size_t>(width));
  for (size_t s = 0; s < static_cast<size_t>(kS); ++s) {
    for (size_t i = 0; i < static_cast<size_t>(width); ++i) {
      out[s * static_cast<size_t>(width) + i] = unique[s * half + i % half];
    }
  }
  return out;
}

// 全宽 Half 模式参考（用例 1）：tables [S, kD]
static std::vector<float> rope_full_ref(const std::vector<float> &x,
  const std::vector<float> &cos_t, const std::vector<float> &sin_t) {
  std::vector<float> out(x.size(), 0.0F);
  for (size_t s = 0; s < static_cast<size_t>(kS); ++s) {
    for (size_t h = 0; h < static_cast<size_t>(kN); ++h) {
      const size_t base =
        (s * static_cast<size_t>(kN) + h) * static_cast<size_t>(kD);
      for (size_t i = 0; i < static_cast<size_t>(kD); ++i) {
        const size_t pair = i < kD / 2 ? i + kD / 2 : i - kD / 2;
        const float  rx   = i < kD / 2 ? -x[base + pair] : x[base + pair];
        out[base + i] =
          x[base + i] * cos_t[s * kD + i] + rx * sin_t[s * kD + i];
      }
    }
  }
  return out;
}

// partial 参考（用例 2/3 判据）：前 kRot 维旋转（配对 i±kRot/2，表宽 kRot），
// 其余维度精确透传
static std::vector<float> rope_partial_ref(const std::vector<float> &x,
  const std::vector<float> &cos_t, const std::vector<float> &sin_t) {
  constexpr size_t   kHalfRot = static_cast<size_t>(kRot / 2);
  std::vector<float> out(x.size(), 0.0F);
  for (size_t s = 0; s < static_cast<size_t>(kS); ++s) {
    for (size_t h = 0; h < static_cast<size_t>(kN); ++h) {
      const size_t base =
        (s * static_cast<size_t>(kN) + h) * static_cast<size_t>(kD);
      for (size_t i = 0; i < static_cast<size_t>(kD); ++i) {
        if (i < static_cast<size_t>(kRot)) {
          const size_t pair = i < kHalfRot ? i + kHalfRot : i - kHalfRot;
          const float  rx   = i < kHalfRot ? -x[base + pair] : x[base + pair];
          out[base + i] =
            x[base + i] * cos_t[s * kRot + i] + rx * sin_t[s * kRot + i];
        } else {
          out[base + i] = x[base + i];
        }
      }
    }
  }
  return out;
}

static TensorList feed(const std::string &name, std::vector<float> x,
  std::vector<float> cos, std::vector<float> sin, const Shape &table_shape) {
  // load_from_vector 只收右值；按值接参再 move，调用方保留自己的副本
  // （case2 在 feed 后还要读 x 做透传检查）。
  return {
    op::load_from_vector(std::move(x), {kS, kN, kD}).set_name(name + ".x"),
    op::load_from_vector(std::move(cos), table_shape).set_name(name + ".cos"),
    op::load_from_vector(std::move(sin), table_shape).set_name(name + ".sin")};
}

// ── 用例 1：facade 全宽 baseline ─────────────────────────────────────────
static int case_baseline(GEBackend &backend) {
  auto x   = make_x();
  auto uc  = make_unique_table(0.0F);
  auto us  = make_unique_table(1.3F);
  auto cf  = expand_table(uc, kD);
  auto sf  = expand_table(us, kD);
  auto exp = rope_full_ref(x, cf, sf);

  auto x_in   = op::create_input({kS, kN, kD}, DType::FP32).set_name("b.x");
  auto cos_in = op::create_input({kS, 1, kD}, DType::FP32).set_name("b.cos");
  auto sin_in = op::create_input({kS, 1, kD}, DType::FP32).set_name("b.sin");
  auto out    = op::rotary_position_embedding(
    op::RotaryPositionEmbedding::Mode::Half)(x_in, cos_in, sin_in)
               .set_name("b.out");

  auto graph = backend.build(forch::Graph({out}).set_name("rope_baseline"),
    {.precision_mode = GEBackend::GEBuildOptions::PrecisionMode::AutoToFP16});
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "case1: compile/load failed: " << e.what() << std::endl;
    return 1;
  }
  auto        outputs = graph.run(feed("b", x, cf, sf, {kS, 1, kD}));
  const auto *got =
    reinterpret_cast<const float *>(outputs.get("b.out").get_data());
  if (!almost_equal(got, exp.data(), exp.size())) {
    std::cerr << "case1 baseline: MISMATCH" << std::endl;
    return 1;
  }
  std::cout << "[case1] baseline full-width half rope: PASS" << std::endl;
  return 0;
}

// ── 用例 2：组合兜底 slice → rope(kRot) → cat（生产路径）─────────────────
// 对应 qwen3 的 fuse_rope=true 路径作用于 64 维 slice；eager 参考路径直接
// 复用 finer common-model.hpp::apply_rope_half，不在本测试范围。
static int case_compose(GEBackend &backend) {
  auto x   = make_x();
  auto uc  = make_unique_table(2.5F);
  auto us  = make_unique_table(0.8F);
  auto cr  = expand_table(uc, kRot); // [S, kRot]，配对结构已内建
  auto sr  = expand_table(us, kRot);
  auto exp = rope_partial_ref(x, cr, sr);

  auto x_in = op::create_input({kS, kN, kD}, DType::FP32).set_name("c.x");
  auto head = x_in
    | op::slice({op::Slice::Ellipsis{}, op::Slice::IntervalFromBegin{kRot}});
  auto tail =
    x_in | op::slice({op::Slice::Ellipsis{}, op::Slice::IntervalTillEnd{kRot}});
  auto cos_in = op::create_input({kS, 1, kRot}, DType::FP32).set_name("c.cos");
  auto sin_in = op::create_input({kS, 1, kRot}, DType::FP32).set_name("c.sin");
  auto rot    = op::rotary_position_embedding(
    op::RotaryPositionEmbedding::Mode::Half)(head, cos_in, sin_in);
  // cat 的参数是 dim 索引（0-based，见 common-model 的 size()-1 用法），
  // 3-D 张量沿末维拼接是 2，不是 3
  auto out = op::cat(2)(rot, tail).set_name("c.out");

  auto graph = backend.build(forch::Graph({out}).set_name("rope_compose"),
    {.precision_mode = GEBackend::GEBuildOptions::PrecisionMode::AutoToFP16});
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "case2: compile/load failed: " << e.what() << std::endl;
    return 1;
  }
  auto        outputs = graph.run(feed("c", x, cr, sr, {kS, 1, kRot}));
  const auto *got =
    reinterpret_cast<const float *>(outputs.get("c.out").get_data());
  if (!almost_equal(got, exp.data(), exp.size())) {
    std::cerr << "case2 compose fallback: MISMATCH" << std::endl;
    return 1;
  }
  // 非旋转区必须精确透传
  for (size_t s = 0; s < static_cast<size_t>(kS); ++s) {
    for (size_t h = 0; h < static_cast<size_t>(kN); ++h) {
      for (size_t i = static_cast<size_t>(kRot); i < static_cast<size_t>(kD);
           ++i) {
        const size_t idx =
          (s * static_cast<size_t>(kN) + h) * static_cast<size_t>(kD) + i;
        if (std::fabs(got[idx] - x[idx]) > 1e-5F) {
          std::cerr << "case2: passthrough dim " << i << " not exact"
                    << std::endl;
          return 1;
        }
      }
    }
  }
  std::cout << "[case2] slice→rope→cat fallback: PASS → 生产兜底路径可用"
            << std::endl;
  return 0;
}

// ── 用例 3：kernel 原生 partial 探针（信息性，不计失败）──────────────────
// 2026-09-10 探针结论：REJECTED —— AscendC tiling 的 JudgeLayoutByShape 要求
// cos/sin 与 x 可广播，窄表直接被拒。facade 保持 == D。
// 保留本用例：未来 CANN 升级后重跑，若变 ACCEPTED+MATCH 可再考虑窄表直用。
// 当前运行时本用例会在 facade shape() 处抛 REJECTED，属预期。
static void case_kernel_probe(GEBackend &backend) {
  auto x   = make_x();
  auto uc  = make_unique_table(0.0F);
  auto us  = make_unique_table(1.3F);
  auto cr  = expand_table(uc, kRot);
  auto sr  = expand_table(us, kRot);
  auto exp = rope_partial_ref(x, cr, sr);

  auto x_in   = op::create_input({kS, kN, kD}, DType::FP32).set_name("p.x");
  auto cos_in = op::create_input({kS, 1, kRot}, DType::FP32).set_name("p.cos");
  auto sin_in = op::create_input({kS, 1, kRot}, DType::FP32).set_name("p.sin");
  auto out    = op::rotary_position_embedding(
    op::RotaryPositionEmbedding::Mode::Half)(x_in, cos_in, sin_in)
               .set_name("p.out");

  try {
    auto graph = backend.build(forch::Graph({out}).set_name("rope_probe"),
      {.precision_mode = GEBackend::GEBuildOptions::PrecisionMode::AutoToFP16});
    graph.compile();
    graph.load();
    auto        outputs = graph.run(feed("p", x, cr, sr, {kS, 1, kRot}));
    const auto *got =
      reinterpret_cast<const float *>(outputs.get("p.out").get_data());
    if (almost_equal(got, exp.data(), exp.size())) {
      std::cout
        << "[case3] kernel probe: ACCEPTED + MATCH → kernel 原生支持 partial，"
           "保留 facade 的 <= D 放宽"
        << std::endl;
    } else {
      std::cout
        << "[case3] kernel probe: ACCEPTED but MISMATCH → kernel 接受窄表但"
           "语义不符：收回 facade 放宽（== D），走组合兜底"
        << std::endl;
    }
  } catch (const std::exception &e) {
    std::cout << "[case3] kernel probe: REJECTED (" << e.what()
              << ") → kernel 不支持窄 cos/sin：收回 facade 放宽（== D），"
                 "走组合兜底"
              << std::endl;
  }
}

int main() {
  GEBackend backend(AscendDevice::from_env());

  if (case_baseline(backend) != 0) {
    return 1;
  }
  if (case_compose(backend) != 0) {
    return 2;
  }
  case_kernel_probe(backend);

  std::cout << "rope_partial probe done." << std::endl;
  return 0;
}
