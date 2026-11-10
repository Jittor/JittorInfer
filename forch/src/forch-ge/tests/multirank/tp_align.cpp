// Two branches per case, in one graph: `tp` (sharded weights + collectives)
// and `rf` (full weights, no comms), compared element-wise. A host oracle
// could not catch a wrong assembly -- shard offset off by one, all_reduce vs
// all_gather, the fused op's y pre- or post-residual: it assumes the same
// thing.

// bf16 has 8 mantissa bits (ints exact to 256), fp16 has 11 (exact to 2048).
// Slicing K across ranks keeps each partial sum <= K/world and the reduced
// total <= K, so with 0/1 values and K <= 256 every intermediate is an exact
// integer in any summation order: --hidden-size 2048 is fp16 bit-exact, 256
// bf16.

#include "forch/backend/forch-ge.hpp"
#include "forch/distributed.hpp"
#include "forch/forch.hpp"
#include "forch/graph.hpp"
#include "forch/ops/collective.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/fused_collective_matmul.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/mat_mul.hpp"
#include "forch/ops/reshape.hpp"
#include "forch/ops/rms_norm.hpp"

#include "mr_common.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using forch::Backend;
using forch::DType;
using forch::ProcessGroup;
using forch::Shape;
using forch::Tensor;
using forch::TensorList;

using mr::Args;
using mr::dtype_name;
using mr::FeedArena;
using mr::max_abs_diff;
using mr::max_magnitude;
using mr::read_bits;
using mr::shape_str;
using mr::tol_for;

std::vector<float> tile(
  int64_t rows, int64_t cols, int64_t r0, int64_t c0, int64_t salt) {
  std::vector<float> v(static_cast<size_t>(rows * cols));
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      v[static_cast<size_t>(i * cols + j)] = mr::cell(r0 + i, c0 + j, salt);
    }
  }
  return v;
}

std::vector<float> gen_gamma(int64_t n, int64_t salt) {
  auto v = tile(1, n, 0, 0, salt);
  for (float &x : v) {
    x = 0.5F + 0.5F * x;
  }
  return v;
}

std::vector<float> tile_small(
  int64_t rows, int64_t cols, int64_t r0, int64_t c0, int64_t salt) {
  auto v = tile(rows, cols, r0, c0, salt);
  for (float &x : v) {
    x *= 1.0F / 1024.0F;
  }
  return v;
}

std::vector<double> host_mm(int64_t m, int64_t n, int64_t k, int64_t a_r0,
  int64_t a_c0, int64_t sa, int64_t b_r0, int64_t b_c0, int64_t sb) {
  std::vector<double> out(static_cast<size_t>(m * n), 0.0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (int64_t t = 0; t < k; ++t) {
        s += static_cast<double>(mr::cell(a_r0 + i, a_c0 + t, sa))
          * static_cast<double>(mr::cell(b_r0 + j, b_c0 + t, sb));
      }
      out[static_cast<size_t>(i * n + j)] = s;
    }
  }
  return out;
}

std::vector<double> host_mm_nn(int64_t m, int64_t n, int64_t k, int64_t a_r0,
  int64_t a_c0, int64_t sa, int64_t b_r0, int64_t b_c0, int64_t sb) {
  std::vector<double> out(static_cast<size_t>(m * n), 0.0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (int64_t t = 0; t < k; ++t) {
        s += static_cast<double>(mr::cell(a_r0 + i, a_c0 + t, sa))
          * static_cast<double>(mr::cell(b_r0 + t, b_c0 + j, sb));
      }
      out[static_cast<size_t>(i * n + j)] = s;
    }
  }
  return out;
}

constexpr int64_t S_X   = 1;
constexpr int64_t S_W   = 2;
constexpr int64_t S_RES = 3;
constexpr int64_t S_G   = 4;
constexpr int64_t S_WG  = 5;
constexpr int64_t S_WU  = 6;
constexpr int64_t S_WD  = 7;
constexpr int64_t S_G2  = 8;

// ------------------------------------------------------------- harness

float bits_to_f32(uint32_t b, DType dt) {
  if (dt == DType::FP32) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
  }
  return (dt == DType::BF16) ? mr::bf16_to_f32(static_cast<uint16_t>(b))
                             : mr::f16_to_f32(static_cast<uint16_t>(b));
}

std::vector<uint32_t> col_block(const std::vector<uint32_t> &ref, int64_t rows,
  int64_t cols, int64_t c0, int64_t cw) {
  std::vector<uint32_t> out(static_cast<size_t>(rows * cw));
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cw; ++j) {
      out[static_cast<size_t>(i * cw + j)] =
        ref[static_cast<size_t>(i * cols + c0 + j)];
    }
  }
  return out;
}

std::vector<uint32_t> row_block(
  const std::vector<uint32_t> &ref, int64_t cols, int64_t r0, int64_t rw) {
  const size_t begin = static_cast<size_t>(r0 * cols);
  return std::vector<uint32_t>(ref.begin() + static_cast<long>(begin),
    ref.begin() + static_cast<long>(begin + static_cast<size_t>(rw * cols)));
}

double exact_int_limit(DType dt) {
  switch (dt) {
  case DType::BF16:
    return 256.0;
  case DType::FP16:
    return 2048.0;
  default:
    return 16777216.0;
  }
}

struct Ctx {
  Backend     &backend;
  DType        dt;
  ProcessGroup group;
  int64_t      hidden;
  FeedArena   &arena;

  int64_t rank() const {
    return group.rank_id();
  }
  int64_t world() const {
    return group.world_size();
  }
};

template <class TP, class RF, class XF>
int align_case(const char *name, Ctx &ctx, const TensorList &feed, TP build_tp,
  RF build_rf, XF from_ref, int64_t tp_off = 0,
  const std::vector<double> *host = nullptr) {

  TensorList outs;
  try {
    outs.push_back(build_tp());
    outs.push_back(build_rf());
  } catch (const std::exception &e) {
    std::printf("  %-32s BUILD-FAIL   %s\n", name, e.what());
    return 1;
  }
  outs[0].set_name("tp");
  outs[1].set_name("rf");

  forch::Graph gir(outs);
  gir.set_name(name);

  auto graph = ctx.backend.build(gir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::printf("  %-32s COMPILE-FAIL %s\n", name, e.what());
    return 2;
  }

  try {
    auto   res = graph.run(feed);
    Tensor got = res.get("tp");
    Tensor ref = res.get("rf");
    if (got.shape() != outs[0].shape() || ref.shape() != outs[1].shape()
      || got.dtype() != ctx.dt || ref.dtype() != ctx.dt
      || outs[0].dtype() != ctx.dt || outs[1].dtype() != ctx.dt) {
      std::printf("  %-32s META-FAIL   dtype tp=%d rf=%d want=%d\n", name,
        static_cast<int>(got.dtype()), static_cast<int>(ref.dtype()),
        static_cast<int>(ctx.dt));
      return 3;
    }

    const auto got_bits = read_bits(got);
    const auto want     = from_ref(ref);

    if (tp_off < 0
      || tp_off + static_cast<int64_t>(want.size())
        > static_cast<int64_t>(got_bits.size())) {
      std::printf("  %-32s META-FAIL   tp=%s cannot cover rf's %zu elements "
                  "(off=%lld)\n",
        name, shape_str(got.shape()).c_str(), want.size(),
        static_cast<long long>(tp_off));
      return 3;
    }

    const double lim   = exact_int_limit(ctx.dt);
    bool         exact = true;
    for (uint32_t b : want) {
      const double v = bits_to_f32(b, ctx.dt);
      if (!std::isfinite(v))
        mr::fail("non-finite TP reference");
      if (v != std::floor(v) || std::fabs(v) > lim) {
        exact = false;
        break;
      }
    }

    if (exact) {
      size_t nbad = 0, first = 0;
      for (size_t i = 0; i < want.size(); ++i) {
        if (got_bits[static_cast<size_t>(tp_off) + i] != want[i]) {
          if (nbad == 0)
            first = i;
          ++nbad;
        }
      }
      if (nbad != 0) {
        const uint32_t g = got_bits[static_cast<size_t>(tp_off) + first];
        std::printf(
          "  %-32s VALUE-FAIL  %zu/%zu elements differ, first idx=%zu "
          "tp=0x%04x(%g) rf=0x%04x(%g)\n",
          name, nbad, want.size(), first, g,
          static_cast<double>(bits_to_f32(g, ctx.dt)), want[first],
          static_cast<double>(bits_to_f32(want[first], ctx.dt)));
        return 4;
      }
      if (host != nullptr) {
        if (host->size() != want.size()) {
          std::printf("  %-32s BAD-SPEC    host expects %zu != %zu\n", name,
            host->size(), want.size());
          return 1;
        }
        for (size_t i = 0; i < want.size(); ++i) {
          if (static_cast<double>(static_cast<float>((*host)[i]))
            != static_cast<double>(bits_to_f32(want[i], ctx.dt))) {
            std::printf("  %-32s HOST-FAIL    idx=%zu rf=%g host=%g "
                        "(both branches wrong)\n",
              name, i, static_cast<double>(bits_to_f32(want[i], ctx.dt)),
              (*host)[i]);
            return 5;
          }
        }
        std::printf(
          "  %-32s PASS(bit-exact + host)  n=%zu\n", name, want.size());
      } else {
        std::printf("  %-32s PASS(bit-exact)  n=%zu\n", name, want.size());
      }
      return 0;
    }

    std::vector<float> a(want.size()), b(want.size());
    for (size_t i = 0; i < want.size(); ++i) {
      a[i] = bits_to_f32(got_bits[static_cast<size_t>(tp_off) + i], ctx.dt);
      b[i] = bits_to_f32(want[i], ctx.dt);
    }
    const double mag   = max_magnitude(b);
    const double tol   = tol_for(ctx.dt, mag, ctx.world());
    const double worst = max_abs_diff(a, b);
    if (worst > tol) {
      std::printf("  %-32s VALUE-FAIL  max_err=%.6g tol=%.6g |ref|max=%g\n",
        name, worst, tol, mag);
      return 4;
    }

    std::printf("  %-32s PASS(tol)  max_err=%.3g tol=%.3g |ref|max=%.4g\n",
      name, worst, tol, mag);
    return 0;
  } catch (const std::exception &e) {
    std::printf("  %-32s RUN-FAIL    %s\n", name, e.what());
    return 6;
  }
}

// ---------------------------------------------------------- test cases

constexpr int64_t T = 16;

int case_col_parallel(Ctx &ctx) {
  const int64_t hd  = ctx.hidden;
  const int64_t nsh = hd / ctx.world();
  const int64_t c0  = ctx.rank() * nsh;
  const Shape   sx = {T, hd}, sc = {nsh, hd}, sf = {hd, hd};

  auto feed = TensorList{
    ctx.arena.feed(sx, ctx.dt, tile(T, hd, 0, 0, S_X), "x"),
    ctx.arena.feed(sc, ctx.dt, tile(nsh, hd, c0, 0, S_W), "wcol"),
    ctx.arena.feed(sf, ctx.dt, tile(hd, hd, 0, 0, S_W), "wfull"),
  };
  auto x  = forch::op::create_input(sx, ctx.dt).set_name("x");
  auto wc = forch::op::create_input(sc, ctx.dt).set_name("wcol");
  auto wf = forch::op::create_input(sf, ctx.dt).set_name("wfull");

  auto host = host_mm(T, nsh, hd, 0, 0, S_X, c0, 0, S_W);
  return align_case(
    "col_parallel_indexing", ctx, feed,
    [&] { return forch::op::mat_mul(false, true)(x, wc); },
    [&] { return forch::op::mat_mul(false, true)(x, wf); },
    [&](
      const Tensor &ref) { return col_block(read_bits(ref), T, hd, c0, nsh); },
    0, &host);
}

int case_row_parallel(Ctx &ctx) {
  const int64_t hd  = ctx.hidden;
  const int64_t ksh = hd / ctx.world();
  const int64_t k0  = ctx.rank() * ksh;
  const Shape   sxr = {T, ksh}, swr = {hd, ksh};
  const Shape   sx = {T, hd}, sw = {hd, hd};

  auto feed = TensorList{
    ctx.arena.feed(sxr, ctx.dt, tile(T, ksh, 0, k0, S_X), "xr"),
    ctx.arena.feed(swr, ctx.dt, tile(hd, ksh, 0, k0, S_W), "wrow"),
    ctx.arena.feed(sx, ctx.dt, tile(T, hd, 0, 0, S_X), "x"),
    ctx.arena.feed(sw, ctx.dt, tile(hd, hd, 0, 0, S_W), "wfull"),
  };
  auto xr = forch::op::create_input(sxr, ctx.dt).set_name("xr");
  auto wr = forch::op::create_input(swr, ctx.dt).set_name("wrow");
  auto x  = forch::op::create_input(sx, ctx.dt).set_name("x");
  auto wf = forch::op::create_input(sw, ctx.dt).set_name("wfull");

  auto host = host_mm(T, hd, hd, 0, 0, S_X, 0, 0, S_W);
  return align_case(
    "row_parallel_all_reduce", ctx, feed,
    [&] {
      return forch::op::all_reduce(
        forch::op::mat_mul(false, true)(xr, wr), ctx.group);
    },
    [&] { return forch::op::mat_mul(false, true)(x, wf); },
    [&](const Tensor &ref) { return read_bits(ref); }, 0, &host);
}

int case_row_parallel_fused(Ctx &ctx) {
  const int64_t hd  = ctx.hidden;
  const int64_t ksh = hd / ctx.world();
  const int64_t k0  = ctx.rank() * ksh;
  const Shape   sxr = {T, ksh}, swr = {hd, ksh};
  const Shape   sx = {T, hd}, sw = {hd, hd};

  auto feed = TensorList{
    ctx.arena.feed(sxr, ctx.dt, tile(T, ksh, 0, k0, S_X), "xr"),
    ctx.arena.feed(swr, ctx.dt, tile(hd, ksh, 0, k0, S_W), "wrow"),
    ctx.arena.feed(sx, ctx.dt, tile(T, hd, 0, 0, S_X), "x"),
    ctx.arena.feed(sw, ctx.dt, tile(hd, hd, 0, 0, S_W), "wfull"),
  };
  auto xr = forch::op::create_input(sxr, ctx.dt).set_name("xr");
  auto wr = forch::op::create_input(swr, ctx.dt).set_name("wrow");
  auto x  = forch::op::create_input(sx, ctx.dt).set_name("x");
  auto wf = forch::op::create_input(sw, ctx.dt).set_name("wfull");

  return align_case(
    "row_parallel_fused_all_reduce", ctx, feed,
    [&] { return forch::op::fused_matmul_all_reduce(xr, wr, ctx.group); },
    [&] { return forch::op::mat_mul(false, true)(x, wf); },
    [&](const Tensor &ref) { return read_bits(ref); });
}

int case_fused_add_rms_norm(Ctx &ctx) {
  const int64_t hd  = ctx.hidden;
  const int64_t ksh = hd / ctx.world();
  const int64_t k0  = ctx.rank() * ksh;
  const Shape   sxr = {T, ksh}, swr = {hd, ksh};
  const Shape   sx = {T, hd}, sw = {hd, hd};
  const Shape   sr = {1, T, hd};

  auto feed = TensorList{
    ctx.arena.feed(sxr, ctx.dt, tile(T, ksh, 0, k0, S_X), "xr"),
    ctx.arena.feed(swr, ctx.dt, tile(hd, ksh, 0, k0, S_W), "wrow"),
    ctx.arena.feed(sx, ctx.dt, tile(T, hd, 0, 0, S_X), "x"),
    ctx.arena.feed(sw, ctx.dt, tile(hd, hd, 0, 0, S_W), "wfull"),
    ctx.arena.feed(sr, ctx.dt, tile(T, hd, 0, 0, S_RES), "res"),
    ctx.arena.feed({hd}, ctx.dt, gen_gamma(hd, S_G), "gamma"),
  };
  auto xr = forch::op::create_input(sxr, ctx.dt).set_name("xr");
  auto wr = forch::op::create_input(swr, ctx.dt).set_name("wrow");
  auto x  = forch::op::create_input(sx, ctx.dt).set_name("x");
  auto wf = forch::op::create_input(sw, ctx.dt).set_name("wfull");
  auto rs = forch::op::create_input(sr, ctx.dt).set_name("res");
  auto ga = forch::op::create_input({hd}, ctx.dt).set_name("gamma");

  return align_case(
    "row_parallel_fused_add_rms_norm", ctx, feed,
    [&] {
      return forch::op::fused_matmul_all_reduce_add_rms_norm(
        xr, wr, rs, ga, ctx.group);
    },
    [&] {
      return forch::op::add(
        forch::op::reshape({T, hd})(forch::op::mat_mul(false, true)(x, wf)),
        forch::op::reshape({T, hd})(rs));
    },
    [&](const Tensor &ref) { return read_bits(ref); });
}

int case_vocab_parallel(Ctx &ctx) {
  const int64_t hd  = ctx.hidden;
  const int64_t vsh = hd / ctx.world();
  const int64_t v0  = ctx.rank() * vsh;
  const Shape   sh = {hd, T}, sv = {vsh, hd}, svf = {hd, hd};

  auto feed = TensorList{
    ctx.arena.feed(sh, ctx.dt, tile(hd, T, 0, 0, S_X), "h"),
    ctx.arena.feed(sv, ctx.dt, tile(vsh, hd, v0, 0, S_W), "wv"),
    ctx.arena.feed(svf, ctx.dt, tile(hd, hd, 0, 0, S_W), "wvfull"),
  };
  auto h  = forch::op::create_input(sh, ctx.dt).set_name("h");
  auto wv = forch::op::create_input(sv, ctx.dt).set_name("wv");
  auto wf = forch::op::create_input(svf, ctx.dt).set_name("wvfull");

  auto host = host_mm_nn(hd, T, hd, 0, 0, S_W, 0, 0, S_X);
  return align_case(
    "vocab_parallel_all_gather", ctx, feed,
    [&] {
      return forch::op::all_gather(forch::op::mat_mul()(wv, h), ctx.group);
    },
    [&] { return forch::op::mat_mul()(wf, h); },
    [&](const Tensor &ref) { return read_bits(ref); }, 0, &host);
}

int case_fused_all_gather_matmul(Ctx &ctx) {
  const int64_t hd  = ctx.hidden;
  const int64_t m0  = ctx.rank() * T;
  const Shape   sxr = {T, hd}, sw = {hd, hd};

  auto feed = TensorList{
    ctx.arena.feed(sxr, ctx.dt, tile(T, hd, m0, 0, S_X), "xr"),
    ctx.arena.feed(sw, ctx.dt, tile(hd, hd, 0, 0, S_W), "w"),
  };
  auto xr = forch::op::create_input(sxr, ctx.dt).set_name("xr");
  auto w  = forch::op::create_input(sw, ctx.dt).set_name("w");

  auto host = host_mm(T, hd, hd, m0, 0, S_X, 0, 0, S_W);
  return align_case(
    "fused_all_gather_matmul", ctx, feed,
    [&] { return forch::op::fused_all_gather_matmul(xr, w, ctx.group); },
    [&] { return forch::op::mat_mul(false, true)(xr, w); },
    [&](const Tensor &ref) { return read_bits(ref); }, m0 * hd, &host);
}

int case_fused_matmul_reduce_scatter(Ctx &ctx) {
  const int64_t hd  = ctx.hidden;
  const int64_t ksh = hd / ctx.world();
  const int64_t k0  = ctx.rank() * ksh;
  const int64_t rsh = T / ctx.world();
  const Shape   sxr = {T, ksh}, swr = {hd, ksh};
  const Shape   sx = {T, hd}, sw = {hd, hd};

  auto feed = TensorList{
    ctx.arena.feed(sxr, ctx.dt, tile(T, ksh, 0, k0, S_X), "xr"),
    ctx.arena.feed(swr, ctx.dt, tile(hd, ksh, 0, k0, S_W), "wrow"),
    ctx.arena.feed(sx, ctx.dt, tile(T, hd, 0, 0, S_X), "x"),
    ctx.arena.feed(sw, ctx.dt, tile(hd, hd, 0, 0, S_W), "wfull"),
  };
  auto xr = forch::op::create_input(sxr, ctx.dt).set_name("xr");
  auto wr = forch::op::create_input(swr, ctx.dt).set_name("wrow");
  auto x  = forch::op::create_input(sx, ctx.dt).set_name("x");
  auto wf = forch::op::create_input(sw, ctx.dt).set_name("wfull");

  return align_case(
    "fused_matmul_reduce_scatter", ctx, feed,
    [&] { return forch::op::fused_matmul_reduce_scatter(xr, wr, ctx.group); },
    [&] { return forch::op::mat_mul(false, true)(x, wf); },
    [&](const Tensor &ref) {
      return row_block(read_bits(ref), hd, ctx.rank() * rsh, rsh);
    });
}

int case_qwen3_layer(Ctx &ctx) {
  const int64_t   hd  = ctx.hidden;
  const int64_t   ksh = hd / ctx.world();
  const int64_t   k0  = ctx.rank() * ksh;
  const int64_t   nsh = hd / ctx.world();
  const int64_t   n0  = ctx.rank() * nsh;
  constexpr float EPS = 1e-6F;

  const Shape sxr = {T, ksh}, swr = {hd, ksh};
  const Shape sx = {T, hd}, sw = {hd, hd};
  const Shape sgn = {nsh, hd};
  const Shape sgf = {hd, hd};
  const Shape sr  = {1, T, hd};

  auto feed = TensorList{
    ctx.arena.feed(sxr, ctx.dt, tile(T, ksh, 0, k0, S_X), "h_r"),
    ctx.arena.feed(swr, ctx.dt, tile_small(hd, ksh, 0, k0, S_W), "wo_r"),
    ctx.arena.feed(sx, ctx.dt, tile(T, hd, 0, 0, S_X), "h"),
    ctx.arena.feed(sw, ctx.dt, tile_small(hd, hd, 0, 0, S_W), "wo"),
    ctx.arena.feed(sr, ctx.dt, tile(T, hd, 0, 0, S_RES), "res"),
    ctx.arena.feed({hd}, ctx.dt, gen_gamma(hd, S_G), "g1"),
    ctx.arena.feed({hd}, ctx.dt, gen_gamma(hd, S_G2), "g2"),
    ctx.arena.feed(sgn, ctx.dt, tile_small(nsh, hd, n0, 0, S_WG), "wg_r"),
    ctx.arena.feed(sgn, ctx.dt, tile_small(nsh, hd, n0, 0, S_WU), "wu_r"),
    ctx.arena.feed(sgf, ctx.dt, tile_small(hd, hd, 0, 0, S_WG), "wg"),
    ctx.arena.feed(sgf, ctx.dt, tile_small(hd, hd, 0, 0, S_WU), "wu"),
    ctx.arena.feed(swr, ctx.dt, tile_small(hd, ksh, 0, k0, S_WD), "wd_r"),
    ctx.arena.feed(sw, ctx.dt, tile_small(hd, hd, 0, 0, S_WD), "wd"),
  };

  auto h_r  = forch::op::create_input(sxr, ctx.dt).set_name("h_r");
  auto wo_r = forch::op::create_input(swr, ctx.dt).set_name("wo_r");
  auto h    = forch::op::create_input(sx, ctx.dt).set_name("h");
  auto wo   = forch::op::create_input(sw, ctx.dt).set_name("wo");
  auto rs   = forch::op::create_input(sr, ctx.dt).set_name("res");
  auto g1   = forch::op::create_input({hd}, ctx.dt).set_name("g1");
  auto g2   = forch::op::create_input({hd}, ctx.dt).set_name("g2");
  auto wgr  = forch::op::create_input(sgn, ctx.dt).set_name("wg_r");
  auto wur  = forch::op::create_input(sgn, ctx.dt).set_name("wu_r");
  auto wg   = forch::op::create_input(sgf, ctx.dt).set_name("wg");
  auto wu   = forch::op::create_input(sgf, ctx.dt).set_name("wu");
  auto wdr  = forch::op::create_input(swr, ctx.dt).set_name("wd_r");
  auto wd   = forch::op::create_input(sw, ctx.dt).set_name("wd");

  auto tp = [&] {
    auto y1_3 = forch::op::fused_matmul_all_reduce_add_rms_norm(
      h_r, wo_r, rs, g1, ctx.group);
    auto y1 = forch::op::reshape({T, hd})(y1_3);
    auto n1 = forch::op::rms_norm(EPS)(y1, g2);
    auto g  = forch::op::mat_mul(false, true)(n1, wgr);
    auto u  = forch::op::mat_mul(false, true)(n1, wur);
    auto a  = forch::op::mul(forch::op::silu(g), u);
    return forch::op::add(
      forch::op::fused_matmul_all_reduce(a, wdr, ctx.group), y1);
  };

  auto rf = [&] {
    auto y1 = forch::op::add(
      forch::op::mat_mul(false, true)(h, wo), forch::op::reshape({T, hd})(rs));
    auto n1 = forch::op::rms_norm(EPS)(y1, g2);
    auto g  = forch::op::mat_mul(false, true)(n1, wg);
    auto u  = forch::op::mat_mul(false, true)(n1, wu);
    auto a  = forch::op::mul(forch::op::silu(g), u);
    return forch::op::add(forch::op::mat_mul(false, true)(a, wd), y1);
  };

  return align_case("qwen3_layer", ctx, feed, tp, rf,
    [&](const Tensor &ref) { return read_bits(ref); });
}

using CaseFn = int (*)(Ctx &);

struct CaseEntry {
  const char *name;
  CaseFn      fn;
  bool        fused;
};

const CaseEntry kCases[] = {
  {"col_parallel_indexing", case_col_parallel, false},
  {"row_parallel_all_reduce", case_row_parallel, false},
  {"row_parallel_fused_all_reduce", case_row_parallel_fused, true},
  {"row_parallel_fused_add_rms_norm", case_fused_add_rms_norm, true},
  {"vocab_parallel_all_gather", case_vocab_parallel, false},
  {"fused_all_gather_matmul", case_fused_all_gather_matmul, true},
  {"fused_matmul_reduce_scatter", case_fused_matmul_reduce_scatter, true},
  {"qwen3_layer", case_qwen3_layer, true},
};

} // namespace

int main(int argc, char **argv) {
  const Args    args(argc, argv, {"dtype", "case", "hidden-size"});
  const DType   dt     = args.get_dtype("dtype", DType::FP16);
  const int64_t hidden = args.get_int("hidden-size", 2048);
  const auto    only   = args.get("case", "all");
  if (only != "all" && std::ranges::none_of(kCases, [&](const auto &entry) {
        return only == entry.name;
      })) {
    mr::fail("unknown --case '" + only + "'");
  }

  const ProcessGroup group = forch::process_group_from_env();

  std::printf("[rank %lld/%lld] dtype=%s hidden-size=%lld case=%s\n",
    static_cast<long long>(group.rank_id()),
    static_cast<long long>(group.world_size()), dtype_name(dt),
    static_cast<long long>(hidden), only.c_str());

  if (group.world_size() < 2) {
    // Both branches of every case need a real group: the tp branch sends
    // traffic and the rf branch exists only to be compared against it.
    std::printf("[rank 0] nothing to compare: world_size is 1. Launch with "
                "script/forch-ge-run.py --devices <a,b> to get a group.\n");
    return 0;
  }

  // Every case partitions this fixed token count. Rejecting hidden-size first
  // would turn a three-rank run into an error instead of a clean skip.
  if (T % group.world_size() != 0) {
    std::printf("[rank %lld] SKIP: T(%lld) is not divisible by world(%lld)\n",
      static_cast<long long>(group.rank_id()), static_cast<long long>(T),
      static_cast<long long>(group.world_size()));
    return 0;
  }

  if (hidden % group.world_size() != 0 || hidden < 256 || hidden % 8 != 0) {
    std::printf("[rank %lld] bad --hidden-size %lld: must be >=256, divisible "
                "by 8 and by world=%lld\n",
      static_cast<long long>(group.rank_id()), static_cast<long long>(hidden),
      static_cast<long long>(group.world_size()));
    return 10;
  }

  // Explicit Options rather than the default constructor, so the programmatic
  // path stays exercised; collectives.cpp covers the environment one.
  forch::GEBackend backend_impl(
    forch::AscendDevice::from_env(), forch::GEBackend::Options::from_env());
  Backend   backend = backend_impl;
  FeedArena arena;

  int total = 0, passed = 0, skipped = 0, failed = 0;

  for (const auto &c : kCases) {
    if (only != "all" && only != c.name)
      continue;
    ++total;

    if (c.fused) {
      if (!mr::supports_fused_world(group.world_size())) {
        std::printf("  %-32s SKIP        fused ops need world_size in "
                    "{2,4,8}\n",
          c.name);
        ++skipped;
        continue;
      }
      if (dt == DType::FP32) {
        std::printf(
          "  %-32s SKIP        fused ops support FP16/BF16 only\n", c.name);
        ++skipped;
        continue;
      }
    }
    Ctx       ctx{backend, dt, group, hidden, arena};
    const int rc = c.fn(ctx);
    if (rc == 0) {
      ++passed;
    } else {
      ++failed;
      std::printf("  ^^ %s failed, rc=%d\n", c.name, rc);
    }
  }

  std::printf("[rank %lld] total %d cases: %d PASS, %d SKIP, %d FAIL\n",
    static_cast<long long>(group.rank_id()), total, passed, skipped, failed);
  return failed == 0 ? 0 : 1;
}
