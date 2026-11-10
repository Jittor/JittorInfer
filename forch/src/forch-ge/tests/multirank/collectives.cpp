// Per-op correctness for the collective and fused-collective ops: each case
// feeds rank-dependent data, runs one graph, and compares against a host
// reference computed from what every rank contributed.
//
// Rank, world size, device and rank table come from the environment the
// launcher exports; only --dtype and --case are read here.

#include "forch/backend/forch-ge.hpp"
#include "forch/distributed.hpp"
#include "forch/forch.hpp"
#include "forch/graph.hpp"
#include "forch/ops/collective.hpp"
#include "forch/ops/fused_collective_matmul.hpp"
#include "forch/ops/input.hpp"

#include "mr_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
using forch::op::ReduceOp;

using mr::Args;
using mr::dtype_name;
using mr::FeedArena;
using mr::numel;
using mr::read_back;
using mr::shape_str;
using mr::tol_for;

// An empty expected entry means that output is not checked (non-root ranks).
template <class F>
int run_case(const char *name, Backend &backend, DType dt, int64_t world,
  const TensorList &feed, F build,
  const std::vector<std::vector<float>> &expected) {
  TensorList outs;
  try {
    outs = build();
  } catch (const std::exception &e) {
    std::printf("  %-37s BUILD-FAIL  %s\n", name, e.what());
    return 1;
  }
  if (outs.size() != expected.size()) {
    std::printf("  %-37s BAD-SPEC    %zu outputs != %zu expected\n", name,
      outs.size(), expected.size());
    return 1;
  }

  std::vector<std::string> names;
  names.reserve(outs.size());
  for (size_t i = 0; i < outs.size(); ++i) {
    names.push_back(std::string(name) + "_out" + std::to_string(i));
    outs[i].set_name(names[i]);
  }

  forch::Graph gir(outs);
  gir.set_name(name);

  auto graph = backend.build(gir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::printf("  %-37s COMPILE-FAIL %s\n", name, e.what());
    return 2;
  }

  try {
    auto res = graph.run(feed);

    int    bad       = 0;
    double worst_all = 0.0;
    for (size_t i = 0; i < names.size(); ++i) {
      if (expected[i].empty())
        continue;

      Tensor       t = res.get(names[i]);
      const size_t n = numel(t.shape());
      if (t.shape() != outs[i].shape() || n != expected[i].size()
        || t.dtype() != dt || outs[i].dtype() != dt) {
        std::printf("  %-37s META-FAIL   out%zu shape=%s dtype=%d\n", name, i,
          shape_str(t.shape()).c_str(), static_cast<int>(t.dtype()));
        bad = 1;
        continue;
      }

      auto   got  = read_back(t.get_data(), n, dt);
      double maxv = 0.0;
      for (float v : expected[i]) {
        if (!std::isfinite(v))
          mr::fail("non-finite collective reference");
        maxv = std::max(maxv, static_cast<double>(std::fabs(v)));
      }
      const double tol = tol_for(dt, maxv, world);

      double worst = 0.0;
      size_t at    = 0;
      for (size_t k = 0; k < n; ++k) {
        if (!std::isfinite(got[k])) {
          worst = std::numeric_limits<double>::infinity();
          at    = k;
          break;
        }
        const double d = std::fabs(
          static_cast<double>(got[k]) - static_cast<double>(expected[i][k]));
        if (d > worst) {
          worst = d;
          at    = k;
        }
      }
      worst_all = std::max(worst_all, worst);
      if (worst > tol) {
        std::printf("  %-37s VALUE-FAIL  out%zu idx=%zu got=%g want=%g "
                    "max_err=%g tol=%g\n",
          name, i, at, static_cast<double>(got[at]), expected[i][at], worst,
          tol);
        bad = 1;
      }
    }
    if (bad != 0)
      return 3;
    std::printf("  %-37s PASS        max_err=%.3g\n", name, worst_all);
    return 0;
  } catch (const std::exception &e) {
    std::printf("  %-37s RUN-FAIL    %s\n", name, e.what());
    return 4;
  }
}

// Values are exactly representable in the target dtype (multiples of 0.5
// summing to <= 16; 0/1 for matmul), so max_err == 0 under fp16 is a derived
// bound rather than a tuned one.
float base_val(int64_t i, int64_t rank) {
  return 0.5F * static_cast<float>(1 + ((i + rank) % 4));
}

std::vector<float> gen(const Shape &shape, int64_t rank, int64_t offset) {
  std::vector<float> v(numel(shape));
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = base_val(static_cast<int64_t>(i), rank + offset);
  }
  return v;
}

std::vector<float> gen_mm(const Shape &shape, int64_t salt) {
  const int64_t      cols = shape.back();
  std::vector<float> v(numel(shape));
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = mr::cell(
      static_cast<int64_t>(i) / cols, static_cast<int64_t>(i) % cols, salt);
  }
  return v;
}

std::vector<double> ref_mm(const std::vector<float> &a, int64_t m, int64_t k,
  const std::vector<float> &b, int64_t n) {
  std::vector<double> out(static_cast<size_t>(m * n), 0.0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (int64_t t = 0; t < k; ++t) {
        s += static_cast<double>(a[static_cast<size_t>(i * k + t)])
          * static_cast<double>(b[static_cast<size_t>(j * k + t)]);
      }
      out[static_cast<size_t>(i * n + j)] = s;
    }
  }
  return out;
}

std::vector<float> to_f(const std::vector<double> &d) {
  std::vector<float> v(d.size());
  for (size_t i = 0; i < d.size(); ++i) {
    v[i] = static_cast<float>(d[i]);
  }
  return v;
}

// Tiny shapes are deliberate: a collective costs a fixed ~130us per call
// regardless of shape, so a small one answers the same wiring question.
constexpr int64_t R = 2;
constexpr int64_t C = 4;

// FusedAllGatherMatmul's k-axis must be >= 256 ("The k-axis should be in range
// [256, 65535)"); the other fused ops have no such floor.
constexpr int64_t MM_M = 32;
constexpr int64_t MM_K = 256;
constexpr int64_t MM_N = 32;

struct Ctx {
  Backend     &backend;
  DType        dt;
  ProcessGroup group;
  FeedArena   &arena;

  int64_t rank() const {
    return group.rank_id();
  }
  int64_t world() const {
    return group.world_size();
  }
};

int case_all_reduce(Ctx &ctx) {
  const Shape in    = {R, C};
  auto        vals  = gen(in, ctx.rank(), 0);
  auto        feed  = TensorList{ctx.arena.feed(in, ctx.dt, vals, "x")};
  auto        in_ph = forch::op::create_input(in, ctx.dt).set_name("x");

  std::vector<float> want(vals.size());
  for (size_t i = 0; i < want.size(); ++i) {
    double s = 0.0;
    for (int64_t q = 0; q < ctx.world(); ++q)
      s += base_val(static_cast<int64_t>(i), q);
    want[i] = static_cast<float>(s);
  }

  return run_case("all_reduce", ctx.backend, ctx.dt, ctx.world(), feed,
    [&] { return TensorList{forch::op::all_reduce(in_ph, ctx.group)}; },
    {want});
}

int case_all_gather(Ctx &ctx) {
  const Shape in    = {R, C};
  auto        vals  = gen(in, ctx.rank(), 0);
  auto        feed  = TensorList{ctx.arena.feed(in, ctx.dt, vals, "x")};
  auto        in_ph = forch::op::create_input(in, ctx.dt).set_name("x");

  const Shape        out = {R * ctx.world(), C};
  std::vector<float> want(numel(out));
  for (int64_t i = 0; i < R * ctx.world(); ++i) {
    const int64_t src_rank = i / R;
    const int64_t src_row  = i % R;
    for (int64_t j = 0; j < C; ++j) {
      want[static_cast<size_t>(i * C + j)] =
        base_val(src_row * C + j, src_rank);
    }
  }

  return run_case("all_gather", ctx.backend, ctx.dt, ctx.world(), feed,
    [&] { return TensorList{forch::op::all_gather(in_ph, ctx.group)}; },
    {want});
}

int case_reduce_scatter(Ctx &ctx) {
  const Shape in    = {R * ctx.world(), C};
  auto        vals  = gen(in, ctx.rank(), 0);
  auto        feed  = TensorList{ctx.arena.feed(in, ctx.dt, vals, "x")};
  auto        in_ph = forch::op::create_input(in, ctx.dt).set_name("x");

  const Shape        out = {R, C};
  std::vector<float> want(numel(out));
  for (int64_t i = 0; i < R; ++i) {
    for (int64_t j = 0; j < C; ++j) {
      const int64_t grow = ctx.rank() * R + i;
      double        s    = 0.0;
      for (int64_t q = 0; q < ctx.world(); ++q) {
        s += base_val(grow * C + j, q);
      }
      want[static_cast<size_t>(i * C + j)] = static_cast<float>(s);
    }
  }

  return run_case("reduce_scatter", ctx.backend, ctx.dt, ctx.world(), feed,
    [&] { return TensorList{forch::op::reduce_scatter(in_ph, ctx.group)}; },
    {want});
}

int case_broadcast(Ctx &ctx) {
  const Shape in    = {R, C};
  auto        vals  = gen(in, ctx.rank(), 0);
  auto        feed  = TensorList{ctx.arena.feed(in, ctx.dt, vals, "x")};
  auto        in_ph = forch::op::create_input(in, ctx.dt).set_name("x");

  std::vector<float> want(vals.size());
  for (size_t i = 0; i < want.size(); ++i) {
    want[i] = base_val(static_cast<int64_t>(i), 0);
  }

  return run_case("broadcast", ctx.backend, ctx.dt, ctx.world(), feed,
    [&] {
      return TensorList{forch::op::broadcast(in_ph, ctx.group, /*root=*/0)};
    },
    {want});
}

int case_reduce(Ctx &ctx) {
  const Shape in    = {R, C};
  auto        vals  = gen(in, ctx.rank(), 0);
  auto        feed  = TensorList{ctx.arena.feed(in, ctx.dt, vals, "x")};
  auto        in_ph = forch::op::create_input(in, ctx.dt).set_name("x");

  std::vector<float> want;
  if (ctx.rank() == 0) {
    want.resize(vals.size());
    for (size_t i = 0; i < want.size(); ++i) {
      double s = 0.0;
      for (int64_t q = 0; q < ctx.world(); ++q)
        s += base_val(static_cast<int64_t>(i), q);
      want[i] = static_cast<float>(s);
    }
  }

  return run_case("reduce", ctx.backend, ctx.dt, ctx.world(), feed,
    [&] { return TensorList{forch::op::reduce(in_ph, ctx.group, /*root=*/0)}; },
    {want});
}

int case_all_to_all(Ctx &ctx) {
  const Shape in    = {ctx.world(), C};
  auto        vals  = gen(in, ctx.rank(), 0);
  auto        feed  = TensorList{ctx.arena.feed(in, ctx.dt, vals, "x")};
  auto        in_ph = forch::op::create_input(in, ctx.dt).set_name("x");

  std::vector<float> want(numel(in));
  for (int64_t q = 0; q < ctx.world(); ++q) {
    for (int64_t j = 0; j < C; ++j) {
      want[static_cast<size_t>(q * C + j)] = base_val(ctx.rank() * C + j, q);
    }
  }

  return run_case("all_to_all", ctx.backend, ctx.dt, ctx.world(), feed,
    [&] { return TensorList{forch::op::all_to_all(in_ph, ctx.group)}; },
    {want});
}

int case_fused_matmul_all_reduce(Ctx &ctx) {
  const Shape sa   = {MM_M, MM_K};
  const Shape sb   = {MM_N, MM_K};
  auto        a    = gen_mm(sa, ctx.rank());
  auto        b    = gen_mm(sb, 0);
  auto        feed = TensorList{
    ctx.arena.feed(sa, ctx.dt, a, "x1"), ctx.arena.feed(sb, ctx.dt, b, "x2")};
  auto x1 = forch::op::create_input(sa, ctx.dt).set_name("x1");
  auto x2 = forch::op::create_input(sb, ctx.dt).set_name("x2");

  std::vector<double> acc(static_cast<size_t>(MM_M * MM_N), 0.0);
  for (int64_t q = 0; q < ctx.world(); ++q) {
    auto mm_q = ref_mm(gen_mm(sa, q), MM_M, MM_K, b, MM_N);
    for (size_t i = 0; i < acc.size(); ++i)
      acc[i] += mm_q[i];
  }

  return run_case("fused_matmul_all_reduce", ctx.backend, ctx.dt, ctx.world(),
    feed,
    [&] {
      return TensorList{forch::op::fused_matmul_all_reduce(x1, x2, ctx.group)};
    },
    {to_f(acc)});
}

int case_fused_matmul_all_reduce_add_rms_norm(Ctx &ctx) {
  const Shape sa    = {MM_M, MM_K};
  const Shape sb    = {MM_N, MM_K};
  const Shape sr    = {1, MM_M, MM_N};
  auto        a     = gen_mm(sa, ctx.rank());
  auto        b     = gen_mm(sb, 0);
  auto        res   = gen_mm(sr, 0);
  auto        gamma = std::vector<float>(static_cast<size_t>(MM_N), 1.0F);

  auto feed = TensorList{ctx.arena.feed(sa, ctx.dt, a, "x1"),
    ctx.arena.feed(sb, ctx.dt, b, "x2"),
    ctx.arena.feed(sr, ctx.dt, res, "residual"),
    ctx.arena.feed({MM_N}, ctx.dt, gamma, "gamma")};
  auto x1   = forch::op::create_input(sa, ctx.dt).set_name("x1");
  auto x2   = forch::op::create_input(sb, ctx.dt).set_name("x2");
  auto rs   = forch::op::create_input(sr, ctx.dt).set_name("residual");
  auto ga   = forch::op::create_input({MM_N}, ctx.dt).set_name("gamma");

  std::vector<double> acc(static_cast<size_t>(MM_M * MM_N), 0.0);
  for (int64_t q = 0; q < ctx.world(); ++q) {
    auto mm_q = ref_mm(gen_mm(sa, q), MM_M, MM_K, b, MM_N);
    for (size_t i = 0; i < acc.size(); ++i)
      acc[i] += mm_q[i];
  }
  for (size_t i = 0; i < acc.size(); ++i) {
    acc[i] += static_cast<double>(res[i]);
  }

  return run_case("fused_matmul_all_reduce_add_rms_norm", ctx.backend, ctx.dt,
    ctx.world(), feed,
    [&] {
      return TensorList{forch::op::fused_matmul_all_reduce_add_rms_norm(
        x1, x2, rs, ga, ctx.group)};
    },
    {to_f(acc)});
}

int case_fused_all_gather_matmul(Ctx &ctx) {
  const Shape sa   = {MM_M, MM_K};
  const Shape sb   = {MM_N, MM_K};
  auto        a    = gen_mm(sa, ctx.rank());
  auto        b    = gen_mm(sb, 0);
  auto        feed = TensorList{
    ctx.arena.feed(sa, ctx.dt, a, "x1"), ctx.arena.feed(sb, ctx.dt, b, "x2")};
  auto x1 = forch::op::create_input(sa, ctx.dt).set_name("x1");
  auto x2 = forch::op::create_input(sb, ctx.dt).set_name("x2");

  std::vector<double> want(static_cast<size_t>(MM_M * ctx.world() * MM_N), 0.0);
  for (int64_t q = 0; q < ctx.world(); ++q) {
    auto mm_q = ref_mm(gen_mm(sa, q), MM_M, MM_K, b, MM_N);
    for (int64_t i = 0; i < MM_M; ++i) {
      for (int64_t j = 0; j < MM_N; ++j) {
        want[static_cast<size_t>((q * MM_M + i) * MM_N + j)] =
          mm_q[static_cast<size_t>(i * MM_N + j)];
      }
    }
  }

  return run_case("fused_all_gather_matmul", ctx.backend, ctx.dt, ctx.world(),
    feed,
    [&] {
      return TensorList{forch::op::fused_all_gather_matmul(x1, x2, ctx.group)};
    },
    {to_f(want)});
}

int case_fused_matmul_reduce_scatter(Ctx &ctx) {
  const Shape sa   = {MM_M, MM_K};
  const Shape sb   = {MM_N, MM_K};
  auto        a    = gen_mm(sa, ctx.rank());
  auto        b    = gen_mm(sb, 0);
  auto        feed = TensorList{
    ctx.arena.feed(sa, ctx.dt, a, "x1"), ctx.arena.feed(sb, ctx.dt, b, "x2")};
  auto x1 = forch::op::create_input(sa, ctx.dt).set_name("x1");
  auto x2 = forch::op::create_input(sb, ctx.dt).set_name("x2");

  const int64_t       rows = MM_M / ctx.world();
  std::vector<double> want(static_cast<size_t>(rows * MM_N), 0.0);
  for (int64_t q = 0; q < ctx.world(); ++q) {
    auto mm_q = ref_mm(gen_mm(sa, q), MM_M, MM_K, b, MM_N);
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < MM_N; ++j) {
        want[static_cast<size_t>(i * MM_N + j)] +=
          mm_q[static_cast<size_t>((ctx.rank() * rows + i) * MM_N + j)];
      }
    }
  }

  return run_case("fused_matmul_reduce_scatter", ctx.backend, ctx.dt,
    ctx.world(), feed,
    [&] {
      return TensorList{
        forch::op::fused_matmul_reduce_scatter(x1, x2, ctx.group)};
    },
    {to_f(want)});
}

using CaseFn = int (*)(Ctx &);

struct CaseEntry {
  const char *name;
  CaseFn      fn;
  /// The fused ops run a matmul, so they are FP16/BF16 only and need a real
  /// multi-rank group.
  bool fused;
};

const CaseEntry kCases[] = {
  {"all_reduce", case_all_reduce, false},
  {"all_gather", case_all_gather, false},
  {"reduce_scatter", case_reduce_scatter, false},
  {"broadcast", case_broadcast, false},
  {"reduce", case_reduce, false},
  {"all_to_all", case_all_to_all, false},
  {"fused_matmul_all_reduce", case_fused_matmul_all_reduce, true},
  {"fused_matmul_all_reduce_add_rms_norm",
    case_fused_matmul_all_reduce_add_rms_norm, true},
  {"fused_all_gather_matmul", case_fused_all_gather_matmul, true},
  {"fused_matmul_reduce_scatter", case_fused_matmul_reduce_scatter, true},
};

int expect_throw(const char *what, auto &&fn) {
  try {
    fn();
    std::printf("  %-37s NO-THROW    expected a throw, got none\n", what);
    return 1;
  } catch (const std::exception &e) {
    std::printf(
      "  %-37s REJECT-OK   %s\n", what, mr::first_line(e.what()).c_str());
    return 0;
  }
}

/**
 * The op's own rules, which `Tensor::Impl` resolves at construction and which
 * therefore need no device. A group of two is synthesized so they hold however
 * the process was launched.
 */
int check_op_rules() {
  int bad = 0;

  const ProcessGroup pair{0, 2};

  auto f16 = forch::op::create_input({3, 4}, DType::FP16).set_name("f16");
  bad += expect_throw("reduce_scatter rejects %2",
    [&] { (void)forch::op::reduce_scatter(f16, pair); });
  bad += expect_throw(
    "all_to_all rejects %2", [&] { (void)forch::op::all_to_all(f16, pair); });
  bad += expect_throw("broadcast rejects root >= world",
    [&] { (void)forch::op::broadcast(f16, pair, /*root=*/2); });
  bad += expect_throw(
    "group rejects rank >= world", [] { (void)ProcessGroup{2, 2}; });
  bad += expect_throw(
    "group rejects negative rank", [] { (void)ProcessGroup{-1, 2}; });
  bad +=
    expect_throw("group rejects zero world", [] { (void)ProcessGroup{0, 0}; });

  auto f32 = forch::op::create_input({2, 4}, DType::FP32).set_name("f32");
  auto f64 = forch::op::create_input({2, 4}, DType::FP64).set_name("f64");
  bad += expect_throw("fused matmul rejects mixed dtypes",
    [&] { (void)forch::op::fused_matmul_all_reduce(f32, f64, pair); });
  auto bias = forch::op::create_input({3}, DType::BF16);
  bad += expect_throw("fused matmul rejects mixed bias",
    [&] { (void)forch::op::fused_matmul_all_reduce(f16, f16, bias, pair); });

  return bad;
}

/**
 * Which dtypes the hardware can actually move is HCCL's business, not the
 * collective's, so it is checked by the backend while it emits -- i.e. inside
 * `backend.build()`, not at Tensor construction.
 */
int check_backend_dtype_rules(Backend &backend, ProcessGroup group) {
  int bad = 0;

  auto build = [&](const char *name, const Tensor &out) {
    forch::Graph gir({Tensor(out).set_name("y")});
    gir.set_name(name);
    (void)backend.build(gir);
  };

  auto f64 = forch::op::create_input({2, 4}, DType::FP64).set_name("f64");
  bad += expect_throw("hccl rejects fp64",
    [&] { build("reject_f64", forch::op::all_reduce(f64, group)); });

  // FP32 is fine for a plain collective but not for the fused matmul.
  const Shape sa = {MM_M, MM_K};
  const Shape sb = {MM_N, MM_K};
  auto        x1 = forch::op::create_input(sa, DType::FP32).set_name("x1f32");
  auto        x2 = forch::op::create_input(sb, DType::FP32).set_name("x2f32");
  bad += expect_throw("fused matmul rejects fp32", [&] {
    build(
      "reject_fused_f32", forch::op::fused_matmul_all_reduce(x1, x2, group));
  });

  return bad;
}

/**
 * GE's globals take effect once per process, so a second backend asking for
 * something else has to be refused rather than silently ignored. Runs after
 * the first backend exists.
 */
int check_reconfigure_rejected(const forch::AscendDevice &device) {
  auto other = forch::GEBackend::Options::from_env();
  int  bad   = expect_throw("second backend rejects same options",
       [&] { forch::GEBackend second(device, other); });
  other.hccl_timeout += std::chrono::seconds(1);
  return bad + expect_throw("reconfigure rejects new options", [&] {
    forch::GEBackend second(device, other);
  });
}

} // namespace

int main(int argc, char **argv) {
  const Args  args(argc, argv, {"dtype", "case"});
  const DType dt   = args.get_dtype("dtype", DType::FP16);
  const auto  only = args.get("case", "all");
  if (only != "all" && std::ranges::none_of(kCases, [&](const auto &entry) {
        return only == entry.name;
      })) {
    mr::fail("unknown --case '" + only + "'");
  }

  const ProcessGroup group = forch::process_group_from_env();

  std::printf("[rank %lld/%lld] dtype=%s case=%s\n",
    static_cast<long long>(group.rank_id()),
    static_cast<long long>(group.world_size()), dtype_name(dt), only.c_str());

  std::printf("-- op rules (no device needed) --\n");
  const int preflight_failed = check_op_rules();

  if (group.world_size() < 2) {
    // Every case below is a collective, and HCOM has no topology without a
    // rank table, so there is nothing to run rather than something to fail.
    std::printf("[rank 0] no collective cases: world_size is 1. Launch with "
                "script/forch-ge-run.py --devices <a,b> to get a group.\n");
    return preflight_failed == 0 ? 0 : 1;
  }

  // Default GE options; tp_align.cpp covers explicit options.
  auto             device = forch::AscendDevice::from_env();
  forch::GEBackend backend_impl(device);
  Backend          backend = backend_impl;
  FeedArena        arena;

  int total = 0, passed = 0, skipped = 0, failed = preflight_failed;
  std::printf("-- backend rules --\n");
  failed += check_reconfigure_rejected(device);
  failed += check_backend_dtype_rules(backend, group);

  for (const auto &c : kCases) {
    if (only != "all" && only != c.name)
      continue;
    ++total;

    if (c.fused) {
      if (!mr::supports_fused_world(group.world_size())) {
        std::printf("  %-37s SKIP        fused ops need world_size in "
                    "{2,4,8}\n",
          c.name);
        ++skipped;
        continue;
      }
      if (dt == DType::FP32) {
        std::printf(
          "  %-37s SKIP        fused ops support FP16/BF16 only\n", c.name);
        ++skipped;
        continue;
      }
      if (MM_M % group.world_size() != 0) {
        std::printf(
          "  %-37s SKIP        M(%lld) not divisible by world(%lld)\n", c.name,
          static_cast<long long>(MM_M),
          static_cast<long long>(group.world_size()));
        ++skipped;
        continue;
      }
    }

    Ctx       ctx{backend, dt, group, arena};
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
