// MoeFinalizeRoutingV2 multi-case probe (token-major public contract): the
// row_idx input is a TOKEN-MAJOR keyed rank table — exactly what
// moe_init_routing emits, consumed natively via drop_pad_mode=2. T=2/K=2
// (regression), T=8/K=6 (repro of the backend compare failure), T=8/K=2 and
// T=2/K=6 (isolate whether the trigger is T/num_tokens or K). A final
// init->finalize direct-link case guards the layout contract between the
// two ops (the P1-1 regression class: both unit tests can pass while the
// pairing silently breaks).
#include "forch/ops/ge/moe_finalize_routing.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/ge/moe_init_routing.hpp"
#include "forch/ops/input.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace forch;

/// out(i,j) = Σ_s scales(i,s) * expandedX( row_idx[i*k+s] )  — row_idx is
/// token-major keyed (contract); the kernel's slot-major read is handled by
/// the facade adapter.
static std::vector<float> reference(const std::vector<float> &expanded_x,
  const std::vector<int32_t> &row_idx, const std::vector<float> &scales,
  int64_t t, int64_t k, int64_t d) {
  std::vector<float> out(static_cast<size_t>(t) * d, 0.0F);
  for (int64_t i = 0; i < t; ++i)
    for (int64_t kk = 0; kk < k; ++kk) {
      int64_t exp_idx = row_idx[i * k + kk]; // token-major keyed
      float   s       = scales[i * k + kk];
      for (int64_t j = 0; j < d; ++j)
        out[i * d + j] += s * expanded_x[exp_idx * d + j];
    }
  return out;
}

/// Run one (T, K) case. row_idx = identity (token-major arange), scales and
/// expanded_x rows are simple recognizable values. Identity row_idx still
/// catches a broken adapter whenever k != T (slot-major vs token-major
/// reads then disagree).
static int run_case(Backend &backend, int64_t t, int64_t k, int64_t d) {
  auto in_x = op::create_input({t * k, d}, DType::FP32).set_name("expanded_x");
  auto in_r = op::create_input({t * k}, DType::INT32).set_name("row_idx");
  auto in_s = op::create_input({t, k}, DType::FP32).set_name("scales");
  auto in_e = op::create_input({t, k}, DType::INT32).set_name("expert_idx");
  auto out  = op::ge::moe_finalize_routing()(in_x, in_r, in_s, in_e);
  out.set_name("y");
  Graph graph_ir({out});
  graph_ir.set_name("moe_finalize");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &err) {
    std::cerr << "T=" << t << " K=" << k << " compile failed: " << err.what()
              << std::endl;
    return 1;
  }

  // expanded_x row r = [r, r+0.1, r+0.2, r+0.3]; identity row_idx
  // (token-major); scales[t,k] = (t*K+k+1)*0.1; expert_idx arbitrary
  // (no bias)
  std::vector<float>   x(static_cast<size_t>(t) * k * d);
  std::vector<int32_t> r(static_cast<size_t>(t) * k);
  std::vector<float>   s(static_cast<size_t>(t) * k);
  std::vector<int32_t> e(static_cast<size_t>(t) * k);
  for (int64_t i = 0; i < t * k; ++i) {
    r[i] = static_cast<int32_t>(i);
    for (int64_t j = 0; j < d; ++j)
      x[i * d + j] = static_cast<float>(i) + j * 0.1F;
    e[i] = static_cast<int32_t>(i % 3);
  }
  for (int64_t i = 0; i < t * k; ++i)
    s[i] = static_cast<float>(i + 1) * 0.1F;

  auto x_t =
    op::load_from_vector(std::move(x), {t * k, d}).set_name("expanded_x");
  auto r_t = op::load_from_vector(std::move(r), {t * k}).set_name("row_idx");
  auto s_t = op::load_from_vector(std::move(s), {t, k}).set_name("scales");
  auto e_t = op::load_from_vector(std::move(e), {t, k}).set_name("expert_idx");
  auto outputs = graph.run({x_t, r_t, s_t, e_t});
  const float *py =
    reinterpret_cast<const float *>(outputs.get("y").get_data());

  std::vector<float>   xv(static_cast<size_t>(t) * k * d);
  std::vector<int32_t> rv(static_cast<size_t>(t) * k);
  std::vector<float>   sv(static_cast<size_t>(t) * k);
  for (int64_t i = 0; i < t * k; ++i) {
    rv[i] = static_cast<int32_t>(i);
    for (int64_t j = 0; j < d; ++j)
      xv[i * d + j] = static_cast<float>(i) + j * 0.1F;
    sv[i] = static_cast<float>(i + 1) * 0.1F;
  }
  auto expected = reference(xv, rv, sv, t, k, d);

  double max_abs = 0.0, max_rel = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    double diff = std::fabs(static_cast<double>(py[i]) - expected[i]);
    max_abs     = std::max(max_abs, diff);
    max_rel     = std::max(max_rel, diff / (std::fabs(expected[i]) + 1e-6));
  }
  std::cout << "T=" << t << " K=" << k << ": y[0]=" << py[0] << " (expect "
            << expected[0] << ") max_abs=" << max_abs << " max_rel=" << max_rel
            << (max_abs < 1e-2F ? "  PASS" : "  FAIL") << std::endl;
  return max_abs < 1e-2F ? 0 : 2;
}

/// init -> finalize direct link (token-major contract): the routing op's
/// expanded_row_idx feeds the combine op unchanged — no caller-side
/// permute. CPU reference over the sorted order:
///   y(i) = sum_s scales(i,s) * expanded_x[ row_idx[i*k+s] ]
static int run_init_finalize_link(Backend &backend) {
  const int64_t        T = 2, k = 3, E = 3, D = 2, n = T * k;
  std::vector<int32_t> v = {0, 1, 2, 1, 0, 2}; // flat t-major experts

  // x rows are per-token markers: x0 = [101 102], x1 = [201 202]
  std::vector<float> x(static_cast<size_t>(T * D));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < D; ++j)
      x[t * D + j] = static_cast<float>((t + 1) * 100 + j + 1);
  std::vector<float> w(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    w[i] = static_cast<float>(i + 1) * 0.1F;

  // ---- CPU reference over the sorted order ----
  std::vector<int32_t> order(n);
  for (int64_t p = 0; p < n; ++p)
    order[p] = static_cast<int32_t>(p);
  std::stable_sort(order.begin(), order.end(),
    [&](int32_t a, int32_t b) { return v[a] < v[b]; });
  std::vector<int32_t> row_idx(n); // token-major keyed rank table
  for (int64_t i = 0; i < n; ++i)
    row_idx[order[i]] = static_cast<int32_t>(i);
  std::vector<float> exx(static_cast<size_t>(n * D));
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < D; ++j)
      exx[i * D + j] = x[(order[i] / k) * D + j];

  std::vector<double> yref(static_cast<size_t>(T * D), 0.0);
  for (int64_t i = 0; i < T; ++i)
    for (int64_t s = 0; s < k; ++s) {
      int64_t ex = row_idx[i * k + s];
      for (int64_t j = 0; j < D; ++j)
        yref[i * D + j] += w[i * k + s] * exx[ex * D + j];
    }

  // ---- graph: init -> finalize (direct link, no caller permute) ----
  auto in_x    = op::create_input({T, D}, DType::FP32).set_name("x");
  auto in_e    = op::create_input({T, k}, DType::INT32).set_name("expert_idx");
  auto in_s    = op::create_input({T, k}, DType::FP32).set_name("scales");
  auto routing = op::ge::moe_init_routing(E)(in_x, in_e);
  auto y       = op::ge::moe_finalize_routing()(
    routing.expanded_x, routing.expanded_row_idx, in_s, in_e);
  y.set_name("y");
  Graph graph_ir(TensorList{y});
  graph_ir.set_name("init_finalize_link");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &err) {
    std::cerr << "init_finalize_link: compile/load failed: " << err.what()
              << std::endl;
    return 2;
  }
  auto x_t = op::load_from_vector(std::move(x), Shape({T, D})).set_name("x");
  auto e_t =
    op::load_from_vector(std::move(v), Shape({T, k})).set_name("expert_idx");
  auto s_t =
    op::load_from_vector(std::move(w), Shape({T, k})).set_name("scales");
  auto         outputs = graph.run(TensorList{x_t, e_t, s_t});
  const float *py =
    reinterpret_cast<const float *>(outputs.get("y").get_data());

  double max_abs = 0.0, max_rel = 0.0;
  for (size_t i = 0; i < yref.size(); ++i) {
    double diff = std::fabs(static_cast<double>(py[i]) - yref[i]);
    max_abs     = std::max(max_abs, diff);
    max_rel     = std::max(max_rel, diff / (std::fabs(yref[i]) + 1e-9));
  }
  std::cout << "  [init_finalize_link] max_abs=" << max_abs
            << " max_rel=" << max_rel << (max_abs < 1e-2 ? "  ok" : "  FAIL")
            << std::endl;
  return max_abs < 1e-2 ? 0 : 3;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;
  int       fails   = 0;
  fails += run_case(backend, 2, 2, 4); // regression (passed before)
  fails += run_case(backend, 8, 6, 4); // repro of backend compare failure
  fails += run_case(backend, 8, 2, 4); // isolate T
  fails += run_case(backend, 2, 6, 4); // isolate K
  fails += run_init_finalize_link(backend);
  std::cout << (fails == 0 ? "All Finalize cases passed." : "Finalize FAILED")
            << std::endl;
  return fails == 0 ? 0 : 1;
}
