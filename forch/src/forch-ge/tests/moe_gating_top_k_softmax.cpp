// MoeGatingTopKSoftmaxV2 unit test — the fused MoE router:
//   softmax(x) -> topk -> (renorm ? y/sum(y) : y)
// Per case, against a CPU reference:
//   - y [T, k]: top-k softmax probabilities, descending;
//   - expert_idx [T, k] INT32: the matching expert ids;
//   - renorm=true: each y row sums to 1.
// Logits are constructed pairwise-distinct within each row so no ties can
// hide an index/value mismatch. Cases: k=3 renorm off/on, k=E boundary.
#include "forch/ops/ge/moe_gating_top_k_softmax.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace forch;

namespace {

// Deterministic logits, pairwise-distinct within each row (no ties).
std::vector<float> make_logits(int64_t t, int64_t e) {
  std::vector<float> x(static_cast<size_t>(t) * e);
  for (int64_t i = 0; i < t; ++i)
    for (int64_t j = 0; j < e; ++j) {
      const int64_t p = i * e + j;
      x[p]            = static_cast<float>(p * 37 % 2000) / 100.0F - 10.0F;
    }
  return x;
}

struct Expect {
  std::vector<float>   y;   // [T, k]
  std::vector<int32_t> idx; // [T, k]
};

Expect cpu_reference(
  const std::vector<float> &x, int64_t t, int64_t e, int64_t k, bool renorm) {
  Expect               out;
  std::vector<int32_t> ids(static_cast<size_t>(e));
  std::iota(ids.begin(), ids.end(), 0);
  for (int64_t i = 0; i < t; ++i) {
    const float       *row = x.data() + i * e;
    const float        mx  = *std::max_element(row, row + e);
    std::vector<float> p(static_cast<size_t>(e));
    float              sum = 0.0F;
    for (int64_t j = 0; j < e; ++j) {
      p[j] = std::exp(row[j] - mx);
      sum += p[j];
    }
    for (auto &v : p)
      v /= sum;
    auto idx = ids;
    std::stable_sort(idx.begin(), idx.end(),
      [&](int32_t a, int32_t b) { return p[a] > p[b]; });
    float denom = 0.0F;
    for (int64_t s = 0; s < k; ++s) {
      out.y.push_back(p[idx[s]]);
      out.idx.push_back(idx[s]);
      denom += p[idx[s]];
    }
    if (renorm) {
      const size_t base = out.y.size() - static_cast<size_t>(k);
      for (int64_t s = 0; s < k; ++s)
        out.y[base + static_cast<size_t>(s)] /= denom;
    }
  }
  return out;
}

int run_case(Backend &backend, int64_t t, int64_t e, int64_t k, bool renorm) {
  auto in = op::create_input({t, e}, DType::FP32).set_name("x");
  auto g  = op::ge::moe_gating_top_k_softmax(k, renorm)(in);
  g.y.set_name("y");
  g.expert_idx.set_name("expert_idx");
  Graph graph_ir(TensorList{g.y, g.expert_idx});
  graph_ir.set_name(
    "moe_gating_k" + std::to_string(k) + (renorm ? "_renorm" : ""));
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &err) {
    std::cerr << "T=" << t << " E=" << e << " K=" << k << " renorm=" << renorm
              << " compile failed: " << err.what() << std::endl;
    return 1;
  }
  const auto x = make_logits(t, e);
  auto       x_t =
    op::load_from_vector(std::vector<float>(x), Shape({t, e})).set_name("x");
  auto         outputs = graph.run(TensorList{x_t});
  const float *py =
    reinterpret_cast<const float *>(outputs.get("y").get_data());
  const int32_t *pi =
    reinterpret_cast<const int32_t *>(outputs.get("expert_idx").get_data());

  const auto exp     = cpu_reference(x, t, e, k, renorm);
  double     max_abs = 0.0;
  for (size_t i = 0; i < exp.y.size(); ++i)
    max_abs = std::max(max_abs,
      std::fabs(static_cast<double>(py[i]) - static_cast<double>(exp.y[i])));
  const bool idx_ok  = std::equal(exp.idx.begin(), exp.idx.end(), pi);
  double     sum_err = 0.0;
  if (renorm)
    for (int64_t i = 0; i < t; ++i) {
      double s = 0.0;
      for (int64_t j = 0; j < k; ++j)
        s += py[i * k + j];
      sum_err = std::max(sum_err, std::fabs(s - 1.0));
    }
  const bool ok = max_abs < 1e-4 && idx_ok && (!renorm || sum_err < 1e-4);
  std::cout << "T=" << t << " E=" << e << " K=" << k << " renorm=" << renorm
            << ": max_abs=" << max_abs << " idx_ok=" << idx_ok
            << (renorm ? " sum_err=" + std::to_string(sum_err) : "")
            << (ok ? "  PASS" : "  FAIL") << std::endl;
  return ok ? 0 : 1;
}

} // namespace

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;
  int       fails   = 0;
  fails += run_case(backend, 4, 8, 3, false);
  fails += run_case(backend, 4, 8, 3, true);
  fails += run_case(backend, 5, 8, 8, true); // k == E boundary
  std::cout << (fails == 0 ? "All MoeGatingTopKSoftmax cases passed."
                           : "MoeGatingTopKSoftmax FAILED")
            << std::endl;
  return fails == 0 ? 0 : 1;
}
