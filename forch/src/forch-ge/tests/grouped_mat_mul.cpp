// GroupedMatmul probe — one operator, both weight forms x both transpose
// flags. M=8, K=4, E=4 groups, N=3. x [8, 4] rows pre-sorted into
// per-expert segments; group_list = cumulative end offsets
// (group_list_type=0, last == M). y [8, 3] = per-segment x @ W[g], in the
// same row order as x (what MoeFinalizeRouting consumes).
//   stacked: weights = one [E, N, K] tensor (transpose_weight=true, HF
//            layout [E, inter, D]) or [E, K, N] (false)
//   list:    weights = E separate [N, K] / [K, N] tensors (native HF
//            per-expert layout)
#include "forch/ops/grouped_mat_mul.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace forch;

static bool floats_equal(const float *a, const float *b, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (std::fabs(a[i] - b[i]) > 1e-4F)
      return false;
  return true;
}

/// Reference (per op docs, split_item=2, group_list_type=0): y = concat
/// over groups g of (x[group g rows] @ W[g]). group_list holds CUMULATIVE
/// end offsets (cumsum): group g covers rows [gl[g-1], gl[g]) (g=0:
/// [0, gl[0])). experts[e] is [N, K] when transpose_weight=true (y = x @
/// W^T) or [K, N] when false (y = x @ W). y rows keep x's order.
static std::vector<float> reference(const std::vector<float> &x,
  const std::vector<std::vector<float>> &experts, int64_t k, int64_t n,
  const std::vector<int64_t> &group_list, bool transpose_weight) {
  int64_t            total = group_list.back();
  std::vector<float> y(static_cast<size_t>(total) * n, 0.0F);
  for (size_t g = 0; g < group_list.size(); ++g) {
    const std::vector<float> &w     = experts[g];
    int64_t                   start = g == 0 ? 0 : group_list[g - 1];
    for (int64_t i = start; i < group_list[g]; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        float acc = 0.0F;
        for (int64_t kk = 0; kk < k; ++kk) {
          // transpose_weight=true: W[j, kk] at w[j*k + kk];
          // false: W[kk, j] at w[kk*n + j]
          acc +=
            x[i * k + kk] * (transpose_weight ? w[j * k + kk] : w[kk * n + j]);
        }
        y[i * n + j] = acc;
      }
    }
  }
  return y;
}

static int run_case(Backend &backend, const char *case_name, bool stacked,
  bool transpose_weight, const Shape &w_shape) {
  constexpr int64_t M = 8, K = 4, E = 4, N = 3;

  // x [8, 4], group_list [4] INT64 cumsum with an EMPTY group (model: 64
  // experts, ~6 routed -> most groups have M=0): boundary [2, 2, 5, 8] ->
  // group1 = [2,2) has 0 tokens.
  auto       in_x = op::create_input({M, K}, DType::FP32).set_name("x");
  TensorList in_ws;
  if (stacked) {
    in_ws.push_back(op::create_input({E, w_shape[0], w_shape[1]}, DType::FP32)
                      .set_name("weight"));
  } else {
    for (int64_t e = 0; e < E; ++e) {
      in_ws.push_back(op::create_input(w_shape, DType::FP32)
                        .set_name("weight_" + std::to_string(e)));
    }
  }
  auto in_gl = op::create_input({E}, DType::INT64).set_name("group_list");

  auto out = stacked
    ? op::grouped_mat_mul(transpose_weight)(in_x, in_ws[0], in_gl)
    : op::grouped_mat_mul(transpose_weight)(in_x, in_ws, in_gl);
  out.set_name("y");
  if (out.shape() != Shape{M, N}) {
    std::cerr << case_name << ": shape mismatch: expected [" << M << "," << N
              << "], got ";
    for (auto d : out.shape())
      std::cerr << d << ",";
    std::cerr << std::endl;
    return 1;
  }

  Graph graph_ir({out});
  graph_ir.set_name(case_name);
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
    std::cout << case_name << " compiled" << std::endl;
  } catch (const std::exception &err) {
    std::cerr << case_name << " compile/load failed: " << err.what()
              << std::endl;
    return 2;
  }

  // x rows 0..7: row i = [i+1, (i+1)*2, (i+1)*3, (i+1)*4]
  std::vector<float> x(static_cast<size_t>(M) * K);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < K; ++j)
      x[i * K + j] = static_cast<float>((i + 1) * (j + 1));
  // Four distinct [N=3, K=4] experts so group indexing and the empty group
  // are verifiable:
  //   W0 = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
  //   W1 = [[1,0,0,1],[0,1,0,1],[0,0,1,1]]
  //   W2 = [[1,1,1,1],[1,1,1,1],[1,1,1,1]]
  //   W3 = [[2,0,0,0],[0,2,0,0],[0,0,2,0]]
  std::vector<std::vector<float>> experts = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    {1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0}};
  // transpose_weight=false stores [K, N] (row-major transposes of the above)
  if (!transpose_weight) {
    for (auto &w : experts) {
      std::vector<float> t(w.size());
      for (int64_t i = 0; i < N; ++i)
        for (int64_t j = 0; j < K; ++j)
          t[j * N + i] = w[i * K + j];
      w = std::move(t);
    }
  }
  // cumsum boundaries: group0=[0,2) W0, group1=[2,2) EMPTY, group2=[2,5)
  // W2, group3=[5,8) W3
  std::vector<int64_t> gl = {2, 2, 5, 8};
  auto expected           = reference(x, experts, K, N, gl, transpose_weight);

  TensorList run_inputs{
    op::load_from_vector(std::move(x), {M, K}).set_name("x")};
  if (stacked) {
    std::vector<float> w_all;
    w_all.reserve(static_cast<size_t>(E) * N * K);
    for (const auto &w : experts)
      w_all.insert(w_all.end(), w.begin(), w.end());
    run_inputs.push_back(
      op::load_from_vector(std::move(w_all), {E, w_shape[0], w_shape[1]})
        .set_name("weight"));
  } else {
    for (int64_t e = 0; e < E; ++e) {
      run_inputs.push_back(op::load_from_vector(std::move(experts[e]), w_shape)
                             .set_name("weight_" + std::to_string(e)));
    }
  }
  run_inputs.push_back(
    op::load_from_vector(std::move(gl), {E}).set_name("group_list"));
  auto outputs = graph.run(run_inputs);

  const float *py =
    reinterpret_cast<const float *>(outputs.get("y").get_data());
  std::cout << case_name << " y:" << std::endl;
  for (int64_t i = 0; i < M; ++i) {
    std::cout << "  [";
    for (int64_t j = 0; j < N; ++j)
      std::cout << py[i * N + j] << " ";
    std::cout << "]" << std::endl;
  }

  if (!floats_equal(py, expected.data(), expected.size())) {
    std::cerr << case_name << ": values differ; expected:" << std::endl;
    for (int64_t i = 0; i < M; ++i) {
      std::cerr << "  [";
      for (int64_t j = 0; j < N; ++j)
        std::cerr << expected[i * N + j] << " ";
      std::cerr << "]" << std::endl;
    }
    return 3;
  }
  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  // stacked × HF layout [E, N, K] (transpose_weight=true)
  int rc = run_case(backend, "grouped_mat_mul_stacked_t", true, true, {3, 4});
  if (rc != 0) {
    return rc;
  }
  // stacked × kernel-native layout [E, K, N]
  rc = run_case(backend, "grouped_mat_mul_stacked_nt", true, false, {4, 3});
  if (rc != 0) {
    return rc;
  }
  // list × per-expert [N, K] (transpose_weight=true)
  rc = run_case(backend, "grouped_mat_mul_list_t", false, true, {3, 4});
  if (rc != 0) {
    return rc;
  }
  // list × per-expert [K, N]
  rc = run_case(backend, "grouped_mat_mul_list_nt", false, false, {4, 3});
  if (rc != 0) {
    return rc;
  }
  std::cout << "GroupedMatmul tests passed." << std::endl;
  return 0;
}
