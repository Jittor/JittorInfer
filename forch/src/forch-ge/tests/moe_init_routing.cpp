// MoeInitRouting (CANN MoeInitRoutingV3) probe — verifies the t-major token
// selection:
//   expandedXOut[expandedRowIdxOut[i]] = x[i // k]   (i = t*k + s flat)
//   expandedRowIdxOut[sortedRowIdx[i]] = i           (inverse permutation)
//   expertTokensCountOrCumsumOut[i] = cumsum(histogram(sortedExpertIdx))
//
// Data: T=2, k=3, D=4, E=3. x rows 0/1 marked by first element (1 / 5).
//   expert_idx (t-major flat) = [0,1,2, 1,0,2]
//   stable sort -> sortedExpertIdx [0,0,1,1,2,2], sortedRowIdx [0,4,1,3,2,5]
//   expandedRowIdxOut = [0,2,4,3,1,5]
//   expanded_x row = x[i//3] at row expandedRowIdxOut[i]:
//     row 0..5 = [x0, x1, x0, x1, x0, x1] -> markers [1,5,1,5,1,5]
//   cumsum (flag 0) = [2,4,6]; count (flag 1) = [2,2,2]
#include "forch/ops/ge/moe_init_routing.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <iostream>
#include <vector>

using namespace forch;

static bool ints_equal(const int32_t *a, const int32_t *b, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i])
      return false;
  return true;
}
static bool floats_equal(const float *a, const float *b, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (std::fabs(a[i] - b[i]) > 1e-5F)
      return false;
  return true;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  auto in_x = op::create_input({2, 4}, DType::FP32).set_name("x");
  auto in_e = op::create_input({2, 3}, DType::INT32).set_name("expert_idx");
  // active_num = T*k = 6 (expanded positions), capacity = 6, E = 3
  auto routing = op::ge::moe_init_routing(3)(in_x, in_e);
  auto out_x   = routing.expanded_x;
  auto out_r   = routing.expanded_row_idx;
  auto out_c   = routing.expert_tokens_count_or_cumsum;
  out_x.set_name("expanded_x");
  out_r.set_name("expanded_row_idx");
  out_c.set_name("tokens_cumsum");
  if (out_x.shape() != Shape{6, 4}) {
    std::cerr << "shape mismatch: expanded_x expected [6,4], got ";
    for (auto d : out_x.shape())
      std::cerr << d << ",";
    std::cerr << std::endl;
    return 1;
  }

  Graph graph_ir({out_x, out_r, out_c});
  graph_ir.set_name("moe_v3");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
    std::cout << "moe_v3 compiled" << std::endl;
  } catch (const std::exception &err) {
    std::cerr << "moe_v3 compile/load failed: " << err.what() << std::endl;
    return 2;
  }

  std::vector<float>   x = {1, 2, 3, 4, 5, 6, 7, 8}; // [2, 4]
  std::vector<int32_t> e = {0, 1, 2, 1, 0, 2};       // [2, 3] t-major flat
  auto x_t = op::load_from_vector(std::move(x), {2, 4}).set_name("x");
  auto e_t = op::load_from_vector(std::move(e), {2, 3}).set_name("expert_idx");
  auto outputs = graph.run({x_t, e_t});

  const float *px =
    reinterpret_cast<const float *>(outputs.get("expanded_x").get_data());
  const int32_t *pr = reinterpret_cast<const int32_t *>(
    outputs.get("expanded_row_idx").get_data());
  const int64_t *pc =
    reinterpret_cast<const int64_t *>(outputs.get("tokens_cumsum").get_data());
  std::cout << "expanded_x markers (row first elem): ";
  for (int i = 0; i < 6; ++i)
    std::cout << static_cast<int>(px[i * 4]) << " ";
  std::cout << std::endl;
  std::cout << "expanded_row_idx: ";
  for (int i = 0; i < 6; ++i)
    std::cout << pr[i] << " ";
  std::cout << std::endl;
  std::cout << "tokens_cumsum: ";
  for (int i = 0; i < 3; ++i)
    std::cout << pc[i] << " ";
  std::cout << std::endl;

  // expectations (t-major i//k semantics)
  const float   ex_x[6] = {1, 5, 1, 5, 1, 5}; // [x0,x1,x0,x1,x0,x1]
  const int32_t ex_r[6] = {0, 2, 4, 3, 1, 5};
  const int32_t ex_c[3] = {2, 4, 6}; // cumsum of [2,2,2]
  // cumsum comes back as INT64; narrow a copy to INT32 so the check below
  // compares like-for-like with ex_c.
  int32_t pc32[3];
  for (int i = 0; i < 3; ++i)
    pc32[i] = static_cast<int32_t>(pc[i]);
  // expanded_x rows carry the source-token marker at [i*4] (row first
  // element); compare markers only — they uniquely identify each x row.
  float markers[6];
  for (int i = 0; i < 6; ++i)
    markers[i] = px[i * 4];
  if (!floats_equal(markers, ex_x, 6) || !ints_equal(pr, ex_r, 6)
    || !ints_equal(pc32, ex_c, 3)) {
    std::cerr << "MoeInitRouting values differ" << std::endl;
    return 3;
  }
  std::cout << "MoeInitRouting tests passed (t-major token selection OK)."
            << std::endl;
  return 0;
}
