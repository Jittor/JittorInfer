// From the parent root, on two allocated cards with CANN loaded:
// python forch/script/forch-ge-run.py --tee -- \
//   ./build/forch/src/forch-ge/tests/multirank/forch-ge-mr-parallel_linear
#include "forch/nn/parallel_linear.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/reshape.hpp"

#include "mr_common.hpp"

#include <iostream>
#include <map>

using namespace forch;
using nn::ColumnParallelLinear;
using nn::RowParallelLinear;

namespace {
struct Loader : nn::Loader {
  std::map<std::string, Tensor> tensors;
  Tensor                        load(const std::string &name) override {
    return tensors.at(name);
  }
};
template <class Linear>
void load(Linear &linear, DType dtype, const std::vector<float> &weight,
  const std::vector<float> &bias, int64_t in, int64_t out, bool has_bias) {
  mr::FeedArena storage;
  Loader        loader;
  loader.tensors.emplace(
    "projection.weight", storage.feed({out, in}, dtype, weight, "weight"));
  if (has_bias)
    loader.tensors.emplace(
      "projection.bias", storage.feed({out}, dtype, bias, "bias"));
  linear.load(&loader, "projection");
  // Both source loader and its BF16/FP16 buffers die before graph building.
}
std::vector<float> matrix(int64_t rows, int64_t cols, int salt) {
  std::vector<float> values(rows * cols);
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c)
      values[r * cols + c] = mr::cell(r, c, salt);
  return values;
}
std::vector<float> reference(const std::vector<float> &x,
  const std::vector<float> &w, const std::vector<float> &b, int64_t m,
  int64_t k, int64_t n, bool bias) {
  std::vector<float> y(m * n);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      float sum = bias ? b[j] : 0;
      for (int64_t l = 0; l < k; ++l)
        sum += x[i * k + l] * w[j * k + l];
      y[i * n + j] = sum;
    }
  return y;
}
void check(const Tensor &got, const Shape &shape, DType dtype,
  const std::vector<float> &expected) {
  if (got.shape() != shape || got.dtype() != dtype)
    mr::fail("output metadata mismatch");
  auto values = mr::read_back(got.get_data(), expected.size(), dtype);
  for (size_t i = 0; i < values.size(); ++i)
    if (!std::isfinite(values[i]) || values[i] != expected[i])
      mr::fail("incorrect matmul at " + std::to_string(i) + ": got "
        + std::to_string(values[i]) + " expected "
        + std::to_string(expected[i]));
}
void run(GEBackend &backend, ProcessGroup group, DType dtype, bool bias) {
  constexpr int64_t m = 3, k = 8, n = 8;
  auto x = matrix(m, k, 11), w = matrix(n, k, 23), w2 = matrix(n, n, 47);
  std::vector<float> b(n), b2(n);
  for (int64_t j = 0; j < n; ++j) {
    b[j]  = 1 + j % 3;
    b2[j] = 2 + j % 2;
  }
  auto full      = reference(x, w, b, m, k, n, bias);
  auto chain_ref = reference(full, w2, b2, m, n, n, bias);
  auto local_k = k / group.world_size(), local_n = n / group.world_size();
  std::vector<float> local_x(m * local_k), local_ref(m * local_n);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < local_k; ++j)
      local_x[i * local_k + j] = x[i * k + group.rank_id() * local_k + j];
    for (int64_t j = 0; j < local_n; ++j)
      local_ref[i * local_n + j] = full[i * n + group.rank_id() * local_n + j];
  }
  ColumnParallelLinear column(k, n, group, bias, dtype);
  RowParallelLinear    row(k, n, group, bias, dtype);
  RowParallelLinear    down(n, n, group, bias, dtype);
  load(column, dtype, w, b, k, n, bias);
  load(row, dtype, w, b, k, n, bias);
  load(down, dtype, w2, b2, n, n, bias);
  if (column.weight().shape() != Shape{local_n, k}
    || row.weight().shape() != Shape{n, local_k})
    mr::fail("full checkpoint was retained instead of local parameters");
  auto input       = op::create_input({m, k}, dtype).set_name("x");
  auto local_input = op::create_input({m, local_k}, dtype).set_name("local_x");
  auto col         = column.forward(input).set_name("column");
  auto reduced     = row.forward(local_input).set_name("row");
  auto chain       = down.forward(col).set_name("chain");
  auto batched = down.forward(column.forward(input | op::reshape({1, m, k})))
                   .set_name("batched");
  auto graph =
    backend.build(Graph({col, reduced, chain, batched})
                    .set_name(std::string("parallel_linear_")
                      + mr::dtype_name(dtype) + (bias ? "_bias" : "_no_bias")));
  graph.compile();
  graph.load();
  mr::FeedArena arena;
  auto          outputs = graph.run({arena.feed({m, k}, dtype, x, "x"),
             arena.feed({m, local_k}, dtype, local_x, "local_x")});
  check(outputs.get("column"), {m, local_n}, dtype, local_ref);
  check(outputs.get("row"), {m, n}, dtype, full);
  check(outputs.get("chain"), {m, n}, dtype, chain_ref);
  check(outputs.get("batched"), {1, m, n}, dtype, chain_ref);
  std::cout << "PASS rank=" << group.rank_id()
            << " dtype=" << mr::dtype_name(dtype) << " bias=" << bias
            << " column, row, column->row, batched; host reference exact\n";
}
} // namespace

int main() {
  try {
    auto group = process_group_from_env();
    if (group.world_size() < 2 || 8 % group.world_size()) {
      std::cout << "SKIP: requires 2, 4 or 8 ranks\n";
      return 0;
    }
    GEBackend backend(AscendDevice::from_env());
    for (auto dtype : {DType::FP32, DType::BF16, DType::FP16})
      for (bool bias : {false, true})
        run(backend, group, dtype, bias);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
