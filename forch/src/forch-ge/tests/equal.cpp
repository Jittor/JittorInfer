#include "forch/ops/equal.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <iostream>
#include <vector>

using namespace forch;

static bool masks_match(const int8_t *a, const int8_t *b, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

static size_t numel(const Shape &shape) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

template <class T>
static int run_equal_case(Backend &backend, const Shape &shape,
  std::vector<T> a, std::vector<T> b, const char *case_name) {
  std::cout << "  [" << case_name << "] ... " << std::flush;

  auto in_a = op::create_input(shape, to_dtype<T>())
                .set_name(std::string(case_name) + "_a");
  auto in_b = op::create_input(shape, to_dtype<T>())
                .set_name(std::string(case_name) + "_b");
  auto out = op::equal()(in_a, in_b);
  out.set_name(std::string(case_name) + "_out");
  if (out.dtype() != DType::BOOL) {
    std::cerr << case_name << ": expected BOOL output" << std::endl;
    return 1;
  }

  Graph graph_ir({out});
  graph_ir.set_name(case_name);
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << case_name << ": compile/load failed: " << e.what()
              << std::endl;
    return 2;
  }

  std::vector<int8_t> expected(numel(shape));
  for (size_t i = 0; i < a.size(); ++i) {
    expected[i] = (a[i] == b[i]) ? 1 : 0;
  }
  auto in_a_t = op::load_from_vector(std::move(a), shape)
                  .set_name(std::string(case_name) + "_a");
  auto in_b_t = op::load_from_vector(std::move(b), shape)
                  .set_name(std::string(case_name) + "_b");
  auto outputs = graph.run({in_a_t, in_b_t});
  auto out_t   = outputs.get(std::string(case_name) + "_out");

  const int8_t *ptr = reinterpret_cast<const int8_t *>(out_t.get_data());
  if (!masks_match(ptr, expected.data(), expected.size())) {
    std::cerr << case_name << ": masks differ" << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (ptr[i] != expected[i]) {
        std::cerr << "first mismatch at " << i
                  << ": out=" << static_cast<int>(ptr[i])
                  << ", expected=" << static_cast<int>(expected[i])
                  << std::endl;
        break;
      }
    }
    return 3;
  }

  std::cout << "ok" << std::endl;
  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  {
    Shape                shape = {6};
    std::vector<int32_t> a     = {1, 2, 3, 4, 5, 6};
    std::vector<int32_t> b     = {1, 0, 3, 4, 9, 6};
    int r = run_equal_case(backend, shape, a, b, "equal_int32");
    if (r != 0)
      return r;
  }
  {
    Shape              shape = {4};
    std::vector<float> a     = {1.5F, 2.0F, -3.0F, 4.25F};
    std::vector<float> b     = {1.5F, 2.5F, -3.0F, 4.25F};
    int                r = run_equal_case(backend, shape, a, b, "equal_fp32");
    if (r != 0)
      return r;
  }
  {
    Shape                shape = {2, 3};
    std::vector<int32_t> a     = {0, 1, 2, 3, 4, 5};
    std::vector<int32_t> b     = {5, 4, 3, 2, 1, 0};
    int                  r = run_equal_case(backend, shape, a, b, "equal_2d");
    if (r != 0)
      return r;
  }

  std::cout << "Equal tests passed." << std::endl;
  return 0;
}
