#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/cast.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/permute.hpp"
#include "forch/ops/reshape.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace forch;

static bool almost_equal(
  const float *a, const float *b, size_t n, float tol = 1e-5f) {
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

int test_cast(Backend &backend) {
  // input 2x2
  Shape              shape = {2, 2};
  std::vector<float> data  = {1.0f, 2.0f, 3.0f, 4.0f};

  auto input = op::create_input(shape, DType::FP32).set_name("input_cast");
  auto out   = op::cast(DType::BF16)(input);
  out.set_name(std::string("cast_output"));

  forch::Graph graph_ir({out});
  graph_ir.set_name("test_cast");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "cast: compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  std::vector<float> expected = data;
  auto               input_tensor =
    op::load_from_vector(std::move(data), shape).set_name("input_cast");
  auto outputs = graph.run({input_tensor});
  if (outputs.empty())
    return 3;

  auto out_tensor = outputs.get(std::string("cast_output"));
  if (out_tensor.dtype() != DType::BF16) {
    std::cerr << "cast: dtype mismatch" << std::endl;
    return 4;
  }

  // BF16 stored as uint16_t per element; convert to float for comparison
  const uint16_t *bf_ptr =
    reinterpret_cast<const uint16_t *>(out_tensor.get_data());
  size_t             n = data.size();
  std::vector<float> out_f(n);
  for (size_t i = 0; i < n; ++i) {
    uint32_t bits = static_cast<uint32_t>(bf_ptr[i]) << 16;
    float    v;
    std::memcpy(&v, &bits, sizeof(v));
    out_f[i] = v;
  }

  if (!almost_equal(out_f.data(), expected.data(), n, 1e-2f)) {
    std::cerr << "cast: values differ (bf16->fp32)" << std::endl;
    return 5;
  }

  return 0;
}

int test_permute(Backend &backend) {
  // input 2x3 -> transpose to 3x2
  Shape                shape     = {2, 3};
  std::vector<float>   data      = {1, 2, 3, 4, 5, 6};
  Shape                out_shape = {3, 2};
  std::vector<int64_t> perm      = {1, 0};

  auto input = op::create_input(shape, DType::FP32).set_name("input_permute");
  auto out   = op::permute(perm)(input);
  out.set_name(std::string("permute_output"));

  forch::Graph graph_ir({out});
  graph_ir.set_name("test_permute");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "permute: compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  auto input_tensor =
    op::load_from_vector(std::move(data), shape).set_name("input_permute");
  auto outputs = graph.run({input_tensor});
  if (outputs.empty())
    return 3;

  const float *out_ptr = reinterpret_cast<const float *>(
    outputs.get(std::string("permute_output")).get_data());

  // expected transpose: {1,4,2,5,3,6}
  std::vector<float> expected = {1, 4, 2, 5, 3, 6};
  if (outputs.get(std::string("permute_output")).shape() != out_shape) {
    std::cerr << "permute: shape mismatch" << std::endl;
    return 4;
  }
  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "permute: values differ" << std::endl;
    return 5;
  }

  return 0;
}

int test_reshape(Backend &backend) {
  // input 2x3 -> reshape to 3x2 (row-major)
  Shape              shape     = {2, 3};
  std::vector<float> data      = {1, 2, 3, 4, 5, 6};
  Shape              out_shape = {3, 2};

  auto input = op::create_input(shape, DType::FP32).set_name("input_reshape");
  auto out   = op::reshape(out_shape)(input);
  out.set_name(std::string("reshape_output"));

  forch::Graph graph_ir({out});
  graph_ir.set_name("test_reshape");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "reshape: compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  auto input_tensor =
    op::load_from_vector(std::move(data), shape).set_name("input_reshape");
  auto outputs = graph.run({input_tensor});
  if (outputs.empty())
    return 3;

  const float *out_ptr = reinterpret_cast<const float *>(
    outputs.get(std::string("reshape_output")).get_data());

  // expected reinterpretation: {1,2,3,4,5,6} laid out as 3x2
  std::vector<float> expected = {1, 2, 3, 4, 5, 6};
  if (outputs.get(std::string("reshape_output")).shape() != out_shape) {
    std::cerr << "reshape: shape mismatch" << std::endl;
    return 4;
  }
  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "reshape: values differ" << std::endl;
    return 5;
  }

  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  int r = 0;
  r     = test_cast(backend);
  if (r != 0)
    return r;
  r = test_permute(backend);
  if (r != 0)
    return r;
  r = test_reshape(backend);
  if (r != 0)
    return r;
  std::cout << "All shape tests passed." << std::endl;
  return 0;
}
