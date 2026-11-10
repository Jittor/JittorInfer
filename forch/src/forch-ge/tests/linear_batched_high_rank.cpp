#include "forch/backend/forch-ge.hpp"
#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/nn/linear.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace forch;

static bool almost_equal(
  const float *a, const float *b, size_t n, float tol = 1e-5F) {
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) / (std::fabs(b[i]) + 1e-5F) > tol) {
      return false;
    }
  }
  return true;
}

struct LinearLoader : public nn::Loader {
  std::vector<float> weight_data = {
    1.0F,
    0.0F,
    0.0F,
    0.0F,
    0.0F,
    1.0F,
    0.0F,
    0.0F,
    0.0F,
    0.0F,
    1.0F,
    1.0F,
  }; // [3, 4]
  std::vector<float> bias_data = {0.5F, -1.0F, 2.0F};

  Tensor load(const std::string &name) override {
    if (name == "linear.weight") {
      return op::load_from_vector(std::vector<float>(weight_data), {3, 4})
        .set_name(name);
    }
    if (name == "linear.bias") {
      return op::load_from_vector(std::vector<float>(bias_data), {3})
        .set_name(name);
    }

    throw exception(std::string("unexpected parameter name: ") + name);
  }
};

int test_linear_batched_high_rank(Backend &backend) {
  // input shape: [B0, B1, B2, T, C] where batch dims count is 3 (> 2)
  const Shape input_shape  = {2, 2, 2, 3, 4};
  const Shape output_shape = {2, 2, 2, 3, 3};

  std::vector<float> input_data(2UL * 2 * 2 * 3 * 4);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i) * 0.1F;
  }

  std::vector<float> expected(2UL * 2 * 2 * 3 * 3);
  for (size_t i = 0; i < expected.size() / 3; ++i) {
    const float x0      = input_data[i * 4 + 0];
    const float x1      = input_data[i * 4 + 1];
    const float x2      = input_data[i * 4 + 2];
    const float x3      = input_data[i * 4 + 3];
    expected[i * 3 + 0] = x0 + 0.5F;
    expected[i * 3 + 1] = x1 - 1.0F;
    expected[i * 3 + 2] = x2 + x3 + 2.0F;
  }

  auto input = op::create_input(input_shape, DType::FP32).set_name("input");

  nn::Linear   linear(4, 3, true, DType::FP32);
  LinearLoader loader;
  linear.load(&loader, "linear");

  auto output = linear.forward(input);
  output.set_name(std::string("output"));
  if (output.shape() != output_shape) {
    std::cerr << "linear high-rank: output shape mismatch" << std::endl;
    return 1;
  }

  forch::Graph graph_ir({output});
  graph_ir.set_name("test_linear_batched_high_rank");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "linear high-rank: compile/load failed: " << e.what()
              << std::endl;
    return 2;
  }

  auto runtime_input =
    op::load_from_vector(std::move(input_data), input_shape).set_name("input");
  auto outputs = graph.run({runtime_input});
  if (outputs.empty()) {
    return 3;
  }

  auto out_tensor = outputs.get(std::string("output"));
  if (out_tensor.shape() != output_shape) {
    std::cerr << "linear high-rank: runtime output shape mismatch" << std::endl;
    return 4;
  }

  const auto *out_ptr = reinterpret_cast<const float *>(out_tensor.get_data());
  if (!almost_equal(out_ptr, expected.data(), expected.size(), 1e-3F)) {
    std::cerr << "linear high-rank: output value mismatch" << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::fabs(out_ptr[i] - expected[i]) > 1e-4F) {
        std::cerr << "first mismatch at " << i << ": out=" << out_ptr[i]
                  << ", expected=" << expected[i] << std::endl;
        break;
      }
    }
    return 5;
  }

  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  const int r = test_linear_batched_high_rank(backend);
  if (r != 0) {
    return r;
  }

  std::cout << "Linear high-rank batched test passed." << std::endl;
  return 0;
}
