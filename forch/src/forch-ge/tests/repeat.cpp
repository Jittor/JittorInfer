#include "forch/ops/repeat.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

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

int test_repeat(Backend &backend) {
  // input 2x3 repeated along first dim by 2 -> 4x3
  Shape              shape     = {2, 3};
  std::vector<float> data      = {1, 2, 3, 4, 5, 6};
  Shape              repeats   = {3, 2};
  Shape              out_shape = {6, 6};

  auto input = op::create_input(shape, DType::FP32).set_name("input_repeat");
  auto out   = op::repeat(repeats)(input);
  out.set_name(std::string("repeat_output"));

  if (out.shape() != out_shape) {
    std::cerr << "repeat: output shape mismatch" << std::endl;
    return 1;
  }

  forch::Graph graph_ir({out});
  graph_ir.set_name("test_repeat");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "repeat: compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  // Build expected output before moving input data into Const tensor.
  std::vector<float> expected(out_shape[0] * out_shape[1]);
  for (int64_t i = 0; i < out_shape[0]; ++i)
    for (int64_t j = 0; j < out_shape[1]; ++j)
      expected[i * out_shape[1] + j] =
        data[(i % shape[0]) * shape[1] + (j % shape[1])];

  auto input_tensor =
    op::load_from_vector(std::move(data), shape).set_name("input_repeat");
  auto outputs = graph.run({input_tensor});
  if (outputs.empty())
    return 3;

  const float *out_ptr = reinterpret_cast<const float *>(
    outputs.get(std::string("repeat_output")).get_data());

  if (outputs.get(std::string("repeat_output")).shape() != out_shape) {
    std::cerr << "repeat: shape mismatch" << std::endl;
    return 4;
  }
  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "repeat: values differ" << std::endl;
    // output out_ptr and expected.data()
    for (size_t i = 0; i < expected.size(); ++i) {
      std::cerr << "out[" << i << "] = " << out_ptr[i] << ", expected[" << i
                << "] = " << expected[i] << std::endl;
    }
    return 5;
  }

  return 0;
}

int main() {
  auto backend = GEBackend(AscendDevice::from_env());
  int  r       = test_repeat(backend);
  if (r != 0)
    return r;
  std::cout << "Repeat test passed." << std::endl;
  return 0;
}
