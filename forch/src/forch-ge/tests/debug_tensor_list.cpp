#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/input.hpp"
#include "forch/pass/debug.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace forch;

int main() {
  GEBackend ge_impl(AscendDevice::from_env());
  Backend   backend = ge_impl;

  // small 2x2 test
  Shape              shape      = {2, 2};
  std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F};

  auto in = op::create_input(shape, DType::FP32).set_name("in");

  // intermediate tensor we want to collect as debug output
  auto debug = op::add(in, in).set_name("debug.add");
  auto out   = op::add(debug, in).set_name("final.out");

  // Build graph and append debug tensor via graph pass.
  forch::Graph graph_ir({out});
  graph_ir.set_name("test_debug_tensor_list");
  graph_ir   = pass::AppendDebugTensor({"debug.add"})(graph_ir);
  auto graph = backend.build(graph_ir);

  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  // feed runtime input (pass a copy of input_data)
  auto runtime_in =
    op::load_from_vector(std::vector<float>(input_data), shape).set_name("in");

  auto outputs = graph.run({runtime_in});
  if (outputs.empty()) {
    std::cerr << "outputs empty" << std::endl;
    return 3;
  }

  try {
    auto debug_out = outputs.get(std::string("debug.add"));
    auto final_out = outputs.get(std::string("final.out"));

    if (debug_out.shape() != shape) {
      std::cerr << "debug shape mismatch" << std::endl;
      return 4;
    }
    if (final_out.shape() != shape) {
      std::cerr << "final shape mismatch" << std::endl;
      return 5;
    }

    const float *debug_ptr =
      reinterpret_cast<const float *>(debug_out.get_data());
    const float *final_ptr =
      reinterpret_cast<const float *>(final_out.get_data());

    // expected: debug = 2 * input, final = 3 * input
    for (size_t i = 0; i < input_data.size(); ++i) {
      float exp_dbg = input_data[i] * 2.0F;
      float exp_fin = input_data[i] * 3.0F;
      if (std::fabs(debug_ptr[i] - exp_dbg) > 1e-3F) {
        std::cerr << "debug value mismatch at " << i << std::endl;
        return 6;
      }
      if (std::fabs(final_ptr[i] - exp_fin) > 1e-3F) {
        std::cerr << "final value mismatch at " << i << std::endl;
        return 7;
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "lookup failed: " << e.what() << std::endl;
    return 8;
  }

  std::cout << "Debug tensor list test passed." << std::endl;
  return 0;
}
