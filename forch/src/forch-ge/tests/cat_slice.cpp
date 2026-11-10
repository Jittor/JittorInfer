#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/cat.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/slice.hpp"

#include <cmath>
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

int test_cat(Backend &backend) {
  // concat along dim 1: shapes {2,2} + {2,3} -> {2,5}
  Shape              a_shape   = {2, 2};
  Shape              b_shape   = {2, 3};
  std::vector<float> a_data    = {1, 2, 3, 4};
  std::vector<float> b_data    = {5, 6, 7, 8, 9, 10};
  Shape              out_shape = {2, 5};

  auto a   = op::create_input(a_shape, DType::FP32).set_name("a_cat");
  auto b   = op::create_input(b_shape, DType::FP32).set_name("b_cat");
  auto out = op::cat(1)(a, b);
  out.set_name(std::string("cat_output"));

  forch::Graph graph_ir({out});
  graph_ir.set_name("test_cat");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "cat: compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  auto a_tensor =
    op::load_from_vector(std::move(a_data), a_shape).set_name("a_cat");
  auto b_tensor =
    op::load_from_vector(std::move(b_data), b_shape).set_name("b_cat");

  auto outputs = graph.run({a_tensor, b_tensor});
  if (outputs.empty())
    return 3;

  const float *out_ptr = reinterpret_cast<const float *>(
    outputs.get(std::string("cat_output")).get_data());

  std::vector<float> expected = {1, 2, 5, 6, 7, 3, 4, 8, 9, 10};
  // expected laid out row-major: row0: [1,2,5,6,7], row1: [3,4,8,9,10]

  if (outputs.get(std::string("cat_output")).shape() != out_shape) {
    std::cerr << "cat: shape mismatch" << std::endl;
    auto shape = outputs.get(std::string("cat_output")).shape();
    for (size_t i = 0; i < shape.size(); ++i) {
      std::cerr << shape[i] << (i + 1 < shape.size() ? "," : "\n");
    }
    return 4;
  }
  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "cat: values differ" << std::endl;
    return 5;
  }

  return 0;
}

int test_slice(Backend &backend) {
  // input 3x4, slice begin {1,1}, end {3,3} -> output 2x2
  Shape              in_shape  = {3, 4};
  std::vector<float> in_data   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Shape              begin     = {1, 1};
  Shape              end       = {3, 3};
  Shape              out_shape = {2, 2};

  auto input = op::create_input(in_shape, DType::FP32).set_name("input_slice");
  auto out   = op::slice(begin, end)(input);
  out.set_name(std::string("slice_output"));

  forch::Graph graph_ir({out});
  graph_ir.set_name("test_slice");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "slice: compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  auto in_tensor =
    op::load_from_vector(std::move(in_data), in_shape).set_name("input_slice");
  auto outputs = graph.run({in_tensor});
  if (outputs.empty())
    return 3;

  const float *out_ptr = reinterpret_cast<const float *>(
    outputs.get(std::string("slice_output")).get_data());

  // expected slice (rows 1..2, cols 1..2): [[6,7],[10,11]]
  std::vector<float> expected = {6, 7, 10, 11};

  if (outputs.get(std::string("slice_output")).shape() != out_shape) {
    std::cerr << "slice: shape mismatch" << std::endl;
    return 4;
  }
  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "slice: values differ" << std::endl;
    return 5;
  }

  return 0;
}

int test_rotate_half_by_slice_cat(Backend &backend) {
  // rotate half on dim 1 by swapping two equal halves using slice + cat
  // input shape {2,4}: [x1, x2] -> [x2, x1]
  Shape              in_shape  = {2, 4};
  std::vector<float> in_data   = {1, 2, 3, 4, 5, 6, 7, 8};
  Shape              out_shape = {2, 4};

  auto input =
    op::create_input(in_shape, DType::FP32).set_name("input_rotate_half");
  auto first_half =
    op::slice({op::Slice::Ellipsis{}, op::Slice::IntervalFromBegin{2}})(input);
  auto second_half = op::slice({op::Slice::IntervalAll{}, op::Slice::Ellipsis{},
    op::Slice::IntervalTillEnd{-2}})(input);
  auto out         = op::cat(1)(second_half | op::neg, first_half);
  if (out.shape() != out_shape) {
    std::cerr << "rotate_half: output shape mismatch" << std::endl;
    for (size_t i = 0; i < out.shape().size(); ++i) {
      std::cerr << out.shape()[i] << (i + 1 < out.shape().size() ? "," : "\n");
    }
    return 1;
  }

  out.set_name(std::string("rotate_output"));
  forch::Graph graph_ir({out});
  graph_ir.set_name("test_rotate_half_by_slice_cat");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "rotate_half: compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  auto in_tensor = op::load_from_vector(std::move(in_data), in_shape)
                     .set_name("input_rotate_half");
  auto outputs = graph.run({in_tensor});
  if (outputs.empty())
    return 3;

  const float *out_ptr = reinterpret_cast<const float *>(
    outputs.get(std::string("rotate_output")).get_data());

  std::vector<float> expected = {-3, -4, 1, 2, -7, -8, 5, 6};

  if (outputs.get(std::string("rotate_output")).shape() != out_shape) {
    std::cerr << "rotate_half: shape mismatch" << std::endl;
    return 4;
  }
  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "rotate_half: values differ" << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
      std::cerr << out_ptr[i] << " vs " << expected[i] << std::endl;
    }
    return 5;
  }

  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  int r = 0;
  r     = test_cat(backend);
  if (r != 0)
    return r;
  r = test_slice(backend);
  if (r != 0)
    return r;
  r = test_rotate_half_by_slice_cat(backend);
  if (r != 0)
    return r;

  std::cout << "Cat and Slice tests passed." << std::endl;
  return 0;
}
