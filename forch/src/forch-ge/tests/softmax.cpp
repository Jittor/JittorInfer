#include "forch/ops/softmax.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace forch;

static bool almost_equal(
  const float *a, const float *b, size_t n, float tol = 1e-5F) {
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) > tol) {
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

static std::vector<float> softmax_reference(
  const std::vector<float> &input, const Shape &shape, int64_t dim) {
  int64_t rank = static_cast<int64_t>(shape.size());
  if (dim < 0) {
    dim += rank;
  }

  size_t outer = 1;
  for (int64_t i = 0; i < dim; ++i) {
    outer *= static_cast<size_t>(shape[i]);
  }

  size_t axis = static_cast<size_t>(shape[dim]);

  size_t inner = 1;
  for (int64_t i = dim + 1; i < rank; ++i) {
    inner *= static_cast<size_t>(shape[i]);
  }

  std::vector<float> out(input.size());
  for (size_t o = 0; o < outer; ++o) {
    for (size_t in = 0; in < inner; ++in) {
      size_t base = o * axis * inner + in;

      float max_v = -std::numeric_limits<float>::infinity();
      for (size_t a = 0; a < axis; ++a) {
        float v = input[base + a * inner];
        if (v > max_v) {
          max_v = v;
        }
      }

      float denom = 0.0F;
      for (size_t a = 0; a < axis; ++a) {
        float e               = std::exp(input[base + a * inner] - max_v);
        out[base + a * inner] = e;
        denom += e;
      }

      for (size_t a = 0; a < axis; ++a) {
        out[base + a * inner] /= denom;
      }
    }
  }
  return out;
}

int run_softmax_case(Backend &backend, const Shape &shape,
  std::vector<float> &data, int64_t dim, const char *case_name) {
  auto        input_name = std::string(case_name) + "_input";
  auto        input = op::create_input(shape, DType::FP32).set_name(input_name);
  auto        out   = op::softmax(dim)(input);
  std::string out_name = std::string(case_name) + "_output";
  out.set_name(out_name);

  if (out.shape() != shape) {
    std::cerr << case_name << ": output shape mismatch" << std::endl;
    return 1;
  }

  forch::Graph graph_ir({out});
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

  auto expected = softmax_reference(data, shape, dim);
  auto in_tensor =
    op::load_from_vector(std::move(data), shape).set_name(input_name);
  auto outputs = graph.run({in_tensor});
  if (outputs.empty()) {
    return 3;
  }

  auto out_tensor = outputs.get(out_name);
  if (out_tensor.shape() != shape) {
    std::cerr << case_name << ": runtime shape mismatch" << std::endl;
    return 4;
  }

  const float *out_ptr = reinterpret_cast<const float *>(out_tensor.get_data());
  constexpr float tol  = 1e-3F;
  if (!almost_equal(out_ptr, expected.data(), expected.size(), tol)) {
    std::cerr << case_name << ": values differ" << std::endl;
    size_t n = numel(shape);
    for (size_t i = 0; i < n; ++i) {
      if (std::fabs(out_ptr[i] - expected[i]) > tol) {
        std::cerr << "first mismatch at " << i << ": out=" << out_ptr[i]
                  << ", expected=" << expected[i] << std::endl;
        break;
      }
    }
    return 5;
  }

  return 0;
}

int test_softmax_multi_dim(Backend &backend) {
  {
    Shape              shape = {5};
    std::vector<float> data  = {1.0F, 2.0F, -1.0F, 3.5F, 0.5F};
    int r = run_softmax_case(backend, shape, data, 0, "softmax_1d_dim0");
    if (r != 0) {
      return r;
    }
  }

  {
    Shape              shape = {2, 4};
    std::vector<float> data  = {
      1.0F,
      2.0F,
      3.0F,
      4.0F,
      -2.0F,
      0.0F,
      2.0F,
      6.0F,
    };
    int r = run_softmax_case(backend, shape, data, 1, "softmax_2d_dim1");
    if (r != 0) {
      return r;
    }
  }

  {
    Shape              shape = {2, 3, 2};
    std::vector<float> data  = {
      1.0F,
      2.0F,
      0.0F,
      -1.0F,
      3.0F,
      1.0F,
      4.0F,
      -2.0F,
      2.0F,
      0.5F,
      -1.0F,
      3.0F,
    };
    int r = run_softmax_case(backend, shape, data, 1, "softmax_3d_dim1");
    if (r != 0) {
      return r;
    }
  }

  {
    Shape              shape = {2, 2, 3};
    std::vector<float> data  = {
      1.0F,
      2.0F,
      3.0F,
      -1.0F,
      0.0F,
      1.0F,
      4.0F,
      3.0F,
      2.0F,
      0.0F,
      -2.0F,
      -4.0F,
    };
    int r = run_softmax_case(backend, shape, data, 2, "softmax_3d_dim2");
    if (r != 0) {
      return r;
    }
  }

  // Cases involving -inf inputs: entries that are -inf should produce 0.
  {
    Shape              shape = {4};
    std::vector<float> data  = {1.0F, -std::numeric_limits<float>::infinity(),
       0.0F, -std::numeric_limits<float>::infinity()};
    int                r =
      run_softmax_case(backend, shape, data, 0, "softmax_1d_with_neg_inf");
    if (r != 0) {
      return r;
    }
  }

  {
    Shape              shape = {2, 3};
    std::vector<float> data  = {
      1.0F,
      -std::numeric_limits<float>::infinity(),
      0.0F,
      -1.0F,
      2.0F,
      -std::numeric_limits<float>::infinity(),
    };
    int r =
      run_softmax_case(backend, shape, data, 1, "softmax_2d_with_neg_inf_dim1");
    if (r != 0) {
      return r;
    }
  }

  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  int r = test_softmax_multi_dim(backend);
  if (r != 0) {
    return r;
  }

  std::cout << "Softmax tests passed." << std::endl;
  return 0;
}
