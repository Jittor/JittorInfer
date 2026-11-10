#include "forch/ops/rotary_position_embedding.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace forch;

static bool almost_equal(
  const float *a, const float *b, size_t n, float tol = 1e-1F) {
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) / (std::fabs(std::abs(a[i])) + 1) > tol) {
      std::cerr << "first mismatch at index " << i << ": got=" << a[i]
                << ", expected=" << b[i] << std::endl;
      return false;
    }
  }
  return true;
}

static std::vector<float> rotary_reference(const std::vector<float> &x,
  const std::vector<float> &cos, const std::vector<float> &sin,
  const Shape &shape, op::RotaryPositionEmbedding::Mode mode) {
  const size_t hidden_size = static_cast<size_t>(shape.back());
  const size_t total       = x.size();
  const size_t blocks      = total / hidden_size;

  std::vector<float> x_rotate(total, 0.0F);
  std::vector<float> out(total, 0.0F);

  for (size_t b = 0; b < blocks; ++b) {
    const size_t base = b * hidden_size;

    if (mode == op::RotaryPositionEmbedding::Mode::Half) {
      const size_t half = hidden_size / 2;
      for (size_t i = 0; i < half; ++i) {
        const float x1            = x[base + i];
        const float x2            = x[base + half + i];
        x_rotate[base + i]        = -x2;
        x_rotate[base + half + i] = x1;
      }
    } else if (mode == op::RotaryPositionEmbedding::Mode::Interleave) {
      const size_t half = hidden_size / 2;
      for (size_t i = 0; i < half; ++i) {
        const size_t even_i = base + 2 * i;
        const size_t odd_i  = even_i + 1;
        x_rotate[even_i]    = -x[odd_i];
        x_rotate[odd_i]     = x[even_i];
      }
    } else {
      throw std::runtime_error(
        "test reference only supports half/interleave modes");
    }

    for (size_t i = 0; i < hidden_size; ++i) {
      const size_t idx = base + i;
      out[idx]         = x[idx] * cos[idx] + x_rotate[idx] * sin[idx];
    }
  }

  return out;
}

static int run_case(GEBackend &backend, op::RotaryPositionEmbedding::Mode mode,
  const char *name) {
  const Shape shape = {1, 2, 8};

  std::vector<float> x = {
    1.0F,
    2.0F,
    3.0F,
    4.0F,
    5.0F,
    6.0F,
    7.0F,
    8.0F,
    9.0F,
    10.0F,
    11.0F,
    12.0F,
    13.0F,
    14.0F,
    15.0F,
    16.0F,
  };
  std::vector<float> cos(x.size(), 0.8F);
  std::vector<float> sin(x.size(), 0.6F);

  const auto expected = rotary_reference(x, cos, sin, shape, mode);

  const auto x_name   = std::string(name) + ".x";
  const auto cos_name = std::string(name) + ".cos";
  const auto sin_name = std::string(name) + ".sin";
  const auto out_name = std::string(name) + ".out";

  auto x_input   = op::create_input(shape, DType::FP32).set_name(x_name);
  auto cos_input = op::create_input(shape, DType::FP32).set_name(cos_name);
  auto sin_input = op::create_input(shape, DType::FP32).set_name(sin_name);

  auto out = op::rotary_position_embedding(mode)(x_input, cos_input, sin_input)
               .set_name(out_name);

  if (out.shape() != shape) {
    std::cerr << name << ": output shape mismatch" << std::endl;
    return 1;
  }

  auto graph = backend.build(forch::Graph({out}).set_name(name),
    {
      .precision_mode = GEBackend::GEBuildOptions::PrecisionMode::AutoToFP16,
    });
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << name << ": compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  auto x_tensor = op::load_from_vector(std::move(x), shape).set_name(x_name);
  auto cos_tensor =
    op::load_from_vector(std::move(cos), shape).set_name(cos_name);
  auto sin_tensor =
    op::load_from_vector(std::move(sin), shape).set_name(sin_name);

  auto outputs = graph.run({x_tensor, cos_tensor, sin_tensor});
  if (outputs.empty()) {
    std::cerr << name << ": no outputs" << std::endl;
    return 3;
  }

  const auto out_tensor = outputs.get(out_name);
  if (out_tensor.shape() != shape) {
    std::cerr << name << ": runtime shape mismatch" << std::endl;
    return 4;
  }

  const auto *out_ptr = reinterpret_cast<const float *>(out_tensor.get_data());
  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    return 5;
  }

  return 0;
}

int main() {
  GEBackend backend(AscendDevice::from_env());

  int ret = run_case(
    backend, op::RotaryPositionEmbedding::Mode::Half, "rope_half_mode");
  if (ret != 0) {
    return ret;
  }

  ret = run_case(backend, op::RotaryPositionEmbedding::Mode::Interleave,
    "rope_interleave_mode");
  if (ret != 0) {
    return ret;
  }

  std::cout << "RotaryPositionEmbedding tests passed." << std::endl;
  return 0;
}