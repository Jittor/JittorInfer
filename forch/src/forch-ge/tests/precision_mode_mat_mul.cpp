#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/mat_mul.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace forch;

namespace {

using PrecisionMode = GEBackend::GEBuildOptions::PrecisionMode;

const char *to_string(PrecisionMode mode) {
  switch (mode) {
  case PrecisionMode::Origin:
    return "Origin(FP32)";
  case PrecisionMode::AutoToFP16:
    return "AutoToFP16";
  case PrecisionMode::AutoToBF16:
    return "AutoToBF16";
  }
  return "Unknown";
}

std::vector<float> cpu_mat_mul_2x2(
  const std::vector<float> &a, const std::vector<float> &b) {
  std::vector<float> out(4, 0.0F);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      float sum = 0.0F;
      for (int k = 0; k < 2; ++k) {
        sum += a[i * 2 + k] * b[k * 2 + j];
      }
      out[i * 2 + j] = sum;
    }
  }
  return out;
}

float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
  float max_diff = 0.0F;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
  }
  return max_diff;
}

void print_2x2(const char *title, const std::vector<float> &m) {
  std::cout << title << "\n";
  std::cout << std::fixed << std::setprecision(9);
  std::cout << "  [" << std::setw(13) << m[0] << ", " << std::setw(13) << m[1]
            << "]\n";
  std::cout << "  [" << std::setw(13) << m[2] << ", " << std::setw(13) << m[3]
            << "]\n";
}

int run_mode(GEBackend &backend, PrecisionMode mode,
  const std::vector<float> &a_data, const std::vector<float> &b_data,
  std::vector<float> *out) {
  const Shape a_shape = {2, 2};
  const Shape b_shape = {2, 2};

  const std::string mode_name = to_string(mode);
  const std::string a_name = std::string("precision_mode.") + mode_name + ".A";
  const std::string b_name = std::string("precision_mode.") + mode_name + ".B";
  const std::string y_name = std::string("precision_mode.") + mode_name + ".Y";

  auto a = op::create_input(a_shape, DType::FP32).set_name(a_name);
  auto b = op::create_input(b_shape, DType::FP32).set_name(b_name);
  auto y = op::mat_mul()(a, b).set_name(y_name);

  forch::Graph graph_ir({y});
  graph_ir.set_name(std::string("precision_mode_mat_mul.") + mode_name);

  auto graph = backend.build(graph_ir, {.precision_mode = mode});
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "[" << mode_name << "] compile/load failed: " << e.what()
              << std::endl;
    return 1;
  }

  auto a_tensor =
    op::load_from_vector(std::vector<float>(a_data), a_shape).set_name(a_name);
  auto b_tensor =
    op::load_from_vector(std::vector<float>(b_data), b_shape).set_name(b_name);

  auto outputs = graph.run({a_tensor, b_tensor});
  if (outputs.empty()) {
    std::cerr << "[" << mode_name << "] outputs is empty" << std::endl;
    return 2;
  }

  const auto out_tensor = outputs.get(y_name);
  if (out_tensor.shape() != Shape({2, 2})) {
    std::cerr << "[" << mode_name << "] output shape mismatch" << std::endl;
    return 3;
  }
  if (out_tensor.dtype() != DType::FP32) {
    std::cerr << "[" << mode_name << "] output dtype is not FP32" << std::endl;
    return 4;
  }

  const auto *ptr = reinterpret_cast<const float *>(out_tensor.get_data());
  out->assign(ptr, ptr + 4);
  return 0;
}

} // namespace

int main() {
  // Test motivation:
  // - Validate GEBackend::GEBuildOptions::precision_mode really changes
  //   matmul compute precision while keeping FP32 input/output interfaces.
  //
  // Principle for this crafted 2x2x2 matmul:
  // - A is close to 1.0 with tiny offsets +/-3/4096, and B uses large scale
  //   4096. In FP32 this keeps the tiny offsets and yields exactly +/-3.
  // - With AutoToFP16, offsets are rounded to +/-4/4096 before compute,
  //   producing +/-4.
  // - With AutoToBF16, offsets are rounded away to 0 before compute,
  //   producing 0.
  //
  // This makes FP32 / FP16 / BF16 output differences visible in one run.
  const float              delta  = 3.0F / 4096.0F;
  const std::vector<float> a_data = {
    1.0F + delta,
    1.0F,
    1.0F,
    1.0F - delta,
  };
  const std::vector<float> b_data = {
    4096.0F,
    -4096.0F,
    -4096.0F,
    4096.0F,
  };

  const auto ref_fp32 = cpu_mat_mul_2x2(a_data, b_data);

  std::vector<float> out_origin;
  std::vector<float> out_fp16;
  std::vector<float> out_bf16;

  GEBackend backend(AscendDevice::from_env());

  int rc =
    run_mode(backend, PrecisionMode::Origin, a_data, b_data, &out_origin);
  if (rc != 0) {
    return 10 + rc;
  }
  rc = run_mode(backend, PrecisionMode::AutoToFP16, a_data, b_data, &out_fp16);
  if (rc != 0) {
    return 20 + rc;
  }
  rc = run_mode(backend, PrecisionMode::AutoToBF16, a_data, b_data, &out_bf16);
  if (rc != 0) {
    return 30 + rc;
  }

  std::cout << "==== PrecisionMode 2x2x2 MatMul Test ====\n";
  std::cout << "Input/Output dtype are FP32. Only compute precision policy "
               "changes by precision_mode.\n";
  print_2x2("A:", a_data);
  print_2x2("B:", b_data);
  print_2x2("CPU FP32 reference (expected near +/-3):", ref_fp32);
  print_2x2("GE Origin output:", out_origin);
  print_2x2("GE AutoToFP16 output:", out_fp16);
  print_2x2("GE AutoToBF16 output:", out_bf16);

  const float diff_origin_ref = max_abs_diff(out_origin, ref_fp32);
  const float diff_fp16_ref   = max_abs_diff(out_fp16, ref_fp32);
  const float diff_bf16_ref   = max_abs_diff(out_bf16, ref_fp32);

  std::cout << "max|Origin - FP32_ref|    = " << diff_origin_ref << "\n";
  std::cout << "max|AutoToFP16 - FP32_ref| = " << diff_fp16_ref << "\n";
  std::cout << "max|AutoToBF16 - FP32_ref| = " << diff_bf16_ref << "\n";

  /*
  Standard output:
  max|Origin - FP32_ref|     = 0.000000000
  max|AutoToFP16 - FP32_ref| = 1.000000000
  max|AutoToBF16 - FP32_ref| = 3.000000000
  */

  if (diff_origin_ref > 1e-5F) {
    std::cerr << "Origin mode output differs from FP32 reference more than "
                 "tolerance.\n";
    return 1;
  }
  if (diff_fp16_ref < 0.99F || diff_fp16_ref > 1.01F) {
    std::cerr << "AutoToFP16 mode output does not differ from FP32 reference "
                 "by about 1.0 as expected.\n";
    return 2;
  }
  if (diff_bf16_ref < 2.99F || diff_bf16_ref > 3.01F) {
    std::cerr << "AutoToBF16 mode output does not differ from FP32 reference "
                 "by about 3.0 as expected.\n";
    return 3;
  }

  return 0;
}