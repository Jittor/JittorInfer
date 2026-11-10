#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/exp.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/rsqrt.hpp"
#include "forch/ops/sigmoid.hpp"
#include "forch/ops/softplus.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace forch;

static size_t numel(const Shape &shape) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

static uint16_t fp32_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb           = (bits >> 16U) & 1U;
  const uint32_t rounding_bias = 0x7fffU + lsb;
  return static_cast<uint16_t>((bits + rounding_bias) >> 16U);
}

static float bf16_to_fp32(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float          out  = 0.0F;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static bool close_enough(
  float actual, float expected, float abs_tol, float rel_tol) {
  // 实际值若为 NaN/inf 直接判失败:fabs 相减无法察觉 NaN
  if (!std::isfinite(actual)) {
    return false;
  }
  const auto diff = std::fabs(actual - expected);
  return diff <= abs_tol + rel_tol * std::fabs(expected);
}

using RefFn = float (*)(float);

/// 对单个单输入逐元素算子跑一种 dtype 的 pointwise 用例,
/// reference 为 <cmath> 独立参考实现。
static int run_unary_case(Backend &backend, const char *op_name,
  const std::function<Tensor(const Tensor &)> &build, RefFn reference,
  const Shape &shape, const std::vector<float> &data, DType dtype,
  float rel_tol) {
  const auto case_name =
    std::string(op_name) + (dtype == DType::FP32 ? "_fp32" : "_bf16");
  const auto input_name  = case_name + "_input";
  const auto output_name = case_name + "_output";

  auto input = op::create_input(shape, dtype).set_name(input_name);
  auto out   = build(input);
  out.set_name(output_name);

  // 声明期元数据:shape 与输入一致,dtype 跟随输入(不隐式升精度)
  if (out.shape() != shape) {
    std::cerr << case_name << ": declared shape mismatch" << std::endl;
    return 1;
  }
  if (out.dtype() != dtype) {
    std::cerr << case_name << ": declared dtype mismatch" << std::endl;
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

  const auto n = numel(shape);

  // 参考值基于量化后的输入计算:kernel 实际看到的就是量化后的数
  std::vector<float> quantized(data);
  if (dtype == DType::BF16) {
    for (size_t i = 0; i < n; ++i) {
      quantized[i] = bf16_to_fp32(fp32_to_bf16(data[i]));
    }
  }
  std::vector<float> expected(n);
  for (size_t i = 0; i < n; ++i) {
    expected[i] = reference(quantized[i]);
  }

  if (dtype == DType::FP32) {
    auto quant_copy = quantized;
    auto in_tensor =
      op::load_from_vector(std::move(quant_copy), shape).set_name(input_name);
    auto outputs    = graph.run({in_tensor});
    auto out_tensor = outputs.get(output_name);
    if (out_tensor.shape() != shape || out_tensor.dtype() != DType::FP32) {
      std::cerr << case_name << ": runtime shape/dtype mismatch" << std::endl;
      return 4;
    }
    const float *ptr = reinterpret_cast<const float *>(out_tensor.get_data());
    for (size_t i = 0; i < n; ++i) {
      if (!close_enough(ptr[i], expected[i], 1e-5F, rel_tol)) {
        std::cerr << case_name << ": value mismatch at " << i
                  << ": out=" << ptr[i] << ", expected=" << expected[i]
                  << std::endl;
        return 5;
      }
    }
  } else {
    std::vector<uint16_t> in_bf16(n);
    for (size_t i = 0; i < n; ++i) {
      in_bf16[i] = fp32_to_bf16(data[i]);
    }
    auto in_tensor = op::load_from_buffer(
      reinterpret_cast<Buffer>(in_bf16.data()), shape, DType::BF16)
                       .set_name(input_name);
    auto outputs    = graph.run({in_tensor});
    auto out_tensor = outputs.get(output_name);
    if (out_tensor.shape() != shape || out_tensor.dtype() != DType::BF16) {
      std::cerr << case_name << ": runtime shape/dtype mismatch" << std::endl;
      return 4;
    }
    const uint16_t *ptr =
      reinterpret_cast<const uint16_t *>(out_tensor.get_data());
    for (size_t i = 0; i < n; ++i) {
      if (!close_enough(bf16_to_fp32(ptr[i]), expected[i], 1e-3F, rel_tol)) {
        std::cerr << case_name << ": value mismatch at " << i
                  << ": out=" << bf16_to_fp32(ptr[i])
                  << ", expected=" << expected[i] << std::endl;
        return 5;
      }
    }
  }

  std::cout << "  [" << case_name << "] ok" << std::endl;
  return 0;
}

static float ref_sigmoid(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}
static float ref_softplus(float x) {
  // 与 torch 一致的数值稳定形式
  return x > 20.0F ? x : std::log1p(std::exp(x));
}
static float ref_exp(float x) {
  return std::exp(x);
}
static float ref_rsqrt(float x) {
  return 1.0F / std::sqrt(x);
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  const Shape shape = {2, 5};
  // sigmoid/softplus 覆盖正负区间;exp 控制在 [-2, 2] 防溢出;
  // rsqrt 取正值且远离 0
  const std::vector<float> data_mid = {
    -3.5F, -1.25F, -0.5F, 0.0F, 0.75F, 1.5F, 2.75F, 4.0F, -2.25F, 3.25F};
  const std::vector<float> data_exp = {
    -2.0F, -1.5F, -0.75F, -0.25F, 0.0F, 0.25F, 0.75F, 1.5F, 2.0F, 1.0F};
  const std::vector<float> data_pos = {
    0.25F, 0.5F, 1.0F, 2.0F, 3.0F, 4.0F, 5.5F, 7.0F, 8.0F, 6.25F};

  struct UnaryOp {
    const char                           *name;
    std::function<Tensor(const Tensor &)> build;
    RefFn                                 reference;
    const std::vector<float>             *data;
    float                                 rel_tol;
  };
  // bf16 输出量化误差 ~0.4%,留 1e-2 相对容差;fp32 用 1e-5
  const std::vector<UnaryOp> ops = {
    {"sigmoid", [](const Tensor &t) { return op::sigmoid()(t); }, ref_sigmoid,
      &data_mid, 1e-2F},
    {"softplus", [](const Tensor &t) { return op::softplus()(t); },
      ref_softplus, &data_mid, 1e-2F},
    {"exp", [](const Tensor &t) { return op::exp()(t); }, ref_exp, &data_exp,
      1e-2F},
    {"rsqrt", [](const Tensor &t) { return op::rsqrt()(t); }, ref_rsqrt,
      &data_pos, 1e-2F},
  };

  for (const auto &entry : ops) {
    // 需求:每种算子 fp32 与 bf16 输入各一
    for (auto dtype : {DType::FP32, DType::BF16}) {
      const float tol = dtype == DType::FP32 ? 1e-5F : entry.rel_tol;
      if (int rc = run_unary_case(backend, entry.name, entry.build,
            entry.reference, shape, *entry.data, dtype, tol);
          rc != 0) {
        return rc;
      }
    }
  }

  std::cout << "Unary math tests passed." << std::endl;
  return 0;
}
