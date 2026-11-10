#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/mat_mul.hpp"
#include "forch/ops/rms_norm.hpp"

#include "forch/backend/forch-ge.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace forch;

// Helper to create a randomized weight buffer
std::vector<float> create_random_buffer(int size, float range = 0.1f) {
  std::vector<float>                    buffer(size);
  std::default_random_engine            generator;
  std::uniform_real_distribution<float> distribution(-range, range);
  for (int i = 0; i < size; ++i) {
    buffer[i] = distribution(generator);
  }
  return buffer;
}

struct Layer {
  std::vector<float> weight_data;
  std::vector<float> bias_data;
  std::vector<float> gamma_data;
  Tensor             weight;
  Tensor             bias;
  Tensor             gamma;
  std::string        name;

  Layer(int in_features, int out_features, const std::string &name)
      : weight_data(create_random_buffer(in_features * out_features)),
        bias_data(create_random_buffer(out_features)),
        gamma_data(create_random_buffer(out_features, 2)),
        weight(
          op::load_from_buffer(reinterpret_cast<Buffer>(weight_data.data()),
            {out_features, in_features}, DType::FP32)
            .set_name(name + ".weight")),
        bias(op::load_from_buffer(reinterpret_cast<Buffer>(bias_data.data()),
          {out_features, 1}, DType::FP32)
               .set_name(name + ".bias")),
        gamma(op::load_from_buffer(reinterpret_cast<Buffer>(gamma_data.data()),
          {out_features, 1}, DType::FP32)
                .set_name(name + ".norm")),
        name(name) {}

  Tensor forward(Tensor input) {
    auto mm   = op::mat_mul()(weight, input).set_name(name + ".mm");
    auto add  = op::add(mm, bias).set_name(name + ".add");
    auto silu = op::silu(add).set_name(name + ".silu");
    auto norm = op::rms_norm(0.00001f)(silu, gamma).set_name(name + ".rmsnorm");
    return norm;
  }

  std::vector<float> forward_cpu(const std::vector<float> &input,
    int in_features, int out_features, int batch_size) {
    std::vector<float> output(out_features * batch_size, 0.0f);

    // GEMM: output = weight * input + bias
    for (int b = 0; b < batch_size; ++b) {
      for (int i = 0; i < out_features; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < in_features; ++j) {
          sum += weight_data[i * in_features + j] * input[j * batch_size + b];
        }
        output[i * batch_size + b] = sum + bias_data[i];
      }
    }

    // SiLU and RMSNorm
    for (int b = 0; b < batch_size; ++b) {
      // 1. SiLU: x * sigmoid(x)
      for (int i = 0; i < out_features; ++i) {
        float x                    = output[i * batch_size + b];
        output[i * batch_size + b] = x / (1.0f + std::exp(-x));
      }

      // 2. RMSNorm
      float ms = 0.0f;
      for (int i = 0; i < out_features; ++i) {
        float x = output[i * batch_size + b];
        ms += x * x;
      }
      ms /= out_features;
      float rrms = 1.0f / std::sqrt(ms + 0.00001f);
      for (int i = 0; i < out_features; ++i) {
        output[i * batch_size + b] *= rrms * gamma_data[i];
      }
    }

    return output;
  }
};

int main(int argc, char **argv) {
  // MLP Parameters
  int input_dim   = 16;
  int hidden_dim1 = 32;
  int hidden_dim2 = 32;
  int output_dim  = 10;
  int batch_size  = 1;

  if (argc > 1)
    input_dim = std::atoi(argv[1]);
  if (argc > 2)
    hidden_dim1 = std::atoi(argv[2]);
  if (argc > 3)
    hidden_dim2 = std::atoi(argv[3]);
  if (argc > 4)
    output_dim = std::atoi(argv[4]);
  if (argc > 5)
    batch_size = std::atoi(argv[5]);

  std::cout << "Creating 3-layer MLP: " << input_dim << " -> " << hidden_dim1
            << " -> " << hidden_dim2 << " -> " << output_dim << std::endl;

  // Define Inputs
  auto input =
    op::create_input({input_dim, batch_size}, DType::FP32).set_name("input");

  // Define Layers
  Layer layer1(input_dim, hidden_dim1, "layer1");
  Layer layer2(hidden_dim1, hidden_dim2, "layer2");
  Layer layer3(hidden_dim2, output_dim, "layer3");

  // Forward Pass
  auto h1  = layer1.forward(input);
  auto h2  = layer2.forward(h1);
  auto out = layer3.forward(h2);

  // GE Backend Build
  Backend      backend = GEBackend(AscendDevice::from_env());
  forch::Graph graph_ir({out});
  graph_ir.set_name("ge-mlp");
  auto graph = backend.build(graph_ir);

  std::cout << "=== Compiling Graph ===" << std::endl;
  graph.compile();

  std::cout << "=== Loading Graph ===" << std::endl;
  graph.load();

  // Prepare random input data
  auto input_data = create_random_buffer(input_dim * batch_size);
  auto input_tensor =
    op::load_from_buffer(reinterpret_cast<Buffer>(input_data.data()),
      {input_dim, batch_size}, DType::FP32)
      .set_name("input");

  std::cout << "=== Running Graph ===" << std::endl;
  auto outputs = graph.run({input_tensor});

  std::cout << "Output Tensor Info: ";
  outputs[0].get_op()->dump(std::cout);
  std::cout << std::endl;

  const float *out_ptr = reinterpret_cast<const float *>(outputs[0].get_data());

  // CPU Verification
  std::cout << "=== Running CPU Verification ===" << std::endl;
  auto cpu_h1 =
    layer1.forward_cpu(input_data, input_dim, hidden_dim1, batch_size);
  auto cpu_h2 =
    layer2.forward_cpu(cpu_h1, hidden_dim1, hidden_dim2, batch_size);
  auto cpu_out =
    layer3.forward_cpu(cpu_h2, hidden_dim2, output_dim, batch_size);

  float max_diff = 0.0f;
  float sum_diff = 0.0;
  for (int i = 0; i < output_dim * batch_size; ++i) {
    float diff =
      std::abs(cpu_out[i] - out_ptr[i]) / (std::abs(cpu_out[i]) + 1e-6f);
    max_diff = std::max(max_diff, diff);
    sum_diff += diff;

    std::cout << "output[" << i << "]: cpu = " << cpu_out[i]
              << ", ge = " << out_ptr[i] << ", diff = " << diff << std::endl;
  }

  std::cout << "Verification Results:" << std::endl;
  std::cout << "Max Difference: " << max_diff << std::endl;
  std::cout << "Mean Difference: " << sum_diff / (output_dim * batch_size)
            << std::endl;

  return 0;
}
