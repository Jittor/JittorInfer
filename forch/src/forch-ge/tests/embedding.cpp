#include "forch/nn/embedding.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace forch;

static bool almost_equal(
  const float *a, const float *b, size_t n, float tol = 1e-6F) {
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

struct EmbeddingLoader : public nn::Loader {
  // [6, 3]
  std::vector<float> weight = {
    0.0F,
    0.1F,
    0.2F,
    1.0F,
    1.1F,
    1.2F,
    2.0F,
    2.1F,
    2.2F,
    3.0F,
    3.1F,
    3.2F,
    4.0F,
    4.1F,
    4.2F,
    5.0F,
    5.1F,
    5.2F,
  };

  Tensor load(const std::string &name) override {
    if (name == "embedding.weight") {
      return op::load_from_vector(std::vector<float>(weight), {6, 3})
        .set_name(name);
    }
    throw exception(std::string("unexpected parameter name: ") + name);
  }
};

int test_embedding(Backend &backend) {
  const Shape token_shape = {4};
  const Shape out_shape   = {4, 3};

  std::vector<float> expected = {
    3.0F,
    3.1F,
    3.2F,
    1.0F,
    1.1F,
    1.2F,
    4.0F,
    4.1F,
    4.2F,
    1.0F,
    1.1F,
    1.2F,
  };

  auto token_input =
    op::create_input(token_shape, DType::INT64).set_name("token_ids");

  nn::Embedding   embedding(6, 3, DType::FP32);
  EmbeddingLoader loader;
  embedding.load(&loader, "embedding");

  auto out = embedding.forward(token_input);
  out.set_name(std::string("embedding_output"));
  if (out.shape() != out_shape) {
    std::cerr << "embedding: compile-time output shape mismatch" << std::endl;
    return 1;
  }

  forch::Graph graph_ir({out});
  graph_ir.set_name("test_embedding");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "embedding: compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  auto runtime_tokens =
    op::load_from_vector(std::vector<int64_t>{3, 1, 4, 1}, token_shape)
      .set_name("token_ids");
  auto outputs = graph.run({runtime_tokens});

  if (outputs.empty()) {
    std::cerr << "embedding: runtime outputs is empty" << std::endl;
    return 3;
  }
  auto out_tensor = outputs.get(std::string("embedding_output"));
  if (out_tensor.shape() != out_shape) {
    std::cerr << "embedding: runtime output shape mismatch" << std::endl;
    return 4;
  }
  if (out_tensor.dtype() != DType::FP32) {
    std::cerr << "embedding: runtime output dtype mismatch" << std::endl;
    return 5;
  }

  const auto *out_ptr = reinterpret_cast<const float *>(out_tensor.get_data());
  if (!almost_equal(out_ptr, expected.data(), expected.size(), 1e-2F)) {
    std::cerr << "embedding: output value mismatch" << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::fabs(out_ptr[i] - expected[i]) > 1e-2F) {
        std::cerr << "first mismatch at " << i << ": out=" << out_ptr[i]
                  << ", expected=" << expected[i] << std::endl;
        break;
      }
    }
    return 6;
  }

  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  const int r = test_embedding(backend);
  if (r != 0) {
    return r;
  }

  std::cout << "Embedding test passed." << std::endl;
  return 0;
}
