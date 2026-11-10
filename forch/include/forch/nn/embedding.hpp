#pragma once

#include "forch/forch.hpp"
#include "forch/ops/index_select.hpp"

#include "forch/nn/module.hpp"
#include "forch/nn/parameter.hpp"

namespace forch::nn {

class Embedding : public Module {
public:
  int64_t num_embeddings;
  int64_t embedding_dim;

  Embedding(
    int64_t num_embeddings, int64_t embedding_dim, DType dtype = DType::FP32)
      : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
    weight = register_loadable(
      "weight", Parameter({num_embeddings, embedding_dim}, dtype));
  }

  Tensor forward(const Tensor &input) {
    return op::index_select(0)(weight->get(), input);
  }

private:
  Parameter *weight;
};

} // namespace forch::nn
