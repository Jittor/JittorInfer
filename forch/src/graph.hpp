#pragma once

#include "forch/graph.hpp"

#include <utility>

namespace forch {

struct Graph::Impl {
  TensorList  outputs;
  std::string graph_name;

  explicit Impl(TensorList graph_outputs)
      : outputs(std::move(graph_outputs)), graph_name("") {}

  virtual ~Impl() = default;
};

struct Backend::Graph::Impl {
  virtual ~Impl() = default;

  virtual void       compile()                     = 0;
  virtual void       load()                        = 0;
  virtual TensorList run(const TensorList &inputs) = 0;
};

struct Backend::Impl {
  virtual ~Impl() = default;

  virtual Backend::Graph build(
    std::shared_ptr<Backend::Impl> backend_impl, const forch::Graph &graph) = 0;

  virtual Tensor move_to_device(const Tensor &tensor) const = 0;
};

} // namespace forch
