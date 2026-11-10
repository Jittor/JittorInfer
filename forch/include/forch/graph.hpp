#pragma once

#include "forch/forch.hpp"
#include <memory>
#include <utility>
#include <vector>

namespace forch {

/**
 * General graph for backend-independent optimizations and transformations.
 */
struct Graph {
  Graph(TensorList outputs);

  std::string name() const;
  Graph      &set_name(const std::string &name);

  TensorList outputs() const;

protected:
  struct Impl;
  std::shared_ptr<Impl> _impl;
};

struct Backend {
  /**
   * Interface for backend specific graph, after general optimizations and
   * transformations.
   */
  struct Graph {
    struct Output : private TensorList {
      using TensorList::operator[];
      using TensorList::empty;
      using TensorList::size;
      using TensorList::TensorList;

      explicit Output(TensorList tensors) : TensorList(std::move(tensors)) {}

      TensorList filter_name(const std::string &name) const;
      Tensor     get(const std::string &name) const;
    };

    /**
     * Compile graph to backend-specific executable format. This step may
     * include backend-specific optimizations and transformations.
     */
    Graph &compile();

    /**
     * Load graph from host to device.
     * Including memory allocation and weight data copy if necessary.
     */
    Graph &load();

    /**
     * inputs: Tensor with op::Const or op::Runtime
     * outputs: Tensor with op::Runtime
     */
    Output run(const TensorList &inputs);

  protected:
    struct Impl;
    std::shared_ptr<Impl> _impl;
    Graph(std::shared_ptr<Impl> impl);
  };

  /**
   * convert a forch::Graph to a backend-specific Graph
   */
  Graph build(const forch::Graph &graph);

  /**
   * inputs: const op::Const
   * outputs: Tensor with op::Runtime
   */
  virtual Tensor move_to_device(const Tensor &tensor) const;

protected:
  struct Impl;
  std::shared_ptr<Impl> _impl;
  Backend(std::shared_ptr<Impl> impl);
};

struct ToDevice {
  Backend backend;
  Tensor  operator()(const Tensor &t) const;
};

inline ToDevice to_device(Backend backend) {
  return ToDevice{std::move(backend)};
}

} // namespace forch