#pragma once

#include "forch/exception.hpp"
#include "forch/forch.hpp"

namespace forch::op {

struct Const;
struct Input;
struct Function;
struct Runtime;
struct MultiOutputPort;

struct Visitor {
  virtual ~Visitor() = default;

  virtual void visit(const op::Const * /*unused*/, const Tensor & /*unused*/) {
    throw exception("Unsupported op");
  }
  virtual void visit(
    const op::Function * /*unused*/, const Tensor & /*unused*/) {
    throw exception("Unsupported op");
  }
  virtual void visit(const op::Input * /*unused*/, const Tensor & /*unused*/) {
    throw exception("Unsupported op");
  }
  virtual void visit(
    const op::Runtime * /*unused*/, const Tensor & /*unused*/) {
    throw exception("Unsupported op");
  }
  virtual void visit(
    const op::MultiOutputPort * /*unused*/, const Tensor & /*unused*/) {
    throw exception("Unsupported op");
  }
};

// unified ops across backends
struct Base {
  virtual ~Base() = default;

  virtual void accept(Visitor *visitor, const Tensor &self) const = 0;

  virtual void dump(std::ostream & /*unused*/) const {}

  virtual Shape shape(const TensorList & /*unused*/ = {}) const {
    throw exception("shape() not implemented for this op");
  }

  virtual DType dtype(const TensorList & /*unused*/ = {}) const {
    throw exception("dtype() not implemented for this op");
  }
};

} // namespace forch::op