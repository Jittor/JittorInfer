#pragma once

#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/ops.hpp"

namespace forch::op {

struct Runtime : public Base {
  Runtime()          = default;
  virtual ~Runtime() = default;

  void accept(Visitor *visitor, const Tensor &self) const override {
    visitor->visit(this, self);
  }

  virtual const void *get_data() const {
    throw exception("unimplemented get_data");
  }
};

} // namespace forch::op
