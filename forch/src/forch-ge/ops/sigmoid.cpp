#include "forch/ops/sigmoid.hpp"
#include "op-registry.hpp"

#include "es_Sigmoid_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Sigmoid>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Sigmoid *) {
      return EsSigmoid(ctx.inputs[0]);
    });
}

} // namespace
