#include "forch/ops/softmax.hpp"
#include "op-registry.hpp"

#include "es_SoftmaxV2_c.h"

#include <cstdint>

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Softmax>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Softmax *op) {
      const int64_t axes[1] = {op->dim};
      return EsSoftmaxV2(ctx.inputs[0], axes, /*axes_num=*/1,
        /*half_to_float=*/false);
    });
}
} // namespace
