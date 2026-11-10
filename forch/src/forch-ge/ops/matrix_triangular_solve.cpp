#include "forch/ops/matrix_triangular_solve.hpp"
#include "op-registry.hpp"

#include "es_MatrixTriangularSolve_c.h"

namespace {

__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::MatrixTriangularSolve>(
    [](const forch::detail::GEOpContext      &ctx,
      const forch::op::MatrixTriangularSolve *op) {
      return EsMatrixTriangularSolve(
        ctx.inputs[0], ctx.inputs[1], op->lower, op->adjoint);
    });
}

} // namespace
