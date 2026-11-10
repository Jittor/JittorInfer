#include "forch/exception.hpp"
#include "op-registry.hpp"

#include "es_Add_c.h"
#include "es_Cos_c.h"
#include "es_Mul_c.h"
#include "es_Sin_c.h"
#include "es_Swish_c.h"

#include "magic_enum.hpp"

#include <string>

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Function>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Function *op) {
      using Function = forch::op::Function;
      using Type     = Function::Type;

      switch (op->type) {
      case Type::Add:
        return EsAdd(ctx.inputs[0], ctx.inputs[1]);
      case Type::Mul:
        return EsMul(ctx.inputs[0], ctx.inputs[1]);
      case Type::Silu:
        return EsSwish(ctx.inputs[0], /*scale=*/1.0F);
      case Type::Sin:
        return EsSin(ctx.inputs[0]);
      case Type::Cos:
        return EsCos(ctx.inputs[0]);
      default:
        throw forch::exception(
          std::string("forch-ge: unsupported op function type: ")
          + std::string(magic_enum::enum_name(op->type)));
      }
    });
}
} // namespace
