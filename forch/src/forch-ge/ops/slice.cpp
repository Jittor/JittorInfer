#include "forch/ops/slice.hpp"
#include "op-registry.hpp"

#include "es_Slice_c.h"

#include <cstdint>
#include <vector>

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::Slice>(
    [](const forch::detail::GEOpContext &ctx, const forch::op::Slice *op) {
      std::vector<int64_t> offsets;
      std::vector<int64_t> size;
      offsets.reserve(op->slice.size());
      size.reserve(op->slice.size());
      for (const auto &[begin, end] : op->slice) {
        offsets.push_back(begin);
        size.push_back(end - begin);
      }
      auto *offsets_h = EsCreateVectorInt64(
        ctx.builder, offsets.data(), static_cast<int64_t>(offsets.size()));
      auto *size_h = EsCreateVectorInt64(
        ctx.builder, size.data(), static_cast<int64_t>(size.size()));
      return EsSlice(ctx.inputs[0], offsets_h, size_h);
    });
}
} // namespace
