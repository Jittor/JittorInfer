#pragma once
#include <forch/nn/parallel_linear.hpp>
#include <forch/ops/cast.hpp>
#include <forch/ops/collective.hpp>
#include <forch/ops/const.hpp>
#include <forch/ops/function.hpp>
#include <forch/ops/index_select.hpp>
#include <forch/ops/permute.hpp>
#include <forch/ops/reshape.hpp>
#include <forch/ops/topk.hpp>

namespace finer::models {
// Vocabulary rows split evenly across ranks (vocab % tp == 0, enforced by
// ColumnParallelLinear). Inheriting keeps the checkpoint key, including tied
// embedding keys. Each rank takes a local TopK; the gathered candidates are
// reduced to the global TopK with vocabulary IDs.
class VocabParallelLMHead : public forch::nn::ColumnParallelLinear {
public:
  VocabParallelLMHead(int64_t hidden, int64_t vocab, int64_t top_k,
    const forch::ProcessGroup &group, forch::DType dtype)
      : ColumnParallelLinear(hidden, vocab, group, false, dtype),
        top_k_(top_k) {}

  forch::op::TopKOutputs forward(const forch::Tensor &hidden) const {
    using namespace forch;
    auto logits = ColumnParallelLinear::forward(hidden);
    if (group().world_size() == 1)
      return op::topk(top_k_, -1, true)(logits);

    const auto world   = group().world_size();
    const auto rows    = logits.shape()[0];
    const auto local_k = std::min(top_k_, local_out_features());
    auto local = op::topk(local_k, -1, true)(logits | op::cast(DType::FP32));
    auto ids   = op::add(local.indices | op::cast(DType::INT64),
        op::load_from_vector(
        std::vector<int64_t>{group().rank_id() * local_out_features()}, {}));
    // all_gather stacks ranks on dim 0; regroup candidates per token row.
    const auto width      = world * local_k;
    auto       rows_first = [&](const Tensor &x) {
      return op::all_gather(x, group()) | op::reshape({world, rows, local_k})
        | op::permute({1, 0, 2}) | op::reshape({rows, width});
    };
    auto candidates = rows_first(ids);
    auto result     = op::topk(top_k_, -1, true)(rows_first(local.values));
    // TopK indices address candidate columns, not vocabulary IDs. Flatten
    // with a row offset so each token selects its own gathered candidates.
    std::vector<int64_t> offsets(rows);
    for (int64_t row = 0; row < rows; ++row)
      offsets[row] = row * width;
    auto positions = op::add(result.indices | op::cast(DType::INT64),
                       op::load_from_vector(std::move(offsets), {rows, 1}))
      | op::reshape({rows * top_k_});
    auto global_ids =
      op::index_select(0)(candidates | op::reshape({rows * width}), positions)
      | op::reshape({rows, top_k_});
    return {result.values, global_ids};
  }

private:
  int64_t top_k_;
};

} // namespace finer::models
