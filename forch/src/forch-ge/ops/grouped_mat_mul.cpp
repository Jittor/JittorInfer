#include "forch/ops/grouped_mat_mul.hpp"
#include "ge_session_shim.h"
#include "op-registry.hpp"

#include "es_Const_c.h"
#include "es_GroupedMatmul_c.h"

#include <cstdint>
#include <vector>

namespace {
__attribute__((constructor)) void register_op() {
  forch::detail::GEOpRegistry::register_op<forch::op::GroupedMatMul>(
    [](const forch::detail::GEOpContext &ctx,
      const forch::op::GroupedMatMul    *op) {
      // One CANN op, two weight forms (resolved by shape() at Tensor
      // construction):
      //   stacked: [x, weight(E,K,N or E,N,K), group_list] — weight_num=1;
      //     the kernel advances offsets per group, so bias is ONE stacked
      //     zero [E, N] const.
      //   list: [x, w_0..w_{E-1}, group_list], each [K,N] or [N,K] —
      //     weight_num=E; aclnn bias is per-weight 1D in this mode (no
      //     stacked [E, N] form), so bias is E zero [N] consts.
      //
      // es_ GroupedMatmul declares ALL 7 list inputs as DYNAMIC (op proto);
      // the FE requires every dynamic input to have >= 1 instance, while the
      // infershape CheckNonQuant requires scale/offset/antiquant_* to be
      // absent for non-quant x/weight. Resolved (both forms): bias gets the
      // zero const(s) described above; scale/offset/antiquant get a single
      // empty scalar const (0 elements) — passes the FE count check while
      // reading as empty.
      auto make_const = [&ctx](const void *value, size_t bytes,
                          const int64_t *dims, size_t dim_num, C_DataType dt) {
        EsCTensor *t = forch_ge_create_tensor(value, bytes, dims, dim_num, dt);
        if (t == nullptr) {
          throw forch::exception(std::string("forch-ge: make_const failed: ")
            + forch_ge_last_error());
        }
        EsCTensorHolder *h = EsConst(ctx.builder, t);
        if (h == nullptr) {
          throw forch::exception(
            std::string("forch-ge: EsConst failed: ") + forch_ge_last_error());
        }
        return h;
      };

      // E/N resolved from the weight tensor(s) by shape() at Tensor
      // construction
      if (op->num_groups < 0 || op->out_features < 0) {
        throw forch::exception(
          "forch-ge: GroupedMatMul shape() not resolved before mapping");
      }
      const int64_t e = op->num_groups;
      const int64_t n = op->out_features;

      std::vector<EsCTensorHolder *> weight_list;
      std::vector<EsCTensorHolder *> bias_list;
      if (op->weight_form == forch::op::GroupedMatMul::WeightForm::Stacked) {
        const int64_t      bias_dims[2] = {e, n};
        std::vector<float> zero_eb(
          static_cast<size_t>(e) * n, 0.0F); // SetData copies, buffer reusable
        weight_list.push_back(ctx.inputs[1]);
        bias_list.push_back(make_const(zero_eb.data(),
          zero_eb.size() * sizeof(float), bias_dims, 2, C_DT_FLOAT));
      } else {
        const int64_t      bias_dims[1] = {n};
        std::vector<float> zero_b(static_cast<size_t>(n), 0.0F);
        weight_list.reserve(static_cast<size_t>(e));
        bias_list.reserve(static_cast<size_t>(e));
        for (int64_t i = 0; i < e; ++i) {
          weight_list.push_back(ctx.inputs[static_cast<size_t>(i) + 1]);
          bias_list.push_back(make_const(zero_b.data(),
            zero_b.size() * sizeof(float), bias_dims, 1, C_DT_FLOAT));
        }
      }

      // empty scalar const for the unused quant inputs
      EsCTensorHolder *empty = make_const(nullptr, 0, nullptr, 0, C_DT_FLOAT16);

      const int64_t    k_tuning[1]   = {0};
      EsCTensorHolder *x_list[1]     = {ctx.inputs[0]};
      EsCTensorHolder *empty_list[1] = {empty};

      // Fixed attrs for the DeepSeek-MoE usage: group_type=0 (M split from
      // group_list), group_list_type=0 (cumsum — matches
      // moe_init_routing cumsum output), split_item=2 (single y tensor).
      auto out = EsGroupedMatmul(
        /*x=*/x_list, /*x_num=*/1,
        /*weight=*/weight_list.data(),
        /*weight_num=*/static_cast<int64_t>(weight_list.size()),
        /*bias=*/bias_list.data(),
        /*bias_num=*/static_cast<int64_t>(bias_list.size()),
        /*scale=*/empty_list, /*scale_num=*/1,
        /*offset=*/empty_list, /*offset_num=*/1,
        /*antiquant_scale=*/empty_list, /*antiquant_scale_num=*/1,
        /*antiquant_offset=*/empty_list, /*antiquant_offset_num=*/1,
        /*group_list=*/ctx.inputs.back(),
        /*per_token_scale=*/nullptr,
        /*y_num=*/1, /*split_item=*/2,
        /*dtype=*/0,
        /*transpose_weight=*/op->transpose_weight,
        /*transpose_x=*/false,
        /*group_type=*/0,
        /*group_list_type=*/0,
        /*act_type=*/0, k_tuning, /*tuning_config_num=*/1);
      return out.y[0];
    });
}
} // namespace
