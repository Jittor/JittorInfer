#include "ascend_graph_ops_create.h"

#include <numeric>

// ES API 版本：对MOE_FUSED算子中的GroupMatmul算子进行封装，预定义好一些属性
ge::es::EsTensorHolder create_moe_grouped_matmul_es(
    ge::es::EsGraphBuilder& graph_builder, const ge::es::EsTensorHolder& x,
    const ge::es::EsTensorHolder& weight,
    const std::vector<int64_t>& bias_shape,
    const ge::es::EsTensorHolder& group_list) {
    int64_t len = std::accumulate(bias_shape.begin(), bias_shape.end(), 1,
                                  std::multiplies<int64_t>());
    auto es_bias =
        graph_builder
            .CreateConst(std::vector<ge::float32_t>(len / 2, 0.0f), bias_shape)
            .SetDataType(ge::DT_FLOAT16);

    // 准备 GroupedMatmul 的输入（单个张量包装成 vector）
    std::vector<ge::es::EsTensorHolder> es_x_vec = {x};
    std::vector<ge::es::EsTensorHolder> es_weight_vec = {weight};
    std::vector<ge::es::EsTensorHolder> es_bias_vec = {es_bias};

    // scale, offset, antiquant_scale, antiquant_offset 传空 vector
    std::vector<ge::es::EsTensorHolder> empty_vec = {};

    // per_token_scale 传 nullptr
    // 调用 ES API 的 GroupedMatmul
    // 参数：split_item=2, group_list_type=1, group_type=0, act_type=0, y_num=1
    auto result_vec = ge::es::GroupedMatmul(
        es_x_vec, es_weight_vec, es_bias_vec, empty_vec, empty_vec, empty_vec,
        empty_vec,  // scale, offset, antiquant_scale, antiquant_offset
        group_list, nullptr,  // group_list, per_token_scale
        1,                    // y_num
        2,                    // split_item
        0,                    // dtype
        false, false,         // transpose_weight, transpose_x
        0,                    // group_type
        1,                    // group_list_type
        0                     // act_type
    );

    // 返回第一个输出
    return result_vec[0];
}
