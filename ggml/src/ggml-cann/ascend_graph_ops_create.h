#ifndef _ASCEND_GRAPH_OPS_CREATE_H_
#define _ASCEND_GRAPH_OPS_CREATE_H_

#include <map>
#include <vector>

#include "es_all_ops.h"
#include "ggml.h"

// ES API 版本：对MOE_FUSED算子中的GroupMatmul算子进行封装，预定义好一些属性
ge::es::EsTensorHolder create_moe_grouped_matmul_es(
    ge::es::EsGraphBuilder& graph_builder, const ge::es::EsTensorHolder& x,
    const ge::es::EsTensorHolder& weight,
    const std::vector<int64_t>& bias_shape,
    const ge::es::EsTensorHolder& group_list);

#endif  // _ASCEND_GRAPH_OPS_CREATE_H_
