#ifndef _ASCEND_GRAPH_OPS_CREATE_H_
#define _ASCEND_GRAPH_OPS_CREATE_H_

#include <map>
#include <vector>

#include "all_ops.h"
#include "ggml.h"
#include "graph/graph.h"

ge::Operator create_reshape_op(ge::Graph& graph, const std::string& prefix,
                               const std::string& suffix, ge::Operator src,
                               std::vector<int64_t>&& new_shape);

ge::Operator create_squeeze_op(ge::Graph& graph, const std::string& prefix,
                               const std::string& suffix, ge::Operator src,
                               std::vector<int64_t>&& remove_dims);

ge::Operator create_unsqueeze_op(ge::Graph& graph, const std::string& prefix,
                                 const std::string& suffix, ge::Operator src,
                                 std::vector<int64_t>&& add_dims);

ge::Operator create_permute_op(ge::Graph& graph, const std::string& prefix,
                               const std::string& suffix, ge::Operator src,
                               std::vector<int64_t>&& permute_dims);

ge::Operator create_identity_op_by_name(ge::Graph& graph,
                                        const std::string& prefix,
                                        const std::string& suffix,
                                        ge::Operator src,
                                        const std::string& name);

// 对MOE_FUSED算子中的GroupMatmul算子进行封装，预定义好一些属性
ge::Operator create_moe_grouped_matmul_op(
    ge::Graph& graph, const std::string& prefix, const std::string& suffix,
    ge::DataType weight_type, ge::Operator x, ge::Operator weight,
    std::vector<int64_t>&& bias_shape, ge::Operator group_list);

ge::Operator create_const_float_op(ge::Graph& graph, const std::string& prefix,
                                   const std::string& suffix,
                                   std::vector<int64_t>&& shape,
                                   ge::DataType const_type, float value);

ge::Operator create_const_float16_zero_op(ge::Graph& graph,
                                          const std::string& prefix,
                                          const std::string& suffix,
                                          std::vector<int64_t>&& shape);

ge::Operator create_gather_op(ge::Graph& graph, const std::string& prefix,
                              const std::string& suffix, ge::Operator src,
                              ge::Operator indices, int axis = 0,
                              int batch_dims = 0);

ge::Operator create_cast_op(ge::Graph& graph, const std::string& prefix,
                            const std::string& suffix, ge::Operator src,
                            ge::DataType dst_type);

ge::Operator create_concat_op(ge::Graph& graph, const std::string& prefix,
                              const std::string& suffix,
                              std::vector<ge::Operator>&& op_list, int axis);

#endif  // _ASCEND_GRAPH_OPS_CREATE_H_