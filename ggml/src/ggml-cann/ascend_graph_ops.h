#ifndef _ASCEND_GRAPH_OPS_H_
#define _ASCEND_GRAPH_OPS_H_

#include <map>
#include <vector>

#include "all_ops.h"
// ES (Eager Style) API support
#include "es_all_ops.h"
#include "common.h"
#include "ggml.h"
#include "graph/graph.h"

ge::DataType get_data_type(enum ggml_type type);
/**
 * @brief Handles the creation of an ADD operation in the computational graph
 *
 * This function creates an ADD operation in the graph, connecting it with its
 * inputs which may be existing operators or newly created data operators.
 *
 * @param graph The computational graph
 * @param node The tensor node representing the ADD operation
 * @param gmml_tensor_to_ge_op_map Map of tensors to their corresponding
 * operators
 * @param op_index Index for generating unique operator names
 * @return The created ADD operator
 */
ge::Operator handle_add_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理ADD（加法）操作的函数
 *
 * 使用ES API在计算图中创建一个加法操作，支持两个输入张量的形状不同时的广播处理
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示ADD操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的ADD操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_add_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_mul_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_matmul_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_softmax_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_repeat_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_silu_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_argsort_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_scale_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_cpy_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_reshape_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_permute_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_transpose_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);
ge::Operator handle_concat_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_view_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_cont_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_rms_norm_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);
ge::Operator handle_rope_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index, ggml_backend_cann_context &cann_ctx);
ge::Operator handle_moe_fused_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);
ge::Operator handle_arange_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);
ge::Operator handle_stridedslicev2_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);
ge::Operator handle_flash_attn_prompt_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_set_slice_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_get_rows_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator handle_pad_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

ge::Operator create_const_1d_op(ge::Graph &graph, const std::string &name,
                                const std::vector<int64_t> &values,
                                ge::DataType type);

/**
 * @brief 创建通用的Reshape操作
 *
 * 这是一个通用的reshape函数，可以被多个算子调用
 * 避免重复代码，统一reshape操作的创建方式
 *
 * @param graph 计算图引用
 * @param input_op 输入算子
 * @param target_shape 目标形状向量
 * @param op_name 操作名称
 * @param data_type 数据类型
 * @return 创建的Reshape算子
 */
ge::Operator create_reshape_op(ge::Graph &graph, ge::Operator &input_op,
                               const std::vector<int64_t> &target_shape,
                               const std::string &op_name,
                               ge::DataType data_type);

#endif  // _ASCEND_GRAPH_OPS_H_