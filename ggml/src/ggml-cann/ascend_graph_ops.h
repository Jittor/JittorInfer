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

/**
 * @brief ES版本：处理MUL（乘法）操作的函数
 *
 * 使用ES API在计算图中创建一个乘法操作，支持两个输入张量的形状不同时的广播处理
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示MUL操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的MUL操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_mul_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_matmul_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理MATMUL（矩阵乘法）操作的函数
 *
 * 使用ES API在计算图中创建一个矩阵乘法操作
 * 注意：在昇腾中，矩阵乘法的输入顺序和转置设置需要特殊处理
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示MATMUL操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的MATMUL操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_matmul_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_softmax_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理SOFTMAX（Softmax）操作的函数
 *
 * 使用ES API在计算图中创建一个Softmax操作，支持scale缩放和mask掩码处理（包括ALiBi）
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示SOFTMAX操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的SOFTMAX操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_softmax_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_repeat_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理REPEAT（重复/广播）操作的函数
 *
 * 使用ES API在计算图中创建一个Tile（重复）操作，用于实现张量的重复和广播
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示REPEAT操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的REPEAT操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_repeat_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_silu_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理SILU（Swish激活函数）操作的函数
 *
 * 使用ES API在计算图中创建一个Swish操作，SILU是Swish的一种特殊情况
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示SILU操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的SILU操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_silu_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_argsort_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理ARGSORT（参数排序）操作的函数
 *
 * 使用ES API在计算图中创建一个Sort操作，返回排序后的索引
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示ARGSORT操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的ARGSORT操作的ES张量持有者（返回indices）
 */
ge::es::EsTensorHolder handle_argsort_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_scale_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理SCALE（缩放）操作的函数
 *
 * 使用ES API在计算图中创建一个缩放操作，将输入张量与标量常量相乘
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示SCALE操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的SCALE操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_scale_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_cpy_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理CPY（复制/类型转换）操作的函数
 *
 * 使用ES API在计算图中创建一个Cast操作，实现张量的类型转换
 * 输出形状与目标张量一致
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示CPY操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的CPY操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_cpy_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_reshape_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理RESHAPE（重塑形状）操作的函数
 *
 * 使用ES API在计算图中创建一个重塑操作，在不改变数据内容的情况下改变张量的维度结构
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示RESHAPE操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的RESHAPE操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_reshape_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_permute_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理PERMUTE（维度置换）操作的函数
 *
 * 使用ES API在计算图中创建一个维度置换操作，按照指定的顺序重新排列维度
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示PERMUTE操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的PERMUTE操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_permute_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_transpose_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理TRANSPOSE（转置）操作的函数
 *
 * 使用ES API在计算图中创建一个转置操作，交换最后两个维度
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示TRANSPOSE操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的TRANSPOSE操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_transpose_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_concat_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理CONCAT（拼接）操作的函数
 *
 * 使用ES API在计算图中创建一个ConcatV2操作，将两个张量在指定维度上拼接
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示CONCAT操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的CONCAT操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_concat_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_view_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理VIEW（视图）操作的函数
 *
 * 使用ES API在计算图中创建一个ViewCopy操作，实现张量的视图（不复制数据，只改变形状和步长）
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示VIEW操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的VIEW操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_view_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_cont_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理CONT（连续化）操作的函数
 *
 * 使用ES API在计算图中创建一个Identity或Reshape操作，将张量转换为内存连续排列
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示CONT操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的CONT操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_cont_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_rms_norm_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index);

/**
 * @brief ES版本：处理RMS_NORM（RMS归一化）操作的函数
 *
 * 使用ES API在计算图中创建一个RmsNorm操作，实现Root Mean Square归一化
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示RMS_NORM操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的RMS_NORM操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_rms_norm_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

ge::Operator handle_rope_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index, ggml_backend_cann_context &cann_ctx);

/**
 * @brief ES版本：处理ROPE（旋转位置编码）操作的函数
 *
 * 使用ES API在计算图中创建一个RoPE操作，实现旋转位置编码
 * 注意：此操作非常复杂，涉及Reshape、Transpose、Gather、自定义算子等多个步骤
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示ROPE操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @param cann_ctx CANN上下文，用于RopeCache
 * @return 创建的ROPE操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_rope_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
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