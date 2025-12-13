#ifndef _ASCEND_GRAPH_OPS_H_
#define _ASCEND_GRAPH_OPS_H_

#include <map>
#include <vector>

#include "all_ops.h"
// ES (Eager Style) API support
#include "common.h"
#include "es_all_ops.h"
#include "ggml.h"
#include "graph/graph.h"

ge::DataType get_data_type(enum ggml_type type);
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

/**
 * @brief ES版本：处理SOFTMAX（Softmax）操作的函数
 *
 * 使用ES
 * API在计算图中创建一个Softmax操作，支持scale缩放和mask掩码处理（包括ALiBi）
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

/**
 * @brief ES版本：处理RESHAPE（重塑形状）操作的函数
 *
 * 使用ES
 * API在计算图中创建一个重塑操作，在不改变数据内容的情况下改变张量的维度结构
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

/**
 * @brief ES版本：处理VIEW（视图）操作的函数
 *
 * 使用ES
 * API在计算图中创建一个ViewCopy操作，实现张量的视图（不复制数据，只改变形状和步长）
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

/**
 * @brief ES版本：处理MOE_FUSED（混合专家融合）操作的函数
 *
 * 使用ES API在计算图中创建一个MoE Fused操作，实现混合专家模型的前向传播
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示MOE_FUSED操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的MOE_FUSED操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_moe_fused_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

/**
 * @brief ES版本：处理ARANGE（等差数列生成）操作的函数
 *
 * 使用ES
 * API在计算图中创建一个Range操作，生成从start到limit（不包含）步长为delta的等差数列
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示ARANGE操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的ARANGE操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_arange_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

/**
 * @brief ES版本：处理StridedSliceV2（步长切片）操作的函数
 *
 * 使用ES API在计算图中创建一个StridedSliceV2操作，实现张量的步长切片操作
 * 从输入张量中提取指定步长的切片，支持多维张量的灵活切片
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示StridedSliceV2操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的StridedSliceV2操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_stridedslicev2_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

/**
 * @brief ES版本：处理Flash Attention Prompt操作的函数
 *
 * 使用ES API在计算图中创建一个PromptFlashAttention操作，实现Flash
 * Attention的prompt阶段
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示Flash Attention Prompt操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的Flash Attention Prompt操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_flash_attn_prompt_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

/**
 * @brief ES版本：处理Set Slice（切片赋值）操作的函数
 *
 * 使用ES API在计算图中创建一个ScatterUpdate操作，实现张量的切片赋值
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示Set Slice操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的Set Slice操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_set_slice_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

/**
 * @brief ES版本：处理Get Rows操作的函数
 *
 * 使用ES API在计算图中创建一个GatherV2操作，实现按行索引获取张量行
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示Get Rows操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的Get Rows操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_get_rows_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

/**
 * @brief ES版本：处理Pad（填充）操作的函数
 *
 * 使用ES API在计算图中创建一个PadV3操作，实现张量的填充
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示Pad操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的Pad操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_pad_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index);

#endif  // _ASCEND_GRAPH_OPS_H_