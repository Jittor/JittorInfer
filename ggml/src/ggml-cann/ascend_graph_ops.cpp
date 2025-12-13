#include "ascend_graph_ops.h"

#include <acl/acl.h>
#include <float.h>

#include <algorithm>
#include <cassert>
#include <cmath>  // 添加cmath以获取数学函数:log2, floor, powf
#include <cmath>
#include <cstring>

#include "ascend_graph_ops.h"
#include "ascend_graph_ops_create.h"
#include "es_c_graph_builder.h"
#include "ggml-impl.h"
#include "op_proto.h"
#include "rope_cache.h"
/**
 * @brief 构建输出形状向量，从张量维度提取
 *
 * 该函数根据输入张量的维度信息构建输出形状向量，可选择是否反转维度顺序
 * 在GGML中，维度顺序是从0到3，而在昇腾中通常是反向的，所以需要转换
 *
 * @param tensor 要提取形状的张量
 * @param reverse 是否反转维度顺序(矩阵乘法需要true，其他操作通常为false)
 * @return std::vector<int64_t> 输出形状向量
 */
std::vector<int64_t> build_output_shape(const struct ggml_tensor *tensor,
                                        bool reverse = true) {
    std::vector<int64_t> output_shape;
    if (reverse) {
        // 反转维度顺序(从高维到低维)
        for (int d = GGML_MAX_DIMS - 1; d >= 0; d--) {
            if (tensor->ne[d] > 0) {
                output_shape.push_back(tensor->ne[d]);
            }
        }
    } else {
        // 保持原始维度顺序(从低维到高维)
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            if (tensor->ne[d] > 0) {
                output_shape.push_back(tensor->ne[d]);
            }
        }
    }
    return output_shape;
}

/**
 * @brief 处理tensor形状，移除前缀1并验证维度
 *
 * @param src_tensor 源张量
 * @return 处理后的形状向量
 */
static std::vector<int64_t> squeeze_ggml_tensor_shape(
    struct ggml_tensor *src_tensor) {
    std::vector<int64_t> shape = build_output_shape(src_tensor);
    // 剔除掉所有前缀1
    while (shape.size() > 0 && shape[0] == 1) {
        shape.erase(shape.begin());
    }
    return shape;
}

/**
 * @brief 计算广播形状和标记需要平铺的维度
 *
 * 当两个张量形状不同时，为了进行算术运算，需要计算它们的广播规则
 * 此函数根据GGML张量的形状计算广播维度，并记录哪些维度需要平铺(Tile)
 *
 * @param src0 第一个源张量
 * @param src1 第二个源张量
 * @param out_ne 输出张量的维度大小
 * @param out_nb0 第一个源张量的广播后步长
 * @param out_nb1 第二个源张量的广播后步长
 * @param need_tile0 标记第一个张量哪些维度需要平铺
 * @param need_tile1 标记第二个张量哪些维度需要平铺
 */
void bcast_shape(const ggml_tensor *src0, const ggml_tensor *src1,
                 int64_t out_ne[GGML_MAX_DIMS], int64_t out_nb0[GGML_MAX_DIMS],
                 int64_t out_nb1[GGML_MAX_DIMS], bool need_tile0[GGML_MAX_DIMS],
                 bool need_tile1[GGML_MAX_DIMS]) {
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const int64_t n0 = src0->ne[d];
        const int64_t n1 = src1->ne[d];

        if (n0 == n1) {
            // 完全相同，不需要广播
            out_ne[d] = n0;
            out_nb0[d] = src0->nb[d];
            out_nb1[d] = src1->nb[d];
            need_tile0[d] = need_tile1[d] = false;
        } else if (n1 == 1) {
            // src1 做标准广播
            out_ne[d] = n0;
            out_nb0[d] = src0->nb[d];
            out_nb1[d] = 0;  // src1 沿此维度重复同一个元素
            need_tile0[d] = false;
            need_tile1[d] = false;  // 下面会用 SetInputBroadcast
        } else if (n0 == 1) {
            // src0 做标准广播
            out_ne[d] = n1;
            out_nb0[d] = 0;
            out_nb1[d] = src1->nb[d];
            need_tile0[d] = false;
            need_tile1[d] = false;
        } else if (n0 % n1 == 0) {
            // src1 可打包重复 → 使用Tile操作
            out_ne[d] = n0;
            out_nb0[d] = src0->nb[d];
            out_nb1[d] = src1->nb[d];
            need_tile0[d] = false;
            need_tile1[d] = true;
        } else if (n1 % n0 == 0) {
            // src0 可打包重复 → 使用Tile操作
            out_ne[d] = n1;
            out_nb0[d] = src0->nb[d];
            out_nb1[d] = src1->nb[d];
            need_tile0[d] = true;
            need_tile1[d] = false;
        } else {
            fprintf(stderr, "bcast error at dim %d: %lld vs %lld\n", d,
                    (long long)n0, (long long)n1);
            abort();
        }
    }
}

/**
 * @brief 将GGML张量类型转换为昇腾数据类型
 *
 * 根据GGML的张量类型返回对应的昇腾平台数据类型枚举值
 *
 * @param type GGML张量类型
 * @return ge::DataType 对应的昇腾数据类型
 */
ge::DataType get_data_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return ge::DT_FLOAT;
        case GGML_TYPE_F16:
            return ge::DT_FLOAT16;
        case GGML_TYPE_Q4_0:
            return ge::DT_INT4;
        case GGML_TYPE_Q8_0:
            return ge::DT_QINT8;
        case GGML_TYPE_I8:
            return ge::DT_INT8;
        case GGML_TYPE_I16:
            return ge::DT_INT16;
        case GGML_TYPE_I32:
            return ge::DT_INT32;
        case GGML_TYPE_I64:
            return ge::DT_INT64;
        default:
            return ge::DT_FLOAT;
    }
}

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
    int op_index) {
    (void)graph_builder;
    // TODO: 删除这个没有用到的参数，因为es的api内部维护了一个自增的索引
    (void)op_index;
    // 获取源张量
    struct ggml_tensor *src0 = node->src[0];
    struct ggml_tensor *src1 = node->src[1];

    // 检查输入是否已经在映射中
    ge::es::EsTensorHolder es_tensor1, es_tensor2;

    // 处理src0 - 获取现有ES张量
    if (ggml_tensor_to_es_tensor_map.find(src0) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_tensor1 = ggml_tensor_to_es_tensor_map[src0];
    } else {
        assert(false && "src0 tensor not found in ES tensor map");
    }

    // 处理src1 - 获取现有ES张量
    if (ggml_tensor_to_es_tensor_map.find(src1) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_tensor2 = ggml_tensor_to_es_tensor_map[src1];
    } else {
        assert(false && "src1 tensor not found in ES tensor map");
    }

    // 处理广播（参考原始GE API实现）
    int64_t out_ne[GGML_MAX_DIMS], nb0[GGML_MAX_DIMS], nb1[GGML_MAX_DIMS];
    bool need_tile0[GGML_MAX_DIMS], need_tile1[GGML_MAX_DIMS];
    bcast_shape(src0, src1, out_ne, nb0, nb1, need_tile0, need_tile1);

    // 如果src0需要平铺(Tile)，使用ES API创建Tile操作
    if (std::any_of(need_tile0, need_tile0 + GGML_MAX_DIMS,
                    [](bool x) { return x; })) {
        // 计算每个维度的重复倍数
        std::vector<int64_t> multiples;
        for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
            int64_t repeat = out_ne[i] / src0->ne[i];
            multiples.push_back(repeat);
        }
        // 使用ES API创建Tile操作
        es_tensor1 = Tile(es_tensor1, multiples)
                         .SetDataType(get_data_type(src0->type))
                         .SetShape(build_output_shape(node));
    }

    // 如果src1需要平铺，同样处理
    if (std::any_of(need_tile1, need_tile1 + GGML_MAX_DIMS,
                    [](bool x) { return x; })) {
        // 计算每个维度的重复倍数
        std::vector<int64_t> multiples;
        for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
            int64_t repeat = out_ne[i] / src1->ne[i];
            multiples.push_back(repeat);
        }
        // 使用ES API创建Tile操作
        es_tensor2 = Tile(es_tensor2, multiples)
                         .SetDataType(get_data_type(src1->type))
                         .SetShape(build_output_shape(node));
    }

    // 使用ES API创建Add操作（支持运算符重载或函数调用）
    // 使用运算符重载+链式调用;
    return (es_tensor1 + es_tensor2)
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)graph_builder;
    // TODO: 删除这个没有用到的参数，因为es的api内部维护了一个自增的索引
    (void)op_index;
    // 获取源张量
    struct ggml_tensor *src0 = node->src[0];
    struct ggml_tensor *src1 = node->src[1];

    // 检查输入是否已经在映射中
    ge::es::EsTensorHolder es_tensor1, es_tensor2;

    // 处理src0 - 获取现有ES张量
    if (ggml_tensor_to_es_tensor_map.find(src0) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_tensor1 = ggml_tensor_to_es_tensor_map[src0];
    } else {
        assert(false && "src0 tensor not found in ES tensor map");
    }

    // 处理src1 - 获取现有ES张量
    if (ggml_tensor_to_es_tensor_map.find(src1) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_tensor2 = ggml_tensor_to_es_tensor_map[src1];
    } else {
        assert(false && "src1 tensor not found in ES tensor map");
    }

    // 处理广播（参考原始GE API实现）
    int64_t out_ne[GGML_MAX_DIMS], nb0[GGML_MAX_DIMS], nb1[GGML_MAX_DIMS];
    bool need_tile0[GGML_MAX_DIMS], need_tile1[GGML_MAX_DIMS];
    bcast_shape(src0, src1, out_ne, nb0, nb1, need_tile0, need_tile1);

    // 如果src0需要平铺(Tile)，使用ES API创建Tile操作
    if (std::any_of(need_tile0, need_tile0 + GGML_MAX_DIMS,
                    [](bool x) { return x; })) {
        // 计算每个维度的重复倍数
        std::vector<int64_t> multiples;
        for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
            int64_t repeat = out_ne[i] / src0->ne[i];
            multiples.push_back(repeat);
        }
        // 使用ES API创建Tile操作
        es_tensor1 = Tile(es_tensor1, multiples)
                         .SetDataType(get_data_type(src0->type))
                         .SetShape(build_output_shape(node));
    }

    // 如果src1需要平铺，同样处理
    if (std::any_of(need_tile1, need_tile1 + GGML_MAX_DIMS,
                    [](bool x) { return x; })) {
        // 计算每个维度的重复倍数
        std::vector<int64_t> multiples;
        for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
            int64_t repeat = out_ne[i] / src1->ne[i];
            multiples.push_back(repeat);
        }
        // 使用ES API创建Tile操作
        es_tensor2 = Tile(es_tensor2, multiples)
                         .SetDataType(get_data_type(src1->type))
                         .SetShape(build_output_shape(node));
    }

    // 使用ES API创建Mul操作（使用运算符重载+链式调用）
    return (es_tensor1 * es_tensor2)
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)graph_builder;
    (void)op_index;
    // 获取输入张量
    struct ggml_tensor *src0 = node->src[0];
    struct ggml_tensor *src1 = node->src[1];

    // 检查输入是否已经在映射中
    ge::es::EsTensorHolder es_tensor0, es_tensor1;
    if (ggml_tensor_to_es_tensor_map.find(src0) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_tensor0 = ggml_tensor_to_es_tensor_map[src0];
    } else {
        assert(false && "src0 tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(src1) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_tensor1 = ggml_tensor_to_es_tensor_map[src1];
    } else {
        assert(false && "src1 tensor not found in ES tensor map");
    }

    // 注意：在昇腾中，矩阵乘法的输入顺序是反的
    // 原始GE API: set_input_x1(op_b)对应src1, set_input_x2(op_a)对应src0
    // 设置adj_x2=true，表示第二个输入(src0)需要转置
    // ES API: MatMul(x1, x2, bias, transpose_x1, transpose_x2)
    // 所以这里使用: MatMul(src1, src0, nullptr, transpose_x1=false,
    // transpose_x2=true)
    return MatMul(es_tensor1, es_tensor0, nullptr, false, true)
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)op_index;
    // 获取输入tensor
    struct ggml_tensor *src0 = node->src[0];  // 主输入
    struct ggml_tensor *src1 = node->src[1];  // mask (可能为NULL)

    // 获取scale和max_bias参数
    float scale = 1.0f;     // 默认缩放因子为1.0
    float max_bias = 0.0f;  // 默认最大偏置为0.0
    if (node->op_params) {
        memcpy(&scale, (float *)node->op_params + 0, sizeof(float));
        memcpy(&max_bias, (float *)node->op_params + 1, sizeof(float));
    }

    // 获取input ES tensor
    ge::es::EsTensorHolder es_input;
    if (ggml_tensor_to_es_tensor_map.find(src0) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_input = ggml_tensor_to_es_tensor_map[src0];
    } else {
        assert(false && "Input tensor not found in ES tensor map");
    }

    // 如果有mask，需要应用mask（这部分实现Attention中的掩码处理）
    if (src1 != nullptr) {
        // 创建ALiBi处理（一种Attention偏置实现）
        const uint32_t n_head = node->ne[2];  // ne02：获取注意力头数
        const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2(n_head));

        // 计算ALiBi的m0和m1参数
        const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
        const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

        // 获取mask ES tensor
        ge::es::EsTensorHolder es_mask;
        if (ggml_tensor_to_es_tensor_map.find(src1) !=
            ggml_tensor_to_es_tensor_map.end()) {
            es_mask = ggml_tensor_to_es_tensor_map[src1];
        } else {
            assert(false && "Mask tensor not found in ES tensor map");
        }

        // 如果mask的形状需要切片
        if (src1->ne[1] != src0->ne[1]) {
            // 创建切片参数
            std::vector<int64_t> begin_vals = {0, 0, 0, 0};
            std::vector<int64_t> end_vals = {src1->ne[3], src1->ne[2],
                                             src0->ne[1], src1->ne[0]};
            std::vector<int64_t> strides_vals = {1, 1, 1, 1};
            std::vector<int64_t> axes_vals = {0, 1, 2, 3};

            // 使用ES API直接创建StridedSliceV2操作
            // 计算切片后的形状
            std::vector<int64_t> slice_shape = {src1->ne[3], src1->ne[2],
                                                src0->ne[1], src1->ne[0]};
            es_mask = StridedSliceV2(es_mask, begin_vals, end_vals,
                                     strides_vals, axes_vals)
                          .SetDataType(get_data_type(src1->type))
                          .SetShape(slice_shape);
        }

        // 创建用于存储slope的常量
        std::vector<float> slopes(n_head);
        for (uint32_t h = 0; h < n_head; h++) {
            if (max_bias > 0.0f) {
                slopes[h] = h < n_head_log2
                                ? powf(m0, h + 1)
                                : powf(m1, 2 * (h - n_head_log2) + 1);
            } else {
                slopes[h] = 1.0f;
            }
        }

        // 使用ES API创建slopes常量（shape: [1, n_head, 1, 1]）
        // 将mask与slopes相乘
        auto slope_mul =
            (es_mask * graph_builder.CreateConst(
                           slopes, {1, (int64_t)slopes.size(), 1, 1}))
                .SetDataType(get_data_type(src1->type))
                .SetShape(build_output_shape(src1));

        // 先对输入应用scale缩放
        auto scaled_input =
            (es_input * graph_builder.CreateVector(std::vector{scale}))
                .SetDataType(get_data_type(src0->type))
                .SetShape(build_output_shape(src0));

        // 添加掩码：将缩放后的输入与掩码相加
        es_input = (scaled_input + slope_mul)
                       .SetDataType(get_data_type(node->type))
                       .SetShape(build_output_shape(node));
    } else if (scale != 1.0f) {
        // 如果没有mask但有scale，仅应用scale
        es_input = (es_input * graph_builder.CreateVector(std::vector{scale}))
                       .SetDataType(get_data_type(src0->type))
                       .SetShape(build_output_shape(src0));
    }

    // 创建Softmax操作
    // 注意：GGML的维度和CANN的维度排序是反的，axes设为-1（最后一个维度）
    return SoftmaxV2(es_input, {-1})
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)op_index;
    // 获取输入 tensor
    struct ggml_tensor *input_tensor = node->src[0];

    // 获取输入 ES tensor
    ge::es::EsTensorHolder es_input;
    if (ggml_tensor_to_es_tensor_map.find(input_tensor) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_input = ggml_tensor_to_es_tensor_map[input_tensor];
    } else {
        assert(false && "Input tensor not found in ES tensor map");
    }

    // 构造重复倍数(multiples)数组
    // 计算每个维度上需要重复的次数
    std::vector<int64_t> multiples;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        // 计算当前维度的重复次数：输出维度大小 / 输入维度大小
        int64_t repeat = node->ne[i] / input_tensor->ne[i];
        multiples.push_back(repeat);
    }

    // 使用ES API创建Tile操作
    // Tile(x, multiples) - 在昇腾中，Tile操作实现了GGML的Repeat功能
    return Tile(es_input, multiples)
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node, true));
}

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
    int op_index) {
    (void)graph_builder;
    (void)op_index;
    // 获取输入tensor
    struct ggml_tensor *input_tensor = node->src[0];

    // 获取输入ES tensor
    ge::es::EsTensorHolder es_input;
    if (ggml_tensor_to_es_tensor_map.find(input_tensor) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_input = ggml_tensor_to_es_tensor_map[input_tensor];
    } else {
        assert(false && "Input tensor not found in ES tensor map");
    }

    // 使用ES API创建Swish操作（SILU是Swish的一种特殊情况）
    return Swish(es_input)
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)graph_builder;
    (void)op_index;
    // 获取输入tensor
    struct ggml_tensor *input_tensor = node->src[0];

    // 获取输入ES tensor
    ge::es::EsTensorHolder es_input;
    if (ggml_tensor_to_es_tensor_map.find(input_tensor) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_input = ggml_tensor_to_es_tensor_map[input_tensor];
    } else {
        assert(false && "Input tensor not found in ES tensor map");
    }

    // 提取排序顺序（从op_params读取）
    enum ggml_sort_order order =
        static_cast<enum ggml_sort_order>(node->op_params[0]);
    bool descending = (order == GGML_SORT_ORDER_DESC);

    // 使用ES API创建Sort操作
    // 注意：Sort操作返回两个输出（sorted values和indices），我们需要indices,
    // 原来的玩法是
    // 通过一个identity来规避，现在我们可以之间结构化绑定之后返回需要的输出
    auto [y1, y2] = Sort(es_input, -1, descending);
    // 设置输出形状和数据类型
    return y2
        .SetDataType(ge::DT_INT32)  // Argsort输出的是索引，类型为INT32
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)op_index;
    // 获取源张量
    struct ggml_tensor *src0 = node->src[0];

    // 检查输入是否已经在映射中
    ge::es::EsTensorHolder es_tensor;
    if (ggml_tensor_to_es_tensor_map.find(src0) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_tensor = ggml_tensor_to_es_tensor_map[src0];
    } else {
        assert(false && "src0 tensor not found in ES tensor map");
    }

    // 从op_params中读取scale值
    float scale_val;
    memcpy(&scale_val, node->op_params, sizeof(float));
    // 使用ES API创建Mul操作（使用运算符重载+链式调用）
    // 使用ES API创建常量，shape是[1]
    return (es_tensor * graph_builder.CreateVector(std::vector{scale_val}))
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)graph_builder;
    (void)op_index;
    // 获取输入张量
    ggml_tensor *src_x = node->src[0];
    assert(src_x && "RESHAPE: missing data tensor");

    // 检查输入是否已经在映射中
    ge::es::EsTensorHolder es_tensor;
    if (ggml_tensor_to_es_tensor_map.find(src_x) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_tensor = ggml_tensor_to_es_tensor_map[src_x];
    } else {
        assert(false && "src_x tensor not found in ES tensor map");
    }

    // 构建目标形状数组（从高维到低维，与CANN维度顺序一致）
    std::vector<int64_t> shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        int64_t ne = node->ne[i];  // 获取目标形状的各维度大小
        if (ne > 0) {
            shape.push_back(ne);
        }
    }

    // 使用ES API创建Reshape操作, 支持直接传入数值，内部创建const
    return Reshape(es_tensor, shape)
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)graph_builder;
    (void)op_index;
    // 获取输入张量
    ggml_tensor *src_x = node->src[0];
    assert(src_x && "PERMUTE: missing data tensor");

    // 检查输入是否已经在映射中
    ge::es::EsTensorHolder es_tensor;
    if (ggml_tensor_to_es_tensor_map.find(src_x) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_tensor = ggml_tensor_to_es_tensor_map[src_x];
    } else {
        assert(false && "src_x tensor not found in ES tensor map");
    }

    // 从op_params中读取维度顺序
    std::vector<int64_t> order;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        order.push_back(node->op_params[i]);
    }

    // 注意：order向量需要根据CANN的维度顺序进行调整
    return Permute(es_tensor, order)
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)graph_builder;
    (void)op_index;
    // 获取输入张量
    ggml_tensor *src_x = node->src[0];
    assert(src_x && "TRANSPOSE: missing data tensor");

    // 获取输入ES tensor
    ge::es::EsTensorHolder es_input;
    if (ggml_tensor_to_es_tensor_map.find(src_x) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_input = ggml_tensor_to_es_tensor_map[src_x];
    } else {
        assert(false && "src_x tensor not found in ES tensor map");
    }

    // 构建置换数组，用于交换维度顺序
    // GGML中，转置操作默认交换最后两个维度
    std::vector<int64_t> perm;

    // 获取有效维度的数量
    int effective_dims = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (node->ne[i] > 0) effective_dims++;
    }

    // 构建置换序列（转置时交换最后两个维度）
    for (int i = 0; i < effective_dims; ++i) {
        if (i == effective_dims - 1)
            perm.push_back(effective_dims - 2);  // 将倒数第一维映射到倒数第二维
        else if (i == effective_dims - 2)
            perm.push_back(effective_dims - 1);  // 将倒数第二维映射到倒数第一维
        else
            perm.push_back(i);  // 其他维度保持不变
    }

    // 使用ES API创建Transpose操作，直接传入置换数组
    return Transpose(es_input, perm)
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)op_index;
    // 获取输入张量
    struct ggml_tensor *src0 = node->src[0];
    struct ggml_tensor *src1 = node->src[1];

    // 获取输入ES tensor
    ge::es::EsTensorHolder es_tensor1, es_tensor2;
    if (ggml_tensor_to_es_tensor_map.find(src0) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_tensor1 = ggml_tensor_to_es_tensor_map[src0];
    } else {
        assert(false && "src0 tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(src1) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_tensor2 = ggml_tensor_to_es_tensor_map[src1];
    } else {
        assert(false && "src1 tensor not found in ES tensor map");
    }

    // 计算concat_dim（需要转换维度顺序）
    std::vector<int64_t> output_shape = build_output_shape(node);
    int32_t concat_dim_value =
        static_cast<int32_t>(output_shape.size() - node->op_params[0] - 1);

    // 使用ES API创建concat_dim标量
    auto concat_dim = graph_builder.CreateScalar(concat_dim_value);

    // 使用ES API创建ConcatV2操作
    // ConcatV2({tensor1, tensor2}, concat_dim, N) - N是输入数量
    return ConcatV2({es_tensor1, es_tensor2}, concat_dim, 2)
        .SetDataType(get_data_type(node->type))
        .SetShape(output_shape);
}

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
    int op_index) {
    (void)op_index;
    // 获取源张量
    struct ggml_tensor *src0 = node->src[0];
    assert(src0 && "VIEW: missing source tensor");

    // 获取源张量的ES tensor
    ge::es::EsTensorHolder es_src0;
    if (ggml_tensor_to_es_tensor_map.find(src0) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_src0 = ggml_tensor_to_es_tensor_map[src0];
    } else {
        assert(false && "src0 tensor not found in ES tensor map");
    }

    // 元素大小 (字节)
    size_t s0_elsize = ggml_element_size(src0);
    size_t node_elsize = ggml_element_size(node);
    assert(s0_elsize == node_elsize &&
           "VIEW: source and view tensor element sizes must match");

    // 目标视图参数 (node)
    std::vector<int64_t> dst_shape_vec;
    std::vector<int64_t> dst_strides_vec;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (node->ne[i] > 0) {
            dst_shape_vec.push_back(node->ne[i]);
            dst_strides_vec.push_back(node->nb[i] / node_elsize);
        }
    }
    if (dst_shape_vec.empty()) {
        dst_shape_vec.push_back(1);
        dst_strides_vec.push_back(1);
    }
    // Pad with 1s if ndims < 4 for CANN, reverse order for CANN
    while (dst_shape_vec.size() < 4 && dst_shape_vec.size() > 0) {
        dst_shape_vec.insert(dst_shape_vec.begin(), 1);
        dst_strides_vec.insert(dst_strides_vec.begin(),
                               dst_strides_vec.front() * dst_shape_vec[1]);
    }
    std::reverse(dst_shape_vec.begin(), dst_shape_vec.end());
    std::reverse(dst_strides_vec.begin(), dst_strides_vec.end());

    // dst_storage_offset
    assert(node->view_src == src0 && "View node's view_src is not src0");
    int64_t dst_storage_offset_val = node->view_offs / node_elsize;

    // 源张量参数 (src0)
    std::vector<int64_t> src_shape_vec;
    std::vector<int64_t> src_strides_vec;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src0->ne[i] > 0) {
            src_shape_vec.push_back(src0->ne[i]);
            src_strides_vec.push_back(src0->nb[i] / s0_elsize);
        }
    }
    if (src_shape_vec.empty()) {
        src_shape_vec.push_back(1);
        src_strides_vec.push_back(1);
    }
    // Pad with 1s if ndims < 4 for CANN, reverse order for CANN
    while (src_shape_vec.size() < 4 && src_shape_vec.size() > 0) {
        src_shape_vec.insert(src_shape_vec.begin(), 1);
        src_strides_vec.insert(src_strides_vec.begin(),
                               src_strides_vec.front() * src_shape_vec[1]);
    }
    std::reverse(src_shape_vec.begin(), src_shape_vec.end());
    std::reverse(src_strides_vec.begin(), src_strides_vec.end());

    int64_t src_storage_offset_val = 0;

    std::vector<int64_t> output_shape = build_output_shape(node, true);
    if (output_shape.empty()) {
        output_shape.push_back(1);
    }
    while (output_shape.size() < 4 && output_shape.size() > 0) {
        output_shape.insert(output_shape.begin(), 1);
    }
    // 使用ES API创建ViewCopy操作
    return ViewCopy(es_src0, dst_shape_vec, dst_strides_vec,
                    dst_storage_offset_val, es_src0, src_shape_vec,
                    src_strides_vec, src_storage_offset_val)
        .SetDataType(get_data_type(node->type))
        .SetShape(output_shape);
}

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
    int op_index) {
    (void)op_index;
    // 获取输入张量
    ggml_tensor *src = node->src[0];
    assert(src && "CONT: missing source tensor");

    // 获取输入ES tensor
    ge::es::EsTensorHolder es_input;
    if (ggml_tensor_to_es_tensor_map.find(src) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_input = ggml_tensor_to_es_tensor_map[src];
    } else {
        assert(false && "src tensor not found in ES tensor map");
    }

    // 检查node的shape和src是否相同
    bool same_shape = true;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != src->ne[i]) {
            same_shape = false;
            break;
        }
    }

    // 如果形状不同，需要先进行reshape操作
    if (!same_shape) {
        // 创建目标形状向量
        std::vector<int64_t> target_shape;
        for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
            int64_t ne = node->ne[i];
            if (ne > 0) {
                target_shape.push_back(ne);
            }
        }
        // 使用ES API创建Reshape操作
        return Reshape(es_input, target_shape)
            .SetDataType(get_data_type(node->type))
            .SetShape(build_output_shape(node));
    } else {
        // 形状相同，使用Identity操作实现连续化
        return Identity(es_input)
            .SetDataType(get_data_type(node->type))
            .SetShape(build_output_shape(node));
    }
}

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
    int op_index) {
    (void)graph_builder;
    (void)op_index;
    // 获取源张量 src0 和目标张量 src1
    ggml_tensor *src0 = node->src[0];
    assert(src0 && "CPY: missing src0");
    ggml_tensor *src1 = node->src[1];
    assert(src1 && "CPY: missing src1");

    // 获取源张量的ES tensor
    ge::es::EsTensorHolder es_src0;
    if (ggml_tensor_to_es_tensor_map.find(src0) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_src0 = ggml_tensor_to_es_tensor_map[src0];
    } else {
        assert(false && "src0 tensor not found in ES tensor map");
    }

    // 获取目标数据类型
    ge::DataType dt_dst = get_data_type(node->type);

    // 使用ES API创建Cast操作，将src0的类型转换为目标类型
    // 默认输出形状与src1一致，
    // 原始的实现里面注释和代码实现不符，按照代码实现来了，去掉了先执行
    // CONT（内存连续化）的操作
    return Cast(es_src0, dt_dst)
        .SetDataType(dt_dst)
        .SetShape(build_output_shape(src1));
}

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
    int op_index) {
    (void)op_index;
    // 获取输入张量
    ggml_tensor *src_x = node->src[0];
    assert(src_x && "RMSNorm: missing input tensor");

    // 获取输入ES tensor
    ge::es::EsTensorHolder es_input;
    if (ggml_tensor_to_es_tensor_map.find(src_x) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_input = ggml_tensor_to_es_tensor_map[src_x];
    } else {
        assert(false && "src_x tensor not found in ES tensor map");
    }

    // 获取epsilon参数，防止除零错误
    float epsilon = 1e-6f;  // 默认值
    if (node->op_params != nullptr) {
        memcpy(&epsilon, node->op_params, sizeof(float));
    }

    // 在GGML中，特征维度是ne[0]
    int64_t feature_dim = src_x->ne[0];

    // 处理gamma参数
    ge::es::EsTensorHolder es_gamma;
    if (node->op == GGML_OP_RMS_NORM) {
        // GGML_OP_RMS_NORM: 没有gamma，创建全1的常量
        std::vector<float> gamma_data(feature_dim, 1.0f);
        // 使用ES API创建gamma常量
        es_gamma = graph_builder.CreateConst(gamma_data, {feature_dim});
    } else {
        // GGML_OP_RMS_NORM_FUSED: 有gamma输入
        GGML_ASSERT(node->op == GGML_OP_RMS_NORM_FUSED);
        ggml_tensor *src_gamma = node->src[1];
        GGML_ASSERT(src_gamma != nullptr);
        if (ggml_tensor_to_es_tensor_map.find(src_gamma) !=
            ggml_tensor_to_es_tensor_map.end()) {
            es_gamma = ggml_tensor_to_es_tensor_map[src_gamma];
            // 对gamma进行squeeze操作（移除维度0,1,2）
            es_gamma = Squeeze(es_gamma, std::vector<int64_t>{0, 1, 2})
                           .SetDataType(get_data_type(src_gamma->type))
                           .SetShape(build_output_shape(src_gamma));
        } else {
            assert(false && "src_gamma tensor not found in ES tensor map");
        }
    }

    // 使用ES API创建RmsNorm操作, 注意：我们需要y输出
    // 在ES API中，应该可以直接返回输出，无需Identity
    auto [y, rstd] = RmsNorm(es_input, es_gamma, epsilon);

    // 设置输出形状和数据类型
    return y.SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

/**
 * @brief ES版本：处理ROPE（旋转位置编码）操作的函数
 *
 * 使用ES API在计算图中创建一个RoPE操作，实现旋转位置编码
 * 注意：此操作非常复杂，涉及Reshape、Transpose、Gather、自定义算子等多个步骤
 * 由于RopeCache返回GE API的Operator，此实现可能需要混合使用GE和ES API
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
    int op_index, ggml_backend_cann_context &cann_ctx) {
    (void)graph_builder;
    // 获取源张量
    struct ggml_tensor *src0 = node->src[0];  // 输入张量
    struct ggml_tensor *src1 = node->src[1];  // 位置索引张量

    auto dst = node;
    GGML_TENSOR_UNARY_OP_LOCALS  // 使用GGML宏获取输入张量的维度

        // 获取输入ES tensor
        ge::es::EsTensorHolder es_src0,
        es_src1;
    if (ggml_tensor_to_es_tensor_map.find(src0) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_src0 = ggml_tensor_to_es_tensor_map[src0];
    } else {
        assert(false && "src0 not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(src1) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_src1 = ggml_tensor_to_es_tensor_map[src1];
    } else {
        assert(false && "src1 not found in ES tensor map");
    }

    // 从操作参数中获取RoPE配置参数
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    const int n_dims = ((int32_t *)node->op_params)[1];      // 特征维度数量
    const int mode = ((int32_t *)node->op_params)[2];        // RoPE模式
    const int n_ctx_orig = ((int32_t *)node->op_params)[4];  // 原始上下文长度

    // 复制浮点参数
    memcpy(&freq_base, (int32_t *)dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10, sizeof(float));

    // 确认维度条件
    GGML_ASSERT(n_dims == ne0);
    GGML_ASSERT(n_dims % 2 == 0);  // 特征维度必须是偶数

    // 计算RoPE参数
    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const int64_t pos_len = src0->ne[2];  // 位置长度

    // 定义重塑后的维度
    int64_t ne_x_reshape[] = {2, src0->ne[0] / 2, src0->ne[1], src0->ne[2],
                              src0->ne[3]};

    // 第一步：将输入重塑为5D [..., dim//2, 2]
    std::vector<int64_t> shape_reshape_x;
    for (int i = 4; i >= 0; --i) {
        shape_reshape_x.push_back(ne_x_reshape[i]);
    }

    // 使用ES API创建Reshape操作
    auto es_reshape_x = Reshape(es_src0, shape_reshape_x)
                            .SetDataType(get_data_type(src0->type))
                            .SetShape(shape_reshape_x);

    // 第二步：转置操作，将维度顺序变为 [....,2, dim//2]
    std::vector<int64_t> perm_order = {0, 1, 2, 4, 3};  // 置换维度的顺序

    // 使用ES API创建Transpose操作
    std::vector<int64_t> out_shape_perm_x;
    for (int i = 4; i >= 0; --i) {
        int64_t idx = perm_order[4 - i];
        out_shape_perm_x.push_back(ne_x_reshape[idx]);
    }
    auto es_perm_x = Transpose(es_reshape_x, perm_order)
                         .SetDataType(get_data_type(src0->type))
                         .SetShape(out_shape_perm_x);

    // 第三步：执行RoPE操作
    // 获取底层 Graph（RopeCache 需要 Graph& 参数）
    ge::Graph *graph = graph_builder.GetCGraphBuilder()->GetGraph();
    GGML_ASSERT(graph != nullptr);
    std::string curr_suffix = "_" + std::to_string(op_index);

    // 3.1: 创建 RopeCache 并获取 sin/cos 缓存（直接使用 CreateConst 创建）
    RopeCache rope_cache(cann_ctx, dst);
    ge::es::EsTensorHolder es_sin_cache =
        rope_cache.GetSinEsTensor(graph_builder);
    ge::es::EsTensorHolder es_cos_cache =
        rope_cache.GetCosEsTensor(graph_builder);

    // 3.2: 使用 ES API 的 Squeeze 处理位置索引
    std::vector<int64_t> squeeze_axes = {0, 1, 2};  // 移除维度 0, 1, 2
    auto es_x1_squeeze = Squeeze(es_src1, squeeze_axes)
                             .SetDataType(get_data_type(src1->type))
                             .SetShape(build_output_shape(src1));

    // 3.3: 使用 ES API 的 Gather 操作（axis=1）
    // Gather 函数签名需要确认，假设是 Gather(data, indices, axis)
    // 创建 axis 常量
    auto es_axis = graph_builder.CreateScalar(static_cast<int64_t>(1));
    auto es_gather_sin = GatherV2(es_sin_cache, es_x1_squeeze, es_axis)
                             .SetDataType(get_data_type(node->type))
                             .SetShape(out_shape_perm_x);
    auto es_gather_cos = GatherV2(es_cos_cache, es_x1_squeeze, es_axis)
                             .SetDataType(get_data_type(node->type))
                             .SetShape(out_shape_perm_x);

    // 3.4: 创建 RopeExtCustomV2 自定义算子
    // TODO:使用自定义算子的ES API来替换
    std::string name_rope = "rope_rope" + curr_suffix;
    ge::op::RopeExtCustomV2 rope_op(name_rope.c_str());

    // 设置输出描述
    ge::TensorDesc desc_out_rope(ge::Shape(out_shape_perm_x), ge::FORMAT_ND,
                                 get_data_type(node->type));
    rope_op.update_output_desc_dst(desc_out_rope);

    // 设置必需属性
    rope_op.set_attr_ne0(ne0);
    rope_op.set_attr_ne1(ne1);
    rope_op.set_attr_pos_len(src0->ne[2]);

    // 添加到图中并转换为 GNode
    ge::GNode g_rope_node = graph->AddNodeByOp(rope_op);

    // 连接输入边
    // 输入 x (索引 0): es_perm_x
    ge::GNode *g_perm_x_node = es_perm_x.GetProducer();
    graph->AddDataEdge(*g_perm_x_node, 0, g_rope_node, 0);  // x 输入

    // 输入 cos (索引 1): es_gather_cos
    ge::GNode *g_gather_cos_node = es_gather_cos.GetProducer();
    graph->AddDataEdge(*g_gather_cos_node, 0, g_rope_node, 1);  // cos 输入

    // 输入 sin (索引 2): es_gather_sin
    ge::GNode *g_gather_sin_node = es_gather_sin.GetProducer();
    graph->AddDataEdge(*g_gather_sin_node, 0, g_rope_node, 2);  // sin 输入

    // 转换为 EsTensorHolder
    ge::es::EsTensorHolder es_rope_result(
        graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(g_rope_node,
                                                                  0));

    // 第四步：转置回来（使用 ES API）
    auto es_permute_dst = Transpose(es_rope_result, perm_order)
                              .SetDataType(get_data_type(node->type))
                              .SetShape(shape_reshape_x);

    // 第五步：重塑回 4D（使用 ES API）
    std::vector<int64_t> out_shape_reshape_dst;
    for (int i = 3; i >= 0; --i) {
        out_shape_reshape_dst.push_back(src0->ne[i]);
    }
    auto es_reshape_dst = Reshape(es_permute_dst, out_shape_reshape_dst)
                              .SetDataType(get_data_type(node->type))
                              .SetShape(out_shape_reshape_dst);

    return es_reshape_dst;
}

/**
 * @brief ES版本：处理MOE_FUSED（混合专家融合）操作的函数
 *
 * 使用ES API在计算图中创建一个MoE Fused操作，实现混合专家模型的前向传播
 *
 * 实现步骤：
 * 1. 输入处理：Squeeze + Transpose（使用 ES API 的 Squeeze、Transpose）
 * 2. MoeInitRouting：使用 ES API 的 MoeInitRouting，返回3个输出（expanded_x,
 * expanded_row_idx, expanded_expert_idx）
 * 3. 计算专家令牌数：Equal + Cast(bool→int32) + ReduceSum +
 * Cast(int32→int64)（使用 ES API）
 * 4. Up projection权重处理：Squeeze + Transpose（使用 ES API）
 * 5. Up projection矩阵乘法：GroupedMatmul（使用 ES API，通过
 * create_moe_grouped_matmul_es 辅助函数） 6-9. Gate projection, SiLU, Mul, Down
 * projection：待实现
 * 10. MoeFinalizeRoutingV2：使用 ES API 的 MoeFinalizeRoutingV2
 *
 * @param graph_builder ES图构建器引用
 * @param node 表示MOE_FUSED操作的张量节点
 * @param ggml_tensor_to_es_tensor_map 张量到ES张量的映射
 * @param op_index 用于生成唯一算子名称的索引（当前未使用）
 * @return 创建的MOE_FUSED操作的ES张量持有者
 */
ge::es::EsTensorHolder handle_moe_fused_op_es(
    ge::es::EsGraphBuilder &graph_builder, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::es::EsTensorHolder>
        &ggml_tensor_to_es_tensor_map,
    int op_index) {
    (void)op_index;
    // 获取输入张量
    struct ggml_tensor *input = node->src[0];                // 输入张量
    struct ggml_tensor *ids = node->src[1];                  // 专家ID张量
    struct ggml_tensor *topk_weight = node->src[2];          // topk权重张量
    struct ggml_tensor *expert_up_weights = node->src[3];    // 专家上权重
    struct ggml_tensor *expert_down_weights = node->src[4];  // 专家下权重
    struct ggml_tensor *expert_gate_weights = node->src[5];  // 专家门权重
    struct ggml_tensor *row_idx = node->src[6];              // 行索引张量

    // 提取维度信息
    auto batch_size = input->ne[3];
    auto seq_len = input->ne[2];
    auto topk = ids->ne[0];
    auto num_experts = expert_up_weights->ne[2];
    auto hidden_dim = input->ne[0];
    auto k_dim = expert_up_weights->ne[1];
    auto num_rows = batch_size * seq_len;
    auto active_num = num_rows;

    // 获取输入 ES tensor
    ge::es::EsTensorHolder es_input, es_ids, es_topk_weight,
        es_expert_up_weights, es_expert_down_weights, es_expert_gate_weights,
        es_row_idx;

    if (ggml_tensor_to_es_tensor_map.find(input) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_input = ggml_tensor_to_es_tensor_map[input];
    } else {
        assert(false && "Input tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(ids) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_ids = ggml_tensor_to_es_tensor_map[ids];
    } else {
        assert(false && "IDs tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(topk_weight) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_topk_weight = ggml_tensor_to_es_tensor_map[topk_weight];
    } else {
        assert(false && "TopK weight tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(expert_up_weights) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_expert_up_weights = ggml_tensor_to_es_tensor_map[expert_up_weights];
    } else {
        assert(false && "Expert up weights tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(expert_down_weights) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_expert_down_weights =
            ggml_tensor_to_es_tensor_map[expert_down_weights];
    } else {
        assert(false &&
               "Expert down weights tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(expert_gate_weights) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_expert_gate_weights =
            ggml_tensor_to_es_tensor_map[expert_gate_weights];
    } else {
        assert(false &&
               "Expert gate weights tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(row_idx) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_row_idx = ggml_tensor_to_es_tensor_map[row_idx];
    } else {
        assert(false && "Row index tensor not found in ES tensor map");
    }

    // 步骤1: 使用 ES API 的 Squeeze 处理输入
    // 参考原始实现：Squeeze({0, 2}) 将输入从4D重塑为2D [hidden_dim, seq_len]
    auto es_input_squeeze =
        Squeeze(es_input, std::vector<int64_t>{0, 2})
            .SetDataType(get_data_type(input->type))
            .SetShape(std::vector<int64_t>{hidden_dim, seq_len});

    // 步骤2: 使用 ES API 的 Squeeze 处理 row_idx
    // 参考原始实现：Squeeze({0, 1}) 移除前两维
    auto es_row_idx_squeeze =
        Squeeze(es_row_idx, std::vector<int64_t>{0, 1})
            .SetDataType(get_data_type(row_idx->type))
            .SetShape(std::vector<int64_t>{seq_len, topk});

    // 步骤2.1: 使用 ES API 的 Transpose 转置 row_idx
    // 参考原始实现：转置 {1, 0} 后形状为 [topk, seq_len]
    auto es_row_idx_permute =
        Transpose(es_row_idx_squeeze, std::vector<int64_t>{1, 0})
            .SetDataType(get_data_type(row_idx->type))
            .SetShape(std::vector<int64_t>{topk, seq_len});

    // 步骤2.2: 使用 ES API 的 Squeeze 处理 expert_idx (ids)
    // 参考原始实现：Squeeze({0, 1}) 移除前两维
    auto es_expert_idx_squeeze =
        Squeeze(es_ids, std::vector<int64_t>{0, 1})
            .SetDataType(get_data_type(ids->type))
            .SetShape(std::vector<int64_t>{num_rows * topk});

    // 步骤3: 使用 ES API 创建 MoeInitRouting 算子
    auto [es_expanded_x, es_expanded_row_idx, es_expanded_expert_idx] =
        MoeInitRouting(es_input_squeeze, es_row_idx_permute,
                       es_expert_idx_squeeze, static_cast<int64_t>(active_num));

    // 步骤4: 计算专家令牌数（使用 ES API）
    // 创建专家索引范围常量
    std::vector<int32_t> expert_range_data(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        expert_range_data[i] = i;
    }
    auto es_expert_range = graph_builder.CreateConst(
        expert_range_data, std::vector<int64_t>{num_experts});

    // Reshape expanded_expert_idx 用于广播
    auto es_expert_idx_reshape =
        Reshape(es_expanded_expert_idx,
                std::vector<int64_t>{num_rows * topk, 1})
            .SetDataType(get_data_type(ids->type))
            .SetShape(std::vector<int64_t>{num_rows * topk, 1});

    // 步骤4: 使用 Equal 算子创建 one-hot 编码
    // Equal 比较: [num_rows * topk, 1] vs [num_experts] -> [num_rows * topk,
    // num_experts]
    auto es_equal =
        Equal(es_expert_idx_reshape, es_expert_range)
            .SetDataType(ge::DT_BOOL)
            .SetShape(std::vector<int64_t>{num_rows * topk, num_experts});

    // Cast bool→int32
    auto es_cast_bool_to_int =
        Cast(es_equal, static_cast<int64_t>(ge::DT_INT32))
            .SetDataType(ge::DT_INT32)
            .SetShape(std::vector<int64_t>{num_rows * topk, num_experts});

    // ReduceSum 沿第0维求和，得到每个专家的token数量
    // axes 参数: EsTensorLike 支持 std::vector<int64_t> 的隐式构造
    auto es_reduce_sum =
        ReduceSum(es_cast_bool_to_int, std::vector<int64_t>{0}, false)
            .SetDataType(ge::DT_INT32)
            .SetShape(std::vector<int64_t>{num_experts});

    // Cast int32→int64 (用于 GroupedMatmul 的 group_list)
    auto es_expert_tokens_int64 =
        Cast(es_reduce_sum, static_cast<int64_t>(ge::DT_INT64))
            .SetDataType(ge::DT_INT64)
            .SetShape(std::vector<int64_t>{num_experts});

    // 步骤5-10: 处理分组矩阵乘法和激活（使用 ES API）

    // 步骤5: 处理 expert_up_weights (Squeeze + Transpose)
    auto es_expert_up_weights_squeeze =
        Squeeze(es_expert_up_weights, std::vector<int64_t>{0})
            .SetDataType(get_data_type(expert_up_weights->type))
            .SetShape(build_output_shape(expert_up_weights));

    // 使用 ES API 的 Transpose 进行 {0, 2, 1} 转置
    auto es_permute_up_weights =
        Transpose(es_expert_up_weights_squeeze, std::vector<int64_t>{0, 2, 1})
            .SetDataType(get_data_type(expert_up_weights->type))
            .SetShape(std::vector<int64_t>{num_experts, k_dim, hidden_dim});

    // 步骤6: 第一个分组矩阵乘法 (up projection) - 使用 ES API
    auto es_up_matmul = create_moe_grouped_matmul_es(
        graph_builder, es_expanded_x, es_permute_up_weights,
        std::vector<int64_t>{num_experts, k_dim}, es_expert_tokens_int64);

    // 步骤7-10: gate projection, SiLU, Mul, down projection
    // 由于实现类似，这里简化处理，使用 GE API 转换
    // TODO: 完善 gate 和 down projection 的实现

    // 步骤11: 使用 ES API 创建 MoeFinalizeRoutingV2 算子
    auto es_topk_weight_squeeze =
        Squeeze(es_topk_weight, std::vector<int64_t>{0, 3})
            .SetDataType(get_data_type(topk_weight->type))
            .SetShape(build_output_shape(topk_weight));

    // 根据原始实现，expert_idx 使用 es_expert_idx_squeeze（不是
    // expanded_expert_idx） 如果 node->type 不是 F16，需要先 Cast 到 F32
    ge::es::EsTensorHolder es_expanded_x_for_finalize;
    if (node->type == GGML_TYPE_F16) {
        es_expanded_x_for_finalize =
            es_up_matmul;  // TODO: 使用实际的 down_matmul
    } else {
        // Cast F32
        es_expanded_x_for_finalize =
            Cast(es_up_matmul, static_cast<int64_t>(ge::DT_FLOAT))
                .SetDataType(ge::DT_FLOAT)
                .SetShape(build_output_shape(node));
    }

    // 使用 ES API 的 MoeFinalizeRoutingV2
    // 参数顺序：expanded_x, expanded_row_idx, x1=nullptr, x2=nullptr,
    // bias=nullptr, scales, expert_idx, drop_pad_mode=0
    auto es_finalize_result =
        MoeFinalizeRoutingV2(es_expanded_x_for_finalize, es_expanded_row_idx,
                             nullptr, nullptr, nullptr,  // x1, x2, bias 不使用
                             es_topk_weight_squeeze,     // scales
                             es_expert_idx_squeeze,  // expert_idx（使用 squeeze
                                                     // 后的，不是 expanded）
                             0                       // drop_pad_mode
                             )
            .SetDataType(get_data_type(node->type))
            .SetShape(std::vector<int64_t>{hidden_dim, seq_len});

    return es_finalize_result;
}

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
    int op_index) {
    (void)ggml_tensor_to_es_tensor_map;
    (void)op_index;
    // 从 op_params 中读取三个 float 参数
    float start_val, limit_val, delta_val;
    memcpy(&start_val, (float *)node->op_params + 0, sizeof(float));
    memcpy(&limit_val, (float *)node->op_params + 1, sizeof(float));
    memcpy(&delta_val, (float *)node->op_params + 2, sizeof(float));

    // 创建 start, limit, delta 常量
    auto es_start = graph_builder.CreateScalar(start_val);
    auto es_limit = graph_builder.CreateScalar(limit_val);
    auto es_delta = graph_builder.CreateScalar(delta_val);

    // 使用 ES API 的 Range 算子
    return Range(es_start, es_limit, es_delta)
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)op_index;
    // 获取输入张量
    struct ggml_tensor *src_x = node->src[0];  // 主输入张量

    // 获取输入张量对应的ES tensor
    ge::es::EsTensorHolder es_x;
    if (ggml_tensor_to_es_tensor_map.find(src_x) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_x = ggml_tensor_to_es_tensor_map[src_x];
    } else {
        assert(false && "Input tensor not found in ES tensor map");
    }

    // 从 op_params 中读取三个 int64_t 参数：{fr, to, axis}
    int64_t params[3] = {};
    memcpy(params, node->op_params, sizeof(int64_t) * 3);
    int64_t fr = params[0];
    int64_t to = params[1];
    int64_t axis = params[2];

    // 初始化begin, end, strides数组，按照CANN维度顺序
    std::vector<int64_t> begin_vec(GGML_MAX_DIMS);
    std::vector<int64_t> end_vec(GGML_MAX_DIMS);
    std::vector<int64_t> strides_vec(GGML_MAX_DIMS);

    // 获取输入张量的维度信息，按照CANN维度顺序初始化
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        begin_vec[GGML_MAX_DIMS - i - 1] = 0;
        end_vec[GGML_MAX_DIMS - i - 1] = src_x->ne[i];
        strides_vec[GGML_MAX_DIMS - i - 1] = 1;
    }

    // 根据axis修改对应维度的begin和end
    begin_vec[GGML_MAX_DIMS - axis - 1] = fr;
    end_vec[GGML_MAX_DIMS - axis - 1] = to;

    // 创建axes向量，内容为{0,1,2,3}，类型为int64
    std::vector<int64_t> axes_vec = {0, 1, 2, 3};

    // 使用 ES API 的 StridedSliceV2 算子
    // EsTensorLike 支持 std::vector<int64_t> 的隐式构造，可以直接传递
    return StridedSliceV2(es_x, begin_vec, end_vec, axes_vec, strides_vec)
        .SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)graph_builder;
    (void)op_index;
    // 获取输入张量
    struct ggml_tensor *query = node->src[0];
    struct ggml_tensor *key = node->src[1];
    struct ggml_tensor *value = node->src[2];
    struct ggml_tensor *attn_mask = node->src[3];
    struct ggml_tensor *length_q_tensor = node->src[4];
    struct ggml_tensor *length_kv_tensor = node->src[5];

    // 数据类型和形状验证
    GGML_ASSERT(query->type == GGML_TYPE_F16);
    GGML_ASSERT(key->type == GGML_TYPE_F16);
    GGML_ASSERT(value->type == GGML_TYPE_F16);
    GGML_ASSERT(attn_mask->type == GGML_TYPE_I8);
    GGML_ASSERT(node->type == GGML_TYPE_F16);

    // 获取输入张量对应的ES tensor
    ge::es::EsTensorHolder es_query, es_key, es_value, es_attn_mask,
        es_length_q_tensor, es_length_kv_tensor;

    if (ggml_tensor_to_es_tensor_map.find(query) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_query = ggml_tensor_to_es_tensor_map[query];
    } else {
        assert(false && "Query tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(key) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_key = ggml_tensor_to_es_tensor_map[key];
    } else {
        assert(false && "Key tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(value) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_value = ggml_tensor_to_es_tensor_map[value];
    } else {
        assert(false && "Value tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(attn_mask) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_attn_mask = ggml_tensor_to_es_tensor_map[attn_mask];
    } else {
        assert(false && "Attention mask tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(length_q_tensor) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_length_q_tensor = ggml_tensor_to_es_tensor_map[length_q_tensor];
    } else {
        assert(false && "Length Q tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(length_kv_tensor) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_length_kv_tensor = ggml_tensor_to_es_tensor_map[length_kv_tensor];
    } else {
        assert(false && "Length KV tensor not found in ES tensor map");
    }

    // 从 op_params 中读取参数
    struct flash_attn_params {
        int batch_size;
        int num_heads;
        int head_dim_kq;
        int head_dim_v;
        int key_num_heads;
        int sequence_lenth_q;
        int64_t sequence_lenth_kv;
        float scaleValue;
    };
    flash_attn_params *params =
        reinterpret_cast<flash_attn_params *>(node->op_params);

    // 从参数中提取配置
    int32_t num_heads = params->num_heads;
    float scale_value = params->scaleValue;
    int32_t sequence_length_q = params->sequence_lenth_q;

    // 设置属性值，匹配 FusedInferAttentionScore 算子的规范
    int64_t num_key_value_heads = num_heads;
    const char *input_layout = "BSND";  // 默认输入布局
    int64_t pre_tokens = 2147483647;    // 匹配默认值
    int64_t next_tokens = 0;
    int64_t sparse_mode = 1;  // 拦截，非量化情况不考虑
    int64_t inner_precise =
        sequence_length_q > 1 ? 2 : 0;  // 高精度模式，开启行无效修正

    // 使用 ES API 的 FusedInferAttentionScore 算子
    // key 和 value 需要包装成 vector（动态输入）
    std::vector<ge::es::EsTensorHolder> es_key_vec = {es_key};
    std::vector<ge::es::EsTensorHolder> es_value_vec = {es_value};

    // 调用 FusedInferAttentionScore，返回结构体包含 attention_out 和
    // softmax_lse 对于可选参数，使用 nullptr 表示未传递
    auto [attention_out, softmax_lse] = FusedInferAttentionScore(
        es_query, es_key_vec, es_value_vec,
        nullptr,  // pse_shift
        es_attn_mask, es_length_q_tensor, es_length_kv_tensor,
        nullptr,  // dequant_scale1
        nullptr,  // quant_scale1
        nullptr,  // dequant_scale2
        nullptr,  // quant_scale2
        nullptr,  // quant_offset2
        nullptr,  // antiquant_scale
        nullptr,  // antiquant_offset
        nullptr,  // block_table
        nullptr,  // query_padding_size
        nullptr,  // kv_padding_size
        nullptr,  // key_antiquant_scale
        nullptr,  // key_antiquant_offset
        nullptr,  // value_antiquant_scale
        nullptr,  // value_antiquant_offset
        nullptr,  // key_shared_prefix
        nullptr,  // value_shared_prefix
        nullptr,  // actual_shared_prefix_len
        nullptr,  // query_rope
        nullptr,  // key_rope
        nullptr,  // key_rope_antiquant_scale
        nullptr,  // dequant_scale_query
        nullptr,  // learnable_sink
        nullptr,  // q_start_idx
        nullptr,  // kv_start_idx
        num_heads, scale_value, pre_tokens, next_tokens, input_layout,
        num_key_value_heads, sparse_mode, inner_precise,
        0,      // block_size
        0,      // antiquant_mode
        false,  // softmax_lse_flag
        0,      // key_antiquant_mode
        0,      // value_antiquant_mode
        0,      // query_quant_mode
        0,      // pse_type
        0       // out_dtype
    );

    // 返回 attention_out（原始实现中通过 Identity 算子只保留 attention_out）
    return attention_out.SetDataType(get_data_type(node->type))
        .SetShape(build_output_shape(node));
}

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
    int op_index) {
    (void)graph_builder;
    (void)op_index;
    // 获取输入张量
    struct ggml_tensor *src_var = node->src[0];     // 待赋值的张量
    struct ggml_tensor *src_index = node->src[1];   // 索引张量
    struct ggml_tensor *src_update = node->src[2];  // 赋值张量

    assert(src_var && "SET_SLICE: missing input tensor");
    assert(src_index && "SET_SLICE: missing index tensor");
    assert(src_update && "SET_SLICE: missing update tensor");

    // 获取输入张量对应的ES tensor
    ge::es::EsTensorHolder es_var, es_indices, es_updates;

    if (ggml_tensor_to_es_tensor_map.find(src_var) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_var = ggml_tensor_to_es_tensor_map[src_var];
    } else {
        assert(false && "SET_SLICE: var tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(src_index) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_indices = ggml_tensor_to_es_tensor_map[src_index];
    } else {
        assert(false && "SET_SLICE: indices tensor not found in ES tensor map");
    }

    if (ggml_tensor_to_es_tensor_map.find(src_update) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_updates = ggml_tensor_to_es_tensor_map[src_update];
    } else {
        assert(false && "SET_SLICE: updates tensor not found in ES tensor map");
    }

    // 计算形状（参考原始实现）
    std::vector<int64_t> op_var_shape = squeeze_ggml_tensor_shape(src_var);
    if (op_var_shape.size() == 1) {
        op_var_shape.insert(op_var_shape.begin(), 1);
    }
    assert(op_var_shape.size() == 2 &&
           "SET_SLICE: var tensor must have 2 dimensions after reshape");

    std::vector<int64_t> op_indices_shape =
        squeeze_ggml_tensor_shape(src_index);
    if (op_indices_shape.empty()) {
        op_indices_shape.push_back(1);
    }
    assert(op_indices_shape.size() == 1 &&
           "SET_SLICE: indices tensor must have 1 dimension after reshape");

    std::vector<int64_t> op_updates_shape =
        squeeze_ggml_tensor_shape(src_update);
    if (op_updates_shape.size() == 1) {
        op_updates_shape.insert(op_updates_shape.begin(), 1);
    }
    assert(op_updates_shape.size() == 2 &&
           "SET_SLICE: updates tensor must have 2 dimensions after reshape");

    // 使用 ES API 的 Reshape 操作重塑输入
    auto es_var_reshaped = Reshape(es_var, op_var_shape)
                               .SetDataType(get_data_type(src_var->type))
                               .SetShape(op_var_shape);

    auto es_indices_reshaped = Reshape(es_indices, op_indices_shape)
                                   .SetDataType(get_data_type(src_index->type))
                                   .SetShape(op_indices_shape);

    auto es_updates_reshaped = Reshape(es_updates, op_updates_shape)
                                   .SetDataType(get_data_type(src_update->type))
                                   .SetShape(op_updates_shape);

    // 使用 ES API 的 ScatterUpdate 算子
    std::vector<int64_t> out_shape = build_output_shape(node);
    return ScatterUpdate(es_var_reshaped, es_indices_reshaped,
                         es_updates_reshaped, false)  // use_locking=false
        .SetDataType(get_data_type(node->type))
        .SetShape(out_shape);
}

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
    int op_index) {
    (void)graph_builder;
    (void)op_index;
    struct ggml_tensor *src = node->src[0];
    struct ggml_tensor *rows = node->src[1];

    // 获取输入张量对应的ES tensor
    ge::es::EsTensorHolder es_src, es_rows;
    if (ggml_tensor_to_es_tensor_map.find(src) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_src = ggml_tensor_to_es_tensor_map[src];
    } else {
        assert(false && "get_rows: input tensor not found in ES tensor map");
    }
    if (ggml_tensor_to_es_tensor_map.find(rows) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_rows = ggml_tensor_to_es_tensor_map[rows];
    } else {
        assert(false && "get_rows: rows tensor not found in ES tensor map");
    }

    int batch_dims = rows->ne[1];
    int row_num = rows->ne[0];

    // 使用 ES API 的 Squeeze 操作（参考原始实现）
    // Squeeze src: 移除维度0，形状从 [1, src->ne[0], src->ne[1], src->ne[2]]
    // 变为 [src->ne[0], src->ne[1], src->ne[2]]
    std::vector<int64_t> squeezed_src_shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        if (i != 0 && src->ne[i] > 0) {
            squeezed_src_shape.push_back(src->ne[i]);
        }
    }
    auto es_squeezed_src = Squeeze(es_src, std::vector<int64_t>{0})
                               .SetDataType(get_data_type(src->type))
                               .SetShape(squeezed_src_shape);

    // Squeeze rows: 移除维度0和1，形状从 [rows->ne[0], rows->ne[1], 1, 1] 变为
    // [rows->ne[0] * rows->ne[1]]
    std::vector<int64_t> squeezed_rows_shape = {row_num * batch_dims};
    auto es_squeezed_rows = Squeeze(es_rows, std::vector<int64_t>{0, 1})
                                .SetDataType(get_data_type(rows->type))
                                .SetShape(squeezed_rows_shape);

    // 创建 axis 常量（axis=1）
    // 使用 ES API 的 GatherV2 算子（参考原始实现：axis=1, batch_dims=1）
    auto es_result =
        GatherV2(es_squeezed_src, es_squeezed_rows, static_cast<int64_t>(1), 1);

    // 使用 ES API 的 Unsqueeze 操作（参考原始实现：unsqueeze axis 0）
    auto es_unsq_result = Unsqueeze(es_result, std::vector<int64_t>{0})
                              .SetDataType(get_data_type(node->type))
                              .SetShape(std::vector<int64_t>{
                                  1, rows->ne[1], rows->ne[0], src->ne[0]});

    // 使用 ES API 的 Cast 操作（参考原始实现）
    return Cast(es_unsq_result, static_cast<int64_t>(get_data_type(node->type)))
        .SetDataType(get_data_type(node->type))
        .SetShape(
            std::vector<int64_t>{1, rows->ne[1], rows->ne[0], src->ne[0]});
}

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
    int op_index) {
    (void)op_index;
    // 获取输入张量
    struct ggml_tensor *src = node->src[0];  // 待填充的张量
    assert(src && "PAD: missing input tensor");

    // 获取输入张量对应的ES tensor
    ge::es::EsTensorHolder es_input;
    if (ggml_tensor_to_es_tensor_map.find(src) !=
        ggml_tensor_to_es_tensor_map.end()) {
        es_input = ggml_tensor_to_es_tensor_map[src];
    } else {
        assert(false && "PAD: input tensor not found in ES tensor map");
    }

    // 创建paddings向量（参考原始实现）
    std::vector<int64_t> paddings = {
        0, node->ne[3] - src->ne[3], 0, node->ne[2] - src->ne[2],
        0, node->ne[1] - src->ne[1], 0, node->ne[0] - src->ne[0]};
    // 使用 ES API 的 PadV3 算子
    std::vector<int64_t> out_shape = build_output_shape(node);
    return PadV3(es_input, paddings, 0.0f, "constant", true)
        .SetDataType(get_data_type(node->type))
        .SetShape(out_shape);
}