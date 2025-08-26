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
 * @brief 打印算子的输出形状信息
 *
 * 用于调试目的，打印特定算子的输出张量形状
 *
 * @param op 要打印形状的算子
 * @param reverse 是否以反转顺序打印维度(默认为true)
 */
void print_op_shape(ge::Operator &op, bool reverse = true) {
    auto output_desc = op.GetOutputDesc(0);
    auto shape = output_desc.GetShape();
    std::cout << op.GetName() << " output shape: ";
    if (reverse) {
        for (size_t i = shape.GetDimNum(); i > 0; --i) {
            std::cout << shape.GetDim(i - 1) << " ";
        }
    } else {
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            std::cout << shape.GetDim(i) << " ";
        }
    }
    std::cout << std::endl;
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
 * @brief 处理ADD（加法）操作的函数
 *
 * 在计算图中创建一个加法操作，将它与输入连接起来
 * 支持两个输入张量的形状不同时的广播处理
 *
 * @param graph 计算图引用
 * @param node 表示ADD操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的ADD算子
 */
ge::Operator handle_add_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // 获取源张量
    struct ggml_tensor *src0 = node->src[0];
    struct ggml_tensor *src1 = node->src[1];

    // 检查输入是否已经在映射中
    ge::Operator op_x1, op_x2;

    // 处理src0 - 获取现有算子或创建新的数据算子
    if (gmml_tensor_to_ge_op_map.find(src0) != gmml_tensor_to_ge_op_map.end()) {
        op_x1 = gmml_tensor_to_ge_op_map[src0];
    } else {
        assert(false);
    }

    // 处理src1 - 获取现有算子或创建新的数据算子
    if (gmml_tensor_to_ge_op_map.find(src1) != gmml_tensor_to_ge_op_map.end()) {
        op_x2 = gmml_tensor_to_ge_op_map[src1];
    } else {
        assert(false);
    }

    // 处理广播
    int64_t out_ne[GGML_MAX_DIMS], nb0[GGML_MAX_DIMS], nb1[GGML_MAX_DIMS];
    bool need_tile0[GGML_MAX_DIMS], need_tile1[GGML_MAX_DIMS];
    bcast_shape(src0, src1, out_ne, nb0, nb1, need_tile0, need_tile1);

    // 如果src0需要平铺(Tile)，调用handle_repeat_op
    if (std::any_of(need_tile0, need_tile0 + GGML_MAX_DIMS,
                    [](bool x) { return x; })) {
        op_x1 =
            handle_repeat_op(graph, node, gmml_tensor_to_ge_op_map, op_index);
    }
    // 如果src1需要平铺，同样处理
    if (std::any_of(need_tile1, need_tile1 + GGML_MAX_DIMS,
                    [](bool x) { return x; })) {
        struct ggml_tensor *node1 = node;
        node1->src[0] = node1->src[1];
        op_x2 =
            handle_repeat_op(graph, node1, gmml_tensor_to_ge_op_map, op_index);
    }

    // 创建Add算子
    std::string add_name = "add_" + std::to_string(op_index);
    ge::op::Add add_op(add_name);
    // 设置输入
    add_op.set_input_x1(op_x1);
    add_op.set_input_x2(op_x2);

    // 设置输出张量描述符
    std::vector<int64_t> output_shape = build_output_shape(node);
    ge::DataType dataType = get_data_type(node->type);
    ge::TensorDesc desc_out(ge::Shape(output_shape), ge::FORMAT_ND, dataType);
    add_op.update_output_desc_y(desc_out);

    // 将算子添加到图中
    graph.AddOp(add_op);

    return add_op;
}

/**
 * @brief 处理MUL（乘法）操作的函数
 *
 * 在计算图中创建一个乘法操作，处理两个输入张量
 * 支持形状不同时的广播处理
 *
 * @param graph 计算图引用
 * @param node 表示MUL操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的MUL算子
 */
ge::Operator handle_mul_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    ggml_tensor *src0 = node->src[0];
    ggml_tensor *src1 = node->src[1];

    ge::Operator op_x1, op_x2;
    // 处理src0 - 获取现有算子
    if (gmml_tensor_to_ge_op_map.find(src0) != gmml_tensor_to_ge_op_map.end()) {
        op_x1 = gmml_tensor_to_ge_op_map[src0];
    } else {
        assert(false);
    }

    // 处理src1 - 获取现有算子
    if (gmml_tensor_to_ge_op_map.find(src1) != gmml_tensor_to_ge_op_map.end()) {
        op_x2 = gmml_tensor_to_ge_op_map[src1];
    } else {
        assert(false);
    }

    // 处理广播情况
    int64_t out_ne[GGML_MAX_DIMS], nb0[GGML_MAX_DIMS], nb1[GGML_MAX_DIMS];
    bool need_tile0[GGML_MAX_DIMS], need_tile1[GGML_MAX_DIMS];
    bcast_shape(src0, src1, out_ne, nb0, nb1, need_tile0, need_tile1);

    // 处理需要平铺的情况
    if (std::any_of(need_tile0, need_tile0 + GGML_MAX_DIMS,
                    [](bool x) { return x; })) {
        op_x1 =
            handle_repeat_op(graph, node, gmml_tensor_to_ge_op_map, op_index);
    }
    if (std::any_of(need_tile1, need_tile1 + GGML_MAX_DIMS,
                    [](bool x) { return x; })) {
        ggml_tensor *saved = node->src[0];
        node->src[0] = node->src[1];
        op_x2 =
            handle_repeat_op(graph, node, gmml_tensor_to_ge_op_map, op_index);
        node->src[0] = saved;
    }

    // 创建乘法算子
    std::string mul_name = "mul_" + std::to_string(op_index);
    ge::op::Mul mul_op(mul_name);
    mul_op.set_input_x1(op_x1);
    mul_op.set_input_x2(op_x2);

    // 设置输出形状和数据类型
    std::vector<int64_t> output_shape = build_output_shape(node);
    ge::DataType dataType = get_data_type(node->type);
    ge::TensorDesc desc(ge::Shape(output_shape), ge::FORMAT_ND, dataType);
    mul_op.update_output_desc_y(desc);

    // 将算子添加到图中
    graph.AddOp(mul_op);
    return mul_op;
}

/**
 * @brief 处理矩阵乘法(MATMUL)操作的函数
 *
 * 在计算图中创建一个矩阵乘法操作
 * 注意矩阵乘法需要特别处理转置等问题
 *
 * @param graph 计算图引用
 * @param node 表示矩阵乘法操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的矩阵乘法算子
 */
ge::Operator handle_matmul_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    ggml_tensor *src0 = node->src[0];
    ggml_tensor *src1 = node->src[1];

    ge::Operator op_a, op_b;
    if (gmml_tensor_to_ge_op_map.find(src0) != gmml_tensor_to_ge_op_map.end()) {
        op_a = gmml_tensor_to_ge_op_map[src0];
    } else {
        assert(false);
    }

    // 处理src1 - 获取现有算子
    if (gmml_tensor_to_ge_op_map.find(src1) != gmml_tensor_to_ge_op_map.end()) {
        op_b = gmml_tensor_to_ge_op_map[src1];
    } else {
        assert(false);
    }

    // 创建矩阵乘法算子
    std::string name = "matmul_" + std::to_string(op_index);
    ge::op::BatchMatMulV2 matmul_op(name);

    // 注意：在昇腾中，矩阵乘法的输入顺序是反的
    matmul_op.set_input_x1(op_b);  // 对应GGML中的src1
    matmul_op.set_input_x2(op_a);  // 对应GGML中的src0

    // 设置adj_x2=true，表示第二个输入(GGML中的src0)需要转置
    matmul_op.set_attr_adj_x2(true);

    // 设置输出形状和数据类型
    std::vector<int64_t> output_shape = build_output_shape(node);
    ge::DataType dataType = get_data_type(node->type);
    ge::TensorDesc desc_out(ge::Shape(output_shape), ge::FORMAT_ND, dataType);
    matmul_op.update_output_desc_y(desc_out);

    // 添加到图中
    graph.AddOp(matmul_op);

    return matmul_op;
}

/**
 * @brief 处理Softmax操作的函数
 *
 * 在计算图中创建Softmax操作，支持带掩码(mask)和缩放(scale)的情况
 * 处理包括Attention中的ALiBi缩放等特殊情况
 *
 * @param graph 计算图引用
 * @param node 表示Softmax操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的Softmax算子
 */
ge::Operator handle_softmax_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
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

    // 获取input operator
    ge::Operator op_input;
    if (gmml_tensor_to_ge_op_map.find(src0) != gmml_tensor_to_ge_op_map.end()) {
        op_input = gmml_tensor_to_ge_op_map[src0];
    } else {
        assert(false && "Input tensor not found in gmml_tensor_to_ge_op_map");
    }

    // 如果有mask，需要应用mask（这部分实现Attention中的掩码处理）
    if (src1 != nullptr) {
        // 创建ALiBi处理（一种Attention偏置实现）
        const uint32_t n_head = node->ne[2];  // ne02：获取注意力头数
        const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2(n_head));

        // 计算ALiBi的m0和m1参数
        // 这些参数控制不同头的衰减率
        const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
        const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

        // 获取mask operator
        ge::Operator op_mask;
        if (gmml_tensor_to_ge_op_map.find(src1) !=
            gmml_tensor_to_ge_op_map.end()) {
            op_mask = gmml_tensor_to_ge_op_map[src1];
        } else {
            assert(false &&
                   "Mask tensor not found in gmml_tensor_to_ge_op_map");
        }
        if (src1->ne[1] != src0->ne[1]) {
            // op_mask = op_mask[:,:, src0->ne[1], :] slice 参考stride_slice_op
            // 创建切片参数常量
            std::vector<int64_t> begin_vals = {0, 0, 0, 0};
            std::vector<int64_t> end_vals = {src1->ne[3], src1->ne[2],
                                             src0->ne[1], src1->ne[0]};
            std::vector<int64_t> strides_vals = {1, 1, 1, 1};

            // 创建begin常量
            std::string begin_name =
                "softmax_mask_slice_begin_" + std::to_string(op_index);
            auto begin_const =
                create_const_1d_op(graph, begin_name, begin_vals, ge::DT_INT64);

            // 创建end常量
            std::string end_name =
                "softmax_mask_slice_end_" + std::to_string(op_index);
            auto end_const =
                create_const_1d_op(graph, end_name, end_vals, ge::DT_INT64);

            // 创建strides常量
            std::string strides_name =
                "softmax_mask_slice_strides_" + std::to_string(op_index);
            auto strides_const = create_const_1d_op(graph, strides_name,
                                                    strides_vals, ge::DT_INT64);

            // 创建axes常量
            std::vector<int64_t> axes_vals = {0, 1, 2, 3};
            std::string axes_name =
                "softmax_mask_slice_axes_" + std::to_string(op_index);
            auto axes_const =
                create_const_1d_op(graph, axes_name, axes_vals, ge::DT_INT64);

            // 创建StridedSliceV2操作
            std::string slice_name =
                "softmax_mask_slice_" + std::to_string(op_index);
            ge::op::StridedSliceV2 slice_op(slice_name);
            slice_op.set_input_x(op_mask);
            slice_op.set_input_begin(begin_const);
            slice_op.set_input_end(end_const);
            slice_op.set_input_strides(strides_const);
            slice_op.set_input_axes(axes_const);

            // 设置输出形状
            std::vector<int64_t> slice_shape = {src1->ne[3], src1->ne[2],
                                                src0->ne[1], src1->ne[0]};
            ge::TensorDesc slice_desc(ge::Shape(slice_shape), ge::FORMAT_ND,
                                      get_data_type(src1->type));
            slice_op.update_output_desc_y(slice_desc);

            graph.AddOp(slice_op);
            op_mask = slice_op;
        }
        // 创建用于存储slope的常量
        // slopes数组包含每个注意力头的缩放因子
        std::vector<float> slopes(n_head);
        for (uint32_t h = 0; h < n_head; h++) {
            if (max_bias > 0.0f) {
                // 根据头索引计算slope值
                slopes[h] = h < n_head_log2
                                ? powf(m0, h + 1)
                                : powf(m1, 2 * (h - n_head_log2) + 1);
            } else {
                slopes[h] = 1.0f;  // 不使用ALiBi时默认为1.0
            }
        }

        // 创建slopes常量操作
        std::string slopes_name = "softmax_slopes_" + std::to_string(op_index);
        ge::op::Const slopes_const_op(slopes_name.c_str());

        // 设置slopes常量的形状和数据
        ge::TensorDesc slopes_desc(ge::Shape({1, (int64_t)slopes.size(), 1, 1}),
                                   ge::FORMAT_ND, ge::DT_FLOAT);
        ge::Tensor slopes_tensor(slopes_desc,
                                 reinterpret_cast<uint8_t *>(slopes.data()),
                                 slopes.size() * sizeof(float));
        slopes_const_op.set_attr_value(slopes_tensor);

        // 将mask与slopes相乘，实现每个头的不同缩放
        std::string slope_mul_name =
            "softmax_slope_mul_" + std::to_string(op_index);
        ge::op::Mul slope_mul_op(slope_mul_name.c_str());
        slope_mul_op.set_input_x1(op_mask);
        slope_mul_op.set_input_x2(slopes_const_op);
        slope_mul_op.update_output_desc_y(op_mask.GetOutputDesc(0));
        graph.AddOp(slope_mul_op);

        // 使用Add操作将掩码添加到输入
        std::string add_name = "softmax_mask_add_" + std::to_string(op_index);
        ge::op::Add add_op(add_name.c_str());

        // 先对输入应用scale缩放
        std::string scale_name = "softmax_scale_" + std::to_string(op_index);
        ge::op::Const scale_const_op(scale_name.c_str());
        ge::TensorDesc scale_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
        ge::Tensor scale_tensor(scale_desc, reinterpret_cast<uint8_t *>(&scale),
                                sizeof(float));
        scale_const_op.set_attr_value(scale_tensor);

        // 创建乘法算子应用scale
        std::string mul_name = "softmax_scale_mul_" + std::to_string(op_index);
        ge::op::Mul mul_op(mul_name.c_str());
        mul_op.set_input_x1(op_input);
        mul_op.set_input_x2(scale_const_op);

        // 设置乘法输出形状
        std::vector<int64_t> input_shape = build_output_shape(src0);
        ge::TensorDesc mul_desc(ge::Shape(input_shape), ge::FORMAT_ND,
                                get_data_type(src0->type));
        mul_op.update_output_desc_y(mul_desc);

        graph.AddOp(mul_op);

        // 添加掩码：将缩放后的输入与掩码相加
        add_op.set_input_x1(mul_op);
        add_op.set_input_x2(slope_mul_op);

        // 设置加法输出形状
        std::vector<int64_t> output_shape = build_output_shape(node);
        ge::TensorDesc add_desc(ge::Shape(output_shape), ge::FORMAT_ND,
                                get_data_type(node->type));
        add_op.update_output_desc_y(add_desc);

        graph.AddOp(add_op);

        // 更新op_input为添加了mask的值，用于后续的Softmax操作
        op_input = add_op;
    } else if (scale != 1.0f) {
        // 如果没有mask但有scale，仅应用scale
        std::string scale_name = "softmax_scale_" + std::to_string(op_index);
        ge::op::Const scale_const_op(scale_name.c_str());
        ge::TensorDesc scale_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
        ge::Tensor scale_tensor(scale_desc, reinterpret_cast<uint8_t *>(&scale),
                                sizeof(float));
        scale_const_op.set_attr_value(scale_tensor);

        // 创建乘法算子应用scale
        std::string mul_name = "softmax_scale_mul_" + std::to_string(op_index);
        ge::op::Mul mul_op(mul_name.c_str());
        mul_op.set_input_x1(op_input);
        mul_op.set_input_x2(scale_const_op);

        // 设置乘法输出形状
        std::vector<int64_t> input_shape = build_output_shape(src0);
        ge::TensorDesc mul_desc(ge::Shape(input_shape), ge::FORMAT_ND,
                                get_data_type(src0->type));
        mul_op.update_output_desc_y(mul_desc);

        graph.AddOp(mul_op);

        // 更新op_input为缩放后的值
        op_input = mul_op;
    }

    // 创建SoftmaxV2操作
    std::string softmax_name = "softmax_" + std::to_string(op_index);
    ge::op::SoftmaxV2 softmax_op(softmax_name.c_str());

    // 设置输入
    softmax_op.set_input_x(op_input);

    // 设置axes属性
    // 在GGML中，维度顺序是从0开始的，需要确定正确的softmax轴
    // 注意：GGML的维度和CANN的维度排序是反的
    // 在GGML中是[n0, n1, n2, n3]，但在CANN中会变成[n3, n2, n1, n0]
    // 所以axes设为-1（最后一个维度）
    std::vector<int64_t> output_shape = build_output_shape(node);
    std::vector<int64_t> axes = {-1};  // 使用最后一个维度，与文档中的默认值一致
    softmax_op.set_attr_axes(axes);

    // 设置输出描述
    ge::DataType dataType = get_data_type(node->type);
    ge::TensorDesc desc_out(ge::Shape(output_shape), ge::FORMAT_ND, dataType);
    softmax_op.update_output_desc_y(desc_out);

    // 添加到图中
    graph.AddOp(softmax_op);

    return softmax_op;
}

/**
 * @brief 处理重复(Repeat)操作的函数
 *
 * 实现张量的重复操作，使用昇腾的Tile算子
 * 根据输入张量和输出张量的维度差异，计算重复倍数并生成重复操作
 *
 * @param graph 计算图引用
 * @param node 表示Repeat操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的Tile(重复)算子
 */
ge::Operator handle_repeat_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // 获取输入 tensor
    struct ggml_tensor *input_tensor = node->src[0];

    // 找到输入 operator
    ge::Operator input_op;
    if (gmml_tensor_to_ge_op_map.find(input_tensor) !=
        gmml_tensor_to_ge_op_map.end()) {
        input_op = gmml_tensor_to_ge_op_map[input_tensor];
    } else {
        assert(false && "Input tensor not found in gmml_tensor_to_ge_op_map");
    }

    // 构造重复倍数(multiples)数组
    // 计算每个维度上需要重复的次数
    std::vector<int64_t> multiples;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        // 计算当前维度的重复次数：输出维度大小 / 输入维度大小
        int64_t repeat = node->ne[i] / input_tensor->ne[i];
        multiples.push_back(repeat);
    }

    // 构造multiples常量张量
    std::string const_name = "repeat_multiples_" + std::to_string(op_index);
    ge::op::Const multiples_const_op(const_name);

    // 设置multiples常量的形状和数据
    ge::TensorDesc multiples_desc(ge::Shape({(int64_t)multiples.size()}),
                                  ge::FORMAT_ND, ge::DT_INT64);
    ge::Tensor multiples_tensor(multiples_desc,
                                reinterpret_cast<uint8_t *>(multiples.data()),
                                multiples.size() * sizeof(int64_t));
    multiples_const_op.set_attr_value(multiples_tensor);

    // 将常量操作添加到图中，确保输入能正确连接
    graph.AddOp(multiples_const_op);

    // 构造 Tile 算子（在昇腾中，Tile操作实现了GGML的Repeat功能）
    std::string tile_name = "repeat_tile_" + std::to_string(op_index);
    ge::op::Tile tile_op(tile_name);
    tile_op.set_input_x(input_op);                    // 设置输入
    tile_op.set_input_multiples(multiples_const_op);  // 设置重复倍数

    // 设置输出描述
    std::vector<int64_t> output_shape = build_output_shape(node, true);
    ge::DataType dataType = get_data_type(node->type);
    ge::TensorDesc out_desc(ge::Shape(output_shape), ge::FORMAT_ND, dataType);
    tile_op.update_output_desc_y(out_desc);

    // 添加 Tile 算子到图中
    graph.AddOp(tile_op);

    return tile_op;
}

ge::Operator handle_silu_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // Get input tensor
    struct ggml_tensor *input_tensor = node->src[0];

    // Find input operator in the map
    ge::Operator input_op;
    if (gmml_tensor_to_ge_op_map.find(input_tensor) !=
        gmml_tensor_to_ge_op_map.end()) {
        input_op = gmml_tensor_to_ge_op_map[input_tensor];
    } else {
        assert(false && "Input tensor not found in gmml_tensor_to_ge_op_map");
    }

    // Create Swish operator
    std::string swish_name = "silu_" + std::to_string(op_index);
    ge::op::Swish swish_op(swish_name);

    // Set input
    swish_op.set_input_x(input_op);

    // Set output tensor descriptor
    std::vector<int64_t> output_shape = build_output_shape(node);
    ge::DataType dataType = get_data_type(node->type);
    ge::TensorDesc desc_out(ge::Shape(output_shape), ge::FORMAT_ND, dataType);
    swish_op.update_output_desc_y(desc_out);

    // Add operator to graph
    graph.AddOp(swish_op);

    return swish_op;
}

ge::Operator handle_argsort_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // Get input tensor
    struct ggml_tensor *input_tensor = node->src[0];

    // Get operator for input tensor
    auto it = gmml_tensor_to_ge_op_map.find(input_tensor);
    if (it == gmml_tensor_to_ge_op_map.end()) {
        fprintf(stderr,
                "Error: input tensor not found in gmml_tensor_to_ge_op_map\n");
        assert(false && "Input tensor not found in gmml_tensor_to_ge_op_map");
    }
    ge::Operator input_op = it->second;

    // Create Sort operator
    std::string sort_name = "argsort_" + std::to_string(op_index);
    ge::op::Sort sort_op(sort_name);

    // Set input
    sort_op.set_input_x(input_op);

    // Set attributes
    // For GGML_OP_ARGSORT, we assume axis is the last dimension
    // In GGML it's always sorting along the last axis
    // sort_op.set_attr_axis(-1);

    // Set descending order based on GGML sort order (GGML enum has ASC/DESC
    // options) Extract sort order from op_params
    enum ggml_sort_order order =
        static_cast<enum ggml_sort_order>(node->op_params[0]);
    sort_op.set_attr_descending(order == GGML_SORT_ORDER_DESC);

    // Set stable sort (default to false as per the operator definition)
    // sort_op.set_attr_stable(false);

    // Set output tensor descriptors
    std::vector<int64_t> output_shape = build_output_shape(node);

    // GGML_OP_ARGSORT只需要indices输出，不需要sorted values
    // 只设置indices输出，使其成为单输出操作
    ge::TensorDesc desc_indices(ge::Shape(output_shape), ge::FORMAT_ND,
                                ge::DT_INT32);
    sort_op.update_output_desc_y2(desc_indices);

    // Add operator to graph
    graph.AddOp(sort_op);

    // 多一个Identity算子，因为Argsort的输出是单输出
    std::string identity_name = "argsort_identity_" + std::to_string(op_index);
    ge::op::Identity identity_op(identity_name);
    identity_op.set_input_x_by_name(sort_op, "y2");
    identity_op.update_output_desc_y(desc_indices);
    graph.AddOp(identity_op);

    return identity_op;
}

ge::Operator handle_scale_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    struct ggml_tensor *src0 = node->src[0];
    ge::Operator op_x, op_s;
    if (gmml_tensor_to_ge_op_map.count(src0)) {
        op_x = gmml_tensor_to_ge_op_map[src0];
    } else {
        assert(false && "SCALE: missing x in gmml_tensor_to_ge_op_map");
    }

    float scale_val;
    memcpy(&scale_val, node->op_params, sizeof(float));

    std::vector<float> scale;

    scale.push_back(scale_val);

    std::string const_name = "scale_const_" + std::to_string(op_index);
    ge::op::Const const_op(const_name);

    ge::TensorDesc scale_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor scale_tensor(scale_desc,
                            reinterpret_cast<uint8_t *>(scale.data()),
                            scale.size() * sizeof(float));
    const_op.set_attr_value(scale_tensor);

    // 将常量操作添加到图中，确保输入能正确连接
    graph.AddOp(const_op);

    op_s = const_op;

    std::string name = "scale_mul_" + std::to_string(op_index);
    ge::op::Mul mul_op(name);

    mul_op.set_input_x1(op_x);
    mul_op.set_input_x2(op_s);

    std::vector<int64_t> output_shape = build_output_shape(node);
    ge::DataType dt = get_data_type(node->type);
    ge::TensorDesc desc_out(ge::Shape(output_shape), ge::FORMAT_ND, dt);
    mul_op.update_output_desc_y(desc_out);

    graph.AddOp(mul_op);

    return mul_op;
}

/**
 * @brief 处理重塑(Reshape)操作的函数
 *
 * 实现张量形状的重塑操作，在不改变数据内容的情况下改变张量的维度结构
 *
 * @param graph 计算图引用
 * @param node 表示Reshape操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的Reshape算子
 */
ge::Operator handle_reshape_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // 获取输入张量
    ggml_tensor *src_x = node->src[0];
    assert(src_x && "RESHAPE: missing data tensor");

    // 获取输入张量对应的操作符
    assert(gmml_tensor_to_ge_op_map.count(src_x));
    ge::Operator op_x = gmml_tensor_to_ge_op_map[src_x];

    // 构建新的形状数组
    std::vector<int64_t> shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        int64_t ne = node->ne[i];  // 获取目标形状的各维度大小
        shape.push_back(ne);
    }

    // 创建形状常量操作符
    std::string const_name = "reshape_shape_" + std::to_string(op_index);
    ge::op::Const shape_const_op(const_name);

    // 设置形状常量的值
    ge::TensorDesc shape_desc(ge::Shape({(int64_t)shape.size()}), ge::FORMAT_ND,
                              ge::DT_INT64);
    ge::Tensor shape_tensor(shape_desc,
                            reinterpret_cast<uint8_t *>(shape.data()),
                            shape.size() * sizeof(int64_t));
    shape_const_op.set_attr_value(shape_tensor);

    // 将常量操作添加到图中，确保输入能正确连接
    graph.AddOp(shape_const_op);

    // 创建Reshape操作符
    std::string name = "reshape_" + std::to_string(op_index);
    ge::op::Reshape reshape_op(name);

    // 设置Reshape的输入和形状
    reshape_op.set_input_x(op_x);
    reshape_op.set_input_shape(shape_const_op);

    // 设置重塑属性
    reshape_op.set_attr_axis(0);       // 从第一个维度开始重塑
    reshape_op.set_attr_num_axes(-1);  // 重塑所有维度

    // 设置输出形状和类型
    std::vector<int64_t> out_shape = build_output_shape(node);
    ge::TensorDesc desc_out(ge::Shape(out_shape), ge::FORMAT_ND,
                            get_data_type(node->type));
    reshape_op.update_output_desc_y(desc_out);

    // 添加到图中
    graph.AddOp(reshape_op);

    return reshape_op;
}

ge::Operator handle_permute_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    ggml_tensor *src_x = node->src[0];
    assert(src_x && "PERMUTE: missing data tensor");

    auto it = gmml_tensor_to_ge_op_map.find(src_x);
    assert(it != gmml_tensor_to_ge_op_map.end());
    ge::Operator op_x = it->second;

    std::vector<int64_t> order;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        order.push_back(node->op_params[i]);
    }

    std::string name = "permute_" + std::to_string(op_index);
    ge::op::Permute perm_op(name);

    perm_op.set_input_x(op_x);

    perm_op.set_attr_order(order);

    std::vector<int64_t> out_shape = build_output_shape(node);
    ge::TensorDesc desc_out(ge::Shape(out_shape), ge::FORMAT_ND,
                            get_data_type(node->type));
    perm_op.update_output_desc_y(desc_out);

    graph.AddOp(perm_op);
    return perm_op;
}

/**
 * @brief 处理转置(Transpose)操作的函数
 *
 * 实现张量的维度转置，交换维度顺序
 *
 * @param graph 计算图引用
 * @param node 表示Transpose操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的Transpose算子
 */
ge::Operator handle_transpose_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // 获取输入张量
    ggml_tensor *src_x = node->src[0];
    assert(src_x && "TRANSPOSE: missing data tensor");
    assert(gmml_tensor_to_ge_op_map.count(src_x));

    // 获取输入张量对应的操作符
    ge::Operator op_x = gmml_tensor_to_ge_op_map[src_x];

    // 构建置换数组，用于交换维度顺序
    // GGML中，转置操作默认交换0维和1维
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

    // 创建维度置换序列的常量操作符
    std::string const_name = "transpose_perm_" + std::to_string(op_index);
    ge::op::Const perm_const_op(const_name);

    // 设置置换序列常量的值
    ge::TensorDesc perm_desc(ge::Shape({(int64_t)perm.size()}), ge::FORMAT_ND,
                             ge::DT_INT64);
    ge::Tensor perm_tensor(perm_desc, reinterpret_cast<uint8_t *>(perm.data()),
                           perm.size() * sizeof(int64_t));
    perm_const_op.set_attr_value(perm_tensor);

    // 将常量操作添加到图中，确保输入能正确连接
    graph.AddOp(perm_const_op);

    // 创建Transpose操作符
    std::string name = "transpose_" + std::to_string(op_index);
    ge::op::Transpose transpose_op(name);

    // 设置Transpose的输入和置换序列
    transpose_op.set_input_x(op_x);
    transpose_op.set_input_perm(perm_const_op);

    // 设置输出形状和类型
    std::vector<int64_t> out_shape = build_output_shape(node);
    ge::TensorDesc desc_out(ge::Shape(out_shape), ge::FORMAT_ND,
                            get_data_type(node->type));
    transpose_op.update_output_desc_y(desc_out);

    // 添加到图中
    graph.AddOp(transpose_op);
    return transpose_op;
}

ge::Operator handle_concat_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    struct ggml_tensor *src0 = node->src[0];
    struct ggml_tensor *src1 = node->src[1];

    // Check if inputs are already in the map
    ge::Operator op_x1, op_x2;

    // Handle src0 - either get existing operator or create a new data operator
    if (gmml_tensor_to_ge_op_map.find(src0) != gmml_tensor_to_ge_op_map.end()) {
        op_x1 = gmml_tensor_to_ge_op_map[src0];
    } else {
        assert(false);
    }

    // Handle src1 - either get existing operator or create a new data operator
    if (gmml_tensor_to_ge_op_map.find(src1) != gmml_tensor_to_ge_op_map.end()) {
        op_x2 = gmml_tensor_to_ge_op_map[src1];
    } else {
        assert(false);
    }

    std::vector<int64_t> output_shape = build_output_shape(node);

    std::string concat_name = "concat_" + std::to_string(op_index);
    ge::TensorDesc dim_tensor_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
    int32_t concat_dim_value = output_shape.size() - node->op_params[0] - 1;
    ge::Tensor dim_tensor(dim_tensor_desc,
                          reinterpret_cast<uint8_t *>(&concat_dim_value),
                          sizeof(int32_t));
    auto concat_dim = ge::op::Const((concat_name + "_dim").c_str());
    concat_dim.set_attr_value(dim_tensor);
    concat_dim.update_output_desc_y(
        ge::TensorDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32));

    // 将常量操作添加到图中，确保输入能正确连接
    graph.AddOp(concat_dim);

    ge::op::Concat concat_op(concat_name.c_str());
    concat_op.create_dynamic_input_x(2);

    concat_op.set_dynamic_input_x(0, op_x1);
    concat_op.set_dynamic_input_x(1, op_x2);

    concat_op.set_input_concat_dim(concat_dim).set_attr_N(2);
    concat_op.update_input_desc_concat_dim(dim_tensor_desc);

    ge::DataType dataType = get_data_type(node->type);
    ge::TensorDesc desc_out(ge::Shape(output_shape), ge::FORMAT_ND, dataType);
    concat_op.update_output_desc_y(desc_out);

    // GGML_LOG_INFO("concat_%d: %s, src0: %s(%d), src1: %s(%d)\n", op_index,
    // node->name, src0->name, op_x1.GetOutputDesc(0).GetDataType(), src1->name,
    // op_x2.GetOutputDesc(0).GetDataType());

    // Add operator to graph
    graph.AddOp(concat_op);
    return concat_op;
}

// Helper function to create a scalar Const operator
ge::Operator create_const_scalar_op(ge::Graph &graph, const std::string &name,
                                    int64_t value, ge::DataType type) {
    ge::op::Const const_op(name);
    ge::TensorDesc desc(ge::Shape({1}), ge::FORMAT_ND, type);
    ge::Tensor tensor(desc);
    if (type == ge::DT_INT32) {
        int32_t val = static_cast<int32_t>(value);
        tensor.SetData(reinterpret_cast<uint8_t *>(&val), sizeof(int32_t));
    } else {  // Default to INT64
        tensor.SetData(reinterpret_cast<uint8_t *>(&value), sizeof(int64_t));
    }
    const_op.set_attr_value(tensor);
    graph.AddOp(const_op);
    return const_op;
}

// Helper function to create a 1D Const operator from a vector
ge::Operator create_const_1d_op(ge::Graph &graph, const std::string &name,
                                const std::vector<int64_t> &values,
                                ge::DataType type) {
    ge::op::Const const_op(name);
    ge::TensorDesc desc(ge::Shape({(int64_t)values.size()}), ge::FORMAT_ND,
                        type);
    ge::Tensor tensor(desc);

    if (type == ge::DT_INT32) {
        std::vector<int32_t> int32_values;
        int32_values.reserve(values.size());
        for (int64_t val : values) {
            int32_values.push_back(static_cast<int32_t>(val));
        }
        tensor.SetData(reinterpret_cast<uint8_t *>(int32_values.data()),
                       int32_values.size() * sizeof(int32_t));
    } else {  // Default to INT64
        // Create a mutable copy for SetData
        std::vector<int64_t> mutable_values = values;
        tensor.SetData(reinterpret_cast<uint8_t *>(mutable_values.data()),
                       mutable_values.size() * sizeof(int64_t));
    }
    const_op.set_attr_value(tensor);
    graph.AddOp(const_op);
    return const_op;
}

/**
 * @brief 处理视图(View)操作的函数
 *
 * 使用ViewCopy操作创建张量的视图。
 *
 * @param graph 计算图引用
 * @param node 表示View操作的张量节点 (目标视图)
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的ViewCopy算子
 */
ge::Operator handle_view_op(
    ge::Graph &graph, struct ggml_tensor *node,  // node is the destination view
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // 获取源张量
    struct ggml_tensor *src0 = node->src[0];
    assert(src0 && "VIEW: missing source tensor");
    assert(gmml_tensor_to_ge_op_map.count(src0) &&
           "VIEW: source tensor not in map");
    ge::Operator op_s0 = gmml_tensor_to_ge_op_map[src0];

    // 元素大小 (字节)
    size_t s0_elsize = ggml_element_size(src0);
    size_t node_elsize = ggml_element_size(node);
    assert(s0_elsize == node_elsize &&
           "VIEW: source and view tensor element sizes must match");

    // 目标视图参数 (node)
    std::vector<int64_t> dst_shape_vec;
    std::vector<int64_t> dst_strides_vec;
    int dst_ndims = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (node->ne[i] > 0) {
            dst_shape_vec.push_back(node->ne[i]);
            dst_strides_vec.push_back(node->nb[i] /
                                      node_elsize);  // element-wise stride
            dst_ndims++;
        }
    }
    if (dst_shape_vec.empty()) {  // Scalar case
        dst_shape_vec.push_back(1);
        dst_strides_vec.push_back(1);  // Stride for scalar is typically 1
        dst_ndims = 1;
    }
    // Pad with 1s if ndims < 4 for CANN, reverse order for CANN
    while (dst_shape_vec.size() < 4 && dst_shape_vec.size() > 0) {
        dst_shape_vec.insert(dst_shape_vec.begin(), 1);
        dst_strides_vec.insert(
            dst_strides_vec.begin(),
            dst_strides_vec.front() *
                dst_shape_vec[1]);  // maintain contiguity for prepended dims
    }
    std::reverse(dst_shape_vec.begin(), dst_shape_vec.end());
    std::reverse(dst_strides_vec.begin(), dst_strides_vec.end());

    // dst_storage_offset is the offset of the view (node) within its source
    // (src0)
    assert(node->view_src == src0 && "View node's view_src is not src0");
    int64_t dst_storage_offset_val =
        node->view_offs / node_elsize;  // element-wise offset

    // 源张量参数 (src0)
    std::vector<int64_t> src_shape_vec;
    std::vector<int64_t> src_strides_vec;
    int src_ndims = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src0->ne[i] > 0) {
            src_shape_vec.push_back(src0->ne[i]);
            src_strides_vec.push_back(src0->nb[i] /
                                      s0_elsize);  // element-wise stride
            src_ndims++;
        }
    }
    if (src_shape_vec.empty()) {  // Scalar case
        src_shape_vec.push_back(1);
        src_strides_vec.push_back(1);
        src_ndims = 1;
    }
    // Pad with 1s if ndims < 4 for CANN, reverse order for CANN
    while (src_shape_vec.size() < 4 && src_shape_vec.size() > 0) {
        src_shape_vec.insert(src_shape_vec.begin(), 1);
        src_strides_vec.insert(src_strides_vec.begin(),
                               src_strides_vec.front() * src_shape_vec[1]);
    }
    std::reverse(src_shape_vec.begin(), src_shape_vec.end());
    std::reverse(src_strides_vec.begin(), src_strides_vec.end());

    // src_storage_offset for ViewCopy is relative to the src tensor itself.
    // If src0 is already a view, its op_s0 already points to the correct data
    // region. So, the offset relative to op_s0's data is 0.
    int64_t src_storage_offset_val = 0;  // element-wise offset

    // 创建Const操作符
    // dst: A tensor. (used as a template for shape/type by ViewCopy, data comes
    // from src) dst_size: A tensor. Must be one of the following types: int32,
    // int64. dst_stride: A tensor. Must be one of the following types: int32,
    // int64. dst_storage_offset: A tensor. Must be one of the following types:
    // int32, int64. src: A tensor. src_size: A tensor. Must be one of the
    // following types: int32, int64. src_stride: A tensor. Must be one of the
    // following types: int32, int64. src_storage_offset: the storage_offset of
    // src tensor . Must be one of the following types: int32, int64.

    ge::Operator const_dst_size_op =
        create_const_1d_op(graph, "view_dst_size_" + std::to_string(op_index),
                           dst_shape_vec, ge::DT_INT64);
    ge::Operator const_dst_stride_op =
        create_const_1d_op(graph, "view_dst_stride_" + std::to_string(op_index),
                           dst_strides_vec, ge::DT_INT64);
    ge::Operator const_dst_storage_offset_op = create_const_scalar_op(
        graph, "view_dst_offset_" + std::to_string(op_index),
        dst_storage_offset_val, ge::DT_INT64);

    ge::Operator const_src_size_op =
        create_const_1d_op(graph, "view_src_size_" + std::to_string(op_index),
                           src_shape_vec, ge::DT_INT64);
    ge::Operator const_src_stride_op =
        create_const_1d_op(graph, "view_src_stride_" + std::to_string(op_index),
                           src_strides_vec, ge::DT_INT64);
    ge::Operator const_src_storage_offset_op = create_const_scalar_op(
        graph, "view_src_offset_" + std::to_string(op_index),
        src_storage_offset_val, ge::DT_INT64);

    // 创建 ViewCopy 操作
    std::string view_copy_name = "view_copy_" + std::to_string(op_index);
    ge::op::ViewCopy view_copy_op(view_copy_name);

    // 设置输入
    // The 'dst' input to ViewCopy is a bit misleading. It's more of a template
    // for the output tensor's metadata if the operator were to create a new
    // buffer. However, ViewCopy reuses the src's buffer. We pass op_s0 as it
    // has the correct data type. The actual view parameters (shape, strides,
    // offset for the *view* itself) are passed via dst_size, dst_stride,
    // dst_storage_offset.
    view_copy_op.set_input_dst(
        op_s0);  // Input tensor to be viewed (acts as template)
    view_copy_op.set_input_dst_size(const_dst_size_op);
    view_copy_op.set_input_dst_stride(const_dst_stride_op);
    view_copy_op.set_input_dst_storage_offset(const_dst_storage_offset_op);

    view_copy_op.set_input_src(op_s0);  // Source tensor providing the data
    view_copy_op.set_input_src_size(const_src_size_op);
    view_copy_op.set_input_src_stride(const_src_stride_op);
    view_copy_op.set_input_src_storage_offset(const_src_storage_offset_op);

    // 设置输出描述符 (与 node 一致)
    // The output shape of ViewCopy should match the *view's* shape (node->ne)
    // build_output_shape typically reverses dimensions for CANN.
    std::vector<int64_t> output_shape_for_desc = build_output_shape(node, true);
    if (output_shape_for_desc.empty()) {  // Scalar case
        output_shape_for_desc.push_back(1);
    }
    // Pad with 1s if ndims < 4 for CANN
    while (output_shape_for_desc.size() < 4 &&
           output_shape_for_desc.size() > 0) {
        output_shape_for_desc.insert(output_shape_for_desc.begin(), 1);
    }

    ge::TensorDesc desc_out(ge::Shape(output_shape_for_desc), ge::FORMAT_ND,
                            get_data_type(node->type));
    view_copy_op.update_output_desc_dst(
        desc_out);  // Output is named 'dst' for ViewCopy

    graph.AddOp(view_copy_op);
    return view_copy_op;
}

/**
 * @brief 处理连续(Cont)操作的函数
 *
 * 将张量转换为内存连续排列的新张量
 * 对于深度学习加速器，连续内存排列的张量通常能获得更好的性能
 *
 * @param graph 计算图引用
 * @param node 表示Cont操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的Identity算子，实现连续化
 */
ge::Operator handle_cont_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // 获取输入张量
    ggml_tensor *src = node->src[0];
    assert(src && "CONT: missing source tensor");

    // 获取输入张量对应的操作符
    assert(gmml_tensor_to_ge_op_map.count(src));
    ge::Operator op_x = gmml_tensor_to_ge_op_map[src];

    // 检查node的shape和src是否相同，不同则调用reshape
    bool same_shape = true;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != src->ne[i]) {
            same_shape = false;
            break;
        }
    }

    // 如果形状不同，需要先进行reshape操作
    ge::Operator output_op;
    if (!same_shape) {
        // 创建目标形状向量
        std::vector<int64_t> target_shape = build_output_shape(node);
        // 使用通用的reshape函数创建reshape操作
        std::string reshape_name = "cont_reshape_" + std::to_string(op_index);
        output_op = create_reshape_op(graph, op_x, target_shape, reshape_name,
                                      get_data_type(node->type));
    } else {
        // 创建Identity操作符，实现连续化
        // 在昇腾中，不需要特别处理连续化，通过Identity操作实现
        std::string name = "cont_" + std::to_string(op_index);
        ge::op::Identity cont_op(name);

        // 设置Identity的输入
        cont_op.set_input_x(op_x);

        // 设置输出张量的描述：形状和数据类型与输入一致
        std::vector<int64_t> out_shape = build_output_shape(node);
        ge::TensorDesc desc_out(
            ge::Shape(out_shape),  // 和输入一样的形状
            ge::FORMAT_ND,
            get_data_type(node->type)  // 和输入一样的数据类型
        );
        cont_op.update_output_desc_y(desc_out);

        // 添加到图中
        graph.AddOp(cont_op);
        output_op = cont_op;
    }
    return output_op;
}

ge::Operator handle_cpy_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // 1) 取出源张量 src0 和占位 dst src1
    ggml_tensor *src0 = node->src[0];
    assert(src0 && "CPY: missing src0");
    ggml_tensor *src1 = node->src[1];
    assert(src1 && "CPY: missing src1");

    // 2) 先执行 CONT（内存连续化）
    assert(gmml_tensor_to_ge_op_map.count(src0));
    // ge::Operator op_cont = handle_cont_op(graph, node,
    // gmml_tensor_to_ge_op_map, op_index);

    // 3) 如果 CONT 出来的类型 != node 期望的类型，就插入 Cast
    ge::Operator op_x = gmml_tensor_to_ge_op_map[src0];
    ge::DataType dt_dst = get_data_type(node->type);

    // 构造 Cast 算子
    std::string cast_name = "cpy_cast_" + std::to_string(op_index);
    ge::op::Cast cast_op(cast_name);

    cast_op.set_input_x(op_x);
    // 设置目标类型
    cast_op.set_attr_dst_type(dt_dst);

    // 输出 descriptor: shape 和 src1 一致
    std::vector<int64_t> dst_shape = build_output_shape(src1);
    ge::TensorDesc desc_cast_out(ge::Shape(dst_shape), ge::FORMAT_ND, dt_dst);
    cast_op.update_output_desc_y(desc_cast_out);

    op_x = cast_op;

    // 4) 最终把 op_x 加入图，并返回
    graph.AddOp(op_x);
    return op_x;
}

/**
 * @brief 处理RMS归一化(RMSNorm)操作的函数
 *
 * 实现Root Mean Square归一化，是Transformer架构中常用的归一化方法
 * 与LayerNorm相比，RMSNorm仅使用均方根而不减去均值，计算更简单高效
 *
 * @param graph 计算图引用
 * @param node 表示RMSNorm操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的RMSNorm算子
 */
ge::Operator handle_rms_norm_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // 获取输入张量
    ggml_tensor *src_x = node->src[0];
    assert(src_x && "RMSNorm: missing input tensor");

    // 获取输入张量对应的操作符
    assert(gmml_tensor_to_ge_op_map.count(src_x));
    ge::Operator op_x = gmml_tensor_to_ge_op_map[src_x];

    // 获取epsilon参数，防止除零错误
    float epsilon = 1e-6f;  // 默认值
    if (node->op_params != nullptr) {
        memcpy(&epsilon, node->op_params, sizeof(float));
    }

    auto get_shape_str = [](const ge::Operator &op) {
        std::string shape_str = "[";
        for (const auto &dim : op.GetOutputDesc(0).GetShape().GetDims()) {
            shape_str += std::to_string(dim) + ", ";
        }
        if (!shape_str.empty()) {
            shape_str.pop_back();
            shape_str.pop_back();
        }
        shape_str += "]";
        return shape_str;
    };

    // 在GGML中，特征维度是ne[0]
    int64_t feature_dim = src_x->ne[0];

    // GGML_LOG_INFO("FRMSNorm with input %s: %s, feature_dim = %ld\n",
    // src_x->name, get_shape_str(op_x).c_str(), feature_dim);

    // 创建RMSNorm操作符
    std::string name = "rmsnorm_" + std::to_string(op_index);
    ge::op::RmsNorm rms_norm_op(name);

    // 设置RMSNorm的输入和参数
    rms_norm_op.set_input_x(op_x);          // 主输入
    rms_norm_op.set_attr_epsilon(epsilon);  // 设置epsilon值，防止除零

    if (node->op == GGML_OP_RMS_NORM) {
        GGML_ASSERT(node->src[1] == nullptr);
        std::vector<ggml_fp16_t> gamma_data(feature_dim);
        for (size_t i = 0; i < (size_t)feature_dim; i++) {
            gamma_data[i] = ggml_fp32_to_fp16(1.0f);
        }
        std::string const_name = "rms_norm_gamma_" + std::to_string(op_index);
        ge::op::Const gamma_const_op(const_name);

        // 设置gamma常量的形状和数据
        ge::TensorDesc gamma_desc(ge::Shape({feature_dim}), ge::FORMAT_ND,
                                  ge::DT_FLOAT16);
        ge::Tensor gamma_tensor(gamma_desc,
                                reinterpret_cast<uint8_t *>(gamma_data.data()),
                                gamma_data.size() * sizeof(ggml_fp16_t));
        gamma_const_op.set_attr_value(gamma_tensor);

        graph.AddOp(gamma_const_op);

        rms_norm_op.set_input_gamma(gamma_const_op);  // 缩放因子
    } else {
        GGML_ASSERT(node->op == GGML_OP_RMS_NORM_FUSED);
        ggml_tensor *src_gamma = node->src[1];
        GGML_ASSERT(src_gamma != nullptr);
        GGML_ASSERT(gmml_tensor_to_ge_op_map.count(src_gamma));
        ge::Operator op_gamma = gmml_tensor_to_ge_op_map[src_gamma];
        ge::Operator op_squeeze_gamma = create_squeeze_op(
            graph, "rms_norm_", "_" + std::to_string(op_index), op_gamma,
            {0, 1, 2});
        rms_norm_op.set_input_gamma(op_squeeze_gamma);
    }

    // 设置输出形状和类型
    std::vector<int64_t> out_shape = build_output_shape(node);
    ge::TensorDesc desc_out(ge::Shape(out_shape), ge::FORMAT_ND,
                            get_data_type(node->type));

    // 推理阶段只需要RMS Norm的单输出
    rms_norm_op.update_output_desc_y(desc_out);
    // 添加到图中
    graph.AddOp(rms_norm_op);

    // 多一个Identity算子，因为RMS Norm的输出是单输出
    std::string identity_name = "rms_norm_identity_" + std::to_string(op_index);
    ge::op::Identity identity_op(identity_name);
    identity_op.set_input_x_by_name(rms_norm_op, "y");
    identity_op.update_output_desc_y(desc_out);
    graph.AddOp(identity_op);

    return identity_op;
}

/**
 * @brief 处理RoPE（旋转位置编码）操作的函数
 *
 * 实现RoPE（Rotary Position
 * Embedding）算法，这是Transformer模型中的位置编码方法
 * 该方法通过旋转词向量的特征维度来编码位置信息
 *
 * @param graph 计算图引用
 * @param node 表示RoPE操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的RoPE算子
 */
ge::Operator handle_rope_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index, ggml_backend_cann_context &cann_ctx) {
    // 获取源张量
    struct ggml_tensor *src0 = node->src[0];  // 输入张量
    struct ggml_tensor *src1 = node->src[1];  // 位置索引张量

    auto dst = node;
    GGML_TENSOR_UNARY_OP_LOCALS  // 使用GGML宏获取输入张量的维度

        // 检查算子映射中是否存在输入
        ge::Operator op_x0;
    ge::Operator op_x1;

    // 获取src0（主输入）操作符
    if (gmml_tensor_to_ge_op_map.find(src0) != gmml_tensor_to_ge_op_map.end()) {
        op_x0 = gmml_tensor_to_ge_op_map[src0];
    } else {
        printf("src0 not found in gmml_tensor_to_ge_op_map\n");
        assert(false);
    }

    // 获取src1（位置索引）操作符
    if (gmml_tensor_to_ge_op_map.find(src1) != gmml_tensor_to_ge_op_map.end()) {
        op_x1 = gmml_tensor_to_ge_op_map[src1];
    } else {
        printf("src1 not found in gmml_tensor_to_ge_op_map\n");
        assert(false);
    }

    // 从操作参数中获取RoPE配置参数
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    // const int n_past     = ((int32_t *) dst->op_params)[0]; // 过去的序列长度
    const int n_dims = ((int32_t *)node->op_params)[1];  // 特征维度数量
    const int mode = ((int32_t *)node->op_params)[2];    // RoPE模式
    // const int n_ctx      = ((int32_t *) dst->op_params)[3]; // 上下文长度
    const int n_ctx_orig = ((int32_t *)node->op_params)[4];  // 原始上下文长度

    // 复制浮点参数
    memcpy(&freq_base, (int32_t *)dst->op_params + 5,
           sizeof(float));  // 频率基数
    memcpy(&freq_scale, (int32_t *)dst->op_params + 6,
           sizeof(float));  // 频率缩放
    memcpy(&ext_factor, (int32_t *)dst->op_params + 7,
           sizeof(float));  // 扩展因子
    memcpy(&attn_factor, (int32_t *)dst->op_params + 8,
           sizeof(float));  // 注意力因子
    memcpy(&beta_fast, (int32_t *)dst->op_params + 9,
           sizeof(float));  // Beta快速
    memcpy(&beta_slow, (int32_t *)dst->op_params + 10,
           sizeof(float));  // Beta慢速

    // 确认维度条件
    GGML_ASSERT(n_dims == ne0);
    GGML_ASSERT(n_dims % 2 == 0);  // 特征维度必须是偶数

    // 计算RoPE参数
    const float theta_scale = powf(freq_base, -2.0f / n_dims);  // 频率衰减因子
    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);  // 步长1
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);  // 步长2

    // 计算维度校正因子
    float corr_dims[2];
    const int64_t nr = ggml_nrows(src0);  // 行数
    const int64_t pos_len = src0->ne[2];  // 位置长度
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims);

    const float logf_1_freq_scale = logf(1.0f / freq_scale);

    // 定义重塑后的维度和步长
    int64_t ne_x_reshape[] = {2, src0->ne[0] / 2, src0->ne[1], src0->ne[2],
                              src0->ne[3]};
    size_t nb_x_reshape[5];
    nb_x_reshape[0] = ggml_type_size(src0->type);
    nb_x_reshape[1] =
        nb_x_reshape[0] * (ne_x_reshape[0] / ggml_blck_size(src0->type));
    for (int i = 2; i < 5; i++) {
        nb_x_reshape[i] = nb_x_reshape[i - 1] * ne_x_reshape[i - 1];
    }

    // 第一步：将输入重塑为5D [..., dim//2, 2]
    std::vector<int64_t> shape_reshape_x;
    for (int i = 4; i >= 0; --i) {
        shape_reshape_x.push_back(ne_x_reshape[i]);
    }

    // 创建重塑形状的常量操作
    std::string const_name = "rope_reshape_shape_x_" + std::to_string(op_index);
    ge::op::Const shape_const_op(const_name);

    ge::TensorDesc shape_desc(ge::Shape({(int64_t)shape_reshape_x.size()}),
                              ge::FORMAT_ND, ge::DT_INT64);
    ge::Tensor shape_tensor(shape_desc,
                            reinterpret_cast<uint8_t *>(shape_reshape_x.data()),
                            shape_reshape_x.size() * sizeof(int64_t));
    shape_const_op.set_attr_value(shape_tensor);
    graph.AddOp(shape_const_op);  // 添加到图中

    // 创建重塑操作
    std::string name_reshape_x = "rope_reshape_x_" + std::to_string(op_index);
    ge::op::Reshape reshape_x_op(name_reshape_x);

    reshape_x_op.set_input_x(op_x0);
    reshape_x_op.set_input_shape(shape_const_op);

    reshape_x_op.set_attr_axis(0);
    reshape_x_op.set_attr_num_axes(-1);

    // 设置重塑操作的输出描述
    std::vector<int64_t> out_shape_reshape_x;
    for (int i = 4; i >= 0; --i) {
        out_shape_reshape_x.push_back(ne_x_reshape[i]);
    }
    ge::TensorDesc desc_out_reshape_x(ge::Shape(out_shape_reshape_x),
                                      ge::FORMAT_ND, get_data_type(node->type));
    reshape_x_op.update_output_desc_y(desc_out_reshape_x);
    graph.AddOp(reshape_x_op);

    // 第二步：转置操作，将维度顺序变为 [....,2, dim//2]
    int64_t ne_x_permute[] = {ne_x_reshape[1], ne_x_reshape[0], ne_x_reshape[2],
                              ne_x_reshape[3], ne_x_reshape[4]};
    size_t nb_x_permute[5];
    nb_x_permute[0] = ggml_type_size(src0->type);
    nb_x_permute[1] =
        nb_x_permute[0] * (ne_x_permute[0] / ggml_blck_size(src0->type));
    for (int i = 2; i < 5; i++) {
        nb_x_permute[i] = nb_x_permute[i - 1] * ne_x_permute[i - 1];
    }

    // 定义维度置换顺序
    std::vector<int64_t> order;
    int64_t permute_dim[] = {0, 1, 2, 4, 3};  // 置换维度的顺序
    for (int i = 0; i < 5; i++) {
        order.push_back(permute_dim[i]);
    }

    // 创建转置维度常量
    std::string tranpose_const_x_name =
        "rope_transpose_const_x_" + std::to_string(op_index);
    ge::op::Const tranpose_const_x_op(tranpose_const_x_name);

    ge::TensorDesc transpse_const_x_desc(ge::Shape({(int64_t)order.size()}),
                                         ge::FORMAT_ND, ge::DT_INT64);
    ge::Tensor transpse_const_x_tensor(
        transpse_const_x_desc, reinterpret_cast<uint8_t *>(order.data()),
        order.size() * sizeof(int64_t));
    tranpose_const_x_op.set_attr_value(transpse_const_x_tensor);
    graph.AddOp(tranpose_const_x_op);  // 添加到图中

    // 创建转置操作
    std::string name_permute_x = "rope_permute_x_" + std::to_string(op_index);
    ge::op::Transpose perm_x_op(name_permute_x);

    perm_x_op.set_input_x(reshape_x_op);
    perm_x_op.set_input_perm(tranpose_const_x_op);

    // 设置转置操作的输出描述
    std::vector<int64_t> out_shape_perm_x;
    for (int i = 4; i >= 0; --i) {
        out_shape_perm_x.push_back(ne_x_permute[i]);
    }
    ge::TensorDesc desc_out_perm_x(ge::Shape(out_shape_perm_x), ge::FORMAT_ND,
                                   get_data_type(node->type));
    perm_x_op.update_output_desc_y(desc_out_perm_x);

    graph.AddOp(perm_x_op);

    // 第三步：执行RoPE操作
    // TODO: 在同一个模型中多个RoPE也可能共享一个sin、cos缓存
    std::string curr_suffix = "_" + std::to_string(op_index);
    RopeCache rope_cache(cann_ctx, dst);
    ge::Operator rope_sin_cache =
        rope_cache.GetSinOp(graph, "rope_sin_tensor" + curr_suffix);
    ge::Operator rope_cos_cache =
        rope_cache.GetCosOp(graph, "rope_cos_tensor" + curr_suffix);

    ge::Operator op_x1_squeeze = create_squeeze_op(
        graph, "rope_squeeze_x1_", curr_suffix, op_x1, {0, 1, 2});

    ge::Operator gather_sin_cache =
        create_gather_op(graph, "rope_gather_sin_cache_", curr_suffix,
                         rope_sin_cache, op_x1_squeeze, 1);
    ge::Operator gather_cos_cache =
        create_gather_op(graph, "rope_gather_cos_cache_", curr_suffix,
                         rope_cos_cache, op_x1_squeeze, 1);
    std::string name_rope = "rope_rope" + curr_suffix;
    ge::op::RopeExtCustomV2 rope_op(name_rope.c_str());

    // // 设置RoPE操作的输入
    rope_op.set_input_x(perm_x_op);           // 主输入
    rope_op.set_input_cos(gather_cos_cache);  // 预先计算好的cos输入
    rope_op.set_input_sin(gather_sin_cache);  // 预先计算好的sin输入

    // 设置RoPE操作的输出描述
    ge::TensorDesc desc_out_rope(ge::Shape(out_shape_perm_x), ge::FORMAT_ND,
                                 get_data_type(node->type));
    rope_op.update_output_desc_dst(desc_out_rope);

    // 只设置3个属性
    rope_op.set_attr_ne0(ne0);
    rope_op.set_attr_ne1(ne1);
    rope_op.set_attr_pos_len(src0->ne[2]);

    // std::string name_rope = "rope_rope_" + std::to_string(op_index);
    // ge::op::RopeExtCustom rope_op(name_rope);

    // // 设置RoPE操作的输入
    // rope_op.set_input_x(perm_x_op);  // 转置后的输入
    // rope_op.set_input_pos(op_x1);    // 位置索引

    // // 设置RoPE操作的输出描述
    // ge::TensorDesc desc_out_rope(ge::Shape(out_shape_perm_x), ge::FORMAT_ND,
    //                              get_data_type(node->type));
    // rope_op.update_output_desc_dst(desc_out_rope);

    // // 设置RoPE操作的各种属性参数
    // rope_op.set_attr_ne0(ne0);
    // rope_op.set_attr_ne1(ne1);
    // rope_op.set_attr_s1(s01);
    // rope_op.set_attr_s2(s02);
    // rope_op.set_attr_n_dims(n_dims);
    // rope_op.set_attr_freq_scale(freq_scale);
    // rope_op.set_attr_theta_scale(theta_scale);
    // rope_op.set_attr_ext_factor(ext_factor);
    // rope_op.set_attr_attn_factor(attn_factor);
    // rope_op.set_attr_corr_dims_v_0(corr_dims[0]);
    // rope_op.set_attr_corr_dims_v_1(corr_dims[1]);
    // rope_op.set_attr_logf_1_freq_scale(logf_1_freq_scale);
    // rope_op.set_attr_pos_len(pos_len);

    // 添加RoPE操作到图中
    graph.AddOp(rope_op);

    // 第四步：将维度顺序恢复为 [...., dim//2, 2]
    std::string name_permute_dst =
        "rope_permute_dst_" + std::to_string(op_index);
    ge::op::Transpose permute_dst_op(name_permute_dst);
    permute_dst_op.set_input_x(rope_op);
    permute_dst_op.set_input_perm(tranpose_const_x_op);
    ge::TensorDesc desc_out_permute_dst(ge::Shape(out_shape_reshape_x),
                                        ge::FORMAT_ND,
                                        get_data_type(node->type));
    permute_dst_op.update_output_desc_y(desc_out_permute_dst);
    graph.AddOp(permute_dst_op);

    // 第五步：将5D张量重塑回4D [...., dim]
    std::string name_reshape_dst =
        "rope_reshape_dst_" + std::to_string(op_index);
    ge::op::Reshape op_reshape_dst(name_reshape_dst);

    // 构建最终输出形状
    std::vector<int64_t> out_shape_reshape_dst;
    for (int i = 3; i >= 0; --i) {
        out_shape_reshape_dst.push_back(src0->ne[i]);
    }

    // 创建最终形状常量
    std::string const_dst_name =
        "rope_reshape_shape_dst_" + std::to_string(op_index);
    ge::op::Const shape_const_dst_op(const_dst_name);
    ge::TensorDesc shape_desc_dst(
        ge::Shape({(int64_t)out_shape_reshape_dst.size()}), ge::FORMAT_ND,
        ge::DT_INT64);
    ge::Tensor shape_dst_tensor(
        shape_desc_dst,
        reinterpret_cast<uint8_t *>(out_shape_reshape_dst.data()),
        out_shape_reshape_dst.size() * sizeof(int64_t));
    shape_const_dst_op.set_attr_value(shape_dst_tensor);
    graph.AddOp(shape_const_dst_op);  // 添加到图中

    // 设置最终重塑操作
    op_reshape_dst.set_input_x(permute_dst_op);
    op_reshape_dst.set_input_shape(shape_const_dst_op);
    op_reshape_dst.set_attr_axis(0);
    op_reshape_dst.set_attr_num_axes(-1);

    // 设置最终输出描述
    ge::TensorDesc desc_out_reshape_dst(ge::Shape(out_shape_reshape_dst),
                                        ge::FORMAT_ND,
                                        get_data_type(node->type));
    op_reshape_dst.update_output_desc_y(desc_out_reshape_dst);
    graph.AddOp(op_reshape_dst);

    // 返回最终操作
    return op_reshape_dst;
}

/**
 * @brief 处理ARANGE（等差数列生成）操作的函数
 *
 * 在计算图中创建一个Range操作，生成从start到limit（不包含）步长为delta的等差数列
 * 使用昇腾的Range算子实现，通过三个常量算子提供参数
 *
 * @param graph 计算图引用
 * @param node 表示ARANGE操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的Range算子
 */
ge::Operator handle_arange_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // 从 op_params 中读取三个 float 参数
    float start_val, limit_val, delta_val;
    memcpy(&start_val, (float *)node->op_params + 0, sizeof(float));
    memcpy(&limit_val, (float *)node->op_params + 1, sizeof(float));
    memcpy(&delta_val, (float *)node->op_params + 2, sizeof(float));

    // 创建 start 常量算子
    std::string start_const_name = "arange_start_" + std::to_string(op_index);
    ge::op::Const start_const_op(start_const_name);
    ge::TensorDesc start_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor start_tensor(start_desc, reinterpret_cast<uint8_t *>(&start_val),
                            sizeof(float));
    start_const_op.set_attr_value(start_tensor);

    // 创建 limit 常量算子
    std::string limit_const_name = "arange_limit_" + std::to_string(op_index);
    ge::op::Const limit_const_op(limit_const_name);
    ge::TensorDesc limit_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor limit_tensor(limit_desc, reinterpret_cast<uint8_t *>(&limit_val),
                            sizeof(float));
    limit_const_op.set_attr_value(limit_tensor);

    // 创建 delta 常量算子
    std::string delta_const_name = "arange_delta_" + std::to_string(op_index);
    ge::op::Const delta_const_op(delta_const_name);
    ge::TensorDesc delta_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor delta_tensor(delta_desc, reinterpret_cast<uint8_t *>(&delta_val),
                            sizeof(float));
    delta_const_op.set_attr_value(delta_tensor);

    // 创建 Range 算子
    std::string range_name = "arange_" + std::to_string(op_index);
    ge::op::Range range_op(range_name);

    // 设置三个输入
    range_op.set_input_start(start_const_op);
    range_op.set_input_limit(limit_const_op);
    range_op.set_input_delta(delta_const_op);

    // 设置输出描述
    std::vector<int64_t> output_shape = build_output_shape(node);
    ge::DataType dataType = get_data_type(node->type);
    ge::TensorDesc desc_out(ge::Shape(output_shape), ge::FORMAT_ND, dataType);
    range_op.update_output_desc_y(desc_out);

    // 将算子添加到图中
    graph.AddOp(start_const_op);
    graph.AddOp(limit_const_op);
    graph.AddOp(delta_const_op);
    graph.AddOp(range_op);

    return range_op;
}

void print_op_shape(ge::Operator &op, int idx, const char *name) {
    ge::TensorDesc output_desc = op.GetOutputDesc(idx);
    std::vector<int64_t> dims = output_desc.GetShape().GetDims();
    printf("%s shape: ", name);
    for (auto x : dims) {
        printf("%lld ", x);
    }
    printf("\n");
}

ge::Operator handle_moe_fused_op(
    ge::Graph &graph, ggml_tensor *node,
    std::map<ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    std::string op_suffix = "_" + std::to_string(op_index);
    // 获取输入张量 - 参考 ggml_cann_moe_fused 的参数顺序
    struct ggml_tensor *input = node->src[0];                // 输入张量
    struct ggml_tensor *ids = node->src[1];                  // 专家ID张量
    struct ggml_tensor *topk_weight = node->src[2];          // topk权重张量
    struct ggml_tensor *expert_up_weights = node->src[3];    // 专家上权重
    struct ggml_tensor *expert_down_weights = node->src[4];  // 专家下权重
    struct ggml_tensor *expert_gate_weights = node->src[5];  // 专家门权重
    struct ggml_tensor *row_idx = node->src[6];  // 行索引张量 - 作为参数传入

    // 获取操作参数
    int32_t start_idx = node->op_params[0];
    int32_t end_idx = node->op_params[1];

    // 提取维度信息
    auto batch_size = input->ne[3];
    auto seq_len = input->ne[2];
    auto topk = ids->ne[0];
    auto num_experts = expert_up_weights->ne[2];
    auto hidden_dim = input->ne[0];
    auto k_dim = expert_up_weights->ne[1];
    auto num_rows = batch_size * seq_len;
    auto active_num = num_rows;
    auto num_length = seq_len * topk;

    // 获取输入操作
    ge::Operator op_input, op_ids, op_topk_weight, op_expert_up_weights,
        op_expert_down_weights, op_expert_gate_weights, op_row_idx;

    {
        if (gmml_tensor_to_ge_op_map.find(input) !=
            gmml_tensor_to_ge_op_map.end()) {
            op_input = gmml_tensor_to_ge_op_map[input];
        } else {
            assert(false && "Input tensor not found in map");
        }

        if (gmml_tensor_to_ge_op_map.find(ids) !=
            gmml_tensor_to_ge_op_map.end()) {
            op_ids = gmml_tensor_to_ge_op_map[ids];
        } else {
            assert(false && "IDs tensor not found in map");
        }

        if (gmml_tensor_to_ge_op_map.find(topk_weight) !=
            gmml_tensor_to_ge_op_map.end()) {
            op_topk_weight = gmml_tensor_to_ge_op_map[topk_weight];
        } else {
            assert(false && "TopK weight tensor not found in map");
        }

        if (gmml_tensor_to_ge_op_map.find(expert_up_weights) !=
            gmml_tensor_to_ge_op_map.end()) {
            op_expert_up_weights = gmml_tensor_to_ge_op_map[expert_up_weights];
        } else {
            assert(false && "Expert up weights tensor not found in map");
        }

        if (gmml_tensor_to_ge_op_map.find(expert_down_weights) !=
            gmml_tensor_to_ge_op_map.end()) {
            op_expert_down_weights =
                gmml_tensor_to_ge_op_map[expert_down_weights];
        } else {
            assert(false && "Expert down weights tensor not found in map");
        }

        if (gmml_tensor_to_ge_op_map.find(expert_gate_weights) !=
            gmml_tensor_to_ge_op_map.end()) {
            op_expert_gate_weights =
                gmml_tensor_to_ge_op_map[expert_gate_weights];
        } else {
            assert(false && "Expert gate weights tensor not found in map");
        }

        if (gmml_tensor_to_ge_op_map.find(row_idx) !=
            gmml_tensor_to_ge_op_map.end()) {
            op_row_idx = gmml_tensor_to_ge_op_map[row_idx];
        } else {
            assert(false && "Row index tensor not found in map");
        }
    }

    // 步骤1: 将输入重塑为2D张量 [hidden_dim, seq_len] (注意与GGML ne的倒序关系)
    // 参考 ggml_cann_moe_fused: input_ne[] = {input->ne[0], input->ne[1] *
    // input->ne[2]} = {hidden_dim, seq_len} ge::Operator input_reshape_op =
    // create_reshape_op(
    //     graph, "moe_input_", op_suffix, op_input, {hidden_dim, seq_len});
    ge::Operator input_squeeze_op =
        create_squeeze_op(graph, "moe_input_", op_suffix, op_input, {0, 2});

    // 步骤2: 首先对 op_row_idx 进行 Squeeze 操作，移除前两维
    ge::Operator row_idx_squeeze_op =
        create_squeeze_op(graph, "moe_row_idx_", op_suffix, op_row_idx, {0, 1});

    // 步骤2.1: 创建 row_idx 的转置版本 [topk, seq_len] 用于 MoeInitRouting
    // 参考 ggml_cann_moe_fused 中的实现
    ge::Operator row_idx_permute_op = create_permute_op(
        graph, "moe_row_idx_", op_suffix, row_idx_squeeze_op, {1, 0});

    // 转置后形状为 [topk, seq_len]

    ge::Operator expert_idx_squeeze_op =
        create_squeeze_op(graph, "moe_expert_idx_", op_suffix, op_ids, {0, 1});
    // ge::Operator expert_idx_permute_op = create_permute_op(
    //     graph, "moe_expert_idx_", op_suffix, expert_idx_squeeze_op, {1, 0});

    // 步骤3: MoE 初始化路由 - 使用 MoeInitRouting 算子
    std::string moe_init_name = "moe_init_routing" + op_suffix;
    ge::op::MoeInitRouting moe_init_op(moe_init_name.c_str());

    // 设置输入 - 新接口需要 x, row_idx, expert_idx
    moe_init_op.set_input_x(input_squeeze_op);
    moe_init_op.set_input_row_idx(row_idx_permute_op);
    moe_init_op.set_input_expert_idx(expert_idx_squeeze_op);

    // 设置属性 - 新接口只需要 active_num
    moe_init_op.set_attr_active_num(active_num);

    // 设置输出描述 - 新接口有3个输出
    // 参考 ggml_cann_moe_fused: expand_input_ne[2] = {hidden_dim,
    // std::min(num_rows, active_num) * topk}
    std::vector<int64_t> expanded_x_shape = {
        hidden_dim, std::min(num_rows, active_num) * topk};
    std::vector<int64_t> expanded_row_idx_shape = {num_rows * topk};
    std::vector<int64_t> expanded_expert_idx_shape = {num_rows * topk};

    ge::TensorDesc expanded_x_desc(ge::Shape(expanded_x_shape), ge::FORMAT_ND,
                                   get_data_type(input->type));
    ge::TensorDesc expanded_row_idx_desc(ge::Shape(expanded_row_idx_shape),
                                         ge::FORMAT_ND, ge::DT_INT32);
    ge::TensorDesc expanded_expert_idx_desc(
        ge::Shape(expanded_expert_idx_shape), ge::FORMAT_ND, ge::DT_INT32);

    moe_init_op.update_output_desc_expanded_x(expanded_x_desc);
    moe_init_op.update_output_desc_expanded_row_idx(expanded_row_idx_desc);
    moe_init_op.update_output_desc_expanded_expert_idx(
        expanded_expert_idx_desc);
    graph.AddOp(moe_init_op);

    // 为多输出操作符的每个输出创建单独的Identity算子
    // 输出1: expanded_x
    ge::Operator expanded_x_identity_op = create_identity_op_by_name(
        graph, "moe_expanded_x_", op_suffix, moe_init_op, "expanded_x");
    // expanded_x_identity_op.UpdateOutputDesc((uint32_t)0, expanded_x_desc);
    // return expanded_x_identity_op;
    // 输出2: expanded_row_idx
    ge::Operator expanded_row_idx_identity_op =
        create_identity_op_by_name(graph, "moe_expanded_row_idx_", op_suffix,
                                   moe_init_op, "expanded_row_idx");
    // expanded_row_idx_identity_op.UpdateOutputDesc((uint32_t)0,
    // expanded_row_idx_desc); 输出3: expanded_expert_idx
    ge::Operator expanded_expert_idx_identity_op =
        create_identity_op_by_name(graph, "moe_expanded_expert_idx_", op_suffix,
                                   moe_init_op, "expanded_expert_idx");
    // expanded_expert_idx_identity_op.UpdateOutputDesc((uint32_t)0,
    // expanded_expert_idx_desc); return expanded_expert_idx_identity_op; 步骤4:
    // 计算专家令牌数 - 使用 ReduceSum 算子来模拟 MoeComputeExpertTokens
    // 首先为每个专家创建一个one-hot编码矩阵
    std::string expert_tokens_compute_name =
        "moe_expert_tokens_compute" + op_suffix;

    // 创建expert索引范围 [0, 1, 2, ..., num_experts-1]
    std::string expert_range_const_name = "moe_expert_range_const" + op_suffix;
    ge::op::Const expert_range_const_op(expert_range_const_name);
    std::vector<int32_t> expert_range_data(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        expert_range_data[i] = i;
    }
    std::vector<int64_t> expert_range_shape = {num_experts};
    ge::TensorDesc expert_range_desc(ge::Shape(expert_range_shape),
                                     ge::FORMAT_ND, ge::DT_INT32);
    ge::Tensor expert_range_tensor(
        expert_range_desc,
        reinterpret_cast<uint8_t *>(expert_range_data.data()),
        expert_range_data.size() * sizeof(int32_t));
    expert_range_const_op.set_attr_value(expert_range_tensor);
    graph.AddOp(expert_range_const_op);

    // 使用Equal算子来创建one-hot编码
    std::string equal_name = "moe_equal" + op_suffix;
    ge::op::Equal equal_op(equal_name);

    // 首先需要reshape expanded_expert_idx 到 [num_rows*topk, 1] 用于广播
    ge::Operator expert_idx_reshape_op = create_reshape_op(
        graph, "moe_expert_idx_", op_suffix, expanded_expert_idx_identity_op,
        {num_rows * topk, 1});

    // 现在进行广播比较
    equal_op.set_input_x1(expert_idx_reshape_op);
    equal_op.set_input_x2(expert_range_const_op);

    std::vector<int64_t> equal_output_shape = {num_rows * topk, num_experts};
    ge::TensorDesc equal_desc(ge::Shape(equal_output_shape), ge::FORMAT_ND,
                              ge::DT_BOOL);
    equal_op.update_output_desc_y(equal_desc);
    graph.AddOp(equal_op);

    // 转换bool到int32
    std::string cast_bool_to_int_name = "moe_cast_bool_to_int" + op_suffix;
    ge::op::Cast cast_bool_to_int_op(cast_bool_to_int_name);
    cast_bool_to_int_op.set_input_x(equal_op);
    cast_bool_to_int_op.set_attr_dst_type(ge::DT_INT32);

    ge::TensorDesc cast_bool_to_int_desc(ge::Shape(equal_output_shape),
                                         ge::FORMAT_ND, ge::DT_INT32);
    cast_bool_to_int_op.update_output_desc_y(cast_bool_to_int_desc);
    graph.AddOp(cast_bool_to_int_op);

    // 使用ReduceSum沿着第0维求和得到每个专家的token数量
    std::string reduce_sum_name = "moe_reduce_sum" + op_suffix;
    ge::op::ReduceSum reduce_sum_op(reduce_sum_name);

    // 创建reduction维度常量 [0]
    std::string reduce_axes_const_name = "moe_reduce_axes_const" + op_suffix;
    ge::op::Const reduce_axes_const_op(reduce_axes_const_name);
    std::vector<int32_t> reduce_axes = {0};
    ge::TensorDesc reduce_axes_desc(ge::Shape({1}), ge::FORMAT_ND,
                                    ge::DT_INT32);
    ge::Tensor reduce_axes_tensor(
        reduce_axes_desc, reinterpret_cast<uint8_t *>(reduce_axes.data()),
        reduce_axes.size() * sizeof(int32_t));
    reduce_axes_const_op.set_attr_value(reduce_axes_tensor);
    graph.AddOp(reduce_axes_const_op);

    reduce_sum_op.set_input_x(cast_bool_to_int_op);
    reduce_sum_op.set_input_axes(reduce_axes_const_op);
    reduce_sum_op.set_attr_keep_dims(false);

    std::vector<int64_t> expert_tokens_shape = {num_experts};
    ge::TensorDesc expert_tokens_desc(ge::Shape(expert_tokens_shape),
                                      ge::FORMAT_ND, ge::DT_INT32);
    reduce_sum_op.update_output_desc_y(expert_tokens_desc);
    graph.AddOp(reduce_sum_op);
    // return reduce_sum_op;

    // 步骤5: 专家令牌数转换为 INT64 - 使用 Cast 算子
    std::string expert_tokens_cast_name = "moe_expert_tokens_cast" + op_suffix;
    ge::op::Cast expert_tokens_cast_op(expert_tokens_cast_name);
    expert_tokens_cast_op.set_input_x(reduce_sum_op);
    expert_tokens_cast_op.set_attr_dst_type(ge::DT_INT64);

    ge::TensorDesc expert_tokens_int64_desc(ge::Shape(expert_tokens_shape),
                                            ge::FORMAT_ND, ge::DT_INT64);
    expert_tokens_cast_op.update_output_desc_y(expert_tokens_int64_desc);
    graph.AddOp(expert_tokens_cast_op);
    // debug_operators.push_back(expert_tokens_cast_op);

    // 移除第一个维度，交换后两个维度的位置
    ge::Operator squeezed_up_weight_op = create_squeeze_op(
        graph, "moe_expert_up_weights_", op_suffix, op_expert_up_weights, {0});
    ge::Operator permute_up_weights_op =
        create_permute_op(graph, "moe_expert_up_weights_", op_suffix,
                          squeezed_up_weight_op, {0, 2, 1});
    // 步骤6: 第一个分组矩阵乘法 (up projection) - 使用 GroupedMatmul 算子
    // 生成全0的bias
    // ge::TensorDesc bias_desc(ge::Shape({num_experts, k_dim}), ge::FORMAT_ND,
    // get_data_type(expert_up_weights->type)); ge::Tensor
    // bias_tensor(bias_desc,
    //                        std::vector<uint8_t>(num_experts * k_dim *
    //                        sizeof(float), 0));
    // ge::op::Const grouped_matmul_bias_op("moe_grouped_matmul_bias" +
    // op_suffix); grouped_matmul_bias_op.set_attr_value(bias_tensor);
    // grouped_matmul_bias_op.update_output_desc_y(bias_desc);
    // graph.AddOp(grouped_matmul_bias_op);

    ge::Operator up_matmul_op = create_moe_grouped_matmul_op(
        graph, "moe_up_", op_suffix, get_data_type(expert_up_weights->type),
        expanded_x_identity_op, permute_up_weights_op, {num_experts, k_dim},
        expert_tokens_cast_op);
    // return up_matmul_op;

    // 步骤7: 第二个分组矩阵乘法 (gate projection) - 使用 GroupedMatmul 算子

    ge::Operator squeezed_gate_weight_op =
        create_squeeze_op(graph, "moe_expert_gate_weights_", op_suffix,
                          op_expert_gate_weights, {0});
    ge::Operator permute_gate_weights_op =
        create_permute_op(graph, "moe_expert_gate_weights_", op_suffix,
                          squeezed_gate_weight_op, {0, 2, 1});

    ge::Operator gate_matmul_op = create_moe_grouped_matmul_op(
        graph, "moe_gate_", op_suffix, get_data_type(expert_gate_weights->type),
        expanded_x_identity_op, permute_gate_weights_op, {num_experts, k_dim},
        expert_tokens_cast_op);

    // 步骤8: SiLU 激活 - 使用 Swish 算子
    std::string silu_name = "moe_silu" + op_suffix;
    ge::op::Swish silu_op(silu_name);
    silu_op.set_input_x(gate_matmul_op);
    silu_op.set_attr_scale(1.0f);

    // ge::TensorDesc silu_desc(ge::Shape(gate_output_shape), ge::FORMAT_ND,
    //                          get_data_type(expert_gate_weights->type));
    // silu_op.update_output_desc_y(silu_desc);
    graph.AddOp(silu_op);

    // 步骤9: 元素乘法 (up * gate_silu) - 使用 Mul 算子
    std::string mul_name = "moe_mul" + op_suffix;
    ge::op::Mul mul_op(mul_name);
    mul_op.set_input_x1(up_matmul_op);
    mul_op.set_input_x2(silu_op);

    // ge::TensorDesc mul_desc(ge::Shape(up_output_shape), ge::FORMAT_ND,
    // get_data_type(expert_up_weights->type));
    // mul_op.update_output_desc_y(mul_desc);
    graph.AddOp(mul_op);

    // 步骤10: 第三个分组矩阵乘法 (down projection) - 使用 GroupedMatmul 算子

    ge::Operator squeezed_down_weight_op =
        create_squeeze_op(graph, "moe_expert_down_weights_", op_suffix,
                          op_expert_down_weights, {0});
    ge::Operator permute_down_weights_op =
        create_permute_op(graph, "moe_expert_down_weights_", op_suffix,
                          squeezed_down_weight_op, {0, 2, 1});

    // ge::TensorDesc down_bias_desc(ge::Shape({num_experts, hidden_dim}),
    // ge::FORMAT_ND, get_data_type(expert_up_weights->type)); ge::Tensor
    // down_bias_tensor(down_bias_desc,
    //                        std::vector<uint8_t>(num_experts * hidden_dim *
    //                        sizeof(float), 0));
    // ge::op::Const down_grouped_matmul_bias_op("moe_down_grouped_matmul_bias"
    // + op_suffix);
    // down_grouped_matmul_bias_op.set_attr_value(down_bias_tensor);
    // down_grouped_matmul_bias_op.update_output_desc_y(down_bias_desc);
    // graph.AddOp(down_grouped_matmul_bias_op);

    ge::Operator down_matmul_op = create_moe_grouped_matmul_op(
        graph, "moe_down_", op_suffix, get_data_type(expert_down_weights->type),
        mul_op, permute_down_weights_op, {num_experts, hidden_dim},
        expert_tokens_cast_op);

    // 步骤11: 最终化路由 - 使用 MoeFinalizeRouting 算子

    ge::Operator op_topk_weight_squeezed = create_squeeze_op(
        graph, "moe_topk_weight_", op_suffix, op_topk_weight, {0, 3});

    std::string finalize_name = "moe_finalize_routing" + op_suffix;
    ge::op::MoeFinalizeRoutingV2 finalize_op(finalize_name.c_str());

    if (node->type == GGML_TYPE_F16) {
        finalize_op.set_input_expanded_x(down_matmul_op);
    } else {
        std::string cast_f32_name = "moe_cast_f32" + op_suffix;
        ge::op::Cast cast_f32_op(cast_f32_name);
        cast_f32_op.set_input_x(down_matmul_op);
        cast_f32_op.set_attr_dst_type(ge::DT_FLOAT);
        graph.AddOp(cast_f32_op);
        finalize_op.set_input_expanded_x(cast_f32_op);
    }
    finalize_op.set_input_expanded_row_idx(
        moe_init_op, 1);  // 使用第2个输出 expanded_row_idx
    // finalize_op.set_input_x1(op_topk_weight);
    // finalize_op.set_input_bias(op_topk_weight); // 使用 topk_weight 作为 bias
    finalize_op.set_input_scales(op_topk_weight_squeezed);
    finalize_op.set_input_expert_idx(expert_idx_squeeze_op,
                                     0);  // 使用第3个输出 expanded_expert_idx

    // 设置最终输出描述
    // 参考 ggml_cann_moe_fused: f_dst_ne[2] = {hidden_dim, seq_len}
    std::vector<int64_t> final_shape = {hidden_dim, seq_len};
    ge::TensorDesc final_desc(ge::Shape(final_shape), ge::FORMAT_ND,
                              get_data_type(node->type));
    finalize_op.update_output_desc_y(final_desc);
    graph.AddOp(finalize_op);

    return finalize_op;
}

/**
 * @brief 处理StridedSliceV2操作的函数
 *
 * 实现张量的步长切片操作，从输入张量中提取指定步长的切片
 * 支持多维张量的灵活切片，包括起始位置、结束位置、步长和轴向等参数
 *
 * @param graph 计算图引用
 * @param node 表示StridedSliceV2操作的张量节点
 * @param ggml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的StridedSliceV2算子
 */

ge::Operator handle_stridedslicev2_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &ggml_tensor_to_ge_op_map,
    int op_index) {
    // 获取输入张量
    struct ggml_tensor *src_x = node->src[0];  // 主输入张量

    // 获取输入张量对应的操作符
    assert(ggml_tensor_to_ge_op_map.count(src_x));
    ge::Operator op_x = ggml_tensor_to_ge_op_map[src_x];

    // 通过{ fr, to, axis
    // }，构造出begin，end，strides，然后作为const_op添加到图中
    int64_t params[3] = {};
    // use memcpy to copy params to begin, end, strides
    memcpy(params, node->op_params, sizeof(int64_t) * 3);
    int64_t fr = params[0];
    int64_t to = params[1];
    int64_t axis = params[2];

    // 初始化begin, end, strides数组
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

    // 创建begin常量操作
    std::string begin_const_name =
        "strided_slice_begin_" + std::to_string(op_index);
    auto begin_const_op =
        create_const_1d_op(graph, begin_const_name, begin_vec, ge::DT_INT64);

    // 创建end常量操作
    std::string end_const_name =
        "strided_slice_end_" + std::to_string(op_index);
    auto end_const_op =
        create_const_1d_op(graph, end_const_name, end_vec, ge::DT_INT64);

    // 创建strides常量操作
    std::string strides_const_name =
        "strided_slice_strides_" + std::to_string(op_index);
    auto strides_const_op = create_const_1d_op(graph, strides_const_name,
                                               strides_vec, ge::DT_INT64);

    // 创建axes常量节点，内容为{0,1,2,3}，类型为int64
    std::vector<int64_t> axes_vec = {0, 1, 2, 3};
    std::string axes_const_name =
        "strided_slice_axes_" + std::to_string(op_index);
    auto axes_const_op =
        create_const_1d_op(graph, axes_const_name, axes_vec, ge::DT_INT64);

    // 创建StridedSliceV2操作
    ge::op::StridedSliceV2 strided_slice_op("strided_slice_v2_" +
                                            std::to_string(op_index));

    strided_slice_op.set_input_x(op_x);
    strided_slice_op.set_input_begin(begin_const_op);
    strided_slice_op.set_input_end(end_const_op);
    strided_slice_op.set_input_strides(strides_const_op);
    strided_slice_op.set_input_axes(axes_const_op);

    // 设置输出形状和类型
    std::vector<int64_t> out_shape = build_output_shape(node);
    ge::TensorDesc desc_out(ge::Shape(out_shape), ge::FORMAT_ND,
                            get_data_type(node->type));
    strided_slice_op.update_output_desc_y(desc_out);
    // 添加到图中
    graph.AddOp(strided_slice_op);
    // 返回创建的StridedSliceV2操作符
    return strided_slice_op;
}

/**
 * @brief 处理StridedSliceAssignV2操作的函数
 *
 * 实现张量的步长切片操作，从输入张量中提取指定步长的切片
 * 支持多维张量的灵活切片，包括起始位置、结束位置、步长和轴向等参数
 *
 * @param graph 计算图引用
 * @param node 表示StridedSliceAssignV2操作的张量节点
 * @param ggml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的StridedSliceAssignV2算子
 */

ge::Operator handle_set_slice_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &ggml_tensor_to_ge_op_map,
    int op_index) {
    // 获取输入张量
    struct ggml_tensor *src_var = node->src[0];     // 待赋值的张量
    struct ggml_tensor *src_index = node->src[1];   // 索引张量
    struct ggml_tensor *src_update = node->src[2];  // 赋值张量

    assert(src_var && "SET_SLICE: missing input tensor");
    assert(src_index && "SET_SLICE: missing index tensor");
    assert(src_update && "SET_SLICE: missing update tensor");

    // 获取输入张量对应的操作符
    ge::Operator op_var, op_indices, op_updates;

    // 处理var - 获取现有算子
    if (ggml_tensor_to_ge_op_map.find(src_var) !=
        ggml_tensor_to_ge_op_map.end()) {
        op_var = ggml_tensor_to_ge_op_map[src_var];
    } else {
        assert(false && "SET_SLICE: var tensor not found in map");
    }

    // 处理indices - 获取现有算子
    if (ggml_tensor_to_ge_op_map.find(src_index) !=
        ggml_tensor_to_ge_op_map.end()) {
        op_indices = ggml_tensor_to_ge_op_map[src_index];
    } else {
        assert(false && "SET_SLICE: indices tensor not found in map");
    }

    ge::DataType indices_dtype = get_data_type(src_index->type);
    if (indices_dtype != ge::DT_INT32 && indices_dtype != ge::DT_INT64) {
        printf(
            "SET_SLICE: indices tensor type is not INT32 or INT64, type: %d\n",
            indices_dtype);
    }

    // 处理updates - 获取现有算子
    if (ggml_tensor_to_ge_op_map.find(src_update) !=
        ggml_tensor_to_ge_op_map.end()) {
        op_updates = ggml_tensor_to_ge_op_map[src_update];
    } else {
        assert(false && "SET_SLICE: updates tensor not found in map");
    }

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
    op_var = create_reshape_op(graph, op_var, op_var_shape,
                               "set_slice_var_" + std::to_string(op_index),
                               get_data_type(src_var->type));
    op_indices =
        create_reshape_op(graph, op_indices, op_indices_shape,
                          "set_slice_indices_" + std::to_string(op_index),
                          get_data_type(src_index->type));
    op_updates =
        create_reshape_op(graph, op_updates, op_updates_shape,
                          "set_slice_updates_" + std::to_string(op_index),
                          get_data_type(src_update->type));
    // 创建ScatterUpdate算子
    std::string scatter_update_name =
        "scatter_update_" + std::to_string(op_index);
    ge::op::ScatterUpdate scatter_update_op(scatter_update_name.c_str());

    ge::DataType var_dtype = get_data_type(src_var->type);
    ge::DataType updates_dtype = get_data_type(src_update->type);

    if (var_dtype != updates_dtype) {
        printf(
            "SET_SLICE: var_dtype != updates_dtype, var_dtype: %d, "
            "updates_dtype: %d\n",
            var_dtype, updates_dtype);
    }

    // 设置输入
    scatter_update_op.set_input_var(op_var);          // 要更新的变量
    scatter_update_op.set_input_indices(op_indices);  // 索引位置
    scatter_update_op.set_input_updates(op_updates);  // 更新值

    // 设置属性
    scatter_update_op.set_attr_use_locking(false);  // 默认不使用锁定

    // 设置输出形状和数据类型（与输入var保持一致）
    std::vector<int64_t> output_shape = build_output_shape(node);
    ge::DataType dataType = get_data_type(node->type);
    ge::TensorDesc desc_out(ge::Shape(output_shape), ge::FORMAT_ND, dataType);
    scatter_update_op.update_output_desc_var(desc_out);

    // 将算子添加到图中
    graph.AddOp(scatter_update_op);

    return scatter_update_op;
}

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
                               ge::DataType data_type) {
    // 创建形状常量操作符
    std::string const_name = op_name + "_shape_const";
    ge::op::Const shape_const_op(const_name.c_str());

    // 设置形状常量的值
    ge::TensorDesc shape_desc(ge::Shape({(int64_t)target_shape.size()}),
                              ge::FORMAT_ND, ge::DT_INT64);

    // 创建可修改的目标形状副本
    std::vector<int64_t> mutable_shape = target_shape;
    ge::Tensor shape_tensor(shape_desc,
                            reinterpret_cast<uint8_t *>(mutable_shape.data()),
                            mutable_shape.size() * sizeof(int64_t));
    shape_const_op.set_attr_value(shape_tensor);
    shape_const_op.UpdateOutputDesc("y", shape_desc);
    // 将常量操作添加到图中
    graph.AddOp(shape_const_op);

    // 创建Reshape操作符
    ge::op::Reshape reshape_op(op_name + "_reshape");

    // 设置Reshape的输入和形状
    // print_op_shape(input_op);
    // print_op_shape(shape_const_op);
    reshape_op.set_input_x(input_op);
    reshape_op.set_input_shape(shape_const_op);

    // 设置重塑属性
    reshape_op.set_attr_axis(0);       // 从第一个维度开始重塑
    reshape_op.set_attr_num_axes(-1);  // 重塑所有维度

    // 设置输出形状和类型
    ge::TensorDesc desc_out(ge::Shape(target_shape), ge::FORMAT_ND, data_type);
    reshape_op.update_output_desc_y(desc_out);

    // 添加到图中
    graph.AddOp(reshape_op);

    return reshape_op;
}
//  * @brief 处理 Flash Attention Prompt 操作的函数
//  *
//  * 在计算图中创建一个 Flash Attention Prompt 操作，基于 GE 算子实现
//  *
//  * @param graph 计算图引用
//  * @param node 表示 Flash Attention Prompt 操作的张量节点
//  * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
//  * @param op_index 用于生成唯一算子名称的索引
//  * @return 创建的 Flash Attention Prompt 算子
//  */
ge::Operator handle_flash_attn_prompt_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // return op_sequence_length_kv;
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
    // GGML_ASSERT(sequence_length_kv_tensor->type == GGML_TYPE_I32);  //
    // 验证sequence_length_kv张量类型
    GGML_ASSERT(node->type == GGML_TYPE_F16);

    // 检查输入是否已经在映射中
    ge::Operator op_query;
    ge::Operator op_key;
    ge::Operator op_value;
    ge::Operator op_attn_mask;
    ge::Operator op_length_q_tensor;
    ge::Operator op_length_kv_tensor;

    if (gmml_tensor_to_ge_op_map.find(query) !=
        gmml_tensor_to_ge_op_map.end()) {
        op_query = gmml_tensor_to_ge_op_map[query];
    } else {
        assert(false);
    }

    if (gmml_tensor_to_ge_op_map.find(key) != gmml_tensor_to_ge_op_map.end()) {
        op_key = gmml_tensor_to_ge_op_map[key];
    } else {
        assert(false);
    }

    if (gmml_tensor_to_ge_op_map.find(value) !=
        gmml_tensor_to_ge_op_map.end()) {
        op_value = gmml_tensor_to_ge_op_map[value];
    } else {
        assert(false);
    }

    if (gmml_tensor_to_ge_op_map.find(attn_mask) !=
        gmml_tensor_to_ge_op_map.end()) {
        op_attn_mask = gmml_tensor_to_ge_op_map[attn_mask];
    } else {
        assert(false);
    }

    if (gmml_tensor_to_ge_op_map.find(length_q_tensor) !=
        gmml_tensor_to_ge_op_map.end()) {
        op_length_q_tensor = gmml_tensor_to_ge_op_map[length_q_tensor];
    } else {
        assert(false);
    }

    if (gmml_tensor_to_ge_op_map.find(length_kv_tensor) !=
        gmml_tensor_to_ge_op_map.end()) {
        op_length_kv_tensor = gmml_tensor_to_ge_op_map[length_kv_tensor];
    } else {
        assert(false);
    }

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
    int32_t batch_size = params->batch_size;
    int32_t num_heads = params->num_heads;
    int32_t head_dim_kq = params->head_dim_kq;
    int32_t head_dim_v = params->head_dim_v;
    int32_t key_num_heads = params->key_num_heads;
    // 这个参数仅用于判断开启精度模式
    int32_t sequence_length_q = params->sequence_lenth_q;
    float scale_value = params->scaleValue;

    // 设置属性值，匹配 PromptFlashAttention 算子的规范
    int64_t num_key_value_heads = num_heads;
    std::string input_layout = "BSND";  // 默认输入布局
    int64_t pre_tokens = 2147483647;  // 匹配默认值 214748647 -> 2147483647
    int64_t next_tokens = 0;
    int64_t sparse_mode = 1;  // 拦截，非量化情况不考虑
    int64_t inner_precise =
        sequence_length_q > 1 ? 2 : 0;  // 高精度模式，开启行无效修正

    // // 创建 actual_seq_lengths 常量张量
    // std::string actual_seq_lengths_q_name =
    //     "flash_attn_actual_seq_lengths_q_" + std::to_string(op_index);
    // ge::op::Const actual_seq_lengths_q_op(actual_seq_lengths_q_name);

    // std::vector<int64_t> actual_seq_q_values = {sequence_length_q};
    // ge::TensorDesc actual_seq_q_desc(ge::Shape({1}), ge::FORMAT_ND,
    //                                  ge::DT_INT64);
    // ge::Tensor actual_seq_q_tensor(
    //     actual_seq_q_desc,
    //     reinterpret_cast<uint8_t *>(actual_seq_q_values.data()),
    //     actual_seq_q_values.size() * sizeof(int64_t));
    // actual_seq_lengths_q_op.set_attr_value(actual_seq_q_tensor);
    // graph.AddOp(actual_seq_lengths_q_op);

    // 创建 PromptFlashAttention 算子
    std::string flash_attn_name = "flash_attn_" + std::to_string(op_index);
    ge::op::FusedInferAttentionScore flash_attn_op(flash_attn_name);

    // 设置输入 - 必选输入
    flash_attn_op.set_input_query(op_query);

    // 设置key输入
    flash_attn_op.create_dynamic_input_byindex_key(1, 1);
    flash_attn_op.set_dynamic_input_key(0, op_key);

    // 设置value输入
    flash_attn_op.create_dynamic_input_byindex_value(1, 2);
    flash_attn_op.set_dynamic_input_value(0, op_value);

    // 设置可选输入
    flash_attn_op.set_input_atten_mask(op_attn_mask);
    flash_attn_op.set_input_actual_seq_lengths(op_length_q_tensor);
    flash_attn_op.set_input_actual_seq_lengths_kv(op_length_kv_tensor);

    // 设置属性 - 按照 PromptFlashAttention 的接口规范
    flash_attn_op.set_attr_num_heads(num_heads);        // 必选属性
    flash_attn_op.set_attr_scale(scale_value);          // 默认 1.0
    flash_attn_op.set_attr_pre_tokens(pre_tokens);      // 默认 214748647
    flash_attn_op.set_attr_next_tokens(next_tokens);    // 默认 0
    flash_attn_op.set_attr_input_layout(input_layout);  // 默认 "BSH"
    flash_attn_op.set_attr_num_key_value_heads(num_key_value_heads);  // 默认 0
    flash_attn_op.set_attr_sparse_mode(sparse_mode);      // 默认 0
    flash_attn_op.set_attr_inner_precise(inner_precise);  // 默认 1
    flash_attn_op.set_attr_softmax_lse_flag(false);

    // 计算输出形状
    std::vector<int64_t> output_shape = build_output_shape(node);
    ge::DataType dataType = get_data_type(node->type);

    // 设置输出描述
    ge::TensorDesc desc_out(ge::Shape(output_shape), ge::FORMAT_ND, dataType);
    flash_attn_op.update_output_desc_attention_out(desc_out);

    // 添加算子到图中
    graph.AddOp(flash_attn_op);

    // 多一个Identity算子，仅保留flash attention的out部分
    std::string identity_name =
        "flashattn_identity_" + std::to_string(op_index);
    ge::op::Identity identity_op(identity_name);
    identity_op.set_input_x_by_name(flash_attn_op, "attention_out");
    identity_op.update_output_desc_y(desc_out);
    graph.AddOp(identity_op);

    return identity_op;
}

/**
 * @brief 处理Pad操作的函数
 *
 * 实现张量的填充操作，支持多种填充模式
 * 支持constant、reflect、edge模式，并可指定填充值
 *
 * @param graph 计算图引用
 * @param node 表示Pad操作的张量节点
 * @param gmml_tensor_to_ge_op_map 张量到对应算子的映射
 * @param op_index 用于生成唯一算子名称的索引
 * @return 创建的PadV3算子
 */
ge::Operator handle_pad_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    // 获取输入张量
    struct ggml_tensor *src = node->src[0];  // 待填充的张量
    assert(src && "PAD: missing input tensor");

    // 获取输入张量对应的操作符
    ge::Operator op_input;
    if (gmml_tensor_to_ge_op_map.find(src) != gmml_tensor_to_ge_op_map.end()) {
        op_input = gmml_tensor_to_ge_op_map[src];
    } else {
        assert(false && "PAD: input tensor not found in map");
    }

    // 创建PadV3算子
    std::string pad_name = "pad_v3_" + std::to_string(op_index);
    ge::op::PadV3 pad_op(pad_name);

    // 设置输入
    pad_op.set_input_x(op_input);
    // 设置padding
    std::vector<int64_t> paddings = {
        0, node->ne[3] - src->ne[3], 0, node->ne[2] - src->ne[2],
        0, node->ne[1] - src->ne[1], 0, node->ne[0] - src->ne[0]};
    // 反转
    //  std::reverse(paddings.begin(), paddings.end());
    //  for(auto pad:paddings){
    //  printf("%lld ", pad);
    // }
    // printf("\n");
    auto paddings_const_op =
        create_const_1d_op(graph, "pad_paddings" + std::to_string(op_index),
                           paddings, ge::DT_INT64);
    // graph.AddOp(paddings_const_op);
    pad_op.set_input_paddings(paddings_const_op);

    float pad_const_value = {0.0f};  // 默认填充值为0.0f
    std::string const_value_name = "pad_const_value" + std::to_string(op_index);
    ge::op::Const const_value_op(const_value_name);
    ge::TensorDesc const_value_desc(ge::Shape({(int64_t)1}), ge::FORMAT_ND,
                                    ge::DT_FLOAT);
    ge::Tensor tensor(const_value_desc);
    tensor.SetData(reinterpret_cast<uint8_t *>(&pad_const_value),
                   sizeof(float));
    const_value_op.set_attr_value(tensor);
    graph.AddOp(const_value_op);
    pad_op.set_input_constant_values(const_value_op);

    // 设置属性
    pad_op.set_attr_mode("constant");           // 默认使用constant模式
    pad_op.set_attr_paddings_contiguous(true);  // 默认使用连续的paddings格式

    // 设置输出描述
    auto output_shape = build_output_shape(node);
    ge::DataType data_type = get_data_type(node->type);
    ge::TensorDesc desc_out(ge::Shape(output_shape), ge::FORMAT_ND, data_type);
    pad_op.update_output_desc_y(desc_out);

    // 将算子添加到图中
    graph.AddOp(pad_op);

    return pad_op;
}

ge::Operator handle_get_rows_op(
    ge::Graph &graph, struct ggml_tensor *node,
    std::map<struct ggml_tensor *, ge::Operator> &gmml_tensor_to_ge_op_map,
    int op_index) {
    std::string op_suffix = "_" + std::to_string(op_index);

    struct ggml_tensor *src = node->src[0];
    struct ggml_tensor *rows = node->src[1];

    ge::Operator op_src, op_rows;
    if (gmml_tensor_to_ge_op_map.find(src) != gmml_tensor_to_ge_op_map.end()) {
        op_src = gmml_tensor_to_ge_op_map[src];
    } else {
        assert(false && "get_rows: input tensor not found in map");
    }
    if (gmml_tensor_to_ge_op_map.find(rows) != gmml_tensor_to_ge_op_map.end()) {
        op_rows = gmml_tensor_to_ge_op_map[rows];
    } else {
        assert(false && "get_rows: input tensor not found in map");
    }

    int batch_dims = rows->ne[1];
    int row_num = rows->ne[0];

    ge::Operator op_squeezed_src =
        create_squeeze_op(graph, "get_rows_src_", op_suffix, op_src, {0});

    // ge::Operator op_reshape_rows = create_reshape_op(
    //     graph, "get_rows_rows_", op_suffix, op_rows,
    //     {row_num * batch_dims});
    ge::Operator op_squeezed_rows =
        create_squeeze_op(graph, "get_rows_rows_", op_suffix, op_rows, {0, 1});

    // ge::Operator op_cast_rows = create_cast_op(
    //     graph, "get_rows_rows_", op_suffix, op_squeezed_rows, ge::DT_INT64);
    ge::Operator op_result = create_gather_op(
        graph, "get_rows_", op_suffix, op_squeezed_src, op_squeezed_rows, 1, 1);

    ge::Operator op_unsq_result =
        create_unsqueeze_op(graph, "get_rows_", op_suffix, op_result, {0});

    ge::Operator op_cast_result =
        create_cast_op(graph, "get_rows_res_", op_suffix, op_unsq_result,
                       get_data_type(node->type));
    ge::TensorDesc result_desc(
        ge::Shape({1, rows->ne[1], rows->ne[0], src->ne[0]}), ge::FORMAT_ND,
        get_data_type(node->type));
    op_cast_result.UpdateOutputDesc((uint32_t)0, result_desc);
    return op_cast_result;
}
