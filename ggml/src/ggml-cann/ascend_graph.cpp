/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ggml-cann/ascend_graph.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

#include "ggml-cann/ascend_graph_ops.h"

using namespace std;

void apply_tensor_desc(gert::Tensor& tensor,
                       const ge::TensorDesc& tensor_desc) {
    tensor.MutableOriginShape().SetDimNum(tensor_desc.GetShape().GetDimNum());
    for (size_t i = 0; i < tensor_desc.GetShape().GetDimNum(); i++) {
        tensor.MutableOriginShape().SetDim(i, tensor_desc.GetShape().GetDim(i));
    }
    tensor.MutableStorageShape().SetDimNum(tensor_desc.GetShape().GetDimNum());

    for (size_t i = 0; i < tensor_desc.GetShape().GetDimNum(); i++) {
        tensor.MutableStorageShape().SetDim(i,
                                            tensor_desc.GetShape().GetDim(i));
    }
    tensor.SetOriginFormat(tensor_desc.GetFormat());
    tensor.SetStorageFormat(tensor_desc.GetFormat());
    tensor.SetPlacement(gert::TensorPlacement::kOnDeviceHbm);
    tensor.SetDataType(tensor_desc.GetDataType());
}

/**
 * @brief 为GGML张量创建GE张量描述符
 *
 * @param node GGML张量指针
 * @return 创建的GE张量描述符
 */
ge::TensorDesc create_tensor_desc_for_node(ggml_tensor* node) {
    // 设置张量描述符
    std::vector<int64_t> shape;
    for (int d = GGML_MAX_DIMS - 1; d >= 0; d--) {
        if (node->ne[d] > 0) {
            shape.push_back(node->ne[d]);
        }
    }

    // 创建张量描述符
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND,
                        get_data_type(node->type));
    desc.SetPlacement(ge::Placement::kPlacementDevice);

    return desc;
}

/**
 * @brief 创建gert::Tensor并绑定指定的数据指针
 *
 * @param desc GE张量描述符
 * @param data_ptr 数据指针
 * @param data_size 数据大小（字节）
 * @return 创建并绑定数据的gert::Tensor
 */
gert::Tensor create_bound_tensor_with_ptr(const ge::TensorDesc& tensor_desc,
                                          void* data_ptr, size_t data_size) {
    gert::Tensor tensor;

    tensor.MutableOriginShape().SetDimNum(tensor_desc.GetShape().GetDimNum());

    for (size_t i = 0; i < tensor_desc.GetShape().GetDimNum(); i++) {
        tensor.MutableOriginShape().SetDim(i, tensor_desc.GetShape().GetDim(i));
    }

    tensor.MutableStorageShape().SetDimNum(tensor_desc.GetShape().GetDimNum());

    for (size_t i = 0; i < tensor_desc.GetShape().GetDimNum(); i++) {
        tensor.MutableStorageShape().SetDim(i,
                                            tensor_desc.GetShape().GetDim(i));
    }

    tensor.SetOriginFormat(tensor_desc.GetFormat());
    tensor.SetStorageFormat(tensor_desc.GetFormat());
    tensor.SetPlacement(gert::TensorPlacement::kOnDeviceHbm);
    tensor.SetDataType(tensor_desc.GetDataType());

    tensor.SetData(gert::TensorData(reinterpret_cast<uint8_t*>(data_ptr),
                                    nullptr, data_size,
                                    gert::TensorPlacement::kOnDeviceHbm));

    return tensor;
}

/**
 * @brief 创建并配置图输入张量
 *
 * 为给定的源张量创建对应的Ascend Tensor，并添加到图输入列表中
 *
 * @param src_tensor 源GGML张量
 * @param ggml_tensor_to_ge_op_map 张量到算子的映射
 * @param graph_inputs 图输入算子列表
 * @param input_init 输入张量初始化列表
 */
void create_graph_input_tensor(
    ggml_tensor* src_tensor,
    std::map<ggml_tensor*, Operator>& ggml_tensor_to_ge_op_map,
    std::vector<Operator>& graph_inputs,
    std::vector<gert::Tensor>& input_init) {
    // 将对应的Data操作符加入graph_inputs
    graph_inputs.push_back(ggml_tensor_to_ge_op_map[src_tensor]);

    // 获取输出描述符，使用输出端口0
    ge::TensorDesc tensor_desc =
        ggml_tensor_to_ge_op_map[src_tensor].GetOutputDesc(0);
    tensor_desc.SetPlacement(ge::Placement::kPlacementDevice);

    // 使用辅助函数创建并绑定张量
    gert::Tensor input_tensor = create_bound_tensor_with_ptr(
        tensor_desc, src_tensor->data, ggml_nbytes(src_tensor));
    input_init.push_back(std::move(input_tensor));
}

/**
 * @brief 创建gert::Tensor并绑定指定的主机侧数据指针
 *
 * @param desc GE张量描述符
 * @param data_ptr 主机侧数据指针
 * @param data_size 数据大小（字节）
 * @return 创建并绑定主机侧数据的gert::Tensor
 */
gert::Tensor create_bound_tensor_with_host_ptr(
    const ge::TensorDesc& tensor_desc, void* data_ptr, size_t data_size) {
    gert::Tensor tensor;

    tensor.MutableOriginShape().SetDimNum(tensor_desc.GetShape().GetDimNum());

    for (size_t i = 0; i < tensor_desc.GetShape().GetDimNum(); i++) {
        tensor.MutableOriginShape().SetDim(i, tensor_desc.GetShape().GetDim(i));
    }

    tensor.MutableStorageShape().SetDimNum(tensor_desc.GetShape().GetDimNum());

    for (size_t i = 0; i < tensor_desc.GetShape().GetDimNum(); i++) {
        tensor.MutableStorageShape().SetDim(i,
                                            tensor_desc.GetShape().GetDim(i));
    }

    tensor.SetOriginFormat(tensor_desc.GetFormat());
    tensor.SetStorageFormat(tensor_desc.GetFormat());
    tensor.SetPlacement(gert::TensorPlacement::kOnHost);  // 设置为主机侧
    tensor.SetDataType(tensor_desc.GetDataType());

    tensor.SetData(gert::TensorData(
        reinterpret_cast<uint8_t*>(data_ptr), nullptr, data_size,
        gert::TensorPlacement::kOnHost));  // 设置为主机侧

    return tensor;
}

/**
 * @brief 查找计算图中最后一个操作节点
 *
 * @param cgraph GGML计算图
 * @return 最后一个操作节点，如果没有找到返回nullptr
 */
ggml_tensor* find_last_op_node(ggml_cgraph* cgraph) {
    ggml_tensor* last_op_node = nullptr;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];
        if (!ggml_is_empty(node) && node->op != GGML_OP_NONE) {
            last_op_node = node;
        }
    }
    return last_op_node;
}

/**
 * @brief 创建输出张量列表
 *
 * @param graph_outputs 图输出操作符列表
 * @param last_op_node 最后一个操作节点
 * @param output_init 输出张量初始化列表
 */
void create_output_tensors(const std::vector<Operator>& graph_outputs,
                           ggml_tensor* last_op_node,
                           std::vector<gert::Tensor>& output_init) {
    if (!graph_outputs.empty() && last_op_node != nullptr) {
        for (auto& output_op : graph_outputs) {
            // 获取输出描述符
            ge::TensorDesc tensor_desc = output_op.GetOutputDesc(0);
            tensor_desc.SetPlacement(ge::Placement::kPlacementDevice);

            // 使用辅助函数创建输出张量并绑定数据
            gert::Tensor output_tensor = create_bound_tensor_with_ptr(
                tensor_desc, last_op_node->data, ggml_nbytes(last_op_node));

            // 添加到output_init
            output_init.push_back(std::move(output_tensor));
        }
    }
}

/**
 * @brief 处理输入张量（支持构建模式和复用模式）
 *
 * @param tensor_array 张量数组指针
 * @param count 数组元素数量
 * @param input_init 输入张量初始化列表
 * @param build_mode
 * 是否为构建模式（需要创建算子），false为复用模式（只绑定数据）
 * @param name_prefix 算子名称前缀（仅构建模式使用）
 * @param index_offset 索引偏移量（仅构建模式使用）
 * @param graph Ascend图引用（仅构建模式使用）
 * @param ggml_tensor_to_ge_op_map 张量到算子的映射（仅构建模式使用）
 * @param graph_inputs 图输入算子列表（仅构建模式使用）
 */
void process_input_tensors(
    ggml_tensor** tensor_array, int count,
    std::vector<gert::Tensor>& input_init, bool build_mode = true,
    const std::string& name_prefix = "", int index_offset = 0,
    Graph* graph = nullptr,
    std::map<ggml_tensor*, Operator>* ggml_tensor_to_ge_op_map = nullptr,
    std::vector<Operator>* graph_inputs = nullptr) {
    auto create_data = [&](ggml_tensor* node) {
        if (ggml_tensor_to_ge_op_map->find(node) !=
            ggml_tensor_to_ge_op_map->end()) {
            return;
        }

        if (build_mode) {
            // 使用辅助函数创建张量描述符
            ge::TensorDesc desc = create_tensor_desc_for_node(node);

            // 为此输入创建数据算子
            std::string name = name_prefix + std::string(node->name);
            op::Data data_op(name.c_str());
            data_op.update_output_desc_y(desc);

            // 添加到图和映射中
            graph->AddOp(data_op);
            (*ggml_tensor_to_ge_op_map)[node] = data_op;
            create_graph_input_tensor(node, *ggml_tensor_to_ge_op_map,
                                      *graph_inputs, input_init);
        } else {
            ge::TensorDesc desc = create_tensor_desc_for_node(node);
            gert::Tensor input_tensor = create_bound_tensor_with_ptr(
                desc, node->data, ggml_nbytes(node));
            input_init.push_back(std::move(input_tensor));
        }
    };
    for (int i = 0; i < count; i++) {
        ggml_tensor* node = tensor_array[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor* src = node->src[j];
            if (src == nullptr || ggml_is_empty(src) ||
                src->op != GGML_OP_NONE) {
                continue;
            }
            create_data(src);
        }
        if (ggml_is_empty(node) || node->op != GGML_OP_NONE) {
            continue;
        }
        create_data(node);
    }
}

/**
 * @brief 构建Ascend(昇腾)计算图
 *
 * 将GGML计算图(cgraph)转换为Ascend计算图，支持各种算子类型
 * 该函数是CANN后端的核心功能之一，负责算子到昇腾算子的映射和转换
 *
 * @param cgraph GGML计算图
 * @param input_init 输出参数，存储图的输入Tensor
 * @param output_init 输出参数，存储图的输出Tensor
 * @return 构建好的Ascend计算图
 */
ge::Graph build_ascend_graph(ggml_cgraph* cgraph,
                             ggml_backend_cann_context& cann_ctx,
                             std::vector<gert::Tensor>& input_init,
                             std::vector<gert::Tensor>& output_init) {
    // 创建新的Ascend图
    Graph graph("Graph");
    cann_ctx.n_ctx = cgraph->n_ctx;

    // 用于跟踪已处理的张量，键为GGML张量指针，值为对应的Ascend算子
    std::map<ggml_tensor*, Operator> ggml_tensor_to_ge_op_map;
    std::vector<Operator> graph_inputs, graph_outputs;

    // 第一部分：处理叶子节点（通常是输入或常量）
    // --------------------------------------------------------------
    // 叶子节点肯定是输入或常量，所以先保存一份
    process_input_tensors(cgraph->leafs, cgraph->n_leafs, input_init, true,
                          "leaf_", 0, &graph, &ggml_tensor_to_ge_op_map,
                          &graph_inputs);

    // 第二部分：处理GGML_OP_NONE节点（输入张量）
    // --------------------------------------------------------------
    process_input_tensors(cgraph->nodes, cgraph->n_nodes, input_init, true,
                          "data_", cgraph->n_leafs, &graph,
                          &ggml_tensor_to_ge_op_map, &graph_inputs);

    // 第三部分：查找首个和最后一个非NONE算子节点
    // --------------------------------------------------------------
    ggml_tensor* first_op_node = nullptr;
    ggml_tensor* last_op_node = find_last_op_node(cgraph);

    // 查找第一个操作节点
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];
        if (!ggml_is_empty(node) && node->op != GGML_OP_NONE) {
            first_op_node = node;
            break;
        }
    }

    // 第四部分：处理算子节点
    // --------------------------------------------------------------
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) {
            continue;  // 跳过空节点和输入节点（已经处理过）
        }

        // 根据不同的操作类型处理
        switch (node->op) {
            case GGML_OP_ADD: {
                // 使用ascend_graph_ops.h中的函数处理ADD操作
                Operator add_op =
                    handle_add_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = add_op;

                // 如果这是最后一个操作，将其添加到输出中
                if (node == last_op_node) {
                    graph_outputs.push_back(add_op);
                }
                break;
            }

            case GGML_OP_MUL: {
                // 处理MUL（乘法）操作
                Operator mul_op =
                    handle_mul_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = mul_op;

                // 如果这是最后一个操作，将其添加到输出中
                if (node == last_op_node) {
                    graph_outputs.push_back(mul_op);
                }
                break;
            }

            case GGML_OP_MUL_MAT: {
                // 处理矩阵乘法操作
                ge::Operator matmul_op =
                    handle_matmul_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = matmul_op;

                if (node == last_op_node) {
                    graph_outputs.push_back(matmul_op);
                }
                break;
            }

            case GGML_OP_SCALE: {
                // 处理缩放操作
                Operator scale_op =
                    handle_scale_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = scale_op;

                if (node == last_op_node) {
                    graph_outputs.push_back(scale_op);
                }
                break;
            }

            case GGML_OP_SOFT_MAX: {
                // 处理Softmax操作
                Operator softmax_op =
                    handle_softmax_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = softmax_op;

                if (node == last_op_node) {
                    graph_outputs.push_back(softmax_op);
                }
                break;
            }

            case GGML_OP_REPEAT: {
                // 处理重复操作
                Operator repeat_op =
                    handle_repeat_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = repeat_op;

                if (node == last_op_node) {
                    graph_outputs.push_back(repeat_op);
                }
                break;
            }

            case GGML_OP_RESHAPE: {
                // 处理重塑形状操作
                Operator reshape_op =
                    handle_reshape_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = reshape_op;

                if (node == last_op_node) {
                    graph_outputs.push_back(reshape_op);
                }
                break;
            }

            case GGML_OP_PERMUTE: {
                // 处理维度置换操作
                Operator permute_op =
                    handle_permute_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = permute_op;

                if (node == last_op_node) {
                    graph_outputs.push_back(permute_op);
                }
                break;
            }
            case GGML_OP_PAD: {
                Operator pad_op =
                    handle_pad_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = pad_op;
                if (node == last_op_node) {
                    graph_outputs.push_back(pad_op);
                }
                break;
            }
            case GGML_OP_VIEW: {
                // 处理视图操作
                Operator view_op =
                    handle_view_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = view_op;

                if (node == last_op_node) {
                    graph_outputs.push_back(view_op);
                }
                break;
            }

            case GGML_OP_TRANSPOSE: {
                // 处理转置操作
                Operator transpose_op = handle_transpose_op(
                    graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = transpose_op;

                if (node == last_op_node) {
                    graph_outputs.push_back(transpose_op);
                }
                break;
            }

            case GGML_OP_CONT: {
                // 处理连续存储操作
                Operator cont_op =
                    handle_cont_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = cont_op;

                if (node == last_op_node) {
                    graph_outputs.push_back(cont_op);
                }
                break;
            }

            case GGML_OP_CPY: {
                // 处理复制操作
                Operator cpy_op =
                    handle_cpy_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = cpy_op;

                if (node == last_op_node) {
                    graph_outputs.push_back(cpy_op);
                }
                break;
            }

            case GGML_OP_UNARY: {
                // 处理一元操作（如激活函数）
                switch (ggml_get_unary_op(node)) {
                    case GGML_UNARY_OP_SILU: {
                        Operator silu_op = handle_silu_op(
                            graph, node, ggml_tensor_to_ge_op_map, i);
                        ggml_tensor_to_ge_op_map[node] = silu_op;
                        if (node == last_op_node) {
                            graph_outputs.push_back(silu_op);
                        }
                        break;
                    }
                    default:
                        std::cerr << "Unhandled operation type: " << node->op
                                  << std::endl;
                        break;
                }
                break;
            }

            case GGML_OP_ARGSORT: {
                // 处理ArgSort操作（返回排序后的索引）
                Operator argsort_op =
                    handle_argsort_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = argsort_op;
                if (node == last_op_node) {
                    graph_outputs.push_back(argsort_op);
                }
                break;
            }

            case GGML_OP_CONCAT: {
                // 处理拼接操作
                Operator concat_op =
                    handle_concat_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = concat_op;
                if (node == last_op_node) {
                    graph_outputs.push_back(concat_op);
                }
                break;
            }
            case GGML_OP_RMS_NORM_FUSED:
            case GGML_OP_RMS_NORM: {
                // 处理RMS归一化操作
                Operator rms_norm_op = handle_rms_norm_op(
                    graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = rms_norm_op;
                if (node == last_op_node) {
                    graph_outputs.push_back(rms_norm_op);
                }
                break;
            }

            case GGML_OP_ROPE: {
                // 处理旋转位置编码(RoPE)操作
                Operator rope_op = handle_rope_op(
                    graph, node, ggml_tensor_to_ge_op_map, i, cann_ctx);
                ggml_tensor_to_ge_op_map[node] = rope_op;
                if (node == last_op_node) {
                    graph_outputs.push_back(rope_op);
                }
                break;
            }

            case GGML_OP_MOE_FUSED: {
                // 处理MOE Fused操作
                Operator moe_fused_op = handle_moe_fused_op(
                    graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = moe_fused_op;
                if (node == last_op_node) {
                    graph_outputs.push_back(moe_fused_op);
                }
                break;
            }

            case GGML_OP_ARANGE: {
                // 处理Arange操作
                Operator arange_op =
                    handle_arange_op(graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = arange_op;
                if (node == last_op_node) {
                    graph_outputs.push_back(arange_op);
                }
                break;
            }
            case GGML_OP_GET_SLICE: {
                // 处理步长切片操作
                Operator strided_slice_op = handle_stridedslicev2_op(
                    graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = strided_slice_op;
                if (node == last_op_node) {
                    graph_outputs.push_back(strided_slice_op);
                }
                break;
            }
            // case GGML_OP_STRIDED_SLICE_V2: {
            //     // 处理步长切片操作
            //     Operator strided_slice_op = handle_stridedslicev2_op(
            //         graph, node, ggml_tensor_to_ge_op_map, i);
            //     ggml_tensor_to_ge_op_map[node] = strided_slice_op;
            //     if (node == last_op_node) {
            //         graph_outputs.push_back(strided_slice_op);
            //     }
            //     break;
            // }
            case GGML_OP_SCATTER_UPDATE: {
                // 处理步长切片赋值操作
                Operator set_slice_op = handle_set_slice_op(
                    graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = set_slice_op;
                if (node == last_op_node) {
                    graph_outputs.push_back(set_slice_op);
                }
                break;
            }
            case GGML_OP_FLASH_ATTN_PROMPT: {
                Operator op_node = handle_flash_attn_prompt_op(
                    graph, node, ggml_tensor_to_ge_op_map, i);

                ggml_tensor_to_ge_op_map[node] = op_node;
                if (node == last_op_node) {
                    graph_outputs.push_back(op_node);
                }
                break;
            }
            case GGML_OP_GET_ROWS: {
                // 处理获取行操作
                Operator get_rows_op = handle_get_rows_op(
                    graph, node, ggml_tensor_to_ge_op_map, i);
                ggml_tensor_to_ge_op_map[node] = get_rows_op;

                if (node == last_op_node) {
                    graph_outputs.push_back(get_rows_op);
                }
                break;
            }
            default:
                // 未处理的操作类型
                std::cerr << "Unhandled operation type: " << node->op
                          << std::endl;
                break;
        }
    }

    // 第五部分：设置图的输入和输出，并处理output_init
    // --------------------------------------------------------------
    if (!graph_inputs.empty() && !graph_outputs.empty()) {
        // 创建一个std::vector<std::pair<ge::Operator,
        // std::vector<size_t>>>来指定输出算子及其输出索引
        int prev_size = graph_outputs.size();
        create_output_tensors(graph_outputs, last_op_node, output_init);
        // for(auto& op: debug_operators) {
        //     graph_outputs.push_back(op);
        // }
        // create_debug_tensors(graph_outputs, last_op_node, output_init,
        // prev_size);
        std::vector<std::pair<ge::Operator, std::vector<size_t>>>
            indexed_graph_outputs;
        for (const auto& op : graph_outputs) {
            indexed_graph_outputs.push_back(
                {op, {0}});  // 默认使用第0个输出端口
        }
        graph.SetInputs(graph_inputs).SetOutputs(indexed_graph_outputs);

        // 为图中的每个输出创建对应的output_init张量
    } else {
        std::cerr << "Graph inputs or outputs are empty." << std::endl;
    }

    return graph;
}

Status reuse_ascend_graph(uint32_t graph_idx, ge::Session* session,
                          ggml_cgraph* cgraph, const aclrtStream& stream,
                          std::vector<gert::Tensor>& input_init,
                          std::vector<gert::Tensor>& output_init) {
    // 重新清空输入输出张量向量
    // input_init.clear();
    // output_init.clear();

    // 第一部分：重新创建输入张量
    // --------------------------------------------------------------

    // 处理叶子节点（输入张量）
    // process_input_tensors(cgraph->leafs, cgraph->n_leafs, input_init, false);

    // 处理nodes中的GGML_OP_NONE张量（也是输入）
    // process_input_tensors(cgraph->nodes, cgraph->n_nodes, input_init, false);

    // 第二部分：重新创建输出张量
    // --------------------------------------------------------------

    // 找到最后一个操作节点（输出节点）
    // ggml_tensor* last_op_node = find_last_op_node(cgraph);

    // // 创建输出张量（复用模式下只有一个输出）
    // if (last_op_node != nullptr) {
    //     //
    //     为了和build_ascend_graph保持一致，我们创建一个假的graph_outputs列表
    //     // 但实际上复用模式下我们只需要直接创建输出张量
    //     ge::TensorDesc tensor_desc =
    //     create_tensor_desc_for_node(last_op_node); gert::Tensor output_tensor
    //     = create_bound_tensor_with_ptr(
    //         tensor_desc, last_op_node->data, ggml_nbytes(last_op_node));
    //     output_init.push_back(std::move(output_tensor));
    // }

    return SUCCESS;
}