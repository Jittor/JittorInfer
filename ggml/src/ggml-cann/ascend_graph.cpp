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
// 引入es构图api的聚合头文件
#include "es_all_ops.h"

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
 * @param ggml_tensor_to_es_tensor_map 张量映射
 * @param input_init 输入张量初始化列表
 */
void create_graph_input_tensor_es(
    ggml_tensor* src_tensor,
    std::map<ggml_tensor*, ge::es::EsTensorHolder>*
        ggml_tensor_to_es_tensor_map,
    std::vector<gert::Tensor>& input_init) {
    // 获取输出描述符，使用输出端口0
    ge::es::EsTensorHolder tensor_holder =
        ggml_tensor_to_es_tensor_map->at(src_tensor);

    // tensor_holder当前并没有提供SetPlacement方法，所以用下面这种方式绕一下；
    // TODO:EsTensorHolder提供SetPlacement方法
    ge::TensorDesc tensor_desc;
    (void)tensor_holder.GetProducer()->GetOutputDesc(0, tensor_desc);
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

namespace {
// ES version helper function declarations
void process_input_tensors_es(
    ggml_tensor** tensor_array, int count,
    std::vector<gert::Tensor>& input_init, const std::string& name_prefix,
    int index_offset, ge::es::EsGraphBuilder* graph_builder,
    std::map<ggml_tensor*, ge::es::EsTensorHolder>*
        ggml_tensor_to_es_tensor_map,
    std::vector<ge::es::EsTensorHolder>* graph_inputs) {
    auto create_data = [&](ggml_tensor* node) {
        if (ggml_tensor_to_es_tensor_map->find(node) !=
            ggml_tensor_to_es_tensor_map->end()) {
            return;
        }

        // 使用辅助函数创建张量描述符
        ge::TensorDesc desc = create_tensor_desc_for_node(node);

        // 构建输入名称
        std::string name = name_prefix + std::string(node->name);

        // 使用ES API创建输入
        int input_index = static_cast<int>(graph_inputs->size()) + index_offset;
        auto es_input = graph_builder->CreateInput(
            input_index, name.c_str(), desc.GetDataType(), desc.GetFormat(),
            desc.GetShape().GetDims());

        // 添加到映射和输入列表
        (*ggml_tensor_to_es_tensor_map)[node] = es_input;
        graph_inputs->push_back(es_input);

        // 创建对应的gert::Tensor用于input_init
        create_graph_input_tensor_es(node, ggml_tensor_to_es_tensor_map,
                                     input_init);
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
 * @brief 创建输出张量列表（ES API版本）
 *
 * @param graph_outputs 图输出ES张量列表
 * @param last_op_node 最后一个操作节点
 * @param output_init 输出张量初始化列表
 */
void create_output_tensors_es(
    std::vector<ge::es::EsTensorHolder>& graph_outputs,
    ggml_tensor* last_op_node, std::vector<gert::Tensor>& output_init) {
    if (!graph_outputs.empty() && last_op_node != nullptr) {
        for (auto& es_output : graph_outputs) {
            ge::TensorDesc tensor_desc;
            (void)es_output.GetProducer()->GetOutputDesc(
                es_output.GetProducerOutIndex(), tensor_desc);
            tensor_desc.SetPlacement(ge::Placement::kPlacementDevice);

            // 使用辅助函数创建并绑定张量
            gert::Tensor output_tensor = create_bound_tensor_with_ptr(
                tensor_desc, last_op_node->data, ggml_nbytes(last_op_node));
            output_init.push_back(std::move(output_tensor));
        }
    }
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
    // 使用 ES API 版本构建图
    return build_ascend_graph_es(cgraph, cann_ctx, input_init, output_init);
}

ge::Graph build_ascend_graph_es(ggml_cgraph* cgraph,
                                ggml_backend_cann_context& cann_ctx,
                                std::vector<gert::Tensor>& input_init,
                                std::vector<gert::Tensor>& output_init) {
    // 创建ES Graph Builder
    ge::es::EsGraphBuilder graph_builder("Graph");
    cann_ctx.n_ctx = cgraph->n_ctx;

    // 用于跟踪已处理的张量，键为GGML张量指针，值为对应的ES张量
    std::map<ggml_tensor*, ge::es::EsTensorHolder> ggml_tensor_to_es_tensor_map;
    std::vector<ge::es::EsTensorHolder> graph_inputs;
    std::vector<ge::es::EsTensorHolder> graph_outputs;

    // 第一部分：处理叶子节点（通常是输入或常量）
    // --------------------------------------------------------------
    process_input_tensors_es(cgraph->leafs, cgraph->n_leafs, input_init, "leaf_",
                             0, &graph_builder, &ggml_tensor_to_es_tensor_map,
                             &graph_inputs);

    // 第二部分：处理GGML_OP_NONE节点（输入张量）
    // --------------------------------------------------------------
    process_input_tensors_es(cgraph->nodes, cgraph->n_nodes, input_init, "data_",
                             cgraph->n_leafs, &graph_builder,
                             &ggml_tensor_to_es_tensor_map, &graph_inputs);

    // 第三部分：查找最后一个非NONE算子节点
    // --------------------------------------------------------------
    ggml_tensor* last_op_node = find_last_op_node(cgraph);

    // 第四部分：处理算子节点
    // --------------------------------------------------------------
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) {
            continue;  // 跳过空节点和输入节点（已经处理过）
        }

        // 根据不同的操作类型处理
        ge::es::EsTensorHolder es_result;
        switch (node->op) {
            case GGML_OP_ADD: {
                es_result = handle_add_op_es(graph_builder, node,
                                             ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_MUL: {
                es_result = handle_mul_op_es(graph_builder, node,
                                               ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_MUL_MAT: {
                es_result = handle_matmul_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_SCALE: {
                es_result = handle_scale_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_SOFT_MAX: {
                es_result = handle_softmax_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_REPEAT: {
                es_result = handle_repeat_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_RESHAPE: {
                es_result = handle_reshape_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_PERMUTE: {
                es_result = handle_permute_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_PAD: {
                es_result = handle_pad_op_es(graph_builder, node,
                                              ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_VIEW: {
                es_result = handle_view_op_es(graph_builder, node,
                                               ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_TRANSPOSE: {
                es_result = handle_transpose_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_CONT: {
                es_result = handle_cont_op_es(graph_builder, node,
                                               ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_CPY: {
                es_result = handle_cpy_op_es(graph_builder, node,
                                              ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_UNARY: {
                // 处理一元操作（如激活函数）
                switch (ggml_get_unary_op(node)) {
                    case GGML_UNARY_OP_SILU: {
                        es_result = handle_silu_op_es(
                            graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                        ggml_tensor_to_es_tensor_map[node] = es_result;
                        if (node == last_op_node) {
                            graph_outputs.push_back(es_result);
                        }
                        break;
                    }
                    default:
                        std::cerr << "Unhandled unary operation type: "
                                  << ggml_get_unary_op(node) << std::endl;
                        break;
                }
                break;
            }

            case GGML_OP_ARGSORT: {
                es_result = handle_argsort_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_CONCAT: {
                es_result = handle_concat_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_RMS_NORM_FUSED:
            case GGML_OP_RMS_NORM: {
                es_result = handle_rms_norm_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_ROPE: {
                es_result = handle_rope_op_es(graph_builder, node,
                                               ggml_tensor_to_es_tensor_map, i,
                                               cann_ctx);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_MOE_FUSED: {
                es_result = handle_moe_fused_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_ARANGE: {
                es_result = handle_arange_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_GET_SLICE: {
                es_result = handle_stridedslicev2_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_SCATTER_UPDATE: {
                es_result = handle_set_slice_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_FLASH_ATTN_PROMPT: {
                es_result = handle_flash_attn_prompt_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
                }
                break;
            }

            case GGML_OP_GET_ROWS: {
                es_result = handle_get_rows_op_es(
                    graph_builder, node, ggml_tensor_to_es_tensor_map, i);
                ggml_tensor_to_es_tensor_map[node] = es_result;
                if (node == last_op_node) {
                    graph_outputs.push_back(es_result);
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

    // 第五部分：创建output_init并设置图的输出
    // --------------------------------------------------------------
    if (!graph_inputs.empty() && !graph_outputs.empty()) {
        // 创建output_init
        create_output_tensors_es(graph_outputs, last_op_node, output_init);
        // 设置多个输出并构建图
        return *graph_builder.Build(graph_outputs);
    } else {
        std::cerr << "Graph inputs or outputs are empty." << std::endl;
        // 返回空图
        return ge::Graph("EmptyGraph");
    }
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