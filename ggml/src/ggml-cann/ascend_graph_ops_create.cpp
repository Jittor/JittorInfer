#include "ascend_graph_ops_create.h"

#include <numeric>

#include "selection.h"

#define FLOAT16_SIZE 2
#define BFLOAT16_SIZE 2

ge::Operator create_reshape_op(ge::Graph& graph, const std::string& prefix,
                               const std::string& suffix, ge::Operator src,
                               std::vector<int64_t>&& new_shape) {
    std::string op_name = prefix + "reshape" + suffix;
    ge::op::Reshape reshape_op(op_name.c_str());
    std::string reshape_const_name = prefix + "reshape_const" + suffix;
    ge::op::Const input_shape_const_op(reshape_const_name.c_str());
    ge::TensorDesc input_shape_desc(ge::Shape({(int64_t)new_shape.size()}),
                                    ge::FORMAT_ND, ge::DT_INT64);
    ge::Tensor input_shape_tensor(input_shape_desc,
                                  reinterpret_cast<uint8_t*>(new_shape.data()),
                                  new_shape.size() * sizeof(int64_t));
    input_shape_const_op.set_attr_value(input_shape_tensor);

    reshape_op.set_input_x(src);
    reshape_op.set_input_shape(input_shape_const_op);
    graph.AddOp(input_shape_const_op);
    graph.AddOp(reshape_op);
    return reshape_op;
}

ge::Operator create_squeeze_op(ge::Graph& graph, const std::string& prefix,
                               const std::string& suffix, ge::Operator src,
                               std::vector<int64_t>&& remove_dims) {
    std::string squeeze_name = prefix + "squeeze" + suffix;
    ge::op::Squeeze squeeze_op(squeeze_name.c_str());
    squeeze_op.set_input_x(src);
    squeeze_op.set_attr_axis(remove_dims);
    graph.AddOp(squeeze_op);
    return squeeze_op;
}

ge::Operator create_unsqueeze_op(ge::Graph& graph, const std::string& prefix,
                                 const std::string& suffix, ge::Operator src,
                                 std::vector<int64_t>&& add_dims) {
    std::string squeeze_name = prefix + "unsqueeze" + suffix;
    ge::op::Unsqueeze squeeze_op(squeeze_name.c_str());
    squeeze_op.set_input_x(src);
    squeeze_op.set_attr_axes(add_dims);
    graph.AddOp(squeeze_op);
    return squeeze_op;
}

ge::Operator create_permute_op(ge::Graph& graph, const std::string& prefix,
                               const std::string& suffix, ge::Operator src,
                               std::vector<int64_t>&& permute_dims) {
    // 创建转置维度常量
    std::string permute_const_name = prefix + "permute_const" + suffix;
    ge::op::Const permute_const_op(permute_const_name.c_str());
    ge::TensorDesc permute_const_desc(ge::Shape({(int64_t)permute_dims.size()}),
                                      ge::FORMAT_ND, ge::DT_INT64);
    ge::Tensor permute_tensor(permute_const_desc,
                              reinterpret_cast<uint8_t*>(permute_dims.data()),
                              permute_dims.size() * sizeof(int64_t));
    permute_const_op.set_attr_value(permute_tensor);
    graph.AddOp(permute_const_op);

    // 创建转置操作
    std::string name_permute = prefix + "permute" + suffix;
    ge::op::Transpose perm_x_op(name_permute.c_str());

    perm_x_op.set_input_x(src);
    perm_x_op.set_input_perm(permute_const_op);

    graph.AddOp(perm_x_op);
    return perm_x_op;
}

ge::Operator create_identity_op_by_name(ge::Graph& graph,
                                        const std::string& prefix,
                                        const std::string& suffix,
                                        ge::Operator src,
                                        const std::string& name) {
    std::string identity_name = prefix + "identity" + suffix;
    ge::op::Identity identity_op(identity_name.c_str());
    identity_op.set_input_x_by_name(src, name.c_str());
    graph.AddOp(identity_op);
    return identity_op;
}

ge::Operator create_const_int32_op(ge::Graph& graph, const std::string& prefix,
                                   const std::string& suffix,
                                   std::vector<int64_t>&& shape,
                                   ge::DataType const_type, int32_t value) {
    ge::TensorDesc tensor_desc(ge::Shape(shape), ge::FORMAT_ND, const_type);
    int64_t len = std::accumulate(shape.begin(), shape.end(), 1,
                                  std::multiplies<int64_t>());
    std::vector<int32_t> const_data(len, value);
    ge::Tensor const_tensor(tensor_desc,
                            reinterpret_cast<uint8_t*>(const_data.data()),
                            len * sizeof(int32_t));
    ge::op::Const const_op((prefix + "const" + suffix).c_str());
    const_op.set_attr_value(const_tensor);
    const_op.update_output_desc_y(tensor_desc);
    graph.AddOp(const_op);
    return const_op;
}

ge::Operator create_const_float_op(ge::Graph& graph, const std::string& prefix,
                                   const std::string& suffix,
                                   std::vector<int64_t>&& shape,
                                   ge::DataType const_type, float value) {
    ge::TensorDesc tensor_desc(ge::Shape(shape), ge::FORMAT_ND, const_type);
    int64_t len = std::accumulate(shape.begin(), shape.end(), 1,
                                  std::multiplies<int64_t>());
    std::vector<float> const_data(len, value);
    ge::Tensor const_tensor(tensor_desc,
                            reinterpret_cast<uint8_t*>(const_data.data()),
                            len * sizeof(float));
    ge::op::Const const_op((prefix + "const" + suffix).c_str());
    const_op.set_attr_value(const_tensor);
    const_op.update_output_desc_y(tensor_desc);
    graph.AddOp(const_op);
    return const_op;
}

ge::Operator create_const_float16_zero_op(ge::Graph& graph,
                                          const std::string& prefix,
                                          const std::string& suffix,
                                          std::vector<int64_t>&& shape) {
    ge::TensorDesc tensor_desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT16);
    int64_t len = std::accumulate(shape.begin(), shape.end(), 1,
                                  std::multiplies<int64_t>());
    ge::Tensor const_tensor(tensor_desc,
                            std::vector<uint8_t>(len * FLOAT16_SIZE, 0));
    ge::op::Const const_op((prefix + "const" + suffix).c_str());
    const_op.set_attr_value(const_tensor);
    const_op.update_output_desc_y(tensor_desc);
    graph.AddOp(const_op);
    return const_op;
}

ge::Operator create_const_bf16_zero_op(ge::Graph& graph,
                                       const std::string& prefix,
                                       const std::string& suffix,
                                       std::vector<int64_t>&& shape) {
    ge::TensorDesc tensor_desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_BF16);
    int64_t len = std::accumulate(shape.begin(), shape.end(), 1,
                                  std::multiplies<int64_t>());
    ge::Tensor const_tensor(tensor_desc,
                            std::vector<uint8_t>(len * BFLOAT16_SIZE, 0));
    ge::op::Const const_op((prefix + "const" + suffix).c_str());
    const_op.set_attr_value(const_tensor);
    const_op.update_output_desc_y(tensor_desc);
    graph.AddOp(const_op);
    return const_op;
}

// 对MOE_FUSED算子中的GroupMatmul算子进行封装，预定义好一些属性
ge::Operator create_moe_grouped_matmul_op(
    ge::Graph& graph, const std::string& prefix, const std::string& suffix,
    ge::DataType weight_type, ge::Operator x, ge::Operator weight,
    std::vector<int64_t>&& bias_shape, ge::Operator group_list) {
    ge::Operator bias;
    if (weight_type == ge::DT_FLOAT16) {
        bias = create_const_float16_zero_op(
            graph, prefix + "bias", suffix,
            std::forward<std::vector<int64_t>>(bias_shape));
    } else if (weight_type == ge::DT_BF16) {
        bias = create_const_bf16_zero_op(
            graph, prefix + "bias", suffix,
            std::forward<std::vector<int64_t>>(bias_shape));
    } else if (weight_type == ge::DT_FLOAT) {
        bias = create_const_float_op(
            graph, prefix + "bias", suffix,
            std::forward<std::vector<int64_t>>(bias_shape), weight_type, 0.0f);
    } else {
        GGML_ABORT("Unsupported weight_type for MOE grouped matmul: %d",
                   weight_type);
    }
    // ge::Operator scale_const_op = create_const_float_op(graph, prefix +
    // "_scale_", suffix, {0}, ge::DT_FLOAT, 0.0f);
    ge::TensorDesc tensor_desc(ge::Shape({0}), ge::FORMAT_ND, weight_type);
    ge::Tensor scale_tensor = ge::Tensor(tensor_desc, nullptr, 0);
    ge::op::Const scale_const_op =
        ge::op::Const(prefix + "scale_const" + suffix);
    scale_const_op.set_attr_value(
        scale_tensor);  // 用于占位，如果不需要scale和offset可以传入空操作

    std::string op_name = prefix + "grouped_matmul" + suffix;
    ge::op::GroupedMatmul matmul_op(op_name.c_str());
    // 创建动态输入
    matmul_op.create_dynamic_input_byindex_x(1, 0);
    matmul_op.create_dynamic_input_byindex_weight(1, 1);
    matmul_op.create_dynamic_input_byindex_bias(1, 2);
    matmul_op.create_dynamic_input_byindex_scale(1, 3);
    matmul_op.create_dynamic_input_byindex_offset(1, 4);
    matmul_op.create_dynamic_input_byindex_antiquant_scale(1, 5);
    matmul_op.create_dynamic_input_byindex_antiquant_offset(1, 6);
    // 设置动态输入 - 使用Identity算子的输出
    matmul_op.set_dynamic_input_x(0, x);
    matmul_op.set_dynamic_input_weight(0, weight);
    matmul_op.set_dynamic_input_bias(0, bias);
    matmul_op.set_dynamic_input_scale(0, scale_const_op);
    matmul_op.set_dynamic_input_offset(0, scale_const_op);
    matmul_op.set_dynamic_input_antiquant_scale(0, scale_const_op);
    matmul_op.set_dynamic_input_antiquant_offset(0, scale_const_op);
    matmul_op.set_input_group_list(group_list);

    // 设置属性
    matmul_op.set_attr_split_item(2);
    // up_matmul_op.set_attr_dtype(0);
    matmul_op.set_attr_transpose_weight(false);
    matmul_op.set_attr_transpose_x(false);
    matmul_op.set_attr_group_type(0);
    matmul_op.set_attr_group_list_type(1);
    matmul_op.set_attr_act_type(0);

    // 创建动态输出
    matmul_op.create_dynamic_output_y(1);

    // 参考 ggml_cann_moe_fused: moe_up_ne[2] = {k_dim, num_length}
    // std::vector<int64_t> up_output_shape = {k_dim, num_length};
    // ge::TensorDesc up_output_desc(ge::Shape(up_output_shape), ge::FORMAT_ND,
    //                               get_data_type(expert_up_weights->type));
    // matmul_op.update_dynamic_output_desc_y(0, up_output_desc);
    graph.AddOp(matmul_op);
    return matmul_op;
}

ge::Operator create_gather_op(ge::Graph& graph, const std::string& prefix,
                              const std::string& suffix, ge::Operator src,
                              ge::Operator indices, int axis, int batch_dims) {
    std::string op_name = prefix + "gather" + suffix;
    ge::op::GatherV2 index_by_tensor_op(op_name.c_str());
    ge::Operator axis_op = create_const_int32_op(
        graph, prefix + "axis_", suffix, {1}, ge::DT_INT32, axis);
    index_by_tensor_op.set_input_x(src)
        .set_input_indices(indices)
        .set_input_axis(axis_op)
        .set_attr_batch_dims(batch_dims);
    graph.AddOp(index_by_tensor_op);
    return index_by_tensor_op;
}

ge::Operator create_cast_op(ge::Graph& graph, const std::string& prefix,
                            const std::string& suffix, ge::Operator src,
                            ge::DataType dst_type) {
    std::string op_name = prefix + "cast" + suffix;
    ge::op::Cast cast_op(op_name.c_str());
    cast_op.set_input_x(src);
    cast_op.set_attr_dst_type(dst_type);
    graph.AddOp(cast_op);
    return cast_op;
}

ge::Operator create_concat_op(ge::Graph& graph, const std::string& prefix,
                              const std::string& suffix,
                              std::vector<ge::Operator>&& op_list, int axis) {
    std::string op_name = prefix + "concat" + suffix;
    ge::op::Concat concat_op(op_name.c_str());
    concat_op.create_dynamic_input_x(op_list.size());
    for (int i = 0; i < op_list.size(); i++) {
        concat_op.set_dynamic_input_x(i, op_list[i]);
    }
    ge::Operator aixs_op = create_const_int32_op(
        graph, prefix + "concat_", suffix, {1}, ge::DT_INT32, axis);
    concat_op.set_input_concat_dim(aixs_op).set_attr_N(op_list.size());
    graph.AddOp(concat_op);
    return concat_op;
}
