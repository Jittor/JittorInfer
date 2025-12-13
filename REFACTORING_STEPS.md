# GE API 到 ES API 重构步骤总结

## 重构目标

将 `build_ascend_graph` 函数从使用 GE (Graph Engine) API 重构为使用 ES (Eager Style) API 进行图构建，确保每个修改步骤都是小步前进、可编译、可验证的。

## 参考资源

- ES API 使用示例目录：`https://gitcode.com/cann/ge-dev/tree/master/examples/es`

---

## 阶段零：工程适配（重构前提）

### 步骤 0.1：在 CMakeLists.txt 中引入 ES API 支持

**修改文件：**
- `ggml/src/ggml-cann/CMakeLists.txt`

**具体修改：**

1. **引入 ES API 生成工具：**
   ```cmake
   list(APPEND CMAKE_MODULE_PATH "${CANN_INSTALL_DIR}/include/ge/cmake")
   find_package(GenerateEsPackage REQUIRED)
   ```

2. **定义原型库：**
   ```cmake
   add_library(opgraph_all INTERFACE)
   set_target_properties(opgraph_all PROPERTIES
       INTERFACE_LIBRARY_OUTPUT_DIRECTORY "${CANN_INSTALL_DIR}/opp/built-in/op_proto"
   )
   ```

3. **生成 ES API 包：**
   ```cmake
   add_es_library(
       ES_LINKABLE_AND_ALL_TARGET es_all
       OPP_PROTO_TARGET  opgraph_all
       OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
   )
   ```

4. **链接 ES API 库：**
   ```cmake
   target_link_libraries(ggml-cann PRIVATE
       es_all # 自动获得依赖、头文件和库
   )
   ```

5. **修复编译问题：**
   ```cmake
   # CANN包本身的BUG, 引入es_all之后头文件搜索路径变化，搜索到了不包含GE_FUNC_VISIBILITY的头文件ge_error_codes.h，导致编译失败
   # 这里手动定义一个空的GE_FUNC_VISIBILITY，避免编译失败
   target_compile_definitions(ggml-cann PRIVATE "-DGE_FUNC_VISIBILITY=")
   ```

**状态：** ✅ 已完成（重构前提条件）

**说明：** 这一步是后续所有 ES API 重构工作的基础，必须首先完成。

---

## 阶段一：基础准备

### 步骤 1.1：添加 ES API 头文件和命名空间支持

**修改文件：**
- `ggml/src/ggml-cann/ascend_graph.h`
- `ggml/src/ggml-cann/ascend_graph.cpp`
- `ggml/src/ggml-cann/ascend_graph_ops.h`
- `ggml/src/ggml-cann/ascend_graph_ops.cpp`

**具体修改：**

1. **在 `ascend_graph.h` 中添加：**
   - 声明 `build_ascend_graph_es()` 函数（仅此函数声明，无其他前置声明）

2. **在 `ascend_graph.cpp` 中添加：**
   - `#include "es_all_ops.h"`
   - 在匿名 namespace 中实现 `create_graph_input_tensor_es()` 和 `process_input_tensors_es()`（这些函数不在头文件中声明）

3. **在 `ascend_graph_ops.h` 中添加：**
   - `#include "es_all_ops.h"`

4. **在 `ascend_graph_ops.cpp` 中添加：**
   - `using namespace ge::es;`（如果需要）

**状态：** ✅ 已完成

**注意：** 
- `process_input_tensors_es()` 和 `create_graph_input_tensor_es()` 是在 `ascend_graph.cpp` 的匿名 namespace 中实现的，**不在头文件中声明**
- `ascend_graph.h` 中只有 `build_ascend_graph_es()` 的函数声明

---

## 阶段二：创建 ES 版本的辅助函数（并行实现）

### 步骤 2.1：创建 `create_graph_input_tensor_es()` 函数

**修改文件：**
- `ggml/src/ggml-cann/ascend_graph.cpp`（在匿名 namespace 中实现）

**功能说明：**
- ES 版本的图输入张量创建函数
- 对应 GE API 版本的 `create_graph_input_tensor()`
- 从 `EsTensorHolder` 获取张量描述符并创建 `gert::Tensor`

**实现位置：**
```cpp
namespace {
void create_graph_input_tensor_es(
    ggml_tensor* src_tensor,
    std::map<ggml_tensor*, ge::es::EsTensorHolder>*
        ggml_tensor_to_es_tensor_map,
    std::vector<gert::Tensor>& input_init)
```

**关键实现细节：**
- 通过 `tensor_holder.GetProducer()->GetOutputDesc(tensor_holder.GetProducerOutIndex(), tensor_desc)` 获取张量描述符
- 设置 `Placement` 为 `kPlacementDevice`
- 使用 `create_bound_tensor_with_ptr()` 创建并绑定数据

**TODO 标记：**
- `// TODO:EsTensorHolder直接提供SetPlacement方法` - 当前通过 `GetProducer()->GetOutputDesc()` 绕过了这个问题

**状态：** ✅ 已完成

---

### 步骤 2.2：创建 `process_input_tensors_es()` 函数

**修改文件：**
- `ggml/src/ggml-cann/ascend_graph.cpp`（在匿名 namespace 中实现）

**功能说明：**
- ES 版本的输入张量处理函数
- 对应 GE API 版本的 `process_input_tensors()`
- 使用 `EsGraphBuilder::CreateInput()` 创建图输入

**实现位置：**
```cpp
namespace {
void process_input_tensors_es(
    ggml_tensor** tensor_array, int count,
    std::vector<gert::Tensor>& input_init, const std::string& name_prefix,
    int index_offset, ge::es::EsGraphBuilder* graph_builder,
    std::map<ggml_tensor*, ge::es::EsTensorHolder>*
        ggml_tensor_to_es_tensor_map,
    std::vector<ge::es::EsTensorHolder>* graph_inputs)
```

**关键实现细节：**
- 使用 `graph_builder->CreateInput()` 创建输入张量
- 维护 `ggml_tensor_to_es_tensor_map` 映射关系
- 调用 `create_graph_input_tensor_es()` 创建对应的 `gert::Tensor`

**状态：** ✅ 已完成

---

## 阶段三：实现所有算子的 ES 版本（必须全部完成）

**重要说明：** 
- **在实现 `build_ascend_graph_es()` 之前，必须完成所有 `handle_xx_op_es()` 函数的实现**
- 因为 `build_ascend_graph_es()` 要完全替换 `build_ascend_graph()`，所以需要支持所有原有的算子
- 所有在 `build_ascend_graph()` 中使用的 `handle_xx_op()` 函数，都必须有对应的 `handle_xx_op_es()` 实现

**修改文件：**
- `ggml/src/ggml-cann/ascend_graph_ops.h`（函数声明）
- `ggml/src/ggml-cann/ascend_graph_ops.cpp`（函数实现）

### 已完成的算子 ES 版本

#### 3.1：`handle_add_op_es()` - ADD 算子

**功能：** 使用 ES API 创建加法操作

**实现方式：**
- 使用运算符重载：`es_tensor1 + es_tensor2`
- 链式调用设置数据类型和形状：`.SetDataType().SetShape()`

**状态：** ✅ 已完成


---

#### 3.2：`handle_mul_op_es()` - MUL 算子

**功能：** 使用 ES API 创建乘法操作

**实现方式：**
- 使用运算符重载：`es_tensor1 * es_tensor2`
- 链式调用设置数据类型和形状

**状态：** ✅ 已完成

---

#### 3.3：`handle_scale_op_es()` - SCALE 算子

**功能：** 使用 ES API 创建缩放操作

**实现方式：**
- 使用 `graph_builder->CreateVector()` 创建常量标量
- 使用运算符重载：`es_tensor * scale_scalar`
- 链式调用设置数据类型和形状

**关键代码：**
```cpp
return (es_tensor * graph_builder.CreateVector(std::vector{scale_val}))
    .SetDataType(get_data_type(node->type))
    .SetShape(build_output_shape(node));
```

**状态：** ✅ 已完成

---

#### 3.4：`handle_reshape_op_es()` - RESHAPE 算子

**功能：** 使用 ES API 创建形状重塑操作

**实现方式：**
- 使用 `Reshape()` 函数，直接传入形状向量
- ES API 内部会自动创建常量

**关键代码：**
```cpp
return Reshape(es_tensor, shape)
    .SetDataType(get_data_type(node->type))
    .SetShape(build_output_shape(node));
```

**状态：** ✅ 已完成

---

#### 3.5：`handle_permute_op_es()` - PERMUTE 算子

**功能：** 使用 ES API 创建维度置换操作

**实现方式：**
- 使用 `Permute()` 函数
- 从 `node->op_params` 读取维度顺序

**关键代码：**
```cpp
return Permute(es_tensor, order)
    .SetDataType(get_data_type(node->type))
    .SetShape(build_output_shape(node));
```

**状态：** ✅ 已完成

---

#### 3.6：`handle_matmul_op_es()` - MATMUL 算子

**功能：** 使用 ES API 创建矩阵乘法操作

**实现方式：**
- 使用 `MatMul()` 函数
- 保持与原始 GE API 相同的逻辑：
  - 输入顺序反转：`MatMul(src1, src0, ...)`
  - 第二个输入转置：`transpose_x2=true`

**关键代码：**
```cpp
return MatMul(es_tensor1, es_tensor0, nullptr, false, true)
    .SetDataType(get_data_type(node->type))
    .SetShape(build_output_shape(node));
```

**状态：** ✅ 已完成

---

#### 3.7：`handle_softmax_op_es()` - SOFTMAX 算子

**功能：** 使用 ES API 创建 Softmax 操作

**实现方式：**
- 支持 scale 缩放和 mask 掩码处理（包括 ALiBi）
- 使用 `StridedSliceV2()` 直接处理 mask 切片（无需等待专门的函数实现）
- 使用 `SoftmaxV2()` 函数创建 Softmax 操作

**状态：** ✅ 已完成

---

#### 3.8：`handle_repeat_op_es()` - REPEAT 算子

**功能：** 使用 ES API 创建重复/广播操作

**实现方式：**
- 使用 `Tile()` 函数实现重复操作
- 计算每个维度的重复倍数

**状态：** ✅ 已完成

---

#### 3.9：`handle_silu_op_es()` - SILU 算子

**功能：** 使用 ES API 创建 Swish 激活函数操作

**实现方式：**
- 使用 `Swish()` 函数（SILU 是 Swish 的特殊情况）

**状态：** ✅ 已完成

---

#### 3.10：`handle_argsort_op_es()` - ARGSORT 算子

**功能：** 使用 ES API 创建参数排序操作

**实现方式：**
- 使用 `Sort()` 函数
- **使用结构化绑定直接获取第二个输出（indices）**，无需像 GE API 那样创建 Identity 算子

**关键代码：**
```cpp
// 使用结构化绑定获取 Sort 的两个输出
auto [y1, y2] = Sort(es_input, -1, descending);
// 直接返回 indices（y2），无需额外的 Identity 算子
return y2.SetDataType(ge::DT_INT32).SetShape(build_output_shape(node));
```

**改进点：**
- ES API 支持结构化绑定，可以直接获取多输出算子的特定输出
- 避免了 GE API 中需要创建 Identity 算子的额外开销
- 代码更简洁，性能更好

**状态：** ✅ 已完成


#### 3.14：`handle_stridedslicev2_op_es()` - StridedSliceV2 算子

**功能：** 使用 ES API 创建步长切片操作

**实现方式：**
- 使用 `StridedSliceV2()` 函数
- 从 `op_params` 读取 `{fr, to, axis}` 参数
- 构造 `begin_vec`、`end_vec`、`strides_vec`、`axes_vec`
- `EsTensorLike` 支持 `std::vector<int64_t>` 的隐式构造，可以直接传递向量

**状态：** ✅ 已完成

---

#### 3.15：`handle_flash_attn_prompt_op_es()` - Flash Attention Prompt 算子

**功能：** 使用 ES API 创建 Flash Attention 的 prompt 阶段操作

**实现方式：**
- 使用 `FusedInferAttentionScore()` 函数（与原始 GE 实现一致）
- `key` 和 `value` 需要包装成 `std::vector<EsTensorHolder>`（动态输入）
- 返回 `FusedInferAttentionScoreOutput` 结构体，包含 `attention_out` 和 `softmax_lse`
- 只返回 `attention_out`（与原始实现一致）
- 对于可选参数，使用 `nullptr` 表示未传递

**关键代码：**
```cpp
std::vector<ge::es::EsTensorHolder> es_key_vec = {es_key};
std::vector<ge::es::EsTensorHolder> es_value_vec = {es_value};
auto [attention_out, softmax_lse] = FusedInferAttentionScore(
    es_query, es_key_vec, es_value_vec,
    nullptr,  // pse_shift
    es_attn_mask, es_length_q_tensor, es_length_kv_tensor,
    nullptr,  // 其他可选参数...
    num_heads, scale_value, pre_tokens, next_tokens, input_layout,
    num_key_value_heads, sparse_mode, inner_precise, ...);
return attention_out.SetDataType(...).SetShape(...);
```

**状态：** ✅ 已完成

---

#### 3.16：`handle_set_slice_op_es()` - Set Slice 算子

**功能：** 使用 ES API 创建切片赋值操作

**实现方式：**
- 使用 `ScatterUpdate()` 函数（与原始 GE 实现一致）
- 先对 `var`、`indices`、`updates` 进行形状处理（squeeze + reshape）
- 使用 ES API 的 `Reshape` 和 `ScatterUpdate`

**状态：** ✅ 已完成

---

#### 3.17：`handle_get_rows_op_es()` - Get Rows 算子

**功能：** 使用 ES API 创建按行索引获取张量行的操作

**实现方式：**
- 使用 `GatherV2()` 函数（与原始 GE 实现一致）
- 处理流程：Squeeze → GatherV2 → Unsqueeze → Cast
- `GatherV2` 的 `axis` 参数直接传递 `int64_t` 值，无需创建标量常量

**关键代码：**
```cpp
auto es_squeezed_src = Squeeze(es_src, std::vector<int64_t>{0});
auto es_squeezed_rows = Squeeze(es_rows, std::vector<int64_t>{0, 1});
auto es_result = GatherV2(es_squeezed_src, es_squeezed_rows, static_cast<int64_t>(1), 1);
auto es_unsq_result = Unsqueeze(es_result, std::vector<int64_t>{0});
return Cast(es_unsq_result, ...);
```

**状态：** ✅ 已完成

---

#### 3.18：`handle_pad_op_es()` - Pad 算子

**功能：** 使用 ES API 创建张量填充操作

**实现方式：**
- 使用 `PadV3()` 函数（与原始 GE 实现一致）
- `paddings` 直接传递 `std::vector<int64_t>`（支持隐式构造）
- `constant_values` 直接传递标量值 `0.0f`（支持隐式构造）

**关键代码：**
```cpp
std::vector<int64_t> paddings = {
    0, node->ne[3] - src->ne[3], 0, node->ne[2] - src->ne[2],
    0, node->ne[1] - src->ne[1], 0, node->ne[0] - src->ne[0]};
return PadV3(es_input, paddings, 0.0f, "constant", true)
    .SetDataType(...).SetShape(...);
```

**状态：** ✅ 已完成

---

---

### 待实现的算子 ES 版本（必须全部完成）

根据 `ascend_graph_ops.h` 中的函数声明，以下算子的 ES 版本**必须全部实现**：

1. ~~`handle_softmax_op_es()` - Softmax 算子~~ ✅ 已完成
2. ~~`handle_repeat_op_es()` - Repeat 算子~~ ✅ 已完成
3. ~~`handle_silu_op_es()` - SiLU 算子~~ ✅ 已完成
4. ~~`handle_argsort_op_es()` - Argsort 算子~~ ✅ 已完成
5. ~~`handle_cpy_op_es()` - Copy 算子~~ ✅ 已完成
6. ~~`handle_transpose_op_es()` - Transpose 算子~~ ✅ 已完成
7. ~~`handle_concat_op_es()` - Concat 算子~~ ✅ 已完成
8. ~~`handle_view_op_es()` - View 算子~~ ✅ 已完成
9. ~~`handle_cont_op_es()` - Contiguous 算子~~ ✅ 已完成
10. ~~`handle_rms_norm_op_es()` - RMS Norm 算子~~ ✅ 已完成
11. ~~`handle_rope_op_es()` - RoPE 算子~~ ✅ 已完成（有 TODO 标记，但不影响功能）
12. `handle_moe_fused_op_es()` - MoE Fused 算子 ✅ 已完成
13. ~~`handle_arange_op_es()` - Arange 算子~~ ✅ 已完成
14. ~~`handle_stridedslicev2_op_es()` - StridedSliceV2 算子~~ ✅ 已完成
15. ~~`handle_flash_attn_prompt_op_es()` - Flash Attention Prompt 算子~~ ✅ 已完成
16. ~~`handle_set_slice_op_es()` - Set Slice 算子~~ ✅ 已完成
17. ~~`handle_get_rows_op_es()` - Get Rows 算子~~ ✅ 已完成
18. ~~`handle_pad_op_es()` - Pad 算子~~ ✅ 已完成

**总计：** 0 个算子待实现（已完成 24 个，其中 1 个TODO改善实现）

**重要：** 在实现 `build_ascend_graph_es()` 之前，这 24 个算子的 ES 版本必须全部完成（其中  1 个TODO改善实现）

---


### 待完善的 TODO 项

在已实现的算子中，有以下 TODO 需要完善：

1. **删除未使用的 `op_index` 参数：**
   - 位置：`handle_add_op_es()`, `handle_mul_op_es()`等
   - 原因：ES API 内部维护了自增索引，不需要外部传入

2. **实现 ES 版本的广播处理：**
   - 位置：`handle_add_op_es()`, `handle_mul_op_es()`
   - 说明：当前依赖 ES API 自动处理广播，但可能需要显式调用 `handle_repeat_op_es()` 来处理某些情况

3. **RoPE 算子的实现：**
   - 位置：`handle_rope_op_es()` ✅ **已完成**
   - **实现思路：** 由于 RoPE 算子涉及 RopeCache 和自定义算子 `RopeExtCustomV2`，采用 GE 和 ES API 混合方式：
     - **RopeCache 优化：** 在 `RopeCache` 类中添加了 `GetSinEsTensor()` 和 `GetCosEsTensor()` 方法，直接使用 `EsGraphBuilder::CreateConst()` 创建常量节点，避免了 GE API 到 ES API 的转换：
       ```cpp
       // RopeCache 内部实现
       ge::es::EsTensorHolder GetSinEsTensor(ge::es::EsGraphBuilder& graph_builder) const {
           std::vector<int64_t> shape(final_shape.begin(), final_shape.end());
           const float* data = static_cast<const float*>(sin_final_buffer);
           size_t num_elements = final_size / sizeof(float);
           std::vector<float> value(data, data + num_elements);
           return graph_builder.CreateConst(value, shape);
       }
       ```
     - **核心转换机制（用于 RopeExtCustomV2）：**
       ```cpp
       // 获取底层 Graph
       ge::Graph* graph = graph_builder.GetCGraphBuilder()->GetGraph();
       
       // Operator → GNode → EsTensorHolder
       ge::GNode g_node = graph->AddNodeByOp(operator);
       EsCTensorHolder* es_c_tensor = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(g_node, output_index);
       ge::es::EsTensorHolder es_tensor(es_c_tensor);
       
       // EsTensorHolder → GNode（用于连接边）
       ge::GNode* g_node_ptr = es_tensor.GetProducer();
       graph->AddDataEdge(*g_node_ptr, src_port, dst_node, dst_port);
       ```
     - **实现步骤：**
       1. **RopeCache 的 sin/cos：** 直接调用 `rope_cache.GetSinEsTensor(graph_builder)` 和 `rope_cache.GetCosEsTensor(graph_builder)`，内部使用 `CreateConst` 创建常量
       2. **Squeeze 操作：** 使用 ES API 的 `Squeeze(es_src1, {0, 1, 2})` 函数
       3. **Gather 操作：** 使用 ES API 的 `GatherV2()` 函数，axis 参数通过 `graph_builder.CreateScalar(1)` 创建：
          ```cpp
          auto es_axis = graph_builder.CreateScalar(static_cast<int64_t>(1));
          auto es_gather_sin = GatherV2(es_sin_cache, es_x1_squeeze, es_axis);
          ```
       4. **RopeExtCustomV2：** 创建 `ge::op::RopeExtCustomV2` → `AddNodeByOp` → `AddDataEdge` 连接输入（x索引0, cos索引1, sin索引2）→ `GetTensorHolderFromNode`
       5. **后续 Transpose/Reshape：** 使用 ES API 的 `Transpose()` 和 `Reshape()` 函数
     - **关键优化点：**
       - RopeCache 直接使用 ES API 的 `CreateConst` 创建常量，避免了 Operator → GNode → EsTensorHolder 的转换步骤
       - 代码更简洁，性能更好
     - **TODO：**
       - `// TODO: 在同一个模型中多个RoPE也可能共享一个sin、cos缓存`
       - `// TODO:使用自定义算子的ES API来替换` RopeExtCustomV2 的 GE API 实现

4. **EsTensorHolder 的 SetPlacement 方法：**
   - 位置：`create_graph_input_tensor_es()`
   - TODO：`// TODO:EsTensorHolder提供SetPlacement方法`

---

## 阶段四：实现主函数 `build_ascend_graph_es()`

**目标：** 实现完整的 ES 版本图构建函数，使用 ES API 构建 Ascend 计算图

**实现要点：**

1. **创建 `EsGraphBuilder`：** 直接创建 `EsGraphBuilder graph_builder("Graph")`（不使用 unique_ptr）

2. **处理输入：** 使用 `process_input_tensors_es()` 处理叶子节点和 `GGML_OP_NONE` 节点

3. **处理算子：** 遍历 `cgraph->nodes`，根据 `node->op` 调用对应的 `handle_*_op_es()` 函数，维护 `ggml_tensor_to_es_tensor_map` 映射

4. **创建输出：** 实现 `create_output_tensors_es()` 辅助函数，使用 `GetProducerOutIndex()` 获取正确的输出索引

5. **构建图：** 使用 `graph_builder.Build(graph_outputs)` 一次性设置多个输出并构建图

**关键实现细节：**
- ✅ 支持所有 24 个算子类型（与 `build_ascend_graph()` 保持一致）
- ✅ 特殊算子处理：`GGML_OP_UNARY` 子类型、`GGML_OP_RMS_NORM_FUSED`/`GGML_OP_RMS_NORM` 保持原逻辑
- ✅ 使用 `EsTensorHolder::GetProducerOutIndex()` 获取输出索引（而非硬编码 0）

**状态：** ✅ 已完成

---

## 阶段四第二阶段：让 `build_ascend_graph()` 调用 `build_ascend_graph_es()`

**目标：** 将 `build_ascend_graph()` 改为调用 ES 版本，完成 API 切换

**实现方式：**
- 简化 `build_ascend_graph()` 函数，直接调用 `build_ascend_graph_es()`
- 保持函数签名不变，确保向后兼容

**状态：** ✅ 已完成

---

## 阶段四第三阶段：删除不需要的老的 GE 创建代码

**目标：** 清理不再使用的 GE API 辅助函数，简化代码

**删除的函数：**
- ✅ `create_graph_input_tensor()` (GE版本) - 已被 `create_graph_input_tensor_es()` 替代
- ✅ `create_output_tensors()` (GE版本) - 已被 `create_output_tensors_es()` 替代
- ✅ `process_input_tensors()` (GE版本) - 已被 `process_input_tensors_es()` 替代

**注意：** 
- GE版本的 `handle_*_op()` 函数删除
- `reuse_ascend_graph()` 函数保留（虽然大部分代码被注释，但函数签名仍在使用）

**状态：** ✅ 已完成

---

## 阶段五：测试和验证

1. **编译验证：** ✅ 确保所有代码能够成功编译
source /pkg/latest/bin/setenv.bash
cmake --build cmake-build-debug/ --target ggml-cann -j16

**状态：** 完成

2. **补充更多llt用例** 

**状态：：** TODO

---

## 文件修改清单

### 已修改的文件

1. **`ggml/src/ggml-cann/CMakeLists.txt`**
   - ✅ 添加 ES API 生成工具和库链接
   - ✅ 添加 `GE_FUNC_VISIBILITY` 宏定义修复

2. **`ggml/src/ggml-cann/ascend_graph.h`**
   - ✅ 添加 `build_ascend_graph_es()` 函数声明
   - ⚠️ **注意：** 没有添加前置声明和 `process_input_tensors_es()` 声明（这些在匿名 namespace 中实现）

3. **`ggml/src/ggml-cann/ascend_graph.cpp`**
   - ✅ 添加 `#include "es_all_ops.h"`
   - ✅ 在匿名 namespace 中实现 `create_graph_input_tensor_es()`
   - ✅ 在匿名 namespace 中实现 `process_input_tensors_es()`
   - ✅ 在匿名 namespace 中实现 `create_output_tensors_es()`（使用 `GetProducerOutIndex()`）
   - ✅ 实现 `build_ascend_graph_es()` 主函数（支持所有 24 个算子类型）
   - ✅ 修改 `build_ascend_graph()` 直接调用 `build_ascend_graph_es()`（阶段四第二阶段）
   - ✅ 删除不再使用的 GE 版本辅助函数：`create_graph_input_tensor()`、`create_output_tensors()`、`process_input_tensors()`（阶段四第三阶段）

4. **`ggml/src/ggml-cann/ascend_graph_ops.h`**
   - ✅ 添加 `#include "es_all_ops.h"`
   - ✅ 添加所有 24 个算子的 ES 版本函数声明

5. **`ggml/src/ggml-cann/ascend_graph_ops.cpp`**
   - ✅ 添加 `using namespace ge::es;`
   - ✅ 实现所有 24 个算子的 ES 版本

6. **`ggml/src/ggml-cann/ascend_graph_ops_create.h`**
   - ✅ 添加 `#include "es_all_ops.h"`
   - ✅ 添加 `create_moe_grouped_matmul_es()` 函数声明（ES API 版本）

7. **`ggml/src/ggml-cann/ascend_graph_ops_create.cpp`**
   - ✅ 添加 `#include "es_all_ops.h"`
   - ✅ 添加 `#include "es_c_graph_builder.h"`
   - ✅ 实现 `create_moe_grouped_matmul_es()` 函数

8. **`ggml/src/ggml-cann/rope_cache.h`**
   - ✅ 添加 `#include "es_graph_builder.h"`
   - ✅ 添加 `GetSinEsTensor()` 和 `GetCosEsTensor()` 方法声明

9. **`ggml/src/ggml-cann/rope_cache.cpp`**
   - ✅ 添加 `#include "es_graph_builder.h"`
   - ✅ 添加 `#include "es_tensor_holder.h"`
   - ✅ 实现 `GetSinEsTensor()` 和 `GetCosEsTensor()` 方法
   - ✅ 直接使用 `EsGraphBuilder::CreateConst()` 创建常量，避免 GE API 转换
   - ✅ 优化了 RoPE 算子的实现，代码更简洁

---

## 技术要点

### ES API 与 GE API 的主要区别

1. **图构建方式：**
   - GE API：使用 `ge::Graph` 和 `ge::Operator`，显式调用 `graph.AddOp()`
   - ES API：使用 `EsGraphBuilder` 和 `EsTensorHolder`，支持运算符重载

2. **算子创建：**
   - GE API：创建 `ge::op::*` 算子对象，设置属性，添加到图
   - ES API：直接调用函数（如 `Add()`, `Mul()`, `MatMul()`）或使用运算符重载

3. **张量表示：**
   - GE API：`ge::Operator` 表示算子，通过 `GetOutputDesc()` 获取张量描述
   - ES API：`EsTensorHolder` 直接表示张量，支持链式调用

4. **常量创建：**
   - GE API：需要创建 `op::Const` 算子，设置 `attr_value`，然后 `graph.AddOp()`
   - ES API：使用 `CreateConst()` 方法，直接传入数据和形状：
     ```cpp
     // ES API 方式：直接创建常量
     std::vector<float> value(data, data + size);
     std::vector<int64_t> shape = {1, 2, 3, 4};
     auto const_tensor = graph_builder.CreateConst(value, shape);
     ```
   - **优化示例：** RopeCache 类中添加了 `GetSinEsTensor()` 和 `GetCosEsTensor()` 方法，直接使用 `CreateConst` 创建常量，避免了 GE API 的转换步骤

5. **多输出算子的处理：**
   - GE API：对于多输出算子（如 Sort 返回 sorted values 和 indices），如果需要特定输出，需要创建额外的 Identity 算子来提取该输出
     ```cpp
     // GE API 方式：需要 Identity 算子
     ge::op::Sort sort_op(...);
     graph.AddOp(sort_op);
     ge::op::Identity identity_op(...);
     identity_op.set_input_x_by_name(sort_op, "y2");  // 提取第二个输出
     graph.AddOp(identity_op);
     return identity_op;
     ```
   - ES API：支持结构化绑定，可以直接获取多个输出，无需额外的拷贝操作
     ```cpp
     // ES API 方式：直接结构化绑定获取输出
     auto [y1, y2] = Sort(es_input, -1, descending);
     return y2.SetDataType(...).SetShape(...);  // 直接返回需要的输出
     ```
   - **优势：** ES API 的方式更简洁高效，避免了不必要的 Identity 算子，减少了图构建的开销

6. **GE 和 ES API 混合使用（自定义算子场景）：**
   - **场景：** 当遇到自定义算子（如 `RopeExtCustomV2`）时，ES API 可能没有对应的函数，需要混合使用 GE 和 ES API
   - **转换机制：**
     ```cpp
     // 1. 获取底层 Graph
     ge::Graph* graph = graph_builder.GetCGraphBuilder()->GetGraph();
     
     // 2. 创建 GE API 算子并设置属性
     ge::op::RopeExtCustomV2 rope_op(name.c_str());
     rope_op.set_attr_ne0(ne0);
     rope_op.set_attr_ne1(ne1);
     
     // 3. Operator → GNode
     ge::GNode g_node = graph->AddNodeByOp(rope_op);
     
     // 4. 连接输入边（从 EsTensorHolder 获取 GNode）
     ge::GNode* input_g_node = es_input.GetProducer();
     graph->AddDataEdge(*input_g_node, 0, g_node, 0);
     
     // 5. GNode → EsTensorHolder
     EsCTensorHolder* es_c_tensor = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(g_node, 0);
     ge::es::EsTensorHolder es_result(es_c_tensor);
     ```
   - **关键 API：**
     - `EsTensorHolder::GetProducer()` → `GNode*`（用于连接边）
     - `Graph::AddNodeByOp(Operator)` → `GNode`（添加算子到图）
     - `Graph::AddDataEdge(GNode&, src_port, GNode&, dst_port)`（连接数据边）
     - `EsCGraphBuilder::GetTensorHolderFromNode(GNode, output_index)` → `EsCTensorHolder*`（转换为 ES 张量）

7. 设置图的输入和输出

**GE API 方式：**
```cpp
std::vector<std::pair<ge::Operator, std::vector<size_t>>> indexed_graph_outputs;
for (const auto& op : graph_outputs) {
    indexed_graph_outputs.push_back({op, {0}});
}
graph.SetInputs(graph_inputs).SetOutputs(indexed_graph_outputs);
```

**ES API 方式**
- 使用 `graph_builder->SetOutput(es_tensor, output_index)` 设置输出
- 可以多次调用 `SetOutput()` 设置多个输出
- 也可以使用 `BuildAndReset({tensor1, tensor2, ...})` 一次性设置多个输出
- **输入不需要显式设置**：`CreateInput()` 创建的输入会自动成为图的输入

注： BuildAndReset的函数名之前命名为了Build, 最近才调整为了BuildAndReset以表达更清晰语义，当前重构使用的社区版本（12/01号版本仍然是Build命名）

**参考代码：**
```cpp
// 设置单个输出
graph_builder->SetOutput(result, 0);

// 设置多个输出（方式1：多次调用）
graph_builder->SetOutput(output1, 0);
graph_builder->SetOutput(output2, 1);

// 设置多个输出（方式2：BuildAndReset 时传递）
return graph_builder->BuildAndReset({output1, output2});
```

### 常见模式

1. **获取输入张量：**
   ```cpp
   ge::es::EsTensorHolder es_tensor;
   if (ggml_tensor_to_es_tensor_map.find(src) != 
       ggml_tensor_to_es_tensor_map.end()) {
       es_tensor = ggml_tensor_to_es_tensor_map[src];
   } else {
       assert(false && "tensor not found");
   }
   ```

2. **创建算子并设置输出的信息：**
   ```cpp
   return OperatorFunction(es_tensor, ...)
       .SetDataType(get_data_type(node->type))
       .SetShape(build_output_shape(node));
   ```

3. **创建常量：**
   ```cpp
   // 方式1：使用 CreateConst（推荐，用于大常量）
   std::vector<float> value(data, data + size);
   std::vector<int64_t> shape = {1, 2, 3, 4};
   auto const_tensor = graph_builder.CreateConst(value, shape);
   
   // 方式2：使用 CreateScalar（用于标量）
   auto scalar = graph_builder.CreateScalar(static_cast<int64_t>(1));
   
   // 方式3：使用 CreateVector（用于向量）
   auto vector = graph_builder.CreateVector(std::vector{value});
   ```

4. **处理多输出算子（使用结构化绑定）：**
   ```cpp
   // 对于返回多个输出的算子（如 Sort），使用结构化绑定直接获取
   auto [output1, output2] = MultiOutputOp(input, ...);
   return output2.SetDataType(...).SetShape(...);  // 返回需要的输出
   ```

---

