
#include "rope_cpu.h"


void build_rope_graph_cpu(
    const std::vector<float> &input_host,
    const std::vector<int32_t> &positions_host,
    std::vector<float> &output_host,
    ggml_backend_t backend,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dims,
    int64_t sequence_length,
    int n_dims,
    int mode
) {
    std::cout << "Running CPU RoPE implementation..." << std::endl;

    // 创建CPU上下文
    size_t mem_size = 256 * 1024 * 1024; // 256MB
    ggml_init_params cpu_params = {
        /* .mem_size = */ mem_size,
        /* .mem_base = */ NULL,
        /* .no_alloc = */ false,  // 允许分配
    };
    ggml_context* cpu_ctx = ggml_init(cpu_params);

    // 构建CPU计算图
    ggml_cgraph* cpu_gf = ggml_new_graph(cpu_ctx);

    // 创建输入张量
    ggml_tensor* cpu_input = ggml_new_tensor_4d(cpu_ctx, GGML_TYPE_F32, 
        head_dims, sequence_length, num_heads, batch_size);
        
    // 创建位置张量  
    ggml_tensor* cpu_positions = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, num_heads);

    // 设置数据
    memcpy(cpu_input->data, input_host.data(), ggml_nbytes(cpu_input));
    memcpy(cpu_positions->data, positions_host.data(), ggml_nbytes(cpu_positions));

    // 调用GGML的RoPE函数
    ggml_tensor* cpu_output = ggml_rope(cpu_ctx, cpu_input, cpu_positions, n_dims, mode);

    ggml_build_forward_expand(cpu_gf, cpu_output);

    // 执行CPU计算
    GGML_ASSERT(ggml_backend_graph_compute(backend, cpu_gf) == GGML_STATUS_SUCCESS);

    // 获取结果
    output_host.resize(ggml_nelements(cpu_output));
    memcpy(output_host.data(), cpu_output->data, ggml_nbytes(cpu_output));

    ggml_free(cpu_ctx);
}
