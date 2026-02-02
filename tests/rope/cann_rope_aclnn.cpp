#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include "rope_cpu.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>

const int max_graph_nodes = 128;

void build_rope_graph_cann(
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
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * max_graph_nodes + ggml_graph_overhead(),
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };
    ggml_context* ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    ggml_cgraph* gf = ggml_new_graph(ctx);

    // build graph
    // 输入张量: [batch_size, num_heads, sequence_length, head_dims]
    ggml_tensor* input_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
        head_dims, sequence_length, num_heads, batch_size);
    
    // 频率张量: [sequence_length, n_dims/2]
    ggml_tensor* positions_tensor  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, num_heads);

    // 输出张量: [batch_size, num_heads, sequence_length, head_dims]
    ggml_tensor* output_tensor = ggml_rope(
        ctx, input_tensor, positions_tensor , n_dims, mode);
    
    ggml_build_forward_expand(gf, output_tensor);

    // 分配空间
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);

    // 设置输入
    // 设置输入 - 移除断言检查，直接设置数据
    std::cout << "Input tensor bytes: " << ggml_nbytes(input_tensor) 
              << ", Host data bytes: " << input_host.size() * sizeof(float) << std::endl;
    std::cout << "Positions tensor bytes: " << ggml_nbytes(positions_tensor) 
              << ", Host data bytes: " << positions_host.size() * sizeof(int32_t) << std::endl;
    ggml_backend_tensor_set(input_tensor, input_host.data(), 0, ggml_nbytes(input_tensor));
    ggml_backend_tensor_set(positions_tensor, positions_host.data(), 0, ggml_nbytes(positions_tensor));

    // 执行计算
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    // 获取输出
    output_host.resize(ggml_nelements(output_tensor));
    ggml_backend_tensor_get(output_tensor, output_host.data(), 0, ggml_nbytes(output_tensor));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}


int main() {
    std::cout << "Starting RoPE test..." << std::endl;
    printf("%d\n",mode);

    // Define tensor dimensions
    int64_t batch_size = 1;
    int64_t num_heads = 16;
    int64_t head_dims = 256;
    int64_t sequence_length = 64;
    int n_dims = 256;  // RoPE dimensions
    int mode = 2;     // RoPE mode

    std::cout << "Dimensions: batch_size=" << batch_size 
            << ", num_heads=" << num_heads 
            << ", head_dims=" << head_dims 
            << ", seq_length=" << sequence_length 
            << ", n_dims=" << n_dims 
            << ", mode=" << mode << std::endl;

    // Calculate sizes for tensors
    int64_t input_size = batch_size * num_heads * sequence_length * head_dims;
    int64_t positions_size = num_heads;  // 每个头一个位置索引
    int64_t output_size = input_size;  // Same shape as input

    std::cout << "Allocating memory for tensors..." << std::endl;
    std::cout << "Input size: " << input_size << std::endl;
    std::cout << "Positions size: " << positions_size << " elements" << std::endl;
    std::cout << "Output size: " << output_size << std::endl;

    // Initialize tensors
    std::vector<float> input_host(input_size);
    std::vector<int32_t> positions_host(positions_size);
    std::vector<float> output_host_cpu(output_size);
    std::vector<float> output_host_cann(output_size);

    // Random initialization
    std::cout << "Initializing tensors with random values..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.1, 0.1);
    
    for (auto &val : input_host) { val = dis(gen); }
    
    // 初始化位置信息：每个头一个位置值
    for (int64_t i = 0; i < num_heads; ++i) {
        // 使用位置索引，从0开始
        positions_host[i] = i;  // 或者使用其他有意义的序列
    }
    
    std::cout << "Input sample (first few values):" << std::endl;
    for (int i = 0; i < std::min(static_cast<int64_t>(10), input_size); i++) {
        std::cout << std::fixed << std::setprecision(4) << static_cast<float>(input_host[i]) << " ";
    }
    std::cout << std::endl;

    int num_devices = ggml_backend_dev_count();

    ggml_backend_dev_t cann_dev = nullptr;
    for (int i = 0; i < num_devices; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::cout << "Device " << i << " name: " << ggml_backend_dev_name(dev) << std::endl;
        if (std::string(ggml_backend_dev_name(dev)).find("CANN") != std::string::npos) {
            std::cout << "CANN device found" << std::endl;
            cann_dev = dev;
            break;
        }
    }
    if (cann_dev == nullptr) {
        std::cerr << "CANN device not found" << std::endl;
        return 1;
    }

    ggml_backend_t cann_backend = ggml_backend_dev_init(cann_dev, NULL);
    GGML_ASSERT(cann_backend != NULL);

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cann_dev);

    printf("  Device description: %s\n", ggml_backend_dev_description(cann_dev));
    size_t free, total;  // NOLINT
    ggml_backend_dev_memory(cann_dev, &free, &total);
    printf("  Device memory: %zu MB (%zu MB free)\n", total / 1024 / 1024, free / 1024 / 1024);
    printf("\n");

    // Call RoPE function
    std::cout << "Calling RoPE function..." << std::endl;
    build_rope_graph_cann(
        input_host, positions_host, output_host_cann,
        cann_backend, batch_size, num_heads, head_dims, sequence_length,
        n_dims, mode);
    
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    build_rope_graph_cpu(
        input_host, positions_host, output_host_cpu,
        cpu_backend, batch_size, num_heads, head_dims, sequence_length,
        n_dims, mode);
    std::cout << "RoPE completed successfully." << std::endl;
    std::cout << "Output tensor sample (first few values):" << std::endl;

    // Print only a small sample of the output to avoid flooding the console
    for (int i = 0; i < std::min(static_cast<int64_t>(10), output_size); i++) {
        std::cout << std::fixed << std::setprecision(4) << static_cast<float>(output_host_cann[i]) << " ";
    }
    std::cout << std::endl;

    // 比较 CPU 和 CANN 实现的结果
    std::cout << "\nComparing CPU and CANN implementations:" << std::endl;
    
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    int max_abs_error_idx = 0;
    int max_rel_error_idx = 0;
    
    for (int64_t i = 0; i < output_size; ++i) {
        double abs_error = std::fabs(output_host_cpu[i] - output_host_cann[i]);
        double rel_error = 0.0;
        
        // 计算相对误差，避免除以0
        if (std::fabs(output_host_cpu[i]) > 1e-10) {
            rel_error = abs_error / std::fabs(output_host_cpu[i]);
        } else if (std::fabs(output_host_cann[i]) > 1e-10) {
            rel_error = abs_error / std::fabs(output_host_cann[i]);
        }
        
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            max_abs_error_idx = i;
        }
        
        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
            max_rel_error_idx = i;
        }
    }
    
    std::cout << "Maximum absolute error: " << max_abs_error 
            << " at index " << max_abs_error_idx 
            << " (CPU: " << output_host_cpu[max_abs_error_idx] 
            << ", CANN: " << output_host_cann[max_abs_error_idx] << ")" << std::endl;
    std::cout << "Maximum relative error: " << max_rel_error 
            << " at index " << max_rel_error_idx 
            << " (CPU: " << output_host_cpu[max_rel_error_idx] 
            << ", CANN: " << output_host_cann[max_rel_error_idx] << ")" << std::endl;
    
    return 0;
}

