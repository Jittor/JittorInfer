#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include "flash_attn_cpu.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>

const int max_graph_nodes = 128;

void build_flash_attention_graph(
    const std::vector<float> &query_host,
    const std::vector<float> &key_host,
    const std::vector<float> &value_host,
    const std::vector<int8_t> &attn_mask_host,
    std::vector<float> &output_host,
    ggml_backend_t backend,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dims_kq,
    int64_t head_dims_v,
    int64_t key_num_heads,
    int64_t sequence_lenth_q,
    int64_t sequence_lenth_kv,
    float scaleValue
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
    ggml_tensor* q_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
        head_dims_kq, sequence_lenth_q, num_heads, batch_size);
    // 键张量: [batch_size, key_num_heads, sequence_lenth_kv, head_dims]
    ggml_tensor* k_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
        head_dims_kq, sequence_lenth_kv, key_num_heads, batch_size);
    // 值张量: [batch_size, key_num_heads, sequence_lenth_kv, head_dims]
    ggml_tensor* v_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
        head_dims_v, sequence_lenth_kv, key_num_heads, batch_size);

    ggml_tensor* q_tensor_f16 = ggml_cast(ctx, q_tensor, GGML_TYPE_F16);
    ggml_tensor* k_tensor_f16 = ggml_cast(ctx, k_tensor, GGML_TYPE_F16);
    ggml_tensor* v_tensor_f16 = ggml_cast(ctx, v_tensor, GGML_TYPE_F16);

    // 掩码张量: [batch_size, 1, sequence_lenth_q, sequence_lenth_kv]
    ggml_tensor* mask_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_I8, 
    sequence_lenth_kv, sequence_lenth_q, 1, batch_size);

    // 输出张量: [batch_size, num_heads, sequence_lenth_q, head_dims]
    ggml_tensor* output_tensor = ggml_flash_attn_jittor_v1(
        ctx, q_tensor_f16, k_tensor_f16, v_tensor_f16, mask_tensor, batch_size, num_heads,
        head_dims_kq, head_dims_v, key_num_heads, sequence_lenth_q,
        sequence_lenth_kv, nullptr, nullptr, scaleValue);
    
    output_tensor = ggml_cast(ctx, output_tensor, GGML_TYPE_F32);

    ggml_build_forward_expand(gf, output_tensor);

    // 分配空间
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);

    // 设置输入
    GGML_ASSERT(ggml_nbytes(q_tensor) == query_host.size() * sizeof(float));
    GGML_ASSERT(ggml_nbytes(k_tensor) == key_host.size() * sizeof(float));
    GGML_ASSERT(ggml_nbytes(v_tensor) == value_host.size() * sizeof(float));
    GGML_ASSERT(ggml_nbytes(mask_tensor) == attn_mask_host.size() * sizeof(int8_t));
    ggml_backend_tensor_set(q_tensor, query_host.data(), 0, ggml_nbytes(q_tensor));
    ggml_backend_tensor_set(k_tensor, key_host.data(), 0, ggml_nbytes(k_tensor));
    ggml_backend_tensor_set(v_tensor, value_host.data(), 0, ggml_nbytes(v_tensor));
    ggml_backend_tensor_set(mask_tensor, attn_mask_host.data(), 0, ggml_nbytes(mask_tensor));

    // 执行计算
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    // 获取输出
    output_host.resize(ggml_nelements(output_tensor));
    ggml_backend_tensor_get(output_tensor, output_host.data(), 0, ggml_nbytes(output_tensor));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    std::cout << "Starting Flash Attention test..." << std::endl;

    // Define tensor dimensions
    int64_t batch_size = 1;
    int64_t num_heads = 16;  // Reduced from 128 to a smaller value

    int64_t head_dims_kq = 192;  // Reduced from 192 to a smaller value  // 576 
    int64_t head_dims_v = 192;  // Reduced from 192 to a smaller value   // 512
    int64_t key_num_heads = 16;                                   // 1

    int64_t sequence_lenth_q = 4;
    int64_t sequence_lenth_kv = 2048;
    float scaleValue = 1.0f / std::sqrt(static_cast<float>(76));

    std::cout << "Dimensions: batch_size=" << batch_size 
              << ", num_heads=" << num_heads 
              << ", head_dims_kq=" << head_dims_kq 
              << ", head_dims_v=" << head_dims_v 
              << ", seq_q=" << sequence_lenth_q 
              << ", seq_kv=" << sequence_lenth_kv << std::endl;

    // Calculate sizes for tensors
    int64_t query_size = batch_size * num_heads * sequence_lenth_q * head_dims_kq;
    int64_t key_size = batch_size * key_num_heads * sequence_lenth_kv * head_dims_kq;
    int64_t value_size = batch_size * key_num_heads * sequence_lenth_kv * head_dims_v;
    int64_t attn_mask_size = batch_size * 1 * sequence_lenth_q * sequence_lenth_kv;
    int64_t output_size = batch_size * num_heads * sequence_lenth_q * head_dims_v;

    std::cout << "Allocating memory for tensors..." << std::endl;
    std::cout << "Query size: " << query_size << std::endl;
    std::cout << "Key size: " << key_size << std::endl;
    std::cout << "Value size: " << value_size << std::endl;
    std::cout << "Attention mask size: " << attn_mask_size << std::endl;
    std::cout << "Output size: " << output_size << std::endl;

    // Initialize tensors
    std::vector<float> query_host(query_size);
    std::vector<float> key_host(key_size);
    std::vector<float> value_host(value_size);
    std::vector<int8_t> attn_mask(attn_mask_size, 0);  // Initialize all to 0
    
    // 随机设置一些 attn_mask 元素为 1（表示需要被屏蔽的位置）
    // 使用与初始化张量相同的随机数生成器
    std::random_device mask_rd;
    std::mt19937 mask_gen(mask_rd());
    std::bernoulli_distribution mask_dist(0.3);  // 30% 的概率设置为 1
    for (int64_t i = 0; i < attn_mask_size; ++i) {
        attn_mask[i] = mask_dist(mask_gen) ? 1 : 0;
    }
    
    // 打印部分 attn_mask 信息
    std::cout << "Attention mask sample:" << std::endl;
    for (int i = 0; i < std::min(static_cast<int64_t>(16), static_cast<int64_t>(attn_mask_size)); ++i) {
        std::cout << static_cast<int>(attn_mask[i]) << " ";
    }
    std::cout << std::endl;
    
    std::vector<float> output_host(output_size);

    // Random initialization
    std::cout << "Initializing tensors with random values..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.1, 0.1);  // Smaller range for numerical stability
    
    for (auto &val : query_host) { val = dis(gen); }
    for (auto &val : key_host) { val = dis(gen); }
    for (auto &val : value_host) { val = dis(gen); }

    memset(output_host.data(), 0, output_size * sizeof(float));
    flash_attn_cpu(query_host, key_host, value_host, attn_mask, output_host,
        batch_size, num_heads, head_dims_kq, head_dims_v, key_num_heads,
        sequence_lenth_q, sequence_lenth_kv, scaleValue);

    for (int i = 0; i < std::min(static_cast<int64_t>(10), output_size); i++) {
        std::cout << std::fixed << std::setprecision(4) << static_cast<float>(output_host[i]) << " ";
    }
    std::cout << std::endl;
    std::vector<float> output_host_cann(output_size);
    memset(output_host_cann.data(), 0, output_size * sizeof(float));

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

    // Call flash attention function
    std::cout << "Calling flash attention function..." << std::endl;
    build_flash_attention_graph(
        query_host, key_host, value_host, attn_mask, output_host_cann,
        cann_backend, batch_size, num_heads, head_dims_kq, head_dims_v, key_num_heads,
        sequence_lenth_q, sequence_lenth_kv, scaleValue);

    std::cout << "Flash attention completed successfully." << std::endl;
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
        double abs_error = std::fabs(output_host[i] - output_host_cann[i]);
        double rel_error = 0.0;
        
        // 计算相对误差，避免除以0
        if (std::fabs(output_host[i]) > 1e-10) {
            rel_error = abs_error / std::fabs(output_host[i]);
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
              << " (CPU: " << output_host[max_abs_error_idx] 
              << ", CANN: " << output_host_cann[max_abs_error_idx] << ")" << std::endl;
              
    std::cout << "Maximum relative error: " << max_rel_error 
              << " at index " << max_rel_error_idx 
              << " (CPU: " << output_host[max_rel_error_idx] 
              << ", CANN: " << output_host_cann[max_rel_error_idx] << ")" << std::endl;
    
    return 0;
}

