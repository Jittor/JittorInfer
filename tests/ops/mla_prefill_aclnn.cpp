#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>
#include "mla_prefill_attn_cpu.h"

const int max_graph_nodes = 128;

void build_mla_prefill_graph(    
    const std::vector<float> &query_host,
    const std::vector<float> &query_rope_host,
    const std::vector<float> &key_host,
    const std::vector<float> &key_rope_host,
    const std::vector<float> &value_host,
    const std::vector<int64_t> &qSeqLen_host,
    const std::vector<int64_t> &kvSeqLen_host,
    const std::vector<float> &mask_host,
    std::vector<float> &output_host,
    ggml_backend_t backend,
    int batchSize,
    int headNum,
    int kvHeadNum,
    int embeddim, 
    int embeddimV, 
    int maxSeqlen,
    float qkScale)
    {
        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * max_graph_nodes + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx);

        ggml_cgraph* gf = ggml_new_graph(ctx);

        int tokenNum = batchSize * maxSeqlen;

        // build graph
        ggml_tensor* query_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
            headNum * embeddimV, tokenNum);
        ggml_tensor* query_rope_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
            headNum * (embeddim - embeddimV), tokenNum);
        ggml_tensor* key_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 
            kvHeadNum * embeddimV, maxSeqlen, batchSize);
        ggml_tensor* key_rope_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 
            kvHeadNum * (embeddim - embeddimV), maxSeqlen, batchSize);
        ggml_tensor* value_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 
            kvHeadNum * embeddimV, maxSeqlen, batchSize);
        ggml_tensor* qSeqLen_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 
            batchSize);
        ggml_tensor* kvSeqLen_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 
            batchSize);
        ggml_tensor* mask_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
            512, 512);

        ggml_tensor* query_tensor_f16 = ggml_cast(ctx, query_tensor, GGML_TYPE_F16);
        ggml_tensor* query_rope_tensor_f16 = ggml_cast(ctx, query_rope_tensor, GGML_TYPE_F16);
        ggml_tensor* key_tensor_f16 = ggml_cast(ctx, key_tensor, GGML_TYPE_F16);
        ggml_tensor* key_rope_tensor_f16 = ggml_cast(ctx, key_rope_tensor, GGML_TYPE_F16);
        ggml_tensor* value_tensor_f16 = ggml_cast(ctx, value_tensor, GGML_TYPE_F16);
        ggml_tensor* mask_tensor_f16 = ggml_cast(ctx, mask_tensor, GGML_TYPE_F16);

        ggml_tensor* output_tensor = ggml_mla_prefill_jittor(
            ctx, query_tensor_f16, query_rope_tensor_f16, key_tensor_f16, key_rope_tensor_f16, value_tensor_f16, 
            qSeqLen_tensor, kvSeqLen_tensor, mask_tensor_f16, batchSize, headNum, kvHeadNum, embeddim, embeddimV, maxSeqlen, qkScale);
        
        output_tensor = ggml_cast(ctx, output_tensor, GGML_TYPE_F32);

        ggml_build_forward_expand(gf, output_tensor);

        // 分配空间
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buf);

        // 设置输入
        GGML_ASSERT(ggml_nbytes(query_tensor) == query_host.size() * sizeof(float));
        GGML_ASSERT(ggml_nbytes(query_rope_tensor) == query_rope_host.size() * sizeof(float));
        GGML_ASSERT(ggml_nbytes(key_tensor) == key_host.size() * sizeof(float));
        GGML_ASSERT(ggml_nbytes(key_rope_tensor) == key_rope_host.size() * sizeof(float));
        GGML_ASSERT(ggml_nbytes(value_tensor) == value_host.size() * sizeof(float));
        GGML_ASSERT(ggml_nbytes(qSeqLen_tensor) == qSeqLen_host.size() * sizeof(int64_t));
        GGML_ASSERT(ggml_nbytes(kvSeqLen_tensor) == kvSeqLen_host.size() * sizeof(int64_t));
        GGML_ASSERT(ggml_nbytes(mask_tensor) == mask_host.size() * sizeof(float));
        ggml_backend_tensor_set(query_tensor, query_host.data(), 0, ggml_nbytes(query_tensor));
        ggml_backend_tensor_set(query_rope_tensor, query_rope_host.data(), 0, ggml_nbytes(query_rope_tensor));
        ggml_backend_tensor_set(key_tensor, key_host.data(), 0, ggml_nbytes(key_tensor));
        ggml_backend_tensor_set(key_rope_tensor, key_rope_host.data(), 0, ggml_nbytes(key_rope_tensor));
        ggml_backend_tensor_set(value_tensor, value_host.data(), 0, ggml_nbytes(value_tensor));
        ggml_backend_tensor_set(qSeqLen_tensor, qSeqLen_host.data(), 0, ggml_nbytes(qSeqLen_tensor));
        ggml_backend_tensor_set(kvSeqLen_tensor, kvSeqLen_host.data(), 0, ggml_nbytes(kvSeqLen_tensor));
        ggml_backend_tensor_set(mask_tensor, mask_host.data(), 0, ggml_nbytes(mask_tensor));

        // 执行计算
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        // 获取输出
        output_host.resize(ggml_nelements(output_tensor));
        ggml_backend_tensor_get(output_tensor, output_host.data(), 0, ggml_nbytes(output_tensor));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

int main() {
    std::cout << "Starting MLA Prefill test..." << std::endl;

    // Define tensor dimensions

    int batchSize = 1;
    int maxSeqlen = 512;
    int headNum = 1;
    int kvHeadNum = 1;
    int embeddim = 192;
    int embeddimV = 128;
    float qkScale = 0.1352667747812271;
    int tokenNum = batchSize * maxSeqlen;

    std::cout << "Dimensions: batch_size=" << batchSize 
              << ", max_seq_len=" << maxSeqlen 
              << ", token_num=" << tokenNum 
              << ", head_num=" << headNum 
              << ", kv_head_num=" << kvHeadNum 
              << ", embeddim=" << embeddim 
              << ", embeddimV=" << embeddimV << std::endl;

    // Calculate sizes for tensors
    int64_t query_size = tokenNum * headNum * embeddimV;
    int64_t query_rope_size = tokenNum * headNum * 64;
    int64_t key_size = batchSize * maxSeqlen * kvHeadNum * embeddimV;
    int64_t key_rope_size = batchSize * maxSeqlen * kvHeadNum * 64;
    int64_t value_size = batchSize * maxSeqlen * kvHeadNum * embeddimV;
    int64_t output_size = tokenNum * headNum * embeddimV;

    std::cout << "Allocating memory for tensors..." << std::endl;
    std::cout << "Query size: " << query_size << std::endl;
    std::cout << "Query Rope size: " << query_rope_size << std::endl;
    std::cout << "Key size: " << key_size << std::endl;
    std::cout << "Key Rope size: " << key_rope_size << std::endl;
    std::cout << "Value size: " << value_size << std::endl;
    std::cout << "Output size: " << output_size << std::endl;

    // Initialize tensors
    std::vector<float> query_host(query_size);
    std::vector<float> query_rope_host(query_rope_size);
    std::vector<float> key_host(key_size);
    std::vector<float> key_rope_host(key_rope_size);
    std::vector<float> value_host(value_size);
    std::vector<int64_t> qSeqLen_host(batchSize, maxSeqlen);
    std::vector<int64_t> kvSeqLen_host(batchSize, maxSeqlen);
    std::vector<float> mask_host(512 * 512);

    // Random initialization
    std::cout << "Initializing tensors with random values..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.1, 0.1);  // Smaller range for numerical stability
    
    for (auto &val : query_host) { val = dis(gen); }
    for (auto &val : query_rope_host) { val = dis(gen); }
    for (auto &val : key_host) { val = dis(gen); }
    for (auto &val : key_rope_host) { val = dis(gen); }
    for (auto &val : value_host) { val = dis(gen); }

    const float zero = float(0.0f);
    const float neg_inf = -INFINITY;
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j <= i; ++j) {
            mask_host[i * 512 + j] = zero;
        }
        for (int j = i + 1; j < 512; ++j) {
            mask_host[i * 512 + j] = neg_inf;
        }
    }

    std::vector<float> output_host(output_size);
    memset(output_host.data(), 0, output_size * sizeof(float));

    mla_prefill_using_attention_cpu(query_host, query_rope_host, key_host, key_rope_host,
    value_host, mask_host, output_host, batchSize, maxSeqlen, maxSeqlen, headNum, kvHeadNum, qkScale, 128, 64);

    std::cout << "CPU output tensor sample (first few values):" << std::endl;
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
    std::cout << "Calling MLA Prefill function..." << std::endl;
    build_mla_prefill_graph(
        query_host, query_rope_host, key_host, key_rope_host, value_host,
        qSeqLen_host, kvSeqLen_host, mask_host, output_host_cann, cann_backend,
        batchSize, headNum, kvHeadNum, embeddim, embeddimV, maxSeqlen, qkScale);

    std::cout << "MLA Prefill completed successfully." << std::endl;
    std::cout << "Cann output tensor sample (first few values):" << std::endl;

    // Print only a small sample of the output to avoid flooding the console
    for (int i = 0; i < std::min(static_cast<int64_t>(10), output_size); i++) {
        std::cout << std::fixed << std::setprecision(4) << static_cast<float>(output_host_cann[i]) << " ";
    }

    std::cout << std::endl;

    std::cout << "\nComparing CPU and CANN implementations:" << std::endl;
    
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    int max_abs_error_idx = 0;
    int max_rel_error_idx = 0;

    for (int64_t i = 0; i < output_size; ++i) {
        double abs_error = std::fabs(output_host[i] - output_host_cann[i]);
        double rel_error = 0.0;
        
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

