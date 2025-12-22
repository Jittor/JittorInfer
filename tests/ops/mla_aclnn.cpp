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
#include "cached_attn_cpu.h"

const int max_graph_nodes = 128;

void build_mla_graph(    
    const std::vector<float> &query_host,
    const std::vector<float> &query_rope_host,
    const std::vector<float> &ctKV_host,
    const std::vector<float> &kRope_host,
    const std::vector<int32_t> &blockTables_host,
    const std::vector<int64_t> &contextLens_host,
    const std::vector<float> &mask_host,
    std::vector<float> &output_host,
    ggml_backend_t backend,
    int batchSize,
    int tokenNum,
    int headNum,
    int kvHeadNum,
    int kSeqLen,
    float qkScale,
    int blockSize)
    {
        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * max_graph_nodes + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx);

        ggml_cgraph* gf = ggml_new_graph(ctx);

        int maxBlockNumPerSeq = (kSeqLen + blockSize - 1) / blockSize;
        int blockNum = tokenNum * maxBlockNumPerSeq;
        
        // build graph
        ggml_tensor* query_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 
            512, headNum, tokenNum);
        ggml_tensor* query_rope_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 
            64, headNum, tokenNum);
        ggml_tensor* ctKV_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
            512, kvHeadNum, blockSize, blockNum);
        ggml_tensor* key_rope_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
            64, kvHeadNum, blockSize, blockNum);
        ggml_tensor* block_table_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 
            maxBlockNumPerSeq, batchSize);
        ggml_tensor* contextLen_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 
            batchSize);
        ggml_tensor* mask_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
            kSeqLen, tokenNum);

        ggml_tensor* query_tensor_f16 = ggml_cast(ctx, query_tensor, GGML_TYPE_F16);
        ggml_tensor* query_rope_tensor_f16 = ggml_cast(ctx, query_rope_tensor, GGML_TYPE_F16);
        ggml_tensor* ctKV_tensor_f16 = ggml_cast(ctx, ctKV_tensor, GGML_TYPE_F16);
        ggml_tensor* kRope_tensor_f16 = ggml_cast(ctx, key_rope_tensor, GGML_TYPE_F16);
        ggml_tensor* mask_tensor_f16 = ggml_cast(ctx, mask_tensor, GGML_TYPE_F16);

        ggml_tensor* output_tensor = ggml_mla_jittor(
            ctx, query_tensor_f16, query_rope_tensor_f16, ctKV_tensor_f16, kRope_tensor_f16, block_table_tensor, contextLen_tensor, mask_tensor_f16,
            batchSize, tokenNum, headNum, kvHeadNum, kSeqLen, qkScale, blockSize);
        
        output_tensor = ggml_cast(ctx, output_tensor, GGML_TYPE_F32);

        ggml_build_forward_expand(gf, output_tensor);

        // 分配空间
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buf);

        // 设置输入
        GGML_ASSERT(ggml_nbytes(query_tensor) == query_host.size() * sizeof(float));
        GGML_ASSERT(ggml_nbytes(query_rope_tensor) == query_rope_host.size() * sizeof(float));
        GGML_ASSERT(ggml_nbytes(ctKV_tensor) == ctKV_host.size() * sizeof(float));
        GGML_ASSERT(ggml_nbytes(key_rope_tensor) == kRope_host.size() * sizeof(float));
        GGML_ASSERT(ggml_nbytes(block_table_tensor) == blockTables_host.size() * sizeof(int32_t));
        GGML_ASSERT(ggml_nbytes(contextLen_tensor) == contextLens_host.size() * sizeof(int64_t));
        GGML_ASSERT(ggml_nbytes(mask_tensor) == mask_host.size() * sizeof(float));
        ggml_backend_tensor_set(query_tensor, query_host.data(), 0, ggml_nbytes(query_tensor));
        ggml_backend_tensor_set(query_rope_tensor, query_rope_host.data(), 0, ggml_nbytes(query_rope_tensor));
        ggml_backend_tensor_set(ctKV_tensor, ctKV_host.data(), 0, ggml_nbytes(ctKV_tensor));
        ggml_backend_tensor_set(key_rope_tensor, kRope_host.data(), 0, ggml_nbytes(key_rope_tensor));
        ggml_backend_tensor_set(block_table_tensor, blockTables_host.data(), 0, ggml_nbytes(block_table_tensor));
        ggml_backend_tensor_set(contextLen_tensor, contextLens_host.data(), 0, ggml_nbytes(contextLen_tensor));
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
    std::cout << "Starting MLA test..." << std::endl;

    // Define tensor dimensions

    // tokenNum = batchSize ?
    int batchSize = 1;
    int tokenNum = 1;

    int headNum = 16;
    int kSeqLen = 526;
    const uint32_t blockSize = 128;
    int kvHeadNum = 1;
    float qkScale = 0.1352667747812271;
    int maxBlockNumPerSeq = (kSeqLen + blockSize - 1) / blockSize;
    int blockNum = tokenNum * maxBlockNumPerSeq;

    std::cout << "Dimensions: batch_size=" << batchSize 
              << ", token_num=" << tokenNum 
              << ", head_num=" << headNum 
              << ", kv_head_num=" << kvHeadNum 
              << ", key_seq_len=" << kSeqLen 
              << ", block_num=" << blockNum << std::endl;

    // Calculate sizes for tensors
    int64_t query_size = tokenNum * headNum * 512;
    int64_t query_rope_size = tokenNum * headNum * 64;
    int64_t ctKV_size = blockNum * blockSize * kvHeadNum * 512;
    int64_t kRope_size = blockNum * blockSize * kvHeadNum * 64;
    int64_t blockTables_size = batchSize * maxBlockNumPerSeq;
    int64_t contextLens_size = batchSize;
    int64_t mask_size = tokenNum * kSeqLen;
    int64_t output_size = tokenNum * headNum * 512;

    std::cout << "Allocating memory for tensors..." << std::endl;
    std::cout << "Query size: " << query_size << std::endl;
    std::cout << "Query Rope size: " << query_rope_size << std::endl;
    std::cout << "Context KV size: " << ctKV_size << std::endl;
    std::cout << "Key Rope size: " << kRope_size << std::endl;
    std::cout << "Block Table size: " << blockTables_size << std::endl;
    std::cout << "Context Length size: " << contextLens_size << std::endl;
    std::cout << "Attention mask size: " << mask_size << std::endl;
    std::cout << "Output size: " << output_size << std::endl;

    // Initialize tensors
    std::vector<float> query_host(query_size);
    std::vector<float> query_rope_host(query_rope_size);
    std::vector<float> ctKV_host(ctKV_size);
    std::vector<float> kRope_host(kRope_size);
    std::vector<int32_t> blockTables_host(blockTables_size);
    std::vector<int64_t> contextLens_host(contextLens_size, kSeqLen);
    std::vector<float> mask_host(mask_size);

    // Random initialization
    std::cout << "Initializing tensors with random values..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.1, 0.1);  // Smaller range for numerical stability
    
    for (auto &val : query_host) { val = dis(gen); }
    for (auto &val : query_rope_host) { val = dis(gen); }
    for (auto &val : ctKV_host) { val = dis(gen); }
    for (auto &val : kRope_host) { val = dis(gen); }
    for (auto &val : mask_host) { val = 0; }

    // 这样的布局使得PageAttention KVCache和原生KVCache一致
    for (size_t i = 0; i < batchSize; i++) {
        for (size_t j = 0; j < maxBlockNumPerSeq; j++) {
            blockTables_host[i * maxBlockNumPerSeq + j] = i * maxBlockNumPerSeq + j;
        }
    }

    std::vector<float> output_host(output_size);
    memset(output_host.data(), 0, output_size * sizeof(float));

    cached_attention_cpu(query_host, query_rope_host, ctKV_host, kRope_host, blockTables_host, contextLens_host, 
        output_host, tokenNum, headNum, kvHeadNum, 576, 512, blockSize, maxBlockNumPerSeq, qkScale);

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
    std::cout << "Calling MLA function..." << std::endl;
    build_mla_graph(
        query_host, query_rope_host, ctKV_host, kRope_host, blockTables_host, contextLens_host, mask_host, output_host_cann, 
        cann_backend,batchSize, tokenNum, headNum, kvHeadNum, kSeqLen, qkScale, blockSize);

    std::cout << "MLA completed successfully." << std::endl;
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

