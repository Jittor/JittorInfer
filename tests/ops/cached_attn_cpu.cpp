#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include "cached_attn_cpu.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>

void mla_attn_cpu(
    const std::vector<float> &query,
    const std::vector<float> &key,
    const std::vector<float> &value,
    const std::vector<float> &attn_mask,
    std::vector<float> &output,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dims_kq,
    int64_t head_dims_v,
    int64_t key_num_heads,
    int64_t sequence_lenth_q,
    int64_t sequence_lenth_kv,
    float scaleValue,
    int64_t kv_context_len,
    const std::string &layerOut
) {
    // 初始化输出向量
    output.resize(batch_size * num_heads * sequence_lenth_q * head_dims_v, 0.0f);
    
    // 使用类似wkv_b_post_process的方式实现
    struct ggml_init_params init_params = {
        /* .mem_size */   ggml_tensor_overhead() * 32 + 1024 * 1024 * 128,  // 分配足够的内存
        /* .mem_buffer */ NULL,
        /* .no_alloc */   false
    };
    
    // 创建GGML上下文
    ggml_context* ctx = ggml_init(init_params);
    if (!ctx) {
        std::cerr << "Failed to initialize GGML context" << std::endl;
        return;
    }
    
    // 创建输入张量
    // 查询张量: [batch_size, num_heads, sequence_lenth_q, head_dims]
    GGML_ASSERT(layerOut == "BNSD");
    ggml_tensor* q_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                            head_dims_kq, sequence_lenth_q, num_heads, batch_size);
    // 键张量: [batch_size, key_num_heads, sequence_lenth_kv, head_dims]
    ggml_tensor* k_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                            head_dims_kq, sequence_lenth_kv, key_num_heads, batch_size);
    // 值张量: [batch_size, key_num_heads, sequence_lenth_kv, head_dims]
    ggml_tensor* v_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                            head_dims_v, sequence_lenth_kv, key_num_heads, batch_size);
    
    // 输出张量: [batch_size, num_heads, sequence_lenth_q, head_dims]
    ggml_tensor* output_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                head_dims_v, sequence_lenth_q, num_heads, batch_size);
    // 复制数据到张量
    memcpy(q_tensor->data, query.data(), query.size() * sizeof(float));
    memcpy(k_tensor->data, key.data(), key.size() * sizeof(float));
    memcpy(v_tensor->data, value.data(), value.size() * sizeof(float));

    ggml_tensor* mask_tensor = nullptr;
    bool use_mask = false;

    if (kv_context_len != 0 || !attn_mask.empty()) {
        use_mask = true;

        // 掩码张量: [batch_size, 1, sequence_lenth_q, sequence_lenth_kv]
        mask_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                        sequence_lenth_kv, sequence_lenth_q, 1, batch_size);
        if(kv_context_len == 0) {
            std::vector<float> mask_data(attn_mask.size());
            for (size_t i = 0; i < attn_mask.size(); ++i) {
                mask_data[i] = attn_mask[i] ? -INFINITY : 0.0f;
            }
            memcpy(mask_tensor->data, mask_data.data(), attn_mask.size() * sizeof(float));   
        }

        else {
            int64_t mask_needed = batch_size * sequence_lenth_q * sequence_lenth_kv;
            std::vector<float> mask_data(mask_needed, -INFINITY); // 默认全部禁止
            
            for (int64_t b = 0; b < batch_size; ++b) {
                for (int64_t q = 0; q < sequence_lenth_q; ++q) {
                    // 当前查询位置在全局序列中的索引（假设是递增的）
                    int64_t current_query_global_idx = q; // 这里需要根据实际逻辑调整
                    for (int64_t k = 0; k < sequence_lenth_kv; ++k) {
                        int64_t idx = (b * sequence_lenth_q + q) * sequence_lenth_kv + k;
                        // 允许关注的条件: 
                        // 1. 键位置k在有效长度内 (k < kv_context_len)
                        // 2. 并且满足因果性 (k <= current_query_global_idx)
                        if (k < kv_context_len && k <= current_query_global_idx) {
                            mask_data[idx] = 0.0f; // 允许关注
                        }
                        // 否则保持 -INFINITY (禁止关注)
                    }
                }
            }  
            memcpy(mask_tensor->data, mask_data.data(), mask_needed * sizeof(float));
        }
    }
    
    // 构建计算图
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 100, false);
    
    ggml_tensor* q_cur = q_tensor;
    
    // 2. 计算 Q*K^T
    ggml_tensor* kq = ggml_mul_mat(ctx, k_tensor, q_cur);  // 矩阵乘法
    
    // 3. 应用缩放因子
    // kq = ggml_scale(ctx, kq, scaleValue);
    
    // 4. 应用注意力掩码（如果有）并计算softmax
    ggml_tensor* kq_soft_max;
    if (use_mask && mask_tensor) {
        // 使用带掩码的softmax
        kq_soft_max = ggml_soft_max_ext(ctx, kq, mask_tensor, scaleValue, 0.0f);
    } else {
        // 使用普通softmax，先缩放再softmax
        ggml_tensor* kq_scaled = ggml_scale(ctx, kq, scaleValue);
        kq_soft_max = ggml_soft_max(ctx, kq_scaled);
    }
    
    // 5. 计算 Attention * V
    ggml_tensor* v_cur = ggml_cont(ctx, ggml_permute(ctx, v_tensor, 1, 0, 2, 3));
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_cur, kq_soft_max);
    
    // 6. 重新排列结果
    // ggml_tensor* kqv_merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // BDSH -> BHSD
    
    // 7. 转换为连续内存布局
    ggml_tensor* result = ggml_cpy(ctx, kqv, output_tensor);
    // ggml_tensor* result = ggml_cpy(ctx, kqv_merged, output_tensor);
    
    // 添加到计算图中
    ggml_build_forward_expand(gf, result);
    
    // 执行计算
    ggml_build_forward_expand(gf, result);
    struct ggml_cplan cplan = ggml_graph_plan(gf, 1, NULL);
    
    // 为 cplan 分配 work_data
    if (cplan.work_size > 0) {
        void* work_memory = malloc(cplan.work_size);
        if (!work_memory) {
            std::cerr << "Failed to allocate work memory for cplan" << std::endl;
            ggml_free(ctx);
            return;
        }
        cplan.work_data = (uint8_t*)work_memory;
    }
    
    ggml_status status = ggml_graph_compute(gf, &cplan);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("attention_cpu failed with status : %d\n", status);
        GGML_ABORT("attention_cpu failed");
    }
    
    // 释放 work_data
    if (cplan.work_size > 0 && cplan.work_data) {
        free(cplan.work_data);
    }
    // 复制结果到输出向量
    memcpy(output.data(), output_tensor->data, output.size() * sizeof(float));
    
    // 释放资源
    ggml_free(ctx);
}

void cached_attention_cpu(
    const std::vector<float> &query_nope,
    const std::vector<float> &query_rope,
    const std::vector<float> &ctKV_nope,
    const std::vector<float> &key_rope,
    const std::vector<int32_t> &blockTables,
    const std::vector<int64_t> &contextLens,
    std::vector<float> &mask,
    std::vector<float> &output,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t kv_num_heads,
    int64_t head_dims_kq,
    int64_t head_dims_v,
    int64_t block_size,
    int64_t maxBlockNumPerSeq,
    float scaleValue
) {
    int blockNum = num_tokens * maxBlockNumPerSeq;

    std::vector<float> query(num_tokens * num_heads * (512 + 64));
    for (int i = 0; i < num_tokens; ++i) {
        for (int h = 0; h < num_heads; ++h) {
            // 复制非rope部分 (512维)
            int src_base = (i * num_heads + h) * 512;
            int tgt_base = (i * num_heads + h) * (512 + 64);
            std::copy(query_nope.begin() + src_base,
                    query_nope.begin() + src_base + 512,
                    query.begin() + tgt_base);
            
            // 复制rope部分 (64维)
            src_base = (i * num_heads + h) * 64;
            tgt_base = (i * num_heads + h) * (512 + 64) + 512;
            std::copy(query_rope.begin() + src_base,
                    query_rope.begin() + src_base + 64,
                    query.begin() + tgt_base);
        }
    }

    // 拼接K缓存数据 - 布局: [num_blocks, block_size, kv_num_heads, head_dims_kq]
    std::vector<float> key_cache(blockNum * block_size * kv_num_heads * (512 + 64));
    for (int b = 0; b < blockNum; ++b) {
        for (int s = 0; s < block_size; ++s) {
            for (int h = 0; h < kv_num_heads; ++h) {
                // 复制非rope部分 (512维)
                int src_base = (b * block_size * kv_num_heads + s * kv_num_heads + h) * 512;
                int tgt_base = (b * block_size * kv_num_heads + s * kv_num_heads + h) * (512 + 64);
                std::copy(ctKV_nope.begin() + src_base,
                        ctKV_nope.begin() + src_base + 512,
                        key_cache.begin() + tgt_base);
                
                // 复制rope部分 (64维)
                src_base = (b * block_size * kv_num_heads + s * kv_num_heads + h) * 64;
                tgt_base = (b * block_size * kv_num_heads + s * kv_num_heads + h) * (512 + 64) + 512;
                std::copy(key_rope.begin() + src_base,
                        key_rope.begin() + src_base + 64,
                        key_cache.begin() + tgt_base);
            }
        }
    }
    
    std::vector<float> value_cache = ctKV_nope;
    
    // 计算压缩后的头数
    int64_t compressed_heads = num_heads / kv_num_heads;
    
    // 初始化输出
    output.resize(num_tokens * num_heads * head_dims_v, 0.0f);
    
    // 为每个token单独处理
    for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        int64_t context_len = contextLens[token_idx];
        if (context_len == 0) continue;
        
        // 获取当前token对应的块表
        const int32_t* block_table = blockTables.data() + token_idx * maxBlockNumPerSeq;
        
        // 为当前token提取完整的K和V序列
        // 布局: [1, kv_num_heads, context_len, head_dims_kq] 和 [1, kv_num_heads, context_len, head_dims_v]
        std::vector<float> token_key(kv_num_heads * context_len * head_dims_kq, 0.0f);
        std::vector<float> token_value(kv_num_heads * context_len * head_dims_v, 0.0f);
        
        // 从KV缓存中提取当前token对应的K和V
        for (int64_t j = 0; j < context_len; ++j) {
            int32_t block_number = block_table[j / block_size];
            int64_t block_offset = j % block_size;
            
            // 计算在KV缓存中的起始位置
            int64_t key_cache_offset = block_number * (block_size * kv_num_heads * head_dims_kq) + 
                                     block_offset * (kv_num_heads * head_dims_kq);
            int64_t value_cache_offset = block_number * (block_size * kv_num_heads * head_dims_v) + 
                                       block_offset * (kv_num_heads * head_dims_v);
            
            // 复制所有头的K和V数据
            for (int64_t h = 0; h < kv_num_heads; ++h) {
                // K数据 - 布局: [1, kv_num_heads, context_len, head_dims_kq]
                int64_t key_src_start = key_cache_offset + h * head_dims_kq;
                int64_t key_tgt_start = (h * context_len + j) * head_dims_kq;
                
                if (key_src_start + head_dims_kq <= key_cache.size() && 
                    key_tgt_start + head_dims_kq <= token_key.size()) {
                    std::copy(key_cache.begin() + key_src_start,
                             key_cache.begin() + key_src_start + head_dims_kq,
                             token_key.begin() + key_tgt_start);
                }
                
                // V数据 - 布局: [1, kv_num_heads, context_len, head_dims_v]
                int64_t value_src_start = value_cache_offset + h * head_dims_v;
                int64_t value_tgt_start = (h * context_len + j) * head_dims_v;
                
                if (value_src_start + head_dims_v <= value_cache.size() && 
                    value_tgt_start + head_dims_v <= token_value.size()) {
                    std::copy(value_cache.begin() + value_src_start,
                             value_cache.begin() + value_src_start + head_dims_v,
                             token_value.begin() + value_tgt_start);
                }
            }
        }
        
        // 重塑当前token的查询 [num_heads, head_dims_kq] -> [1, num_heads, 1, head_dims_kq]
        // 由于compressHead=true，我们需要进一步重塑为 [1, kv_num_heads, compressed_heads, head_dims_kq]
        std::vector<float> reshaped_query(kv_num_heads * compressed_heads * head_dims_kq, 0.0f);
        
        for (int64_t kv_h = 0; kv_h < kv_num_heads; ++kv_h) {
            for (int64_t comp_h = 0; comp_h < compressed_heads; ++comp_h) {
                for (int64_t dim = 0; dim < head_dims_kq; ++dim) {
                    // 源索引：原始查询 [token_idx, num_heads, head_dims_kq]
                    int64_t src_idx = (token_idx * num_heads + (kv_h * compressed_heads + comp_h)) * head_dims_kq + dim;
                    
                    // 目标索引：重塑查询 [kv_h, comp_h, dim]
                    int64_t tgt_idx = ((kv_h * compressed_heads + comp_h) * head_dims_kq + dim);
                    
                    if (src_idx < query.size() && tgt_idx < reshaped_query.size()) {
                        reshaped_query[tgt_idx] = query[src_idx];
                    }
                }
            }
        }
        
        std::vector<float> attention_output(kv_num_heads * compressed_heads * head_dims_v, 0.0f);

        int64_t mask_size = context_len;
        for (auto &val : mask) { val = 0.0;}
        
        mla_attn_cpu(
            reshaped_query,      // 查询: [1, kv_num_heads * compressed_heads, 1, head_dims_kq] 
            token_key,           // 键: [1, kv_num_heads, context_len, head_dims_kq]
            token_value,         // 值: [1, kv_num_heads, context_len, head_dims_v]
            mask,                // mask
            attention_output,    // 输出: [1, kv_num_heads * compressed_heads, 1, head_dims_v]
            1,                   // batch_size = 1 (单个token)
            num_heads,           // num_heads
            head_dims_kq,        // head_dims_kq
            head_dims_v,         // head_dims_v
            kv_num_heads,        // key_num_heads
            1,                   // sequence_lenth_q = 1 (单个查询)
            context_len,         // sequence_lenth_kv
            scaleValue          // 缩放因子
        );
        
        // [1, num_heads, 1, head_dims_v]
        for (int64_t kv_h = 0; kv_h < kv_num_heads; ++kv_h) {
            for (int64_t comp_h = 0; comp_h < compressed_heads; ++comp_h) {
                for (int64_t dim = 0; dim < head_dims_v; ++dim) {

                    int64_t src_idx = ((kv_h * compressed_heads + comp_h) * head_dims_v + dim);
                    int64_t tgt_idx = (token_idx * num_heads + (kv_h * compressed_heads + comp_h)) * head_dims_v + dim;
                    
                    if (src_idx < attention_output.size() && tgt_idx < output.size()) {
                        output[tgt_idx] = attention_output[src_idx];
                    }
                }
            }
        }
    }
}