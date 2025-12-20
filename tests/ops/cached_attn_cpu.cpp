#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include "flash_attn_cpu.h"
#include "cached_attn_cpu.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>

void cached_attention_cpu(
    const std::vector<float> &query_nope,
    const std::vector<float> &query_rope,
    const std::vector<float> &ctKV_nope,
    const std::vector<float> &key_rope,
    const std::vector<int32_t> &blockTables,
    const std::vector<int64_t> &contextLens,
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
        std::vector<int8_t> mask(mask_size);
        for (auto &val : mask) { val = 0;}
        
        flash_attn_cpu(
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
            scaleValue           // 缩放因子
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