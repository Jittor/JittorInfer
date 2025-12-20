#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include "mla_prefill_attn_cpu.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>

void attention_cpu(
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
    float scaleValue
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
    ggml_tensor* q_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                            head_dims_kq, sequence_lenth_q, num_heads, batch_size);
    // 键张量: [batch_size, key_num_heads, sequence_lenth_kv, head_dims]
    ggml_tensor* k_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                            head_dims_kq, sequence_lenth_kv, key_num_heads, batch_size);
    // 值张量: [batch_size, key_num_heads, sequence_lenth_kv, head_dims]
    ggml_tensor* v_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                            head_dims_v, sequence_lenth_kv, key_num_heads, batch_size);
    
    // 掩码张量: [batch_size, 1, sequence_lenth_q, sequence_lenth_kv]
    ggml_tensor* mask_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                    sequence_lenth_kv, sequence_lenth_q, 1, batch_size);
    
    // 输出张量: [batch_size, num_heads, sequence_lenth_q, head_dims]
    ggml_tensor* output_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                head_dims_v, sequence_lenth_q, num_heads, batch_size);
    
    // 复制数据到张量
    memcpy(q_tensor->data, query.data(), query.size() * sizeof(float));
    memcpy(k_tensor->data, key.data(), key.size() * sizeof(float));
    memcpy(v_tensor->data, value.data(), value.size() * sizeof(float));
    memcpy(mask_tensor->data, attn_mask.data(), attn_mask.size() * sizeof(float));
    // 构建计算图
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 100, false);
    
    ggml_tensor* q_cur = q_tensor;
    
    // 2. 计算 Q*K^T
    ggml_tensor* kq = ggml_mul_mat(ctx, k_tensor, q_cur);  // 矩阵乘法
    
    // 3. 应用缩放因子
    // kq = ggml_scale(ctx, kq, scaleValue);
    
    // 4. 应用注意力掩码（如果有）并计算softmax
    ggml_tensor* kq_soft_max = ggml_soft_max_ext(ctx, kq, mask_tensor, scaleValue, 0.0f);
    // if (mask_tensor && !attn_mask.empty()) {
    //     kq_soft_max = ggml_soft_max_ext(ctx, kq, mask_tensor, 1.0f, 0.0f);
    // } else {
    //     kq_soft_max = ggml_soft_max(ctx, kq);
    // }
    
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

std::vector<float> rope_concat(const std::vector<float>& nope,
                                        const std::vector<float>& rope,
                                        int64_t batch_size, int64_t seq_len,
                                        int64_t num_heads, int64_t nope_dim = 128,
                                        int64_t rope_dim = 64) {
    std::vector<float> result(nope.size() + rope.size(), 0.0f);; // 先复制原始base
    int64_t total_dim = nope_dim + rope_dim;
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            for (int64_t h = 0; h < num_heads; ++h) {
                // 复制无位置编码的部分 (前nope_dim个维度)
                for (int64_t d = 0; d < nope_dim; ++d) {
                    int64_t result_idx = (b * seq_len + s) * (num_heads * total_dim) + 
                                       h * total_dim + d;
                    int64_t q_idx = (b * seq_len + s) * (num_heads * nope_dim) + 
                                  h * nope_dim + d;
                    result[result_idx] = nope[q_idx];
                }
                
                // 复制旋转位置编码的部分 (后rope_dim个维度)
                for (int64_t d = 0; d < rope_dim; ++d) {
                    int64_t result_idx = (b * seq_len + s) * (num_heads * total_dim) + 
                                       h * total_dim + nope_dim + d;
                    int64_t rope_idx = (b * seq_len + s) * (num_heads * rope_dim) + 
                                     h * rope_dim + d;
                    result[result_idx] = rope[rope_idx];
                }
            }
        }
    }
    return result;
}

// 辅助函数：重塑张量布局
std::vector<float> reshape_2d_to_4d(const std::vector<float>& input, 
                                   int64_t batch_size, int64_t seq_len, 
                                   int64_t num_heads, int64_t head_dim) {
    std::vector<float> output(batch_size * num_heads * seq_len * head_dim, 0.0f);
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t s = 0; s < seq_len; ++s){
                for (int64_t d = 0; d < head_dim; ++d) {
                    int64_t input_idx = (b * seq_len + s) * (num_heads * head_dim) + h * head_dim + d;
                    int64_t output_idx = b * (num_heads * seq_len * head_dim) + 
                                       h * (seq_len * head_dim) + 
                                       s * head_dim + d;
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
    return output;
}

// 辅助函数：重塑K/V张量布局
std::vector<float> reshape_kv_3d_to_4d(const std::vector<float>& input,
                                      int64_t batch_size, int64_t seq_len,
                                      int64_t num_heads, int64_t head_dim) {
    std::vector<float> output(batch_size * num_heads * seq_len * head_dim, 0.0f);
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t s = 0; s < seq_len; ++s) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    int64_t input_idx = b * (seq_len * num_heads * head_dim) + 
                                      s * (num_heads * head_dim) + 
                                      h * head_dim + d;
                    int64_t output_idx = b * (num_heads * seq_len * head_dim) + 
                                       h * (seq_len * head_dim) + 
                                       s * head_dim + d;
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
    return output;
}

std::vector<float> reshape_output_4d_to_2d(const std::vector<float>& output_4d,
                                          int64_t batch_size, int64_t seq_len,
                                          int64_t num_heads, int64_t head_dim) {
    std::vector<float> output_2d(batch_size * seq_len * num_heads * head_dim, 0.0f);
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t s = 0; s < seq_len; ++s) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    int64_t input_idx = b * (num_heads * seq_len * head_dim) + 
                                      h * (seq_len * head_dim) + 
                                      s * head_dim + d;
                    int64_t output_idx = (b * seq_len + s) * (num_heads * head_dim) + 
                                       h * head_dim + d;
                    output_2d[output_idx] = output_4d[input_idx];
                }
            }
        }
    }
    return output_2d;
}

void mla_prefill_using_attention_cpu(
    const std::vector<float>& Q,           // [batchSize * SeqlenQ, headNum * 128]
    const std::vector<float>& QRope,       // [batchSize * SeqlenQ, headNum * 64]
    const std::vector<float>& K,           // [batchSize, SeqlenKV, kvHeadNum * 128]
    const std::vector<float>& kRope,       // [batchSize, SeqlenKV, kvHeadNum * 64]
    const std::vector<float>& V,           // [batchSize, SeqlenKV, kvHeadNum * 128]
    const std::vector<float>& mask,        // [512, 512]
    std::vector<float>& output,            // [batchSize * SeqlenQ, headNum * 128] - 输出形状与Q相同
    int64_t batch_size,
    int64_t seq_len_q,
    int64_t seq_len_kv,
    int64_t head_num,
    int64_t kv_head_num,
    float scale,
    int64_t head_dim,
    int64_t rope_dim
) {
    // 1. 应用旋转位置编码到Q和K（替换前64个维度，保持形状不变）
    std::vector<float> Q_final = rope_concat(Q, QRope, batch_size, seq_len_q, 
                                                     head_num, head_dim, rope_dim);

    std::vector<float> K_final = rope_concat(K, kRope, batch_size, seq_len_kv, 
                                                       kv_head_num, head_dim, rope_dim);
    
    // 2. 重塑张量布局以适应attention_cpu函数
    // Q: [batch_size, head_num, seq_len_q, head_dim]
    std::vector<float> Q_4d = reshape_2d_to_4d(Q_final, batch_size, seq_len_q, head_num, head_dim + rope_dim);

    // K: [batch_size, kv_head_num, seq_len_kv, head_dim]
    std::vector<float> K_4d = reshape_kv_3d_to_4d(K_final, batch_size, seq_len_kv, kv_head_num, head_dim + rope_dim);

    // V: [batch_size, kv_head_num, seq_len_kv, head_dim]  
    std::vector<float> V_4d = reshape_kv_3d_to_4d(V, batch_size, seq_len_kv, kv_head_num, head_dim);

    // // 5. 调用CPU注意力函数
    std::vector<float> output_4d;

    attention_cpu(Q_4d, K_4d, V_4d, mask, output_4d,
                 batch_size, head_num, head_dim + rope_dim, head_dim, kv_head_num,
                 seq_len_q, seq_len_kv, scale);
    
    // // 6. 将输出重塑回原始布局 [batchSize * SeqlenQ, headNum * 128]
    output = reshape_output_4d_to_2d(output_4d, batch_size, seq_len_q, head_num, head_dim);
}