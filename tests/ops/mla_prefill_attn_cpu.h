#include <vector>
#include <cstdint>
#include <string>

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
);

