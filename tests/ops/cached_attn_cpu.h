#include <vector>
#include <cstdint>
#include <string>

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
);