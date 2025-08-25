#ifndef LLAMA_GRAPH_QWEN2_H
#define LLAMA_GRAPH_QWEN2_H

#include "llama-graph-utils.h"

class llm_qwen2_context : public llm_build_context {
  protected:
    const llama_model &    model;
    const llama_hparams &  hparams;
    const llama_cparams &  cparams;
    const llama_ubatch &   ubatch;
    const llama_kv_cache & kv_self;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_rot;
    const int64_t n_ctx;  // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int32_t n_tokens;
    const int32_t n_kv;  // size of KV cache to consider (n_kv <= kv_self.size)
    const int32_t n_outputs;
    const int32_t n_outputs_enc;
    const int32_t kv_head;  // index of where we store new KV data in the cache
    const int32_t n_ctx_orig;

    const bool flash_attn;

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    const llm_build_cb & cb;

    std::vector<uint8_t> & buf_compute_meta;

    int print_layer_;

    // assistant
    struct ggml_tensor * build_inp_pos();
    struct ggml_tensor * build_inp_KQ_mask(bool causal = true);
    struct ggml_tensor * build_inp_out_ids();

  public:
    llm_qwen2_context(llama_context & lctx, std::vector<uint8_t> & buf_compute_meta, const llama_ubatch & ubatch,
                          const llm_build_cb & cb, bool worst_case, int print_layer = -1);

    void init();

    struct ggml_cgraph * build_qwen2();

    void free();
};

struct ggml_cgraph * llm_build_qwen2(llama_context & lctx, std::vector<uint8_t> & buf_compute_meta,
                                         const llama_ubatch & ubatch, llm_build_cb & cb, bool worst_case, int print_layer = -1);

#endif  // LLAMA_GRAPH_QWEN2_H
