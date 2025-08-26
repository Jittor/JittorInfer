#ifndef LLAMA_GRAPH_UTILS_H
#define LLAMA_GRAPH_UTILS_H

#include <functional>

#include "ggml-backend.h"
#include "llama-hparams.h"
#include "llama-kv-cache.h"

using llm_build_cb = std::function<void(struct ggml_tensor * cur, const char * name, int nl)>;

// operators
enum llm_ffn_op_type {
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR,  // ffn_gate is parallel to ffn_up
};

// llama context
class llm_build_context {
  protected:
    llama_context &       lctx;
    struct ggml_context * ctx0 = nullptr;
  public:
    llm_build_context(llama_context & lctx) : lctx(lctx) {}

    void init();
    void free();

    ggml_context * get_context() { return ctx0; }
};

// input building
struct ggml_tensor * llm_build_inp_embd(struct ggml_context * ctx, struct llama_context & lctx,
                                        const llama_hparams & hparams, const llama_ubatch & ubatch,
                                        struct ggml_tensor * tok_embd, const llm_build_cb & cb,
                                        bool enable_fp16 = false);

// lora
struct ggml_tensor * llm_build_lora_mm(struct llama_context & lctx, struct ggml_context * ctx0, struct ggml_tensor * w,
                                       struct ggml_tensor * cur, bool enable_fp16 = false);

struct ggml_tensor * llm_build_lora_mm_id(struct llama_context & lctx, struct ggml_context * ctx0,
                                          struct ggml_tensor * w,    // struct ggml_tensor * as
                                          struct ggml_tensor * cur,  // struct ggml_tensor * b
                                          struct ggml_tensor * ids);

// norm
struct ggml_tensor * llm_build_norm(struct ggml_context * ctx, struct ggml_tensor * cur, const llama_hparams & hparams,
                                    struct ggml_tensor * mw, struct ggml_tensor * mb, llm_norm_type type,
                                    const llm_build_cb & cb, int il, bool fuse_mul = false);

// ffn
struct ggml_tensor * llm_build_ffn(struct ggml_context * ctx, struct llama_context & lctx, struct ggml_tensor * cur,
                                   struct ggml_tensor * up, struct ggml_tensor * up_b, struct ggml_tensor * up_s,
                                   struct ggml_tensor * gate, struct ggml_tensor * gate_b, struct ggml_tensor * gate_s,
                                   struct ggml_tensor * down, struct ggml_tensor * down_b, struct ggml_tensor * down_s,
                                   struct ggml_tensor * act_scales, llm_ffn_op_type type_op,
                                   llm_ffn_gate_type type_gate, const llm_build_cb & cb, int il,
                                   bool enable_fp16 = false);

// moe ffn
struct ggml_tensor * llm_build_moe_ffn(struct ggml_context * ctx, struct llama_context & lctx, struct ggml_tensor * cur,
                                       struct ggml_tensor * gate_inp, struct ggml_tensor * up_exps,
                                       struct ggml_tensor * gate_exps, struct ggml_tensor * down_exps,
                                       struct ggml_tensor * exp_probs_b, int64_t n_expert, int64_t n_expert_used,
                                       int64_t expert_group_id, int64_t n_expert_groups, llm_ffn_op_type type_op,
                                       bool enable_fused_moe, bool norm_w, bool scale_w, float w_scale,
                                       llama_expert_gating_func_type gating_op, const llm_build_cb & cb, int il,
                                       bool enable_fp16 = false);

// attention
struct ggml_tensor * llm_build_kv(struct ggml_context * ctx, struct llama_context & lctx, const llama_kv_cache & kv,
                                  struct ggml_cgraph * graph, struct ggml_tensor * wo, struct ggml_tensor * wo_b,
                                  struct ggml_tensor * k_cur, struct ggml_tensor * v_cur, struct ggml_tensor * q_cur,
                                  struct ggml_tensor * kq_mask, int32_t n_tokens, int32_t kv_head, int32_t n_kv,
                                  float kq_scale, const llm_build_cb & cb, int il);

struct ggml_tensor * llm_attn_mla(struct ggml_context * ctx, struct llama_context & lctx, const llama_kv_cache & kv,
                                  struct ggml_cgraph * graph, struct ggml_tensor * wo, struct ggml_tensor * wo_b,
                                  struct ggml_tensor * wkv_b, struct ggml_tensor * kv_cur, struct ggml_tensor * q_nope,
                                  struct ggml_tensor * q_pe, struct ggml_tensor * kq_mask, int32_t n_tokens,
                                  int32_t kv_head, int32_t n_kv, int32_t n_embd_head_qk_nope, int32_t n_embd_head_v,
                                  int32_t n_head, int32_t n_embd_head_qk_rope, float kq_scale, const llm_build_cb & cb,
                                  int il);

struct ggml_tensor * llm_build_kv_ge(struct ggml_context * ctx, struct llama_context & lctx, const llama_kv_cache & kv,
                                     struct ggml_cgraph * graph, struct ggml_tensor * wo, struct ggml_tensor * wo_b,
                                     struct ggml_tensor * k_cur, struct ggml_tensor * v_cur, struct ggml_tensor * q_cur,
                                     struct ggml_tensor * indices, struct ggml_tensor * length_q,
                                     struct ggml_tensor * length_kv, int32_t n_tokens, int32_t n_kv, float kq_scale,
                                     const llm_build_cb & cb, int il, bool enable_fp16 = false);
#endif  // LLAMA_GRAPH_UTILS_H
