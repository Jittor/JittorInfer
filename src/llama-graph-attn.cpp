#include "ggml.h"
#include "llama-context.h"
#include "llama-graph-utils.h"

static void llm_build_kv_store(struct ggml_context * ctx, const llama_hparams & hparams, const llama_cparams & cparams,
                               const llama_kv_cache & kv, struct ggml_cgraph * graph, struct ggml_tensor * k_cur,
                               struct ggml_tensor * v_cur, int32_t n_tokens, int32_t kv_head, const llm_build_cb & cb,
                               int64_t il) {
    const int64_t n_ctx = cparams.n_ctx;

    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    GGML_ASSERT(kv.size == n_ctx);

    struct ggml_tensor * k_cache_view =
        ggml_view_1d(ctx, kv.k_l[il], n_tokens * n_embd_k_gqa, ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa) * kv_head);
    cb(k_cache_view, "k_cache_view", il);

    // note: storing RoPE-ed version of K in the KV cache
    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));

    GGML_ASSERT(v_cur->ne[0] == n_embd_v_gqa && v_cur->ne[1] == n_tokens);

    struct ggml_tensor * v_cache_view = nullptr;

    if (cparams.flash_attn || hparams.enable_cann_flash_attention) {
        v_cache_view = ggml_view_1d(ctx, kv.v_l[il], n_tokens * n_embd_v_gqa,
                                    ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa) * kv_head);
    } else {
        // note: the V cache is transposed when not using flash attention
        v_cache_view = ggml_view_2d(ctx, kv.v_l[il], n_tokens, n_embd_v_gqa, (n_ctx) *ggml_element_size(kv.v_l[il]),
                                    (kv_head) *ggml_element_size(kv.v_l[il]));

        v_cur = ggml_transpose(ctx, v_cur);
    }
    cb(v_cache_view, "v_cache_view", il);

    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
}

static struct ggml_tensor * llm_build_kqv(struct ggml_context * ctx, struct llama_context & lctx,
                                          const llama_kv_cache & kv, struct ggml_cgraph * graph,
                                          struct ggml_tensor * wo, struct ggml_tensor * wo_b,
                                          struct ggml_tensor * q_cur, struct ggml_tensor * kq_mask, int32_t n_tokens,
                                          int32_t n_kv, float kq_scale, const llm_build_cb & cb, int il) {
    const llama_model &   model   = lctx.model;
    const llama_hparams & hparams = lctx.model.hparams;
    const llama_cparams & cparams = lctx.cparams;

    const int head_split = (hparams.enable_tensor_parallel && !hparams.enable_data_parallel) ? hparams.num_parallel : 1;
    const int64_t n_ctx  = cparams.n_ctx;
    const int64_t n_head = hparams.n_head(il) / head_split;
    const int64_t n_head_kv     = hparams.n_head_kv(il) / head_split;
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_head_v = hparams.n_embd_head_v;
    const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(il);

    struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
    cb(q, "q", il);

    struct ggml_tensor * k =
        ggml_view_3d(ctx, kv.k_l[il], n_embd_head_k, n_kv, n_head_kv, ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa),
                     ggml_row_size(kv.k_l[il]->type, n_embd_head_k), 0);
    cb(k, "k", il);

    struct ggml_tensor * cur;

    if (model.hparams.enable_cann_flash_attention) {
        const int64_t pad_n_embd = GGML_PAD(std::max(n_embd_head_k, n_embd_head_v), 64);
        const int     n_kv_act   = n_ctx;
        k = ggml_view_3d(ctx, kv.k_l[il], pad_n_embd, n_head_kv, n_kv_act, ggml_row_size(kv.k_l[il]->type, pad_n_embd),
                         ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa), 0);
        cb(k, "k", il);

        q = q_cur;
        cb(q, "q", il);

        const int64_t        pad_n_tokens = GGML_PAD(n_tokens, GGML_KQ_MASK_PAD);
        struct ggml_tensor * kq_view =
            ggml_view_2d(ctx, kv.kq_mask_l[il], n_kv, pad_n_tokens, ggml_row_size(kv.kq_mask_l[il]->type, n_ctx), 0);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, kq_mask, kq_view));
        struct ggml_tensor * kq_full = ggml_view_2d(ctx, kv.kq_mask_l[il], n_kv_act, pad_n_tokens,
                                                    ggml_row_size(kv.kq_mask_l[il]->type, n_ctx), 0);
        struct ggml_tensor * v =
            ggml_view_3d(ctx, kv.v_l[il], pad_n_embd, n_head_kv, n_kv_act, ggml_row_size(kv.v_l[il]->type, pad_n_embd),
                         ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa), 0);
        cb(v, "v", il);
        if (q->type != GGML_TYPE_F16) {
            q = ggml_cast(ctx, q, GGML_TYPE_F16);
        }
        if (k->type != GGML_TYPE_F16) {
            k = ggml_cast(ctx, k, GGML_TYPE_F16);
        }
        if (v->type != GGML_TYPE_F16) {
            v = ggml_cast(ctx, v, GGML_TYPE_F16);
        }
        cur = ggml_flash_attn_prompt(ctx, q, k, v, kq_full, 1, n_head, pad_n_embd, pad_n_embd, n_head, n_tokens, n_kv,
                                     nullptr, nullptr, kq_scale);
        cur = ggml_view_3d(ctx, cur, n_embd_head_v, n_head, n_tokens, ggml_row_size(cur->type, pad_n_embd),
                           ggml_row_size(cur->type, n_embd_v_gqa), 0);
        cur = ggml_cont_2d(ctx, cur, n_embd_head_v * n_head, n_tokens);
        cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
    } else if (cparams.flash_attn) {
        GGML_UNUSED(model);
        GGML_UNUSED(n_ctx);

        // split cached v into n_head heads (not transposed)
        struct ggml_tensor * v =
            ggml_view_3d(ctx, kv.v_l[il], n_embd_head_v, n_kv, n_head_kv, ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa),
                         ggml_row_size(kv.v_l[il]->type, n_embd_head_v), 0);
        cb(v, "v", il);

        cur = ggml_flash_attn_ext(ctx, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);

        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx, cur, n_embd_head_v * n_head, n_tokens);
    } else {
        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (model.arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplyer of 0.08838834764831845
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx, ggml_scale(ctx, kq, 0.08838834764831845f / 30.0f));
            kq = ggml_scale(ctx, kq, 30);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx, kq, 1.0f / hparams.f_attn_logit_softcapping);
            kq = ggml_tanh(ctx, kq);
            kq = ggml_scale(ctx, kq, hparams.f_attn_logit_softcapping);
        }

        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        cb(kq, "kq_soft_max_ext", il);

        GGML_ASSERT(kv.size == n_ctx);

        // split cached v into n_head heads
        struct ggml_tensor * v =
            ggml_view_3d(ctx, kv.v_l[il], n_kv, n_embd_head_v, n_head_kv, ggml_element_size(kv.v_l[il]) * n_ctx,
                         ggml_element_size(kv.v_l[il]) * n_ctx * n_embd_head_v, 0);
        cb(v, "v", il);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
        cb(kqv, "kqv", il);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        cb(kqv_merged, "kqv_merged", il);

        cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_v * n_head, n_tokens);
        cb(cur, "kqv_merged_cont", il);
    }

    ggml_build_forward_expand(graph, cur);

    if (wo) {
        cur = llm_build_lora_mm(lctx, ctx, wo, cur);
        cb(cur, "o_proj", il);
    }

    if (wo_b) {
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx, cur, wo_b);
    }

    return cur;
}

struct ggml_tensor * llm_build_kv(struct ggml_context * ctx, struct llama_context & lctx, const llama_kv_cache & kv,
                                  struct ggml_cgraph * graph, struct ggml_tensor * wo, struct ggml_tensor * wo_b,
                                  struct ggml_tensor * k_cur, struct ggml_tensor * v_cur, struct ggml_tensor * q_cur,
                                  struct ggml_tensor * kq_mask, int32_t n_tokens, int32_t kv_head, int32_t n_kv,
                                  float kq_scale, const llm_build_cb & cb, int il) {
    const llama_hparams & hparams = lctx.model.hparams;
    const llama_cparams & cparams = lctx.cparams;

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    if (hparams.enable_cann_flash_attention) {
        const uint32_t n_head_kv = hparams.n_head_kv(il);
        const uint32_t max_embd  = GGML_PAD(std::max(q_cur->ne[0], k_cur->ne[0]), 64);
        q_cur                    = ggml_pad(ctx, q_cur, max_embd - q_cur->ne[0], 0, 0, 0);
        k_cur                    = ggml_pad(ctx, k_cur, max_embd - k_cur->ne[0], 0, 0, 0);
        v_cur                    = ggml_reshape_3d(ctx, v_cur, hparams.n_embd_head_v, n_head_kv, v_cur->ne[1]);
        v_cur                    = ggml_pad(ctx, v_cur, max_embd - v_cur->ne[0], 0, 0, 0);
        v_cur                    = ggml_reshape_2d(ctx, v_cur, max_embd * n_head_kv, v_cur->ne[2]);
    }

    ggml_build_forward_expand(graph, q_cur);
    ggml_build_forward_expand(graph, k_cur);
    ggml_build_forward_expand(graph, v_cur);

    llm_build_kv_store(ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, cb, il);

    struct ggml_tensor * cur;

    cur = llm_build_kqv(ctx, lctx, kv, graph, wo, wo_b, q_cur, kq_mask, n_tokens, n_kv, kq_scale, cb, il);
    cb(cur, "kqv_out", il);

    return cur;
}

struct ggml_tensor * llm_attn_mla(struct ggml_context * ctx, struct llama_context & lctx, const llama_kv_cache & kv,
                                  struct ggml_cgraph * graph, struct ggml_tensor * wo, struct ggml_tensor * wo_b,
                                  struct ggml_tensor * wkv_b, struct ggml_tensor * kv_cur, struct ggml_tensor * q_nope,
                                  struct ggml_tensor * q_pe, struct ggml_tensor * kq_mask, int32_t n_tokens,
                                  int32_t kv_head, int32_t n_kv, int32_t n_embd_head_qk_nope, int32_t n_embd_head_v,
                                  int32_t n_head, int32_t n_embd_head_qk_rope, float kq_scale, const llm_build_cb & cb,
                                  int il) {
    const llama_hparams & hparams      = lctx.model.hparams;
    const llama_cparams & cparams      = lctx.cparams;
    const int64_t         n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t         n_embd_v_gqa = hparams.n_embd_v_gqa(il);
    const uint32_t        kv_lora_rank = hparams.n_lora_kv;
    const int64_t         n_ctx        = cparams.n_ctx;

    // save load mla kv_cur
    {
        GGML_ASSERT(kv_cur->ne[0] == n_embd_k_gqa && kv_cur->ne[1] == n_tokens);
        struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv.k_l[il], n_tokens * n_embd_k_gqa,
                                                         ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa) * kv_head);
        cb(k_cache_view, "k_cache_view", il);

        // note: storing RoPE-ed version of K in the KV cache
        ggml_build_forward_expand(graph, ggml_cpy(ctx, kv_cur, k_cache_view));

        // note: the V cache is transposed when not using flash attention
        GGML_ASSERT(n_embd_v_gqa == kv_lora_rank);
        struct ggml_tensor * v_cache_view =
            ggml_view_2d(ctx, kv.v_l[il], n_tokens, n_embd_v_gqa, (n_ctx) *ggml_element_size(kv.v_l[il]),
                         (kv_head) *ggml_element_size(kv.v_l[il]));

        struct ggml_tensor * v_cur =
            ggml_view_2d(ctx, kv_cur, kv_lora_rank, n_tokens, ggml_row_size(kv_cur->type, n_embd_k_gqa), 0);
        v_cur = ggml_transpose(ctx, v_cur);
        cb(v_cur, "v_cur", il);

        ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
    }

    // recover k states and v states
    struct ggml_tensor * q_states;
    struct ggml_tensor * k =
        ggml_view_2d(ctx, kv.k_l[il], n_embd_k_gqa, n_kv, ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa), 0);
    cb(k, "k", il);
    struct ggml_tensor * v =
        ggml_view_2d(ctx, kv.v_l[il], n_kv, n_embd_v_gqa, ggml_element_size(kv.v_l[il]) * n_ctx, 0);
    cb(v, "v", il);
    {
        GGML_ASSERT(kv_lora_rank + n_embd_head_qk_rope == k->ne[0]);

        // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_kv} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_kv}
        struct ggml_tensor * wk_b = ggml_view_3d(ctx, wkv_b, n_embd_head_qk_nope, kv_lora_rank, n_head,
                                                 ggml_row_size(wkv_b->type, n_embd_head_qk_nope),
                                                 ggml_row_size(wkv_b->type, n_embd_head_qk_nope * kv_lora_rank), 0);

        q_nope = ggml_permute(ctx, q_nope, 0, 2, 1, 3);
        q_nope = ggml_mul_mat(ctx, wk_b, q_nope);
        cb(q_nope, "q_nope_absorb", il);
        q_pe     = ggml_permute(ctx, q_pe, 0, 2, 1, 3);
        q_states = ggml_concat(ctx, q_nope, q_pe, 0);
        cb(q_states, "q_states", il);
    }

    // attention
    struct ggml_tensor * cur;
    {
        GGML_ASSERT(q_states->ne[0] == kv_lora_rank + n_embd_head_qk_rope);
        GGML_ASSERT(q_states->ne[1] == n_tokens);
        GGML_ASSERT(q_states->ne[2] == n_head);
        struct ggml_tensor * q = ggml_view_2d(ctx, q_states, kv_lora_rank + n_embd_head_qk_rope, n_head * n_tokens,
                                              ggml_row_size(q_states->type, kv_lora_rank + n_embd_head_qk_rope), 0);
        cb(q, "q", il);

        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        cb(kq, "kq", il);

        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_view_3d(ctx, kq, n_kv, n_tokens, n_head, ggml_row_size(kq->type, n_kv),
                          ggml_row_size(kq->type, n_kv * n_tokens), 0);

        GGML_ASSERT(lctx.model.arch != LLM_ARCH_GROK);
        GGML_ASSERT(!hparams.attn_soft_cap);

        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        cb(kq, "kq_soft_max_ext", il);
        kq = ggml_view_2d(ctx, kq, n_kv, n_tokens * n_head, ggml_row_size(kq->type, n_kv), 0);

        GGML_ASSERT(kv.size == cparams.n_ctx);

        struct ggml_tensor * wv_b =
            ggml_view_3d(ctx, wkv_b, kv_lora_rank, n_embd_head_v, n_head, ggml_row_size(wkv_b->type, kv_lora_rank),
                         ggml_row_size(wkv_b->type, kv_lora_rank * n_embd_head_v),
                         ggml_row_size(wkv_b->type, kv_lora_rank * n_embd_head_qk_nope * n_head));

        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
        cb(kqv, "kqv", il);
        kqv = ggml_view_3d(ctx, kqv, kv_lora_rank, n_tokens, n_head, ggml_row_size(kqv->type, kv_lora_rank),
                           ggml_row_size(kqv->type, kv_lora_rank * n_tokens), 0);
        kqv = ggml_mul_mat(ctx, wv_b, kqv);
        cb(kqv, "kqv_absorb", il);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        cb(kqv_merged, "kqv_merged", il);

        cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_v * n_head, n_tokens);
        cb(cur, "kqv_merged_cont", il);
    }

    ggml_build_forward_expand(graph, cur);

    if (wo) {
        cur = llm_build_lora_mm(lctx, ctx, wo, cur);
        cb(cur, "o_proj", il);
    }

    if (wo_b) {
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx, cur, wo_b);
    }

    return cur;
}

struct ggml_tensor * llm_build_kv_ge(struct ggml_context * ctx, struct llama_context & lctx, const llama_kv_cache & kv,
                                     struct ggml_cgraph * graph, struct ggml_tensor * wo, struct ggml_tensor * wo_b,
                                     struct ggml_tensor * k_cur, struct ggml_tensor * v_cur, struct ggml_tensor * q_cur,
                                     struct ggml_tensor * indices, struct ggml_tensor * length_q,
                                     struct ggml_tensor * length_kv, int32_t n_tokens, int32_t n_kv, float kq_scale,
                                     const llm_build_cb & cb, int il, bool enable_fp16) {
    const llama_hparams & hparams = lctx.model.hparams;
    const llama_cparams & cparams = lctx.cparams;

    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_embd_head_v = hparams.n_embd_head_v;
    const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(il);
    const int64_t n_head_kv     = hparams.n_head_kv(il);
    const int64_t n_head        = hparams.n_head(il);
    const int64_t n_ctx         = cparams.n_ctx;

    // TODO(hsh): remove pad op here, need to modify kv cache initialization
    // pad qkv
    const int64_t        pad_n_embd = GGML_PAD(std::max(n_embd_head_k, n_embd_head_v), 64);
    struct ggml_tensor * q          = ggml_pad(ctx, q_cur, pad_n_embd - q_cur->ne[0], 0, 0, 0);
    k_cur                           = ggml_pad(ctx, k_cur, pad_n_embd - k_cur->ne[0], 0, 0, 0);
    k_cur                           = ggml_reshape_2d(ctx, k_cur, n_embd_k_gqa, k_cur->ne[2]);
    v_cur                           = ggml_reshape_3d(ctx, v_cur, n_embd_head_v, n_head_kv, v_cur->ne[1]);
    v_cur                           = ggml_pad(ctx, v_cur, pad_n_embd - v_cur->ne[0], 0, 0, 0);
    v_cur                           = ggml_reshape_2d(ctx, v_cur, n_embd_v_gqa, v_cur->ne[2]);

    ggml_build_forward_expand(graph, q);
    ggml_build_forward_expand(graph, k_cur);
    ggml_build_forward_expand(graph, v_cur);
    ggml_build_forward_expand(graph, indices);

    ggml_tensor * k = kv.k_l[il];
    ggml_tensor * v = kv.v_l[il];
    // save kv cache
    {
        k = ggml_reshape_2d(ctx, k, n_embd_k_gqa, n_ctx);
        v = ggml_reshape_2d(ctx, v, n_embd_v_gqa, n_ctx);
        k = ggml_scatter_update(ctx, k, indices, k_cur);
        v = ggml_scatter_update(ctx, v, indices, v_cur);
    }

    // computing attention
    {
        k = ggml_reshape_3d(ctx, k, pad_n_embd, n_head_kv, n_ctx);
        cb(k, "k_updated", il);
        v = ggml_reshape_3d(ctx, v, pad_n_embd, n_head_kv, n_ctx);
        cb(v, "v_updated", il);

        struct ggml_tensor * cur;
        struct ggml_tensor * kq_full =
            ggml_reshape_2d(ctx, kv.kq_mask_l[il], n_ctx, GGML_PAD(cparams.n_ubatch, GGML_KQ_MASK_PAD));
        if (q->type != GGML_TYPE_F16) {
            q = ggml_cast(ctx, q, GGML_TYPE_F16);
        }
        cur = ggml_flash_attn_prompt(ctx, q, k, v, kq_full, 1, n_head, pad_n_embd, pad_n_embd, n_head, n_tokens, n_kv,
                                     length_q, length_kv, kq_scale);
        cur = ggml_reshape_3d(ctx, cur, pad_n_embd, n_head, n_tokens);
        cur = ggml_get_slice(ctx, cur, 0, n_embd_head_v, 0);
        cur = ggml_reshape_2d(ctx, cur, n_embd_head_v * n_head, n_tokens);

        ggml_build_forward_expand(graph, cur);

        if (wo) {
            cur = llm_build_lora_mm(lctx, ctx, wo, cur, enable_fp16);
            cb(cur, "o_proj", il);
        }

        if (wo_b) {
            cb(cur, "kqv_wo", il);
        }

        if (wo_b) {
            cur = ggml_add(ctx, cur, wo_b);
        }

        return cur;
    }
}
