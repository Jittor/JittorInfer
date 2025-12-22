#include "llama-graph-qwen3.h"

#include <cmath>

#include "llama-context.h"

llm_qwen3_context::llm_qwen3_context(llama_context & lctx, std::vector<uint8_t> & buf_compute_meta,
                                     const llama_ubatch & ubatch, const llm_build_cb & cb, bool worst_case,
                                     int print_layer) :
    llm_build_context(lctx),
    model(lctx.model),
    hparams(model.hparams),
    cparams(lctx.cparams),
    ubatch(ubatch),
    kv_self(lctx.kv_self),
    n_embd(hparams.n_embd),
    n_layer(hparams.n_layer),
    n_rot(hparams.n_rot),
    n_ctx(cparams.n_ctx),
    n_head(hparams.n_head()),
    n_head_kv(hparams.n_head_kv()),
    n_embd_head_k(hparams.n_embd_head_k),
    n_embd_k_gqa(hparams.n_embd_k_gqa()),
    n_embd_head_v(hparams.n_embd_head_v),
    n_embd_v_gqa(hparams.n_embd_v_gqa()),
    n_expert(hparams.n_expert),
    n_expert_used(hparams.n_expert_used),
    freq_base(cparams.rope_freq_base),
    freq_scale(cparams.rope_freq_scale),
    ext_factor(cparams.yarn_ext_factor),
    attn_factor(cparams.yarn_attn_factor),
    beta_fast(cparams.yarn_beta_fast),
    beta_slow(cparams.yarn_beta_slow),
    norm_eps(hparams.f_norm_eps),
    norm_rms_eps(hparams.f_norm_rms_eps),
    n_tokens(ubatch.n_tokens),
    n_kv(worst_case ? kv_self.size : kv_self.n),
    n_outputs(worst_case ? n_tokens : lctx.n_outputs),
    n_outputs_enc(worst_case ? n_tokens : lctx.embd_enc.size() / hparams.n_embd),
    kv_head([this, worst_case]() {
        if (worst_case) {
            return kv_self.recurrent ? 0 : kv_self.size - n_tokens;
        }
        return kv_self.head;
    }()),
    n_ctx_orig(cparams.n_ctx_orig_yarn),
    flash_attn(cparams.flash_attn),
    pooling_type(cparams.pooling_type),
    rope_type(hparams.rope_type),
    cb(cb),
    buf_compute_meta(buf_compute_meta),
    print_layer_(print_layer) {
    // all initializations should be done in init()
}

void llm_qwen3_context::init() {
    struct ggml_init_params params = {
        /*.mem_size   =*/buf_compute_meta.size(),
        /*.mem_buffer =*/buf_compute_meta.data(),
        /*.no_alloc   =*/true,
    };

    ctx0 = ggml_init(params);

    lctx.inp_tokens        = nullptr;
    lctx.inp_embd          = nullptr;
    lctx.inp_pos           = nullptr;
    lctx.inp_out_ids       = nullptr;
    lctx.inp_KQ_mask       = nullptr;
    lctx.inp_KQ_mask_swa   = nullptr;
    lctx.inp_KQ_mask_i8    = nullptr;
    lctx.inp_K_shift       = nullptr;
    lctx.inp_mean          = nullptr;
    lctx.inp_cls           = nullptr;
    lctx.inp_s_copy        = nullptr;
    lctx.inp_s_mask        = nullptr;
    lctx.inp_s_seq         = nullptr;
    lctx.inp_pos_bucket    = nullptr;
    lctx.inp_embd_enc      = nullptr;
    lctx.inp_KQ_mask_cross = nullptr;
    lctx.inp_attn_indices  = nullptr;
    lctx.inp_length_q      = nullptr;
    lctx.inp_length_kv     = nullptr;
}

struct ggml_tensor * llm_qwen3_context::build_inp_pos() {
    lctx.inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    cb(lctx.inp_pos, "inp_pos", -1);
    ggml_set_input(lctx.inp_pos);
    return lctx.inp_pos;
}

struct ggml_tensor * llm_qwen3_context::build_inp_KQ_mask(bool causal) {
    if (model.hparams.enable_cann_flash_attention) {
        lctx.inp_KQ_mask_i8 = ggml_new_tensor_2d(ctx0, GGML_TYPE_I8, n_kv, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        cb(lctx.inp_KQ_mask_i8, "KQ_mask", -1);
        ggml_set_input(lctx.inp_KQ_mask_i8);
        return lctx.inp_KQ_mask_i8;
    }
    lctx.inp_KQ_mask = causal ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD)) :
                                ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    cb(lctx.inp_KQ_mask, "KQ_mask", -1);
    ggml_set_input(lctx.inp_KQ_mask);
    return flash_attn ? ggml_cast(ctx0, lctx.inp_KQ_mask, GGML_TYPE_F16) : lctx.inp_KQ_mask;
}

struct ggml_tensor * llm_qwen3_context::build_inp_out_ids() {
    lctx.inp_out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    cb(lctx.inp_out_ids, "inp_out_ids", -1);
    ggml_set_input(lctx.inp_out_ids);
    return lctx.inp_out_ids;
}

struct ggml_cgraph * llm_qwen3_context::build_qwen3() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q, K, V using merged QKV weight for better performance
            struct ggml_tensor * Qcur;
            struct ggml_tensor * Kcur;
            struct ggml_tensor * Vcur;

            if (model.layers[il].wqkv != nullptr) {
                // Use merged QKV weight
                const int64_t n_embd_head = hparams.n_embd_head_k;
                const int64_t q_dim = n_embd_head * n_head;
                const int64_t k_dim = n_embd_head * n_head_kv;
                const int64_t v_dim = n_embd_head * n_head_kv;

                struct ggml_tensor * QKVcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(QKVcur, "QKVcur", il);

                // Split QKV into Q, K, V
                // QKVcur shape: [qkv_dim, n_tokens]
                // Use ggml_cont to make views contiguous for CANN backend compatibility
                Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, QKVcur, q_dim, n_tokens, QKVcur->nb[1], 0));
                cb(Qcur, "Qcur", il);

                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, QKVcur, k_dim, n_tokens, QKVcur->nb[1], q_dim * ggml_element_size(QKVcur)));
                cb(Kcur, "Kcur", il);

                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, QKVcur, v_dim, n_tokens, QKVcur->nb[1], (q_dim + k_dim) * ggml_element_size(QKVcur)));
                cb(Vcur, "Vcur", il);
            } else {
                // Fallback to separate Q, K, V weights
                Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head, n_tokens);
            Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens);
            Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, model.layers[il].bo, Kcur, Vcur, Qcur,
                               KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head_k)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur                              = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA                            = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        // Use merged gate-up weight if available for better performance
        if (model.layers[il].ffn_gate_up != nullptr) {
            // Use merged FFN gate-up weight
            struct ggml_tensor * gate_up = llm_build_lora_mm(lctx, ctx0, model.layers[il].ffn_gate_up, cur);
            cb(gate_up, "ffn_gate_up", il);

            // Split gate_up into gate and up
            // gate_up shape: [2*n_ff, n_tokens]
            const int64_t n_ff = hparams.n_ff();
            struct ggml_tensor * gate = ggml_cont(ctx0, ggml_view_2d(ctx0, gate_up, n_ff, gate_up->ne[1], 
                                                                      gate_up->nb[1], 0));
            cb(gate, "ffn_gate", il);

            struct ggml_tensor * up = ggml_cont(ctx0, ggml_view_2d(ctx0, gate_up, n_ff, gate_up->ne[1], 
                                                                    gate_up->nb[1], n_ff * ggml_element_size(gate_up)));
            cb(up, "ffn_up", il);

            // Apply activation to gate
            gate = ggml_silu(ctx0, gate);
            cb(gate, "ffn_silu", il);

            // Multiply gate and up
            cur = ggml_mul(ctx0, gate, up);
            cb(cur, "ffn_gate_par", il);

            // Down projection
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].ffn_down, cur);
            cb(cur, "ffn_out", il);
        } else {
            // Fallback to separate gate and up weights
            cur = llm_build_ffn(ctx0, lctx, cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL,
                                model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        // TODO(critical): 我们暂时没有使用adapter!
        cur = lctx.cvec.apply_to(cur);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

void llm_qwen3_context::free() {
    ggml_free(ctx0);
    ctx0 = nullptr;
}

struct ggml_cgraph * llm_build_qwen3(llama_context & lctx, std::vector<uint8_t> & buf_compute_meta,
                                     const llama_ubatch & ubatch, llm_build_cb & cb, bool worst_case, int print_layer) {
    struct ggml_cgraph * result = NULL;

    llm_qwen3_context llm(lctx, buf_compute_meta, ubatch, cb, worst_case, print_layer);

    llm.init();

    result = llm.build_qwen3();

    // add on pooling layer
    GGML_ASSERT(!lctx.cparams.embeddings);

    llm.free();

    return result;
}
