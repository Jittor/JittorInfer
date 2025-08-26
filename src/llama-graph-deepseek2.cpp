#include "llama-graph-deepseek2.h"

#include <cmath>

#include "llama-context.h"

llm_deepseek2_context::llm_deepseek2_context(llama_context & lctx, std::vector<uint8_t> & buf_compute_meta,
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

void llm_deepseek2_context::init() {
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

struct ggml_tensor * llm_deepseek2_context::build_inp_pos() {
    lctx.inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    cb(lctx.inp_pos, "inp_pos", -1);
    ggml_set_input(lctx.inp_pos);
    return lctx.inp_pos;
}

struct ggml_tensor * llm_deepseek2_context::build_inp_KQ_mask(bool causal) {
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

struct ggml_tensor * llm_deepseek2_context::build_inp_out_ids() {
    lctx.inp_out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    cb(lctx.inp_out_ids, "inp_out_ids", -1);
    ggml_set_input(lctx.inp_out_ids);
    return lctx.inp_out_ids;
}

struct ggml_cgraph * llm_deepseek2_context::build_deepseek2() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    bool is_lite = (hparams.n_layer == 27);

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
    const float mscale             = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale           = 1.0f * mscale * mscale / sqrtf(float(hparams.n_embd_head_k));
    const float attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

    const uint32_t n_embd_head_qk_rope = hparams.n_rot;
    const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
    const uint32_t kv_lora_rank        = hparams.n_lora_kv;

    // params changed in parallel
    const int64_t n_head_act      = (hparams.enable_tensor_parallel & !hparams.enable_data_parallel) ?
                                        n_head / lctx.model.params.num_parallel :
                                        n_head;
    const int64_t expert_group_id = hparams.enable_expert_parallel ? lctx.model.params.tp_id : 0;
    const int64_t n_expert_groups = hparams.enable_expert_parallel ? lctx.model.params.num_parallel : 1;
    const bool    run_mlp_only    = lctx.enable_dp_gather && lctx.self_token_size == 0;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;
    struct ggml_tensor * inp_pos;
    struct ggml_tensor * KQ_mask;

    // {n_embd, n_tokens}
    if (!run_mlp_only) {
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        KQ_mask = build_inp_KQ_mask();
    }

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        if (!run_mlp_only) {
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);
        }

        // self_attention
        if (!run_mlp_only) {
            struct ggml_tensor * q = NULL;
            if (!is_lite) {
                // {n_embd, q_lora_rank} * {n_embd, n_tokens} -> {q_lora_rank, n_tokens}
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = llm_build_norm(ctx0, q, hparams, model.layers[il].attn_q_a_norm, NULL, LLM_NORM_RMS, cb, il);
                cb(q, "q", il);

                // {q_lora_rank, n_head * hparams.n_embd_head_k} * {q_lora_rank, n_tokens} -> {n_head * hparams.n_embd_head_k, n_tokens}
                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
            }

            // split into {n_head * n_embd_head_qk_nope, n_tokens}
            struct ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head_act, n_tokens,
                                                       ggml_row_size(q->type, hparams.n_embd_head_k),
                                                       ggml_row_size(q->type, hparams.n_embd_head_k * n_head_act), 0);
            cb(q_nope, "q_nope", il);

            // and {n_head * n_embd_head_qk_rope, n_tokens}
            struct ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head_act, n_tokens,
                                                     ggml_row_size(q->type, hparams.n_embd_head_k),
                                                     ggml_row_size(q->type, hparams.n_embd_head_k * n_head_act),
                                                     ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            // {n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
            struct ggml_tensor * kv_pe_compresseed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_pe_compresseed, "kv_pe_compresseed", il);

            // split into {kv_lora_rank, n_tokens}
            struct ggml_tensor * kv_compressed =
                ggml_view_2d(ctx0, kv_pe_compresseed, kv_lora_rank, n_tokens, kv_pe_compresseed->nb[1], 0);
            cb(kv_compressed, "kv_compressed", il);

            // and {n_embd_head_qk_rope, n_tokens}
            struct ggml_tensor * k_pe =
                ggml_view_3d(ctx0, kv_pe_compresseed, n_embd_head_qk_rope, 1, n_tokens, kv_pe_compresseed->nb[1],
                             kv_pe_compresseed->nb[1], ggml_row_size(kv_pe_compresseed->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            // TODO: the CUDA backend used to not support non-cont. (RMS) norm, investigate removing ggml_cont
            kv_compressed = ggml_cont(ctx0, kv_compressed);
            kv_compressed = llm_build_norm(ctx0, kv_compressed, hparams, model.layers[il].attn_kv_a_norm, NULL,
                                           LLM_NORM_RMS, cb, il);
            cb(kv_compressed, "kv_compressed", il);

            if (hparams.enable_mla) {
                q_pe = ggml_cont(
                    ctx0,
                    q_pe);  // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor_scaled, beta_fast, beta_slow);
                cb(q_pe, "q_pe", il);

                // shared RoPE key
                k_pe = ggml_cont(
                    ctx0,
                    k_pe);  // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor_scaled, beta_fast, beta_slow);
                k_pe = ggml_view_2d(ctx0, k_pe, n_embd_head_qk_rope, n_tokens,
                                    ggml_row_size(k_pe->type, n_embd_head_qk_rope), 0);
                cb(k_pe, "k_pe", il);

                struct ggml_tensor * kv_states = ggml_concat(ctx0, kv_compressed, k_pe, 0);
                cb(kv_states, "kv_states", il);

                cur = llm_attn_mla(ctx0, lctx, kv_self, gf, model.layers[il].wo, NULL, model.layers[il].wkv_b,
                                   kv_states, q_nope, q_pe, KQ_mask, n_tokens, kv_head, n_kv, n_embd_head_qk_nope,
                                   n_embd_head_v, n_head_act, n_embd_head_qk_rope, kq_scale, cb, il);
            } else {
                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                struct ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
                cb(kv, "kv", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                struct ggml_tensor * k_nope = ggml_view_3d(
                    ctx0, kv, n_embd_head_qk_nope, n_head_act, n_tokens,
                    ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                    ggml_row_size(kv->type, n_head_act * (n_embd_head_qk_nope + hparams.n_embd_head_v)), 0);
                cb(k_nope, "k_nope", il);

                // and {n_head * n_embd_head_v, n_tokens}
                struct ggml_tensor * v_states =
                    ggml_view_3d(ctx0, kv, hparams.n_embd_head_v, n_head_act, n_tokens,
                                 ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                                 ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v) * n_head_act),
                                 ggml_row_size(kv->type, (n_embd_head_qk_nope)));
                cb(v_states, "v_states", il);

                v_states = ggml_cont(ctx0, v_states);
                cb(v_states, "v_states", il);

                v_states = ggml_view_2d(ctx0, v_states, hparams.n_embd_head_v * n_head_act, n_tokens,
                                        ggml_row_size(kv->type, hparams.n_embd_head_v * n_head_act), 0);
                cb(v_states, "v_states", il);

                q_pe = ggml_cont(
                    ctx0,
                    q_pe);  // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor_scaled, beta_fast, beta_slow);
                cb(q_pe, "q_pe", il);

                // shared RoPE key
                k_pe = ggml_cont(
                    ctx0,
                    k_pe);  // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor_scaled, beta_fast, beta_slow);
                cb(k_pe, "k_pe", il);

                struct ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(q_states, "q_states", il);

                struct ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(k_states, "k_states", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, NULL, k_states, v_states, q_states,
                                   KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
            }
        }

        if (lctx.model.params.enable_tensor_parallel && !lctx.enable_dp_gather) {
            cur = ggml_all_reduce_sum(ctx0, cur);
            cb(cur, "all_reduce_sum_aft_attn", il);
        }

        struct ggml_tensor * ffn_inp;

        if (!run_mlp_only) {
            if (il == n_layer - 1 && !lctx.enable_dp_gather) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens                         = n_outputs;
                cur                              = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA                            = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);
        } else {
            ffn_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, 0);
        }

        // TODO: This is a rough implementation for data parallelism, may be optimized later.
        if (lctx.enable_dp_gather) {
            ggml_tensor * all_ffn_inp =
                ggml_new_tensor_2d(ctx0, ffn_inp->type, ffn_inp->ne[0], lctx.all_server_token_sum);
            all_ffn_inp = ggml_to_zero(ctx0, all_ffn_inp);
            if (lctx.self_token_size > 0) {
                GGML_ASSERT(ggml_is_contiguous(ffn_inp));
                if (ffn_inp->ne[1] != lctx.self_token_size) {
                    printf("%ld != %d, n_tokens = %d\n", ffn_inp->ne[1], lctx.self_token_size, ubatch.n_tokens);
                }
                GGML_ASSERT(ffn_inp->ne[1] == lctx.self_token_size);
                ggml_tensor * view_ffn_inp = ggml_view_2d(ctx0, all_ffn_inp, ffn_inp->ne[0], ffn_inp->ne[1],
                                                          all_ffn_inp->nb[1], lctx.self_token_offset * ffn_inp->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, ffn_inp, view_ffn_inp));
            }
            ffn_inp = ggml_all_reduce_sum(ctx0, all_ffn_inp);
            ggml_build_forward_expand(gf, ffn_inp);
            cb(ffn_inp, "all_gather_aft_ffn", il);
        }

        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL,
                                NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out = llm_build_moe_ffn(
                ctx0, lctx, cur, model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps, model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used, expert_group_id, n_expert_groups, LLM_FFN_SILU, hparams.enable_fused_moe,
                hparams.expert_weights_norm, true, hparams.expert_weights_scale,
                (enum llama_expert_gating_func_type) hparams.expert_gating_func, cb, il);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp = llm_build_ffn(
                    ctx0, lctx, cur, model.layers[il].ffn_up_shexp, NULL, NULL, model.layers[il].ffn_gate_shexp, NULL,
                    NULL, model.layers[il].ffn_down_shexp, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }

        if (lctx.model.hparams.enable_tensor_parallel) {
            cur = ggml_all_reduce_sum(ctx0, cur);
            cb(cur, "all_reduce_sum_aft_mlp", il);
        }
        ggml_build_forward_expand(gf, cur);

        cur = ggml_add(ctx0, cur, ffn_inp);

        if (lctx.enable_dp_gather && lctx.self_token_size > 0) {
            cur = ggml_view_2d(ctx0, cur, cur->ne[0], lctx.self_token_size, cur->nb[1],
                               lctx.self_token_offset * cur->nb[1]);
        }

        if (il == n_layer - 1 && lctx.enable_dp_gather) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens                         = n_outputs;
            cur                              = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA                            = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        // Simplifed: adapter is not supported.
        cur = lctx.cvec.apply_to(cur);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

void llm_deepseek2_context::free() {
    ggml_free(ctx0);
    ctx0 = nullptr;
}

struct ggml_cgraph * llm_build_deepseek2(llama_context & lctx, std::vector<uint8_t> & buf_compute_meta,
                                         const llama_ubatch & ubatch, llm_build_cb & cb, bool worst_case,
                                         int print_layer) {
    struct ggml_cgraph * result = NULL;

    llm_deepseek2_context llm(lctx, buf_compute_meta, ubatch, cb, worst_case, print_layer);

    llm.init();

    result = llm.build_deepseek2();

    // add on pooling layer
    GGML_ASSERT(!lctx.cparams.embeddings);

    llm.free();

    return result;
}
