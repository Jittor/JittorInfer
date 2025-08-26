#include "llama-graph-deepseek2ge.h"

#include <cmath>
#include <type_traits>

#include "ggml.h"
#include "llama-context.h"

struct ggml_tensor * llm_deepseek2_context_ge::build_attn_indices() {
    lctx.inp_attn_indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    cb(lctx.inp_attn_indices, "inp_attn_indices", -1);
    ggml_set_input(lctx.inp_attn_indices);
    return lctx.inp_attn_indices;
}

struct ggml_tensor * llm_deepseek2_context_ge::build_length_q() {
    lctx.inp_length_q = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, 1);
    cb(lctx.inp_length_q, "inp_length_q", -1);
    ggml_set_input(lctx.inp_length_q);
    return lctx.inp_length_q;
}

struct ggml_tensor * llm_deepseek2_context_ge::build_length_kv() {
    lctx.inp_length_kv = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, 1);
    cb(lctx.inp_length_kv, "inp_length_kv", -1);
    ggml_set_input(lctx.inp_length_kv);
    return lctx.inp_length_kv;
}

struct ggml_cgraph * llm_deepseek2_context_ge::build_deepseek2_ge() {
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
    const int64_t expert_group_id = hparams.enable_expert_parallel ? lctx.model.params.tp_id : 0;
    const int64_t n_expert_groups = hparams.enable_expert_parallel ? lctx.model.params.num_parallel : 1;
    const bool    run_mlp_only    = lctx.enable_dp_gather && lctx.self_token_size == 0;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;
    struct ggml_tensor * inp_pos;
    struct ggml_tensor * indices;
    struct ggml_tensor * length_q;
    struct ggml_tensor * length_kv;

    GGML_ASSERT(!run_mlp_only);
    GGML_ASSERT(!hparams.enable_tensor_parallel);
    GGML_ASSERT(!hparams.enable_data_parallel);
    GGML_ASSERT(!hparams.enable_expert_parallel);

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb, true);

    // cast inpL to fp16
    // inpL = ggml_cast(ctx0, inpL, GGML_TYPE_F16);
    // inp_pos - contains the positions
    inp_pos = build_inp_pos();

    // indices for kv cache
    indices = build_attn_indices();

    length_q  = build_length_q();
    length_kv = build_length_kv();

    for (int il = 0; il < n_layer; ++il) {
        if (print_layer_ >= 0 && il != print_layer_) {
            continue;
        }
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il, true);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            struct ggml_tensor * q = NULL;
            if (!is_lite) {
                // {n_embd, q_lora_rank} * {n_embd, n_tokens} -> {q_lora_rank, n_tokens}
                q = ggml_mul_mat_fp16(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = llm_build_norm(ctx0, q, hparams, model.layers[il].attn_q_a_norm, NULL, LLM_NORM_RMS, cb, il, true);
                cb(q, "q", il);

                // {q_lora_rank, n_head * hparams.n_embd_head_k} * {q_lora_rank, n_tokens} -> {n_head * hparams.n_embd_head_k, n_tokens}
                q = ggml_mul_mat_fp16(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);
            } else {
                q = ggml_mul_mat_fp16(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
            }

            q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
            GGML_ASSERT(n_embd_head_k == n_embd_head_qk_nope + n_embd_head_qk_rope);

            // q_nope = q[:, :, :n_embd_head_qk_nope]
            ggml_tensor * q_nope = ggml_get_slice(ctx0, q, 0, n_embd_head_qk_nope, 0);
            ggml_set_name(q_nope, "q_nope");

            // q_rope = q[:, :, n_embd_head_qk_nope:]
            ggml_tensor * q_pe = ggml_get_slice(ctx0, q, n_embd_head_qk_nope, n_embd_head_k, 0);
            ggml_set_name(q_pe, "q_pe");

            // {n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
            struct ggml_tensor * kv_pe_compresseed = ggml_mul_mat_fp16(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_pe_compresseed, "kv_pe_compresseed", il);

            // kv_compressed = kv_pe_compresseed[:, :kv_lora_rank]
            struct ggml_tensor * kv_compressed = ggml_get_slice(ctx0, kv_pe_compresseed, 0, kv_lora_rank, 0);
            ggml_set_name(kv_compressed, "kv_compressed");

            // k_pe = kv_pe_compresseed[:, kv_lora_rank:]
            ggml_tensor * k_pe =
                ggml_get_slice(ctx0, kv_pe_compresseed, kv_lora_rank, kv_lora_rank + n_embd_head_qk_rope, 0);
            k_pe = ggml_reshape_3d(ctx0, k_pe, n_embd_head_qk_rope, 1, n_tokens);
            ggml_set_name(k_pe, "k_pe");

            kv_compressed = llm_build_norm(ctx0, kv_compressed, hparams, model.layers[il].attn_kv_a_norm, NULL,
                                           LLM_NORM_RMS, cb, il, true);
            cb(kv_compressed, "kv_compressed", il);

            if (hparams.enable_mla) {
                GGML_ABORT("mla not supported now.");
            } else {
                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                struct ggml_tensor * kv = ggml_mul_mat_fp16(ctx0, model.layers[il].wkv_b, kv_compressed);
                kv = ggml_reshape_3d(ctx0, kv, n_embd_head_qk_nope + n_embd_head_v, n_head, n_tokens);
                cb(kv, "kv", il);

                // k_nope = kv[:, :, :n_embd_head_qk_nope]
                struct ggml_tensor * k_nope = ggml_get_slice(ctx0, kv, 0, n_embd_head_qk_nope, 0);
                ggml_set_name(k_nope, "k_nope");

                // v_states = kv[:, :, n_embd_head_qk_nope:]
                struct ggml_tensor * v_states =
                    ggml_get_slice(ctx0, kv, n_embd_head_qk_nope, n_embd_head_qk_nope + n_embd_head_v, 0);
                v_states = ggml_reshape_2d(ctx0, v_states, n_embd_head_v * n_head, n_tokens);
                ggml_set_name(v_states, "v_states");

                // cast q_pe to fp32
                // q_pe = ggml_cast(ctx0, q_pe, GGML_TYPE_F32);
                // k_pe = ggml_cast(ctx0, k_pe, GGML_TYPE_F32);

                q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor_scaled, beta_fast, beta_slow);
                cb(q_pe, "q_pe", il);

                // shared RoPE key
                k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor_scaled, beta_fast, beta_slow);
                cb(k_pe, "k_pe", il);

                // cast q_pe and k_pe to fp16
                // q_pe = ggml_cast(ctx0, q_pe, GGML_TYPE_F16);
                // k_pe = ggml_cast(ctx0, k_pe, GGML_TYPE_F16);

                struct ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(q_states, "q_states", il);

                struct ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(k_states, "k_states", il);

                cur = llm_build_kv_ge(ctx0, lctx, kv_self, gf, model.layers[il].wo, NULL, k_states, v_states, q_states,
                                      indices, length_q, length_kv, n_tokens, n_kv, kq_scale, cb, il, true);
            }
        }

        if (lctx.model.params.enable_tensor_parallel && !lctx.enable_dp_gather) {
            cur = ggml_all_reduce_sum(ctx0, cur);
            cb(cur, "all_reduce_sum_aft_attn", il);
        }

        struct ggml_tensor * ffn_inp;

        ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        if (lctx.enable_dp_gather) {
            GGML_ABORT("dp is not implemented.");
        }

        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il, true);
        cb(cur, "ffn_norm", il);

        // TODO(hsh): remove this cast op
        // cast cur to fp32
        // cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL,
                                NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il,
                                false);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out = llm_build_moe_ffn(
                ctx0, lctx, cur, model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps, model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used, expert_group_id, n_expert_groups, LLM_FFN_SILU, hparams.enable_fused_moe,
                hparams.expert_weights_norm, true, hparams.expert_weights_scale,
                (enum llama_expert_gating_func_type) hparams.expert_gating_func, cb, il, false);
            cb(moe_out, "ffn_moe_out", il);
            // moe_out = ggml_cast(ctx0, moe_out, GGML_TYPE_F16);
            // cb(moe_out, "ffn_moe_out_cast", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp = llm_build_ffn(
                    ctx0, lctx, cur, model.layers[il].ffn_up_shexp, NULL, NULL, model.layers[il].ffn_gate_shexp, NULL,
                    NULL, model.layers[il].ffn_down_shexp, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, false);
                cb(ffn_shexp, "ffn_shexp", il);
                // cur = ggml_cast(ctx0, moe_out, GGML_TYPE_F32);
                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }

        if (lctx.model.hparams.enable_tensor_parallel) {
            cur = ggml_all_reduce_sum(ctx0, cur);
            cb(cur, "all_reduce_sum_aft_mlp", il);
        }
        ggml_build_forward_expand(gf, cur);
        // cast cur to fp16
        if (cur->type != GGML_TYPE_F16) {
            cur = ggml_cast(ctx0, cur, GGML_TYPE_F16);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        if (lctx.enable_dp_gather && lctx.self_token_size > 0) {
            GGML_ABORT("dp is not implemented.");
        }

        cur = lctx.cvec.apply_to(cur);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = ggml_get_rows(ctx0, cur, build_inp_out_ids());

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1, true);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

struct ggml_cgraph * llm_build_deepseek2_ge(llama_context & lctx, std::vector<uint8_t> & buf_compute_meta,
                                            const llama_ubatch & ubatch, llm_build_cb & cb, bool worst_case,
                                            int print_layer) {
    struct ggml_cgraph * result = NULL;

    llm_deepseek2_context_ge llm(lctx, buf_compute_meta, ubatch, cb, worst_case, print_layer);

    llm.init();

    result = llm.build_deepseek2_ge();
    ggml_graph_set_n_ctx(result, lctx.cparams.n_ctx);
    // add on pooling layer
    GGML_ASSERT(!lctx.cparams.embeddings);

    llm.free();

    return result;
}

void llm_update_deepseek2_ge(llama_context & lctx) {
    ggml_cgraph * graph   = lctx.graph_decode;
    int           n_nodes = ggml_graph_n_nodes(graph);

    struct flash_attn_params {
        int     batch_size;
        int     num_heads;
        int     head_dim_kq;
        int     head_dim_v;
        int     key_num_heads;
        int     sequence_lenth_q;
        int64_t sequence_lenth_kv;
        float   scaleValue;
    };

    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * cur = ggml_graph_node(graph, i);
        if (cur->op == GGML_OP_FLASH_ATTN_PROMPT) {
            flash_attn_params * params = reinterpret_cast<flash_attn_params *>(cur->op_params);
            params->sequence_lenth_kv  = lctx.kv_self.n;
        }
    }
}
