#include "llama-graph-qwen3ge.h"

#include <cmath>
#include <cstdio>
#include <type_traits>

#include "ggml.h"
#include "llama-context.h"
#include "llama-hparams.h"

struct ggml_tensor * llm_qwen3_context_ge::build_attn_indices() {
    lctx.inp_attn_indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    cb(lctx.inp_attn_indices, "inp_attn_indices", -1);
    ggml_set_input(lctx.inp_attn_indices);
    return lctx.inp_attn_indices;
}

struct ggml_tensor * llm_qwen3_context_ge::build_length_q() {
    lctx.inp_length_q = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, 1);
    cb(lctx.inp_length_q, "inp_length_q", -1);
    ggml_set_input(lctx.inp_length_q);
    return lctx.inp_length_q;
}

struct ggml_tensor * llm_qwen3_context_ge::build_length_kv() {
    lctx.inp_length_kv = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, 1);
    cb(lctx.inp_length_kv, "inp_length_kv", -1);
    ggml_set_input(lctx.inp_length_kv);
    return lctx.inp_length_kv;
}

struct ggml_cgraph * llm_qwen3_context_ge::build_qwen3_ge() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    // params changed in parallel
    const int64_t n_split = (hparams.enable_tensor_parallel & !hparams.enable_data_parallel) ? hparams.num_parallel : 1;
    const int64_t n_head_act = n_head / n_split;

    // KV heads handling for GQA with few KV heads
    const bool replicate_kv = (n_head_kv < n_split) || (n_head_kv % n_split != 0);
    int64_t    n_head_kv_act;
    int64_t    kv_head_start = 0;  // Start index of KV heads for this device

    if (replicate_kv && n_head_kv > 0) {
        // Calculate which KV heads this device needs based on its Q heads
        const int64_t tp_id          = hparams.tp_id;
        const int64_t n_rep          = n_head / n_head_kv;  // GQA ratio
        const int64_t n_q_per_device = n_head / n_split;
        kv_head_start                = (tp_id * n_q_per_device) / n_rep;
        const int64_t kv_head_end    = ((tp_id + 1) * n_q_per_device + n_rep - 1) / n_rep;  // ceil
        n_head_kv_act                = kv_head_end - kv_head_start;
    } else {
        n_head_kv_act = n_head_kv / n_split;
    }

    const int64_t expert_group_id = hparams.enable_expert_parallel ? lctx.model.params.tp_id : 0;
    const int64_t n_expert_groups = hparams.enable_expert_parallel ? lctx.model.params.num_parallel : 1;
    const bool    run_mlp_only    = lctx.enable_dp_gather && lctx.self_token_size == 0;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;
    struct ggml_tensor * inp_pos;
    struct ggml_tensor * indices;
    struct ggml_tensor * length_q;
    struct ggml_tensor * length_kv;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb, true);

    // inp_pos - contains the positions
    inp_pos = build_inp_pos();

    // indices for kv cache
    indices = build_attn_indices();

    length_q  = build_length_q();
    length_kv = build_length_kv();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il, true);
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
                const int64_t q_dim       = n_embd_head * n_head_act;
                const int64_t k_dim       = n_embd_head * n_head_kv_act;
                const int64_t v_dim       = n_embd_head * n_head_kv_act;

                struct ggml_tensor * QKVcur = ggml_mul_mat_fp16(ctx0, model.layers[il].wqkv, cur);
                cb(QKVcur, "QKVcur", il);

                // Split QKV into Q, K, V using ggml_get_slice (GE backend doesn't support view)
                // QKVcur shape: [qkv_dim, n_tokens]
                Qcur = ggml_get_slice(ctx0, QKVcur, 0, q_dim, 0);
                cb(Qcur, "Qcur", il);

                Kcur = ggml_get_slice(ctx0, QKVcur, q_dim, q_dim + k_dim, 0);
                cb(Kcur, "Kcur", il);

                Vcur = ggml_get_slice(ctx0, QKVcur, q_dim + k_dim, q_dim + k_dim + v_dim, 0);
                cb(Vcur, "Vcur", il);
            } else {
                // Fallback to separate Q, K, V weights
                Qcur = ggml_mul_mat_fp16(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                Kcur = ggml_mul_mat_fp16(ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                Vcur = ggml_mul_mat_fp16(ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                // If KV is replicated (n_head_kv < n_split), slice out the KV heads this device needs
                // GE backend doesn't support view, use ggml_get_slice instead
                if (replicate_kv && n_head_kv_act < n_head_kv) {
                    const int64_t k_start = kv_head_start * n_embd_head_k;
                    const int64_t k_end   = k_start + n_embd_head_k * n_head_kv_act;
                    Kcur                  = ggml_get_slice(ctx0, Kcur, k_start, k_end, 0);
                    cb(Kcur, "Kcur_sliced", il);

                    const int64_t v_start = kv_head_start * n_embd_head_v;
                    const int64_t v_end   = v_start + n_embd_head_v * n_head_kv_act;
                    Vcur                  = ggml_get_slice(ctx0, Vcur, v_start, v_end, 0);
                    cb(Vcur, "Vcur_sliced", il);
                }
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head_act, n_tokens);
            // apply q_norm
            Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, cb, il, true);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv_act, n_tokens);
            // apply k_norm
            Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, cb, il, true);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv_ge(ctx0, lctx, kv_self, gf, model.layers[il].wo, model.layers[il].bo, Kcur, Vcur, Qcur,
                                  indices, length_q, length_kv, n_tokens, n_kv, kq_scale, cb, il, true);
        }

        if (lctx.model.params.enable_tensor_parallel && !lctx.enable_dp_gather) {
            cur = ggml_all_reduce_sum(ctx0, cur);
            cb(cur, "all_reduce_sum_aft_attn", il);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il, true);
        cb(cur, "ffn_norm", il);

        // Use merged gate-up weight if available for better performance
        if (model.layers[il].ffn_gate_up != nullptr) {
            // Use merged FFN gate-up weight
            struct ggml_tensor * gate_up = ggml_mul_mat_fp16(ctx0, model.layers[il].ffn_gate_up, cur);
            cb(gate_up, "ffn_gate_up", il);

            // Split gate_up into gate and up using ggml_get_slice (GE backend doesn't support view)
            // gate_up shape: [2*n_ff, n_tokens]
            const int64_t        n_ff = hparams.n_ff() / n_split;
            struct ggml_tensor * gate = ggml_get_slice(ctx0, gate_up, 0, n_ff, 0);
            cb(gate, "ffn_gate", il);

            struct ggml_tensor * up = ggml_get_slice(ctx0, gate_up, n_ff, 2 * n_ff, 0);
            cb(up, "ffn_up", il);

            // Apply activation to gate
            gate = ggml_silu(ctx0, gate);
            cb(gate, "ffn_silu", il);

            // Multiply gate and up
            cur = ggml_mul(ctx0, gate, up);
            cb(cur, "ffn_gate_par", il);

            // Down projection
            cur = ggml_mul_mat_fp16(ctx0, model.layers[il].ffn_down, cur);
            cb(cur, "ffn_out", il);
        } else {
            // Fallback to separate gate and up weights
            cur = llm_build_ffn(ctx0, lctx, cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL,
                                NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il,
                                false);
            cb(cur, "ffn_out", il);
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

struct ggml_cgraph * llm_build_qwen3_ge(llama_context & lctx, std::vector<uint8_t> & buf_compute_meta,
                                        const llama_ubatch & ubatch, llm_build_cb & cb, bool worst_case,
                                        int print_layer) {
    struct ggml_cgraph * result = NULL;

    llm_qwen3_context_ge llm(lctx, buf_compute_meta, ubatch, cb, worst_case, print_layer);

    llm.init();

    result = llm.build_qwen3_ge();
    ggml_graph_set_n_ctx(result, lctx.cparams.n_ctx);
    // add on pooling layer
    GGML_ASSERT(!lctx.cparams.embeddings);

    llm.free();

    return result;
}

void llm_update_qwen3_ge(llama_context & lctx) {
    ggml_cgraph * graph   = lctx.graph_decode;
    int           n_nodes = ggml_graph_n_nodes(graph);
    llama_hparams hparams = lctx.model.hparams;
    const int64_t head_split =
        (hparams.enable_tensor_parallel & !hparams.enable_data_parallel) ? hparams.num_parallel : 1;
    const int64_t n_head_kv = hparams.n_head_kv();
    const int64_t n_head    = hparams.n_head();

    // KV heads handling for GQA with few KV heads
    const bool replicate_kv = (n_head_kv < head_split) || (n_head_kv % head_split != 0);
    int64_t    n_head_kv_act;

    if (replicate_kv && n_head_kv > 0) {
        const int64_t tp_id          = hparams.tp_id;
        const int64_t n_rep          = n_head / n_head_kv;
        const int64_t n_q_per_device = n_head / head_split;
        const int64_t kv_head_start  = (tp_id * n_q_per_device) / n_rep;
        const int64_t kv_head_end    = ((tp_id + 1) * n_q_per_device + n_rep - 1) / n_rep;
        n_head_kv_act                = kv_head_end - kv_head_start;
    } else {
        n_head_kv_act = n_head_kv / head_split;
    }

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
            // Use actual KV heads for this device
            params->key_num_heads      = n_head_kv_act;
        }
    }
}
