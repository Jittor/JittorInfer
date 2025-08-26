#include "ggml.h"
#include "llama-graph-utils.h"

struct ggml_tensor * llm_build_ffn(struct ggml_context * ctx, struct llama_context & lctx, struct ggml_tensor * cur,
                                   struct ggml_tensor * up, struct ggml_tensor * up_b, struct ggml_tensor * up_s,
                                   struct ggml_tensor * gate, struct ggml_tensor * gate_b, struct ggml_tensor * gate_s,
                                   struct ggml_tensor * down, struct ggml_tensor * down_b, struct ggml_tensor * down_s,
                                   struct ggml_tensor * act_scales, llm_ffn_op_type type_op,
                                   llm_ffn_gate_type type_gate, const llm_build_cb & cb, int il, bool enable_fp16) {
    struct ggml_tensor * tmp = up ? llm_build_lora_mm(lctx, ctx, up, cur, enable_fp16) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = llm_build_lora_mm(lctx, ctx, gate, tmp, enable_fp16);
                    cb(cur, "ffn_gate", il);
                }
                break;
            case LLM_FFN_PAR:
                {
                    cur = llm_build_lora_mm(lctx, ctx, gate, cur, enable_fp16);
                    cb(cur, "ffn_gate", il);
                }
                break;
        }

        if (gate_b) {
            cur = ggml_add(ctx, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            {
                cur = ggml_silu(ctx, cur);
                cb(cur, "ffn_silu", il);
            }
            break;
        case LLM_FFN_GELU:
            {
                cur = ggml_gelu(ctx, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            }
            break;
        case LLM_FFN_RELU:
            {
                cur = ggml_relu(ctx, cur);
                cb(cur, "ffn_relu", il);
            }
            break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx, cur);
                cb(cur, "ffn_sqr(relu)", il);
            }
            break;
        case LLM_FFN_SWIGLU:
            {
                // Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
                int64_t              split_point = cur->ne[0] / 2;
                struct ggml_tensor * x0 =
                    ggml_cont(ctx, ggml_view_2d(ctx, cur, split_point, cur->ne[1], cur->nb[1], 0));
                struct ggml_tensor * x1 = ggml_cont(ctx, ggml_view_2d(ctx, cur, split_point, cur->ne[1], cur->nb[1],
                                                                      split_point * ggml_element_size(cur)));

                x0 = ggml_silu(ctx, x0);
                cb(cur, "ffn_silu", il);

                cur = ggml_mul(ctx, x0, x1);
                cb(cur, "ffn_mul", il);
            }
            break;
    }

    if (type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = llm_build_lora_mm(lctx, ctx, down, cur, enable_fp16);
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

struct ggml_tensor * llm_build_moe_ffn(struct ggml_context * ctx, struct llama_context & lctx, struct ggml_tensor * cur,
                                       struct ggml_tensor * gate_inp, struct ggml_tensor * up_exps,
                                       struct ggml_tensor * gate_exps, struct ggml_tensor * down_exps,
                                       struct ggml_tensor * exp_probs_b, int64_t n_expert, int64_t n_expert_used,
                                       int64_t expert_group_id, int64_t n_expert_groups, llm_ffn_op_type type_op,
                                       bool enable_fused_moe, bool norm_w, bool scale_w, float w_scale,
                                       llama_expert_gating_func_type gating_op, const llm_build_cb & cb, int il,
                                       bool enable_fp16) {
    ggml_tensor * cur_f32;
    if (cur->type != GGML_TYPE_F32) {
        cur_f32 = ggml_cast(ctx, cur, GGML_TYPE_F32);
        cb(cur_f32, "moe_cast", il);
    } else {
        cur_f32 = cur;
    }
    int64_t n_embd   = cur->ne[0];
    int64_t n_tokens = cur->ne[1];

    ggml_tensor * logits = llm_build_lora_mm(lctx, ctx, gate_inp, cur_f32);  // [n_expert, n_tokens]
    cb(logits, "ffn_moe_logits", il);

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx, logits);  // [n_expert, n_tokens]
            }
            break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx, logits);  // [n_expert, n_tokens]
            }
            break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select experts
    ggml_tensor * ffn_moe_argsort  = ggml_argsort(ctx, selection_probs, GGML_SORT_ORDER_DESC);
    ggml_tensor * selected_experts = ggml_get_slice(ctx, ffn_moe_argsort, 0, n_expert_used, 0);
    cb(ffn_moe_argsort, "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens),
                                          selected_experts);  // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);

    if (norm_w) {
        weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights);  // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        weights = ggml_div(ctx, weights, weights_sum);  // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    }
    if (scale_w) {
        weights = ggml_scale(ctx, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    cur                   = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    ggml_tensor * experts = nullptr;

    // aggregate experts
    ggml_tensor * moe_out = nullptr;

    if (enable_fused_moe) {
        GGML_ASSERT(n_expert % n_expert_groups == 0);
        if (n_tokens == 0) {
            moe_out = ggml_new_tensor_2d(ctx, cur->type, n_embd, n_tokens);
            cb(moe_out, "moe_out_after_cast", il);
            return moe_out;
        }
        const int32_t start_expert        = expert_group_id * (n_expert / n_expert_groups);
        const int32_t end_expert          = (expert_group_id + 1) * (n_expert / n_expert_groups);
        ggml_tensor * selected_experts_id = selected_experts;
        cb(selected_experts_id, "selected_experts_id", il);
        ggml_tensor * row_idx       = ggml_arange(ctx, 0, n_expert_used * n_tokens, 1);
        row_idx                     = ggml_reshape_2d(ctx, row_idx, n_tokens, n_expert_used);
        ggml_tensor * row_idx_int32 = ggml_cast(ctx, row_idx, GGML_TYPE_I32);
        ggml_tensor * cur_new;
        if (cur->type != up_exps->type) {
            cur_new = ggml_cast(ctx, cur, up_exps->type);
            cb(cur_new, "casted_hidden_states", il);
        } else {
            cur_new = cur;
        }
        if (enable_fp16) {
            moe_out = ggml_moe_fused_fp16(ctx, cur_new, selected_experts_id, weights, up_exps, down_exps, gate_exps,
                                          row_idx_int32, start_expert, end_expert - 1);
        } else {
            moe_out = ggml_moe_fused(ctx, cur_new, selected_experts_id, weights, up_exps, down_exps, gate_exps,
                                     row_idx_int32, start_expert, end_expert - 1);
        }

        cb(moe_out, "moe_out_after_cast", il);
    } else {
        ggml_tensor * up =
            llm_build_lora_mm_id(lctx, ctx, up_exps, cur, selected_experts);  // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        ggml_tensor * gate =
            llm_build_lora_mm_id(lctx, ctx, gate_exps, cur, selected_experts);  // [n_ff, n_expert_used, n_tokens]
        cb(gate, "ffn_moe_gate", il);

        switch (type_op) {
            case LLM_FFN_SILU:
                {
                    gate = ggml_silu(ctx, gate);
                    cb(gate, "ffn_moe_silu", il);
                }
                break;
            case LLM_FFN_GELU:
                {
                    gate = ggml_gelu(ctx, gate);
                    cb(gate, "ffn_moe_gelu", il);
                }
                break;
            default:
                GGML_ABORT("fatal error");
        }

        ggml_tensor * par = ggml_mul(ctx, up, gate);  // [n_ff, n_expert_used, n_tokens]
        cb(par, "ffn_moe_gate_par", il);

        experts =
            llm_build_lora_mm_id(lctx, ctx, down_exps, par, selected_experts);  // [n_embd, n_expert_used, n_tokens]
        cb(experts, "ffn_moe_down", il);

        experts = ggml_mul(ctx, experts, weights);

        for (int i = 0; i < n_expert_used; ++i) {
            ggml_tensor * cur_expert = ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], i * experts->nb[1]);

            if (i == 0) {
                moe_out = cur_expert;
            } else {
                moe_out = ggml_add(ctx, moe_out, cur_expert);
            }
        }

        if (n_expert_used == 1) {
            // avoid returning a non-contiguous tensor
            moe_out = ggml_cont(ctx, moe_out);
        }
    }

    return moe_out;
}

struct ggml_tensor * llm_build_moe_ffn_ge(struct ggml_context * ctx, struct llama_context & lctx,
                                          struct ggml_tensor * cur, struct ggml_tensor * gate_inp,
                                          struct ggml_tensor * up_exps, struct ggml_tensor * gate_exps,
                                          struct ggml_tensor * down_exps, struct ggml_tensor * exp_probs_b,
                                          int64_t n_expert, int64_t n_expert_used, int64_t expert_group_id,
                                          int64_t n_expert_groups, llm_ffn_op_type type_op, bool enable_fused_moe,
                                          bool norm_w, bool scale_w, float w_scale,
                                          llama_expert_gating_func_type gating_op, const llm_build_cb & cb, int il) {
    int64_t n_embd   = cur->ne[0];
    int64_t n_tokens = cur->ne[1];

    ggml_tensor * logits = llm_build_lora_mm(lctx, ctx, gate_inp, cur);  // [n_expert, n_tokens]
    cb(logits, "ffn_moe_logits", il);

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx, logits);  // [n_expert, n_tokens]
            }
            break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx, logits);  // [n_expert, n_tokens]
            }
            break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select experts
    ggml_tensor * selected_experts = ggml_top_k(ctx, selection_probs, n_expert_used);  // [n_expert_used, n_tokens]
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens),
                                          selected_experts);  // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);

    if (norm_w) {
        weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights);  // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        weights = ggml_div(ctx, weights, weights_sum);  // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    }
    if (scale_w) {
        weights = ggml_scale(ctx, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    cur = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);

    // aggregate experts
    ggml_tensor * moe_out = nullptr;

    if (enable_fused_moe) {
        GGML_ASSERT(n_expert % n_expert_groups == 0);
        if (n_tokens == 0) {
            moe_out = ggml_new_tensor_2d(ctx, cur->type, n_embd, n_tokens);
            cb(moe_out, "moe_out_after_cast", il);
            return moe_out;
        }
        const int32_t start_expert        = expert_group_id * (n_expert / n_expert_groups);
        const int32_t end_expert          = (expert_group_id + 1) * (n_expert / n_expert_groups);
        ggml_tensor * selected_experts_id = selected_experts;
        cb(selected_experts_id, "selected_experts_id", il);
        ggml_tensor * row_idx       = ggml_arange(ctx, 0, n_expert_used * n_tokens, 1);
        row_idx                     = ggml_reshape_2d(ctx, row_idx, n_tokens, n_expert_used);
        ggml_tensor * row_idx_int32 = ggml_cast(ctx, row_idx, GGML_TYPE_I32);
        ggml_tensor * cur_new       = ggml_cast(ctx, cur, up_exps->type);
        cb(cur_new, "casted_hidden_states", il);
        // weights = ggml_cast(ctx, weights, cur_new->type);
        moe_out = ggml_moe_fused(ctx, cur_new, selected_experts_id, weights, up_exps, down_exps, gate_exps,
                                 row_idx_int32, start_expert, end_expert - 1);
        // moe_out = ggml_cast(ctx, moe_out, GGML_TYPE_F32);
        cb(moe_out, "moe_out_after_cast", il);
    } else {
        GGML_ABORT("fatal error");
    }

    return moe_out;
}
