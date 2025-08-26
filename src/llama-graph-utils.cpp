#include "llama-graph-utils.h"

#include "ggml.h"
#include "llama-batch.h"
#include "llama-context.h"

struct ggml_tensor * llm_build_inp_embd(struct ggml_context * ctx, struct llama_context & lctx,
                                        const llama_hparams & hparams, const llama_ubatch & ubatch,
                                        struct ggml_tensor * tok_embd, const llm_build_cb & cb, bool enable_fp16) {
    const int64_t n_embd = hparams.n_embd;

    struct ggml_tensor * inpL;

    if (ubatch.token) {
        lctx.inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ubatch.n_tokens);
        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);

        inpL = enable_fp16 ? ggml_get_rows_fp16(ctx, tok_embd, lctx.inp_tokens) :
                             ggml_get_rows(ctx, tok_embd, lctx.inp_tokens);

        // apply lora for embedding tokens if needed
        GGML_ASSERT(lctx.lora.empty());
    } else {
        lctx.inp_embd = ggml_new_tensor_2d(ctx, enable_fp16 ? GGML_TYPE_F16 : GGML_TYPE_F32, n_embd, ubatch.n_tokens);
        inpL          = lctx.inp_embd;
        ggml_set_input(lctx.inp_embd);
    }

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx, inpL, hparams.f_embedding_scale);
    }

    cb(inpL, "inp_embd", -1);

    return inpL;
}

// do mat_mul, while optionally apply lora
struct ggml_tensor * llm_build_lora_mm(struct llama_context & lctx, struct ggml_context * ctx0, struct ggml_tensor * w,
                                       struct ggml_tensor * cur, bool enable_fp16) {
    struct ggml_tensor * res = nullptr;
    if (enable_fp16) {
        res = ggml_mul_mat_fp16(ctx0, w, cur);
    } else {
        res = ggml_mul_mat(ctx0, w, cur);
    }
    GGML_ASSERT(lctx.lora.empty());
    return res;
}

// do mat_mul_id, while optionally apply lora
struct ggml_tensor * llm_build_lora_mm_id(struct llama_context & lctx, struct ggml_context * ctx0,
                                          struct ggml_tensor * w,    // struct ggml_tensor * as
                                          struct ggml_tensor * cur,  // struct ggml_tensor * b
                                          struct ggml_tensor * ids) {
    struct ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    GGML_ASSERT(lctx.lora.empty());
    return res;
}

struct ggml_tensor * llm_build_norm(struct ggml_context * ctx, struct ggml_tensor * cur, const llama_hparams & hparams,
                                    struct ggml_tensor * mw, struct ggml_tensor * mb, llm_norm_type type,
                                    const llm_build_cb & cb, int il, bool fuse_mul) {
    switch (type) {
        case LLM_NORM:
            cur = ggml_norm(ctx, cur, hparams.f_norm_eps);
            break;
        case LLM_NORM_RMS:
            if (fuse_mul) {
                GGML_ASSERT(mb == nullptr);
                if (mw) {
                    return ggml_rms_norm_fused(ctx, cur, mw, hparams.f_norm_rms_eps);
                }
                return ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);
            }
            cur = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);
            break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[2]);
            }
            break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        // cast mw to fp16
        mw  = ggml_cast(ctx, mw, GGML_TYPE_F16);
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx, cur, mb);
    }

    return cur;
}

void llm_build_context::init() {
    struct ggml_init_params params = {
        /*.mem_size   =*/lctx.buf_compute_meta.size(),
        /*.mem_buffer =*/lctx.buf_compute_meta.data(),
        /*.no_alloc   =*/true,
    };

    ctx0 = ggml_init(params);
}

void llm_build_context::free() {
    ggml_free(ctx0);
}
