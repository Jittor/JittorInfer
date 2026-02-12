#include "llama-context.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama-impl.h"
#include "llama.h"

#ifdef LLAMA_MPI_SUPPORT
#    define OMPI_SKIP_MPICXX 1
#    include <mpi.h>
#endif

void llama_free(struct llama_context * ctx) {
    delete ctx;
}

void llama_set_abort_callback(struct llama_context * ctx, bool (*abort_callback)(void * data),
                              void *                 abort_callback_data) {
    ctx->abort_callback      = abort_callback;
    ctx->abort_callback_data = abort_callback_data;

    for (auto & backend : ctx->backends) {
        auto * reg                   = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_set_abort_callback");
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), ctx->abort_callback, ctx->abort_callback_data);
        }
    }
}

void llama_clear_adapter_lora(struct llama_context * ctx) {
    ctx->lora.clear();
}

void llama_kv_cache_clear(struct llama_context * ctx) {
    llama_kv_cache_clear(ctx->kv_self);
    llama_kv_cache_restore_rope_cache(ctx->kv_self);
}

void llama_kv_cache_restore_rope_cache(struct llama_context * ctx) {
    llama_kv_cache_restore_rope_cache(ctx->kv_self);
}

void llama_set_k_shift(struct llama_context & lctx) {
    const int64_t kv_size = lctx.kv_self.size;

    assert(ggml_backend_buffer_is_host(lctx.inp_K_shift->buffer));

    int32_t * data = (int32_t *) lctx.inp_K_shift->data;

    for (int i = 0; i < kv_size; ++i) {
        data[i] = lctx.kv_self.cells[i].delta;
    }
}

bool llama_kv_cache_can_shift(struct llama_context * ctx) {
    return llama_kv_cache_can_shift(ctx->kv_self);
}

void llama_synchronize(struct llama_context * ctx) {
    ggml_backend_sched_synchronize(ctx->sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (ctx->n_queued_tokens == 1) {
        if (!ctx->cparams.no_perf) {
            ctx->t_eval_us += ggml_time_us() - ctx->t_compute_start_us;
        }
        ctx->n_eval++;
    } else if (ctx->n_queued_tokens > 1) {
        if (!ctx->cparams.no_perf) {
            ctx->t_p_eval_us += ggml_time_us() - ctx->t_compute_start_us;
        }
        ctx->n_p_eval += ctx->n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (ctx->n_queued_tokens > 0 && !ctx->has_evaluated_once) {
        ctx->t_load_us          = ggml_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }

    ctx->n_queued_tokens    = 0;
    ctx->t_compute_start_us = 0;
}

const struct llama_model * llama_get_model(const struct llama_context * ctx) {
    return &ctx->model;
}

enum llama_pooling_type llama_pooling_type(const struct llama_context * ctx) {
    return ctx->cparams.pooling_type;
}

llama_logits llama_get_logits_ith(struct llama_context * ctx, int32_t i) {
    if (i < 0) {
        i += ctx->output_ids.size();
    }
    GGML_ASSERT(i >= 0);
    GGML_ASSERT((size_t) i < ctx->output_ids.size());
    int32_t j = ctx->output_ids[i];
    GGML_ASSERT(j >= 0);

    // printf("### TEST: j=%d, n_outputs=%d\n", j, ctx->n_outputs);
    // FIXME: this would be false positive if ge is enabled
    /*
    if (j >= ctx->n_outputs) {
        printf("### ERROR: j=%d, n_outputs=%d\n", j, ctx->n_outputs);
        // This should not happen
        throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, ctx->n_outputs));
    }
    */

    return ctx->logits.next(j);
}

void llama_attach_threadpool(struct llama_context * ctx, ggml_threadpool_t threadpool,
                             ggml_threadpool_t threadpool_batch) {
    ctx->threadpool       = threadpool;
    ctx->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

size_t llama_output_reserve(struct llama_context & lctx, size_t n_outputs) {
    const auto & cparams = lctx.cparams;
    const auto & hparams = lctx.model.hparams;
    const auto & vocab   = lctx.model.vocab;

    const size_t n_outputs_max = std::max(n_outputs, (size_t) cparams.n_seq_max);

    const auto n_batch = cparams.n_batch;
    const auto n_vocab = vocab.n_tokens();
    const auto n_embd  = hparams.n_embd;

    // TODO: use a per-batch flag for logits presence instead
    const bool   has_logits = !cparams.embeddings;
    const bool   has_embd   = cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE);
    const size_t embd_size  = has_embd ? n_embd * n_outputs_max : 0;

    if (lctx.output_ids.empty()) {
        // init, never resized afterwards
        lctx.output_ids.resize(n_batch);
    }

    llama_logits::llama_logits_type logits_type;
    int64_t                         logits_len;
    if (lctx.cparams.presample_count == -1) {
        logits_type = llama_logits::LLAMA_LOGITS_TYPE_RAW;
        logits_len  = n_vocab;
    } else {
        GGML_ASSERT(lctx.cparams.presample_count > 0);
        logits_type = llama_logits::LLAMA_LOGITS_TYPE_TOPK;
        logits_len  = lctx.cparams.presample_count;
    }

    const size_t logits_size = has_logits ? llama_logits::get_size(logits_type, logits_len, n_outputs_max) : 0;
    const size_t new_size    = logits_size + embd_size * sizeof(float);

    const size_t prev_size = lctx.buf_output ? ggml_backend_buffer_get_size(lctx.buf_output.get()) : 0;

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!lctx.buf_output || prev_size < new_size) {
        if (lctx.buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_INFO("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__,
                           prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            lctx.buf_output = nullptr;
            lctx.embd       = nullptr;
        }

        auto * buft                 = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev           = lctx.model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        lctx.buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (lctx.buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__,
                            new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    char * output_base = (char *) ggml_backend_buffer_get_base(lctx.buf_output.get());

    if (has_logits) {
        lctx.logits.init(output_base, logits_type, logits_len, n_outputs_max);
        output_base += logits_size;
    }
    if (has_embd) {
        lctx.embd = reinterpret_cast<float *>(output_base);
    }

    lctx.output_size = n_outputs_max;
    lctx.embd_size   = embd_size;

    // set all ids as invalid (negative)
    // is it really unused?? I think it is not used.
    // memset(lctx.output_ids.data(), 0xFF, lctx.output_ids.size() * sizeof(int));
    // ggml_backend_buffer_clear(lctx.buf_output.get(), 0);

    lctx.n_outputs = 0;

    return n_outputs_max;
}

// llama input

static int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket   = 0;
    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }
    int32_t relative_position_if_large =
        floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) /
                               log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);
    return relative_bucket;
}

void llama_set_inputs(llama_context & lctx, const llama_ubatch & ubatch) {
    //
    // set input data
    //

    const auto & hparams = lctx.model.hparams;
    const auto & cparams = lctx.cparams;
    const auto & kv_self = lctx.kv_self;

    if (ubatch.token) {
        const int64_t n_tokens = ubatch.n_tokens;

        if (cparams.enable_ge) {
            std::vector<llama_token> inp_tokens(cparams.n_ubatch);
            for (int i = 0; i < n_tokens; i++) {
                inp_tokens[i] = ubatch.token[i];
            }
            for (int i = n_tokens; i < cparams.n_ubatch; i++) {
                inp_tokens[i] = 0;
            }
            ggml_backend_tensor_set(lctx.inp_tokens, inp_tokens.data(), 0,
                                    cparams.n_ubatch * ggml_element_size(lctx.inp_tokens));
        } else {
            ggml_backend_tensor_set(lctx.inp_tokens, ubatch.token, 0, n_tokens * ggml_element_size(lctx.inp_tokens));
        }
    }

    if (ubatch.embd) {
        const int64_t n_embd   = hparams.n_embd;
        const int64_t n_tokens = ubatch.n_tokens;

        ggml_backend_tensor_memset(lctx.inp_embd, 0, 0, cparams.n_ubatch * n_embd * ggml_element_size(lctx.inp_embd));
        ggml_backend_tensor_set(lctx.inp_embd, ubatch.embd, 0, n_tokens * n_embd * ggml_element_size(lctx.inp_embd));
    }

    if (ubatch.pos && lctx.inp_pos) {
        const int64_t n_tokens = ubatch.n_tokens;
        auto          n_pos    = lctx.n_pos_per_token;
        ggml_backend_tensor_memset(lctx.inp_pos, 0, 0, ggml_nbytes(lctx.inp_pos));
        ggml_backend_tensor_set(lctx.inp_pos, ubatch.pos, 0, n_tokens * n_pos * ggml_element_size(lctx.inp_pos));
    }

    if (hparams.causal_attn || cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        //GGML_ASSERT(lctx.inp_out_ids && "every model that can must skip unused outputs");

        if (!lctx.inp_out_ids) {
            LLAMA_LOG_WARN("%s: 'lctx.inp_out_ids' is not created\n", __func__);
        } else {
            const int64_t n_tokens = ubatch.n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_out_ids->buffer));
            int32_t * data = (int32_t *) lctx.inp_out_ids->data;

            if (lctx.n_outputs == n_tokens) {
                for (int i = 0; i < n_tokens; ++i) {
                    data[i] = i;
                }
            } else if (ubatch.output) {
                int32_t n_outputs = 0;
                for (int i = 0; i < n_tokens; ++i) {
                    if (ubatch.output[i]) {
                        data[n_outputs++] = i;
                    }
                }
                // the graph needs to have been passed the correct number of outputs
                GGML_ASSERT(lctx.n_outputs == n_outputs);
            } else if (lctx.n_outputs == 1) {
                // only keep last output
                data[0] = n_tokens - 1;
            } else {
                GGML_ASSERT(lctx.n_outputs == 0);
            }
            int32_t dummy_idx = 0;
            while (dummy_idx < cparams.n_ubatch) {
                bool flag = true;
                for (int i = 0; i < lctx.n_outputs; ++i) {
                    if (data[i] == dummy_idx) {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    break;
                }
                dummy_idx++;
            }
            if (dummy_idx < cparams.n_ubatch) {
                for (int i = lctx.n_outputs; i < lctx.inp_out_ids->ne[0]; ++i) {
                    data[i] = n_tokens;
                }
            }
        }
    }

    GGML_ASSERT(
        // (!a || b) is a logical implication (a -> b)
        // !hparams.causal_attn -> !cparams.causal_attn
        (hparams.causal_attn || !cparams.causal_attn) && "causal attention is not supported by this model");

    if (lctx.inp_KQ_mask || lctx.inp_KQ_mask_swa || lctx.inp_KQ_mask_i8) {
        // NOTE: hparams.causal_attn indicates the model is capable of generation and uses the kv cache.
        GGML_ASSERT(cparams.causal_attn && !lctx.is_encoding);
        {
            const int64_t n_kv         = kv_self.n;
            const int64_t n_tokens     = ubatch.n_tokens;
            const int64_t n_seq_tokens = ubatch.n_seq_tokens;
            const int64_t n_seqs       = ubatch.n_seqs;

            float *  data     = nullptr;
            float *  data_swa = nullptr;
            int8_t * data_i8  = nullptr;

            if (lctx.inp_KQ_mask) {
                GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask->buffer));
                data = (float *) lctx.inp_KQ_mask->data;
            }

            if (lctx.inp_KQ_mask_swa) {
                GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask_swa->buffer));
                data_swa = (float *) lctx.inp_KQ_mask_swa->data;
            }

            if (lctx.inp_KQ_mask_i8) {
                GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask_i8->buffer));
                data_i8 = (int8_t *) lctx.inp_KQ_mask_i8->data;
            }

            // For causal attention, use only the previous KV cells
            // of the correct sequence for each token of the ubatch.
            // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
            for (int h = 0; h < 1; ++h) {
                for (int s = 0; s < n_seqs; ++s) {
                    const llama_seq_id seq_id = ubatch.seq_id[s][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const llama_pos pos = ubatch.pos[s * n_seq_tokens + j];

                        for (int i = 0; i < n_kv; ++i) {
                            float  f;
                            int8_t i8;
                            if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                                f  = -INFINITY;
                                i8 = 1;
                            } else {
                                if (hparams.use_alibi) {
                                    f = -std::abs(kv_self.cells[i].pos - pos);
                                } else {
                                    f = 0.0f;
                                }
                                i8 = 0;
                            }

                            if (data) {
                                data[h * (n_kv * n_tokens) + s * (n_kv * n_seq_tokens) + j * n_kv + i] = f;
                            }

                            // may need to cut off old tokens for sliding window
                            if (data_swa) {
                                if (pos - kv_self.cells[i].pos >= (int32_t) hparams.n_swa) {
                                    f = -INFINITY;
                                }
                                data_swa[h * (n_kv * n_tokens) + s * (n_kv * n_seq_tokens) + j * n_kv + i] = f;
                            }

                            if (data_i8) {
                                data_i8[h * (n_kv * n_tokens) + s * (n_kv * n_seq_tokens) + j * n_kv + i] = i8;
                            }
                        }
                    }
                }

                if (data) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data[h * (n_kv * n_tokens) + i * n_kv + j] = -INFINITY;
                        }
                    }
                }

                if (data_swa) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data_swa[h * (n_kv * n_tokens) + i * n_kv + j] = -INFINITY;
                        }
                    }
                }

                if (data_i8) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data_i8[h * (n_kv * n_tokens) + i * n_kv + j] = 1;
                        }
                    }
                }
            }
        }
    }

    if (lctx.inp_attn_indices) {
        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_attn_indices->buffer));
        int32_t * data_attn_indices = (int32_t *) lctx.inp_attn_indices->data;
        for (size_t i = 0; i < lctx.kv_slots.size(); ++i) {
            data_attn_indices[i] = lctx.kv_slots[i];
        }
        for (size_t i = lctx.kv_slots.size(); i < lctx.inp_attn_indices->ne[0]; ++i) {
            data_attn_indices[i] = 0;
        }
    }
    if (lctx.inp_length_q) {
        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_length_q->buffer));
        int64_t * data_length_q = (int64_t *) lctx.inp_length_q->data;
        data_length_q[0]        = ubatch.n_tokens;
        // avoid possible flash attn bug for precision mode.
        // Maybe it's not needed.
        if (cparams.n_ubatch > 1 && ubatch.n_tokens == 1) {
            data_length_q[0] = 2;
        }
    }
    if (lctx.inp_length_kv) {
        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_length_kv->buffer));
        // Dummy Process of Page Attention
        if (cparams.page_attention) {
            int64_t * data_length_kv = (int64_t *) lctx.inp_length_kv->data;
            for (int i = 0; i < ubatch.n_tokens; ++i) {
                llama_seq_id seq_id       = ubatch.seq_id[i][0];
                int32_t      count_larger = 0;
                for (int j = 0; j < ubatch.n_tokens; ++j) {
                    llama_seq_id seq_id_j = ubatch.seq_id[j][0];
                    int32_t      pos      = ubatch.pos[j];
                    if (seq_id_j == seq_id && pos > ubatch.pos[i]) {
                        count_larger++;
                    }
                }
                data_length_kv[i] = lctx.kv_self.seq_lengths[seq_id] - count_larger;
            }
            for (int i = ubatch.n_tokens; i < lctx.inp_length_kv->ne[0]; ++i) {
                data_length_kv[i] = 0;
            }
        } else {
            int64_t * data_length_kv = (int64_t *) lctx.inp_length_kv->data;
            data_length_kv[0]        = kv_self.n;
        }
    }
    if (lctx.inp_page_table) {
        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_page_table->buffer));
        int32_t * data_page_table = (int32_t *) lctx.inp_page_table->data;
        for (int i = 0; i < ubatch.n_tokens; ++i) {
            llama_seq_id seq_id = ubatch.seq_id[i][0];
            for (int j = 0; j < lctx.kv_self.page_num_per_seq; ++j) {
                int32_t page_id = lctx.kv_self.page_table[seq_id * lctx.kv_self.page_num_per_seq + j];
                data_page_table[i * lctx.kv_self.page_num_per_seq + j] = page_id;
            }
        }
        for (int i = ubatch.n_tokens; i < lctx.inp_page_table->ne[1]; ++i) {
            for (int j = 0; j < lctx.inp_page_table->ne[0]; ++j) {
                data_page_table[i * lctx.inp_page_table->ne[0] + j] = 0;
            }
        }
    }

    // mask is not used for mla.
    if (cparams.enable_ge && !hparams.enable_mla) {
        llama_kv_cache &    kv_self      = lctx.kv_self;
        const int64_t       n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t       n_seqs       = ubatch.n_seqs;
        const int64_t       n_kv         = kv_self.n;
        const int64_t       n_tokens     = ubatch.n_tokens;
        const int64_t       n_ctx        = cparams.n_ctx;
        std::vector<int8_t> kqmask(n_tokens * n_kv);
        GGML_ASSERT(n_seqs * n_seq_tokens == n_tokens);
        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            for (int j = 0; j < n_seq_tokens; ++j) {
                const llama_pos pos = ubatch.pos[s * n_seq_tokens + j];
                for (int i = 0; i < n_kv; ++i) {
                    int8_t i8;
                    if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                        i8 = 1;
                    } else {
                        i8 = 0;
                    }
                    kqmask[s * (n_kv * n_seq_tokens) + j * n_kv + i] = i8;
                }
            }
        }
        for (int i = 0; i < (int) kv_self.kq_masks.size(); i++) {
            struct ggml_init_params params = {
                /*.mem_size   =*/size_t(5u * ggml_tensor_overhead()),
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/true,
            };
            ggml_context *       ctx = ggml_init(params);
            struct ggml_cgraph * gf  = ggml_new_graph_custom(ctx, 5, false);
            ggml_tensor *        src = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, n_kv, n_tokens);
            ggml_tensor *        dst = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, n_ctx, n_tokens);
            ggml_tensor * dst_view   = ggml_view_2d(ctx, dst, n_kv, n_tokens, ggml_row_size(GGML_TYPE_I8, n_ctx), 0);
            ggml_tensor * dst_cpy    = ggml_cpy(ctx, src, dst_view);
            ggml_build_forward_expand(gf, dst_cpy);
            // set intput
            // TODO: This part of the code is designed for CANN, and may have issues with other backends.
            src->data     = kv_self.kq_masks_tmp[i]->data;
            dst_cpy->data = kv_self.kq_masks[i]->data;
            ggml_backend_tensor_set(kv_self.kq_masks_tmp[i], kqmask.data(), 0, ggml_nbytes(src));

            ggml_backend_buffer_type_t buft    = ggml_backend_buffer_get_type(kv_self.kq_masks_tmp[i]->buffer);
            ggml_backend_dev_t         dev     = ggml_backend_buft_get_device(buft);
            ggml_backend_t             backend = nullptr;
            for (int j = 0; j < (int) lctx.backends.size(); j++) {
                if (ggml_backend_get_device(lctx.backends[j].get()) == dev) {
                    backend = lctx.backends[j].get();
                    break;
                }
            }
            if (!backend) {
                throw std::runtime_error("no backend found");
            }
            ggml_backend_graph_compute_async(backend, gf);
            ggml_backend_synchronize(backend);
        }
    }

    GGML_ASSERT(!(cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN));
    GGML_ASSERT(!(cparams.embeddings &&
                  (cparams.pooling_type == LLAMA_POOLING_TYPE_CLS || cparams.pooling_type == LLAMA_POOLING_TYPE_RANK)));
    GGML_ASSERT(!(cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_LAST));
    GGML_ASSERT(!kv_self.recurrent);
    GGML_ASSERT(!lctx.inp_pos_bucket);
    GGML_ASSERT(!(!lctx.is_encoding && lctx.inp_embd_enc));
    GGML_ASSERT(!(!lctx.is_encoding && lctx.inp_KQ_mask_cross));
}

// ============= Function =============

uint32_t llama_n_ctx(const struct llama_context * ctx) {
    return ctx->cparams.n_ctx;
}

uint32_t llama_n_batch(const struct llama_context * ctx) {
    return ctx->cparams.n_batch;
}

void llama_set_embeddings(struct llama_context * ctx, bool embeddings) {
    ctx->cparams.embeddings = embeddings;
}

int llama_all_processed_tokens(const struct llama_context * ctx) {
    return ctx->all_processed_token;
}

void llama_prepare_multiserver_data(struct llama_context & lctx, int n_tokens) {
#ifdef LLAMA_MPI_SUPPORT
    GGML_ASSERT(lctx.model.hparams.num_parallel <= MAX_PARALLEL_SERVERS);
    lctx.enable_dp_gather     = true;
    lctx.self_token_size      = n_tokens;
    lctx.self_token_offset    = 0;
    lctx.all_server_token_sum = 0;
    MPI_Allgather(&lctx.self_token_size, 1, MPI_INT, lctx.all_server_tokens, 1, MPI_INT, MPI_COMM_WORLD);
    for (size_t i = 0; i < lctx.model.hparams.num_parallel; ++i) {
        lctx.all_server_token_sum += lctx.all_server_tokens[i];
        if ((int) i < lctx.model.hparams.tp_id) {
            lctx.self_token_offset += lctx.all_server_tokens[i];
        }
    }
    lctx.all_processed_token += lctx.all_server_token_sum;
#else
    GGML_ABORT("LLAMA_MPI_SUPPORT not enabled");
#endif
}
