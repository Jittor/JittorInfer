#include <cstdio>
#include <cstring>

#include "ggml-cann.h"
#include "ggml.h"
#include "llama-context.h"
#include "llama-graph-builder.h"
#include "llama-impl.h"
#include "llama.h"

#define OMPI_SKIP_MPICXX 1
#include "mpi.h"

template <int Id> __attribute__((noinline)) static int llama_debug_barrier() {
    static double tot_time  = 0.0;
    static int    tot_count = 0;

    double t_before = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    double t_after = MPI_Wtime();

    tot_time += t_after - t_before;
    tot_count += 1;
    if (tot_count == 32) {
        // printf every 1024 calls
        printf("Debug barrier<%d>: avg time = %.6f us over %d calls\n", Id, (tot_time / tot_count) * 1'000'000.0,
               tot_count);
        tot_time  = 0.0;
        tot_count = 0;
    }

    return Id;
}

static int llama_prepare_sbatch(llama_context & lctx, const llama_batch & batch, uint32_t & n_outputs) {
    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    const uint32_t n_tokens_all = batch.n_tokens;
    const int64_t  n_embd       = hparams.n_embd;

    // this indicates we are doing pooled embedding, so we ignore batch.logits and output all tokens
    const bool embd_pooled = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;

    // 输入验证和检查
    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd));  // NOLINT
    if (batch.token) {
        for (uint32_t i = 0; i < n_tokens_all; ++i) {
            if (batch.token[i] < 0 || uint32_t(batch.token[i]) >= model.vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return -1;
            }
        }
    }
    GGML_ASSERT(n_tokens_all <= cparams.n_batch);
    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) &&
                "non-causal attention requires n_ubatch >= n_tokens");

    lctx.n_queued_tokens += n_tokens_all;
    lctx.embd_seq.clear();

    // count outputs
    if (batch.logits && !embd_pooled) {
        for (uint32_t i = 0; i < n_tokens_all; ++i) {
            n_outputs += batch.logits[i] != 0;
        }
    } else if (lctx.logits_all || embd_pooled) {
        n_outputs = n_tokens_all;
    } else {
        n_outputs = 1;
    }

    lctx.sbatch.from_batch(batch, n_embd,
                           /* simple_split */ !lctx.kv_self.recurrent,
                           /* logits_all   */ n_outputs == n_tokens_all);

    if (llama_output_reserve(lctx, n_outputs) < n_outputs) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_outputs);
        return -2;
    };

    return 0;
}

static int llama_prepare_ubatch(llama_context & lctx, llama_kv_slot_restorer & kv_slot_restorer, llama_ubatch & ubatch,
                                const uint32_t n_outputs, const uint32_t n_tokens_all) {
    GGML_ASSERT(lctx.sbatch.n_tokens > 0);

    auto &       kv_self = lctx.kv_self;
    const auto & cparams = lctx.cparams;
    const auto & hparams = lctx.model.hparams;

    // this indicates we are doing pooled embedding, so we ignore batch.logits and output all tokens
    // const bool embd_pooled = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;

    // 从sbatch构造ubatch
    // Simplified: We do not support recurrent model
    GGML_ASSERT(!lctx.kv_self.recurrent);
    { ubatch = lctx.sbatch.split_simple(cparams.n_ubatch); }

    // count the outputs in this u_batch
    {
        int32_t n_outputs_new = 0;

        if (n_outputs == n_tokens_all) {
            n_outputs_new = ubatch.n_tokens;
        } else {
            GGML_ASSERT(ubatch.output);
            for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                n_outputs_new += int32_t(ubatch.output[i] != 0);
            }
        }

        // needs to happen before the graph is built
        lctx.n_outputs = n_outputs_new;
    }

    // non-causal masks do not use the KV cache
    if (hparams.causal_attn) {
        llama_kv_cache_update(&lctx);

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (kv_self.head > kv_self.used + 2 * ubatch.n_tokens) {
            kv_self.head = 0;
        }

        const auto slot = cparams.enable_scatter_kv ?
                              llama_kv_cache_find_scatter_slot(kv_self, ubatch, cparams.n_ubatch) :
                              llama_kv_cache_find_slot(kv_self, ubatch);
        if (!slot) {
            return 1;
        }
        lctx.kv_slots = slot.slot_ids;

        kv_slot_restorer.save(slot);

        if (!kv_self.recurrent) {
            // a heuristic, to avoid attending the full cache if it is not yet utilized
            // after enough generations, the benefit from this heuristic disappears
            // if we start defragmenting the cache, the benefit from this will be more important
            const uint32_t pad = llama_kv_cache_get_padding(cparams);
            if (cparams.enable_ge) {
                kv_self.n = std::min(kv_self.size, llama_kv_cache_cell_max(kv_self));
            } else {
                kv_self.n = std::min(kv_self.size, std::max(pad, GGML_PAD(llama_kv_cache_cell_max(kv_self), pad)));
            }
            //kv_self.n = llama_kv_cache_cell_max(kv_self);
        }
    }

    return 0;
}

static void llama_decode_presample_cann(ggml_backend_t backend, const ggml_tensor * logits_in, float * logits_out,
                                        int64_t * indices_out, int64_t k) {
    GGML_ASSERT(logits_in->type == GGML_TYPE_F32);
    int64_t n_batch = logits_in->ne[1];

    ggml_context * ctx = ggml_init({
        .mem_size   = ggml_tensor_overhead() * 2 + 4096,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    });

    ggml_tensor *         values  = ggml_new_tensor_2d(ctx, logits_in->type, k, n_batch);
    ggml_tensor *         indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, k, n_batch);
    ggml_backend_buffer_t buffer  = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_cann_presample(backend, logits_in, values, indices, k);
    ggml_backend_tensor_get_async(backend, values, logits_out, 0, k * n_batch * sizeof(float));
    ggml_backend_tensor_get_async(backend, indices, indices_out, 0, k * n_batch * sizeof(int64_t));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static int llama_decode_impl(llama_context & lctx, llama_batch inp_batch, bool sync_all_servers) {
    lctx.is_encoding = false;

    if (inp_batch.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // temporarily allocate memory for the input batch if needed
    llama_batch_allocr  batch_allocr(inp_batch, inp_batch.pos ? -1 : lctx.kv_self.max_pos() + 1);
    const llama_batch & batch = batch_allocr.batch;

    const auto & model   = lctx.model;
    const auto & vocab   = model.vocab;
    // const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    if (lctx.t_compute_start_us == 0) {
        lctx.t_compute_start_us = ggml_time_us();
    }
    auto &                 kv_self = lctx.kv_self;
    llama_kv_slot_restorer kv_slot_restorer(kv_self);

    // const int64_t n_embd  = hparams.n_embd;
    const int64_t n_vocab = vocab.n_tokens();

    uint32_t n_outputs      = 0;
    uint32_t n_outputs_prev = 0;

    {
        const int ret = llama_prepare_sbatch(lctx, batch, n_outputs);
        if (ret != 0) {
            return ret;
        }
    }

    while (lctx.sbatch.n_tokens > 0) {
        llama_ubatch ubatch;
        {
            const int ret = llama_prepare_ubatch(lctx, kv_slot_restorer, ubatch, n_outputs, batch.n_tokens);
            if (ret != 0) {
                return ret;
            }
        }

        const int         n_threads  = ubatch.n_tokens == 1 ? cparams.n_threads : cparams.n_threads_batch;
        ggml_threadpool_t threadpool = ubatch.n_tokens == 1 ? lctx.threadpool : lctx.threadpool_batch;

        GGML_ASSERT(n_threads > 0);

        ggml_cgraph *        gf    = nullptr;
        ggml_backend_sched_t sched = nullptr;

        if (!cparams.enable_ge) {
            ggml_backend_sched_reset(lctx.sched.get());
            ggml_backend_sched_set_eval_callback(lctx.sched.get(), lctx.cparams.cb_eval,
                                                 lctx.cparams.cb_eval_user_data);
            if (sync_all_servers) {
                llama_prepare_multiserver_data(lctx, ubatch.n_tokens);
            } else {
                lctx.enable_dp_gather = false;
            }
            gf = llama_graph_builder::llama_build_graph(lctx, lctx.buf_compute_meta, ubatch, false);

            ggml_backend_sched_alloc_graph(lctx.sched.get(), gf);
            sched = lctx.sched.get();
        } else {
            GGML_ASSERT(lctx.graph_decode);
            gf    = lctx.graph_decode;
            sched = lctx.sched_decode.get();
            llama_graph_builder::llama_update_graph(lctx, ubatch, false);
        }
        lctx.enable_dp_gather = false;

        // the output is always the last tensor in the graph
        struct ggml_tensor * res  = ggml_graph_node(gf, -1);
        struct ggml_tensor * embd = ggml_graph_node(gf, -2);

        if (lctx.n_outputs == 0) {
            // no output
            res  = nullptr;
            embd = nullptr;
        } else if (cparams.embeddings) {
            res  = nullptr;  // do not extract logits for embedding case
            embd = nullptr;
            for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
                if (strcmp(ggml_graph_node(gf, i)->name, "result_embd_pooled") == 0) {
                    embd = ggml_graph_node(gf, i);
                    break;
                }
            }
            GGML_ASSERT(embd != nullptr && "missing embeddings tensor");
        } else {
            embd = nullptr;  // do not extract embeddings when not needed
            GGML_ASSERT(strcmp(res->name, "result_output") == 0 && "missing result_output tensor");
        }

        llama_set_inputs(lctx, ubatch);

        const auto compute_status = llama_graph_builder::llama_graph_compute(lctx, gf, sched, n_threads, threadpool);

        // TODO: Exception handling
        GGML_ASSERT(compute_status == GGML_STATUS_SUCCESS);

        // update the kv ring buffer
        if (lctx.cparams.enable_scatter_kv) {
            kv_self.head = 0;
        } else {
            kv_self.head += ubatch.n_tokens;

            // Ensure kv cache head points to a valid index.
            if (kv_self.head >= kv_self.size) {
                kv_self.head = 0;
            }
        }

        // extract logits
        if (res) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched, res);
            GGML_ASSERT(backend_res != nullptr);

            const int32_t n_outputs_new = lctx.n_outputs;

            if (n_outputs_new) {
                auto logits = lctx.logits.next(n_outputs_prev);
                if (logits.type == llama_logits::LLAMA_LOGITS_TYPE_TOPK) {
                    llama_decode_presample_cann(backend_res, res, logits.values, logits.indices, logits.len);
                } else {
                    ggml_backend_tensor_get_async(backend_res, res, logits.values, 0,
                                                  n_outputs_new * logits.len * sizeof(float));
                }
                ggml_backend_synchronize(backend_res);
            }
        }

        // Simplifed: we do not support embeddings
        GGML_ASSERT(embd == nullptr);
        n_outputs_prev += lctx.n_outputs;
    }

    // set output mappings
    {
        bool sorted_output = true;

        GGML_ASSERT(lctx.sbatch.out_ids.size() == n_outputs);

        for (size_t i = 0; i < n_outputs; ++i) {
            size_t out_id           = lctx.sbatch.out_ids[i];
            lctx.output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        if (sorted_output) {
            lctx.sbatch.out_ids.clear();
        }
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    lctx.n_outputs = n_outputs;

    // wait for the computation to finish (automatically done when obtaining the model output)
    //llama_synchronize(&lctx);

    // decide if we need to defrag the kv cache
    if (cparams.causal_attn && cparams.defrag_thold > 0.0f) {
        // - do not defrag small contexts (i.e. < 2048 tokens)
        // - count the padding towards the number of used tokens
        const float fragmentation =
            kv_self.n >= 2048 ?
                std::max(0.0f, 1.0f - (float(kv_self.used + llama_kv_cache_get_padding(cparams)) / float(kv_self.n))) :
                0.0f;

        // queue defragmentation for next llama_kv_cache_update
        if (fragmentation > cparams.defrag_thold) {
            LLAMA_LOG_DEBUG("%s: fragmentation: %.2f - requesting defrag\n", __func__, fragmentation);

            llama_kv_cache_defrag(kv_self);
        }
    }

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(lctx.sched.get());

    return 0;
}

// decode a batch of tokens by evaluating the transformer
// in case of unsuccessful decoding (error or warning),
// the kv_cache state will be returned to its original state
// (for non-recurrent models) or cleaned (for recurrent models)
//
//   - lctx:      llama context
//   - inp_batch: batch to evaluate
//
// return 0 on success
// return positive int on warning
// return negative int on error
//
int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch, bool sync_all_servers) {
    const int ret = llama_decode_impl(*ctx, batch, sync_all_servers);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

static bool llama_empty_run_impl(struct llama_context & lctx) {
    ggml_backend_sched_reset(lctx.sched.get());
    ggml_backend_sched_set_eval_callback(lctx.sched.get(), lctx.cparams.cb_eval, lctx.cparams.cb_eval_user_data);

    llama_prepare_multiserver_data(lctx, 0);
    if (lctx.all_server_token_sum == 0) {
        return false;
    }
    const auto & cparams = lctx.cparams;
    llama_ubatch ubatch;
    ubatch.n_tokens  = 0;
    ggml_cgraph * gf = llama_graph_builder::llama_build_graph(lctx, lctx.buf_compute_meta, ubatch, false);

    ggml_backend_sched_alloc_graph(lctx.sched.get(), gf);
    // llama_set_inputs(lctx, ubatch);

    const int         n_threads  = cparams.n_threads;
    ggml_threadpool_t threadpool = lctx.threadpool;
    const auto        compute_status =
        llama_graph_builder::llama_graph_compute(lctx, gf, lctx.sched.get(), n_threads, threadpool);
    // TODO: Exception handling
    GGML_ASSERT(compute_status == GGML_STATUS_SUCCESS);
    return true;
}

bool llama_empty_run(struct llama_context * ctx) {
    return llama_empty_run_impl(*ctx);
}
