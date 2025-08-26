#include <cstring>
#include <mutex>
#include <thread>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama-context.h"
#include "llama-cparams.h"
#include "llama-graph-builder.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama.h"

#ifdef LLAMA_MPI_SUPPORT
#    define OMPI_SKIP_MPICXX 1
#    include <mpi.h>
#endif

static bool check_llama_init_context(llama_model * model, const llama_context_params & params);
static void build_cparams_by_params_and_hparams(struct llama_cparams &              cparams,
                                                const struct llama_context_params & params,
                                                const struct llama_hparams &        hparams);
static bool set_context_backends_from_model(struct llama_context * ctx, const llama_model * model);
static void print_kv_cache_init_info(const struct llama_context * ctx, ggml_type type_k, ggml_type type_v);
static bool initialize_sched_and_reserve(struct llama_context * ctx, const struct llama_model * model,
                                         const struct llama_context_params &       params,
                                         std::vector<ggml_backend_buffer_type_t> & backend_buft,
                                         std::vector<ggml_backend_t> & backend_ptrs, std::vector<uint8_t> & buf_meta,
                                         ggml_backend_sched_ptr & sched);

struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params) {
    // check params valid.
    if (!check_llama_init_context(model, params)) {
        return nullptr;
    }

    // new context. only init model pointer and model load time.
    llama_context * ctx = new llama_context(*model);

    const auto & hparams = model->hparams;
    auto &       cparams = ctx->cparams;

    // build cparams by hparams and params.
    build_cparams_by_params_and_hparams(cparams, params, hparams);

    ctx->logits_all = params.logits_all;

    // Simplified: deepseek does not have encoder.
    GGML_ASSERT(!llama_model_has_encoder(model));
    ctx->is_encoding = false;

    uint32_t  kv_size = cparams.n_ctx;
    ggml_type type_k  = params.type_k;
    ggml_type type_v  = params.type_v;

    // Simplified: deepseek does not have recurrent model.
    GGML_ASSERT(!llama_model_is_recurrent(model));

    GGML_ASSERT(hparams.n_embd_head_k % ggml_blck_size(type_k) == 0);
    GGML_ASSERT(hparams.n_embd_head_v % ggml_blck_size(type_v) == 0);

    if (!hparams.vocab_only) {
        // migrate backends from model.
        set_context_backends_from_model(ctx, model);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : ctx->backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn =
                    (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    ctx->set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        // set abort callback for context.
        llama_set_abort_callback(ctx, params.abort_callback, params.abort_callback_data);

        // initialize self-attention cache.
        if (!llama_kv_cache_init(ctx->kv_self, ctx->model, ctx->cparams, type_k, type_v, kv_size,
                                 cparams.offload_kqv)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }

        // print kv cache init info.
        print_kv_cache_init_info(ctx, type_k, type_v);

        // graph outputs buffer
        {
            // resized during inference when a batch uses more outputs
            if (llama_output_reserve(*ctx, params.n_seq_max) < params.n_seq_max) {
                LLAMA_LOG_ERROR("%s: failed to reserve initial output buffer\n", __func__);
                llama_free(ctx);
                return nullptr;
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                           ggml_backend_buffer_name(ctx->buf_output.get()),
                           ggml_backend_buffer_get_size(ctx->buf_output.get()) / 1024.0 / 1024.0);
        }

        // scheduler and compute buffers
        {
            // buffer types used for the compute buffer of each backend
            std::vector<ggml_backend_buffer_type_t> backend_buft;
            std::vector<ggml_backend_t>             backend_ptrs;
            for (auto & backend : ctx->backends) {
                auto * buft         = ggml_backend_get_default_buffer_type(backend.get());
                auto   backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model->devices.empty()) {
                    // use the host buffer of the first device CPU for faster transfer of the intermediate state
                    auto * dev       = model->devices[0];
                    auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
                    if (host_buft) {
                        buft = host_buft;
                    }
                }
                backend_buft.push_back(buft);
                backend_ptrs.push_back(backend.get());
            }

            // initialize scheduler and compute buffers.
            if (!initialize_sched_and_reserve(ctx, model, params, backend_buft, backend_ptrs, ctx->buf_compute_meta,
                                              ctx->sched)) {
                return nullptr;
            }

            if (cparams.enable_ge) {
                if (!hparams.enable_cann_flash_attention) {
                    LLAMA_LOG_ERROR("cann flash attention should be enabled when ge is enabled");
                    return nullptr;
                }
                if (!initialize_sched_and_reserve(ctx, model, params, backend_buft, backend_ptrs,
                                                  ctx->buf_compute_meta_decode, ctx->sched_decode)) {
                    return nullptr;
                }
                // persistent, directly apply space.
                llama_token token =
                    ctx->model.vocab
                        .token_bos();  // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
                uint32_t     n_tokens  = std::min(ctx->cparams.n_ctx, ctx->cparams.n_ubatch);
                uint32_t     n_seqs    = 1;  // TODO: worst-case number of sequences
                llama_ubatch ubatch_pp = { true,    n_tokens, n_tokens / n_seqs, n_seqs,  &token,
                                           nullptr, nullptr,  nullptr,           nullptr, nullptr };
                ggml_backend_sched_reset(ctx->sched_decode.get());
                ggml_backend_sched_set_eval_callback(ctx->sched_decode.get(), ctx->cparams.cb_eval,
                                                     ctx->cparams.cb_eval_user_data);
                const char * print_layer_env = std::getenv("LLAMA_PRINT_LAYER");
                if (print_layer_env) {
                    int         print_layer = std::atoi(print_layer_env);
                    std::string graph_name  = "llama_graph_deepseek2_ge_" + std::to_string(print_layer) + ".dot";
                    LLAMA_LOG_INFO("llama_graph_deepseek2_ge: print layer %d to %s\n", print_layer, graph_name.c_str());
                    auto * graph_for_print = llama_graph_builder::llama_build_graph(*ctx, ctx->buf_compute_meta_decode,
                                                                                    ubatch_pp, true, print_layer);
                    ggml_graph_print(graph_for_print);
                    ggml_graph_dump_dot(graph_for_print, nullptr, graph_name.c_str());
                }
                ctx->graph_decode =
                    llama_graph_builder::llama_build_graph(*ctx, ctx->buf_compute_meta_decode, ubatch_pp, true);
                ggml_graph_set_flags(ctx->graph_decode, 3);
                ggml_backend_sched_alloc_graph(ctx->sched_decode.get(), ctx->graph_decode);
                LLAMA_LOG_INFO("Runing Compute to warmup, please wait...\n");
                float time = ggml_time_us();
                llama_graph_builder::llama_graph_compute(*ctx, ctx->graph_decode, ctx->sched_decode.get(),
                                                         ctx->cparams.n_threads, nullptr);
                ggml_graph_set_flags(ctx->graph_decode, 1);
                ggml_backend_sched_set_flags(ctx->sched_decode.get(), 1);
                LLAMA_LOG_INFO("Warmup time: %f ms\n", (ggml_time_us() - time) / 1000.0);
            }

            // print scheduler buffer size.
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                ggml_backend_t             backend = backend_ptrs[i];
                ggml_backend_buffer_type_t buft    = backend_buft[i];
                size_t                     size    = ggml_backend_sched_get_buffer_size(ctx->sched.get(), backend);
                if (size > 1) {
                    LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__, ggml_backend_buft_name(buft),
                                   size / 1024.0 / 1024.0);
                }
            }
        }
    }

    return ctx;
}

bool check_llama_init_context(llama_model * model, const llama_context_params & params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return false;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return false;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return false;
    }

    if (params.flash_attn && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_ERROR("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        return false;
    }

    if (params.flash_attn && model->hparams.n_embd_head_k != model->hparams.n_embd_head_v) {
        LLAMA_LOG_ERROR("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        return false;
    }

    if (ggml_is_quantized(params.type_v) && !params.flash_attn) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return false;
    }

    if (params.enable_ge) {
        if (model->hparams.enable_mla) {
            LLAMA_LOG_ERROR("%s: enable_ge is not compatible with enable_mla - forcing off\n", __func__);
            return false;
        }
        if (!model->hparams.enable_cann_flash_attention) {
            LLAMA_LOG_ERROR("%s: cann flash attention is required when enable_ge is set\n", __func__);
            return false;
        }
    }

    return true;
}

void build_cparams_by_params_and_hparams(struct llama_cparams & cparams, const struct llama_context_params & params,
                                         const struct llama_hparams & hparams) {
    cparams.n_seq_max         = std::max(1u, params.n_seq_max);
    cparams.n_threads         = params.n_threads;
    cparams.n_threads_batch   = params.n_threads_batch;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.defrag_thold      = params.defrag_thold;
    cparams.embeddings        = params.embeddings;
    cparams.offload_kqv       = params.offload_kqv;
    cparams.flash_attn        = params.flash_attn || hparams.enable_cann_flash_attention;
    cparams.no_perf           = params.no_perf;
    cparams.pooling_type      = params.pooling_type;
    cparams.enable_ge         = params.enable_ge;
    cparams.enable_scatter_kv = params.enable_scatter_kv;
    cparams.presample_count   = params.presample_count;

    cparams.n_ctx           = params.n_ctx == 0 ? hparams.n_ctx_train : params.n_ctx;
    cparams.rope_freq_base  = params.rope_freq_base == 0.0f ? hparams.rope_freq_base_train : params.rope_freq_base;
    cparams.rope_freq_scale = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    // this is necessary due to kv_self.n being padded later during inference
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, llama_kv_cache_get_padding(cparams));

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = hparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    // the batch has to be at least GGML_KQ_MASK_PAD because we will be padding the KQ_mask
    // this is required by GPU kernels in order to avoid out-of-bounds accesses (e.g. ggml_flash_attn_ext)
    // ref: https://github.com/ggerganov/llama.cpp/pull/5021
    if (cparams.n_batch < GGML_KQ_MASK_PAD && !cparams.enable_ge) {
        LLAMA_LOG_WARN("%s: n_batch is less than GGML_KQ_MASK_PAD - increasing to %d\n", __func__, GGML_KQ_MASK_PAD);
        cparams.n_batch = GGML_KQ_MASK_PAD;
    }

    if (cparams.enable_ge) {
        if (!cparams.enable_scatter_kv) {
            LLAMA_LOG_WARN("%s: enable_scatter_kv is not set when enable_ge is set - forcing on\n", __func__);
            cparams.enable_scatter_kv = true;
        }
    }

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    if (params.yarn_orig_ctx != 0) {
        cparams.n_ctx_orig_yarn = params.yarn_orig_ctx;
    } else if (hparams.n_ctx_orig_yarn != 0) {
        cparams.n_ctx_orig_yarn = hparams.n_ctx_orig_yarn;
    } else {
        cparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    }

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f;  // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) {  // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    const uint32_t n_ctx_per_seq = cparams.n_ctx / cparams.n_seq_max;

    // LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    // LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    // LLAMA_LOG_INFO("%s: n_ctx_per_seq = %u\n",   __func__, n_ctx_per_seq);
    // LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    // LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    // LLAMA_LOG_INFO("%s: flash_attn    = %d\n",   __func__, cparams.flash_attn);
    // LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    // LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);

    if (n_ctx_per_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN(
            "%s: n_ctx_per_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
            __func__, n_ctx_per_seq, hparams.n_ctx_train);
    }

    if (n_ctx_per_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_pre_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n", __func__,
                       n_ctx_per_seq, hparams.n_ctx_train);
    }
}

bool set_context_backends_from_model(struct llama_context * ctx, const llama_model * model) {
    // GPU backends
    char * param_init = nullptr;
    size_t param_size = 0;
    if (model->params.enable_tensor_parallel) {
        ggml_backend_reg_t reg = nullptr;
        for (auto * dev : model->devices) {
            if (reg == nullptr) {
                reg = ggml_backend_dev_backend_reg(dev);
            } else {
                GGML_ASSERT(reg == ggml_backend_dev_backend_reg(dev));
            }
        }
        const int tp_id                 = model->hparams.tp_id;
        auto *    get_comm_init_size_fn = (ggml_backend_comm_init_overhead_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_comm_init_overhead");
        if (get_comm_init_size_fn == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to get ggml_backend_comm_init_overhead\n", __func__);
            return false;
        }
        param_size = get_comm_init_size_fn();
        param_init = new char[param_size];
        if (tp_id == 0) {
            auto * get_comm_init_fn =
                (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
            if (get_comm_init_fn == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to get ggml_backend_comm_init\n", __func__);
                return false;
            }
            get_comm_init_fn(param_init, model->hparams.num_parallel);
        }
        MPI_Bcast(param_init, param_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        auto * set_comm_rank_fn =
            (ggml_backend_comm_set_rank_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_set_rank");
        if (set_comm_rank_fn == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to get ggml_backend_comm_set_rank\n", __func__);
            return false;
        }
        LLAMA_LOG_INFO("%s: initializing tensor parallelism communication...\n", __func__);
        set_comm_rank_fn(param_init, tp_id);
        ggml_backend_dev_t dev     = model->devices[tp_id];
        ggml_backend_t     backend = ggml_backend_dev_init(dev, param_init);
        if (backend == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
            delete[] param_init;
            return false;
        }

        LLAMA_LOG_INFO("%s: initializing tensor parallelism communication success.\n", __func__);

        ctx->backends.emplace_back(backend);
        delete[] param_init;
    } else {
        for (auto * dev : model->devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
                llama_free(ctx);
                return false;
            }
            ctx->backends.emplace_back(backend);
        }
    }

    // add ACCEL backends (such as BLAS)
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
                llama_free(ctx);
                return false;
            }
            ctx->backends.emplace_back(backend);
        }
    }

    // add CPU backend
    ctx->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (ctx->backend_cpu == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to initialize CPU backend\n", __func__);
        llama_free(ctx);
        return false;
    }
    ctx->backends.emplace_back(ctx->backend_cpu);

    return true;
}

void print_kv_cache_init_info(const struct llama_context * ctx, ggml_type type_k, ggml_type type_v) {
    size_t memory_size_k = 0;
    size_t memory_size_v = 0;

    for (const auto & k : ctx->kv_self.k_l) {
        memory_size_k += ggml_nbytes(k);
    }

    for (const auto & v : ctx->kv_self.v_l) {
        memory_size_v += ggml_nbytes(v);
    }

    LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                   (float) (memory_size_k + memory_size_v) / (1024.0f * 1024.0f), ggml_type_name(type_k),
                   (float) memory_size_k / (1024.0f * 1024.0f), ggml_type_name(type_v),
                   (float) memory_size_v / (1024.0f * 1024.0f));
}

bool initialize_sched_and_reserve(struct llama_context * ctx, const struct llama_model * model,
                                  const struct llama_context_params &       params,
                                  std::vector<ggml_backend_buffer_type_t> & backend_buft,
                                  std::vector<ggml_backend_t> & backend_ptrs, std::vector<uint8_t> & buf_meta,
                                  ggml_backend_sched_ptr & sched) {
    llama_cparams & cparams   = ctx->cparams;
    const size_t    max_nodes = model->max_nodes();

    // buffer used to store the computation graph and the tensor meta data
    buf_meta.resize((ggml_tensor_overhead() * max_nodes) + ggml_graph_overhead_custom(max_nodes, false));

    // TODO: move these checks to ggml_backend_sched
    // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
    bool pipeline_parallel = model->n_devices() > 1 && model->params.n_gpu_layers > (int) model->hparams.n_layer &&
                             model->params.split_mode == LLAMA_SPLIT_MODE_LAYER && params.offload_kqv;

    // pipeline parallelism requires support for async compute and events in all devices
    if (pipeline_parallel) {
        for (auto & backend : ctx->backends) {
            auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                // ignore CPU backend
                continue;
            }
            auto *                 dev = ggml_backend_get_device(backend.get());
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);
            if (!props.caps.async || !props.caps.events) {
                // device does not support async compute or events
                pipeline_parallel = false;
                break;
            }
        }
    }

    if (pipeline_parallel) {
        GGML_ASSERT(!model->hparams.enable_tensor_parallel);
    }

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes,
                                       pipeline_parallel));

    if (model->hparams.enable_tensor_parallel) {
        ggml_backend_sched_init_threadpool(sched.get(), model->n_devices());
    }

    if (pipeline_parallel) {
        LLAMA_LOG_INFO("%s: pipeline parallelism enabled (n_copies=%d)\n", __func__,
                       ggml_backend_sched_get_n_copies(sched.get()));
    }

    uint32_t    n_seqs   = 1;  // TODO: worst-case number of sequences
    uint32_t    n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
    llama_token token =
        ctx->model.vocab
            .token_bos();  // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph

    llama_ubatch  ubatch_pp = { true,    n_tokens, n_tokens / n_seqs, n_seqs,  &token,
                                nullptr, nullptr,  nullptr,           nullptr, nullptr };
    ggml_cgraph * gf_pp     = llama_graph_builder::llama_build_graph(*ctx, buf_meta, ubatch_pp, true);

    // reserve pp graph first so that buffers are only allocated once
    ggml_backend_sched_reserve(sched.get(), gf_pp);
    int n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
    int n_nodes_pp  = ggml_graph_n_nodes(gf_pp);

    // reserve with tg graph to get the number of splits and nodes
    llama_ubatch  ubatch_tg = { true, 1, 1, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr };
    ggml_cgraph * gf_tg     = llama_graph_builder::llama_build_graph(*ctx, buf_meta, ubatch_tg, true);
    ggml_backend_sched_reserve(sched.get(), gf_tg);
    int n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
    int n_nodes_tg  = ggml_graph_n_nodes(gf_tg);

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    gf_pp = llama_graph_builder::llama_build_graph(*ctx, buf_meta, ubatch_pp, true);
    if (!ggml_backend_sched_reserve(sched.get(), gf_pp)) {
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        llama_free(ctx);
        return false;
    }
    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens,
                       n_nodes_tg);
    }
    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens,
                       n_splits_tg);
    }
    return true;
}
