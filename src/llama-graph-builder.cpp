#include "llama-graph-builder.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-context.h"
#include "llama-graph-deepseek2.h"
#include "llama-graph-deepseek2ge.h"
#include "llama-graph-qwen2.h"
#include "llama-graph-defrag.h"
#include "llama-graph-utils.h"
#include "llama-impl.h"

struct ggml_cgraph * llama_graph_builder::llama_build_graph(llama_context &        lctx,
                                                            std::vector<uint8_t> & buf_compute_meta,
                                                            const llama_ubatch & ubatch, bool worst_case, int print_layer) {
    const auto & model = lctx.model;

    // this callback allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
    llm_build_cb cb = [&](struct ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!lctx.cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(lctx.sched.get(), cur, lctx.backend_cpu);
            }
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = lctx.model.params.n_gpu_layers > (int) lctx.model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = lctx.model.dev_layer(il);
                for (auto & backend : lctx.backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(lctx.sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };

    llm_build_cb cb_tp = [&](struct ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!lctx.cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(lctx.sched.get(), cur, lctx.backend_cpu);
            }
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        if (il != -1) {
            const auto & dev_layer = lctx.model.dev_layer(il);
            bool         found     = false;
            for (auto & backend : lctx.backends) {
                if (ggml_backend_get_device(backend.get()) == dev_layer) {
                    GGML_ASSERT(ggml_backend_supports_op(backend.get(), cur));
                    ggml_backend_sched_set_tensor_backend(lctx.sched.get(), cur, backend.get());
                    found = true;
                    break;
                }
            }
            GGML_ASSERT(found);
        }
    };

    if (lctx.model.params.enable_tensor_parallel) {
        cb = cb_tp;
    }

    if (lctx.cparams.enable_ge) {
        switch (model.arch) {
            case LLM_ARCH_DEEPSEEK2:
                return llm_build_deepseek2_ge(lctx, buf_compute_meta, ubatch, cb, worst_case, print_layer);
            default:
                GGML_ABORT("Unsupported model architecture");
        }
    }

    switch (model.arch) {
        case LLM_ARCH_DEEPSEEK2:
            return llm_build_deepseek2(lctx, buf_compute_meta, ubatch, cb, worst_case, print_layer);
        
        case LLM_ARCH_QWEN2:
            return llm_build_qwen2(lctx, buf_compute_meta, ubatch, cb, worst_case, print_layer);
        default:
            GGML_ABORT("Unsupported model architecture");
    }
}

enum ggml_status llama_graph_builder::llama_graph_compute(llama_context & lctx, ggml_cgraph * gf,
                                                          ggml_backend_sched_t sched, int n_threads,
                                                          ggml_threadpool * threadpool) {
    if (lctx.backend_cpu != nullptr) {
        auto * reg               = ggml_backend_dev_backend_reg(ggml_backend_get_device(lctx.backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_cpu_set_threadpool");
        set_threadpool_fn(lctx.backend_cpu, threadpool);
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : lctx.set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    enum ggml_status status;
    status = ggml_backend_sched_graph_compute_async(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    return status;
}

struct ggml_cgraph * llama_graph_builder::llama_build_graph_defrag(llama_context &               lctx,
                                                                   const std::vector<uint32_t> & ids) {
    return llm_build_defrag(lctx, ids);
}

void llama_graph_builder::llama_update_graph(llama_context & lctx, const llama_ubatch & /*ubatch*/,
                                             bool /*worst_case*/) {
    GGML_ASSERT(lctx.cparams.enable_ge);
    const auto & model = lctx.model;
    {
        switch (model.arch) {
            case LLM_ARCH_DEEPSEEK2:
                llm_update_deepseek2_ge(lctx);
                break;
            default:
                GGML_ABORT("Unsupported model architecture");
        }
    }
}
