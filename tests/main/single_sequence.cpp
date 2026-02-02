// #include "common.h"
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "chat_local.h"
#include "common_local.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-context.h"
#include "llama.h"
#include "sampling_local.h"

static const char * DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant";
static bool         is_interacting         = false;

const int         num_inputs         = 3;
const std::string inputs[num_inputs] = {
    "I'm thinking a number between 1 and 20, please guess it.\n",
    "it's too small.\n",
    "it's too large.\n",
};

static struct local_cpu_params {
    int                      n_threads                   = 128;
    bool                     cpumask[GGML_MAX_N_THREADS] = { false };  // CPU affinity mask.
    bool                     mask_valid                  = false;      // Default: any CPU
    enum ggml_sched_priority priority =
        GGML_SCHED_PRIO_NORMAL;   // Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
    bool     strict_cpu = false;  // Use strict CPU placement
    uint32_t poll       = 50;     // Polling (busywait) level (0 - no polling, 100 - mostly polling)
} default_cpu_params, default_cpu_params_batch;

static struct DefaultMiniParams {
    int                   main_gpu          = 1;
    int                   n_gpu_layers      = 99;
    enum llama_split_mode split_mode        = LLAMA_SPLIT_MODE_LAYER;  // how to split the model across GPUs
    float                 tensor_split[128] = { 0 };  // how split tensors should be distributed across GPUs
    bool                  use_mmap          = false;   // use mmap for faster loads
    bool                  use_mlock         = false;  // use mlock to keep model in memory
    bool                  check_tensors     = false;  // validate tensor data

    std::string model = "/root/data/Qwen2.5-7B-Instruct-f16.gguf";  // Will be set from command line argument

    uint32_t n_ctx = 1024;                                            // context size

    // cpu
    uint32_t n_threads       = 128;                                // number of threads to use for computation
    uint32_t n_threads_batch = 128;                                // number of threads to use for batch processing

    float                                 defrag_thold = 0.05f;   // defragmentation threshold
    bool                                  no_perf      = false;  // disable performance metrics
    std::vector<common_adapter_lora_info> lora_adapters;         // lora adapter path with user defined scale

    // some runtime parameters
    int32_t                  n_batch = 1024;  // logical batch size for prompt processing (must be >=32 to use BLAS)
    bool                     enable_chat_template = true;
    bool                     escape               = true;
    common_conversation_mode conversation_mode    = COMMON_CONVERSATION_MODE_ENABLED;

} default_mini_params;

static llama_model_params common_model_params_to_llama_local() {
    auto mparams = llama_model_default_params();

    mparams.main_gpu                    = default_mini_params.main_gpu;
    mparams.split_mode                  = default_mini_params.split_mode;
    mparams.tensor_split                = default_mini_params.tensor_split;
    mparams.use_mmap                    = default_mini_params.use_mmap;
    mparams.use_mlock                   = default_mini_params.use_mlock;
    mparams.check_tensors               = default_mini_params.check_tensors;
    mparams.n_gpu_layers                = default_mini_params.n_gpu_layers;
    mparams.kv_overrides                = NULL;
    // 开启张量并行
    mparams.enable_tensor_parallel      = false;
    mparams.enable_fused_moe            = false;
    mparams.offload_input               = true;
    mparams.enable_cann_flash_attention = false;
    return mparams;
}

static llama_context_params common_context_params_to_llama_local() {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = default_mini_params.n_ctx;
    cparams.n_batch           = default_mini_params.n_batch;
    cparams.n_threads         = default_mini_params.n_threads;
    cparams.n_threads_batch   = default_mini_params.n_threads_batch;
    cparams.defrag_thold      = default_mini_params.defrag_thold;
    cparams.no_perf           = default_mini_params.no_perf;
    cparams.enable_ge         = false;
    cparams.enable_scatter_kv = false;
    cparams.presample_count   = 64;
    return cparams;
}

static common_init_result common_init_from_params_local() {
    common_init_result iparams;
    auto               mparams = common_model_params_to_llama_local();

    llama_model * model = nullptr;

    model = llama_model_load_from_file(default_mini_params.model.c_str(), mparams);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto cparams = common_context_params_to_llama_local();

    llama_context * lctx = llama_init_from_model(model, cparams);
    if (lctx == NULL) {
        std::cerr << __func__ << ": failed to create context with model '" << default_mini_params.model << "'\n";
        llama_model_free(model);
        return iparams;
    }

    // Simplified: lora is not implemented now.
    GGML_ASSERT(default_mini_params.lora_adapters.empty());
    // common_set_adapter_lora_local(lctx, default_mini_params.lora_adapters);
    llama_clear_adapter_lora(lctx);

    std::cout << __func__ << ": warming up the model with an empty run - please wait ... (--no-warmup to disable)\n";

    std::vector<llama_token> tmp;
    llama_token              bos = llama_vocab_bos(vocab);
    llama_token              eos = llama_vocab_eos(vocab);

    // some models (e.g. T5) don't have a BOS token
    if (bos != LLAMA_TOKEN_NULL) {
        tmp.push_back(bos);
    }
    if (eos != LLAMA_TOKEN_NULL) {
        tmp.push_back(eos);
    }
    if (tmp.empty()) {
        tmp.push_back(0);
    }

    // Simplified: encoder is not implemented now (for deepseek).
    GGML_ASSERT(!llama_model_has_encoder(model));

    if (llama_model_has_decoder(model)) {
        llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) default_mini_params.n_batch)));
    }
    llama_kv_cache_clear(lctx);
    llama_synchronize(lctx);
    llama_perf_context_reset(lctx);

    iparams.model.reset(model);
    iparams.context.reset(lctx);

    return iparams;
}

static ggml_threadpool_params ggml_threadpool_params_from_cpu_params_local(const local_cpu_params & cpuparams) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, cpuparams.n_threads);  // setup the defaults

    if (cpuparams.mask_valid) {
        std::memcpy(&tpp.cpumask, &cpuparams.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = cpuparams.priority;
    tpp.poll       = cpuparams.poll;
    tpp.strict_cpu = cpuparams.strict_cpu;

    return tpp;
}

int main() {
    common_params_local_sampling sparams;

    llama_backend_init();

    llama_model *          model = nullptr;
    llama_context *        ctx   = nullptr;
    common_sampler_local * smpl  = nullptr;

    std::vector<common_chat_local_msg> chat_msgs;

    // load the model and apply lora adapter, if any
    std::cout << "Loading model and applying lora adapter...\n";
    common_init_result llama_init = common_init_from_params_local();

    model = llama_init.model.get();
    ctx   = llama_init.context.get();

    const llama_vocab * vocab          = llama_model_get_vocab(model);
    auto                chat_templates = common_chat_local_templates_init(model);

    std::cout << "Initializing threadpool with " << default_mini_params.n_threads << " threads\n";

    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    auto * ggml_threadpool_new_fn =
        (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn =
        (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch = ggml_threadpool_params_from_cpu_params_local(default_cpu_params_batch);
    struct ggml_threadpool_params tpp       = ggml_threadpool_params_from_cpu_params_local(default_cpu_params);

    set_process_priority(default_cpu_params.priority);

    struct ggml_threadpool * threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            std::cerr << "Batch threadpool creation failed: n_threads " << tpp_batch.n_threads << "\n";
            return 1;
        }

        // Start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        std::cerr << "Threadpool creation failed: n_threads " << tpp.n_threads << "\n";
        return 1;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);

    std::vector<llama_token> session_tokens;

    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

    std::vector<llama_token> embd_inp;

    auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
        common_chat_local_msg new_msg;
        new_msg.role    = role;
        new_msg.content = content;
        auto formatted =
            common_chat_local_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", false);
        chat_msgs.push_back(new_msg);
        return formatted;
    };

    {
        auto prompt = chat_add_and_format("system", DEFAULT_SYSTEM_MESSAGE);
        embd_inp    = common_tokenize(ctx, prompt, true, true);
    }

    smpl = common_sampler_local_init(model, sparams);

    is_interacting = true;

    int n_past     = 0;
    int n_consumed = 0;
    int input_idx  = 0;

    std::ostringstream assistant_ss;  // for storing current assistant message, used in conversation mode

    std::vector<llama_token> embd;

    // single-token antiprompts
    std::vector<llama_token> antiprompt_token;

    while (true) {
        if (!embd.empty()) {
            for (int i = 0; i < (int) embd.size(); i += default_mini_params.n_batch) {
                int n_eval = (int) embd.size() - i;
                n_eval     = std::min(n_eval, default_mini_params.n_batch);

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                    std::cerr << "Failed to eval\n";
                    return 1;
                }

                n_past += n_eval;
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, /* accept_grammar= */ true);

            embd.push_back(id);

        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;
                if ((int) embd.size() >= default_mini_params.n_batch) {
                    break;
                }
            }
        }

        std::cout << assistant_ss.str() << '\n';
        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // deal with end of generation tokens in interactive mode
            if (llama_vocab_is_eog(vocab, common_sampler_local_last(smpl))) {
                chat_add_and_format("assistant", assistant_ss.str());
                is_interacting = true;
                std::cout << assistant_ss.str() << '\n';
            }

            // if current token is not EOG, we add it to current assistant message
            if (default_mini_params.conversation_mode) {
                const auto id = common_sampler_local_last(smpl);
                assistant_ss << common_token_to_piece(ctx, id, false);
            }
            if (n_past > 0 && is_interacting) {
                if (input_idx >= num_inputs) {
                    break;
                }
                std::string buffer;
                std::string line = inputs[input_idx];
                input_idx += 1;
                buffer += line;
                std::cout << "> " << buffer;

                if (default_mini_params.escape) {
                    string_process_escapes(buffer);
                }

                std::string user_inp = chat_add_and_format("user", buffer);
                const auto  line_inp = common_tokenize(ctx, user_inp, false, true);

                embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                // reset assistant message
                assistant_ss.str("");
            }

            if (n_past > 0) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;
            }
        }
    }

    if (smpl) {
        common_perf_print(ctx, smpl);
    }

    common_sampler_local_free(smpl);

    llama_backend_free();

    ggml_threadpool_free_fn(threadpool);
    ggml_threadpool_free_fn(threadpool_batch);

    return 0;
}
