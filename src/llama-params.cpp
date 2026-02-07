#include "llama.h"

struct llama_model_params llama_model_default_params() {
    struct llama_model_params result = {
        /*.devices                     =*/nullptr,
        /*.n_gpu_layers                =*/0,
        /*.split_mode                  =*/LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/0,
        /*.tensor_split                =*/nullptr,
        /*.enable_tensor_parallel      =*/false,
        /*.enable_expert_parallel      =*/false,
        /*.enable_data_parallel        =*/false,
        /*.enable_mpi                  =*/false,
        /*.offload_input               =*/false,
        /*.enable_mla                  =*/false,
        /*.enable_cann_flash_attention =*/false,
        /*.tp_id                       =*/-1,
        /*.num_parallel                =*/-1,
        /*.progress_callback           =*/nullptr,
        /*.progress_callback_user_data =*/nullptr,
        /*.kv_overrides                =*/nullptr,
        /*.vocab_only                  =*/false,
        /*.use_mmap                    =*/true,
        /*.use_mlock                   =*/false,
        /*.check_tensors               =*/false,
        /*.enable_fused_moe            =*/false,
    };

#ifdef GGML_USE_METAL
    // note: we usually have plenty of VRAM, so by default offload all layers to the GPU
    result.n_gpu_layers = 999;
#endif

    return result;
}

struct llama_context_params llama_context_default_params() {
    struct llama_context_params result = {
        /*.n_ctx                       =*/512,
        /*.n_batch                     =*/2048,
        /*.n_ubatch                    =*/512,
        /*.n_seq_max                   =*/1,
        /*.n_threads                   =*/GGML_DEFAULT_N_THREADS,  // TODO: better default
        /*.n_threads_batch             =*/GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.rope_freq_base              =*/0.0f,
        /*.rope_freq_scale             =*/0.0f,
        /*.yarn_ext_factor             =*/-1.0f,
        /*.yarn_attn_factor            =*/1.0f,
        /*.yarn_beta_fast              =*/32.0f,
        /*.yarn_beta_slow              =*/1.0f,
        /*.yarn_orig_ctx               =*/0,
        /*.defrag_thold                =*/-1.0f,
        /*.cb_eval                     =*/nullptr,
        /*.cb_eval_user_data           =*/nullptr,
        /*.type_k                      =*/GGML_TYPE_F16,
        /*.type_v                      =*/GGML_TYPE_F16,
        /*.logits_all                  =*/false,
        /*.embeddings                  =*/false,
        /*.offload_kqv                 =*/true,
        /*.flash_attn                  =*/false,
        /*.no_perf                     =*/true,
        /*.enable_ge                   =*/false,
        /*.enable_scatter_kv           =*/false,
        /*.page_attention              =*/false,
        /*.presample_count             =*/-1,
        /*.abort_callback              =*/nullptr,
        /*.abort_callback_data         =*/nullptr,
    };

    return result;
}
