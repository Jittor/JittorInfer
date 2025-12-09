#include <map>
#include <set>
#include <vector>

#include "ggml-cpp.h"
#include "ggml.h"
#include "llama-adapter.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama.h"

const int MAX_PARALLEL_SERVERS = 8;

// logits for a entire batch
struct llama_logits {
    enum llama_logits_type { LLAMA_LOGITS_TYPE_RAW, LLAMA_LOGITS_TYPE_TOPK } type = LLAMA_LOGITS_TYPE_RAW;

    float *   values  = nullptr;  // logits value
    int64_t * indices = nullptr;  // logits indices
    int64_t   len     = -1;       // length(stride) of the logits, -1 means not initialized
    int64_t   num     = -1;       // number of logits in this batch, -1 means not initialized

    static size_t get_size(llama_logits_type type, int64_t len, int64_t num) {
        if (type == LLAMA_LOGITS_TYPE_RAW) {
            return len * num * sizeof(float);
        } else if (type == LLAMA_LOGITS_TYPE_TOPK) {
            return len * num * (sizeof(float) + sizeof(int64_t));
        } else {
            GGML_ABORT("unknown logits type");
            return 0;
        }
    }

    void init(void * base_addr, llama_logits_type type, int64_t len, int64_t num) {
        GGML_ASSERT(base_addr != nullptr);
        this->type = type;
        this->len  = len;
        this->num  = num;

        values = (float *) base_addr;
        if (type == LLAMA_LOGITS_TYPE_TOPK) {
            indices = (int64_t *) (values + len * num);
        } else {
            indices = nullptr;
        }
    }

    /**
     * Get the logits of k-th batch.
     */
    llama_logits next(int64_t k) {
        GGML_ASSERT(0 <= k && k < num);
        return { .type    = type,
                 .values  = values + k * len,
                 .indices = type == LLAMA_LOGITS_TYPE_RAW ? nullptr : indices + k * len,
                 .len     = len,
                 .num     = num - k };
    }
};

struct llama_context {
    llama_context(const llama_model & model) :
        model(model),
        t_start_us(model.t_start_us),
        t_load_us(model.t_load_us),
        enable_dp_gather(false) {}

    const struct llama_model & model;

    struct llama_cparams      cparams;
    struct llama_sbatch       sbatch;  // TODO: revisit if needed
    struct llama_kv_cache     kv_self;
    struct llama_adapter_cvec cvec;

    std::unordered_map<struct llama_adapter_lora *, float> lora;

    std::vector<ggml_backend_ptr>                                        backends;
    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    ggml_backend_t backend_cpu = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    bool has_evaluated_once = false;

    mutable int64_t t_start_us;
    mutable int64_t t_load_us;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0;  // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0;  // number of eval calls

    // kv cache slots for this ubatch
    std::vector<int32_t> kv_slots;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    llama_logits logits;                   // logits for all batch

    std::vector<int32_t> output_ids;       // map batch token positions to ids of the logits and embd buffers
    size_t               output_size = 0;  // capacity (of tokens positions) for the output buffers
    int32_t              n_outputs = 0;  // number of actually-used outputs in the current ubatch or last logical batch

    bool logits_all = false;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0;  // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // TODO: find a better way to accommodate mutli-dimension position encoding methods
    // number of position id each token get, 1 for each token in most cases.
    // when using m-rope, it will be 3 position ids per token to representing 3 dimension coordinate.
    int n_pos_per_token = 1;

    // output of the encoder part of the encoder-decoder models
    std::vector<float>                  embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;

    // memory buffers used to evaluate the model
    std::vector<uint8_t>   buf_compute_meta;
    std::vector<uint8_t>   buf_compute_meta_decode;
    ggml_backend_sched_ptr sched;
    ggml_backend_sched_ptr sched_decode;
    ggml_cgraph *          graph_decode = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    // input tensors
    struct ggml_tensor * inp_tokens;         // I32 [n_batch]
    struct ggml_tensor * inp_embd;           // F32 [n_embd, n_batch]
    struct ggml_tensor * inp_pos;            // I32 [n_batch]
    struct ggml_tensor * inp_out_ids;        // I32 [n_outputs]
    struct ggml_tensor * inp_KQ_mask;        // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_swa;    // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_i8;     // I8  [kv_size, n_batch]
    struct ggml_tensor * inp_K_shift;        // I32 [kv_size]
    struct ggml_tensor * inp_mean;           // F32 [n_batch, n_batch]
    struct ggml_tensor * inp_cls;            // I32 [n_batch]
    struct ggml_tensor * inp_s_copy;         // I32 [kv_size]
    struct ggml_tensor * inp_s_mask;         // F32 [1, n_kv]
    struct ggml_tensor * inp_s_seq;          // I32 [n_kv, n_batch]
    struct ggml_tensor * inp_pos_bucket;     // I32 [n_batch|n_kv, n_batch]
    struct ggml_tensor * inp_embd_enc;       // F32 [n_embd, n_outputs_enc]
    struct ggml_tensor * inp_KQ_mask_cross;  // F32 [n_outputs_enc, n_batch]
    struct ggml_tensor * inp_attn_indices;   // I32 [n_tokens]
    struct ggml_tensor * inp_length_q;       // I64 [1]
    struct ggml_tensor * inp_length_kv;      // I64 [1]

    // multi_server_status
    int  all_server_tokens[MAX_PARALLEL_SERVERS];
    int  all_server_token_sum;
    int  self_token_size;
    int  self_token_offset;
    // 是否启用数据并行，attention之后需要进行gather
    bool enable_dp_gather;
};

// TODO: make these methods of llama_context
void llama_set_k_shift(struct llama_context & lctx);

size_t llama_output_reserve(struct llama_context & lctx, size_t n_outputs);

void llama_set_inputs(llama_context & lctx, const llama_ubatch & ubatch);

// 处理multi-server数据，用于数据并行+MoE
void llama_prepare_multiserver_data(struct llama_context & lctx, int n_tokens);
