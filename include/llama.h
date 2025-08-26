#ifndef LLAMA_MINI_H
#define LLAMA_MINI_H

#include <memory>

#include "ggml-backend.h"

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define MINI_API __declspec(dllexport)
#        else
#            define MINI_API __declspec(dllimport)
#        endif
#    else
#        define MINI_API __attribute__((visibility("default")))
#    endif
#else
#    define MINI_API
#endif

#define LLAMA_TOKEN_NULL   (-1)
#define LLAMA_DEFAULT_SEED 0xFFFFFFFF

struct llama_ubatch;
struct llama_sbatch;
struct llama_vocab;
struct llama_sampler;
struct llama_grammar;
struct llama_model;
struct llama_logits;
struct llama_context;

typedef int32_t llama_token;
typedef int32_t llama_pos;
typedef int32_t llama_token;
typedef int32_t llama_seq_id;
typedef void *  llama_sampler_context_t;

enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NONE = 0,  // For models without vocab
    LLAMA_VOCAB_TYPE_SPM  = 1,  // LLaMA tokenizer based on byte-level BPE with byte fallback
    LLAMA_VOCAB_TYPE_BPE  = 2,  // GPT-2 tokenizer based on byte-level BPE
    LLAMA_VOCAB_TYPE_WPM  = 3,  // BERT tokenizer based on WordPiece
    LLAMA_VOCAB_TYPE_UGM  = 4,  // T5 tokenizer based on Unigram
    LLAMA_VOCAB_TYPE_RWKV = 5,  // RWKV tokenizer based on greedy tokenization
};

enum llama_token_type {  //TODO: remove, required until per token attributes are available from GGUF file
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
    LLAMA_TOKEN_TYPE_NORMAL       = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
    LLAMA_TOKEN_TYPE_CONTROL      = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED       = 5,
    LLAMA_TOKEN_TYPE_BYTE         = 6,
};

// pre-tokenization types
enum llama_vocab_pre_type {
    LLAMA_VOCAB_PRE_TYPE_DEFAULT        = 0,
    LLAMA_VOCAB_PRE_TYPE_LLAMA3         = 1,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM   = 2,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,
    LLAMA_VOCAB_PRE_TYPE_FALCON         = 4,
    LLAMA_VOCAB_PRE_TYPE_MPT            = 5,
    LLAMA_VOCAB_PRE_TYPE_STARCODER      = 6,
    LLAMA_VOCAB_PRE_TYPE_GPT2           = 7,
    LLAMA_VOCAB_PRE_TYPE_REFACT         = 8,
    LLAMA_VOCAB_PRE_TYPE_COMMAND_R      = 9,
    LLAMA_VOCAB_PRE_TYPE_STABLELM2      = 10,
    LLAMA_VOCAB_PRE_TYPE_QWEN2          = 11,
    LLAMA_VOCAB_PRE_TYPE_OLMO           = 12,
    LLAMA_VOCAB_PRE_TYPE_DBRX           = 13,
    LLAMA_VOCAB_PRE_TYPE_SMAUG          = 14,
    LLAMA_VOCAB_PRE_TYPE_PORO           = 15,
    LLAMA_VOCAB_PRE_TYPE_CHATGLM3       = 16,
    LLAMA_VOCAB_PRE_TYPE_CHATGLM4       = 17,
    LLAMA_VOCAB_PRE_TYPE_VIKING         = 18,
    LLAMA_VOCAB_PRE_TYPE_JAIS           = 19,
    LLAMA_VOCAB_PRE_TYPE_TEKKEN         = 20,
    LLAMA_VOCAB_PRE_TYPE_SMOLLM         = 21,
    LLAMA_VOCAB_PRE_TYPE_CODESHELL      = 22,
    LLAMA_VOCAB_PRE_TYPE_BLOOM          = 23,
    LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH   = 24,
    LLAMA_VOCAB_PRE_TYPE_EXAONE         = 25,
    LLAMA_VOCAB_PRE_TYPE_CHAMELEON      = 26,
    LLAMA_VOCAB_PRE_TYPE_MINERVA        = 27,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM  = 28,
};

// model file types
enum llama_ftype {
    LLAMA_FTYPE_ALL_F32        = 0,
    LLAMA_FTYPE_MOSTLY_F16     = 1,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_0    = 2,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_1    = 3,  // except 1d tensors
    // LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
    // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
    // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
    LLAMA_FTYPE_MOSTLY_Q8_0    = 7,   // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_0    = 8,   // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_1    = 9,   // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q2_K    = 10,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_S  = 11,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_M  = 12,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_L  = 13,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_S  = 14,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_M  = 15,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_S  = 16,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_M  = 17,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q6_K    = 18,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_XS  = 20,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q2_K_S  = 21,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_XS  = 22,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ1_S   = 24,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ4_NL  = 25,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_S   = 26,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_M   = 27,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_S   = 28,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_M   = 29,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ4_XS  = 30,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ1_M   = 31,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_BF16    = 32,  // except 1d tensors
    //LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
    //LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
    //LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
    LLAMA_FTYPE_MOSTLY_TQ1_0   = 36,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_TQ2_0   = 37,  // except 1d tensors

    LLAMA_FTYPE_GUESSED = 1024,       // not specified in the model file
};

enum llama_token_attr {
    LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
    LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
    LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
    LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,
    LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
    LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,
    LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
    LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,
    LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
    LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
};

struct llama_chat_message {
    const char * role;
    const char * content;
};

// TODO: simplify (https://github.com/ggml-org/llama.cpp/pull/9294#pullrequestreview-2286561979)
typedef struct llama_token_data {
    llama_token id;     // token id
    float       logit;  // log-odds of the token
    float       p;      // probability of the token
} llama_token_data;

struct llama_token_data_array {
    // TODO: consider SoA
    // NOTE: this pointer can be modified by the samplers
    llama_token_data * data;
    size_t             size;
    int64_t            selected;  // this is the index in the data array (i.e. not the token id)
    bool               sorted;
};

// Input data for llama_decode
// A llama_batch object can contain input about one or many sequences
// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
//
// - token  : the token ids of the input (used when embd is NULL)
// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
// - pos    : the positions of the respective token in the sequence
//            (if set to NULL, the token position will be tracked automatically by llama_decode)
// - seq_id : the sequence to which the respective token belongs
//            (if set to NULL, the sequence ID will be assumed to be 0)
// - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
//            (if set to NULL, only the logits for last token will be returned)
//
typedef struct llama_batch {
    int32_t n_tokens;

    llama_token *   token;
    float *         embd;
    llama_pos *     pos;
    int32_t *       n_seq_id;
    llama_seq_id ** seq_id;
    int8_t *        logits;  // TODO: rename this to "output"
} llama_batch;

enum llama_split_mode {
    LLAMA_SPLIT_MODE_NONE  = 0,  // single GPU
    LLAMA_SPLIT_MODE_LAYER = 1,  // split layers and KV across GPUs
    LLAMA_SPLIT_MODE_ROW   = 2,  // split layers and KV across GPUs, use tensor parallelism if supported
};

enum llama_model_kv_override_type {
    LLAMA_KV_OVERRIDE_TYPE_INT,
    LLAMA_KV_OVERRIDE_TYPE_FLOAT,
    LLAMA_KV_OVERRIDE_TYPE_BOOL,
    LLAMA_KV_OVERRIDE_TYPE_STR,
};

enum llama_rope_type {
    LLAMA_ROPE_TYPE_NONE   = -1,
    LLAMA_ROPE_TYPE_NORM   = 0,
    LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
    LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
    LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
};

struct llama_model_kv_override {
    enum llama_model_kv_override_type tag;

    char key[128];

    union {
        int64_t val_i64;
        double  val_f64;
        bool    val_bool;
        char    val_str[128];
    };
};

enum llama_rope_scaling_type {
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
    LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
    LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
    LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
    LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3,
    LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE,
};

enum llama_pooling_type {
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
    LLAMA_POOLING_TYPE_NONE        = 0,
    LLAMA_POOLING_TYPE_MEAN        = 1,
    LLAMA_POOLING_TYPE_CLS         = 2,
    LLAMA_POOLING_TYPE_LAST        = 3,
    LLAMA_POOLING_TYPE_RANK        = 4,  // used by reranking models to attach the classification head to the graph
};

enum llama_attention_type {
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
    LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
    LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1,
};

typedef struct llama_logit_bias {
    llama_token token;
    float       bias;
} llama_logit_bias;

struct llama_sampler_chain_params {
    bool no_perf;
};

// user code can implement the interface below in order to create custom llama_sampler
struct llama_sampler_i {
    const char * (*name)(const struct llama_sampler * smpl);                     // can be NULL
    void (*accept)(struct llama_sampler * smpl, llama_token token);              // can be NULL
    void (*apply)(struct llama_sampler * smpl, llama_token_data_array * cur_p);  // required
    void (*reset)(struct llama_sampler * smpl);                                  // can be NULL
    struct llama_sampler * (*clone)(const struct llama_sampler * smpl);          // can be NULL if ctx is NULL
    void (*free)(struct llama_sampler * smpl);                                   // can be NULL if ctx is NULL

    // TODO: API for internal libllama usage for appending the sampling to an existing ggml_cgraph
    //void (*apply_ggml) (struct llama_sampler * smpl, ...);
};

MINI_API llama_sampler_chain_params llama_sampler_chain_default_params();

MINI_API struct llama_sampler * llama_sampler_init_grammar(const struct llama_vocab * vocab, const char * grammar_str,
                                                           const char * grammar_root);

/// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
MINI_API struct llama_sampler * llama_sampler_init_top_k(int32_t k);

/// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
MINI_API struct llama_sampler * llama_sampler_init_top_p(float p, size_t min_keep);

/// @details Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
MINI_API struct llama_sampler * llama_sampler_init_min_p(float p, size_t min_keep);

/// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
MINI_API struct llama_sampler * llama_sampler_init_xtc(float p, float t, size_t min_keep, uint32_t seed);

/// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
MINI_API struct llama_sampler * llama_sampler_init_typical(float p, size_t min_keep);

/// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
MINI_API struct llama_sampler * llama_sampler_init_temp_ext(float t, float delta, float exponent);

/// @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
MINI_API struct llama_sampler * llama_sampler_init_dry(const struct llama_vocab * vocab, int32_t n_ctx_train,
                                                       float dry_multiplier, float dry_base, int32_t dry_allowed_length,
                                                       int32_t dry_penalty_last_n, const char ** seq_breakers,
                                                       size_t num_breakers);

/// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
MINI_API struct llama_sampler * llama_sampler_init_penalties(
    int32_t penalty_last_n,  // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   penalty_repeat,  // 1.0 = disabled
    float   penalty_freq,    // 0.0 = disabled
    float   penalty_present);  // 0.0 = disabled

// llama_sampler_chain
// a type of llama_sampler that can chain multiple samplers one after another
MINI_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);

// available samplers:
MINI_API struct llama_sampler * llama_sampler_init_dist(uint32_t seed);

// important: takes ownership of the sampler object and will free it when llama_sampler_free is called
MINI_API void                   llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl);
MINI_API struct llama_sampler * llama_sampler_chain_get(const struct llama_sampler * chain, int32_t i);
MINI_API int                    llama_sampler_chain_n(const struct llama_sampler * chain);

MINI_API struct llama_sampler * llama_sampler_init_logit_bias(int32_t n_vocab, int32_t n_logit_bias,
                                                              const llama_logit_bias * logit_bias);

// important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
MINI_API void llama_sampler_free(struct llama_sampler * smpl);

MINI_API void llama_sampler_accept(struct llama_sampler * smpl, llama_token token);
MINI_API void llama_sampler_reset(struct llama_sampler * smpl);
MINI_API void llama_sampler_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p);

// mirror of llama_sampler_i:
MINI_API struct llama_sampler * llama_sampler_init(const struct llama_sampler_i * iface, llama_sampler_context_t ctx);

// Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
MINI_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);

// Special tokens
MINI_API llama_token llama_vocab_bos(const struct llama_vocab * vocab);  // beginning-of-sentence
MINI_API llama_token llama_vocab_eos(const struct llama_vocab * vocab);  // end-of-sentence
MINI_API llama_token llama_vocab_sep(const struct llama_vocab * vocab);  // sentence separator

MINI_API bool llama_vocab_get_add_eos(const struct llama_vocab * vocab);
MINI_API bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);

// Get the model's RoPE frequency scaling factor
MINI_API int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);

/// @details Convert the provided text into tokens.
/// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
/// @return Returns the number of tokens on success, no more than n_tokens_max
/// @return Returns a negative number on failure - the number of tokens that would have been returned
/// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
/// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
///                      as plaintext. Does not insert a leading space.
MINI_API int32_t llama_tokenize(const struct llama_vocab * vocab, const char * text, int32_t text_len,
                                llama_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special);

// Token Id -> Piece.
// Uses the vocabulary in the provided context.
// Does not write null terminator to the buffer.
// User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
// @param special If true, special tokens are rendered in the output.
MINI_API int32_t llama_token_to_piece(const struct llama_vocab * vocab, llama_token token, char * buf, int32_t length,
                                      int32_t lstrip, bool special);

/// @details Convert the provided tokens into text (inverse of llama_tokenize()).
/// @param text The char pointer must be large enough to hold the resulting text.
/// @return Returns the number of chars/bytes on success, no more than text_len_max.
/// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
/// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
/// @param unparse_special If true, special tokens are rendered in the output.
MINI_API int32_t llama_detokenize(const struct llama_vocab * vocab, const llama_token * tokens, int32_t n_tokens,
                                  char * text, int32_t text_len_max, bool remove_special, bool unparse_special);

// ====================== kv cache ======================

// TODO: remove llama_kv_cache_view_* API

// Information associated with an individual cell in the KV cache view.
struct llama_kv_cache_view_cell {
    // The position for this cell. Takes KV cache shifts into account.
    // May be negative if the cell is not populated.
    llama_pos pos;
};

// An updateable view of the KV cache.
struct llama_kv_cache_view {
    // Number of KV cache cells. This will be the same as the context size.
    int32_t n_cells;

    // Maximum number of sequences that can exist in a cell. It's not an error
    // if there are more sequences in a cell than this value, however they will
    // not be visible in the view cells_sequences.
    int32_t n_seq_max;

    // Number of tokens in the cache. For example, if there are two populated
    // cells, the first with 1 sequence id in it and the second with 2 sequence
    // ids then you'll have 3 tokens.
    int32_t token_count;

    // Number of populated cache cells.
    int32_t used_cells;

    // Maximum contiguous empty slots in the cache.
    int32_t max_contiguous;

    // Index to the start of the max_contiguous slot range. Can be negative
    // when cache is full.
    int32_t max_contiguous_idx;

    // Information for an individual cell.
    struct llama_kv_cache_view_cell * cells;

    // The sequences for each cell. There will be n_seq_max items per cell.
    llama_seq_id * cells_sequences;
};

// Create an empty KV cache view. (use only for debugging purposes)
MINI_API struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_context * ctx, int32_t n_seq_max);

// Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
// TODO: change signature to llama_kv_cache_view_update(struct llama_kv_cache_view * view, const struct llama_context * ctx)
MINI_API void llama_kv_cache_view_update(const struct llama_context * ctx, struct llama_kv_cache_view * view);

// Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
// Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
// seq_id < 0 : match any sequence
// p0 < 0     : [0,  p1]
// p1 < 0     : [p0, inf)
MINI_API bool llama_kv_cache_seq_rm(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1);

// Copy all tokens that belong to the specified sequence to another sequence
// Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
MINI_API void llama_kv_cache_seq_cp(struct llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst,
                                    llama_pos p0, llama_pos p1);

// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
// If the KV cache is RoPEd, the KV data is updated accordingly:
//   - lazily on next llama_decode()
//   - explicitly with llama_kv_cache_update()
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
MINI_API void llama_kv_cache_seq_add(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                                     llama_pos delta);

// ====================== 参数 ======================

typedef bool (*llama_progress_callback)(float progress, void * user_data);

struct llama_model_params {
    // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
    ggml_backend_dev_t * devices;

    int32_t               n_gpu_layers;  // number of layers to store in VRAM
    enum llama_split_mode split_mode;    // how to split the model across multiple GPUs

    // the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
    int32_t main_gpu;

    // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
    const float * tensor_split;

    // 启用张量并行，这里先不考虑其它设置
    bool enable_tensor_parallel;
    bool enable_expert_parallel;
    bool enable_data_parallel;
    bool enable_mpi;
    bool offload_input;
    bool enable_mla;
    bool enable_cann_flash_attention;
    int  tp_id;
    int  num_parallel;

    // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
    // If the provided progress_callback returns true, model loading continues.
    // If it returns false, model loading is immediately aborted.
    llama_progress_callback progress_callback;

    // context pointer passed to the progress callback
    void * progress_callback_user_data;

    // override key-value pairs of the model meta data
    const struct llama_model_kv_override * kv_overrides;

    // Keep the booleans together to avoid misalignment during copy-by-value.
    bool vocab_only;        // only load the vocabulary, no weights
    bool use_mmap;          // use mmap if possible
    bool use_mlock;         // force system to keep model in RAM
    bool check_tensors;     // validate model tensor data
    bool enable_fused_moe;  // enable fused moe
};

// NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
//       https://github.com/ggml-org/llama.cpp/pull/7544
struct llama_context_params {
    uint32_t n_ctx;            // text context, 0 = from model
    uint32_t n_batch;          // logical maximum batch size that can be submitted to llama_decode
    uint32_t n_ubatch;         // physical maximum batch size
    uint32_t n_seq_max;        // max number of sequences (i.e. distinct states for recurrent models)
    int32_t  n_threads;        // number of threads to use for generation
    int32_t  n_threads_batch;  // number of threads to use for batch processing

    enum llama_rope_scaling_type rope_scaling_type;  // RoPE scaling type, from `enum llama_rope_scaling_type`
    enum llama_pooling_type      pooling_type;       // whether to pool (sum) embedding results by sequence id
    enum llama_attention_type    attention_type;     // attention type to use for embeddings

    // ref: https://github.com/ggml-org/llama.cpp/pull/2054
    float    rope_freq_base;    // RoPE base frequency, 0 = from model
    float    rope_freq_scale;   // RoPE frequency scaling factor, 0 = from model
    float    yarn_ext_factor;   // YaRN extrapolation mix factor, negative = from model
    float    yarn_attn_factor;  // YaRN magnitude scaling factor
    float    yarn_beta_fast;    // YaRN low correction dim
    float    yarn_beta_slow;    // YaRN high correction dim
    uint32_t yarn_orig_ctx;     // YaRN original context size
    float    defrag_thold;      // defragment the KV cache if holes/size > thold, < 0 disabled (default)

    ggml_backend_sched_eval_callback cb_eval;
    void *                           cb_eval_user_data;

    enum ggml_type type_k;  // data type for K cache [EXPERIMENTAL]
    enum ggml_type type_v;  // data type for V cache [EXPERIMENTAL]

    // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
    // TODO: move at the end of the struct
    bool
        logits_all;  // the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
    bool embeddings;         // if true, extract embeddings (together with logits)
    bool offload_kqv;        // whether to offload the KQV ops (including the KV cache) to GPU
    bool flash_attn;         // whether to use flash attention [EXPERIMENTAL]
    bool no_perf;            // whether to measure performance timings
    bool enable_ge;          // whether to enable Graph Engine
    bool enable_scatter_kv;  // whether to enable scatter kv
    int  presample_count;    // number of tokens to presample on npu

    // Abort callback
    // if it returns true, execution of llama_decode() will be aborted
    // currently works only with CPU execution
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

// ====================== 加载模型（函数） ======================
struct llama_model_params   llama_model_default_params(void);
struct llama_context_params llama_context_default_params(void);

struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params params);

struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params);

MINI_API bool llama_supports_gpu_offload(void);
MINI_API bool llama_supports_rpc(void);

// ====================== 释放模型（函数） ======================
void llama_model_free(struct llama_model * model);

// ====================== 模型功能（函数） ======================
bool          llama_model_is_recurrent(const struct llama_model * model);
void          llama_free(struct llama_context * ctx);
MINI_API void llama_set_embeddings(struct llama_context * ctx, bool embeddings);
void          llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback,
                                       void * abort_callback_data);
void          llama_clear_adapter_lora(struct llama_context * ctx);
int32_t       llama_set_adapter_lora(struct llama_context * ctx, struct llama_adapter_lora * adapter, float scale);

const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
enum llama_rope_type       llama_model_rope_type(const struct llama_model * model);
// ====================== 并行相关简单函数 ======================
MINI_API uint32_t          llama_n_ctx(const struct llama_context * ctx);
MINI_API uint32_t          llama_n_batch(const struct llama_context * ctx);

bool        llama_model_has_encoder(const struct llama_model * model);
int32_t     llama_encode(struct llama_context * ctx, struct llama_batch batch);
llama_token llama_model_decoder_start_token(const struct llama_model * model);

// Returns true if the model contains a decoder that requires llama_decode() call
bool llama_model_has_decoder(const struct llama_model * model);

// Return batch for single sequence of tokens
// The sequence ID will be fixed to 0
// The position of the tokens will be tracked automatically by llama_decode
//
// NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
//
MINI_API struct llama_batch llama_batch_get_one(llama_token * tokens, int32_t n_tokens);

// Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
// Each token can be assigned up to n_seq_max sequence ids
// The batch has to be freed with llama_batch_free()
// If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
// Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
// The rest of the llama_batch members are allocated with size n_tokens
// All members are left uninitialized
MINI_API struct llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);

// Frees a batch of tokens allocated with llama_batch_init()
MINI_API void llama_batch_free(struct llama_batch batch);

// Positive return values does not mean a fatal error, but rather a warning.
//   0 - success
//   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
// < 0 - error. the KV cache state is restored to the state before this call
int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch, bool sync_all_servers = false);

bool llama_empty_run(struct llama_context * ctx);

int llama_all_processed_tokens(const struct llama_context * ctx);

// Clear the KV cache - both cell info is erased and KV data is zeroed
void llama_kv_cache_clear(struct llama_context * ctx);

// Check if the context supports KV cache shifting
bool llama_kv_cache_can_shift(struct llama_context * ctx);

// Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
void llama_kv_cache_update(struct llama_context * ctx);
// Wait until all computations are finished
// This is automatically done when using one of the functions below to obtain the computation results
// and is not necessary to call it explicitly in most cases
void llama_synchronize(struct llama_context * ctx);

// Get the default chat template. Returns nullptr if not available
// If name is NULL, returns the default chat template
const char * llama_model_chat_template(const struct llama_model * model, const char * name);

const struct llama_model *       llama_get_model(const struct llama_context * ctx);
MINI_API enum llama_pooling_type llama_pooling_type(const struct llama_context * ctx);
int32_t                          llama_model_n_ctx_train(const struct llama_model * model);

// Logits for the ith token. For positive indices, Equivalent to:
// llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
// Negative indicies can be used to access logits in reverse order, -1 is the last logit.
// returns NULL for invalid ids.
llama_logits llama_get_logits_ith(struct llama_context * ctx, int32_t i);

// Optional: an auto threadpool gets created in ggml if not passed explicitly
void llama_attach_threadpool(struct llama_context * ctx, ggml_threadpool_t threadpool,
                             ggml_threadpool_t threadpool_batch);

/// Apply chat template. Inspired by hf apply_chat_template() on python.
/// Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
/// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
/// @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
/// @param chat Pointer to a list of multiple llama_chat_message
/// @param n_msg Number of llama_chat_message in this chat
/// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
/// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
/// @param length The size of the allocated buffer
/// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
MINI_API int32_t llama_chat_apply_template(const char * tmpl, const struct llama_chat_message * chat, size_t n_msg,
                                           bool add_ass, char * buf, int32_t length);

// Initialize the llama + ggml backend
// If numa is true, use NUMA optimizations
// Call once at the start of the program
MINI_API void llama_backend_init(void);

// Call once at the end of the program - currently only used for MPI
MINI_API void llama_backend_free(void);

// 性能统计
struct llama_perf_context_data {
    double t_start_ms;
    double t_load_ms;
    double t_p_eval_ms;
    double t_eval_ms;

    int32_t n_p_eval;
    int32_t n_eval;
};

struct llama_perf_sampler_data {
    double t_sample_ms;

    int32_t n_sample;
};

// NOTE: the following work only with samplers constructed via llama_sampler_chain_init
MINI_API struct llama_perf_sampler_data llama_perf_sampler(const struct llama_sampler * chain);
MINI_API void                           llama_perf_sampler_print(const struct llama_sampler * chain);
MINI_API void                           llama_perf_sampler_reset(struct llama_sampler * chain);

MINI_API struct llama_perf_context_data llama_perf_context(const struct llama_context * ctx);
MINI_API void                           llama_perf_context_print(const struct llama_context * ctx);
MINI_API void                           llama_perf_context_reset(struct llama_context * ctx);

struct llama_model_deleter {
    void operator()(llama_model * model) { llama_model_free(model); }
};

struct llama_context_deleter {
    void operator()(llama_context * context) { llama_free(context); }
};

typedef std::unique_ptr<llama_model, llama_model_deleter>     llama_model_ptr;
typedef std::unique_ptr<llama_context, llama_context_deleter> llama_context_ptr;

#endif  // LLAMA_MINI_H
