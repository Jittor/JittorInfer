// Various helper functions and utilities

#pragma once

#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"

#ifdef _WIN32
#    define DIRECTORY_SEPARATOR '\\'
#else
#    define DIRECTORY_SEPARATOR '/'
#endif  // _WIN32

#define die(msg)                           \
    do {                                   \
        fputs("error: " msg "\n", stderr); \
        exit(1);                           \
    } while (0)
#define die_fmt(fmt, ...)                                 \
    do {                                                  \
        fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); \
        exit(1);                                          \
    } while (0)

#define DEFAULT_MODEL_PATH "models/7B/ggml-model-f16.gguf"

using llama_tokens = std::vector<llama_token>;

struct common_control_vector_load_info;

//
// Common params
//

enum llama_example {
    LLAMA_EXAMPLE_COMMON,
    LLAMA_EXAMPLE_SPECULATIVE,
    LLAMA_EXAMPLE_MAIN,
    LLAMA_EXAMPLE_INFILL,
    LLAMA_EXAMPLE_EMBEDDING,
    LLAMA_EXAMPLE_PERPLEXITY,
    LLAMA_EXAMPLE_RETRIEVAL,
    LLAMA_EXAMPLE_PASSKEY,
    LLAMA_EXAMPLE_IMATRIX,
    LLAMA_EXAMPLE_BENCH,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CVECTOR_GENERATOR,
    LLAMA_EXAMPLE_EXPORT_LORA,
    LLAMA_EXAMPLE_LLAVA,
    LLAMA_EXAMPLE_LOOKUP,
    LLAMA_EXAMPLE_PARALLEL,
    LLAMA_EXAMPLE_TTS,

    LLAMA_EXAMPLE_COUNT,
};

// dimensionality reduction methods, used by cvector-generator
enum dimre_method {
    DIMRE_METHOD_PCA,
    DIMRE_METHOD_MEAN,
};

enum common_conversation_mode {
    COMMON_CONVERSATION_MODE_DISABLED = 0,
    COMMON_CONVERSATION_MODE_ENABLED  = 1,
    COMMON_CONVERSATION_MODE_AUTO     = 2,
};

struct common_grammar_trigger {
    std::string word;
    bool        at_start;
};

enum common_params_local_type {
    COMMON_SAMPLER_TYPE_LOCAL_NONE        = 0,
    COMMON_SAMPLER_TYPE_LOCAL_DRY         = 1,
    COMMON_SAMPLER_TYPE_LOCAL_TOP_K       = 2,
    COMMON_SAMPLER_TYPE_LOCAL_TOP_P       = 3,
    COMMON_SAMPLER_TYPE_LOCAL_MIN_P       = 4,
    //COMMON_SAMPLER_TYPE_LOCAL_TFS_Z       = 5,
    COMMON_SAMPLER_TYPE_LOCAL_TYPICAL_P   = 6,
    COMMON_SAMPLER_TYPE_LOCAL_TEMPERATURE = 7,
    COMMON_SAMPLER_TYPE_LOCAL_XTC         = 8,
    COMMON_SAMPLER_TYPE_LOCAL_INFILL      = 9,
    COMMON_SAMPLER_TYPE_LOCAL_PENALTIES   = 10,
};

// sampling parameters
struct common_params_sampling {
    uint32_t seed = LLAMA_DEFAULT_SEED;  // the seed used to initialize llama_sampler

    int32_t n_prev            = 64;      // number of previous tokens to remember
    int32_t n_probs           = 0;       // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t min_keep          = 0;       // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t top_k             = 40;      // <= 0 to use vocab size
    float   top_p             = 0.95f;   // 1.0 = disabled
    float   min_p             = 0.05f;   // 0.0 = disabled
    float   xtc_probability   = 0.00f;   // 0.0 = disabled
    float   xtc_threshold     = 0.10f;   // > 0.5 disables XTC
    float   typ_p             = 1.00f;   // typical_p, 1.0 = disabled
    float   temp              = 0.80f;   // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float   dynatemp_range    = 0.00f;   // 0.0 = disabled
    float   dynatemp_exponent = 1.00f;   // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t penalty_last_n    = 64;      // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   penalty_repeat    = 1.00f;   // 1.0 = disabled
    float   penalty_freq      = 0.00f;   // 0.0 = disabled
    float   penalty_present   = 0.00f;   // 0.0 = disabled
    float   dry_multiplier    = 0.0f;    // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
    float   dry_base =
        1.75f;  // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
    int32_t dry_allowed_length = 2;  // tokens extending repetitions beyond this receive penalty
    int32_t dry_penalty_last_n =
        -1;                          // how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
    int32_t                             mirostat         = 0;       // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float                               top_n_sigma      = -1.00f;  // -1.0 = disabled
    float                               mirostat_tau     = 5.00f;   // target entropy
    float                               mirostat_eta     = 0.10f;   // learning rate
    bool                                ignore_eos       = false;
    bool                                no_perf          = false;   // disable performance metrics
    bool                                timing_per_token = false;
    std::string                         grammar;                    // optional BNF-like grammar to constrain sampling
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_trigger_words;      // optional trigger words to trigger lazy grammar
    std::vector<llama_token>
        grammar_trigger_tokens;  // optional trigger tokens to trigger lazy grammar and print trigger special tokens.
    std::set<llama_token> preserved_tokens;

    std::vector<std::string> dry_sequence_breakers = { "\n", ":", "\"", "*" };  // default sequence breakers for DRY

    std::vector<llama_logit_bias> logit_bias;                                   // logit biases to apply

    std::vector<enum common_params_local_type> samplers = {
        COMMON_SAMPLER_TYPE_LOCAL_PENALTIES, COMMON_SAMPLER_TYPE_LOCAL_DRY,         COMMON_SAMPLER_TYPE_LOCAL_TOP_K,
        COMMON_SAMPLER_TYPE_LOCAL_TYPICAL_P, COMMON_SAMPLER_TYPE_LOCAL_TOP_P,       COMMON_SAMPLER_TYPE_LOCAL_MIN_P,
        COMMON_SAMPLER_TYPE_LOCAL_XTC,       COMMON_SAMPLER_TYPE_LOCAL_TEMPERATURE,
    };

    // print the parameters into a string
    std::string print() const;
};

struct common_params_vocoder {
    std::string model = "";         // model path                                                // NOLINT

    bool use_guide_tokens = false;  // enable guide tokens to improve TTS accuracy            // NOLINT
};

#include "yaml-cpp/yaml.h"

enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_DEEPSEEK,  // Extract thinking tag contents and return as `message.reasoning_content`
};

struct common_params {
    int                   main_gpu          = 1;
    int                   n_gpu_layers      = 99;
    enum llama_split_mode split_mode        = LLAMA_SPLIT_MODE_LAYER;  // how to split the model across GPUs
    float                 tensor_split[128] = { 0 };  // how split tensors should be distributed across GPUs
    bool                  use_mmap          = true;   // use mmap for faster loads
    bool                  use_mlock         = false;  // use mlock to keep model in memory
    bool                  check_tensors     = false;  // validate tensor data
    bool                  offload_input     = false;  // offload input tensors to CPU
    bool                  enable_ge         = false;  // use GraphEngine
    bool                  display_chat      = false;  // display chat
    int                   presample_count   = -1;     // CANN NPU presampling count, -1 for not use

    bool enable_mla                  = false;
    bool enable_fused_moe            = true;
    bool enable_cann_flash_attention = true;

    enum ggml_type type_k = GGML_TYPE_F16;  // data type for K cache
    enum ggml_type type_v = GGML_TYPE_F16;  // data type for V cache

    uint32_t n_threads       = 64;          // number of threads to use for computation
    uint32_t n_threads_batch = 64;          // number of threads to use for batch processing

    int32_t n_predict  = -1;                // new tokens to predict
    int32_t n_ctx      = 1024 * 16 * 16;    // context size
    int32_t n_batch    = 16;                // logical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep     = 0;                 // number of tokens to keep from initial prompt
    int32_t n_parallel = 16;                // number of parallel sequences to decode

    float defrag_thold = 0.1f;              // defragmentation threshold
    bool  no_perf      = false;             // disable performance metrics

    struct common_params_sampling sampling;

    //std::string model = "/home/zzx/DeepSeek-V2-Chat-f16.gguf";
    std::string model =
        "/root/data/DeepSeek-V2-Lite-Chat-f16.gguf";  // model path                                                    // NOLINT
    std::string model_alias = "";    // model alias                                                   // NOLINT

    bool    special        = false;  // enable special token output
    bool    cont_batching  = true;   // insert new sequences for decoding on-the-fly
    bool    reranking      = false;  // enable reranking support on server
    int32_t port           = 8080;   // server listens on this network port
    int32_t n_threads_http = -1;     // number of threads to process HTTP requests (TODO: support threadpool)

    std::string             hostname         = "127.0.0.1";  // NOLINT
    std::string             chat_template    = "";           // NOLINT
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    // Load parameters from YAML file
    bool load_from_yaml(const std::string & yaml_file) {
        try {
            YAML::Node config = YAML::LoadFile(yaml_file);

            // Load main parameters
            if (config["main_gpu"]) {
                main_gpu = config["main_gpu"].as<int>();
            }
            if (config["n_gpu_layers"]) {
                n_gpu_layers = config["n_gpu_layers"].as<int>();
            }
            if (config["split_mode"]) {
                std::string mode = config["split_mode"].as<std::string>();
                if (mode == "LLAMA_SPLIT_MODE_LAYER") {
                    split_mode = LLAMA_SPLIT_MODE_LAYER;
                } else if (mode == "LLAMA_SPLIT_MODE_ROW") {
                    split_mode = LLAMA_SPLIT_MODE_ROW;
                }
            }
            if (config["tensor_split"]) {
                auto splits = config["tensor_split"].as<std::vector<float>>();
                for (size_t i = 0; i < splits.size() && i < 128; i++) {
                    tensor_split[i] = splits[i];
                }
            }
            if (config["offload_input"]) {
                offload_input = config["offload_input"].as<bool>();
            }
            if (config["use_mmap"]) {
                use_mmap = config["use_mmap"].as<bool>();
            }
            if (config["use_mlock"]) {
                use_mlock = config["use_mlock"].as<bool>();
            }
            if (config["check_tensors"]) {
                check_tensors = config["check_tensors"].as<bool>();
            }
            if (config["enable_mla"]) {
                enable_mla = config["enable_mla"].as<bool>();
            }
            if (config["enable_fused_moe"]) {
                enable_fused_moe = config["enable_fused_moe"].as<bool>();
            }
            if (config["enable_ge"]) {
                enable_ge = config["enable_ge"].as<bool>();
            }
            if (config["presample_count"]) {
                presample_count = config["presample_count"].as<int>();
            }
            if (config["enable_cann_flash_attention"]) {
                enable_cann_flash_attention = config["enable_cann_flash_attention"].as<bool>();
            }
            if (config["display_chat"]) {
                display_chat = config["display_chat"].as<bool>();
            }

            // Load KV cache data types
            if (config["type_k"]) {
                std::string type_str = config["type_k"].as<std::string>();
                if (type_str == "f32" || type_str == "F32" || type_str == "GGML_TYPE_F32") {
                    type_k = GGML_TYPE_F32;
                } else if (type_str == "f16" || type_str == "F16" || type_str == "GGML_TYPE_F16") {
                    type_k = GGML_TYPE_F16;
                } else if (type_str == "bf16" || type_str == "BF16" || type_str == "GGML_TYPE_BF16") {
                    type_k = GGML_TYPE_BF16;
                } else {
                    fprintf(stderr, "Warning: Unknown type_k '%s', using default F16\n", type_str.c_str());
                }
            }
            if (config["type_v"]) {
                std::string type_str = config["type_v"].as<std::string>();
                if (type_str == "f32" || type_str == "F32" || type_str == "GGML_TYPE_F32") {
                    type_v = GGML_TYPE_F32;
                } else if (type_str == "f16" || type_str == "F16" || type_str == "GGML_TYPE_F16") {
                    type_v = GGML_TYPE_F16;
                } else if (type_str == "bf16" || type_str == "BF16" || type_str == "GGML_TYPE_BF16") {
                    type_v = GGML_TYPE_BF16;
                } else {
                    fprintf(stderr, "Warning: Unknown type_v '%s', using default F16\n", type_str.c_str());
                }
            }

            // Load generation parameters
            if (config["n_predict"]) {
                n_predict = config["n_predict"].as<int32_t>();
            }
            if (config["n_ctx"]) {
                n_ctx = config["n_ctx"].as<int32_t>();
            }
            if (config["n_batch"]) {
                n_batch = config["n_batch"].as<int32_t>();
            }
            if (config["n_keep"]) {
                n_keep = config["n_keep"].as<int32_t>();
            }
            if (config["n_parallel"]) {
                n_parallel = config["n_parallel"].as<int32_t>();
                if (n_parallel > 32) {
                    n_parallel = 32;
                    fprintf(stderr, "n_parallel must be <= 32, clamping to %d\n", n_parallel);
                }
            }

            // Load model parameters
            if (config["model"]) {
                model = config["model"].as<std::string>();
            }
            if (config["model_alias"]) {
                model_alias = config["model_alias"].as<std::string>();
            }

            // Load server parameters
            if (config["special"]) {
                special = config["special"].as<bool>();
            }
            if (config["cont_batching"]) {
                cont_batching = config["cont_batching"].as<bool>();
            }
            if (config["reranking"]) {
                reranking = config["reranking"].as<bool>();
            }
            if (config["port"]) {
                port = config["port"].as<int32_t>();
            }
            if (config["n_threads_http"]) {
                n_threads_http = config["n_threads_http"].as<int32_t>();
            }
            if (config["hostname"]) {
                hostname = config["hostname"].as<std::string>();
            }
            if (config["chat_template"]) {
                chat_template = config["chat_template"].as<std::string>();
            }

            // Load thread and performance parameters
            if (config["n_threads"]) {
                n_threads = config["n_threads"].as<uint32_t>();
            }
            if (config["n_threads_batch"]) {
                n_threads_batch = config["n_threads_batch"].as<uint32_t>();
            }
            if (config["defrag_thold"]) {
                defrag_thold = config["defrag_thold"].as<float>();
            }
            if (config["no_perf"]) {
                no_perf = config["no_perf"].as<bool>();
            }

            // Load reasoning format
            if (config["reasoning_format"]) {
                std::string format = config["reasoning_format"].as<std::string>();
                if (format == "COMMON_REASONING_FORMAT_NONE") {
                    reasoning_format = COMMON_REASONING_FORMAT_NONE;
                } else if (format == "COMMON_REASONING_FORMAT_DEEPSEEK") {
                    reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
                }
            }

            return true;
        } catch (const YAML::Exception & e) {
            fprintf(stderr, "Error loading YAML file: %s\n", e.what());
            return false;
        }
    }
};

// call once at the start of a program if it uses libcommon
// initializes the logging system and prints info about the build
void common_init();

//
// String utils
//

std::string string_format(const char * fmt, ...);

std::string              string_join(const std::vector<std::string> & values, const std::string & separator);
std::vector<std::string> string_split(const std::string & str, const std::string & delimiter);
std::string              string_repeat(const std::string & str, size_t n);

template <class T> static std::vector<T> string_split(const std::string & str, char delim) {
    static_assert(!std::is_same<T, std::string>::value, "Please use the specialized version for std::string");
    std::vector<T>     values;
    std::istringstream str_stream(str);
    std::string        token;
    while (std::getline(str_stream, token, delim)) {
        T                  value;
        std::istringstream token_stream(token);
        token_stream >> value;
        values.push_back(value);
    }
    return values;
}

template <> std::vector<std::string> string_split<std::string>(const std::string & input, char separator) {
    std::vector<std::string> parts;
    size_t                   begin_pos     = 0;
    size_t                   separator_pos = input.find(separator);
    while (separator_pos != std::string::npos) {
        std::string part = input.substr(begin_pos, separator_pos - begin_pos);
        parts.emplace_back(part);
        begin_pos     = separator_pos + 1;
        separator_pos = input.find(separator, begin_pos);
    }
    parts.emplace_back(input.substr(begin_pos, separator_pos - begin_pos));
    return parts;
}

static bool string_starts_with(const std::string & str,
                               const std::string & prefix) {  // While we wait for C++20's std::string::starts_with...
    return str.rfind(prefix, 0) == 0;
}

static bool string_ends_with(const std::string & str,
                             const std::string & suffix) {  // While we wait for C++20's std::string::ends_with...
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct common_init_result {
    llama_model_ptr   model;
    llama_context_ptr context;
};

struct common_init_result common_init_from_params(common_params & params);

void common_batch_clear(struct llama_batch & batch);

void common_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos,
                      const std::vector<llama_seq_id> & seq_ids, bool logits);

size_t common_lcp(const llama_tokens & a, const llama_tokens & b);

size_t common_lcs(const llama_tokens & a, const llama_tokens & b);

std::vector<llama_token> common_tokenize(const struct llama_context * ctx, const std::string & text, bool add_special,
                                         bool parse_special = false);

std::vector<llama_token> common_tokenize(const struct llama_vocab * vocab, const std::string & text, bool add_special,
                                         bool parse_special = false);

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special = true);

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special = true);

std::string common_detokenize(const struct llama_context * ctx, const std::vector<llama_token> & tokens,
                              bool special = true);

std::string common_detokenize(const struct llama_vocab * vocab, const std::vector<llama_token> & tokens,
                              bool special = true);

struct common_control_vector_data {
    int                n_embd;
    std::vector<float> data;
};

struct common_control_vector_load_info {
    float strength;

    std::string fname;
};

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos);

size_t common_llama_batch_max_comm_size(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max);

void common_load_batch(struct llama_batch & batch, int32_t embd, const char * data);

size_t common_dump_batch(const struct llama_batch & batch, int32_t embd, char * data);
