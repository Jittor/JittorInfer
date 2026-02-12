#if defined(_MSC_VER)
#    define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#include "common_local.h"

#include "ggml.h"
#include "gguf.h"
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <codecvt>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "json-schema-to-grammar_local.h"
#include "json.hpp"
#include "llama.h"

using json = nlohmann::ordered_json;

//
// String utils
//

std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX);  // NOLINT
    std::vector<char> buf(size + 1);
    int               size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t                   start = 0;
    size_t                   end   = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end   = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

//
// Model utils
//

static llama_model_params common_model_params_to_llama_local(common_params & params) {
    auto mparams = llama_model_default_params();

    mparams.main_gpu                    = params.main_gpu;
    mparams.split_mode                  = params.split_mode;
    mparams.tensor_split                = params.tensor_split;
    mparams.use_mmap                    = params.use_mmap;
    mparams.use_mlock                   = params.use_mlock;
    mparams.check_tensors               = params.check_tensors;
    mparams.n_gpu_layers                = params.n_gpu_layers;
    mparams.kv_overrides                = NULL;
    mparams.enable_mla                  = params.enable_mla;
    mparams.enable_fused_moe            = params.enable_fused_moe;
    mparams.enable_cann_flash_attention = params.enable_cann_flash_attention;
    mparams.offload_input               = params.offload_input;

    return mparams;
}

struct common_init_result common_init_from_params(common_params & params) {
    common_init_result iparams;
    auto               mparams = common_model_params_to_llama_local(params);

    llama_model * model = nullptr;

    model = llama_model_load_from_file(params.model.c_str(), mparams);

    if (model == NULL) {
        return iparams;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (params.reranking) {
        bool ok = true;

        if (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL) {
            ok = false;
        }

        if (llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
            ok = false;
        }

        if (llama_vocab_sep(vocab) == LLAMA_TOKEN_NULL) {
            ok = false;
        }

        if (!ok) {
            llama_model_free(model);

            return iparams;
        }
    }

    auto cparams            = llama_context_default_params();
    cparams.n_ctx           = params.n_ctx;
    cparams.n_batch         = params.n_batch;
    cparams.n_seq_max       = params.n_parallel;
    cparams.enable_ge       = params.enable_ge;
    cparams.presample_count = params.presample_count;
    cparams.defrag_thold    = params.defrag_thold;
    cparams.type_k          = params.type_k;
    cparams.type_v          = params.type_v;

    llama_context * lctx = llama_init_from_model(model, cparams);
    if (lctx == NULL) {
        llama_model_free(model);
        return iparams;
    }

    // if (params.warmup) {

    //     std::vector<llama_token> tmp;
    //     llama_token bos = llama_vocab_bos(vocab);
    //     llama_token eos = llama_vocab_eos(vocab);

    //     // some models (e.g. T5) don't have a BOS token
    //     if (bos != LLAMA_TOKEN_NULL) {
    //         tmp.push_back(bos);
    //     }
    //     if (eos != LLAMA_TOKEN_NULL) {
    //         tmp.push_back(eos);
    //     }
    //     if (tmp.empty()) {
    //         tmp.push_back(0);
    //     }

    //     if (llama_model_has_encoder(model)) {
    //         llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size()));
    //         llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
    //         if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
    //             decoder_start_token_id = bos;
    //         }
    //         tmp.clear();
    //         tmp.push_back(decoder_start_token_id);
    //     }
    //     if (llama_model_has_decoder(model)) {
    //         llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch)));
    //     }
    //     llama_kv_cache_clear(lctx);
    //     llama_synchronize(lctx);
    //     llama_perf_context_reset(lctx);
    // }

    iparams.model.reset(model);
    iparams.context.reset(lctx);

    return iparams;
}

//
// Batch utils
//

void common_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void common_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos,
                      const std::vector<llama_seq_id> & seq_ids, bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token[batch.n_tokens]    = id;
    batch.pos[batch.n_tokens]      = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;

    batch.n_tokens++;
}

//
// Token utils
//

size_t common_lcp(const llama_tokens & a, const llama_tokens & b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {
    }

    return i;
}

size_t common_lcs(const llama_tokens & a, const llama_tokens & b) {
    // check for empty sequences
    if (a.empty() || b.empty()) {
        return 0;
    }

    // get the lengths of the input sequences
    size_t a_len = a.size();
    size_t b_len = b.size();

    // initialize the maximum length of the longest common subsequence (LCS)
    size_t max_length = 0;

    // use two rows instead of a 2D matrix to optimize space
    std::vector<size_t> prev_row(b_len + 1, 0);
    std::vector<size_t> curr_row(b_len + 1, 0);

    // iterate through the elements of a
    for (size_t i = 1; i <= a_len; i++) {
        // iterate through the elements of b
        for (size_t j = 1; j <= b_len; j++) {
            // if elements at the current positions match
            if (a[i - 1] == b[j - 1]) {
                // if it's the first element of either sequences, set LCS length to 1
                if (i == 1 || j == 1) {
                    curr_row[j] = 1;
                } else {
                    // increment LCS length by 1 compared to the previous element
                    curr_row[j] = prev_row[j - 1] + 1;
                }

                // update max_length if necessary
                if (curr_row[j] > max_length) {
                    max_length = curr_row[j];
                }
            } else {
                // reset LCS length if elements don't match
                curr_row[j] = 0;
            }
        }

        // update the previous row for the next iteration
        prev_row = curr_row;
    }

    // return the maximum length of the LCS
    return max_length;
}

//
// Vocab utils
//

std::vector<llama_token> common_tokenize(const struct llama_context * ctx, const std::string & text, bool add_special,
                                         bool parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> common_tokenize(const struct llama_vocab * vocab, const std::string & text, bool add_special,
                                         bool parse_special) {
    // upper limit for the number of tokens
    int                      n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens =
        llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check =
            llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    } else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string common_detokenize(const struct llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_detokenize(vocab, tokens, special);
}

std::string common_detokenize(const struct llama_vocab * vocab, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(vocab, tokens.data(), (int32_t) tokens.size(), &text[0], (int32_t) text.size(),
                                       false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(vocab, tokens.data(), (int32_t) tokens.size(), &text[0], (int32_t) text.size(),
                                   false, special);
        GGML_ASSERT(n_chars <=
                    (int32_t) text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//
// Control vector utils
//

static common_control_vector_data common_control_vector_load_one(const common_control_vector_load_info & load_info) {
    common_control_vector_data result = { -1, {} };

    ggml_context *          ctx              = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(load_info.fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.n_embd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float *       dst = result.data.data() + result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }
    }

    if (result.n_embd == -1) {
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos) {
    common_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = common_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        result.data.clear();
    }

    return result;
}

size_t common_llama_batch_max_comm_size(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    const int32_t n_tokens      = n_tokens_alloc;
    const size_t  n_tokens_size = sizeof(n_tokens);
    const size_t  token_size    = (embd ? sizeof(float) * embd : sizeof(llama_token)) * n_tokens;
    const size_t  pos_size      = sizeof(llama_pos) * n_tokens;
    const size_t  n_seq_id_size = sizeof(int32_t) * n_tokens;
    const size_t  logits_size   = sizeof(int8_t) * n_tokens;
    const size_t  seq_id_size   = sizeof(llama_seq_id) * n_seq_max;
    return n_tokens_size + token_size + pos_size + n_seq_id_size + logits_size + seq_id_size;
}

static void read_move(void * dst, const char * src, size_t & off, size_t size) {
    memcpy(dst, src + off, size);
    off += size;
}

void common_load_batch(struct llama_batch & batch, int32_t embd, const char * data) {
    size_t now_id = 0;
    read_move((void *) &batch.n_tokens, data, now_id, sizeof(batch.n_tokens));
    const int32_t n_tokens      = batch.n_tokens;
    const size_t  token_size    = (embd ? sizeof(float) * embd : sizeof(llama_token)) * n_tokens;
    const size_t  pos_size      = sizeof(llama_pos) * n_tokens;
    const size_t  n_seq_id_size = sizeof(int32_t) * n_tokens;
    const size_t  logits_size   = sizeof(int8_t) * n_tokens;
    read_move((void *) batch.token, data, now_id, token_size);
    read_move((void *) batch.pos, data, now_id, pos_size);
    read_move((void *) batch.n_seq_id, data, now_id, n_seq_id_size);
    read_move((void *) batch.logits, data, now_id, logits_size);
    for (int i = 0; i < n_tokens; ++i) {
        read_move((void *) batch.seq_id[i], data, now_id, sizeof(llama_seq_id) * batch.n_seq_id[i]);
    }
}

static void copy_move(const void * src, char * dst, size_t & off, size_t size) {
    memcpy(dst + off, src, size);
    off += size;
}

size_t common_dump_batch(const struct llama_batch & batch, int32_t embd, char * data) {
    size_t        now_id        = 0;
    const int32_t n_tokens      = batch.n_tokens;
    const size_t  token_size    = (embd ? sizeof(float) * embd : sizeof(llama_token)) * n_tokens;
    const size_t  pos_size      = sizeof(llama_pos) * n_tokens;
    const size_t  n_seq_id_size = sizeof(int32_t) * n_tokens;
    const size_t  logits_size   = sizeof(int8_t) * n_tokens;
    copy_move((void *) &batch.n_tokens, data, now_id, sizeof(batch.n_tokens));
    copy_move((void *) batch.token, data, now_id, token_size);
    copy_move((void *) batch.pos, data, now_id, pos_size);
    copy_move((void *) batch.n_seq_id, data, now_id, n_seq_id_size);
    copy_move((void *) batch.logits, data, now_id, logits_size);
    for (int i = 0; i < n_tokens; ++i) {
        copy_move((void *) batch.seq_id[i], data, now_id, sizeof(llama_seq_id) * batch.n_seq_id[i]);
    }
    return now_id;
}
