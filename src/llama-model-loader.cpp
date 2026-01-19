#include "llama-model-loader.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <future>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-impl.h"

#ifdef LLAMA_MPI_SUPPORT
#    define OMPI_SKIP_MPICXX 1
#    include <mpi.h>
#endif

static const size_t kiB = 1024;
static const size_t MiB = 1024 * kiB;
static const size_t GiB = 1024 * MiB;

const static char * llama_file_version_name(llama_fver version) {
    switch (version) {
        case GGUF_FILE_VERSION_V1:
            return "GGUF V1 (support until nov 2023)";
        case GGUF_FILE_VERSION_V2:
            return "GGUF V2";
        case GGUF_FILE_VERSION_V3:
            return "GGUF V3 (latest)";
    }

    return "unknown";
}

static std::string llama_model_ftype_name(llama_ftype ftype) {
    if (ftype & LLAMA_FTYPE_GUESSED) {
        return llama_model_ftype_name((enum llama_ftype)(ftype & ~LLAMA_FTYPE_GUESSED)) + " (guessed)";
    }

    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:
            return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:
            return "F16";
        case LLAMA_FTYPE_MOSTLY_BF16:
            return "BF16";
        case LLAMA_FTYPE_MOSTLY_Q4_0:
            return "Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1:
            return "Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q5_0:
            return "Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1:
            return "Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q8_0:
            return "Q8_0";
        case LLAMA_FTYPE_MOSTLY_Q2_K:
            return "Q2_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
            return "Q2_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
            return "Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
            return "Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:
            return "Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
            return "Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:
            return "Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
            return "Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:
            return "Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K:
            return "Q6_K";
        case LLAMA_FTYPE_MOSTLY_TQ1_0:
            return "TQ1_0 - 1.69 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_TQ2_0:
            return "TQ2_0 - 2.06 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:
            return "IQ2_XXS - 2.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:
            return "IQ2_XS - 2.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_S:
            return "IQ2_S - 2.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_M:
            return "IQ2_M - 2.7 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:
            return "IQ3_XS - 3.3 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:
            return "IQ3_XXS - 3.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_S:
            return "IQ1_S - 1.5625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_M:
            return "IQ1_M - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:
            return "IQ4_NL - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:
            return "IQ4_XS - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_S:
            return "IQ3_S - 3.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_M:
            return "IQ3_S mix - 3.66 bpw";

        default:
            return "unknown, may not work";
    }
}

namespace GGUFMeta {
template <typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int64_t)> struct GKV_Base_Type {
    static constexpr gguf_type gt = gt_;

    static T getter(const gguf_context * ctx, const int kid) { return gfun(ctx, kid); }
};

template <typename T> struct GKV_Base;

template <> struct GKV_Base<bool> : GKV_Base_Type<bool, GGUF_TYPE_BOOL, gguf_get_val_bool> {};

template <> struct GKV_Base<uint8_t> : GKV_Base_Type<uint8_t, GGUF_TYPE_UINT8, gguf_get_val_u8> {};

template <> struct GKV_Base<uint16_t> : GKV_Base_Type<uint16_t, GGUF_TYPE_UINT16, gguf_get_val_u16> {};

template <> struct GKV_Base<uint32_t> : GKV_Base_Type<uint32_t, GGUF_TYPE_UINT32, gguf_get_val_u32> {};

template <> struct GKV_Base<uint64_t> : GKV_Base_Type<uint64_t, GGUF_TYPE_UINT64, gguf_get_val_u64> {};

template <> struct GKV_Base<int8_t> : GKV_Base_Type<int8_t, GGUF_TYPE_INT8, gguf_get_val_i8> {};

template <> struct GKV_Base<int16_t> : GKV_Base_Type<int16_t, GGUF_TYPE_INT16, gguf_get_val_i16> {};

template <> struct GKV_Base<int32_t> : GKV_Base_Type<int32_t, GGUF_TYPE_INT32, gguf_get_val_i32> {};

template <> struct GKV_Base<int64_t> : GKV_Base_Type<int64_t, GGUF_TYPE_INT64, gguf_get_val_i64> {};

template <> struct GKV_Base<float> : GKV_Base_Type<float, GGUF_TYPE_FLOAT32, gguf_get_val_f32> {};

template <> struct GKV_Base<double> : GKV_Base_Type<double, GGUF_TYPE_FLOAT64, gguf_get_val_f64> {};

template <> struct GKV_Base<const char *> : GKV_Base_Type<const char *, GGUF_TYPE_STRING, gguf_get_val_str> {};

template <> struct GKV_Base<std::string> {
    static constexpr gguf_type gt = GGUF_TYPE_STRING;

    static std::string getter(const gguf_context * ctx, const int kid) { return gguf_get_val_str(ctx, kid); }
};

struct ArrayInfo {
    const gguf_type gt;
    const size_t    length;
    const void *    data;
};

template <> struct GKV_Base<ArrayInfo> {
  public:
    static constexpr gguf_type gt = GGUF_TYPE_ARRAY;

    static ArrayInfo getter(const gguf_context * ctx, const int k) {
        const enum gguf_type arr_type = gguf_get_arr_type(ctx, k);
        return ArrayInfo{
            arr_type,
            gguf_get_arr_n(ctx, k),
            arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx, k),
        };
    }
};

template <typename T> class GKV : public GKV_Base<T> {
    GKV() = delete;

  public:
    static T get_kv(const gguf_context * ctx, const int k) {
        const enum gguf_type kt = gguf_get_kv_type(ctx, k);

        if (kt != GKV::gt) {
            throw std::runtime_error(format("key %s has wrong type %s but expected type %s", gguf_get_key(ctx, k),
                                            gguf_type_name(kt), gguf_type_name(GKV::gt)));
        }
        return GKV::getter(ctx, k);
    }

    static const char * override_type_to_str(const llama_model_kv_override_type ty) {
        switch (ty) {
            case LLAMA_KV_OVERRIDE_TYPE_BOOL:
                return "bool";
            case LLAMA_KV_OVERRIDE_TYPE_INT:
                return "int";
            case LLAMA_KV_OVERRIDE_TYPE_FLOAT:
                return "float";
            case LLAMA_KV_OVERRIDE_TYPE_STR:
                return "str";
        }
        return "unknown";
    }

    static bool validate_override(const llama_model_kv_override_type     expected_type,
                                  const struct llama_model_kv_override * ovrd) {
        if (!ovrd) {
            return false;
        }
        if (ovrd->tag == expected_type) {
            LLAMA_LOG_INFO("%s: Using metadata override (%5s) '%s' = ", __func__, override_type_to_str(ovrd->tag),
                           ovrd->key);
            switch (ovrd->tag) {
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:
                    {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_bool ? "true" : "false");
                    }
                    break;
                case LLAMA_KV_OVERRIDE_TYPE_INT:
                    {
                        LLAMA_LOG_INFO("%" PRId64 "\n", ovrd->val_i64);
                    }
                    break;
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT:
                    {
                        LLAMA_LOG_INFO("%.6f\n", ovrd->val_f64);
                    }
                    break;
                case LLAMA_KV_OVERRIDE_TYPE_STR:
                    {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_str);
                    }
                    break;
                default:
                    // Shouldn't be possible to end up here, but just in case...
                    throw std::runtime_error(format("Unsupported attempt to override %s type for metadata key %s\n",
                                                    override_type_to_str(ovrd->tag), ovrd->key));
            }
            return true;
        }
        LLAMA_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n", __func__,
                       ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
        return false;
    }

    template <typename OT>
    static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type try_override(
        OT & target, const struct llama_model_kv_override * ovrd) {
        if (validate_override(LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) {
            target = ovrd->val_bool;
            return true;
        }
        return false;
    }

    template <typename OT>
    static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
    try_override(OT & target, const struct llama_model_kv_override * ovrd) {
        if (validate_override(LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
            target = ovrd->val_i64;
            return true;
        }
        return false;
    }

    template <typename OT>
    static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type try_override(
        T & target, const struct llama_model_kv_override * ovrd) {
        if (validate_override(LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) {
            target = ovrd->val_f64;
            return true;
        }
        return false;
    }

    template <typename OT>
    static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type try_override(
        T & target, const struct llama_model_kv_override * ovrd) {
        if (validate_override(LLAMA_KV_OVERRIDE_TYPE_STR, ovrd)) {
            target = ovrd->val_str;
            return true;
        }
        return false;
    }

    static bool set(const gguf_context * ctx, const int k, T & target,
                    const struct llama_model_kv_override * ovrd = nullptr) {
        if (try_override<T>(target, ovrd)) {
            return true;
        }
        if (k < 0) {
            return false;
        }
        target = get_kv(ctx, k);
        return true;
    }

    static bool set(const gguf_context * ctx, const char * key, T & target,
                    const struct llama_model_kv_override * ovrd = nullptr) {
        return set(ctx, gguf_find_key(ctx, key), target, ovrd);
    }

    static bool set(const gguf_context * ctx, const std::string & key, T & target,
                    const struct llama_model_kv_override * ovrd = nullptr) {
        return set(ctx, key.c_str(), target, ovrd);
    }
};
}  // namespace GGUFMeta

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type llama_model_loader::get_arr_n(const std::string & key,
                                                                                              T &  result,
                                                                                              bool required) {
    const int kid = gguf_find_key(meta.get(), key.c_str());

    if (kid < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }
        return false;
    }

    struct GGUFMeta::ArrayInfo arr_info = GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta.get(), kid);

    result = arr_info.length;
    return true;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type llama_model_loader::get_arr_n(enum llm_kv kid,
                                                                                              T &         result,
                                                                                              bool        required) {
    return get_arr_n(llm_kv(kid), result, required);
}

template bool llama_model_loader::get_arr_n(enum llm_kv kid, uint32_t & result, bool required);

template <typename T>
bool llama_model_loader::get_arr(const std::string & key, std::vector<T> & result, bool required) {
    const int kid = gguf_find_key(meta.get(), key.c_str());

    if (kid < 0 || gguf_get_kv_type(meta.get(), kid) != GGUF_TYPE_ARRAY) {
        if (required) {
            throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
        }
        return false;
    }

    struct GGUFMeta::ArrayInfo arr_info = GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta.get(), kid);

    switch (arr_info.gt) {
        case GGUF_TYPE_FLOAT32:
            GGML_ASSERT((std::is_same<T, float>::value));
            break;
        case GGUF_TYPE_INT32:
            GGML_ASSERT((std::is_same<T, int32_t>::value) || (std::is_same<T, uint32_t>::value));
            break;
        default:
            throw std::runtime_error(format("%s is not a float32, int32 array", key.c_str()));
    }

    result.resize(arr_info.length);
    result.assign((const T *) arr_info.data, (const T *) arr_info.data + arr_info.length);

    return true;
}

template <typename T, size_t N_MAX>
bool llama_model_loader::get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required) {
    const int kid = gguf_find_key(meta.get(), key.c_str());

    if (kid < 0 || gguf_get_kv_type(meta.get(), kid) != GGUF_TYPE_ARRAY) {
        if (required) {
            throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
        }
        return false;
    }

    struct GGUFMeta::ArrayInfo arr_info = GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta.get(), kid);

    switch (arr_info.gt) {
        case GGUF_TYPE_FLOAT32:
            GGML_ASSERT((std::is_same<T, float>::value));
            break;
        case GGUF_TYPE_INT32:
            GGML_ASSERT((std::is_same<T, int32_t>::value) || (std::is_same<T, uint32_t>::value));
            break;
        default:
            throw std::runtime_error(format("%s is not a float32, int32 array", key.c_str()));
    }

    if (arr_info.length > N_MAX) {
        throw std::runtime_error(format("array length %u for key %s exceeds max %u", (uint32_t) arr_info.length,
                                        key.c_str(), (uint32_t) N_MAX));
    }

    std::copy((const T *) arr_info.data, (const T *) arr_info.data + arr_info.length, result.begin());

    return true;
}

template <typename T> bool llama_model_loader::get_arr(enum llm_kv kid, T & result, bool required) {
    return get_arr(llm_kv(kid), result, required);
}

template <typename T> bool llama_model_loader::get_key(const std::string & key, T & result, bool required) {
    auto it = kv_overrides.find(key);

    const struct llama_model_kv_override * override = it != kv_overrides.end() ? &it->second : nullptr;

    const bool found = GGUFMeta::GKV<T>::set(meta.get(), key, result, override);

    if (required && !found) {
        throw std::runtime_error(format("key not found in model: %s", key.c_str()));
    }

    return found;
}

template <typename T> bool llama_model_loader::get_key(enum llm_kv kid, T & result, bool required) {
    return get_key(llm_kv(kid), result, required);
}

template bool llama_model_loader::get_key<bool>(enum llm_kv kid, bool & result, bool required);
template bool llama_model_loader::get_key<float>(enum llm_kv kid, float & result, bool required);
template bool llama_model_loader::get_key<uint32_t>(enum llm_kv kid, uint32_t & result, bool required);
template bool llama_model_loader::get_key<std::string>(enum llm_kv kid, std::string & result, bool required);

template <> bool llama_model_loader::get_key(enum llm_kv kid, enum llama_pooling_type & result, bool required) {
    uint32_t   tmp;
    const bool found = get_key(kid, tmp, required);
    if (found) {
        result = (enum llama_pooling_type) tmp;
    } else {
        result = LLAMA_POOLING_TYPE_UNSPECIFIED;
    }
    return found;
}

// get array of n <= N_MAX elements, or a single element repeated n times
template <typename T, size_t N_MAX>
bool llama_model_loader::get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n,
                                        bool required) {
    const int kid = gguf_find_key(meta.get(), key.c_str());

    if (kid < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }
        return false;
    }

    if (n > N_MAX) {
        throw std::runtime_error(format("n > N_MAX: %u > %u for key %s", n, (uint32_t) N_MAX, key.c_str()));
    }

    if (gguf_get_kv_type(meta.get(), kid) == GGUF_TYPE_ARRAY) {
        struct GGUFMeta::ArrayInfo arr_info = GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta.get(), kid);

        if (n != arr_info.length) {
            throw std::runtime_error(format("key %s has wrong array length; expected %u, got %u", key.c_str(), n,
                                            (uint32_t) arr_info.length));
        }

        return get_arr(key, result, required);
    }

    T value;

    bool ok = get_key(key, value, required);
    if (!ok) {
        return false;
    }

    for (uint32_t i = 0; i < n; i++) {
        result[i] = value;
    }

    return true;
}

template <typename T> bool llama_model_loader::get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required) {
    return get_key_or_arr(llm_kv(kid), result, n, required);
}

// TODO: this is not very clever - figure out something better
template bool llama_model_loader::get_key_or_arr<std::array<int, 4>>(enum llm_kv kid, std::array<int, 4> & result,
                                                                     uint32_t n, bool required);
template bool llama_model_loader::get_key_or_arr<std::array<uint32_t, 512>>(enum llm_kv                 kid,
                                                                            std::array<uint32_t, 512> & result,
                                                                            uint32_t n, bool required);

llama_model_loader::llama_model_loader(const std::string & fname, std::vector<std::string> & /* unused */,
                                       bool use_mmap, bool check_tensors,
                                       const struct llama_model_kv_override * param_overrides_p) {
    int trace = 0;
    if (getenv("LLAMA_TRACE")) {
        trace = atoi(getenv("LLAMA_TRACE"));
    }

    GGML_ASSERT(param_overrides_p == nullptr);

    // Load the main GGUF
    struct ggml_context *   ctx    = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    meta.reset(gguf_init_from_file(fname.c_str(), params));
    if (!meta) {
        throw std::runtime_error(format("%s: failed to load model from %s\n", __func__, fname.c_str()));
    }

    get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
    llm_kv = LLM_KV(llm_arch_from_string(arch_name));

    files.emplace_back(new llama_file(fname.c_str(), "rb"));
    contexts.emplace_back(ctx);

    // Save tensors data offset of the main file.
    // For subsidiary files, `meta` tensor data offset must not be used,
    // so we build a unified tensors index for weights.
    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string tensor_name = std::string(cur->name);
        // make sure there is no duplicated tensor names
        if (weights_map.find(tensor_name) != weights_map.end()) {
            throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
        }
        n_elements += ggml_nelements(cur);
        n_bytes += ggml_nbytes(cur);
        weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, meta.get(), cur));
    }
    uint16_t n_split = 0;
    get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

    // !!! hack !!!
    // n_split = 1;

    // Load additional GGML contexts
    GGML_ASSERT(n_split <= 1);

    n_kv      = gguf_get_n_kv(meta.get());
    n_tensors = weights_map.size();

    fver = (enum llama_fver) gguf_get_version(meta.get());

    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n", __func__, n_kv,
                   n_tensors, fname.c_str(), llama_file_version_name(fver));

    // determine file type based on the number of tensors for each quantization and print meta data
    // TODO: make optional
    {
        std::map<enum ggml_type, uint32_t> n_type;

        uint32_t       n_type_max = 0;
        enum ggml_type type_max   = GGML_TYPE_F32;

        for (const auto & it : weights_map) {
            const llama_tensor_weight & w      = it.second;
            const ggml_tensor *         tensor = w.tensor;

            enum ggml_type type = tensor->type;

            n_type[type]++;

            if (n_type_max < n_type[type]) {
                n_type_max = n_type[type];
                type_max   = type;
            }

            if (trace > 0) {
                const uint16_t sid = w.idx;
                LLAMA_LOG_INFO("%s: - tensor split %2d: %32s %-8s [ %s ]\n", __func__, sid, ggml_get_name(tensor),
                               ggml_type_name(type), llama_format_tensor_shape(tensor).c_str());
            }
        }

        switch (type_max) {
            case GGML_TYPE_F32:
                ftype = LLAMA_FTYPE_ALL_F32;
                break;
            case GGML_TYPE_F16:
                ftype = LLAMA_FTYPE_MOSTLY_F16;
                break;
            case GGML_TYPE_BF16:
                ftype = LLAMA_FTYPE_MOSTLY_BF16;
                break;
            case GGML_TYPE_Q4_0:
                ftype = LLAMA_FTYPE_MOSTLY_Q4_0;
                break;
            case GGML_TYPE_Q4_1:
                ftype = LLAMA_FTYPE_MOSTLY_Q4_1;
                break;
            case GGML_TYPE_Q5_0:
                ftype = LLAMA_FTYPE_MOSTLY_Q5_0;
                break;
            case GGML_TYPE_Q5_1:
                ftype = LLAMA_FTYPE_MOSTLY_Q5_1;
                break;
            case GGML_TYPE_Q8_0:
                ftype = LLAMA_FTYPE_MOSTLY_Q8_0;
                break;
            case GGML_TYPE_Q2_K:
                ftype = LLAMA_FTYPE_MOSTLY_Q2_K;
                break;
            case GGML_TYPE_Q3_K:
                ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;
                break;
            case GGML_TYPE_Q4_K:
                ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
                break;
            case GGML_TYPE_Q5_K:
                ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;
                break;
            case GGML_TYPE_Q6_K:
                ftype = LLAMA_FTYPE_MOSTLY_Q6_K;
                break;
            case GGML_TYPE_TQ1_0:
                ftype = LLAMA_FTYPE_MOSTLY_TQ1_0;
                break;
            case GGML_TYPE_TQ2_0:
                ftype = LLAMA_FTYPE_MOSTLY_TQ2_0;
                break;
            case GGML_TYPE_IQ2_XXS:
                ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS;
                break;
            case GGML_TYPE_IQ2_XS:
                ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;
                break;
            case GGML_TYPE_IQ2_S:
                ftype = LLAMA_FTYPE_MOSTLY_IQ2_S;
                break;
            case GGML_TYPE_IQ3_XXS:
                ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS;
                break;
            case GGML_TYPE_IQ1_S:
                ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;
                break;
            case GGML_TYPE_IQ1_M:
                ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;
                break;
            case GGML_TYPE_IQ4_NL:
                ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;
                break;
            case GGML_TYPE_IQ4_XS:
                ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;
                break;
            case GGML_TYPE_IQ3_S:
                ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;
                break;
            default:
                {
                    LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                    ftype = LLAMA_FTYPE_ALL_F32;
                }
                break;
        }

        // this is a way to mark that we have "guessed" the file type
        ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

        {
            const int kid = gguf_find_key(meta.get(), "general.file_type");  // TODO: use LLM_KV
            if (kid >= 0) {
                ftype = (llama_ftype) gguf_get_val_u32(meta.get(), kid);
            }
        }

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

        for (int i = 0; i < n_kv; i++) {
            const char *         name = gguf_get_key(meta.get(), i);
            const enum gguf_type type = gguf_get_kv_type(meta.get(), i);
            const std::string    type_name =
                type == GGUF_TYPE_ARRAY ?
                       format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(meta.get(), i)),
                              gguf_get_arr_n(meta.get(), i)) :
                       gguf_type_name(type);

            std::string  value         = gguf_kv_to_str(meta.get(), i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    if (!llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        use_mmap = false;
    }

    this->use_mmap      = use_mmap;
    this->check_tensors = check_tensors;
}

std::string llama_model_loader::get_arch_name() const {
    return arch_name;
}

enum llm_arch llama_model_loader::get_arch() const {
    return llm_kv.arch;
}

const llama_model_loader::llama_tensor_weight * llama_model_loader::get_weight(const char * name) const {
    auto pos = weights_map.find(name);
    if (pos != weights_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

const llama_model_loader::llama_tensor_viewer * llama_model_loader::get_spliter(const char * name) const {
    auto pos = spliter_map.find(name);
    if (pos != spliter_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

struct ggml_tensor * llama_model_loader::get_tensor_meta(const char * name) const {
    const auto * weight = get_weight(name);
    if (!weight) {
        return nullptr;
    }
    return weight->tensor;
}

const struct ggml_tensor * llama_model_loader::check_tensor_dims(const std::string &          name,
                                                                 const std::vector<int64_t> & ne, bool required,
                                                                 llama_tensor_viewer spliter) const {
    const struct ggml_tensor * cur = get_tensor_meta(name.c_str());

    if (cur == NULL) {
        if (!required) {
            return NULL;
        }
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }

    {
        bool is_ok = true;
        if (spliter.mode == LLAMA_AVG_SPLIT) {
            int64_t dim_size_ori = 1;
            int64_t dim_size_tar = 1;
            size_t  compare_dim  = 0;
            for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
                dim_size_ori *= spliter.ne[i];
                dim_size_tar *= spliter.ne[i];
                if (i == (size_t) spliter.split_dim) {
                    dim_size_ori *= spliter.split_num;
                    dim_size_tar *= 1;
                }
                if (dim_size_ori == cur->ne[compare_dim]) {
                    if (compare_dim < ne.size() && dim_size_tar != ne[compare_dim]) {
                        is_ok = false;
                        break;
                    }
                    if (compare_dim >= ne.size() && cur->ne[compare_dim] != 1) {
                        is_ok = false;
                        break;
                    }
                    dim_size_ori = 1;
                    dim_size_tar = 1;
                    ++compare_dim;
                    continue;
                }
                if (dim_size_ori > cur->ne[compare_dim]) {
                    is_ok = false;
                    break;
                }
            }
            if (dim_size_ori > 1) {
                is_ok = false;
            }
            while (is_ok && compare_dim < GGML_MAX_DIMS) {
                if (compare_dim < ne.size() && ne[compare_dim] != 1) {
                    is_ok = false;
                    break;
                }
                if (cur->ne[compare_dim] != 1) {
                    is_ok = false;
                    break;
                }
                ++compare_dim;
            }
            if (!is_ok) {
                std::vector<int64_t> ne2;
                ne2.reserve(GGML_MAX_DIMS);
                for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
                    ne2.push_back(spliter.ne[i]);
                }
                throw std::runtime_error(
                    format("%s: tensor '%s' has wrong shape; expected %s, view size: %s, actual size: %s", __func__,
                           name.c_str(), llama_format_tensor_shape(ne).c_str(), llama_format_tensor_shape(ne2).c_str(),
                           llama_format_tensor_shape(cur).c_str()));
            }
        } else {
            for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
                if ((i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)) {
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok) {
                throw std::runtime_error(format("%s: tensor '%s' has wrong shape; expected %s, got %s", __func__,
                                                name.c_str(), llama_format_tensor_shape(ne).c_str(),
                                                llama_format_tensor_shape(cur).c_str()));
            }
        }
    }

    return cur;
}

struct ggml_tensor * llama_model_loader::create_tensor(struct ggml_context * ctx, const std::string & name,
                                                       const std::vector<int64_t> & ne, int flags,
                                                       llama_tensor_viewer spliter) {
    const struct ggml_tensor * cur = check_tensor_dims(name, ne, !(flags & TENSOR_NOT_REQUIRED), spliter);

    if (cur == NULL) {
        return NULL;
    }

    bool duplicated = flags & TENSOR_DUPLICATED;

    struct ggml_tensor * tensor;
    if (spliter.mode == LLAMA_REPEAT) {
        tensor = ggml_dup_tensor(ctx, cur);
    } else if (spliter.mode == LLAMA_AVG_SPLIT) {
        int     cur_split_dim = 0;
        int64_t cur_dim_size  = 1;
        for (int i = 0; i <= spliter.split_dim; ++i) {
            if (cur_dim_size == cur->ne[cur_split_dim]) {
                cur_split_dim++;
                cur_dim_size = 1;
            }
            cur_dim_size *= spliter.ne[i];
        }
        int64_t tar_ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            tar_ne[i] = cur->ne[i];
        }
        GGML_ASSERT(cur->ne[cur_split_dim] % spliter.split_num == 0);
        tar_ne[cur_split_dim] /= spliter.split_num;
        tensor = ggml_new_tensor(ctx, cur->type, GGML_MAX_DIMS, tar_ne);
    } else {
        throw std::runtime_error(format("%s: unknown tensor viewer mode %d", __func__, spliter.mode));
    }

    std::string target_name = spliter.add_tp_id(name);
    spliter_map.emplace(target_name, spliter);
    ggml_set_name(tensor, target_name.c_str());

    if (duplicated) {
        size_data += ggml_nbytes(cur);
    } else {
        n_created++;
    }

    return tensor;
}

void llama_model_loader::done_getting_tensors() const {
    if (n_created != n_tensors) {
        throw std::runtime_error(
            format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
    }
}

void llama_model_loader::init_mappings(bool prefetch, llama_mlocks * mlock_mmaps) {
    if (use_mmap) {
        mappings.reserve(files.size());
        mmaps_used.reserve(files.size());
        for (const auto & file : files) {
            auto * reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
            auto * is_numa_fn =
                (decltype(ggml_is_numa) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_is_numa");
            std::unique_ptr<llama_mmap> mapping =
                std::make_unique<llama_mmap>(file.get(), prefetch ? -1 : 0, is_numa_fn());
            mmaps_used.emplace_back(mapping->size(), 0);
            if (mlock_mmaps) {
                std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                mlock_mmap->init(mapping->addr());
                mlock_mmaps->emplace_back(std::move(mlock_mmap));
            }
            mappings.emplace_back(std::move(mapping));
        }
    }

    // compute the total size of all tensors for progress reporting
    for (const auto & it : weights_map) {
        size_data += ggml_nbytes(it.second.tensor);
    }
}

void llama_model_loader::get_mapping_range(size_t * first, size_t * last, void ** addr, int idx,
                                           ggml_context * ctx) const {
    GGML_ASSERT(!mappings.empty());
    const auto & mapping = mappings.at(idx);

    *first = mapping->size();
    *last  = 0;
    *addr  = mapping->addr();
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        const auto * const spliter = get_spliter(ggml_get_name(tensor));
        const std::string  name    = spliter->del_tp_id(ggml_get_name(tensor));
        const auto *       weight  = get_weight(name.c_str());
        if (!weight || weight->idx != idx) {
            continue;
        }
        *first = std::min(*first, weight->offs);
        *last  = std::max(*last, weight->offs + ggml_nbytes(tensor));
    }
}

bool llama_model_loader::load_all_data(struct ggml_context * ctx, const struct llama_hparams & hparams,
                                       llama_buf_map & bufs, llama_mlocks * lmlocks,
                                       llama_progress_callback progress_callback, void * progress_callback_user_data) {
    GGML_ASSERT(size_data != 0 && "call init_mappings() first");

    std::vector<no_init<uint8_t>>                            read_buf;
    std::vector<std::future<std::pair<ggml_tensor *, bool>>> validation_result;

    // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.
    // NVMe raid configurations might require more / larger buffers.
    constexpr size_t n_buffers   = 4;
    constexpr size_t buffer_size = 1 * 1024 * 1024;  // 1MB

    std::vector<ggml_backend_buffer_t> host_buffers;
    std::vector<ggml_backend_event_t>  events;
    std::vector<void *>                host_ptrs;
    size_t                             buffer_idx = 0;  // buffer to use for async loads

    bool use_async_upload = false;

    ggml_backend_t upload_backend = [&](const char * func) -> ggml_backend_t {
        if (use_mmap || check_tensors) {
            return nullptr;
        }
        // When not using mmaped io use async uploads from pinned memory to GPU memory.
        // First determine if the backend supports the necessary features for async uploads.
        auto * buf = bufs.count(0) ? bufs.at(0) : nullptr;
        if (!buf) {
            LLAMA_LOG_DEBUG("%s: no buffer found for async uploads\n", func);
            return nullptr;
        }

        auto * buft = ggml_backend_buffer_get_type(buf);
        auto * dev  = ggml_backend_buft_get_device(buft);
        if (!dev) {
            LLAMA_LOG_DEBUG("%s: no device found for buffer type %s for async uploads\n", func,
                            ggml_backend_buft_name(buft));
            return nullptr;
        }

        if (buft != ggml_backend_dev_buffer_type(dev)) {
            LLAMA_LOG_DEBUG("%s: buffer type %s is not the default buffer type for device %s for async uploads\n", func,
                            ggml_backend_buft_name(buft), ggml_backend_dev_name(dev));
            return nullptr;
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.host_buffer) {
            LLAMA_LOG_DEBUG("%s: device %s does not support host buffers\n", func, ggml_backend_dev_name(dev));
            return nullptr;
        }

        if (props.caps.async && props.caps.events) {
            use_async_upload = true;
        }

        auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!host_buft) {
            LLAMA_LOG_DEBUG("%s: no host buffer type found for device %s\n", func, ggml_backend_dev_name(dev));
            return nullptr;
        }

        // If the backend is supported, create pinned memory buffers and events for synchronisation.
        for (size_t idx = 0; idx < n_buffers; ++idx) {
            auto * buf = ggml_backend_buft_alloc_buffer(host_buft, buffer_size);
            if (!buf) {
                LLAMA_LOG_DEBUG("%s: failed to allocate host buffer for async uploads for device %s\n", func,
                                ggml_backend_dev_name(dev));
                return nullptr;
            }

            host_buffers.emplace_back(buf);
            host_ptrs.emplace_back(ggml_backend_buffer_get_base(buf));

            if (use_async_upload) {
                auto * event = ggml_backend_event_new(dev);
                if (!event) {
                    LLAMA_LOG_DEBUG("%s: failed to create event for async uploads for device %s\n", func,
                                    ggml_backend_dev_name(dev));
                    return nullptr;
                }

                events.emplace_back(event);
            }
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            LLAMA_LOG_DEBUG("%s: failed to initialize backend for device %s for async uploads\n", func,
                            ggml_backend_dev_name(dev));
            return nullptr;
        }

        return backend;
    }(__func__);

    if (upload_backend) {
        LLAMA_LOG_DEBUG("%s: using async uploads for device %s, buffer type %s, backend %s\n", __func__,
                        ggml_backend_dev_name(ggml_backend_get_device(upload_backend)),
                        ggml_backend_buft_name(ggml_backend_buffer_get_type(bufs.at(0))),
                        ggml_backend_name(upload_backend));
    }

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        const auto * const spliter = get_spliter(ggml_get_name(cur));
        const std::string  name    = spliter->del_tp_id(ggml_get_name(cur));
        const auto *       weight  = get_weight(name.c_str());
        if (weight == nullptr) {
            // this can happen with split experts models
            continue;
        }

        if (progress_callback) {
            if (!progress_callback((float) size_done / size_data, progress_callback_user_data)) {
                return false;
            }
        }

        size_t n_size = ggml_nbytes(cur);

        if (!use_mmap) {
            GGML_ASSERT(!spliter->post_process);
        }

        if (use_mmap) {
            const auto &          mapping  = mappings.at(weight->idx);
            ggml_backend_buffer_t buf_mmap = nullptr;
            if (bufs.count(weight->idx)) {
                buf_mmap = bufs.at(weight->idx);
            }
            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;

            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                    return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                }));
            }

            GGML_ASSERT(buf_mmap ||
                        cur->data);  // either we have a buffer to allocate the tensor in, or it is already allocated
            if (buf_mmap && cur->data == nullptr) {
                GGML_ASSERT(!spliter->post_process);
                ggml_backend_tensor_alloc(buf_mmap, cur, data);
                if (lmlocks) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + n_size);
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first, weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                if (spliter->post_process) {
                    uint8_t * post_process_data = new uint8_t[n_size];
                    spliter->post_process(data, post_process_data, cur->type, n_size, hparams);
                    ggml_backend_tensor_set(cur, post_process_data, 0, n_size);
                    delete[] post_process_data;
                } else {
                    ggml_backend_tensor_set(cur, data, 0, n_size);
                }
            }
        } else {
            const auto & file = files.at(weight->idx);
            if (ggml_backend_buffer_is_host(cur->buffer)) {
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(cur->data, n_size);
                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                        return std::make_pair(cur, ggml_validate_row_data(cur->type, cur->data, n_size));
                    }));
                }
            } else {
                // If upload_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the GPU.
                if (upload_backend) {
                    file->seek(weight->offs, SEEK_SET);

                    size_t bytes_read = 0;

                    while (bytes_read < n_size) {
                        size_t read_iteration = std::min<size_t>(buffer_size, n_size - bytes_read);

                        if (use_async_upload) {
                            ggml_backend_event_synchronize(events[buffer_idx]);
                        }
                        file->read_raw(host_ptrs[buffer_idx], read_iteration);

                        if (use_async_upload) {
                            ggml_backend_tensor_set_async(upload_backend, cur, host_ptrs[buffer_idx], bytes_read,
                                                          read_iteration);
                            ggml_backend_event_record(events[buffer_idx], upload_backend);
                        } else {
                            ggml_backend_tensor_set(cur, host_ptrs[buffer_idx], bytes_read, read_iteration);
                        }

                        bytes_read += read_iteration;
                        ++buffer_idx;
                        buffer_idx %= n_buffers;
                    }
                } else {
                    read_buf.resize(n_size);
                    file->seek(weight->offs, SEEK_SET);
                    file->read_raw(read_buf.data(), n_size);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                    if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                    }
                }
            }
        }

        size_done += n_size;
    }

    // free temporary resources used for async uploads
    for (auto * event : events) {
        ggml_backend_event_synchronize(event);
        ggml_backend_event_free(event);
    }
    for (auto * buf : host_buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_backend_free(upload_backend);

    // check validation results
    bool validation_failed = false;
    for (auto & future : validation_result) {
        auto result = future.get();
        if (!result.second) {
            LLAMA_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, ggml_get_name(result.first));
            validation_failed = true;
        }
    }
    if (validation_failed) {
        throw std::runtime_error("found tensors with invalid data");
    }

    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto &       mapping   = mappings.at(idx);
                mapping->unmap_fragment(0, mmap_used.first);
                if (mmap_used.second != 0) {
                    mapping->unmap_fragment(mmap_used.second, mapping->size());
                }
            }
        }
        if (progress_callback) {
            // Even though the model is done loading, we still honor
            // cancellation since we need to free allocations.
            return progress_callback(1.0f, progress_callback_user_data);
        }
    }

    return true;
}

#ifdef LLAMA_MPI_SUPPORT
bool llama_model_loader::load_all_data_mpi(struct ggml_context * ctx, const struct llama_hparams & hparams,
                                           llama_buf_map &         bufs, llama_mlocks * /* unused */,
                                           llama_progress_callback progress_callback,
                                           void *                  progress_callback_user_data) {
    GGML_ASSERT(size_data != 0 && "call init_mappings() first");

    std::vector<no_init<uint8_t>>                            read_buf;
    std::vector<std::future<std::pair<ggml_tensor *, bool>>> validation_result;

    // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.
    // NVMe raid configurations might require more / larger buffers.
    constexpr size_t n_buffers   = 4;
    constexpr size_t buffer_size = 1 * 1024 * 1024;  // 1MB

    std::vector<ggml_backend_buffer_t> host_buffers;
    std::vector<ggml_backend_event_t>  events;
    std::vector<void *>                host_ptrs;
    size_t                             buffer_idx = 0;  // buffer to use for async loads

    bool use_async_upload = false;

    ggml_backend_t upload_backend = [&](const char * func) -> ggml_backend_t {
        // if (use_mmap || check_tensors) {
        //     return nullptr;
        // }
        // When not using mmaped io use async uploads from pinned memory to GPU memory.
        // First determine if the backend supports the necessary features for async uploads.
        auto * buf = bufs.count(0) ? bufs.at(0) : nullptr;
        if (!buf) {
            LLAMA_LOG_DEBUG("%s: no buffer found for async uploads\n", func);
            return nullptr;
        }

        auto * buft = ggml_backend_buffer_get_type(buf);
        auto * dev  = ggml_backend_buft_get_device(buft);
        if (!dev) {
            LLAMA_LOG_DEBUG("%s: no device found for buffer type %s for async uploads\n", func,
                            ggml_backend_buft_name(buft));
            return nullptr;
        }

        if (buft != ggml_backend_dev_buffer_type(dev)) {
            LLAMA_LOG_DEBUG("%s: buffer type %s is not the default buffer type for device %s for async uploads\n", func,
                            ggml_backend_buft_name(buft), ggml_backend_dev_name(dev));
            return nullptr;
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.host_buffer) {
            LLAMA_LOG_DEBUG("%s: device %s does not support host buffers\n", func, ggml_backend_dev_name(dev));
            return nullptr;
        }

        if (props.caps.async && props.caps.events) {
            use_async_upload = true;
        }

        auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!host_buft) {
            LLAMA_LOG_DEBUG("%s: no host buffer type found for device %s\n", func, ggml_backend_dev_name(dev));
            return nullptr;
        }

        // If the backend is supported, create pinned memory buffers and events for synchronisation.
        for (size_t idx = 0; idx < n_buffers; ++idx) {
            auto * buf = ggml_backend_buft_alloc_buffer(host_buft, buffer_size);
            if (!buf) {
                LLAMA_LOG_DEBUG("%s: failed to allocate host buffer for async uploads for device %s\n", func,
                                ggml_backend_dev_name(dev));
                return nullptr;
            }

            host_buffers.emplace_back(buf);
            host_ptrs.emplace_back(ggml_backend_buffer_get_base(buf));

            if (use_async_upload) {
                auto * event = ggml_backend_event_new(dev);
                if (!event) {
                    LLAMA_LOG_DEBUG("%s: failed to create event for async uploads for device %s\n", func,
                                    ggml_backend_dev_name(dev));
                    return nullptr;
                }

                events.emplace_back(event);
            }
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            LLAMA_LOG_DEBUG("%s: failed to initialize backend for device %s for async uploads\n", func,
                            ggml_backend_dev_name(dev));
            return nullptr;
        }

        return backend;
    }(__func__);

    if (upload_backend) {
        LLAMA_LOG_DEBUG("%s: using async uploads for device %s, buffer type %s, backend %s\n", __func__,
                        ggml_backend_dev_name(ggml_backend_get_device(upload_backend)),
                        ggml_backend_buft_name(ggml_backend_buffer_get_type(bufs.at(0))),
                        ggml_backend_name(upload_backend));
    }

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        const auto * const spliter = get_spliter(ggml_get_name(cur));
        const std::string  name    = spliter->del_tp_id(ggml_get_name(cur));
        const auto *       weight  = get_weight(name.c_str());
        if (weight == nullptr) {
            // this can happen with split experts models
            continue;
        }

        if (progress_callback) {
            if (!progress_callback((float) size_done / size_data, progress_callback_user_data)) {
                return false;
            }
        }

        size_t n_size = ggml_nbytes(cur);

        if (!use_mmap) {
            LLAMA_LOG_ERROR("%s: mmap is required in MPI mode\n", __func__);
            return false;
        }

        const auto & mapping = mappings.at(weight->idx);
        uint8_t *    data    = (uint8_t *) mapping->addr() + weight->offs;

        if (check_tensors) {
            validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
            }));
        }

        GGML_ASSERT(cur->data);

        if (spliter->mode == LLAMA_REPEAT) {
            if (spliter->post_process) {
                uint8_t * post_process_data = new uint8_t[n_size];
                spliter->post_process(data, post_process_data, cur->type, n_size, hparams);
                ggml_backend_tensor_set(cur, post_process_data, 0, n_size);
                delete[] post_process_data;
            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
            size_done += n_size;
        } else {
            GGML_ASSERT(ggml_blck_size(cur->type) == 1);
            const int64_t element_size     = ggml_element_size(cur);
            const int64_t target_cont_size = spliter->get_target_cont_elements() * element_size;
            const int64_t source_cont_size = spliter->get_source_cont_elements() * element_size;
            const int64_t num_blocks       = spliter->get_num_blocks();
            GGML_ASSERT(num_blocks * target_cont_size == (int64_t) n_size);
            const int64_t source_offset_start = target_cont_size * spliter->tp_id;

            size_done += n_size * spliter->split_num;
            if (num_blocks == 1) {
                if (spliter->post_process) {
                    uint8_t * post_process_data = new uint8_t[n_size];
                    spliter->post_process(data + source_offset_start, post_process_data, cur->type, n_size, hparams);
                    ggml_backend_tensor_set(cur, post_process_data, 0, n_size);
                    delete[] post_process_data;
                } else {
                    ggml_backend_tensor_set(cur, data + source_offset_start, 0, n_size);
                }
                continue;
            }
            GGML_ASSERT(!spliter->post_process);

            if (target_cont_size >= (int64_t) buffer_size) {
                // read blocks directly from file
                for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
                    uint8_t * cur_data = data + source_cont_size * block_idx + source_offset_start;
                    ggml_backend_tensor_set(cur, cur_data, target_cont_size * block_idx, target_cont_size);
                }
            } else {
                // we need to split the tensor into multiple blocks
                if (ggml_backend_buffer_is_host(cur->buffer)) {
                    LLAMA_LOG_ERROR("%s: host buffer is not allowed for non-repeat tensors\n", __func__);
                    return false;
                }
                if (!upload_backend) {
                    for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
                        uint8_t * cur_data = data + source_cont_size * block_idx + source_offset_start;

                        ggml_backend_tensor_set(cur, cur_data, target_cont_size * block_idx, target_cont_size);
                    }
                    continue;
                }

                const int64_t blocks_per_buffer = buffer_size / target_cont_size;
                for (int block_id = 0; block_id < num_blocks; block_id += blocks_per_buffer) {
                    const size_t now_blocks_per_buffer = std::min<int64_t>(blocks_per_buffer, num_blocks - block_id);
                    const size_t buffer_read           = now_blocks_per_buffer * target_cont_size;
                    const size_t pre_read              = block_id * target_cont_size;
                    const size_t source_off            = source_offset_start + source_cont_size * block_id;

                    // sync before read
                    if (use_async_upload) {
                        ggml_backend_event_synchronize(events[buffer_idx]);
                    }

                    // read from file to buffer
                    for (size_t buf_off_id = 0; buf_off_id < now_blocks_per_buffer; buf_off_id++) {
                        memcpy((uint8_t *) host_ptrs[buffer_idx] + buf_off_id * target_cont_size,
                               data + source_off + buf_off_id * source_cont_size, (size_t) target_cont_size);
                    }
                    // transfer buffer to device
                    if (use_async_upload) {
                        ggml_backend_tensor_set_async(upload_backend, cur, host_ptrs[buffer_idx], pre_read,
                                                      buffer_read);
                        ggml_backend_event_record(events[buffer_idx], upload_backend);
                    } else {
                        ggml_backend_tensor_set(cur, host_ptrs[buffer_idx], pre_read, buffer_read);
                    }
                    ++buffer_idx;
                    buffer_idx %= n_buffers;
                }
            }
        }
    }

    // free temporary resources used for async uploads
    for (auto * event : events) {
        ggml_backend_event_synchronize(event);
        ggml_backend_event_free(event);
    }
    for (auto * buf : host_buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_backend_free(upload_backend);

    // check validation results
    bool validation_failed = false;
    for (auto & future : validation_result) {
        auto result = future.get();
        if (!result.second) {
            LLAMA_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, ggml_get_name(result.first));
            validation_failed = true;
        }
    }
    if (validation_failed) {
        throw std::runtime_error("found tensors with invalid data");
    }

    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto &       mapping   = mappings.at(idx);
                mapping->unmap_fragment(0, mmap_used.first);
                if (mmap_used.second != 0) {
                    mapping->unmap_fragment(mmap_used.second, mapping->size());
                }
            }
        }
        if (progress_callback) {
            // Even though the model is done loading, we still honor
            // cancellation since we need to free allocations.
            return progress_callback(1.0f, progress_callback_user_data);
        }
    }

    return true;
}
#endif

std::string llama_model_loader::ftype_name() const {
    return llama_model_ftype_name(ftype);
}

void llama_model_loader::print_info() const {
    LLAMA_LOG_INFO("%s: file format = %s\n", __func__, llama_file_version_name(fver));
    LLAMA_LOG_INFO("%s: file type   = %s\n", __func__, llama_model_ftype_name(ftype).c_str());
    if (n_bytes < GiB) {
        LLAMA_LOG_INFO("%s: file size   = %.2f MiB (%.2f BPW) \n", __func__, n_bytes / 1024.0 / 1024.0,
                       n_bytes * 8.0 / n_elements);
    } else {
        LLAMA_LOG_INFO("%s: file size   = %.2f GiB (%.2f BPW) \n", __func__, n_bytes / 1024.0 / 1024.0 / 1024.0,
                       n_bytes * 8.0 / n_elements);
    }
}

llama_model_loader::llama_tensor_viewer llama_model_loader::build_viewer(const std::vector<int64_t> & ne, int split_dim,
                                                                         int split_num, int tp_id) {
    llama_model_loader::llama_tensor_viewer viewer;
    int                                     idx = 0;
    for (auto dim_size : ne) {
        if (idx == split_dim) {
            GGML_ASSERT(dim_size % split_num == 0);
            viewer.ne[idx] = dim_size / split_num;
        } else {
            viewer.ne[idx] = dim_size;
        }
        idx++;
    }
    while (idx < GGML_MAX_DIMS) {
        viewer.ne[idx] = 1;
        idx++;
    }
    viewer.split_dim    = split_dim;
    viewer.split_num    = split_num;
    viewer.tp_id        = tp_id;
    viewer.mode         = LLAMA_AVG_SPLIT;
    viewer.post_process = nullptr;
    return viewer;
}

llama_model_loader::llama_tensor_viewer llama_model_loader::build_repeater(int tp_id) {
    llama_model_loader::llama_tensor_viewer viewer;
    viewer.mode         = LLAMA_REPEAT;
    viewer.split_dim    = -1;
    viewer.split_num    = -1;
    viewer.tp_id        = tp_id;
    viewer.post_process = nullptr;
    return viewer;
}
