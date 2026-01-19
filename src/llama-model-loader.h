#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <unordered_map>

#include "ggml-cpp.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama.h"

using llama_buf_map        = std::unordered_map<uint32_t, ggml_backend_buffer_t>;
using llama_load_post_proc = void (*)(void * src, void * dst, ggml_type type, size_t n_size,
                                      const llama_hparams & hparams);

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

struct llama_model_loader {
    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t idx;   // source file index
        size_t   offs;  // tensor data offset in the original file

        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx,
                            ggml_tensor * tensor) :
            idx(idx),
            tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx, ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(
                    format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete",
                           ggml_get_name(tensor)));
            }
        }
    };

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    // method to split weights
    enum llama_weight_splitter_mode {
        // repeat tensor
        LLAMA_REPEAT,
        // average split
        LLAMA_AVG_SPLIT,
        // specialized split method for merged qkv
        LLAMA_QKV_SPLIT,
    };

    struct llama_tensor_viewer {
        llama_weight_splitter_mode mode;
        // expected dimensions of the tensor
        int64_t                    ne[GGML_MAX_DIMS];
        // which dimension to split
        int                        split_dim;
        // number of splits
        int                        split_num;
        // tp id
        int                        tp_id;
        // buffer id (initialized only when loading)
        int                        buft_id;
        llama_load_post_proc       post_process;

        struct {
            int64_t q_size;
            int64_t k_size;
            int64_t v_size;
            int64_t q_offset_dst;
            int64_t k_offset_dst;
            int64_t v_offset_dst;
            int64_t q_offset_src;
            int64_t k_offset_src;
            int64_t v_offset_src;
            int64_t tot_size_src;
        } qkv_param;

        std::string add_tp_id(const std::string & name) const { return "tp." + std::to_string(tp_id) + "." + name; }

        std::string del_tp_id(const std::string & name) const {
            std::string tpid = "tp." + std::to_string(tp_id) + ".";
            // require name to start with tp_id
            GGML_ASSERT(name.substr(0, tpid.size()) == tpid);
            // return name without tp_id prefix
            return name.substr(tpid.size());
        }

        void set_buffer_id(int id) { buft_id = id; }

        // whether it can be the same tensor with different splits
        bool able_to_be_same_tensor(const llama_tensor_viewer & other) const {
            if (!(mode == other.mode && split_dim == other.split_dim && split_num == other.split_num)) {
                return false;
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (ne[i] != other.ne[i]) {
                    return false;
                }
            }
            return true;
        }

        // return the number of elements in the continuous segment of the target tensor
        int64_t get_target_cont_elements() const {
            int64_t target_cont_size = 1;
            for (int i = 0; i <= split_dim; ++i) {
                target_cont_size *= ne[i];
            }
            return target_cont_size;
        }

        int64_t get_source_cont_elements() const { return get_target_cont_elements() * split_num; }

        int64_t get_num_blocks() const {
            int64_t num_blocks = 1;
            for (int i = split_dim + 1; i < GGML_MAX_DIMS; ++i) {
                num_blocks *= ne[i];
            }
            return num_blocks;
        }
    };

    struct device_weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            int a_device = -1;
            int b_device = -1;
            sscanf(a.c_str(), "tp.%d.", &a_device);
            sscanf(b.c_str(), "tp.%d.", &b_device);
            if (a_device != b_device) {
                return a_device < b_device;
            }
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    static const int TENSOR_NOT_REQUIRED = 1;
    static const int TENSOR_DUPLICATED   = 2;

    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    bool use_mmap = false;
    bool check_tensors;

    llama_files files;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    std::map<std::string, struct llama_tensor_weight, weight_name_comparer>        weights_map;
    std::map<std::string, struct llama_tensor_viewer, device_weight_name_comparer> spliter_map;
    std::unordered_map<std::string, struct llama_model_kv_override>                kv_overrides;

    gguf_context_ptr              meta;
    std::vector<ggml_context_ptr> contexts;

    std::string arch_name;
    LLM_KV      llm_kv = LLM_KV(LLM_ARCH_UNKNOWN);

    size_t                                 size_done = 0;
    size_t                                 size_data = 0;
    std::vector<std::pair<size_t, size_t>> mmaps_used;

    llama_model_loader(
        const std::string &        fname,
        std::vector<std::string> & splits,  // optional, only need if the split does not follow naming scheme
        bool use_mmap, bool check_tensors, const struct llama_model_kv_override * param_overrides_p);

    template <typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type get_arr_n(const std::string & key, T & result,
                                                                              bool required = true);

    template <typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type get_arr_n(enum llm_kv kid, T & result,
                                                                              bool required = true);

    template <typename T> bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);

    template <typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required = true);

    template <typename T> bool get_arr(enum llm_kv kid, T & result, bool required = true);

    template <typename T> bool get_key(const std::string & key, T & result, bool required = true);

    template <typename T> bool get_key(enum llm_kv kid, T & result, bool required = true);

    template <typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required = true);

    template <typename T> bool get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required = true);

    std::string get_arch_name() const;

    enum llm_arch get_arch() const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_viewer * get_spliter(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    const struct ggml_tensor * check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne,
                                                 bool required, llama_tensor_viewer spliter) const;

    struct ggml_tensor * create_tensor(struct ggml_context * ctx, const std::string & name,
                                       const std::vector<int64_t> & ne, int flags = 0,
                                       llama_tensor_viewer spliter = {
                                           LLAMA_REPEAT,
                                           { 1, 1, 1 },
                                           -1,
                                           -1,
                                           -1,
                                           0,
                                           nullptr
    });

    void done_getting_tensors() const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr);

    void get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(struct ggml_context * ctx, const llama_hparams & hparams, llama_buf_map & bufs,
                       llama_mlocks * lmlocks, llama_progress_callback progress_callback,
                       void * progress_callback_user_data);

#ifdef LLAMA_MPI_SUPPORT
    bool load_all_data_mpi(struct ggml_context * ctx, const llama_hparams & hparams, llama_buf_map & bufs,
                           llama_mlocks * lmlocks, llama_progress_callback progress_callback,
                           void * progress_callback_user_data);
#endif
    std::string ftype_name() const;

    void print_info() const;

    static llama_tensor_viewer build_viewer(const std::vector<int64_t> & ne, int split_dim, int split_num, int tp_id);
    static llama_tensor_viewer build_repeater(int tp_id);
};
