#include "llama-model.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model-loader.h"
#include "llama.h"

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

struct llama_model::impl {
    impl() {}

    ~impl() {}

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored
    std::vector<ggml_context_ptr> ctxs;

    // the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_ptr> bufs;

    buft_list_t                               cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t *      buft_list;
    };

    layer_dev              dev_input  = {};
    layer_dev              dev_output = {};
    std::vector<layer_dev> dev_layer;
};

llama_model::llama_model(const struct llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {}

llama_model::~llama_model() {}

void llama_model::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes    = ml.n_bytes;
}

void llama_model::load_arch(llama_model_loader & ml) {
    arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
}

void llama_model::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

// CPU: ACCEL -> CPU extra -> GPU host -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add extra buffer types
    auto * cpu_dev                             = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * cpu_reg                             = ggml_backend_dev_backend_reg(cpu_dev);
    auto   ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        cpu_reg, "ggml_backend_dev_get_extra_bufts");
    if (ggml_backend_dev_get_extra_bufts_fn) {
        ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
        while (extra_bufts && *extra_bufts) {
            buft_list.emplace_back(cpu_dev, *extra_bufts);
            ++extra_bufts;
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    for (auto * dev : devices) {
        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
        if (buft) {
            buft_list.emplace_back(dev, buft);
            break;
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, enum llama_split_mode /* unused */,
                                      const float * /* unused */) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    // if (split_mode == LLAMA_SPLIT_MODE_ROW) {
    //     ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    //     auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
    //         ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
    //     if (ggml_backend_split_buffer_type_fn) {
    //         size_t dev_index = [&]() {
    //             auto * reg = ggml_backend_dev_backend_reg(dev);
    //             for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
    //                 if (ggml_backend_reg_dev_get(reg, i) == dev) {
    //                     return i;
    //                 }
    //             }
    //             throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
    //         }();
    //         auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
    //         if (buft != nullptr) {
    //             buft_list.emplace_back(dev, buft);
    //         }
    //     }
    // }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    return buft_list;
}

// checks if the weight tensor can be used with the specified buffer type and device
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op,
                                  ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    GGML_ASSERT(w != nullptr);

    if (op == GGML_OP_NONE) {
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr ctx_ptr{ ggml_init(params) };
    if (!ctx_ptr) {
        throw std::runtime_error(format("failed to create ggml context"));
    }
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * op_tensor = nullptr;

    switch (op) {
        case GGML_OP_GET_ROWS:
            {
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor       = ggml_get_rows(ctx, w, b);
            }
            break;
        case GGML_OP_MUL_MAT:
            {
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
                op_tensor       = ggml_mul_mat(ctx, w, b);
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            {
                int           n_expert_used = hparams.n_expert_used;
                ggml_tensor * b             = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * ids           = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor                   = ggml_mul_mat_id(ctx, w, b, ids);
            }
            break;
        case GGML_OP_ADD:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor       = ggml_add(ctx, a, w);
            }
            break;
        case GGML_OP_MUL:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor       = ggml_mul(ctx, a, w);
            }
            break;
        case GGML_OP_DIV:
            {
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w->ne[0]);
                op_tensor       = ggml_div(ctx, a, w);
            }
            break;
        case GGML_OP_ROPE:
            {
                int           n_embd_head = hparams.n_embd_head_v;
                int           n_head      = hparams.n_head();
                ggml_tensor * a           = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, 512);
                ggml_tensor * b           = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor                 = ggml_rope_ext(ctx, a, b, w, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            }
            break;
        case GGML_OP_SSM_CONV:
            {
                // FIXME
                ggml_tensor * conv_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 12345, w->ne[1], 6789);
                op_tensor            = ggml_ssm_conv(ctx, conv_x, w);
            }
            break;
        case GGML_OP_SSM_SCAN:
            {
                // FIXME
                const int64_t d_state      = w->ne[0];
                const int64_t d_inner      = w->ne[1];
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 1;
                ggml_tensor * s            = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, d_inner, n_seqs);
                ggml_tensor * x            = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_seq_tokens, n_seqs);
                ggml_tensor * dt           = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_seq_tokens, n_seqs);
                ggml_tensor * B            = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, n_seq_tokens, n_seqs);
                ggml_tensor * C            = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, n_seq_tokens, n_seqs);
                op_tensor                  = ggml_ssm_scan(ctx, s, x, dt, w, B, C);
            }
            break;
        case GGML_OP_RWKV_WKV6:
            {
                // FIXME
                const int64_t S        = 123;
                const int64_t H        = 123;
                const int64_t n_tokens = 123;
                const int64_t n_seqs   = 123;
                ggml_tensor * k        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor * v        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor * r        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor * tf       = w;
                ggml_tensor * td       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor * state    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_seqs, S, H);
                op_tensor              = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state);
            }
            break;
        case GGML_OP_IM2COL:
            {
                const int     n_embd = hparams.n_embd;
                ggml_tensor * b      = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_embd, w->ne[1], 1, 1);
                op_tensor            = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
            }
            break;
        default:
            GGML_ABORT("%s: missing test for op %s for tensor %s", __func__, ggml_op_name(op), w->name);
    }

    // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
    GGML_ASSERT(w->buffer == nullptr);
    w->buffer         = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;

    return op_supported;
}

// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op,
                                                     const buft_list_t & buft_list) {
    GGML_ASSERT(!buft_list.empty());
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t         cur_dev  = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
            return cur_buft;
        }
    }
    return nullptr;
}

enum llm_split_method {
    LLM_SPLIT_REPEAT,
    LLM_SPLIT_3d_MERGE01,
    LLM_SPLIT_3d_MERGE12,
    LLM_SPLIT_2d_DIM0,
    LLM_SPLIT_2d_DIM1,
    LLM_SPLIT_3d_DIM2_MERGE12,
    LLM_SPLIT_3d_DIM1_MERGE12,
    LLM_SPLIT_3d_DIM1_MERGE01,
    LLM_SPLIT_3d_DIM0_MERGE01,
    LLM_SPLIT_3d_DIM2,
    LLM_SPLIT_3d_DIM1,
    LLM_SPLIT_3d_DIM0,
};

static std::vector<int64_t> bulid_target_ne(const std::vector<int64_t> & ori_ne, llm_split_method split_method,
                                            int split_num) {
    switch (split_method) {
        case LLM_SPLIT_REPEAT:
            return ori_ne;
        case LLM_SPLIT_3d_MERGE01:
            GGML_ASSERT(ori_ne.size() == 3);
            return { ori_ne[0] * ori_ne[1], ori_ne[2] };
        case LLM_SPLIT_3d_MERGE12:
            GGML_ASSERT(ori_ne.size() == 3);
            return { ori_ne[0], ori_ne[1] * ori_ne[2] };
        case LLM_SPLIT_2d_DIM0:
            GGML_ASSERT(ori_ne.size() == 2);
            return { ori_ne[0] / split_num, ori_ne[1] };
        case LLM_SPLIT_2d_DIM1:
            GGML_ASSERT(ori_ne.size() == 2);
            return { ori_ne[0], ori_ne[1] / split_num };
        case LLM_SPLIT_3d_DIM2_MERGE12:
            GGML_ASSERT(ori_ne.size() == 3);
            return { ori_ne[0], ori_ne[1] * (ori_ne[2] / split_num) };
        case LLM_SPLIT_3d_DIM1_MERGE12:
            GGML_ASSERT(ori_ne.size() == 3);
            return { ori_ne[0], (ori_ne[1] / split_num) * ori_ne[2] };
        case LLM_SPLIT_3d_DIM1_MERGE01:
            GGML_ASSERT(ori_ne.size() == 3);
            return { ori_ne[0] * (ori_ne[1] / split_num), ori_ne[2] };
        case LLM_SPLIT_3d_DIM0_MERGE01:
            GGML_ASSERT(ori_ne.size() == 3);
            return { (ori_ne[0] / split_num) * ori_ne[1], ori_ne[2] };
        case LLM_SPLIT_3d_DIM2:
            GGML_ASSERT(ori_ne.size() == 3);
            return { ori_ne[0], ori_ne[1], ori_ne[2] / split_num };
        case LLM_SPLIT_3d_DIM1:
            GGML_ASSERT(ori_ne.size() == 3);
            return { ori_ne[0], ori_ne[1] / split_num, ori_ne[2] };
        case LLM_SPLIT_3d_DIM0:
            GGML_ASSERT(ori_ne.size() == 3);
            return { ori_ne[0] / split_num, ori_ne[1], ori_ne[2] };
        default:
            GGML_ABORT("Invalid split method");
    }
}

static llama_model_loader::llama_tensor_viewer build_viewer(const std::vector<int64_t> & ori_ne,
                                                            llm_split_method split_method, int tp_id, int split_num,
                                                            llama_load_post_proc post_process = nullptr) {
    llama_model_loader::llama_tensor_viewer spliter;
    switch (split_method) {
        case LLM_SPLIT_REPEAT:
        case LLM_SPLIT_3d_MERGE01:
        case LLM_SPLIT_3d_MERGE12:
            spliter = llama_model_loader::build_repeater(tp_id);
            break;
        case LLM_SPLIT_2d_DIM0:
            GGML_ASSERT(ori_ne.size() == 2);
            spliter = llama_model_loader::build_viewer(ori_ne, 0, split_num, tp_id);
            break;
        case LLM_SPLIT_2d_DIM1:
            GGML_ASSERT(ori_ne.size() == 2);
            spliter = llama_model_loader::build_viewer(ori_ne, 1, split_num, tp_id);
            break;
        case LLM_SPLIT_3d_DIM0:
        case LLM_SPLIT_3d_DIM0_MERGE01:
            GGML_ASSERT(ori_ne.size() == 3);
            spliter = llama_model_loader::build_viewer(ori_ne, 0, split_num, tp_id);
            break;
        case LLM_SPLIT_3d_DIM1:
        case LLM_SPLIT_3d_DIM1_MERGE12:
        case LLM_SPLIT_3d_DIM1_MERGE01:
            GGML_ASSERT(ori_ne.size() == 3);
            spliter = llama_model_loader::build_viewer(ori_ne, 1, split_num, tp_id);
            break;
        case LLM_SPLIT_3d_DIM2:
        case LLM_SPLIT_3d_DIM2_MERGE12:
            GGML_ASSERT(ori_ne.size() == 3);
            spliter = llama_model_loader::build_viewer(ori_ne, 2, split_num, tp_id);
            break;
        default:
            GGML_ABORT("Invalid split method");
    }
    spliter.post_process = post_process;
    return spliter;
}

static void wkv_b_post_process(void * src, void * dst, ggml_type type, size_t n_size, const llama_hparams & hparams) {
    struct ggml_init_params params              = { /* .mem_size */ ggml_tensor_overhead() * 10,  // 分配16MB内存
                                       /* .mem_buffer */ NULL,
                                       /* .no_alloc */ true };
    const int               n_head              = (hparams.enable_tensor_parallel & !hparams.enable_data_parallel) ?
                                                      hparams.n_head() / hparams.num_parallel :
                                                      hparams.n_head();
    const int32_t           n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
    const int32_t           kv_lora_rank        = hparams.n_lora_kv;
    const int32_t           n_embd_head_v       = hparams.n_embd_head_v;
    ggml_context *          ctx                 = ggml_init(params);
    struct ggml_tensor *    wk_b_src = ggml_new_tensor_3d(ctx, type, n_embd_head_qk_nope, kv_lora_rank, n_head);
    wk_b_src->nb[0]                  = ggml_row_size(type, kv_lora_rank);
    wk_b_src->nb[1]                  = ggml_type_size(type);
    wk_b_src->nb[2]                  = ggml_row_size(type, kv_lora_rank * (n_embd_head_qk_nope + n_embd_head_v));
    wk_b_src->data                   = src;

    struct ggml_tensor * wv_b_src = ggml_new_tensor_3d(ctx, type, kv_lora_rank, n_embd_head_v, n_head);
    wv_b_src->nb[1]               = ggml_row_size(type, kv_lora_rank);
    wv_b_src->nb[2]               = ggml_row_size(type, kv_lora_rank * (n_embd_head_qk_nope + n_embd_head_v));
    wv_b_src->data                = (uint8_t *) src + ggml_row_size(type, kv_lora_rank * n_embd_head_qk_nope);

    struct ggml_tensor * wk_b_dst = ggml_new_tensor_3d(ctx, type, kv_lora_rank, n_embd_head_qk_nope, n_head);
    wk_b_dst->nb[1]               = ggml_row_size(type, kv_lora_rank);
    wk_b_dst->nb[2]               = ggml_row_size(type, kv_lora_rank * n_embd_head_qk_nope);
    wk_b_dst->data                = dst;
    GGML_ASSERT(ggml_is_contiguous(wk_b_dst));

    struct ggml_tensor * wv_b_dst = ggml_new_tensor_3d(ctx, type, kv_lora_rank, n_embd_head_v, n_head);
    wv_b_dst->nb[1]               = ggml_row_size(type, kv_lora_rank);
    wv_b_dst->nb[2]               = ggml_row_size(type, kv_lora_rank * n_embd_head_v);
    wv_b_dst->data                = (uint8_t *) dst + ggml_row_size(type, kv_lora_rank * n_embd_head_qk_nope * n_head);
    GGML_ASSERT(ggml_is_contiguous(wv_b_dst));

    wk_b_dst->op     = GGML_OP_CPY;
    wk_b_dst->src[0] = wk_b_src;
    wv_b_dst->op     = GGML_OP_CPY;
    wv_b_dst->src[0] = wv_b_src;

    GGML_ASSERT(ggml_nbytes(wk_b_dst) + ggml_nbytes(wv_b_dst) == n_size);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 10, false);
    ggml_build_forward_expand(gf, wk_b_dst);
    ggml_build_forward_expand(gf, wv_b_dst);
    struct ggml_cplan cplan  = ggml_graph_plan(gf, 1, NULL);
    ggml_status       status = ggml_graph_compute(gf, &cplan);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("wkv_b_post_process failed with status : %d\n", status);
        GGML_ABORT("wkv_b_post_process failed");
    }
    ggml_free(ctx);
}

bool llama_model::load_tensors(llama_model_loader & ml) {
    const auto & split_mode             = params.split_mode;
    const auto & n_gpu_layers           = params.n_gpu_layers;
    const auto & use_mlock              = params.use_mlock;
    const auto & tensor_split           = params.tensor_split;
    const bool & enable_tensor_parallel = params.enable_tensor_parallel;
    if (enable_tensor_parallel) {
        if (hparams.num_parallel != 8 && hparams.num_parallel != 4 && hparams.num_parallel != 2 &&
            hparams.num_parallel != 1) {
            GGML_ABORT(
                "Tensor parallelism enabled, only 8 or 4 or 2 or 1 devices are supported, but %ld devices are "
                "provided\n",
                n_devices());
        }
        LLAMA_LOG_INFO("Tensor parallelism enabled.\n");
    }
    GGML_ASSERT(0 <= hparams.tp_id && hparams.tp_id < (int) hparams.num_parallel);

    const int n_layer = hparams.n_layer;

    const bool use_mmap_buffer = true;

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (mmap = %s)\n", __func__,
                   ml.use_mmap ? "true" : "false");

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices);
    for (auto * dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev, std::move(buft_list));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr ||
                    std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i];
            size_t             total;
            size_t             free;
            ggml_backend_dev_memory(dev, &free, &total);
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] /= split_sum;
    }

    ggml_backend_dev_t cpu_dev        = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    const int          i_gpu_start    = std::max((int) hparams.n_layer - n_gpu_layers, 0);
    const int          act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, (int) n_layer + 1);
    if (enable_tensor_parallel && act_gpu_layers < n_layer + 1) {
        GGML_ABORT("Tensor parallelism not supported: act_gpu_layers (%d) < n_layer + 1 (%d)", act_gpu_layers,
                   n_layer + 1);
    }
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s\n", il, ggml_backend_dev_name(cpu_dev));
            return { cpu_dev, &pimpl->cpu_buft_list };
        }
        const int layer_gpu =
            std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start) / act_gpu_layers) -
            splits.begin();
        auto * dev = devices.at(layer_gpu);
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s\n", il, ggml_backend_dev_name(dev));
        return { dev, &pimpl->gpu_buft_list.at(dev) };
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    if (params.offload_input) {
        pimpl->dev_input = get_layer_buft_list(0);
    } else {
        pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };
    }

    // assign the repeating layers to the devices according to the splits
    if (!enable_tensor_parallel) {
        pimpl->dev_layer.resize(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            pimpl->dev_layer[il] = get_layer_buft_list(il);
        }
    }
    // assign the output layer
    if (enable_tensor_parallel) {
        auto * dev        = devices.at(hparams.tp_id);
        pimpl->dev_output = { dev, &pimpl->gpu_buft_list.at(dev) };
    } else {
        pimpl->dev_output = get_layer_buft_list(n_layer);
    }

    // one ggml context per buffer type
    int max_n_tensors = ml.n_tensors;
    max_n_tensors += 1;            // duplicated output tensor
    max_n_tensors += n_layer * 2;  // duplicated rope freq tensors
    const size_t ctx_size = ggml_tensor_overhead() * max_n_tensors;

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ctx_size,
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error(format("failed to create ggml context"));
            }

            ctx_map[buft] = ctx;
            pimpl->ctxs.emplace_back(ctx);

            return ctx;
        }
        return it->second;
    };

    const auto TENSOR_DUPLICATED   = llama_model_loader::TENSOR_DUPLICATED;
    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;

    // create tensors for the weights
    {
        // note: cast to int64_t since we will use these for the tensor dimensions
        const int64_t n_head        = hparams.n_head();
        // const int64_t n_head_kv     = hparams.n_head_kv();
        const int64_t n_embd        = hparams.n_embd;
        const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa();
        const int64_t n_embd_head_k = hparams.n_embd_head_k;
        const int64_t n_embd_head_v = hparams.n_embd_head_v;
        const int64_t n_ff          = hparams.n_ff();
        const int64_t n_embd_gqa    = n_embd_v_gqa;
        const int64_t n_vocab       = vocab.n_tokens();
        // const int64_t n_token_types = vocab.n_token_types();
        // const int64_t n_rot         = hparams.n_rot;
        const int64_t n_expert      = hparams.n_expert;
        const int64_t n_expert_used = hparams.n_expert_used;
        // const int64_t n_ctx_train   = hparams.n_ctx_train;

        if (n_expert > 0 && hparams.n_expert_used == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        int                        n_moved_tensors       = 0;
        ggml_tensor *              first_moved_tensor    = nullptr;
        ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
        ggml_backend_buffer_type_t first_moved_to_buft   = nullptr;

        auto create_tensor = [&](const std::initializer_list<int64_t> & ori_ne, llm_split_method split_method,
                                 const LLM_TN_IMPL & tn, int flags, ggml_backend_dev_t dev = nullptr,
                                 llama_load_post_proc post_process = nullptr) -> ggml_tensor * {
            ggml_tensor * t_meta = ml.get_tensor_meta(tn.str().c_str());

            if (!t_meta) {
                if (flags & TENSOR_NOT_REQUIRED) {
                    return nullptr;
                }
                throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
            }

            // some models use the token embedding tensor as the output, but since these are used in different layers and with different ops
            // the tensor is duplicated
            // to handle this, we check if the tensor is duplicated, and if so, we assume that it is being loaded as the output tensor
            llm_tensor tn_tensor = tn.tensor;
            if (tn.tensor == LLM_TENSOR_TOKEN_EMBD && flags & TENSOR_DUPLICATED) {
                tn_tensor = LLM_TENSOR_OUTPUT;
            }

            llm_tensor_info info;
            try {
                info = llm_tensor_info_for(tn_tensor);
            } catch (const std::out_of_range & e) {
                throw std::runtime_error(format("missing tensor info mapping for %s", tn.str().c_str()));
            }

            // skip unused tensors
            if (info.op == GGML_OP_NONE) {
                LLAMA_LOG_WARN("model has unused tensor %s -- ignoring\n", tn.str().c_str());
                ml.n_created++;

                return nullptr;
            }

            // tensors with "bias" suffix are always used with GGML_OP_ADD
            ggml_op op;
            bool    bias = tn.suffix != nullptr && strcmp(tn.suffix, "bias") == 0;
            if (bias) {
                op = GGML_OP_ADD;
            } else {
                op = info.op;
            }

            // sanity checks
            if (info.layer == LLM_TENSOR_LAYER_INPUT || info.layer == LLM_TENSOR_LAYER_OUTPUT) {
                if (tn.bid != -1) {
                    GGML_ABORT("input/output layer tensor %s used with a layer number", tn.str().c_str());
                }
            } else {
                if (tn.bid == -1) {
                    GGML_ABORT("repeating layer tensor %s used without a layer number", tn.str().c_str());
                }
            }

            // select the buffer type for this tensor
            buft_list_t * buft_list;
            switch (info.layer) {
                case LLM_TENSOR_LAYER_INPUT:
                    buft_list = pimpl->dev_input.buft_list;
                    break;
                case LLM_TENSOR_LAYER_OUTPUT:
                    buft_list = pimpl->dev_output.buft_list;
                    break;
                case LLM_TENSOR_LAYER_REPEATING:
                    if (enable_tensor_parallel) {
                        GGML_ASSERT(dev != nullptr);
                        buft_list = &pimpl->gpu_buft_list.at(dev);
                    } else {
                        buft_list = pimpl->dev_layer.at(tn.bid).buft_list;
                    }
                    break;
                default:
                    GGML_ABORT("invalid layer %d for tensor %s", info.layer, tn.str().c_str());
            }

            ggml_backend_buffer_type_t buft = select_weight_buft(hparams, t_meta, op, *buft_list);
            if (!buft) {
                throw std::runtime_error(
                    format("failed to find a compatible buffer type for tensor %s", tn.str().c_str()));
            }

            // avoid using a host buffer when using mmap
            auto * buft_dev = ggml_backend_buft_get_device(buft);
            if (ml.use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                buft           = ggml_backend_dev_buffer_type(cpu_dev);
            }

            if (buft != buft_list->front().second) {
                n_moved_tensors++;
                if (!first_moved_tensor) {
                    first_moved_tensor    = t_meta;
                    first_moved_from_buft = buft_list->front().second;
                    first_moved_to_buft   = buft;
                }
            }

            ggml_context * ctx = ctx_for_buft(buft);

            // if duplicated, check if the original tensor was allocated in the same buffer type context and avoid creating a new one
            if (flags & TENSOR_DUPLICATED) {
                ggml_tensor * t = ggml_get_tensor(ctx, tn.str().c_str());
                if (t) {
                    return t;
                }
            }
            std::vector<int64_t> ne(bulid_target_ne(ori_ne, split_method, hparams.num_parallel));
            return ml.create_tensor(
                ctx, tn, ne, flags,
                build_viewer(ori_ne, split_method, hparams.tp_id, hparams.num_parallel, post_process));
        };

        const int  parallel_size = enable_tensor_parallel ? hparams.num_parallel : 1;
        const bool use_dp        = hparams.enable_data_parallel;
        layers.resize(n_layer);

        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);
        switch (arch) {
            case LLM_ARCH_DEEPSEEK2:
                {
                    const bool is_lite = (hparams.n_layer == 27);

                    const int64_t n_embd_head_qk_rope = hparams.n_rot;
                    const int64_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;

                    const int64_t q_lora_rank  = hparams.n_lora_q;
                    const int64_t kv_lora_rank = hparams.n_lora_kv;

                    const int64_t n_ff_exp        = hparams.n_ff_exp;
                    const int64_t n_expert_shared = hparams.n_expert_shared;

                    GGML_ASSERT(n_head % parallel_size == 0);
                    GGML_ASSERT(n_ff % parallel_size == 0);
                    GGML_ASSERT(n_ff_exp % parallel_size == 0);

                    tok_embd =
                        create_tensor({ n_embd, n_vocab }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), 0);

                    // output
                    output_norm = create_tensor({ n_embd }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), 0);
                    output = create_tensor({ n_embd, n_vocab }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_OUTPUT, "weight"), 0);

                    for (int i = 0; i < n_layer; ++i) {
                        const int          p         = hparams.tp_id;
                        auto &             layer     = layers[i];
                        ggml_backend_dev_t local_dev = enable_tensor_parallel ? devices[p] : nullptr;

                        layer.attn_norm = create_tensor({ n_embd }, LLM_SPLIT_REPEAT,
                                                        tn(LLM_TENSOR_ATTN_NORM, "weight", i), 0, local_dev);
                        if (!is_lite) {
                            layer.attn_q_a_norm =
                                create_tensor({ q_lora_rank }, LLM_SPLIT_REPEAT,
                                              tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), 0, local_dev);
                        }

                        layer.attn_kv_a_norm = create_tensor({ kv_lora_rank }, LLM_SPLIT_REPEAT,
                                                             tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), 0, local_dev);

                        if (!is_lite) {
                            layer.wq_a = create_tensor({ n_embd, q_lora_rank }, LLM_SPLIT_REPEAT,
                                                       tn(LLM_TENSOR_ATTN_Q_A, "weight", i), 0, local_dev);
                            layer.wq_b = create_tensor({ q_lora_rank, n_embd_head_k, n_head },
                                                       use_dp ? LLM_SPLIT_3d_MERGE12 : LLM_SPLIT_3d_DIM2_MERGE12,
                                                       tn(LLM_TENSOR_ATTN_Q_B, "weight", i), 0, local_dev);
                        } else {
                            layer.wq = create_tensor({ n_embd, n_embd_head_k * n_head },
                                                     use_dp ? LLM_SPLIT_REPEAT : LLM_SPLIT_2d_DIM1,
                                                     tn(LLM_TENSOR_ATTN_Q, "weight", i), 0, local_dev);
                        }

                        layer.wkv_a_mqa =
                            create_tensor({ n_embd, kv_lora_rank + (n_embd_head_qk_rope) }, LLM_SPLIT_REPEAT,
                                          tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), 0, local_dev);
                        layer.wkv_b = create_tensor({ kv_lora_rank, n_embd_head_qk_nope + n_embd_head_v, n_head },
                                                    use_dp ? LLM_SPLIT_3d_MERGE12 : LLM_SPLIT_3d_DIM2_MERGE12,
                                                    tn(LLM_TENSOR_ATTN_KV_B, "weight", i), 0, local_dev,
                                                    hparams.enable_mla ? wkv_b_post_process : nullptr);
                        layer.wo    = create_tensor({ n_embd_head_v, n_head, n_embd },
                                                 use_dp ? LLM_SPLIT_3d_MERGE01 : LLM_SPLIT_3d_DIM1_MERGE01,
                                                    tn(LLM_TENSOR_ATTN_OUT, "weight", i), 0, local_dev);

                        layer.ffn_norm = create_tensor({ n_embd }, LLM_SPLIT_REPEAT,
                                                       tn(LLM_TENSOR_FFN_NORM, "weight", i), 0, local_dev);

                        if (i < (int) hparams.n_layer_dense_lead) {
                            layer.ffn_gate = create_tensor({ n_embd, n_ff }, LLM_SPLIT_2d_DIM1,
                                                           tn(LLM_TENSOR_FFN_GATE, "weight", i), 0, local_dev);
                            layer.ffn_down = create_tensor({ n_ff, n_embd }, LLM_SPLIT_2d_DIM0,
                                                           tn(LLM_TENSOR_FFN_DOWN, "weight", i), 0, local_dev);
                            layer.ffn_up   = create_tensor({ n_embd, n_ff }, LLM_SPLIT_2d_DIM1,
                                                           tn(LLM_TENSOR_FFN_UP, "weight", i), 0, local_dev);
                        } else {
                            layer.ffn_gate_inp = create_tensor({ n_embd, n_expert }, LLM_SPLIT_REPEAT,
                                                               tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), 0, local_dev);
                            layer.ffn_exp_probs_b =
                                create_tensor({ n_expert }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i),
                                              TENSOR_NOT_REQUIRED, local_dev);

                            if (n_expert == 0) {
                                throw std::runtime_error("n_expert must be > 0");
                            }
                            if (n_expert_used == 0) {
                                throw std::runtime_error("n_expert_used must be > 0");
                            }

                            // MoE branch
                            if (hparams.enable_expert_parallel) {
                                layer.ffn_gate_exps =
                                    create_tensor({ n_embd, n_ff_exp, n_expert }, LLM_SPLIT_3d_DIM2,
                                                  tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), 0, local_dev);
                                layer.ffn_down_exps =
                                    create_tensor({ n_ff_exp, n_embd, n_expert }, LLM_SPLIT_3d_DIM2,
                                                  tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), 0, local_dev);
                                layer.ffn_up_exps =
                                    create_tensor({ n_embd, n_ff_exp, n_expert }, LLM_SPLIT_3d_DIM2,
                                                  tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), 0, local_dev);
                            } else {
                                layer.ffn_gate_exps =
                                    create_tensor({ n_embd, n_ff_exp, n_expert }, LLM_SPLIT_3d_DIM1,
                                                  tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), 0, local_dev);
                                layer.ffn_down_exps =
                                    create_tensor({ n_ff_exp, n_embd, n_expert }, LLM_SPLIT_3d_DIM0,
                                                  tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), 0, local_dev);
                                layer.ffn_up_exps =
                                    create_tensor({ n_embd, n_ff_exp, n_expert }, LLM_SPLIT_3d_DIM1,
                                                  tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), 0, local_dev);
                            }

                            // Shared expert branch
                            layer.ffn_gate_shexp =
                                create_tensor({ n_embd, n_ff_exp, n_expert_shared }, LLM_SPLIT_3d_DIM1_MERGE12,
                                              tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), 0, local_dev);
                            layer.ffn_down_shexp =
                                create_tensor({ n_ff_exp, n_expert_shared, n_embd }, LLM_SPLIT_3d_DIM0_MERGE01,
                                              tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), 0, local_dev);
                            layer.ffn_up_shexp =
                                create_tensor({ n_embd, n_ff_exp, n_expert_shared }, LLM_SPLIT_3d_DIM1_MERGE12,
                                              tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), 0, local_dev);
                        }
                    }
                }
                break;
            case LLM_ARCH_QWEN:
                {
                    tok_embd =
                        create_tensor({ n_embd, n_vocab }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), 0);

                    // output
                    output_norm = create_tensor({ n_embd }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), 0);
                    output = create_tensor({ n_embd, n_vocab }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_OUTPUT, "weight"), 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm =
                            create_tensor({ n_embd }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_ATTN_NORM, "weight", i), 0);

                        layer.wqkv = create_tensor({ n_embd, n_embd * 3 }, LLM_SPLIT_REPEAT,
                                                   tn(LLM_TENSOR_ATTN_QKV, "weight", i), 0);
                        layer.bqkv =
                            create_tensor({ n_embd * 3 }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_ATTN_QKV, "bias", i), 0);
                        layer.wo = create_tensor({ n_embd, n_embd }, LLM_SPLIT_REPEAT,
                                                 tn(LLM_TENSOR_ATTN_OUT, "weight", i), 0);

                        layer.ffn_norm =
                            create_tensor({ n_embd }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_FFN_NORM, "weight", i), 0);

                        layer.ffn_gate = create_tensor({ n_embd, n_ff / 2 }, LLM_SPLIT_REPEAT,
                                                       tn(LLM_TENSOR_FFN_GATE, "weight", i), 0);
                        layer.ffn_down = create_tensor({ n_ff / 2, n_embd }, LLM_SPLIT_REPEAT,
                                                       tn(LLM_TENSOR_FFN_DOWN, "weight", i), 0);
                        layer.ffn_up   = create_tensor({ n_embd, n_ff / 2 }, LLM_SPLIT_REPEAT,
                                                       tn(LLM_TENSOR_FFN_UP, "weight", i), 0);
                    }
                }
                break;
            case LLM_ARCH_QWEN2:
            case LLM_ARCH_QWEN2VL:
                {
                    tok_embd =
                        create_tensor({ n_embd, n_vocab }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), 0);

                    // output
                    output_norm = create_tensor({ n_embd }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), 0);
                    output      = create_tensor({ n_embd, n_vocab }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_OUTPUT, "weight"),
                                                TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor({ n_embd, n_vocab }, LLM_SPLIT_REPEAT,
                                               tn(LLM_TENSOR_TOKEN_EMBD, "weight"), TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm =
                            create_tensor({ n_embd }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_ATTN_NORM, "weight", i), 0);
                        layer.wq =
                            create_tensor({ n_embd, n_embd }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_ATTN_Q, "weight", i), 0);
                        layer.wk = create_tensor({ n_embd, n_embd_gqa }, LLM_SPLIT_REPEAT,
                                                 tn(LLM_TENSOR_ATTN_K, "weight", i), 0);
                        layer.wv = create_tensor({ n_embd, n_embd_gqa }, LLM_SPLIT_REPEAT,
                                                 tn(LLM_TENSOR_ATTN_V, "weight", i), 0);
                        layer.wo = create_tensor({ n_embd, n_embd }, LLM_SPLIT_REPEAT,
                                                 tn(LLM_TENSOR_ATTN_OUT, "weight", i), 0);

                        // optional bias tensors
                        layer.bq = create_tensor({ n_embd }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_ATTN_Q, "bias", i), 0);
                        layer.bk = create_tensor({ n_embd_gqa }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_ATTN_K, "bias", i), 0);
                        layer.bv = create_tensor({ n_embd_gqa }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_ATTN_V, "bias", i), 0);

                        layer.ffn_norm =
                            create_tensor({ n_embd }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_FFN_NORM, "weight", i), 0);

                        layer.ffn_gate =
                            create_tensor({ n_embd, n_ff }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_FFN_GATE, "weight", i), 0);
                        layer.ffn_down =
                            create_tensor({ n_ff, n_embd }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_FFN_DOWN, "weight", i), 0);
                        layer.ffn_up =
                            create_tensor({ n_embd, n_ff }, LLM_SPLIT_REPEAT, tn(LLM_TENSOR_FFN_UP, "weight", i), 0);
                    }
                }
                break;
            default:
                throw std::runtime_error("unknown architecture");
        }
        if (n_moved_tensors > 0) {
            LLAMA_LOG_DEBUG(
                "%s: tensor '%s' (%s) (and %d others) cannot be used with preferred buffer type %s, using %s instead\n",
                __func__, first_moved_tensor->name, ggml_type_name(first_moved_tensor->type), n_moved_tensors - 1,
                ggml_backend_buft_name(first_moved_from_buft), ggml_backend_buft_name(first_moved_to_buft));
        }
    }
    ml.done_getting_tensors();

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_bufs;
    ctx_bufs.reserve(ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ctx_map.size() * ml.files.size();
    pimpl->bufs.reserve(n_max_backend_buffer);

    for (auto & it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context *             ctx  = it.second;

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        llama_buf_map buf_map;
        buf_map.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft                = buft == ggml_backend_dev_buffer_type(dev);

        // this optimization is not used when tensor parallelism is enabled, because it is not easy to determine the start and end position of each tensor.
        if (!enable_tensor_parallel && ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported &&
            is_default_buft) {
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer, then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void * addr = nullptr;
                size_t first, last;  // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t          max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer_t buf =
                    ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                pimpl->bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        } else {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            pimpl->bufs.emplace_back(buf);
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init(ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }

        if (pimpl->bufs.empty()) {
            throw std::runtime_error("failed to allocate buffer");
        }

        for (auto & buf : buf_map) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.second, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        ctx_bufs.emplace_back(ctx, buf_map);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int) hparams.n_layer) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
        }

        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers       = hparams.n_layer + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers),
                       max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & buf : pimpl->bufs) {
        LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()),
                       ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
    }

    // populate tensors_by_name
    for (auto & ctx : pimpl->ctxs) {
        for (auto * cur = ggml_get_first_tensor(ctx.get()); cur != NULL; cur = ggml_get_next_tensor(ctx.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    // load tensor data
    std::vector<ggml_context *>  ctxs;
    std::vector<llama_buf_map *> bufs_array;

    if (enable_tensor_parallel) {
        GGML_ASSERT(hparams.enable_mpi && "mpi must be enabled when enable_tensor_parallel is true");
#ifdef LLAMA_MPI_SUPPORT
        for (auto & it : ctx_bufs) {
            ggml_context * ctx  = it.first;
            auto &         bufs = it.second;
            if (!ml.load_all_data_mpi(ctx, hparams, bufs, use_mlock ? &pimpl->mlock_mmaps : NULL,
                                      params.progress_callback, params.progress_callback_user_data)) {
                return false;
            }
        }
#endif
    } else {
        for (auto & it : ctx_bufs) {
            ggml_context * ctx  = it.first;
            auto &         bufs = it.second;
            if (!ml.load_all_data(ctx, hparams, bufs, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback,
                                  params.progress_callback_user_data)) {
                return false;
            }
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    return true;
}

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,     "none"     },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,   "linear"   },
    { LLAMA_ROPE_SCALING_TYPE_YARN,     "yarn"     },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE, "longrope" },
};

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

void llama_model::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.meta.get();

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char *      name  = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH, hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH, hparams.n_embd);
    ml.get_key(LLM_KV_BLOCK_COUNT, hparams.n_layer);
    ml.get_key(LLM_KV_EXPERT_COUNT, hparams.n_expert, false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used, false);

    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        ml.get_key(LLM_KV_FEATURES_LENGTH, hparams.n_embd_features);

        ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LLM_KV_POSNET_BLOCK_COUNT, hparams.posnet.n_layer);

        ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT, hparams.convnext.n_layer);
    }

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
    }

    // zero-out the array hparams
    std::fill(hparams.n_head_arr.begin(), hparams.n_head_arr.end(), 0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(), hparams.n_ff_arr.end(), 0);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH, hparams.n_ff_arr, hparams.n_layer, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f / ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k, false);

        hparams.n_embd_head_v = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v, false);

        // sanity check for n_rot (optional)
        hparams.n_rot = hparams.n_embd_head_k;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);

        if (arch == LLM_ARCH_LLAMA || arch == LLM_ARCH_DECI || arch == LLM_ARCH_FALCON) {
            if (hparams.n_rot != hparams.n_embd_head_k) {
                throw std::runtime_error(
                    format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd_head_k));
            }
        }
    } else {
        hparams.n_rot         = 0;
        hparams.n_embd_head_k = 0;
        hparams.n_embd_head_v = 0;
    }

    // for differentiating model types
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

    switch (arch) {
        case LLM_ARCH_DEEPSEEK:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);

                switch (hparams.n_layer) {
                    case 28:
                        type = LLM_TYPE_20B;
                        break;
                    default:
                        type = LLM_TYPE_UNKNOWN;
                }
            }
            break;
        case LLM_ARCH_DEEPSEEK2:
            {
                bool is_lite = (hparams.n_layer == 27);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
                if (!is_lite) {
                    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
                }
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
                    // for compatibility with existing DeepSeek V2 and V2.5 GGUFs
                    // that have no expert_gating_func model parameter set
                    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
                }
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul);

                switch (hparams.n_layer) {
                    case 27:
                        type = LLM_TYPE_16B;
                        break;
                    case 60:
                        type = LLM_TYPE_236B;
                        break;
                    case 61:
                        type = LLM_TYPE_671B;
                        break;
                    default:
                        type = LLM_TYPE_UNKNOWN;
                }
            }
            break;

        case LLM_ARCH_QWEN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32:
                        type = LLM_TYPE_7B;
                        break;
                    case 40:
                        type = LLM_TYPE_13B;
                        break;
                    default:
                        type = LLM_TYPE_UNKNOWN;
                }
            }
            break;
        case LLM_ARCH_QWEN2VL:
            {
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
            }
            // fall through
        case LLM_ARCH_QWEN2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24:
                        type = hparams.n_embd == 1024 ? LLM_TYPE_0_5B : LLM_TYPE_1B;
                        break;
                    case 28:
                        type = hparams.n_embd == 1536 ? LLM_TYPE_1_5B : LLM_TYPE_7B;
                        break;
                    case 32:
                        type = LLM_TYPE_7B;
                        break;
                    case 36:
                        type = LLM_TYPE_3B;
                        break;
                    case 40:
                        type = hparams.n_head() == 20 ? LLM_TYPE_4B : LLM_TYPE_13B;
                        break;
                    case 48:
                        type = LLM_TYPE_14B;
                        break;
                    case 64:
                        type = LLM_TYPE_32B;
                        break;
                    case 80:
                        type = LLM_TYPE_70B;
                        break;
                    default:
                        type = LLM_TYPE_UNKNOWN;
                }
            }
            break;
        case LLM_ARCH_QWEN2MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24:
                        type = LLM_TYPE_A2_7B;
                        break;
                    case 28:
                        type = LLM_TYPE_57B_A14B;
                        break;
                    default:
                        type = LLM_TYPE_UNKNOWN;
                }
            }
            break;
        default:
            throw std::runtime_error("unsupported model architecture");
    }

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_model_rope_type(this);

    if (params.enable_tensor_parallel) {
        hparams.enable_tensor_parallel = true;
        hparams.num_parallel           = params.num_parallel == -1 ? n_devices() : params.num_parallel;
        hparams.enable_mpi             = params.enable_mpi;
        hparams.tp_id                  = params.tp_id;
        hparams.enable_expert_parallel = params.enable_expert_parallel;
        hparams.enable_data_parallel   = params.enable_data_parallel;
    } else {
        hparams.enable_tensor_parallel = false;
        hparams.enable_expert_parallel = false;
        hparams.enable_data_parallel   = false;
        hparams.num_parallel           = 1;
        hparams.enable_mpi             = false;
        hparams.tp_id                  = 0;
    }
    hparams.enable_fused_moe            = params.enable_fused_moe;
    hparams.enable_mla                  = params.enable_mla;
    hparams.enable_cann_flash_attention = params.enable_cann_flash_attention;
}

std::string llama_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string llama_model::type_name() const {
    return llm_type_name(type);
}

std::string llama_model::desc() const {
    return pimpl->desc_str;
}

size_t llama_model::size() const {
    return pimpl->n_bytes;
}

size_t llama_model::max_nodes() const {
    return std::max<size_t>(8192, tensors_by_name.size() * 500);
}

size_t llama_model::n_devices() const {
    return devices.size();
}

uint64_t llama_model::n_elements() const {
    return pimpl->n_elements;
}

ggml_backend_dev_t llama_model::dev_layer(int il) const {
    if (params.enable_tensor_parallel) {
        return devices[hparams.tp_id];
    }
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

bool llama_model_is_recurrent(const struct llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_MAMBA:
            return true;
        case LLM_ARCH_RWKV6:
            return true;
        case LLM_ARCH_RWKV6QWEN2:
            return true;
        default:
            return false;
    }
}

const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model) {
    return &model->vocab;
}

void llama_model_free(struct llama_model * model) {
    delete model;
}

enum llama_rope_type llama_model_rope_type(const struct llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
        case LLM_ARCH_WAVTOKENIZER_DEC:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_ORION:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_CHAMELEON:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_MINICPM3:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
            return LLAMA_ROPE_TYPE_MROPE;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

bool llama_model_has_encoder(const struct llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:
            return true;
        case LLM_ARCH_T5ENCODER:
            return true;
        default:
            return false;
    }
}

llama_token llama_model_decoder_start_token(const struct llama_model * model) {
    return model->hparams.dec_start_token_id;
}

bool llama_model_has_decoder(const struct llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER:
            return false;
        default:
            return true;
    }
}

const char * llama_model_chat_template(const struct llama_model * model, const char * name) {
    const auto   key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE_N) :
                              LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it  = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        return nullptr;
    }

    return it->second.c_str();
}

int32_t llama_model_n_ctx_train(const struct llama_model * model) {
    return model->hparams.n_ctx_train;
}

const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_14M:
            return "14M";
        case LLM_TYPE_17M:
            return "17M";
        case LLM_TYPE_22M:
            return "22M";
        case LLM_TYPE_33M:
            return "33M";
        case LLM_TYPE_60M:
            return "60M";
        case LLM_TYPE_70M:
            return "70M";
        case LLM_TYPE_80M:
            return "80M";
        case LLM_TYPE_109M:
            return "109M";
        case LLM_TYPE_137M:
            return "137M";
        case LLM_TYPE_160M:
            return "160M";
        case LLM_TYPE_220M:
            return "220M";
        case LLM_TYPE_250M:
            return "250M";
        case LLM_TYPE_270M:
            return "270M";
        case LLM_TYPE_335M:
            return "335M";
        case LLM_TYPE_410M:
            return "410M";
        case LLM_TYPE_450M:
            return "450M";
        case LLM_TYPE_770M:
            return "770M";
        case LLM_TYPE_780M:
            return "780M";
        case LLM_TYPE_0_5B:
            return "0.5B";
        case LLM_TYPE_1B:
            return "1B";
        case LLM_TYPE_1_3B:
            return "1.3B";
        case LLM_TYPE_1_4B:
            return "1.4B";
        case LLM_TYPE_1_5B:
            return "1.5B";
        case LLM_TYPE_1_6B:
            return "1.6B";
        case LLM_TYPE_2B:
            return "2B";
        case LLM_TYPE_2_8B:
            return "2.8B";
        case LLM_TYPE_3B:
            return "3B";
        case LLM_TYPE_4B:
            return "4B";
        case LLM_TYPE_6B:
            return "6B";
        case LLM_TYPE_6_9B:
            return "6.9B";
        case LLM_TYPE_7B:
            return "7B";
        case LLM_TYPE_8B:
            return "8B";
        case LLM_TYPE_9B:
            return "9B";
        case LLM_TYPE_11B:
            return "11B";
        case LLM_TYPE_12B:
            return "12B";
        case LLM_TYPE_13B:
            return "13B";
        case LLM_TYPE_14B:
            return "14B";
        case LLM_TYPE_15B:
            return "15B";
        case LLM_TYPE_16B:
            return "16B";
        case LLM_TYPE_20B:
            return "20B";
        case LLM_TYPE_30B:
            return "30B";
        case LLM_TYPE_32B:
            return "32B";
        case LLM_TYPE_34B:
            return "34B";
        case LLM_TYPE_35B:
            return "35B";
        case LLM_TYPE_40B:
            return "40B";
        case LLM_TYPE_65B:
            return "65B";
        case LLM_TYPE_70B:
            return "70B";
        case LLM_TYPE_236B:
            return "236B";
        case LLM_TYPE_314B:
            return "314B";
        case LLM_TYPE_671B:
            return "671B";
        case LLM_TYPE_SMALL:
            return "0.1B";
        case LLM_TYPE_MEDIUM:
            return "0.4B";
        case LLM_TYPE_LARGE:
            return "0.8B";
        case LLM_TYPE_XL:
            return "1.5B";
        case LLM_TYPE_A1_7B:
            return "A1.7B";
        case LLM_TYPE_A2_7B:
            return "A2.7B";
        case LLM_TYPE_8x7B:
            return "8x7B";
        case LLM_TYPE_8x22B:
            return "8x22B";
        case LLM_TYPE_16x12B:
            return "16x12B";
        case LLM_TYPE_16x3_8B:
            return "16x3.8B";
        case LLM_TYPE_10B_128x3_66B:
            return "10B+128x3.66B";
        case LLM_TYPE_57B_A14B:
            return "57B.A14B";
        case LLM_TYPE_27B:
            return "27B";
        default:
            return "?B";
    }
}
