#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <chrono>

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                printf("                                      ..., \n");
                i2 = ne[2] - n;
            }
            printf("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    printf("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                printf("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    printf("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) printf(", ");
                }
                printf("],\n");
            }
            printf("                                      ],\n");
        }
        printf("                                     ]\n");
        printf("                                     sum = %f\n", sum);
    }
}


static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t             nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    {
        // parallel initialization
        static const size_t                            n_threads  = std::thread::hardware_concurrency();
        // static RNG initialization (revisit if n_threads stops being constant)
        static std::vector<std::default_random_engine> generators = []() {
            std::random_device                      rd;
            std::vector<std::default_random_engine> vec;
            vec.reserve(n_threads);
            //for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(1234 + i); } // fixed seed
            for (size_t i = 0; i < n_threads; i++) {
                vec.emplace_back(rd());
            }
            return vec;
        }();

        auto init_thread = [&](size_t ith, size_t start, size_t end) {
            std::uniform_real_distribution<float> distribution(min, max);
            auto &                                gen = generators[ith];
            for (size_t i = start; i < end; i++) {
                data[i] = distribution(gen);
            }
        };

        std::vector<std::future<void>> tasks;
        tasks.reserve(n_threads);
        for (size_t i = 0; i < n_threads; i++) {
            size_t start = i * nels / n_threads;
            size_t end   = (i + 1) * nels / n_threads;
            tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
        }
        for (auto & t : tasks) {
            t.get();
        }
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(nels % ggml_blck_size(tensor->type) == 0);

        // dummy importance matrix
        std::vector<float> imatrix(tensor->ne[0], 1.0f);
        const float *      im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f * (min + max)) {
                im = nullptr;
            }
        }

        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        {
            // parallel quantization by block
            size_t blck_size = ggml_blck_size(tensor->type);
            size_t n_blocks  = nels / blck_size;

            auto quantize_thread = [&](size_t start, size_t end) {
                ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), start * blck_size, end - start, blck_size,
                                    im);
            };

            const size_t                   min_blocks_per_thread = 1;
            const size_t                   n_threads = std::min<size_t>(std::thread::hardware_concurrency() / 2,
                                                                        std::max<size_t>(1, n_blocks / min_blocks_per_thread));
            std::vector<std::future<void>> tasks;
            tasks.reserve(n_threads);
            for (size_t i = 0; i < n_threads; i++) {
                size_t start = i * n_blocks / n_threads;
                size_t end   = (i + 1) * n_blocks / n_threads;
                tasks.push_back(std::async(std::launch::async, quantize_thread, start, end));
            }
            for (auto & t : tasks) {
                t.get();
            }
        }
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_I64) {
        // Integers with a size of 8 bytes can be set by mirroring the float data, the specific values are again not really meaningful.
        const size_t nbytes_half = ggml_nbytes(tensor) / 2;
        ggml_backend_tensor_set(tensor, data.data(), 0 * nbytes_half, nbytes_half);
        ggml_backend_tensor_set(tensor, data.data(), 1 * nbytes_half, nbytes_half);
    } else {
        GGML_ABORT("fatal error");
    }
}

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    const auto *       tt = ggml_get_type_traits(t->type);
    size_t             bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool               quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 / bs * t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t *) &buf[i]));
                    } else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(ggml_bf16_to_fp32(*(ggml_bf16_t *) &buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I64) {
                        tv.push_back((float) *(int64_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float) *(int32_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float) *(int16_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float) *(int8_t *) &buf[i]);
                    } else if (quantized) {
                        tt->to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ABORT("fatal error");
                    }
                }
            }
        }
    }

    return tv;
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];
        if (i < 10 || i > n - 10) {
            printf("a[%zu] = %f, b[%zu] = %f\n", i, a_i, i, b_i);
        }
        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// maximum absolute asymmetry between a and b
// asymmetry: (a - b) / (a + b)
// This is more stable than relative error if one of the values fluctuates towards zero.
// n: number of values to compare.
// expected_vals: optional vector of expected values for a. If expected_vals is not empty, filter out all comparisons where
//     a does not match any of the expected values. Needed for noncontinuous gradients where the numerical calculation can fail.
static double mean_abs_asymm(const float * a, const float * b, const size_t n,
                             const std::vector<float> & expected_vals) {
    double sum = 0.0f;

    size_t nvalid = 0;
    for (size_t i = 0; i < n; i++) {
        if (!expected_vals.empty()) {
            bool matches_any = false;
            for (const float & ev : expected_vals) {
                if (fabsf(a[i] - ev) < 1e-3f) {
                    matches_any = true;
                    break;
                }
            }
            if (!matches_any) {
                continue;
            }
        }

        const float asymm = (a[i] - b[i]) / (a[i] + b[i]);

        sum += fabsf(asymm);
        nvalid++;
    }

    return sum / nvalid;
}

// utils for printing the variables of the test cases

template <typename T> static std::string var_to_str(const T & x) {
    return std::to_string(x);
}

template <typename T, size_t N> static std::string var_to_str(const T (&x)[N]) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

template <typename T, size_t N> static std::string var_to_str(const std::array<T, N> & x) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

static std::string var_to_str(ggml_type type) {
    return ggml_type_name(type);
}

static std::string var_to_str(ggml_op_pool pool) {
    switch (pool) {
        case GGML_OP_POOL_AVG:
            return "avg";
        case GGML_OP_POOL_MAX:
            return "max";
        default:
            return std::to_string(pool);
    }
}

#define VAR_TO_STR(x) (#x "=" + var_to_str(x))

#define VARS_TO_STR1(a)                                VAR_TO_STR(a)
#define VARS_TO_STR2(a, b)                             VAR_TO_STR(a) + "," + VAR_TO_STR(b)
#define VARS_TO_STR3(a, b, c)                          VAR_TO_STR(a) + "," + VARS_TO_STR2(b, c)
#define VARS_TO_STR4(a, b, c, d)                       VAR_TO_STR(a) + "," + VARS_TO_STR3(b, c, d)
#define VARS_TO_STR5(a, b, c, d, e)                    VAR_TO_STR(a) + "," + VARS_TO_STR4(b, c, d, e)
#define VARS_TO_STR6(a, b, c, d, e, f)                 VAR_TO_STR(a) + "," + VARS_TO_STR5(b, c, d, e, f)
#define VARS_TO_STR7(a, b, c, d, e, f, g)              VAR_TO_STR(a) + "," + VARS_TO_STR6(b, c, d, e, f, g)
#define VARS_TO_STR8(a, b, c, d, e, f, g, h)           VAR_TO_STR(a) + "," + VARS_TO_STR7(b, c, d, e, f, g, h)
#define VARS_TO_STR9(a, b, c, d, e, f, g, h, i)        VAR_TO_STR(a) + "," + VARS_TO_STR8(b, c, d, e, f, g, h, i)
#define VARS_TO_STR10(a, b, c, d, e, f, g, h, i, j)    VAR_TO_STR(a) + "," + VARS_TO_STR9(b, c, d, e, f, g, h, i, j)
#define VARS_TO_STR11(a, b, c, d, e, f, g, h, i, j, k) VAR_TO_STR(a) + "," + VARS_TO_STR10(b, c, d, e, f, g, h, i, j, k)
#define VARS_TO_STR12(a, b, c, d, e, f, g, h, i, j, k, l) \
    VAR_TO_STR(a) + "," + VARS_TO_STR11(b, c, d, e, f, g, h, i, j, k, l)

#ifdef GGML_USE_SYCL
static bool inline _isinf(float f) {
    return (*(uint32_t *) &f & 0x7fffffff) == 0x7f800000;
}
#else
static bool inline _isinf(float f) {
    return std::isinf(f);
}
#endif

// accept FLT_MAX as infinity
static bool isinf_or_max(float f) {
    return _isinf(f) || f == FLT_MAX || f == -FLT_MAX;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return 0;  // op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

enum test_mode {
    MODE_TEST,
    MODE_PERF,
    MODE_GRAD,
};

struct test_case {
    virtual ~test_case() {}

    virtual std::string op_desc(ggml_tensor * t) { return ggml_op_desc(t); }

    virtual std::string vars() { return ""; }

    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;

    virtual double max_nmse_err() { return 1e-7; }

    virtual double max_maa_err() { return 1e-4; }

    virtual float grad_eps() { return 1e-1f; }

    // If false, estimate gradient with 2 points, neglects 3rd order derivative and higher.
    // If true,  estimate gradient with 4 points, neglects 5th order derivative and higher.
    virtual bool grad_precise() { return false; }

    // Skip gradient checks if total number of gradients to be checked is larger than this (to speed up the tests).
    virtual int64_t grad_nmax() { return 10000; }

    // No effect if empty.
    // If not empty, skip all gradient checks where the numerical result does not match any of the values.
    // Needed for dealing with noncontinuous gradients (e.g. ReLU) where estimation using finite differences is unreliable.
    virtual std::vector<float> grad_expect() { return {}; }

    virtual void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t);
        }
    }

    virtual size_t op_size(ggml_tensor * t) {
        size_t size = ggml_nbytes(t);
        // add source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i] != NULL) {
                size += ggml_nbytes(t->src[i]);
            }
        }
        return size;
    }

    virtual uint64_t op_flops(ggml_tensor * t) {
        GGML_UNUSED(t);
        return 0;
    }

    ggml_cgraph * gf = nullptr;
    ggml_cgraph * gb = nullptr;

    static const int sentinel_size = 1024;

    test_mode mode;

    std::vector<ggml_tensor *> sentinels;
    std::vector<ggml_tensor *> nodes_to_expand;

    void add_sentinel(ggml_context * ctx) {
        if (mode == MODE_PERF || mode == MODE_GRAD) {
            return;
        }
        ggml_tensor * sentinel = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sentinel_size);
        ggml_format_name(sentinel, "sent_%zu", sentinels.size());
        sentinels.push_back(sentinel);
    }

    // hijack ggml_new_tensor to add sentinels after each tensor to check for overflows in the backend

    ggml_tensor * ggml_new_tensor(ggml_context * ctx, ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * t = ::ggml_new_tensor(ctx, type, n_dims, ne);
        // add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_1d(ggml_context * ctx, ggml_type type, int64_t ne0) {
        ggml_tensor * t = ::ggml_new_tensor_1d(ctx, type, ne0);
        // add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_2d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ::ggml_new_tensor_2d(ctx, type, ne0, ne1);
        // add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_3d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * t = ::ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        // add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_4d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2,
                                     int64_t ne3) {
        ggml_tensor * t = ::ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        // add_sentinel(ctx);
        return t;
    }

    virtual bool eval(ggml_backend_t backend, const char * op_name) {
        mode = MODE_TEST;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx);

        gf = ggml_new_graph(ctx);
        ggml_graph_set_flags(gf, 1);
        ggml_graph_set_n_ctx(gf, 65536);

        // pre-graph sentinel
        // add_sentinel(ctx);

        ggml_tensor * out = build_graph(ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if the backends support the ops
        bool supported = true;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (!ggml_backend_supports_op(backend, t)) {
                printf("not supported [%s] ", ggml_backend_name(backend));
                supported = false;
                break;
            }
        }
        if (!supported) {
            printf("\n");
            ggml_free(ctx);
            return true;
        }

        // post-graph sentinel
        // add_sentinel(ctx);

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend));
            ggml_free(ctx);
            return false;
        }

        // build graph
        ggml_build_forward_expand(gf, out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor * sentinel : sentinels) {
            ggml_graph_add_node(gf, sentinel);
        }

        // randomize tensors
        initialize_tensors(ctx);
        const int max_warmup_iter_num = 10;
        const int max_iter_num = 10000;
        printf("warmup for %d iterations...\n", max_warmup_iter_num);
        for(int i = 0; i < max_warmup_iter_num; i++)
        {
            ggml_backend_graph_compute(backend, gf);
        }
        printf("warmup done, start speed test\n");
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < max_iter_num; i++)
        {
            ggml_backend_graph_compute(backend, gf);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        printf("speed test done, %d iterations, %f us per iteration\n", max_iter_num, (float)duration / max_iter_num);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        return true;
    }

    bool eval_perf(ggml_backend_t backend, const char * op_name) {
        mode = MODE_PERF;

        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params));  // smart ptr
        GGML_ASSERT(ctx);

        ggml_tensor * out = build_graph(ctx.get());

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            return true;
        }

        int len = printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if backends support op
        if (!ggml_backend_supports_op(backend, out)) {
            printf("not supported\n");
            return true;
        }

        // align while also leaving some margin for variations in parameters
        int align = 8;
        int last  = (len + align - 1) / align * align;
        if (last - len < 5) {
            last += align;
        }
        printf("%*s", last - len, "");

        // allocate
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));  // smart ptr

        if (buf == NULL) {
            printf("failed to allocate tensors\n");
            return false;
        }

        // randomize tensors
        initialize_tensors(ctx.get());

        // build graph
        ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
        ggml_build_forward_expand(gf, out);

        // warmup run
        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                    ggml_status_to_string(status));
            return false;
        }

        // determine number of runs
        int  n_runs;
        bool is_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (op_flops(out) > 0) {
            // based on flops
            const uint64_t GFLOP            = 1000 * 1000 * 1000;
            const uint64_t target_flops_cpu = 8ULL * GFLOP;
            const uint64_t target_flops_gpu = 100ULL * GFLOP;
            uint64_t       target_flops     = is_cpu ? target_flops_cpu : target_flops_gpu;
            n_runs = std::min<int>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), target_flops / op_flops(out)) + 1;
        } else {
            // based on memory size
            const size_t GB              = 1ULL << 30;
            const size_t target_size_cpu = 8 * GB;
            const size_t target_size_gpu = 32 * GB;
            size_t       target_size     = is_cpu ? target_size_cpu : target_size_gpu;
            n_runs = std::min<int>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), target_size / op_size(out)) + 1;
        }

        // duplicate the op
        for (int i = 1; i < n_runs; i++) {
            ggml_graph_add_node(gf, out);
        }

        // calculate memory
        size_t mem            = n_runs * op_size(out);
        auto   tensor_op_size = [](ggml_tensor * t) {
            size_t size = ggml_nbytes(t);
            // add source tensors
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (t->src[i] != NULL) {
                    size += ggml_nbytes(t->src[i]);
                }
            }
            return size;
        };
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (ggml_is_view_op(ggml_graph_node(gf, i)->op) || ggml_graph_node(gf, i) == out) {
                continue;
            }
            mem += tensor_op_size(ggml_graph_node(gf, i));
        }

        // run
        int64_t total_time_us = 0;
        int64_t total_mem     = 0;
        int     total_runs    = 0;
        do {
            int64_t     start_time = ggml_time_us();
            ggml_status status     = ggml_backend_graph_compute(backend, gf);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                        ggml_status_to_string(status));
                return false;
            }
            int64_t end_time = ggml_time_us();

            total_time_us += end_time - start_time;
            total_mem += mem;
            total_runs += n_runs;
        } while (total_time_us < 1000 * 1000);  // run for at least 1 second

        printf("    %8d runs - %8.2f us/run - ", total_runs, (double) total_time_us / total_runs);

        if (op_flops(out) > 0) {
            double flops_per_sec = (op_flops(out) * total_runs) / (total_time_us / 1e6);
            auto   format_flops  = [](double flops) -> std::string {
                char buf[256];
                if (flops >= 1e12) {
                    snprintf(buf, sizeof(buf), "%6.2f TFLOP", flops / 1e12);
                } else if (flops >= 1e9) {
                    snprintf(buf, sizeof(buf), "%6.2f GFLOP", flops / 1e9);
                } else if (flops >= 1e6) {
                    snprintf(buf, sizeof(buf), "%6.2f MFLOP", flops / 1e6);
                } else {
                    snprintf(buf, sizeof(buf), "%6.2f KFLOP", flops / 1e3);
                }
                return buf;
            };
            printf("%s/run - \033[1;34m%sS\033[0m", format_flops(op_flops(out)).c_str(),
                   format_flops(flops_per_sec).c_str());

        } else {
            printf("%8zu kB/run - \033[1;34m%7.2f GB/s\033[0m", op_size(out) / 1024,
                   total_mem / (total_time_us / 1e6) / 1024.0 / 1024.0 / 1024.0);
        }
        printf("\n");

        return true;
    }

    bool eval_grad(ggml_backend_t backend, const char * op_name) {
        mode                            = MODE_GRAD;
        const std::vector<float> expect = grad_expect();

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 128 +
                2 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params));  // smart ptr
        GGML_ASSERT(ctx);

        gf = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, true);
        gb = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, true);

        ggml_tensor * out = build_graph(ctx.get());

        if ((op_name != nullptr && op_desc(out) != op_name) || out->op == GGML_OP_OPT_STEP_ADAMW) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            return true;
        }

        printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        if (out->type != GGML_TYPE_F32) {
            printf("not supported [%s->type != FP32]\n", out->name);
            return true;
        }

        // check if the backend supports the ops
        bool supported  = true;
        bool any_params = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!ggml_backend_supports_op(backend, t)) {
                printf("not supported [%s] ", ggml_backend_name(backend));
                supported = false;
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM)) {
                any_params = true;
                if (t->type != GGML_TYPE_F32) {
                    printf("not supported [%s->type != FP32] ", t->name);
                    supported = false;
                    break;
                }
            }
        }
        if (!any_params) {
            printf("not supported [%s] \n", op_desc(out).c_str());
            supported = false;
        }
        if (!supported) {
            printf("\n");
            return true;
        }

        int64_t ngrads = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (t->flags & GGML_TENSOR_FLAG_PARAM) {
                ngrads += ggml_nelements(t);
            }
        }
        if (ngrads > grad_nmax()) {
            printf("skipping large tensors for speed \n");
            return true;
        }

        if (!ggml_is_scalar(out)) {
            out = ggml_sum(ctx.get(), out);
            ggml_set_name(out, "sum_of_out");
        }
        ggml_set_loss(out);

        ggml_build_forward_expand(gf, out);
        ggml_graph_cpy(gf, gb);
        ggml_build_backward_expand(ctx.get(), ctx.get(), gb, false);
        if (expect.size() != 1 || expect[0] != 0.0f) {
            GGML_ASSERT(ggml_graph_n_nodes(gb) > ggml_graph_n_nodes(gf));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL;
                 t               = ggml_get_next_tensor(ctx.get(), t)) {
                GGML_ASSERT(!(t->flags & GGML_TENSOR_FLAG_PARAM) || ggml_graph_get_grad(gb, t)->op != GGML_OP_NONE);
            }
        }

        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!ggml_backend_supports_op(backend, t)) {
                printf("not supported [%s] ", ggml_backend_name(backend));
                supported = false;
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM) && t->type != GGML_TYPE_F32) {
                printf("not supported [%s->type != FP32] ", t->name);
                supported = false;
                break;
            }
        }
        if (!supported) {
            printf("\n");
            return true;
        }

        // allocate
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));  // smart ptr
        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend));
            return false;
        }

        initialize_tensors(ctx.get());  // Randomizes all tensors (including gradients).
        ggml_graph_reset(gb);           // Sets gradients to 1 if loss, 0 otherwise.

        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                    ggml_status_to_string(status));
            return false;
        }
        status = ggml_backend_graph_compute(backend, gb);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                    ggml_status_to_string(status));
            return false;
        }

        bool ok = true;
        for (struct ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr;
             t                      = ggml_get_next_tensor(ctx.get(), t)) {
            if (!(t->flags & GGML_TENSOR_FLAG_PARAM)) {
                continue;
            }

            const char *  bn = ggml_backend_name(backend);
            const int64_t ne = ggml_nelements(t);

            std::vector<float>   ga;
            struct ggml_tensor * grad = ggml_graph_get_grad(gb, t);
            if (grad) {
                ga = tensor_to_float(grad);
            } else {
                ga.resize(ne);  // default value is 0.0f
            }

            for (int64_t i = 0; i < ne; ++i) {  // gradient algebraic
                // check for nans
                if (!std::isfinite(ga[i])) {
                    printf("[%s] nonfinite gradient at index %" PRId64 " (%s=%f) ", ggml_op_desc(t), i, bn, ga[i]);
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                break;
            }

            std::vector<float> gn(ne);  // gradient numeric
            GGML_ASSERT(ga.size() == gn.size());

            std::vector<float> x0 = tensor_to_float(t);  // original t data
            GGML_ASSERT(ggml_is_scalar(out));
            GGML_ASSERT(out->type == GGML_TYPE_F32);

            const float eps = grad_eps();
            for (int64_t i = 0; i < ne; ++i) {
                const float xiu  = x0[i] + 1.0f * eps;  // x, index i, up
                const float xiuh = x0[i] + 0.5f * eps;  // x, index i, up half
                const float xidh = x0[i] - 0.5f * eps;  // x, index i, down half
                const float xid  = x0[i] - 1.0f * eps;  // x, index i, down

                float fu, fuh, fdh, fd;                 // output values for xiu, xiuh, xid, xidh

                ggml_backend_tensor_set(t, &xiu, i * sizeof(float), sizeof(float));
                status = ggml_backend_graph_compute(backend, gf);
                if (status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                            ggml_status_to_string(status));
                    return false;
                }
                ggml_backend_tensor_get(out, &fu, 0, ggml_nbytes(out));

                ggml_backend_tensor_set(t, &xid, i * sizeof(float), sizeof(float));
                status = ggml_backend_graph_compute(backend, gf);
                if (status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                            ggml_status_to_string(status));
                    return false;
                }
                ggml_backend_tensor_get(out, &fd, 0, ggml_nbytes(out));

                if (grad_precise()) {
                    ggml_backend_tensor_set(t, &xiuh, i * sizeof(float), sizeof(float));
                    status = ggml_backend_graph_compute(backend, gf);
                    if (status != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                                ggml_status_to_string(status));
                        return false;
                    }
                    ggml_backend_tensor_get(out, &fuh, 0, ggml_nbytes(out));

                    ggml_backend_tensor_set(t, &xidh, i * sizeof(float), sizeof(float));
                    status = ggml_backend_graph_compute(backend, gf);
                    if (status != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                                ggml_status_to_string(status));
                        return false;
                    }
                    ggml_backend_tensor_get(out, &fdh, 0, ggml_nbytes(out));

                    gn[i] =
                        (8.0 * (double) fuh + (double) fd - (8.0 * (double) fdh + (double) fu)) / (6.0 * (double) eps);
                } else {
                    gn[i] = (fu - fd) / (2.0f * eps);
                }

                ggml_backend_tensor_set(t, x0.data(), 0, ggml_nbytes(t));
            }

            const double err = mean_abs_asymm(gn.data(), ga.data(), gn.size(), expect);
            if (err > max_maa_err()) {
                printf("[%s] MAA = %.9f > %.9f ", ggml_op_desc(t), err, max_maa_err());
                ok = false;
                break;
            }
            if (!ok) {
                break;
            }
        }

        if (!ok) {
            printf("compare failed ");
        }

        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            return true;
        }

        printf("\033[1;31mFAIL\033[0m\n");
        return false;
    }
};


struct test_rope : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne_a;
    int                          n_dims;
    int                          mode;
    int                          n_ctx;  // used to generate positions
    float                        fs;     // freq_scale
    float                        ef;     // ext_factor
    float                        af;     // attn_factor
    bool                         ff;
    int                          v;      // view (1 : non-contiguous a)
    bool                         forward;

    std::string vars() override {
        // forward can be inferred from the op, does not need to be printed
        return VARS_TO_STR10(type, ne_a, n_dims, mode, n_ctx, fs, ef, af, ff, v);
    }

    test_rope(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne_a = { 10, 5, 3, 1 }, int n_dims = 10,
              int mode = 0, int n_ctx = 512, float fs = 1.0f, float ef = 0.0f, float af = 0.0f, bool ff = false,
              int v = 0, bool forward = true) :
        type(type),
        ne_a(ne_a),
        n_dims(n_dims),
        mode(mode),
        n_ctx(n_ctx),
        fs(fs),
        ef(ef),
        af(af),
        ff(ff),
        v(v),
        forward(forward) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a;
            ne[0] *= 2;
            ne[1] *= 4;
            ne[2] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            if (forward) {
                ggml_set_param(ctx, a);
            }
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            if (forward) {
                ggml_set_param(ctx, a);
            }
            ggml_set_name(a, "a");
        }

        const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
        const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

        ggml_tensor * pos;
        if (is_mrope || is_vision) {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2] * 4);
        } else {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2]);
        }
        ggml_set_name(pos, "pos");

        ggml_tensor * freq = nullptr;
        if (ff) {
            freq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims / 2);
            ggml_set_name(freq, "freq");
        }

        ggml_tensor * out;
        if (is_mrope) {
            if (is_vision) {
                GGML_ASSERT(n_dims / 4 > 0);
                int rope_sections[4] = { n_dims / 4, n_dims / 4, 0,
                                         0 };  // Vision-RoPE only use first two dimension for image (x, y) coordinate
                if (forward) {
                    out = ggml_rope_multi(ctx, a, pos, freq, n_dims / 2, rope_sections, mode, 0, 10000.0f, fs, ef, af,
                                          1.0f, 1.0f);
                } else {
                    out = ggml_rope_multi_back(ctx, a, pos, freq, n_dims / 2, rope_sections, mode, 0, 10000.0f, fs, ef,
                                               af, 1.0f, 1.0f);
                }
            } else {
                GGML_ASSERT(n_dims / 3 > 0);
                int rope_sections[4] = { n_dims / 3, n_dims / 3, n_dims / 3, 0 };
                if (forward) {
                    out = ggml_rope_multi(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f,
                                          1.0f);
                } else {
                    out = ggml_rope_multi_back(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af,
                                               1.0f, 1.0f);
                }
            }
        } else {
            if (forward) {
                out = ggml_rope_ext(ctx, a, pos, freq, n_dims, mode, 4096, 10000.0f, fs, ef, af, 32.0f, 1.0f);
            } else {
                out = ggml_rope_ext_back(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
            }
        }
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                const int        num_pos_ids = (mode & GGML_ROPE_TYPE_MROPE) ? ne_a[2] * 4 : ne_a[2];
                std::vector<int> data(num_pos_ids);
                for (int i = 0; i < num_pos_ids; i++) {
                    data[i] = rand() % n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, num_pos_ids * sizeof(int));
            } else {
                if (t->ne[0] == n_dims / 2) {
                    // frequency factors in the range [0.9f, 1.1f]
                    init_tensor_uniform(t, 0.9f, 1.1f);
                } else {
                    init_tensor_uniform(t);
                }
            }
        }
    }

    double max_maa_err() override { return 1e-3; }

    bool grad_precise() override { return true; }
};

static const ggml_type all_types[] = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    // GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, // TODO: implement for all backends
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ4_XS,
};

static const ggml_type base_types[] = { GGML_TYPE_F32,  GGML_TYPE_F16,
                                        GGML_TYPE_Q8_0,  // for I8MM tests
                                        GGML_TYPE_Q4_0,
                                        GGML_TYPE_Q4_1,  // for I8MM tests
                                        GGML_TYPE_Q4_K, GGML_TYPE_IQ2_XXS };

static const ggml_type other_types[] = {
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    // GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, // TODO: implement for all backends
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_BF16,
};


// Test cases for evaluation: should try to cover edge cases while using small input sizes to keep the runtime low
static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {
    std::vector<std::unique_ptr<test_case>> test_cases;
    std::default_random_engine              rng(0);

    for (bool fw : { true }) {  // fw == forward
        bool all = true;

        for (float v : { 0 }) {
            for (float fs : { 0.025f }) {
                for (float ef : { 1.0f }) {
                    for (float af : { 0.730520f }) {
                        for (ggml_type type : { GGML_TYPE_F16 }) {
                            for (bool ff : { false }) {  // freq_factors
                                // test_cases.emplace_back(new test_rope(type, {64,  1, 84, 1}, 64, 0, 4096, fs, ef, af, ff, v, fw)); // llama 7B
                                test_cases.emplace_back(new test_rope(type, { 64, 1, 16, 1 }, 64, 0, 4096, fs, ef, af,
                                    ff, v, fw));
                                test_cases.emplace_back(new test_rope(type, { 64, 16, 16, 1 }, 64, 0, 4096, fs, ef, af,
                                    ff, v, fw));
                                test_cases.emplace_back(new test_rope(type, { 64, 16, 64, 1 }, 64, 0, 4096, fs, ef, af,
                                    ff, v, fw));
                                test_cases.emplace_back(new test_rope(type, { 64, 16, 128, 1 }, 64, 0, 4096, fs, ef, af,
                                    ff, v, fw));
                                // if (all) {
                                //     test_cases.emplace_back(new test_rope(type, {128,  40, 2, 1}, 128, 0, 512, fs, ef, af, ff, v, fw)); // llama 13B
                                //     test_cases.emplace_back(new test_rope(type, {128,  52, 2, 1}, 128, 0, 512, fs, ef, af, ff, v, fw)); // llama 30B
                                //     test_cases.emplace_back(new test_rope(type, {128,  64, 2, 1}, 128, 0, 512, fs, ef, af, ff, v, fw)); // llama 65B
                                // }

                                // if (all) {
                                //     test_cases.emplace_back(new test_rope(type, { 64,   1, 2, 1},  64, 2, 512, fs, ef, af, ff, v, fw)); // neox (falcon 7B)
                                //     test_cases.emplace_back(new test_rope(type, { 64,  71, 2, 1},  64, 2, 512, fs, ef, af, ff, v, fw)); // neox (falcon 7B)
                                //     test_cases.emplace_back(new test_rope(type, { 64,   8, 2, 1},  64, 2, 512, fs, ef, af, ff, v, fw)); // neox (falcon 40B)
                                //     test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  20, 2, 512, fs, ef, af, ff, v, fw)); // neox (stablelm)
                                //     test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  32, 2, 512, fs, ef, af, ff, v, fw)); // neox (phi-2)
                                // }

                                // if (all) {
                                //     test_cases.emplace_back(new test_rope(type, {128,  12, 2, 1}, 128, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl 2B)
                                //     test_cases.emplace_back(new test_rope(type, {128,  28, 2, 1}, 128, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl 7B)
                                //     test_cases.emplace_back(new test_rope(type, { 80,  16, 2, 1},  80, GGML_ROPE_TYPE_VISION, 512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl ViT)
                                // }

                                // test_cases.emplace_back(new test_rope(type, { 64, 128, 2, 1},  64, 2, 512, fs, ef, af, ff, v, fw)); // neox (falcon 40B)
                            }
                        }

                        all = false;
                    }
                }
            }
        }
    }

    return test_cases;
}

#include <iostream>

static bool test_backend(ggml_backend_t backend, test_mode mode, const char * op_name, const char * params_filter) {
    auto filter_test_cases = [](std::vector<std::unique_ptr<test_case>> & test_cases, const char * params_filter) {
        if (params_filter == nullptr) {
            return;
        }

        std::regex params_filter_regex(params_filter);

        for (auto it = test_cases.begin(); it != test_cases.end();) {
            if (!std::regex_search((*it)->vars(), params_filter_regex)) {
                it = test_cases.erase(it);
                continue;
            }

            it++;
        }
    };

    if (mode == MODE_TEST) {
        auto test_cases = make_test_cases_eval();
        filter_test_cases(test_cases, params_filter);
        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->eval(backend, op_name)) {
                n_ok++;
            }
        }
        printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());


        return n_ok == test_cases.size();
    }
    GGML_ABORT("fatal error");
}

static void usage(char ** argv) {
    printf("Usage: %s [mode] [-o <op>] [-b <backend>] [-p <params regex>]\n", argv[0]);
    printf("    valid modes:\n");
    printf("      - test (default, compare with CPU backend for correctness)\n");
    printf("      - grad (compare gradients from backpropagation with method of finite differences)\n");
    printf("      - perf (performance evaluation)\n");
    printf("    op names for -o are as given by ggml_op_desc() (e.g. ADD, MUL_MAT, etc)\n");
}

int main(int argc, char ** argv) {
    test_mode    mode           = MODE_TEST;
    const char * op_name_filter = nullptr;
    const char * backend_filter = nullptr;
    const char * params_filter  = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            mode = MODE_TEST;
        } else if (strcmp(argv[i], "perf") == 0) {
            mode = MODE_PERF;
        } else if (strcmp(argv[i], "grad") == 0) {
            mode = MODE_GRAD;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                op_name_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                backend_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) {
                params_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else {
            usage(argv);
            return 1;
        }
    }

    // load and enumerate backends
    ggml_backend_load_all();

    printf("Testing %zu devices\n\n", ggml_backend_dev_count());

    size_t n_ok = 0;

    for (size_t i = 0; i < 1; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        printf("Backend %zu/%zu: %s\n", i + 1, ggml_backend_dev_count(), ggml_backend_dev_name(dev));

        if (backend_filter != NULL && strcmp(backend_filter, ggml_backend_dev_name(dev)) != 0) {
            printf("  Skipping\n");
            n_ok++;
            continue;
        }

        if (backend_filter == NULL && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && mode != MODE_GRAD) {
            printf("  Skipping CPU backend\n");
            n_ok++;
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        GGML_ASSERT(backend != NULL);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto               ggml_backend_set_n_threads_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            // TODO: better value for n_threads
            ggml_backend_set_n_threads_fn(backend, std::thread::hardware_concurrency());
        }

        printf("  Device description: %s\n", ggml_backend_dev_description(dev));
        size_t free, total;  // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  Device memory: %zu MB (%zu MB free)\n", total / 1024 / 1024, free / 1024 / 1024);
        printf("\n");

        bool ok = test_backend(backend, mode, op_name_filter, params_filter);

        printf("  Backend %s: ", ggml_backend_name(backend));
        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            n_ok++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        printf("\n");

        ggml_backend_free(backend);
    }

    ggml_quantize_free();

    printf("%zu/%zu backends passed\n", n_ok, ggml_backend_dev_count());

    if (n_ok != ggml_backend_dev_count()) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }

    printf("\033[1;32mOK\033[0m\n");
    return 0;
}