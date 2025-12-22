// This file defines tests for various GGML ops and backends.
// For the forward pass it asserts that the results of multiple backends computing the same GGML ops are consistent.
// For the backward pass it asserts that the gradients from backpropagation are consistent
// with the gradients obtained via the method of finite differences ("grad" mode, this is optional).
// It is also possible to check the performance ("perf" mode).
//
// this file has three sections: Section 1 does general setup, section 2 defines the GGML ops to be tested,
// and section 3 defines which tests to run.
// Quick start for adding a new GGML op: Go to section 2 and create a struct that inherits from test_case,
// then go to section 3 and add an instantiation of your struct.

// ##############################
// ## Section 1: General Setup ##
// ##############################

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
    MODE_SPEED
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

    virtual bool eval(ggml_backend_t backend1, ggml_backend_t backend2, const char * op_name) {
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
        for (ggml_backend_t backend : { backend1, backend2 }) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                if (!ggml_backend_supports_op(backend, t)) {
                    printf("not supported [%s] ", ggml_backend_name(backend));
                    supported = false;
                    break;
                }
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
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);

        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend1));
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

        // compare
        struct callback_userdata {
            bool           ok;
            double         max_err;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
        };

        callback_userdata ud{ true, max_nmse_err(), backend1, backend2 };

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            callback_userdata * ud  = (callback_userdata *) user_data;
            const char *        bn1 = ggml_backend_name(ud->backend1);
            const char *        bn2 = ggml_backend_name(ud->backend2);

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    printf("sentinel mismatch: %s ", t1->name);
                    ud->ok = false;
                    return true;
                }
            }

            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%s=%f %s=%f) ", ggml_op_desc(t1), i, bn1, f1[i], bn2, f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) {
                printf("[%s] NMSE = %.9f > %.9f ", ggml_op_desc(t1), err, ud->max_err);
                //for (int i = 0; i < (int) f1.size(); i++) {
                //    printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                //}
                //printf("\n");
                //exit(1);
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };

        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud, true);

        if (!cmp_ok) {
            printf("compare failed ");
        }

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        if (ud.ok && cmp_ok) {
            printf("\033[1;32mOK\033[0m\n");
            return true;
        }

        printf("\033[1;31mFAIL\033[0m\n");
        return false;
    }

    virtual bool eval_speed(ggml_backend_t backend, const char * op_name) {
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

// ###################################
// ## Section 2: GGML Op Defintions ##
// ###################################

// The following is an example showing the bare minimum for creating a test for a GGML op.

// GGML_OP_EXAMPLE
struct test_example : public test_case {
    // Always define these 2 or variants thereof:
    const ggml_type              type;  // The type of the input tensors.
    const std::array<int64_t, 4> ne;    // The shape of the input tensors.

    // For some ops it's necessary to define multiple types or shapes for the inputs.
    // Or they may need additional parameters.

    // Put all parameters needed to fully define the test into one of the VARS_TO_STR macros.
    // In most cases these are just the properties of the struct that you defined above.
    // This is needed for info prints.
    std::string vars() override { return VARS_TO_STR2(type, ne); }

    // Define a constructor for the struct.
    // In most cases it will be sufficient to have the same arguments as the struct has properties
    // and just use initializer lists.
    test_example(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 }) : type(type), ne(ne) {}

    // Define how a simple GGML compute graph can be constructed for the new GGML op.
    ggml_tensor * build_graph(ggml_context * ctx) override {
        // Step 1: create input tensors that don't depend on any other tensors:
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");  // Setting names is optional but it's useful for debugging.

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        // Step 2: use the op that you want to test in the GGML compute graph.
        ggml_tensor * out = ggml_add(ctx, a, b);  // For this example we're just doing a simple addition.
        ggml_set_name(out, "out");

        // Step 3: return the output tensor.
        return out;
    }

    // In order to also check the gradients for your op, add calls like ggml_set_param(ctx, a)
    // immediately after you create the tensors.
    // This is optional and only makes sense if a backward pass has actually been implemented for the new op.
};

// GGML_OP_UNARY
struct test_unary : public test_case {
    const ggml_unary_op          op;
    const ggml_type              type;
    const std::array<int64_t, 4> ne_a;
    int                          v;  // view (1 : non-contiguous a)

    std::string vars() override { return VARS_TO_STR3(type, ne_a, v); }

    test_unary(ggml_unary_op op, ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne_a = { 128, 2, 2, 2 },
               int v = 0) :
        op(op),
        type(type),
        ne_a(ne_a),
        v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const bool grad_supported = op == GGML_UNARY_OP_ABS || op == GGML_UNARY_OP_SGN || op == GGML_UNARY_OP_NEG ||
                                    op == GGML_UNARY_OP_STEP || op == GGML_UNARY_OP_RELU || op == GGML_UNARY_OP_SILU;

        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a;
            ne[0] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            if (grad_supported) {
                ggml_set_param(ctx, a);
            }
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            if (grad_supported) {
                ggml_set_param(ctx, a);
            }
            ggml_set_name(a, "a");
        }

        ggml_tensor * out = ggml_unary(ctx, a, op);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }

    float grad_eps() override { return 15.0f; }

    std::vector<float> grad_expect() override {
        if (op == GGML_UNARY_OP_ABS) {
            return { -1.0f, 1.0f };
        }
        if (op == GGML_UNARY_OP_SGN || op == GGML_UNARY_OP_STEP) {
            return { 0.0f };
        }
        if (op == GGML_UNARY_OP_RELU) {
            return { 0.0f, 1.0f };
        }
        return {};
    }
};

// GGML_OP_GET_ROWS
struct test_get_rows : public test_case {
    const ggml_type type;
    const int       n;  // cols
    const int       m;  // rows
    const int       r;  // rows to get
    const int       b;  // batch size
    const bool      v;  // view (non-contiguous src1)

    std::string vars() override { return VARS_TO_STR6(type, n, m, r, b, v); }

    test_get_rows(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3, int b = 1, bool v = false) :
        type(type),
        n(n),
        m(m),
        r(r),
        b(b),
        v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in = ggml_new_tensor_3d(ctx, type, n, m, b);
        ggml_set_name(in, "in");

        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r, b);
        ggml_set_name(rows, "rows");
        if (v) {
            rows = ggml_view_2d(ctx, rows, r / 2, b, rows->nb[1], 0);
            ggml_set_name(rows, "view_of_rows");
        }

        const bool grad_supported = ggml_is_matrix(in) && ggml_is_vector(rows);
        if (grad_supported) {
            ggml_set_param(ctx, in);
            // rows is a constant input -> no gradients
        }

        ggml_tensor * out = ggml_get_rows(ctx, in, rows);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) {
                    continue;
                }
                // rows
                std::vector<int> data(r * b);
                for (int i = 0; i < r * b; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * b * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
        // for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        //     std::vector<uint8_t> data(ggml_nbytes(t));
        //     ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        //     ggml_print_tensor((uint8_t*)data.data(), t->type, t->ne, t->nb, ggml_nbytes(t));
        // }
    }
};

// GGML_OP_GET_ROWS_BACK
struct test_get_rows_back : public test_case {
    const ggml_type type;
    const int       n;  // cols
    const int       m;  // rows
    const int       r;  // rows to get
    const int       b;  // batch size
    const bool      v;  // view (non-contiguous src1)

    std::string vars() override { return VARS_TO_STR6(type, n, m, r, b, v); }

    test_get_rows_back(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3, int b = 1, bool v = false) :
        type(type),
        n(n),
        m(m),
        r(r),
        b(b),
        v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in_forward = ggml_new_tensor_3d(ctx, type, n, m, b);
        ggml_set_name(in_forward, "in_forward");

        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r, b);
        ggml_set_name(rows, "rows");
        if (v) {
            rows = ggml_view_2d(ctx, rows, r / 2, b, rows->nb[1], 0);
            ggml_set_name(rows, "view_of_rows");
        }

        ggml_tensor * grad = ggml_new_tensor_3d(ctx, type, n, r, b);
        ggml_set_name(grad, "grad");

        ggml_tensor * out = ggml_get_rows_back(ctx, grad, rows, in_forward);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) {
                    continue;
                }
                // rows
                std::vector<int> data(r * b);
                for (int i = 0; i < r * b; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * b * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_ARGMAX
struct test_argmax : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_argmax(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 100, 1, 1 }) : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_argmax(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device         rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    double max_nmse_err() override { return 0.0; }
};

// GGML_OP_COUNT_EQUAL
struct test_count_equal : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_count_equal(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 4, 500, 1, 1 }) :
        type(type),
        ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * a_argmax = ggml_argmax(ctx, a);
        ggml_set_name(a_argmax, "a_argmax");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        ggml_tensor * b_argmax = ggml_argmax(ctx, b);
        ggml_set_name(b_argmax, "b_argmax");

        ggml_tensor * out = ggml_count_equal(ctx, a_argmax, b_argmax);
        ggml_set_name(out, "out");

        return out;
    }

    double max_nmse_err() override { return 0.0; }
};

// GGML_OP_REPEAT
struct test_repeat : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4>     nr;

    std::string vars() override { return VARS_TO_STR3(type, ne, nr); }

    size_t op_size(ggml_tensor * t) override { return ggml_nbytes(t) * 2; }

    test_repeat(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 },
                std::array<int, 4> nr = { 2, 2, 2, 2 }) :
        type(type),
        ne(ne),
        nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * target =
            ggml_new_tensor_4d(ctx, type, ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2], ne[3] * nr[3]);
        ggml_set_name(target, "target");

        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, src);
        ggml_set_name(src, "src");

        ggml_tensor * out = ggml_repeat(ctx, src, target);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_REPEAT_BACK
struct test_repeat_back : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4>     nr;
    const bool                   v;  // whether src is a noncontiguous view

    std::string vars() override { return VARS_TO_STR4(type, ne, nr, v); }

    size_t op_size(ggml_tensor * t) override { return ggml_nbytes(t) * 2; }

    test_repeat_back(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 8, 6, 4, 2 },
                     std::array<int, 4> nr = { 2, 2, 2, 2 }, bool v = false) :
        type(type),
        ne(ne),
        nr(nr),
        v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor_4d(ctx, type, ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2], ne[3] * nr[3]);
        ggml_set_name(src, "src");

        if (v) {
            GGML_ASSERT(ne[0] % 2 == 0);
            GGML_ASSERT(ne[1] % 2 == 0);
            GGML_ASSERT(ne[2] % 2 == 0);
            GGML_ASSERT(ne[3] % 2 == 0);
            GGML_ASSERT(nr[0] % 2 == 0 || nr[0] == 1);
            GGML_ASSERT(nr[1] % 2 == 0 || nr[1] == 1);
            GGML_ASSERT(nr[2] % 2 == 0 || nr[2] == 1);
            GGML_ASSERT(nr[3] % 2 == 0 || nr[3] == 1);

            const int64_t ne00 = nr[0] == 1 ? src->ne[0] : src->ne[0] / 2;
            const int64_t ne01 = nr[1] == 1 ? src->ne[1] : src->ne[1] / 2;
            const int64_t ne02 = nr[2] == 1 ? src->ne[2] : src->ne[2] / 2;
            const int64_t ne03 = nr[3] == 1 ? src->ne[3] : src->ne[3] / 2;

            src = ggml_view_4d(ctx, src, ne00, ne01, ne02, ne03, src->nb[1], src->nb[2], src->nb[3], 0);
        }

        ggml_tensor * target = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(target, "target");

        ggml_tensor * out = ggml_repeat_back(ctx, src, target);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_DUP
struct test_dup : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute;
    bool                         _use_permute;

    std::string vars() override {
        std::string v = VARS_TO_STR2(type, ne);
        if (_use_permute) {
            v += "," + VAR_TO_STR(permute);
        }
        return v;
    }

    test_dup(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 10, 20, 1 },
             std::array<int64_t, 4> permute = { 0, 0, 0, 0 }) :
        type(type),
        ne(ne),
        permute(permute),
        _use_permute(permute[0] + permute[1] + permute[2] + permute[3] > 0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, src);
        ggml_set_name(src, "src");

        if (_use_permute) {
            src = ggml_permute(ctx, src, permute[0], permute[1], permute[2], permute[3]);
            ggml_set_name(src, "src_permuted");
        }

        ggml_tensor * out = ggml_dup(ctx, src);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SET
struct test_set : public test_case {
    const ggml_type              type_src;
    const ggml_type              type_dst;
    const std::array<int64_t, 4> ne;
    const int                    dim;

    std::string vars() override { return VARS_TO_STR4(type_src, type_dst, ne, dim); }

    size_t op_size(ggml_tensor * t) override { return ggml_nbytes(t) + ggml_nbytes(t->src[0]); }

    test_set(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
             std::array<int64_t, 4> ne = { 6, 5, 4, 3 }, int dim = 1) :
        type_src(type_src),
        type_dst(type_dst),
        ne(ne),
        dim(dim) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne.data());
        ggml_set_param(ctx, src);
        ggml_set_name(src, "src");

        auto ne_dst = ne;
        for (int i = 0; i < dim; ++i) {
            ne_dst[i] *= 2;
        }
        ggml_tensor * dst = ggml_new_tensor(ctx, type_dst, 4, ne_dst.data());
        ggml_set_param(ctx, dst);
        ggml_set_name(dst, "dst");

        size_t offset = 0;
        for (int i = 0; i < dim; ++i) {
            offset += ((ne_dst[i] - ne[i]) / 2) * dst->nb[i];
        }
        ggml_tensor * out = ggml_set(ctx, dst, src,
                                     // The backward pass requires setting a contiguous region:
                                     src->nb[1], src->nb[2], src->nb[3], offset);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_CPY
struct test_cpy : public test_case {
    const ggml_type              type_src;
    const ggml_type              type_dst;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute;
    bool                         _src_use_permute;

    std::string vars() override { return VARS_TO_STR4(type_src, type_dst, ne, permute); }

    double max_nmse_err() override { return 1e-6; }

    size_t op_size(ggml_tensor * t) override { return ggml_nbytes(t) + ggml_nbytes(t->src[0]); }

    test_cpy(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
             std::array<int64_t, 4> ne = { 10, 10, 10, 1 }, std::array<int64_t, 4> permute = { 0, 0, 0, 0 }) :
        type_src(type_src),
        type_dst(type_dst),
        ne(ne),
        permute(permute),
        _src_use_permute(permute[0] + permute[1] + permute[2] + permute[3] > 0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne.data());
        ggml_set_param(ctx, src);
        ggml_set_name(src, "src");

        if (_src_use_permute) {
            src = ggml_permute(ctx, src, permute[0], permute[1], permute[2], permute[3]);
            ggml_set_name(src, "src_permuted");
        }

        ggml_tensor * dst = ggml_new_tensor(ctx, type_dst, 4, src->ne);
        ggml_set_param(ctx, dst);
        ggml_set_name(dst, "dst");

        ggml_tensor * out = ggml_cpy(ctx, src, dst);
        ggml_set_name(out, "out");

        out = ggml_cont(ctx, out);
        ggml_set_name(out, "out_cont");

        out = ggml_cpy(ctx, out, out);
        ggml_set_name(out, "out_copy");

        return out;
    }
};

// GGML_OP_SCALE
struct test_scale : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    float                        scale;

    std::string vars() override { return VARS_TO_STR3(type, ne, scale); }

    test_scale(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 10, 10, 10 }, float scale = 2.0f) :
        type(type),
        ne(ne),
        scale(scale) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_scale(ctx, a, scale);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_RESHAPE
struct test_reshape : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> shape;

    std::string vars() override { return VARS_TO_STR3(type, ne, shape); }

    test_reshape(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 2, 3, 4, 5 },
                 std::array<int64_t, 4> shape = { 5, 4, 3, 2 }) :
        type(type),
        ne(ne),
        shape(shape) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(x, "x");

        ggml_tensor * s = ggml_new_tensor(ctx, type, 4, shape.data());
        ggml_set_name(s, "shape");

        ggml_set_param(ctx, x);
        ggml_set_param(ctx, s);

        // 3) 调用 reshape
        ggml_tensor * y = ggml_reshape(ctx, x, s);
        ggml_set_name(y, "y");

        ggml_tensor * out = ggml_cont(ctx, y);
        ggml_set_name(out, "out");

        out = ggml_reshape(ctx, out, s);
        ggml_set_name(out, "out2");

        return out;
    }
};

// GGML_OP_PERMUTE
struct test_permute : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> order;

    std::string vars() override { return VARS_TO_STR3(type, ne, order); }

    test_permute(ggml_type type_ = GGML_TYPE_F32, std::array<int64_t, 4> ne_ = { 2, 3, 4, 5 },
                 std::array<int64_t, 4> order_ = { 3, 2, 1, 0 }) :
        type(type_),
        ne(ne_),
        order(order_) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // 1) 创建输入张量 x
        ggml_tensor * x = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(x, "x");

        ggml_set_param(ctx, x);

        ggml_tensor * y = ggml_permute(ctx, x, order[0], order[1], order[2], order[3]);
        ggml_set_name(y, "y");

        ggml_tensor * out = ggml_cont(ctx, y);
        ggml_set_name(out, "out");

        out = ggml_permute(ctx, x, 0, 1, 2, 3);
        ggml_set_name(out, "out2");

        return out;
    }
};

// GGML_OP_TRANSPOSE
struct test_transpose : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_transpose(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 10, 10, 1 }) :
        type(type),
        ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, src);
        ggml_set_name(src, "src");

        src = ggml_transpose(ctx, src);
        ggml_set_name(src, "src_transposed");

        ggml_tensor * y = ggml_cont(ctx, src);
        ggml_set_name(y, "y");

        y = ggml_transpose(ctx, y);
        ggml_set_name(y, "y_transposed");

        y = ggml_transpose(ctx, y);
        ggml_set_name(y, "y_transposed2");

        return y;
    }
};

// GGML_OP_VIEW
struct test_view : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_view(ggml_type type_ = GGML_TYPE_F32, std::array<int64_t, 4> ne_ = { 64, 5, 4, 3 }) : type(type_), ne(ne_) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(x, "a");

        ggml_set_param(ctx, x);

        ggml_tensor * y = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], 0);
        ggml_set_name(y, "y");

        ggml_tensor * out = ggml_cont(ctx, y);
        ggml_set_name(out, "out");

        out = ggml_view_4d(ctx, out, out->ne[0], out->ne[1], out->ne[2], out->ne[3], out->nb[1], out->nb[2], out->nb[3],
                           0);
        ggml_set_name(out, "out2");

        return out;
    }
};

// GGML_OP_CONT
struct test_cont : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_cont(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 10, 10, 1 }) : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, src);
        ggml_set_name(src, "src");

        src = ggml_transpose(ctx, src);
        ggml_set_name(src, "src_transposed");

        ggml_tensor * out = ggml_cont(ctx, src);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ADD
// GGML_OP_SUB
// GGML_OP_MUL
// GGML_OP_DIV
struct test_bin_bcast : public test_case {
    using op_t = ggml_tensor * (*) (ggml_context *, ggml_tensor *, ggml_tensor *);
    op_t                         op;
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4>     nr;

    std::string vars() override { return VARS_TO_STR3(type, ne, nr); }

    size_t op_size(ggml_tensor * t) override { return ggml_nbytes(t) * 3; }

    test_bin_bcast(op_t op, ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 10, 1, 1 },
                   std::array<int, 4> nr = { 1, 2, 1, 1 }) :
        op(op),
        type(type),
        ne(ne),
        nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2], ne[3] * nr[3]);
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        ggml_tensor * c = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(c, "c");

        // The backward pass supports broadcasting only for GGML_ADD:
        const bool grad_supported = op == ggml_add || ggml_are_same_shape(a, b);
        if (grad_supported) {
            ggml_set_param(ctx, a);
            ggml_set_param(ctx, b);
            ggml_set_param(ctx, c);
        }

        ggml_tensor * out  = op(ctx, a, b);
        ggml_tensor * out2 = op(ctx, out, c);
        ggml_set_name(out2, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (op == ggml_mul || op == ggml_div) {
                // MUL and DIV have numerical issues around zero:
                init_tensor_uniform(t, 0.9f, 1.1f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    float grad_eps() override { return 0.1f * (op == ggml_mul ? ne[0] * ne[1] * ne[2] * ne[3] : 1); }

    bool grad_precise() override { return op == ggml_div; }

    double max_maa_err() override { return op == ggml_add ? 1e-4 : 1e-3; }
};

// GGML_OP_ADD1
struct test_add1 : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_add1(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 }) : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor_1d(ctx, type, 1);
        // ggml_set_param(ctx, b); // TODO: implement
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_add1(ctx, a, b);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override { return 0.1f * ne[0] * ne[1] * ne[2] * ne[3]; }
};

// GGML_OP_SILU_BACK
struct test_silu_back : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    float                        eps;

    std::string vars() override { return VARS_TO_STR3(type, ne, eps); }

    test_silu_back(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 64, 5, 4, 3 }, float eps = 1e-6f) :
        type(type),
        ne(ne),
        eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * grad = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(grad, "grad");

        ggml_tensor * out = ggml_silu_back(ctx, a, grad);
        ggml_set_name(out, "out");

        return out;
    }

    bool grad_precise() override { return true; }
};

// GGML_OP_NORM
struct test_norm : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const bool                   v;  // whether a is a non-contiguous view
    const float                  eps;

    std::string vars() override { return VARS_TO_STR4(type, ne, v, eps); }

    test_norm(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 64, 5, 4, 3 }, bool v = false,
              float eps = 1e-6f) :
        type(type),
        ne(ne),
        v(v),
        eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        if (v) {
            a = ggml_view_4d(ctx, a, a->ne[0] / 2, a->ne[1] / 2, a->ne[2] / 2, a->ne[3] / 2, a->nb[1], a->nb[2],
                             a->nb[3], 0);
            ggml_set_name(a, "view of a");
        }

        ggml_tensor * out = ggml_norm(ctx, a, eps);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_RMS_NORM
struct test_rms_norm : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const bool                   v;  // whether a is a non-contiguous view
    const float                  eps;

    std::string vars() override { return VARS_TO_STR4(type, ne, v, eps); }

    test_rms_norm(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 64, 5, 4, 3 }, bool v = false,
                  float eps = 1e-6f) :
        type(type),
        ne(ne),
        v(v),
        eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        if (v) {
            a = ggml_view_4d(ctx, a, a->ne[0] / 2, a->ne[1] / 2, a->ne[2] / 2, a->ne[3] / 2, a->nb[1], a->nb[2],
                             a->nb[3], 0);
            ggml_set_name(a, "view of a");
        }

        ggml_tensor * out = ggml_rms_norm(ctx, a, eps);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.f, 10.f);
        }
    }

    float grad_eps() override { return 1.0f; }

    bool grad_precise() override { return true; }
};

// GGML_OP_RMS_NORM_BACK
struct test_rms_norm_back : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const float                  eps;

    std::string vars() override { return VARS_TO_STR3(type, ne, eps); }

    test_rms_norm_back(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 64, 5, 4, 3 }, float eps = 1e-6f) :
        type(type),
        ne(ne),
        eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_rms_norm_back(ctx, a, b, eps);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.f, 10.f);
        }
    }
};

// GGML_OP_SSM_CONV
struct test_ssm_conv : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;

    std::string vars() override { return VARS_TO_STR3(type, ne_a, ne_b); }

    test_ssm_conv(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne_a = { 10, 10, 10, 1 },
                  std::array<int64_t, 4> ne_b = { 3, 3, 1, 1 }) :
        type(type),
        ne_a(ne_a),
        ne_b(ne_b) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a   = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * b   = ggml_new_tensor(ctx, type, 4, ne_b.data());
        ggml_tensor * out = ggml_ssm_conv(ctx, a, b);
        return out;
    }
};

// GGML_OP_SSM_SCAN
struct test_ssm_scan : public test_case {
    const ggml_type type;

    const int64_t d_state;
    const int64_t d_inner;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override { return VARS_TO_STR5(type, d_state, d_inner, n_seq_tokens, n_seqs); }

    test_ssm_scan(ggml_type type = GGML_TYPE_F32, int64_t d_state = 32, int64_t d_inner = 32, int64_t n_seq_tokens = 32,
                  int64_t n_seqs = 32) :
        type(type),
        d_state(d_state),
        d_inner(d_inner),
        n_seq_tokens(n_seq_tokens),
        n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * s = ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_state, d_inner, n_seqs, 1 }.data());
        ggml_tensor * x =
            ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_inner, n_seq_tokens, n_seqs, 1 }.data());
        ggml_tensor * dt =
            ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_inner, n_seq_tokens, n_seqs, 1 }.data());
        ggml_tensor * A = ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_state, d_inner, 1, 1 }.data());
        ggml_tensor * B =
            ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_state, n_seq_tokens, n_seqs, 1 }.data());
        ggml_tensor * C =
            ggml_new_tensor(ctx, type, 4, std::vector<int64_t>{ d_state, n_seq_tokens, n_seqs, 1 }.data());
        ggml_tensor * out = ggml_ssm_scan(ctx, s, x, dt, A, B, C);
        return out;
    }
};

// GGML_OP_RWKV_WKV6
struct test_rwkv_wkv6 : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override { return VARS_TO_STR5(type, head_count, head_size, n_seq_tokens, n_seqs); }

    test_rwkv_wkv6(ggml_type type = GGML_TYPE_F32, int64_t head_count = 32, int64_t head_size = 64,
                   int64_t n_seq_tokens = 32, int64_t n_seqs = 32) :
        type(type),
        head_count(head_count),
        head_size(head_size),
        n_seq_tokens(n_seq_tokens),
        n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_tokens = n_seq_tokens * n_seqs;
        ggml_tensor * r = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * k = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * v = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * tf = ggml_new_tensor(ctx, type, 2, std::vector<int64_t>{ head_size, head_count }.data());
        ggml_tensor * td =
            ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * s =
            ggml_new_tensor(ctx, type, 2, std::vector<int64_t>{ head_size * head_size * head_count, n_seqs }.data());
        ggml_tensor * out = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, s);
        return out;
    }
};

// GGML_OP_GATED_LINEAR_ATTN
struct test_gla : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override { return VARS_TO_STR5(type, head_count, head_size, n_seq_tokens, n_seqs); }

    test_gla(ggml_type type = GGML_TYPE_F32, int64_t head_count = 32, int64_t head_size = 64, int64_t n_seq_tokens = 32,
             int64_t n_seqs = 32) :
        type(type),
        head_count(head_count),
        head_size(head_size),
        n_seq_tokens(n_seq_tokens),
        n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_tokens = n_seq_tokens * n_seqs;
        ggml_tensor * q = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * k = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * v = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * g = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * s =
            ggml_new_tensor(ctx, type, 2, std::vector<int64_t>{ head_size * head_size * head_count, n_seqs }.data());
        ggml_tensor * out = ggml_gated_linear_attn(ctx, k, v, q, g, s, pow(head_size, -0.5));
        return out;
    }
};

// GGML_OP_MUL_MAT
struct test_mul_mat : public test_case {
    const ggml_type              type_a;
    const ggml_type              type_b;
    const int64_t                m;
    const int64_t                n;
    const int64_t                k;
    const std::array<int64_t, 2> bs;   // dims 3 and 4
    const std::array<int64_t, 2> nr;   // repeat in dims 3 and 4
    const std::array<int64_t, 4> per;  // permutation of dimensions

    std::string vars() override { return VARS_TO_STR8(type_a, type_b, m, n, k, bs, nr, per); }

    double max_nmse_err() override { return 5e-4; }

    int64_t grad_nmax() override { return 20000; }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * m * n * k * bs[0] * nr[0] * bs[1] * nr[1];
    }

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32, int64_t m = 32, int64_t n = 32,
                 int64_t k = 32, std::array<int64_t, 2> bs = { 10, 10 }, std::array<int64_t, 2> nr = { 2, 2 },
                 std::array<int64_t, 4> per = { 0, 1, 2, 3 }) :
        type_a(type_a),
        type_b(type_b),
        m(m),
        n(n),
        k(k),
        bs(bs),
        nr(nr),
        per(per) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * a;
        ggml_tensor * b;

        const int npermuted = (per[0] != 0) + (per[1] != 1) + (per[2] != 2) + (per[3] != 3);
        if (npermuted > 0) {
            GGML_ASSERT(npermuted == 2);
            GGML_ASSERT(!ggml_is_quantized(type_a) || per[0] == 0);
            GGML_ASSERT(!ggml_is_quantized(type_b) || per[0] == 0);

            // Create tensors with the permuted dimensions, then permute them back to the dimensions given by m,n,k.
            const int64_t ne_a[4] = { k, m, bs[0], bs[1] };
            const int64_t ne_b[4] = { k, n, bs[0] * nr[0], bs[1] * nr[1] };

            a = ggml_new_tensor_4d(ctx, type_a, ne_a[per[0]], ne_a[per[1]], ne_a[per[2]], ne_a[per[3]]);
            b = ggml_new_tensor_4d(ctx, type_b, ne_b[per[0]], ne_b[per[1]], ne_b[per[2]], ne_b[per[3]]);
            if (!ggml_is_quantized(type_a)) {
                if (bs[1] == 1 && nr[1] == 1) {
                    ggml_set_param(ctx, a);
                }
                ggml_set_param(ctx, b);
            }
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");

            a = ggml_permute(ctx, a, per[0], per[1], per[2], per[3]);
            b = ggml_permute(ctx, b, per[0], per[1], per[2], per[3]);
            ggml_set_name(a, "a_permuted");
            ggml_set_name(b, "b_permuted");
        } else {
            a = ggml_new_tensor_4d(ctx, type_a, k, m, bs[0], bs[1]);
            b = ggml_new_tensor_4d(ctx, type_b, k, n, bs[0] * nr[0], bs[1] * nr[1]);
            if (!ggml_is_quantized(type_a)) {
                if (bs[1] == 1 && nr[1] == 1) {
                    ggml_set_param(ctx, a);
                }
                ggml_set_param(ctx, b);
            }
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_MUL_MAT_ID
struct test_mul_mat_id : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int       n_mats;
    const int       n_used;
    const bool      b;  // brodcast b matrix
    const int64_t   m;
    const int64_t   n;
    const int64_t   k;

    std::string vars() override { return VARS_TO_STR8(type_a, type_b, n_mats, n_used, b, m, n, k); }

    double max_nmse_err() override { return 5e-4; }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * m * k * n * n_used;
    }

    test_mul_mat_id(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32, int n_mats = 8, int n_used = 2,
                    bool b = false, int64_t m = 32, int64_t n = 32, int64_t k = 32) :
        type_a(type_a),
        type_b(type_b),
        n_mats(n_mats),
        n_used(n_used),
        b(b),
        m(m),
        n(n),
        k(k) {
        GGML_ASSERT(n_used <= n_mats);
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * as = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
        ggml_set_name(as, "as");

        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, n);
        ggml_set_name(ids, "ids");
        if (n_used != n_mats) {
            ids = ggml_view_2d(ctx, ids, n_used, n, ids->nb[1], 0);
            ggml_set_name(ids, "view_of_ids");
        }

        ggml_tensor * b = ggml_new_tensor_3d(ctx, type_b, k, this->b ? 1 : n_used, n);
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device         rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) {
                    continue;
                }
                // ids
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i % n_mats;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_OUT_PROD
struct test_out_prod : public test_case {
    const ggml_type              type_a;
    const ggml_type              type_b;
    const int64_t                m;
    const int64_t                n;
    const int64_t                k;
    const std::array<int64_t, 2> bs;  // dims 3 and 4
    const std::array<int64_t, 2> nr;  // repeat in dims 3 and 4
    const bool                   trans_b;

    std::string vars() override { return VARS_TO_STR8(type_a, type_b, m, n, k, bs, nr, trans_b); }

    double max_nmse_err() override { return 5e-4; }

    test_out_prod(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32, int64_t m = 32, int64_t n = 32,
                  int64_t k = 32, std::array<int64_t, 2> bs = { 10, 10 }, std::array<int64_t, 2> nr = { 2, 2 },
                  bool trans_b = false) :
        type_a(type_a),
        type_b(type_b),
        m(m),
        n(n),
        k(k),
        bs(bs),
        nr(nr),
        trans_b(trans_b) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type_a, m, k, bs[0], bs[1]);
        ggml_set_name(a, "a");

        ggml_tensor * b;
        if (trans_b) {
            b = ggml_new_tensor_4d(ctx, type_b, k, n, bs[0] * nr[0], bs[1] * nr[1]);
            b = ggml_transpose(ctx, b);
        } else {
            b = ggml_new_tensor_4d(ctx, type_b, n, k, bs[0] * nr[0], bs[1] * nr[1]);
        }
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_out_prod(ctx, a, b);
        ggml_set_name(out, "out");

        return out;
    }
};

struct test_moe_fused: public test_case{
    const ggml_type type_input;
    const ggml_type type_ids;
    const ggml_type type_topk_weights;
    const ggml_type type_expert_down_weights;
    const ggml_type type_expert_up_weights;
    const int64_t   n_experts;
    const int64_t   n_topk;
    const int64_t   n_tokens;
    const int64_t   n_output_dims;
    const int64_t   n_k_dims;

    std::string vars() override {
        return VARS_TO_STR10(type_input, type_ids, type_topk_weights, type_expert_down_weights, type_expert_up_weights,
                             n_experts, n_topk, n_tokens, n_output_dims, n_k_dims);
    }

    virtual double max_nmse_err() override { return 1e-6; }

    test_moe_fused(ggml_type type_input = GGML_TYPE_F32, ggml_type type_ids = GGML_TYPE_I32,
                   ggml_type type_topk_weights = GGML_TYPE_F32, ggml_type type_expert_up_weights = GGML_TYPE_F32,
                   ggml_type type_expert_down_weights = GGML_TYPE_F32, int64_t n_experts = 4, int64_t n_topk = 2,
                   int64_t n_tokens = 10, int64_t n_output_dims = 8, int64_t n_k_dims = 8) :
        type_input(type_input),
        type_ids(type_ids),
        type_topk_weights(type_topk_weights),
        type_expert_down_weights(type_expert_down_weights),
        type_expert_up_weights(type_expert_up_weights),
        n_experts(n_experts),
        n_topk(n_topk),
        n_tokens(n_tokens),
        n_output_dims(n_output_dims),
        n_k_dims(n_k_dims) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // ggml_tensor * hhhhh = ggml_new_tensor_2ds
        ggml_tensor * input = ggml_new_tensor_3d(ctx, type_input, n_output_dims, 1, n_tokens);
        ggml_set_name(input, "input");

        // ggml_tensor * input_zero = ggml_new_tensor_3d(ctx, type_input, n_output_dims,1,n_tokens);
        // ggml_set_name(input_zero, "input_zero");

        ggml_tensor * ids = ggml_new_tensor_2d(ctx, type_ids, n_topk, n_tokens);
        ggml_set_name(ids, "ids");

        ggml_tensor * topk_weights = ggml_new_tensor_3d(ctx, type_topk_weights, 1, n_topk, n_tokens);
        ggml_set_name(topk_weights, "topk_weights");

        ggml_tensor * expert_down_weights =
            ggml_new_tensor_3d(ctx, type_expert_down_weights, n_k_dims, n_output_dims, n_experts);
        ggml_set_name(expert_down_weights, "expert_down_weights");

        ggml_tensor * expert_up_weights =
            ggml_new_tensor_3d(ctx, type_expert_up_weights, n_output_dims, n_k_dims, n_experts);
        ggml_set_name(expert_up_weights, "expert_up_weights");

        ggml_tensor * expert_gate_weights =
            ggml_new_tensor_3d(ctx, type_expert_up_weights, n_output_dims, n_k_dims, n_experts);
        ggml_set_name(expert_gate_weights, "expert_gate_weights");

        ggml_tensor * row_idx = ggml_arange(ctx, 0, n_topk * n_tokens, 1);
        ggml_set_name(row_idx, "row_idx");

        row_idx                     = ggml_reshape_2d(ctx, row_idx, n_tokens, n_topk);
        ggml_tensor * row_idx_int64 = ggml_cast(ctx, row_idx, GGML_TYPE_I32);
        ggml_tensor * out = ggml_moe_fused(ctx, input, ids, topk_weights, expert_up_weights, expert_down_weights,
                                           expert_gate_weights, row_idx_int64, 0, n_experts - 1);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device         rd;
        std::default_random_engine rng(rd());
        rng.seed(1);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) {
                continue;
            }
                std::vector<int32_t> data_init;
                for (int i = 0; i < n_experts; i++) {
                    data_init.push_back(i);
                }
                std::shuffle(data_init.begin(), data_init.end(), rng);

                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = data_init[r * t->ne[0] + i];
                    }
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else if (t->type == GGML_TYPE_F16) {
                float * data = (float *) malloc(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * sizeof(float));
                for (int i = 0; i < t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]; i++) {
                    data[i] = float(i % 10) / 20;
                }
                ggml_fp16_t * data_fp16 =
                    (ggml_fp16_t *) malloc(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * sizeof(ggml_fp16_t));
                ggml_fp32_to_fp16_row(data, data_fp16, t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]);
                ggml_backend_tensor_set(t, data_fp16, 0,
                                        t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * sizeof(ggml_fp16_t));
            } else {
                // init_tensor_uniform(t);
                float * data = (float *) malloc(t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * sizeof(float));
                for (int i = 0; i < t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]; i++) {
                    data[i] = float(i % 10) / 20;
                }
                ggml_backend_tensor_set(t, data, 0, t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * sizeof(float));
            }
        }
    }
};

// GGML_OP_SQR
struct test_sqr : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_sqr(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 }) : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_sqr(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * 0.25f * ne[0] * ne[1] * ne[2] * ne[3];  // 10% of expected value of sum.
    }
};

// GGML_OP_SQRT
struct test_sqrt : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_sqrt(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 3, 3, 2 }) : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_sqrt(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        // fill with positive values
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, 50.0f, 100.0f);
        }
    }

    float grad_eps() override { return 20.0f; }

    bool grad_precise() override { return true; }
};

// GGML_OP_LOG
struct test_log : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_log(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 }) : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_log(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // log(1) == 0, cluster values there to keep the sum low for better precision in the backward pass:
            init_tensor_uniform(t, 0.9f, 1.1f);
        }
    }

    bool grad_precise() override { return true; }
};

// GGML_OP_SIN
struct test_sin : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_sin(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 2, 2, 2 }) : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_sin(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -6.5f, 6.5f);  // Covers interval [-2*pi, 2*pi].
        }
    }

    double max_maa_err() override { return 1e-3; }

    float grad_eps() override { return 0.2f; }

    bool grad_precise() override { return true; }
};

// GGML_OP_COS
struct test_cos : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_cos(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 2, 2, 2 }) : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_cos(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -6.5f, 6.5f);  // Covers interval [-2*pi, 2*pi].
        }
    }

    double max_maa_err() override { return 1e-3; }

    float grad_eps() override { return 0.2f; }

    bool grad_precise() override { return true; }
};

// GGML_OP_CLAMP
struct test_clamp : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    float                        min;
    float                        max;

    std::string vars() override { return VARS_TO_STR4(type, ne, min, max); }

    test_clamp(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 }, float min = -0.5f,
               float max = 0.5f) :
        type(type),
        ne(ne),
        min(min),
        max(max) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_clamp(ctx, a, min, max);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override { return 1e-2f; }

    std::vector<float> grad_expect() override { return { 0.0f, 1.0f }; }
};

// GGML_OP_DIAG_MASK_INF
struct test_diag_mask_inf : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const int                    n_past;

    std::string vars() override { return VARS_TO_STR3(type, ne, n_past); }

    test_diag_mask_inf(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 10, 3, 2 }, int n_past = 5) :
        type(type),
        ne(ne),
        n_past(n_past) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_diag_mask_inf(ctx, a, n_past);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SOFT_MAX
struct test_soft_max : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const bool                   mask;
    const ggml_type              m_prec;
    const float                  scale;
    const float                  max_bias;

    std::string vars() override { return VARS_TO_STR6(type, ne, mask, m_prec, scale, max_bias); }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override { return 1e-6; }

    test_soft_max(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 }, bool mask = false,
                  ggml_type m_prec = GGML_TYPE_F32, float scale = 1.0f, float max_bias = 0.0f) :
        type(type),
        ne(ne),
        mask(mask),
        m_prec(m_prec),
        scale(scale),
        max_bias(max_bias) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * mask = nullptr;
        if (this->mask) {
            mask = ggml_new_tensor_2d(ctx, m_prec, ne[0], ne[1]);
            ggml_set_name(mask, "mask");
        }

        ggml_tensor * out = ggml_soft_max_ext(ctx, a, mask, scale, max_bias);
        ggml_set_name(out, "out");

        return out;
    }

    bool grad_precise() override { return true; }
};

// GGML_OP_SOFT_MAX_BACK
struct test_soft_max_back : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const float                  scale;
    const float                  max_bias;

    std::string vars() override { return VARS_TO_STR4(type, ne, scale, max_bias); }

    test_soft_max_back(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 }, float scale = 1.0f,
                       float max_bias = 0.0f) :
        type(type),
        ne(ne),
        scale(scale),
        max_bias(max_bias) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_soft_max_ext_back(ctx, a, b, scale, max_bias);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ROPE + GGML_OP_ROPE_BACK
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


// GGML_OP_GET_SLICE
struct test_get_slice : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    int64_t                      fr, to, axis;

    std::string vars() override {
        // forward can be inferred from the op, does not need to be printed
        return "TESTGETSLICE";
    }

    test_get_slice(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 2, 3, 1 }, int64_t fr = 2,
                   int64_t to = 5, int64_t axis = 0) :
        type(type),
        ne(ne),
        fr(fr),
        to(to),
        axis(axis) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, src);
        ggml_set_name(src, "src");

        ggml_tensor * out = ggml_get_slice(ctx, src, fr, to, axis);
        ggml_set_name(out, "out");

        return out;
    }

    virtual void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "src") == 0) {
                std::vector<float> data(ggml_nbytes(t) / sizeof(float));
                for (size_t i = 0; i < data.size(); i++) {
                    data[i] = i;
                }
                GGML_ASSERT(t->type == GGML_TYPE_F32);
                ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_POOL2D
struct test_pool2d : public test_case {
    enum ggml_op_pool            pool_type;
    const ggml_type              type_input;
    const std::array<int64_t, 4> ne_input;
    // kernel size
    const int                    k0;
    const int                    k1;
    // stride
    const int                    s0;
    const int                    s1;
    // padding
    const int                    p0;
    const int                    p1;

    std::string vars() override { return VARS_TO_STR9(pool_type, type_input, ne_input, k0, k1, s0, s1, p0, p1); }

    test_pool2d(ggml_op_pool pool_type = GGML_OP_POOL_AVG, ggml_type type_input = GGML_TYPE_F32,
                std::array<int64_t, 4> ne_input = { 10, 10, 3, 1 },  // [input_width, input_height, input_channels, 1]
                int k0 = 3, int k1 = 3, int s0 = 1, int s1 = 1, int p0 = 1, int p1 = 1) :
        pool_type(pool_type),
        type_input(type_input),
        ne_input(ne_input),
        k0(k0),
        k1(k1),
        s0(s0),
        s1(s1),
        p0(p0),
        p1(p1) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_set_param(ctx, input);
        ggml_set_name(input, "input");

        ggml_tensor * out = ggml_pool_2d(ctx, input, pool_type, k0, k1, s0, s1, p0, p1);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_CONV_TRANSPOSE_1D
struct test_conv_transpose_1d : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;

    const int s0;  // stride
    const int p0;  // padding
    const int d0;  // dilation

    std::string vars() override { return VARS_TO_STR5(ne_input, ne_kernel, s0, p0, d0); }

    test_conv_transpose_1d(
        std::array<int64_t, 4> ne_input  = { 197, 32, 1, 1 },  // [input_width, input_height, input_channels, 1]
        std::array<int64_t, 4> ne_kernel = { 16, 32, 32, 1 },  // [kernel_width, kernel_height, input_channels, 1]
        int s0 = 1, int p0 = 0, int d0 = 1) :
        ne_input(ne_input),
        ne_kernel(ne_kernel),
        s0(s0),
        p0(p0),
        d0(d0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, input, s0, p0, d0);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_IM2COL
struct test_im2col : public test_case {
    const ggml_type              type_input;
    const ggml_type              type_kernel;
    const ggml_type              dst_type;
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    // stride
    const int                    s0;
    const int                    s1;
    // padding
    const int                    p0;
    const int                    p1;
    // dilation
    const int                    d0;
    const int                    d1;
    // mode
    const bool                   is_2D;

    std::string vars() override {
        return VARS_TO_STR12(type_input, type_kernel, dst_type, ne_input, ne_kernel, s0, s1, p0, p1, d0, d1, is_2D);
    }

    test_im2col(ggml_type type_input = GGML_TYPE_F32, ggml_type type_kernel = GGML_TYPE_F16,
                ggml_type              dst_type  = GGML_TYPE_F32,
                std::array<int64_t, 4> ne_input  = { 10, 10, 3, 1 },  // [input_width, input_height, input_channels, 1]
                std::array<int64_t, 4> ne_kernel = { 3, 3, 3, 1 },  // [kernel_width, kernel_height, input_channels, 1]
                int s0 = 1, int s1 = 1, int p0 = 1, int p1 = 1, int d0 = 1, int d1 = 1, bool is_2D = true) :
        type_input(type_input),
        type_kernel(type_kernel),
        dst_type(dst_type),
        ne_input(ne_input),
        ne_kernel(ne_kernel),
        s0(s0),
        s1(s1),
        p0(p0),
        p1(p1),
        d0(d0),
        d1(d1),
        is_2D(is_2D) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_set_param(ctx, input);
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_im2col(ctx, kernel, input, s0, s1, p0, p1, d0, d1, is_2D, dst_type);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_CONCAT
struct test_concat : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne_a;
    const int64_t                ne_b_d;
    const int                    dim;
    const int                    v;  // view (1 << 0: non-cont a, 1 << 1: non-cont b)

    std::string vars() override { return VARS_TO_STR5(type, ne_a, ne_b_d, dim, v); }

    test_concat(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne_a = { 10, 5, 5, 5 }, int64_t ne_b_d = 5,
                int dim = 2, int v = 0) :
        type(type),
        ne_a(ne_a),
        ne_b_d(ne_b_d),
        dim(dim),
        v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        auto ne_b = ne_a;
        ne_b[dim] = ne_b_d;
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a;
            ne[0] *= 2;
            ne[1] *= 4;
            ne[2] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_name(a, "a");
        }
        ggml_tensor * b;
        if (v & 2) {
            auto ne = ne_b;
            ne[0] *= 3;
            ne[1] *= 2;
            ne[2] *= 4;
            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(b, "b");

            b = ggml_view_4d(ctx, b, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b->nb[1], b->nb[2], b->nb[3], 0);
            ggml_set_name(b, "view_of_b");
        } else {
            b = ggml_new_tensor(ctx, type, 4, ne_b.data());
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_concat(ctx, a, b, dim);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SCATTER_UPDATE
struct test_set_slice : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne_var;      // 变量张量形状
    const std::array<int64_t, 1> ne_indices;  // 索引张量形状
    const std::array<int64_t, 4> ne_updates;  // 更新张量形状
    const int                    fr;
    const int                    to;

    std::string vars() override { return "TESTSETSLICE"; }

    /*
    ScatterUpdate操作语义：
    var: [3, 8, 1, 1]  - 要被更新的张量
    indices: [2] - 索引张量，包含3个索引值[1, 3]
    updates: [3, 2, 1, 1] - 更新值张量
    操作：var[indices] = updates
    */

    test_set_slice(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne_var = { 3, 8, 1, 1 },     // 主张量：5行10列
                   std::array<int64_t, 1> ne_indices = { 2 },     // [1,3]                                      // 索引：2个位置
                   std::array<int64_t, 4> ne_updates = { 3, 2, 1, 1 },  // 更新值：3行2列
                   int fr = 2, int to = 4) :
        type(type),
        ne_var(ne_var),
        ne_indices(ne_indices),
        ne_updates(ne_updates),
        fr(fr),
        to(to) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // 创建变量张量（要被更新的张量）
        ggml_tensor * var = ggml_new_tensor(ctx, type, 4, ne_var.data());
        ggml_set_param(ctx, var);
        ggml_set_name(var, "var");

        // 创建索引张量
        ggml_tensor * indices = ggml_new_tensor(ctx, GGML_TYPE_I64, 1, ne_indices.data());
        ggml_set_name(indices, "indices");

        // 创建更新值张量
        ggml_tensor * updates = ggml_new_tensor(ctx, type, 4, ne_updates.data());
        ggml_set_name(updates, "updates");

        // 创建SET_SLICE操作
        ggml_tensor * out = ggml_scatter_update(ctx, var, indices, updates);
        ggml_set_name(out, "out");

        return out;
    }

    virtual void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "var") == 0) {
                // 初始化变量张量 - 使用连续的值
                std::vector<float> data(ggml_nbytes(t) / sizeof(float));
                for (size_t i = 0; i < data.size(); i++) {
                    data[i] = i * 0.1f;  // 0.0, 0.1, 0.2, 0.3, ...
                }

                GGML_ASSERT(t->type == GGML_TYPE_F32);
                ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
            } else if (strcmp(t->name, "indices") == 0) {
                // 初始化索引张量 - 指定要更新的位置：[0, 2, 4]
                std::vector<int64_t> indices_data;
                // size_t               num_indices = ggml_nbytes(t) / sizeof(int64_t);

                // 设置要更新的索引位置（0, 2, 4）
                // for (size_t i = 0; i < num_indices; i++) {
                    // indices_data.push_back(static_cast<int64_t>(i+1 ));  // 0, 2, 4
                // }
                for (int i = fr; i < to; i++) {
                    indices_data.push_back(static_cast<int64_t>(i - 1));
                }

                // GGML_ASSERT(t->type == GGML_TYPE_I32);
                ggml_backend_tensor_set(t, indices_data.data(), 0, ggml_nbytes(t));
            } else if (strcmp(t->name, "updates") == 0) {
                // 初始化更新值张量 - 使用不同的值模式
                std::vector<float> data(ggml_nbytes(t) / sizeof(float));
                for (size_t i = 0; i < data.size(); i++) {
                    data[i] = 100.0f + i;  // 100.0, 101.0, 102.0, ...
                }
                GGML_ASSERT(t->type == GGML_TYPE_F32);
                ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
            } else {
                // 其他张量使用默认初始化
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_ARGSORT
struct test_argsort : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    ggml_sort_order              order;

    std::string vars() override { return VARS_TO_STR3(type, ne, order); }

    test_argsort(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 16, 10, 10, 10 },
                 ggml_sort_order order = GGML_SORT_ORDER_ASC) :
        type(type),
        ne(ne),
        order(order) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_argsort(ctx, a, order);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device         rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // indices
                std::vector<int> data(ggml_nelements(t));
                for (int i = 0; i < ggml_nelements(t); i++) {
                    data[i] = rand();
                }
                std::shuffle(data.begin(), data.end(), rng);
                ggml_backend_tensor_set(t, data.data(), 0, ne[0] * ne[1] * ne[2] * ne[3] * sizeof(int));
            } else if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                GGML_ABORT("fatal error");
            }
        }
    }
};

// GGML_OP_SUM
struct test_sum : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_sum(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 }) : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_sum(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override { return 0.1f * sqrtf(ne[0] * ne[1] * ne[2] * ne[3]); }
};

// GGML_OP_SUM_ROWS
struct test_sum_rows : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_sum_rows(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 }) : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_sum_rows(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_MEAN
struct test_mean : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_mean(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 }) : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_mean(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override { return 0.1f * ne[0] * ne[1] * ne[2] * ne[3]; }
};

// GGML_OP_UPSCALE
struct test_upscale : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const int32_t                scale_factor;
    const bool                   transpose;

    std::string vars() override { return VARS_TO_STR4(type, ne, scale_factor, transpose); }

    test_upscale(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 512, 512, 3, 1 },
                 int32_t scale_factor = 2, bool transpose = false) :
        type(type),
        ne(ne),
        scale_factor(scale_factor),
        transpose(transpose) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        if (transpose) {
            a = ggml_transpose(ctx, a);
            ggml_set_name(a, "a_transposed");
        }

        ggml_tensor * out = ggml_upscale(ctx, a, scale_factor);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_UPSCALE (ext)
struct test_upscale_ext : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> ne_tgt;

    std::string vars() override { return VARS_TO_STR3(type, ne, ne_tgt); }

    test_upscale_ext(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 2, 5, 7, 11 },
                     std::array<int64_t, 4> ne_tgt = { 5, 7, 11, 13 }) :
        type(type),
        ne(ne),
        ne_tgt(ne_tgt) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_upscale_ext(ctx, a, ne_tgt[0], ne_tgt[1], ne_tgt[2], ne_tgt[3]);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_GROUP_NORM
struct test_group_norm : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const int32_t                num_groups;
    const float                  eps;

    std::string vars() override { return VARS_TO_STR4(type, ne, num_groups, eps); }

    test_group_norm(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 64, 64, 320, 1 },
                    int32_t num_groups = 32, float eps = 1e-6f) :
        type(type),
        ne(ne),
        num_groups(num_groups),
        eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_group_norm(ctx, a, num_groups, eps);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ACC
struct test_acc : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;

    std::string vars() override { return VARS_TO_STR3(type, ne_a, ne_b); }

    test_acc(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne_a = { 256, 17, 1, 1 },
             std::array<int64_t, 4> ne_b = { 256, 16, 1, 1 }) :
        type(type),
        ne_a(ne_a),
        ne_b(ne_b) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne_b.data());
        ggml_set_param(ctx, b);
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_acc(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], b->nb[1]);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_PAD
struct test_pad : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne_a;
    const int                    pad_0;
    const int                    pad_1;

    std::string vars() override { return VARS_TO_STR4(type, ne_a, pad_0, pad_1); }

    test_pad(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne_a = { 2, 2, 1, 1 }, int pad_0 = 1,
             int pad_1 = 1) :
        type(type),
        ne_a(ne_a),
        pad_0(pad_0),
        pad_1(pad_1) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_pad(ctx, a, pad_0, pad_1, 0, 0);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_PAD_REFLECT_1D
struct test_pad_reflect_1d : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne_a;
    const int                    pad_0;
    const int                    pad_1;

    std::string vars() override { return VARS_TO_STR4(type, ne_a, pad_0, pad_1); }

    test_pad_reflect_1d(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne_a = { 512, 34, 2, 1 }, int pad_0 = 10,
                        int pad_1 = 9) :
        type(type),
        ne_a(ne_a),
        pad_0(pad_0),
        pad_1(pad_1) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 2, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_pad_reflect_1d(ctx, a, pad_0, pad_1);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ARANGE
struct test_arange : public test_case {
    const ggml_type type;
    const float     start;
    const float     stop;
    const float     step;

    std::string vars() override { return VARS_TO_STR4(type, start, stop, step); }

    test_arange(ggml_type type = GGML_TYPE_F32, float start = 0.f, float stop = 10.f, float step = 1.f) :
        type(type),
        start(start),
        stop(stop),
        step(step) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * out = ggml_arange(ctx, start, stop, step);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_TIMESTEP_EMBEDDING
struct test_timestep_embedding : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne_a;
    const int                    dim;
    const int                    max_period;

    std::string vars() override { return VARS_TO_STR4(type, ne_a, dim, max_period); }

    test_timestep_embedding(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne_a = { 2, 1, 1, 1 }, int dim = 320,
                            int max_period = 10000) :
        type(type),
        ne_a(ne_a),
        dim(dim),
        max_period(max_period) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_timestep_embedding(ctx, a, dim, max_period);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_LEAKY_RELU
struct test_leaky_relu : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne_a;
    const float                  negative_slope;

    std::string vars() override { return VARS_TO_STR3(type, ne_a, negative_slope); }

    test_leaky_relu(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne_a = { 10, 5, 4, 3 },
                    float negative_slope = 0.1f) :
        type(type),
        ne_a(ne_a),
        negative_slope(negative_slope) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_leaky_relu(ctx, a, negative_slope, true);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_FLASH_ATTN_EXT
struct test_flash_attn_ext : public test_case {
    const int64_t hs;           // head size
    const int64_t nh;           // num heads
    const int64_t nr;           // repeat in Q, tests for grouped-query attention
    const int64_t kv;           // kv size
    const int64_t nb;           // batch size

    const bool mask;            // use mask

    const float max_bias;       // ALiBi
    const float logit_softcap;  // Gemma 2

    const ggml_type        type_KV;
    std::array<int32_t, 4> permute;

    std::string vars() override {
        return VARS_TO_STR10(hs, nh, nr, kv, nb, mask, max_bias, logit_softcap, type_KV, permute);
    }

    double max_nmse_err() override { return 5e-4; }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        // Just counting matmul costs:
        // Q*K^T is nb x hs x kv, P*V is nb x kv x hs, per head
        return 2 * 2 * nh * nr * nb * hs * kv;
    }

    test_flash_attn_ext(int64_t hs = 128, int64_t nh = 32, int64_t nr = 1, int64_t kv = 96, int64_t nb = 8,
                        bool mask = true, float max_bias = 0.0f, float logit_softcap = 0.0f,
                        ggml_type type_KV = GGML_TYPE_F16, std::array<int32_t, 4> permute = { 0, 1, 2, 3 }) :
        hs(hs),
        nh(nh),
        nr(nr),
        kv(kv),
        nb(nb),
        mask(mask),
        max_bias(max_bias),
        logit_softcap(logit_softcap),
        type_KV(type_KV),
        permute(permute) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t hs_padded = GGML_PAD(hs, ggml_blck_size(type_KV));

        const auto & create_permuted = [&](ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2,
                                           int64_t ne3) -> ggml_tensor * {
            int64_t ne[4] = { ne0, ne1, ne2, ne3 };
            int64_t ne_perm[4];
            for (int i = 0; i < 4; ++i) {
                ne_perm[permute[i]] = ne[i];
            }
            ggml_tensor * t = ggml_new_tensor_4d(ctx, type, ne_perm[0], ne_perm[1], ne_perm[2], ne_perm[3]);
            if (permute != std::array<int32_t, 4>{ 0, 1, 2, 3 }) {
                t = ggml_permute(ctx, t, permute[0], permute[1], permute[2], permute[3]);
            }
            return t;
        };

        ggml_tensor * q = create_permuted(GGML_TYPE_F32, hs_padded, nb, nh * nr, 1);
        ggml_set_name(q, "q");

        ggml_tensor * k = create_permuted(type_KV, hs_padded, kv, nh, 1);
        ggml_set_name(k, "k");

        ggml_tensor * v = create_permuted(type_KV, hs_padded, kv, nh, 1);
        ggml_set_name(v, "v");

        ggml_tensor * m = nullptr;
        if (mask) {
            m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, 1);
            ggml_set_name(m, "m");
        }

        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf(hs), max_bias, logit_softcap);
        ggml_set_name(out, "out");

        return out;
    }

    bool grad_precise() override { return true; }
};


// GGML_OP_CROSS_ENTROPY_LOSS_BACK
struct test_cross_entropy_loss_back : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_cross_entropy_loss_back(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 10, 5, 4, 3 }) :
        type(type),
        ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * grad = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(grad, "grad");

        ggml_tensor * logits = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(logits, "logits");

        ggml_tensor * labels = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(labels, "labels");

        // Ensure labels add up to 1:
        labels = ggml_soft_max(ctx, labels);
        ggml_set_name(labels, "labels_normalized");

        ggml_tensor * out = ggml_cross_entropy_loss_back(ctx, grad, logits, labels);
        ggml_set_name(out, "out");

        return out;
    }
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
};

struct llama_hparams {
    uint32_t                  n_vocab;
    uint32_t                  n_embd;
    uint32_t                  n_head;
    uint32_t                  n_head_kv;
    static constexpr uint32_t n_layer = 1;
    uint32_t                  n_rot;
    uint32_t                  n_embd_head;  // dimension of values (d_v)
    uint32_t                  n_ff;

    float f_norm_eps;
    float f_norm_rms_eps;

    // cparams
    static constexpr uint32_t n_ctx      = 512;  // user-specified context size
    static constexpr uint32_t n_ctx_orig = n_ctx;

    // batch
    int32_t n_tokens;

    // llm_build_context
    static constexpr int32_t n_kv    = 32;  // size of KV cache to consider (n_kv <= n_ctx
    static constexpr int32_t kv_head = 1;   // index of where we store new KV data in the cache

    uint32_t n_embd_gqa() const {           // dimension of key embeddings across all k-v heads
        return n_embd_head * n_head_kv;
    }
};

// LLM base class
struct test_llm : public test_case {
    llama_hparams hp;

  protected:
    test_llm(llama_hparams hp) : hp(std::move(hp)) {}

  public:
    struct ggml_tensor * llm_build_norm(struct ggml_context * ctx, struct ggml_tensor * cur, struct ggml_tensor * mw,
                                        struct ggml_tensor * mb, llm_norm_type type) {
        switch (type) {
            case LLM_NORM:
                cur = ggml_norm(ctx, cur, hp.f_norm_eps);
                break;
            case LLM_NORM_RMS:
                cur = ggml_rms_norm(ctx, cur, hp.f_norm_rms_eps);
                break;
        }
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cur = ggml_add(ctx, cur, mb);
        }
        return cur;
    }

    void llm_build_kv_store(struct ggml_context * ctx, struct ggml_tensor * k_l, struct ggml_tensor * v_l,
                            struct ggml_tensor * k_cur, struct ggml_tensor * v_cur) {
        // compute the transposed [n_tokens, n_embd] V matrix
        struct ggml_tensor * v_cur_t = ggml_transpose(ctx, ggml_reshape_2d(ctx, v_cur, hp.n_embd_gqa(), hp.n_tokens));

        struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, k_l, hp.n_tokens * hp.n_embd_gqa(),
                                                         (ggml_row_size(k_l->type, hp.n_embd_gqa())) *hp.kv_head);

        struct ggml_tensor * v_cache_view =
            ggml_view_2d(ctx, v_l, hp.n_tokens, hp.n_embd_gqa(), (hp.n_ctx) * ggml_element_size(v_l),
                         (hp.kv_head) * ggml_element_size(v_l));

        // important: storing RoPE-ed version of K in the KV cache!
        ggml_cpy(ctx, k_cur, k_cache_view);
        ggml_cpy(ctx, v_cur_t, v_cache_view);
    }

    struct ggml_tensor * llm_build_kqv(struct ggml_context * ctx, struct ggml_tensor * k_l, struct ggml_tensor * v_l,
                                       struct ggml_tensor * q_cur, struct ggml_tensor * kq_mask, float kq_scale) {
        struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);

        struct ggml_tensor * k =
            ggml_view_3d(ctx, k_l, hp.n_embd_head, hp.n_kv, hp.n_head_kv, ggml_row_size(k_l->type, hp.n_embd_gqa()),
                         ggml_row_size(k_l->type, hp.n_embd_head), 0);

        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);

        // split cached v into n_head heads
        struct ggml_tensor * v =
            ggml_view_3d(ctx, v_l, hp.n_kv, hp.n_embd_head, hp.n_head_kv, ggml_element_size(v_l) * hp.n_ctx,
                         ggml_element_size(v_l) * hp.n_ctx * hp.n_embd_head, 0);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);

        struct ggml_tensor * cur = ggml_cont_2d(ctx, kqv_merged, hp.n_embd_head * hp.n_head, hp.n_tokens);

        struct ggml_tensor * wo = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
        cur                     = ggml_mul_mat(ctx, wo, cur);

        return cur;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                std::vector<int> data(hp.n_tokens);
                for (int i = 0; i < hp.n_tokens; i++) {
                    data[i] = rand() % hp.n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, hp.n_tokens * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// Llama
struct test_llama : public test_llm {
    static constexpr float freq_base   = 10000.0f;
    static constexpr float freq_scale  = 1.0f;
    static constexpr float ext_factor  = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast   = 32.0f;
    static constexpr float beta_slow   = 1.0f;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "LLAMA";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return VARS_TO_STR1(n_tokens);
    }

    double max_nmse_err() override { return 2e-3; }

    test_llama(int n_tokens = 1) :
        test_llm({
            /*n_vocab        =*/32000,
            /*n_embd         =*/3200,
            /*n_head         =*/32,
            /*n_head_kv      =*/32,
            /*n_rot          =*/100,
            /*n_embd_head    =*/100,
            /*n_ff           =*/8640,
            /*f_norm_eps     =*/0.f,
            /*f_norm_rms_eps =*/1e-5f,
            /*n_tokens       =*/n_tokens,
        }) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.n_kv, hp.n_tokens, 1);

        ggml_tensor * k_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);
        ggml_tensor * v_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            ggml_tensor * attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            cur                     = llm_build_norm(ctx, inpL, attn_norm, nullptr, LLM_NORM_RMS);

            // self-attention
            {
                ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
                ggml_tensor * wk = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd_gqa());
                ggml_tensor * wv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd_gqa());

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx, wq, cur);
                struct ggml_tensor * Kcur = ggml_mul_mat(ctx, wk, cur);
                struct ggml_tensor * Vcur = ggml_mul_mat(ctx, wv, cur);

                Qcur = ggml_rope_ext(ctx, ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head, hp.n_tokens), inp_pos,
                                     nullptr, hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale, ext_factor,
                                     attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx, ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens),
                                     inp_pos, nullptr, hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale, ext_factor,
                                     attn_factor, beta_fast, beta_slow);

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f / sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);

            // feed-forward network
            ggml_tensor * ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            cur                    = llm_build_norm(ctx, ffn_inp, ffn_norm, nullptr, LLM_NORM_RMS);

            ggml_tensor *        ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            ggml_tensor *        ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_ff, hp.n_embd);
            ggml_tensor *        ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            struct ggml_tensor * tmp      = ggml_mul_mat(ctx, ffn_up, cur);
            cur                           = ggml_mul_mat(ctx, ffn_gate, cur);
            cur                           = ggml_silu(ctx, cur);
            cur                           = ggml_mul(ctx, cur, tmp);
            cur                           = ggml_mul_mat(ctx, ffn_down, cur);

            cur = ggml_add(ctx, cur, ffn_inp);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor * output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        cur                       = llm_build_norm(ctx, cur, output_norm, nullptr, LLM_NORM_RMS);

        // lm_head
        ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_vocab);
        cur                  = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};

// Falcon
struct test_falcon : public test_llm {
    static constexpr float freq_base   = 10000.0f;
    static constexpr float freq_scale  = 1.0f;
    static constexpr float ext_factor  = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast   = 32.0f;
    static constexpr float beta_slow   = 1.0f;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "FALCON";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return VARS_TO_STR1(n_tokens);
    }

    double max_nmse_err() override { return 2e-3; }

    test_falcon(int n_tokens = 1) :
        test_llm({
            /*n_vocab        =*/32000,
            /*n_embd         =*/3200,
            /*n_head         =*/50,
            /*n_head_kv      =*/1,
            /*n_rot          =*/64,
            /*n_embd_head    =*/64,
            /*n_ff           =*/8640,
            /*f_norm_eps     =*/1e-5f,
            /*f_norm_rms_eps =*/0.f,
            /*n_tokens       =*/n_tokens,
        }) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.n_kv, hp.n_tokens, 1);

        ggml_tensor * k_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);
        ggml_tensor * v_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            // norm
            ggml_tensor * attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            ggml_tensor * attn_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            ggml_tensor * attn_norm   = llm_build_norm(ctx, inpL, attn_norm_w, attn_norm_b, LLM_NORM);

            // self-attention
            {
                cur = attn_norm;

                ggml_tensor * wqkv =
                    ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd + 2 * hp.n_embd_gqa());

                cur = ggml_mul_mat(ctx, wqkv, cur);

                struct ggml_tensor * Qcur = ggml_cont(
                    ctx, ggml_view_2d(ctx, cur, hp.n_embd, hp.n_tokens, cur->nb[1], 0 * sizeof(float) * (hp.n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens,
                                                                        cur->nb[1], 1 * sizeof(float) * (hp.n_embd)));
                struct ggml_tensor * Vcur =
                    ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens, cur->nb[1],
                                                1 * sizeof(float) * (hp.n_embd + hp.n_embd_gqa())));

                Qcur = ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head, hp.n_tokens);
                Kcur = ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f / sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor * ffn_inp = cur;

            // feed forward
            {
                ggml_tensor * ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
                ggml_tensor * ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_ff, hp.n_embd);
                cur                    = attn_norm;
                cur                    = ggml_mul_mat(ctx, ffn_up, cur);
                cur                    = ggml_gelu(ctx, cur);
                cur                    = ggml_mul_mat(ctx, ffn_down, cur);
            }

            cur = ggml_add(ctx, cur, ffn_inp);

            cur = ggml_add(ctx, cur, inpL);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor * output_norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        ggml_tensor * output_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        cur                         = llm_build_norm(ctx, cur, output_norm, output_norm_b, LLM_NORM);

        // lm_head
        ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, hp.n_embd, hp.n_vocab);
        cur                  = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};

struct test_mixture_ops : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override { return 1e-6; }

    test_mixture_ops(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 1024, 1024, 32, 32 }) :
        type(type),
        ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        // FFN
        std::array<int64_t, 4> linear_1_shape = { ne[0], ne[0] * 4, 1, 1 };
        ggml_tensor *          linear_1       = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_1, "linear_1");
        std::array<int64_t, 4> linear_2_shape = { ne[0] * 4, ne[0], 1, 1 };
        ggml_tensor *          linear_2       = ggml_new_tensor(ctx, type, 4, linear_2_shape.data());
        ggml_set_name(linear_2, "linear_2");
        ggml_tensor * hidden  = ggml_mul_mat(ctx, linear_1, a);
        ggml_tensor * ffn_out = ggml_mul_mat(ctx, linear_2, hidden);
        ggml_set_name(ffn_out, "ffn_out");

        return ffn_out;

        ggml_tensor * tmp = ffn_out;

        ggml_tensor * out = ggml_view_4d(ctx, tmp, tmp->ne[0], tmp->ne[1], 1, 1, tmp->nb[1], tmp->nb[2], tmp->nb[3], 0);
        ggml_tensor * rms_norm = ggml_rms_norm(ctx, out, 1e-6f);
        ggml_set_name(rms_norm, "rms_norm");

        // 残差连接
        ggml_tensor * add = ggml_add(ctx, out, rms_norm);
        ggml_set_name(add, "add");

        // 应用SiLU激活函数
        ggml_tensor * silu = ggml_silu(ctx, add);
        ggml_set_name(silu, "silu");

        // 乘法
        ggml_tensor * mul = ggml_mul(ctx, silu, rms_norm);
        ggml_set_name(mul, "mul");

        // 残差连接
        ggml_tensor * add2 = ggml_add(ctx, out, mul);
        ggml_set_name(add2, "add2");

        ggml_tensor * soft_max = ggml_soft_max(ctx, out);
        ggml_set_name(soft_max, "soft_max");

        return soft_max;
    }
};

struct test_mixture_ops_20_matmul : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override { return 1e-6; }

    test_mixture_ops_20_matmul(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 1024, 1024, 8, 8 }) :
        type(type),
        ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        // 构建 20 个 matmul 节点
        std::array<int64_t, 4> linear_1_shape = { ne[0], ne[0], 1, 1 };
        ggml_tensor *          linear_1       = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_1, "linear_1");
        ggml_tensor * linear_2 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_2, "linear_2");

        ggml_tensor * linear_3 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_3, "linear_3");

        ggml_tensor * linear_4 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_4, "linear_4");

        ggml_tensor * linear_5 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_5, "linear_5");

        ggml_tensor * linear_6 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_6, "linear_6");

        ggml_tensor * linear_7 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_7, "linear_7");

        ggml_tensor * linear_8 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_8, "linear_8");

        ggml_tensor * linear_9 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_9, "linear_9");

        ggml_tensor * linear_10 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_10, "linear_10");

        ggml_tensor * linear_11 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_11, "linear_11");

        ggml_tensor * linear_12 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_12, "linear_12");

        ggml_tensor * linear_13 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_13, "linear_13");

        ggml_tensor * linear_14 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_14, "linear_14");

        ggml_tensor * linear_15 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_15, "linear_15");

        ggml_tensor * linear_16 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_16, "linear_16");

        ggml_tensor * linear_17 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_17, "linear_17");

        ggml_tensor * linear_18 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_18, "linear_18");

        ggml_tensor * linear_19 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_19, "linear_19");

        ggml_tensor * linear_20 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_20, "linear_20");

        ggml_tensor * hidden   = ggml_mul_mat(ctx, linear_1, a);
        ggml_tensor * hidden2  = ggml_mul_mat(ctx, linear_2, hidden);
        ggml_tensor * hidden3  = ggml_mul_mat(ctx, linear_3, hidden2);
        ggml_tensor * hidden4  = ggml_mul_mat(ctx, linear_4, hidden3);
        ggml_tensor * hidden5  = ggml_mul_mat(ctx, linear_5, hidden4);
        ggml_tensor * hidden6  = ggml_mul_mat(ctx, linear_6, hidden5);
        ggml_tensor * hidden7  = ggml_mul_mat(ctx, linear_7, hidden6);
        ggml_tensor * hidden8  = ggml_mul_mat(ctx, linear_8, hidden7);
        ggml_tensor * hidden9  = ggml_mul_mat(ctx, linear_9, hidden8);
        ggml_tensor * hidden10 = ggml_mul_mat(ctx, linear_10, hidden9);
        ggml_tensor * hidden11 = ggml_mul_mat(ctx, linear_11, hidden10);
        ggml_tensor * hidden12 = ggml_mul_mat(ctx, linear_12, hidden11);
        ggml_tensor * hidden13 = ggml_mul_mat(ctx, linear_13, hidden12);
        ggml_tensor * hidden14 = ggml_mul_mat(ctx, linear_14, hidden13);
        ggml_tensor * hidden15 = ggml_mul_mat(ctx, linear_15, hidden14);
        ggml_tensor * hidden16 = ggml_mul_mat(ctx, linear_16, hidden15);
        ggml_tensor * hidden17 = ggml_mul_mat(ctx, linear_17, hidden16);
        ggml_tensor * hidden18 = ggml_mul_mat(ctx, linear_18, hidden17);
        ggml_tensor * hidden19 = ggml_mul_mat(ctx, linear_19, hidden18);
        ggml_tensor * hidden20 = ggml_mul_mat(ctx, linear_20, hidden19);

        return hidden20;
    }
};

struct test_mixture_ops_2_matmul : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override { return 1e-6; }

    test_mixture_ops_2_matmul(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 1024, 1024, 8, 8 }) :
        type(type),
        ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        // 构建 100 个 matmul 节点
        std::array<int64_t, 4> linear_1_shape = { ne[0], ne[0], 1, 1 };
        ggml_tensor *          linear_1       = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_1, "linear_1");
        ggml_tensor * linear_2 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_2, "linear_2");

        ggml_tensor * hidden  = ggml_mul_mat(ctx, linear_1, a);
        ggml_tensor * hidden2 = ggml_mul_mat(ctx, linear_2, hidden);

        return hidden2;
    }
};

struct test_mixture_ops_5_matmul_fast : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override { return 1e-6; }

    test_mixture_ops_5_matmul_fast(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 512, 512, 8, 8 }) :
        type(type),
        ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        // 构建 100 个 matmul 节点
        std::array<int64_t, 4> linear_1_shape = { ne[0], ne[0], 1, 1 };
        ggml_tensor *          linear_1       = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_1, "linear_1");
        ggml_tensor * linear_2 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_2, "linear_2");

        ggml_tensor * linear_3 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_3, "linear_3");

        ggml_tensor * linear_4 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_4, "linear_4");

        ggml_tensor * linear_5 = ggml_new_tensor(ctx, type, 4, linear_1_shape.data());
        ggml_set_name(linear_5, "linear_5");

        ggml_tensor * hidden  = ggml_mul_mat(ctx, linear_1, a);
        ggml_tensor * hidden2 = ggml_mul_mat(ctx, linear_2, hidden);
        ggml_tensor * hidden3 = ggml_mul_mat(ctx, linear_3, hidden2);
        ggml_tensor * hidden4 = ggml_mul_mat(ctx, linear_4, hidden3);
        ggml_tensor * hidden5 = ggml_mul_mat(ctx, linear_5, hidden4);

        return hidden5;
    }
};

struct test_mixture_ops_bug : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override { return 1e-6; }

    test_mixture_ops_bug(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = { 12, 15, 1, 1 }) :
        type(type),
        ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(ctx, a);
        ggml_set_name(a, "a");

        // RMSNorm
        ggml_tensor * rms_norm = ggml_rms_norm(ctx, a, 1e-6f);
        ggml_set_name(rms_norm, "rms_norm");

        // 乘法
        ggml_tensor * mul = ggml_mul(ctx, a, rms_norm);
        ggml_set_name(mul, "mul");

        return mul;
    }
};

struct test_deepseek_case2 : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne_a;
    int                          seedw;
    int                          seedt;
    std::vector<ggml_tensor *>   nodes_to_expand;
    float                        f_norm_rms_eps;

    int n_head;
    int n_embd;
    int n_embd_head_k;
    int n_embd_head_v;
    int n_embd_head_qk_rope;
    int n_embd_head_qk_nope;
    int kv_lora_rank;
    int n_embd_k_gqa;
    int n_embd_v_gqa;
    int n_head_kv;

    // rum_time
    int n_ctx;
    int n_tokens;
    int n_kv;

    ggml_tensor * attn_norm;
    ggml_tensor * attn_kv_a_norm;
    ggml_tensor * wq;
    ggml_tensor * wkv_a_mqa;
    ggml_tensor * wkv_b;
    ggml_tensor * wo;

    // kv cacche
    ggml_tensor * k_l;
    ggml_tensor * v_l;

    double max_nmse_err() override { return 1e-5; }

    std::string vars() override {
        // forward can be inferred from the op, does not need to be printed
        return "TESTDEEPSEEK2";
    }

    test_deepseek_case2(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne_a = { 32, 4, 1, 1 }, int seedw = 514,
                        int seedt = 233) :
        type(type),
        ne_a(ne_a),
        seedw(seedw),
        seedt(seedt) {
        n_ctx               = 128;
        n_tokens            = 4;
        n_head              = 16;
        n_head_kv           = n_head;
        n_kv                = 64;
        n_embd_head_v       = 128;
        n_embd_head_k       = 192;
        n_embd_v_gqa        = n_embd_head_v * n_head;
        n_embd_k_gqa        = n_embd_head_k * n_head;
        kv_lora_rank        = 64;
        n_embd_head_qk_nope = 128;
        n_embd_head_qk_rope = 64;
        n_embd              = 32;
        f_norm_rms_eps      = 1e-5;
    }

    ggml_tensor * build_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * mw) const {
        cur = ggml_rms_norm(ctx, cur, f_norm_rms_eps);
        cur = ggml_mul(ctx, cur, mw);
        return cur;
    }

    void llm_build_kv_store(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * v_cur) {
        struct ggml_tensor * k_cache_view =
            ggml_view_1d(ctx, k_l, n_tokens * n_embd_k_gqa, ggml_row_size(k_l->type, n_embd_k_gqa) * n_kv);
        nodes_to_expand.push_back(ggml_cpy(ctx, k_cur, k_cache_view));
        struct ggml_tensor * v_cache_view = ggml_view_2d(
            ctx, v_l, n_tokens, n_embd_v_gqa, (n_ctx) *ggml_element_size(v_l), (n_kv) *ggml_element_size(v_l));
        v_cur = ggml_transpose(ctx, v_cur);
        ggml_set_name(v_cache_view, "v_cache_view");
        nodes_to_expand.push_back(ggml_cpy(ctx, v_cur, v_cache_view));
    }

    ggml_tensor * llm_build_kqv(ggml_context * ctx, ggml_tensor * q_cur, ggml_tensor * kq_mask) {
        struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
        ggml_set_name(q, "q");

        struct ggml_tensor * k =
            ggml_view_3d(ctx, k_l, n_embd_head_k, n_kv, n_head_kv, ggml_row_size(k_l->type, n_embd_k_gqa),
                         ggml_row_size(k_l->type, n_embd_head_k), 0);
        ggml_set_name(k, "k");
        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        ggml_set_name(kq, "kq");
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq                     = ggml_soft_max_ext(ctx, kq, kq_mask, 1.0, 0.0);
        struct ggml_tensor * v = ggml_view_3d(ctx, v_l, n_kv, n_embd_head_v, n_head_kv, ggml_element_size(v_l) * n_ctx,
                                              ggml_element_size(v_l) * n_ctx * n_embd_head_v, 0);
        ggml_set_name(v, "v");
        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
        ggml_set_name(kqv, "kqv");
        struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        ggml_set_name(kqv_merged, "kqv_merged");
        struct ggml_tensor * cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_v * n_head, n_tokens);
        ggml_set_name(cur, "kqv_merged_cont");
        nodes_to_expand.push_back(cur);
        cur = ggml_mul_mat(ctx, wo, cur);
        return cur;
    }

    void build_weights(ggml_context * ctx) {
        GGML_ASSERT(ne_a[2] == 1);
        GGML_ASSERT(ne_a[3] == 1);

        n_embd   = ne_a[0];
        n_tokens = ne_a[1];

        // kv cache
        k_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd_k_gqa * n_ctx * n_head_kv);
        ggml_set_input(k_l);

        v_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd_v_gqa * n_ctx * n_head_kv);
        ggml_set_input(v_l);

        wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_head_v * n_head, n_embd);
        ggml_set_input(wo);

        wkv_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_lora_rank, (n_embd_head_qk_nope + n_embd_head_v) * n_head);
        ggml_set_input(wkv_b);

        wkv_a_mqa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, kv_lora_rank + n_embd_head_qk_rope);
        ggml_set_input(wkv_a_mqa);

        wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd_head_k * n_head);
        ggml_set_input(wq);

        attn_kv_a_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kv_lora_rank);
        ggml_set_input(attn_kv_a_norm);

        attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        ggml_set_input(attn_norm);
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        build_weights(ctx);
        // 输入张量
        ggml_tensor * inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inpL);
        ggml_set_name(inpL, "inpL");

        ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        ggml_set_input(kq_mask);
        ggml_set_name(kq_mask, "kq_mask");

        ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp_pos);
        ggml_set_name(inp_pos, "inp_pos");

        // 自注意力层
        ggml_tensor * inpSA = inpL;

        // 层归一化
        ggml_tensor * cur = build_norm(ctx, inpL, attn_norm);
        ggml_set_name(cur, "attn_norm");

        ggml_tensor * q = ggml_mul_mat(ctx, wq, cur);

        struct ggml_tensor * q_nope =
            ggml_view_3d(ctx, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                         ggml_row_size(q->type, n_embd_head_k * n_head), 0);
        ggml_set_name(q_nope, "q_nope");

        struct ggml_tensor * q_pe =
            ggml_view_3d(ctx, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                         ggml_row_size(q->type, n_embd_head_k * n_head), ggml_row_size(q->type, n_embd_head_qk_nope));
        ggml_set_name(q_pe, "q_pe");

        struct ggml_tensor * kv_pe_compresseed = ggml_mul_mat(ctx, wkv_a_mqa, cur);
        ggml_set_name(kv_pe_compresseed, "kv_pe_compresseed");

        struct ggml_tensor * kv_compressed =
            ggml_view_2d(ctx, kv_pe_compresseed, kv_lora_rank, n_tokens, kv_pe_compresseed->nb[1], 0);
        ggml_set_name(kv_compressed, "kv_compressed");

        struct ggml_tensor * k_pe =
            ggml_view_3d(ctx, kv_pe_compresseed, n_embd_head_qk_rope, 1, n_tokens, kv_pe_compresseed->nb[1],
                         kv_pe_compresseed->nb[1], ggml_row_size(kv_pe_compresseed->type, kv_lora_rank));
        ggml_set_name(k_pe, "k_pe");

        kv_compressed = ggml_cont(ctx, kv_compressed);
        kv_compressed = build_norm(ctx, kv_compressed, attn_kv_a_norm);

        struct ggml_tensor * kv = ggml_mul_mat(ctx, wkv_b, kv_compressed);
        ggml_set_name(kv, "kv");

        // split into {n_head * n_embd_head_qk_nope, n_tokens}
        struct ggml_tensor * k_nope =
            ggml_view_3d(ctx, kv, n_embd_head_qk_nope, n_head, n_tokens,
                         ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                         ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + n_embd_head_v)), 0);
        ggml_set_name(k_nope, "k_nope");

        // and {n_head * n_embd_head_v, n_tokens}
        struct ggml_tensor * v_states = ggml_view_3d(
            ctx, kv, n_embd_head_v, n_head, n_tokens, ggml_row_size(kv->type, (n_embd_head_qk_nope + n_embd_head_v)),
            ggml_row_size(kv->type, (n_head * (n_embd_head_qk_nope + n_embd_head_v))),
            ggml_row_size(kv->type, (n_embd_head_qk_nope)));
        ggml_set_name(v_states, "v_states");

        v_states = ggml_cont(ctx, v_states);

        v_states = ggml_view_2d(ctx, v_states, n_embd_head_v * n_head, n_tokens,
                                ggml_row_size(kv->type, n_embd_head_v * n_head), 0);
        ggml_set_name(v_states, "v_states");

        q_pe = ggml_cont(ctx,
                         q_pe);  // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
        q_pe = ggml_rope_ext(ctx, q_pe, inp_pos, nullptr, 64, 0, 4096, 10000, 0.025, 1, 0.73052001, 32, 1);
        ggml_set_name(q_pe, "q_pe");

        // shared RoPE key
        k_pe = ggml_cont(ctx,
                         k_pe);  // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
        k_pe = ggml_rope_ext(ctx, k_pe, inp_pos, nullptr, 64, 0, 4096, 10000, 0.025, 1, 0.73052001, 32, 1);
        ggml_set_name(k_pe, "k_pe");

        struct ggml_tensor * q_states = ggml_concat(ctx, q_nope, q_pe, 0);
        ggml_set_name(q_states, "q_states");

        struct ggml_tensor * k_states = ggml_concat(ctx, k_nope, ggml_repeat(ctx, k_pe, q_pe), 0);
        ggml_set_name(k_states, "k_states");

        nodes_to_expand.push_back(q_states);
        nodes_to_expand.push_back(k_states);
        nodes_to_expand.push_back(v_states);

        llm_build_kv_store(ctx, k_states, v_states);
        cur = llm_build_kqv(ctx, q_states, kq_mask);

        return cur;
    }

    static float rand_generator(int & seed, float last, float low = -2.0, float high = 2.0) {
        // 使用更强的随机算法 - 结合多种技术
        // 1. 使用 Xorshift 算法作为基础
        // 2. 混合使用位操作和非线性变换
        // 3. 结合上一个结果和当前种子

        // 将上一个结果转换为无符号整数
        uint32_t last_bits = *reinterpret_cast<uint32_t *>(&last);

        // 组合种子、上一个结果和时间
        uint32_t x = static_cast<uint32_t>(seed);
        x ^= last_bits;

        // 应用 Xorshift 算法
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;

        // 应用非线性变换 - 使用平方中间取值法
        uint64_t square = static_cast<uint64_t>(x) * static_cast<uint64_t>(x);
        uint32_t middle = static_cast<uint32_t>((square >> 16) & 0xFFFFFFFF);

        // 再次混合
        x = middle ^ (x * 747796405U + 2891336453U);
        x += static_cast<uint32_t>(last_bits * 1664525U + 1013904223U);

        // 更新种子，使下次调用产生不同结果
        seed = static_cast<int>(x);

        // 将结果映射到 [0, 1] 范围
        float normalized = static_cast<float>(x) / static_cast<float>(0xFFFFFFFF);

        // 映射到 [low, high] 范围
        float result = low + normalized * (high - low);

        return result;
    }

    void initialize_tensors(ggml_context * ctx) override {
        // 初始化随机数生成器
        int seed_weights = seedw;  // 权重的种子
        int seed_tensors = seedt;  // 其他张量的种子

        float last_rand_w = 0.0f;  // 上一个权重随机数
        float last_rand_t = 0.0f;  // 上一个张量随机数

        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // 获取张量名称
            const char * name = t->name;

            // 判断是否为权重张量（名称以 weight 开头）
            bool is_weight = (name != nullptr && strncmp(name, "weight", 6) == 0);

            // 基于不同的种子初始化张量
            if (is_weight) {
                // 权重张量使用 seedw 初始化
                size_t             nels = ggml_nelements(t);
                std::vector<float> data(nels);
                std::vector<int>   data_int(nels);

                // 生成随机数据
                if (strcmp(t->name, "kq_mask") == 0) {
                    for (size_t i = 0; i < nels; i++) {
                        last_rand_w = rand_generator(seed_weights, last_rand_w, -1.0f, 1.0f);
                        data[i]     = 0;
                    }
                } else if (strcmp(t->name, "inp_pos") == 0) {
                    for (size_t i = 0; i < nels; i++) {
                        data_int[i] = 0;
                    }
                } else {
                    for (size_t i = 0; i < nels; i++) {
                        last_rand_w = rand_generator(seed_weights, last_rand_w, -1.0f, 1.0f);
                        data[i]     = last_rand_w;
                    }
                }

                // 设置张量数据
                if (t->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(float));
                } else if (t->type == GGML_TYPE_I32) {
                    ggml_backend_tensor_set(t, data_int.data(), 0, nels * sizeof(int));
                } else if (ggml_is_quantized(t->type) || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16) {
                    // 量化张量处理
                    std::vector<uint8_t> dataq(ggml_row_size(t->type, nels));
                    ggml_quantize_chunk(t->type, data.data(), dataq.data(), 0, nels, ggml_blck_size(t->type), nullptr);
                    ggml_backend_tensor_set(t, dataq.data(), 0, dataq.size());
                }
            } else {
                // 非权重张量使用 seedt 初始化
                size_t             nels = ggml_nelements(t);
                std::vector<float> data(nels);
                std::vector<int>   data_int(nels);

                // 生成随机数据
                if (strcmp(t->name, "kq_mask") == 0) {
                    for (size_t i = 0; i < nels; i++) {
                        last_rand_w = rand_generator(seed_weights, last_rand_w, -1.0f, 1.0f);
                        data[i]     = 0;
                    }
                } else if (strcmp(t->name, "inp_pos") == 0) {
                    for (size_t i = 0; i < nels; i++) {
                        data_int[i] = 0;
                    }
                } else {
                    for (size_t i = 0; i < nels; i++) {
                        last_rand_w = rand_generator(seed_weights, last_rand_w, -1.0f, 1.0f);
                        data[i]     = last_rand_w;
                    }
                }

                // 设置张量数据
                if (t->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(float));
                } else if (t->type == GGML_TYPE_I32) {
                    ggml_backend_tensor_set(t, data_int.data(), 0, nels * sizeof(int));
                } else if (ggml_is_quantized(t->type) || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16) {
                    // 量化张量处理
                    std::vector<uint8_t> dataq(ggml_row_size(t->type, nels));
                    ggml_quantize_chunk(t->type, data.data(), dataq.data(), 0, nels, ggml_blck_size(t->type), nullptr);
                    ggml_backend_tensor_set(t, dataq.data(), 0, dataq.size());
                }
            }
        }
    }

    double max_maa_err() override { return 1e-3; }

    bool grad_precise() override { return false; }
};


struct test_deepseek_case3 : public test_case {
    const ggml_type              type;
    static const int             n_layer = 1;
    static const int             n_dense = 1;
    int                          seedw;
    int                          seedt;

    float f_norm_rms_eps;

    int n_head;
    int n_embd;
    int n_embd_head_k;
    int n_embd_head_v;
    int n_embd_head_qk_rope;
    int n_embd_head_qk_nope;
    int kv_lora_rank;
    int n_embd_k_gqa;
    int n_embd_v_gqa;
    int n_head_kv;

    int n_ff;
    int n_ff_exp;
    int n_expert;
    int n_expert_used;
    int n_expert_shared;

    // rum_time
    int n_ctx;
    int n_tokens;
    int n_kv;
    int n_vocab;

    ggml_tensor * attn_norm[n_layer];
    ggml_tensor * attn_kv_a_norm[n_layer];
    ggml_tensor * wq[n_layer];
    ggml_tensor * wkv_a_mqa[n_layer];
    ggml_tensor * wkv_b[n_layer];
    ggml_tensor * wo[n_layer];
    ggml_tensor * ffn_norm[n_layer];

    ggml_tensor * ffn_up[n_layer];
    ggml_tensor * ffn_gate[n_layer];
    ggml_tensor * ffn_down[n_layer];

    ggml_tensor * ffn_gate_inp[n_layer];
    ggml_tensor * ffn_up_exps[n_layer];
    ggml_tensor * ffn_gate_exps[n_layer];
    ggml_tensor * ffn_down_exps[n_layer];
    // ggml_tensor * ffn_exp_probs_b[n_layer];
    ggml_tensor * ffn_gate_shexp[n_layer];
    ggml_tensor * ffn_up_shexp[n_layer];
    ggml_tensor * ffn_down_shexp[n_layer];

    ggml_tensor * out_norm;
    ggml_tensor * output;

    // kv cacche
    ggml_tensor * k_l[n_layer];
    ggml_tensor * v_l[n_layer];
    ggml_tensor * kq_mask_full[n_layer];

    double max_nmse_err() override { return 1e-5; }

    std::string vars() override {
        // forward can be inferred from the op, does not need to be printed
        return "TESTDEEPSEEK3";
    }

    test_deepseek_case3(ggml_type type = GGML_TYPE_F32, int seedw = 514, int seedt = 233) :
        type(type),
        seedw(seedw),
        seedt(seedt) {
        n_ctx               = 2048;
        n_tokens            = 16;
        n_head              = 16;
        n_head_kv           = n_head;
        n_kv                = 64;
        n_embd_head_v       = 128;
        n_embd_head_k       = 192;
        int pad_n_embd      = GGML_PAD(std::max(n_embd_head_k, n_embd_head_v), 128);
        n_embd_v_gqa        = pad_n_embd * n_head;
        n_embd_k_gqa        = pad_n_embd * n_head;
        kv_lora_rank        = 64;
        n_embd_head_qk_nope = 128;
        n_embd_head_qk_rope = 64;
        n_embd              = 64;
        f_norm_rms_eps      = 1e-5;
        n_vocab             = 128;

        n_ff = 1024;
        n_ff_exp = 128;
        n_expert = 64;
        n_expert_used = 6;
        n_expert_shared = 2;
    

        // n_ctx               = 4096;
        // n_tokens            = 64;
        // n_head              = 16;
        // n_head_kv           = n_head;
        // n_kv                = 2048;
        // n_embd_head_v       = 128;
        // n_embd_head_k       = 192;
        // int pad_n_embd      = GGML_PAD(std::max(n_embd_head_k, n_embd_head_v), 128);
        // n_embd_v_gqa        = pad_n_embd * n_head;
        // n_embd_k_gqa        = pad_n_embd * n_head;
        // kv_lora_rank        = 64;
        // n_embd_head_qk_nope = 128;
        // n_embd_head_qk_rope = 64;
        // n_embd              = 2048;
        // f_norm_rms_eps      = 1e-5;
        // // 其实应该是102400
        // n_vocab             = 1024;

        // n_ff = 10944;
        // n_ff_exp = 1408;
        // n_expert = 64;
        // n_expert_used = 6;
        // n_expert_shared = 2;
    }

    ggml_tensor * build_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * mw) const {
        cur = ggml_rms_norm(ctx, cur, f_norm_rms_eps);
        cur = ggml_mul(ctx, cur, mw);
        return cur;
    }

    ggml_tensor * llm_build_kqv(ggml_context * ctx, ggml_tensor * q_cur, ggml_tensor * k_cur, ggml_tensor * v_cur, ggml_tensor * kv_indices, int layer) {
        // pad qkv
        const int64_t pad_n_embd = GGML_PAD(std::max(n_embd_head_k, n_embd_head_v), 128);
        struct ggml_tensor * q = ggml_pad(ctx, q_cur, pad_n_embd - q_cur->ne[0], 0, 0, 0);
        k_cur = ggml_cast(ctx, k_cur, GGML_TYPE_F16);
        v_cur = ggml_cast(ctx, v_cur, GGML_TYPE_F16);
        k_cur = ggml_pad(ctx, k_cur, pad_n_embd - k_cur->ne[0], 0, 0, 0);
        k_cur = ggml_reshape_2d(ctx, k_cur, n_embd_k_gqa, k_cur->ne[2]);
        v_cur = ggml_reshape_3d(ctx, v_cur, n_embd_head_v, n_head_kv, v_cur->ne[1]);
        v_cur = ggml_pad(ctx, v_cur, pad_n_embd - v_cur->ne[0], 0, 0, 0);
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_v_gqa, v_cur->ne[2]);

        // save kv
        ggml_tensor * k = k_l[layer];
        ggml_tensor * v = v_l[layer];
        k = ggml_reshape_2d(ctx, k, n_embd_k_gqa, n_ctx);
        v = ggml_reshape_2d(ctx, v, n_embd_v_gqa, n_ctx);

        // 用最简单的假设情况，batch填满。
        k = ggml_scatter_update(ctx, k, kv_indices, k_cur);
        v = ggml_scatter_update(ctx, v, kv_indices, v_cur);

        k = ggml_reshape_3d(ctx, k, pad_n_embd, n_head_kv, n_ctx);
        v = ggml_reshape_3d(ctx, v, pad_n_embd, n_head_kv, n_ctx);

        if (q->type != GGML_TYPE_F16) {
            q = ggml_cast(ctx, q, GGML_TYPE_F16);
        }
        if (k->type != GGML_TYPE_F16) {
            k = ggml_cast(ctx, k, GGML_TYPE_F16);
        }
        if (v->type != GGML_TYPE_F16) {
            v = ggml_cast(ctx, v, GGML_TYPE_F16);
        }
        ggml_tensor * length_q = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        ggml_tensor * length_kv = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        ggml_set_input(length_q);
        ggml_set_input(length_kv);
        
        // ggml_tensor * cur = ggml_flash_attn_prompt(ctx, q, k, v, kq_mask_full[layer], 1, n_head, pad_n_embd, pad_n_embd,
        //                                            n_head, n_tokens, n_kv, length_q, length_kv, 1.0);
        ggml_tensor * cur = ggml_mul(ctx, k, q);
        cur = ggml_mul(ctx, cur, v);
        cur               = ggml_reshape_3d(ctx, cur, pad_n_embd, n_head, n_tokens);
        cur               = ggml_get_slice(ctx, cur, 0, n_embd_head_v, 0);
        cur               = ggml_reshape_2d(ctx, cur, n_embd_head_v * n_head, n_tokens);
        cur               = ggml_cast(ctx, cur, GGML_TYPE_F32);
        cur               = ggml_mul_mat(ctx, wo[layer], cur);
        return cur;
    }

    static ggml_tensor * build_ffn(ggml_context * ctx, ggml_tensor * cur, ggml_tensor *up, ggml_tensor *gate, ggml_tensor *down) {
        ggml_tensor * tmp = ggml_mul_mat(ctx, up, cur);
        cur = ggml_mul_mat(ctx, gate, cur);
        cur = ggml_silu(ctx, cur);
        cur = ggml_mul(ctx, cur, tmp);
        cur = ggml_mul_mat(ctx, down, cur);
        return cur;
    }

    ggml_tensor * build_moe_ffn(ggml_context * ctx, ggml_tensor * cur, int i) {
        ggml_tensor * logits = ggml_mul_mat(ctx, ffn_gate_inp[i], cur);
        ggml_tensor * probs = ggml_soft_max(ctx, logits);
        ggml_tensor * selection_probs = probs;
        ggml_tensor * selected_experts = ggml_top_k(ctx, selection_probs, 6);
        ggml_tensor * weights = ggml_get_rows(ctx,
            ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens),
            selected_experts);
        weights = ggml_scale(ctx, weights, 1);
        cur = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
        ggml_tensor * row_idx = ggml_arange(ctx, 0, n_tokens * n_expert_used, 1);
        row_idx = ggml_reshape_2d(ctx, row_idx, n_tokens, n_expert_used);
        ggml_tensor * row_idx_int32 = ggml_cast(ctx, row_idx, GGML_TYPE_I32);
        ggml_tensor * cur_new = ggml_cast(ctx, cur, ffn_up_exps[i]->type);
        ggml_tensor * moe_out = ggml_moe_fused(
            ctx,
            cur_new,
            selected_experts,
            weights,
            ffn_up_exps[i],
            ffn_down_exps[i],
            ffn_gate_exps[i],
            row_idx_int32,
            0,
            n_expert - 1);
        return moe_out;
    }

    void build_weights(ggml_context * ctx) {
        for (int i = 0; i < n_layer; i++) {
            // kv cache
            k_l[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_embd_k_gqa * n_ctx);
            ggml_set_input(k_l[i]);

            v_l[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_embd_v_gqa * n_ctx);
            ggml_set_input(v_l[i]);

            kq_mask_full[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, n_ctx, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
            kq_mask_full[i] = ggml_set_name(kq_mask_full[i], "kq_mask_full");
            ggml_set_input(kq_mask_full[i]);

            wo[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd_head_v * n_head, n_embd);
            ggml_set_input(wo[i]);

            wkv_b[i] =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_lora_rank, (n_embd_head_qk_nope + n_embd_head_v) * n_head);
            ggml_set_input(wkv_b[i]);

            wkv_a_mqa[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, kv_lora_rank + n_embd_head_qk_rope);
            ggml_set_input(wkv_a_mqa[i]);

            wq[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_embd_head_k * n_head);
            ggml_set_input(wq[i]);

            attn_kv_a_norm[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, kv_lora_rank);
            ggml_set_input(attn_kv_a_norm[i]);

            attn_norm[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_embd);
            ggml_set_input(attn_norm[i]);

            ffn_norm[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_embd);
            ggml_set_input(ffn_norm[i]);

            if (i < n_dense) {
                ffn_gate[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_ff);
                ggml_set_input(ffn_gate[i]);
                
                ffn_down[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_ff, n_embd);
                ggml_set_input(ffn_down[i]);
                
                ffn_up[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_ff);
                ggml_set_input(ffn_up[i]);
            } else {
                ffn_gate_inp[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_expert);
                ggml_set_input(ffn_gate_inp[i]);
                
                ffn_gate_exps[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd, n_ff_exp, n_expert);
                ggml_set_input(ffn_gate_exps[i]);
                
                ffn_down_exps[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_ff_exp, n_embd, n_expert);
                ggml_set_input(ffn_down_exps[i]);
                
                ffn_up_exps[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd, n_ff_exp, n_expert);
                ggml_set_input(ffn_up_exps[i]);
    
                ffn_gate_shexp[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_ff_exp * n_expert_shared);
                ggml_set_input(ffn_gate_shexp[i]);
                
                ffn_down_shexp[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_ff_exp * n_expert_shared, n_embd);
                ggml_set_input(ffn_down_shexp[i]);
    
                ffn_up_shexp[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_ff_exp * n_expert_shared);
                ggml_set_input(ffn_up_shexp[i]);
            }
        }
        out_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_embd);
        ggml_set_input(out_norm);
        output = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_vocab);
        ggml_set_input(output);
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        build_weights(ctx);
        // 输入张量
        ggml_tensor * inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inpL);
        ggml_set_name(inpL, "inpL");

        ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp_pos);
        ggml_set_name(inp_pos, "inp_pos");

        ggml_tensor * kv_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
        ggml_set_input(kv_indices);
        ggml_set_name(kv_indices, "kv_indices");

        ggml_tensor * cur;
        for (int i = 0; i < n_layer; i++) {
            ggml_tensor *inpSA = inpL;
            // 自注意力层
            {
                // 层归一化
                cur = build_norm(ctx, inpL, attn_norm[i]);
                ggml_set_name(cur, "attn_norm");

                ggml_tensor * q = ggml_mul_mat(ctx, wq[i], cur);
                q               = ggml_reshape_3d(ctx, q, n_embd_head_k, n_head, n_tokens);
                GGML_ASSERT(n_embd_head_k == n_embd_head_qk_nope + n_embd_head_qk_rope);

                // q_nope = q[:, :, :n_embd_head_qk_nope]
                ggml_tensor * q_nope = ggml_get_slice(ctx, q, 0, n_embd_head_qk_nope, 0);
                ggml_set_name(q_nope, "q_nope");

                // q_rope = q[:, :, n_embd_head_qk_nope:]
                ggml_tensor * q_pe = ggml_get_slice(ctx, q, n_embd_head_qk_nope, n_embd_head_k, 0);
                ggml_set_name(q_pe, "q_pe");

                struct ggml_tensor * kv_pe_compresseed = ggml_mul_mat(ctx, wkv_a_mqa[i], cur);
                ggml_set_name(kv_pe_compresseed, "kv_pe_compresseed");

                // kv_compressed = kv_pe_compresseed[:, :kv_lora_rank]
                struct ggml_tensor * kv_compressed = ggml_get_slice(ctx, kv_pe_compresseed, 0, kv_lora_rank, 0);
                ggml_set_name(kv_compressed, "kv_compressed");

                // k_pe = kv_pe_compresseed[:, kv_lora_rank:]
                ggml_tensor * k_pe =
                    ggml_get_slice(ctx, kv_pe_compresseed, kv_lora_rank, kv_lora_rank + n_embd_head_qk_rope, 0);
                k_pe = ggml_reshape_3d(ctx, k_pe, n_embd_head_qk_rope, 1, n_tokens);
                ggml_set_name(k_pe, "k_pe");

                kv_compressed = ggml_cont(ctx, kv_compressed);
                kv_compressed = build_norm(ctx, kv_compressed, attn_kv_a_norm[i]);

                struct ggml_tensor * kv = ggml_mul_mat(ctx, wkv_b[i], kv_compressed);
                kv                      = ggml_reshape_3d(ctx, kv, n_embd_head_qk_nope + n_embd_head_v, n_head, n_tokens);
                ggml_set_name(kv, "kv");

                // k_nope = kv[:, :, :n_embd_head_qk_nope]
                struct ggml_tensor * k_nope = ggml_get_slice(ctx, kv, 0, n_embd_head_qk_nope, 0);
                ggml_set_name(k_nope, "k_nope");

                // v_states = kv[:, :, n_embd_head_qk_nope:]
                struct ggml_tensor * v_states =
                    ggml_get_slice(ctx, kv, n_embd_head_qk_nope, n_embd_head_qk_nope + n_embd_head_v, 0);
                v_states = ggml_reshape_2d(ctx, v_states, n_embd_head_v * n_head, n_tokens);
                ggml_set_name(v_states, "v_states");

                q_pe = ggml_rope_ext(ctx, q_pe, inp_pos, nullptr, 64, 0, 4096, 10000, 0.025, 1, 0.73052001, 32, 1);
                ggml_set_name(q_pe, "q_pe");

                // shared RoPE key
                k_pe = ggml_rope_ext(ctx, k_pe, inp_pos, nullptr, 64, 0, 4096, 10000, 0.025, 1, 0.73052001, 32, 1);

                ggml_set_name(k_pe, "k_pe");

                struct ggml_tensor * q_states = ggml_concat(ctx, q_nope, q_pe, 0);
                ggml_set_name(q_states, "q_states");

                struct ggml_tensor * k_states = ggml_concat(ctx, k_nope, ggml_repeat(ctx, k_pe, q_pe), 0);
                ggml_set_name(k_states, "k_states");

                ggml_tensor* test = ggml_mul(ctx, k_states, q_states);
                return test;

                cur = llm_build_kqv(ctx, q_states, k_states, v_states, kv_indices, i);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);

            cur = build_norm(ctx, ffn_inp, ffn_norm[i]);

            if (i < n_dense) {
                cur = build_ffn(ctx, cur, ffn_up[i], ffn_gate[i], ffn_down[i]);
            } else {
                ggml_tensor * moe_out = build_moe_ffn(ctx, cur, i);
                ggml_tensor * ffn_shexp = build_ffn(ctx, cur, ffn_up_shexp[i], ffn_gate_shexp[i], ffn_down_shexp[i]);
                cur = ggml_add(ctx, moe_out, ffn_shexp);
            }
            cur = ggml_add(ctx, cur, ffn_inp);
            inpL = cur;
        }

        cur = inpL;
        cur = build_norm(ctx, cur, out_norm);
        cur = ggml_mul_mat(ctx, output, cur);
        return cur;
    }

    static float rand_generator(int & seed, float last, float low = -2.0, float high = 2.0) {
        // 使用更强的随机算法 - 结合多种技术
        // 1. 使用 Xorshift 算法作为基础
        // 2. 混合使用位操作和非线性变换
        // 3. 结合上一个结果和当前种子

        // 将上一个结果转换为无符号整数
        uint32_t last_bits = *reinterpret_cast<uint32_t *>(&last);

        // 组合种子、上一个结果和时间
        uint32_t x = static_cast<uint32_t>(seed);
        x ^= last_bits;

        // 应用 Xorshift 算法
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;

        // 应用非线性变换 - 使用平方中间取值法
        uint64_t square = static_cast<uint64_t>(x) * static_cast<uint64_t>(x);
        uint32_t middle = static_cast<uint32_t>((square >> 16) & 0xFFFFFFFF);

        // 再次混合
        x = middle ^ (x * 747796405U + 2891336453U);
        x += static_cast<uint32_t>(last_bits * 1664525U + 1013904223U);

        // 更新种子，使下次调用产生不同结果
        seed = static_cast<int>(x);

        // 将结果映射到 [0, 1] 范围
        float normalized = static_cast<float>(x) / static_cast<float>(0xFFFFFFFF);

        // 映射到 [low, high] 范围
        float result = low + normalized * (high - low);

        return result;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            size_t nels = ggml_nelements(t);
            // 生成随机数据
            if (strcmp(t->name, "inp_pos") == 0) {
                std::vector<int> data(nels);
                for (size_t i = 0; i < nels; i++) {
                    data[i] = i;
                }
                ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(int));
            } else if (strncmp(t->name, "kq_mask_full", 11) == 0) {
                std::vector<int8_t> data(nels);
                for (size_t i = 0; i < nels; i++) {
                    data[i] = 0;
                }
                ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(int8_t));
            } else if (strcmp(t->name, "kv_indices") == 0) {
                GGML_ASSERT(n_kv > (int) nels);
                std::vector<int64_t> data(nels);
                for (size_t i = 0; i < nels; i++) {
                    data[i] = n_kv - nels + i;
                }
                ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(int64_t));
            } else {
                init_tensor_uniform(t, -1.0f, 1.0f);
            }
        }
    }

    double max_maa_err() override { return 1e-3; }

    bool grad_precise() override { return false; }
};

struct test_mla_preprocess_ds : public test_case {
    // 配置参数
    int token_num  = 32;
    int head_num   = 128;
    int block_size = 128;
    int block_num  = 192;

    // 输入/权重维度
    int n_embd_in = 7168;  // input 的最后一维
    int wdqkv_out = 2112;  // 第一阶段 matmul 输出 dim
    int q_dim     = 1536;  // 第二阶段 q dim
    int kv_dim    = 512;   // k/v dim
    int rope_dim  = 64;    // rope dim

    float eps_rms = 1e-5f;

    // ggml 数据类型
    ggml_type type = GGML_TYPE_F16;

    // ---- 基类覆盖 ----
    double max_nmse_err() override { return 0.0; }

    double max_maa_err() override { return 0.0; }

    bool grad_precise() override { return false; }

    std::string vars() override { return "GGML_MLA_PREPROCESS_DS"; }

    test_mla_preprocess_ds(int tok = 32, int head = 128, int bsz = 128, int bnum = 192) :
        token_num(tok),
        head_num(head),
        block_size(bsz),
        block_num(bnum) {}

    // ---- 图构建：直接调用 ggml_mla_preprocess ----
    ggml_tensor * build_graph(ggml_context * ctx) override {
        // 1) 创建输入/权重/常量/缓冲张量，并初始化（按原 Demo 约定）
        auto * hiddenState = ggml_new_tensor_2d(ctx, type, n_embd_in, token_num);
        ggml_set_input(hiddenState);
        ggml_set_name(hiddenState, "hiddenState");

        auto * gamma1 = ggml_new_tensor_1d(ctx, type, n_embd_in);
        ggml_set_input(gamma1);
        ggml_set_name(gamma1, "gamma1");

        auto * beta1 = ggml_new_tensor_1d(ctx, type, n_embd_in);
        ggml_set_input(beta1);
        ggml_set_name(beta1, "beta1");

        auto * quantScale1 = ggml_new_tensor_1d(ctx, type, 1);
        ggml_set_input(quantScale1);
        ggml_set_name(quantScale1, "quantScale1");

        auto * quantOffset1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 1);
        ggml_set_input(quantOffset1);
        ggml_set_name(quantOffset1, "quantOffset1");

        auto * wdqkv = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, n_embd_in, wdqkv_out);
        ggml_set_input(wdqkv);
        ggml_set_name(wdqkv, "nz_wdqkv");

        auto * descale1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, wdqkv_out);
        ggml_set_input(descale1);
        ggml_set_name(descale1, "descale1");

        auto * bias1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, wdqkv_out);
        ggml_set_input(bias1);
        ggml_set_name(bias1, "bias1");

        auto * gamma2 = ggml_new_tensor_1d(ctx, type, q_dim);
        ggml_set_input(gamma2);
        ggml_set_name(gamma2, "gamma2");

        auto * beta2 = ggml_new_tensor_1d(ctx, type, q_dim);
        ggml_set_input(beta2);
        ggml_set_name(beta2, "beta2");

        auto * quantScale2 = ggml_new_tensor_1d(ctx, type, 1);
        ggml_set_input(quantScale2);
        ggml_set_name(quantScale2, "quantScale2");

        auto * quantOffset2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 1);
        ggml_set_input(quantOffset2);
        ggml_set_name(quantOffset2, "quantOffset2");

        auto * wuq = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, /*48*32=*/1536, head_num * 192);
        ggml_set_input(wuq);
        ggml_set_name(wuq, "nz_wuq");

        auto * descale2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, head_num * 192);
        ggml_set_input(descale2);
        ggml_set_name(descale2, "descale2");

        auto * bias2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, head_num * 192);
        ggml_set_input(bias2);
        ggml_set_name(bias2, "bias2");

        auto * gamma3 = ggml_new_tensor_1d(ctx, type, kv_dim);
        ggml_set_input(gamma3);
        ggml_set_name(gamma3, "gamma3");

        auto * cos_tbl = ggml_new_tensor_2d(ctx, type, rope_dim, token_num);
        ggml_set_input(cos_tbl);
        ggml_set_name(cos_tbl, "cos");

        auto * sin_tbl = ggml_new_tensor_2d(ctx, type, rope_dim, token_num);
        ggml_set_input(sin_tbl);
        ggml_set_name(sin_tbl, "sin");

        auto * wuk = ggml_new_tensor_3d(ctx, type, kv_dim, 128, head_num);
        ggml_set_input(wuk);
        ggml_set_name(wuk, "wuk");

        auto * kvCache = ggml_new_tensor_4d(ctx, type, kv_dim, 1, block_size, block_num);
        ggml_set_input(kvCache);
        ggml_set_name(kvCache, "kvCache");

        // 算子中没加入这个参数
        auto * kvCacheRope = ggml_new_tensor_4d(ctx, type, rope_dim, 1, block_size, block_num);
        ggml_set_input(kvCacheRope);
        ggml_set_name(kvCacheRope, "kvCacheRope");

        auto * slotmapping = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_num);
        ggml_set_input(slotmapping);
        ggml_set_name(slotmapping, "slotmapping");

        auto * ctkvScale = ggml_new_tensor_1d(ctx, type, 1);
        ggml_set_input(ctkvScale);
        ggml_set_name(ctkvScale, "ctkvScale");

        auto * qNopeScale = ggml_new_tensor_1d(ctx, type, head_num);
        ggml_set_input(qNopeScale);
        ggml_set_name(qNopeScale, "qNopeScale");

        // 输出张量（由算子写入）
        auto * qOut0 = ggml_new_tensor_3d(ctx, type, kv_dim, head_num, token_num);
        ggml_set_name(qOut0, "qOut0");
        auto * qOut1 = ggml_new_tensor_3d(ctx, type, rope_dim, head_num, token_num);
        ggml_set_name(qOut1, "qOut1");

        // 2) 初始化数据，遵循原 Demo 默认值（必要最小化，足以跑通）

        auto convert_float_f16 = [&](std::vector<float>& data, ggml_tensor* t) {
            size_t nels = ggml_nelements(t);
            std::vector<uint8_t> dataq(ggml_row_size(t->type, nels));
            ggml_quantize_chunk(t->type, data.data(), dataq.data(), 0, nels, ggml_blck_size(t->type), nullptr);
            ggml_backend_tensor_set(t, dataq.data(), 0, dataq.size());
        };

        auto fill_f16 = [&](ggml_tensor * t, float v) {
            size_t nels = ggml_nelements(t);
            std::vector<float> data(nels, v);
            convert_float_f16(data, t);
        };
        auto fill_i32 = [&](ggml_tensor * t) {
            std::vector<int32_t> buf(ggml_nelements(t));
            for (size_t i = 0; i < buf.size(); ++i) {
                buf[i] = (int32_t) i;  // slotmapping: 0..token-1
            }
            ggml_backend_tensor_set(t, buf.data(), 0, buf.size() * sizeof(int32_t));
        };
        // fill_f16(hiddenState, 0.0f); fill_f16(gamma1, 0.0f); fill_f16(beta1, 0.0f);
        // fill_f16(quantScale1, 0.0f); fill_f16(quantOffset1, 1.0f);
        // // wdqkv / wuq / wuk 给个小随机便于非零
        // {
        //     std::vector<float> tmp(ggml_nelements(wdqkv)); for(size_t i=0;i<tmp.size();++i) tmp[i] = ((int)i%13 - 6)*0.01f; convert_float_f16(tmp, wdqkv);
        // }
        // fill_f16(descale1, 1.0f); fill_f16(bias2, 1.0f);
        // fill_f16(gamma2, 0.0f); fill_f16(beta2, 0.0f); fill_f16(quantScale2, 0.0f); fill_f16(quantOffset2, 1.0f);
        // {
        //     std::vector<float> tmp(ggml_nelements(wuq)); for(size_t i=0;i<tmp.size();++i) tmp[i] = ((int)i%11 - 5)*0.01f; convert_float_f16(tmp, wuq);
        // }
        // fill_f16(descale2, 1.0f); fill_f16(bias2, 1.0f);
        // fill_f16(gamma3, 0.0f); // 原 Demo 的 gamma2 初值为 0
        // fill_f16(cos_tbl, 0.0f); fill_f16(sin_tbl, 0.5f);
        // {
        //     std::vector<float> tmp(ggml_nelements(wuk)); for(size_t i=0;i<tmp.size();++i) tmp[i] = ((int)i%7 - 3)*0.01f; convert_float_f16(tmp, wuk);
        // }
        // fill_f16(kvCache, 0.0f); fill_f16(kvCacheRope, 0.0f); fill_i32(slotmapping);
        // fill_f16(ctkvScale, 0.0f); fill_f16(qNopeScale, 0.0f);


        ggml_tensor* exec = ggml_mla_preprocess(ctx, 
            hiddenState,    // [tokenNum,7168]          dtype
            gamma1,         // [7168]                   dtype
            beta1,          // [1]                      dtype
            quantScale1,    // [1]                      dtype
            quantOffset1,   // [1]                      int8
            wdqkv,          // [2112,7168] FRACTAL_NZ   int8
            bias1,          // [2112]                   int32
            gamma2,         // [1536]                   dtype
            beta2,          // [1536]                   dtype
            quantScale2,    // [1]                      dtype
            quantOffset2,   // [1]                      int8
            gamma3,         // [512]                    dtype
            sin_tbl,        // [tokenNum,64]            dtype
            cos_tbl,        // [tokenNum,64]            dtype
            kvCache,        // [blockNum,blockSize,1,512] dtype
            slotmapping,    // [tokenNum]               int32
            wuq,            // [headNum*192,1536] FRACTAL_NZ    int8
            bias2,          // [headNum*192]            int32
            wuk,            // [headNum,128,512]        dtype
            descale1,       // [2112]                   int64
            descale2,       // [headNum*192]            int64
            ctkvScale,      // [1]                      dtype
            qNopeScale,     // [headNum]                dtype
            token_num,
            head_num,
            1,
            0,
            qOut0,          // [tokenNum,headNum,512]   dtype
            // kvCache,        // [blockNum,blockSize,1,512]   dtype
            qOut1           // [tokenNum,headNum,64]    dtype
            // kvCacheRope     // [blockNum,blockSize,1,64]    dtype
        );
        ggml_set_name(exec, "mla_preprocess_exec");


        // 4) 可选：为了单一输出节点，拼接 qOut0/qOut1 作为测试框架抓取点
        // ggml_tensor* out = ggml_concat(ctx, ggml_cont(ctx, qOut0), ggml_cont(ctx, qOut1), 0);
        // ggml_set_name(out, "mla_preprocess_out");


        return exec; // 或返回 exec
    };

    // void initialize_tensors(ggml_context*) override {}
};

// ###########################################
// ## Section 3: GGML Op Test Instantiation ##
// ###########################################
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
    test_cases.emplace_back(new test_get_slice(GGML_TYPE_F32, { 32, 10, 2, 1 }, 2, 5, 1));
    test_cases.emplace_back(new test_set_slice());
    test_cases.emplace_back(new test_deepseek_case2());
    test_cases.emplace_back(new test_deepseek_case3());
    test_cases.emplace_back(new test_mixture_ops_bug());
    test_cases.emplace_back(new test_mixture_ops_2_matmul());
    test_cases.emplace_back(new test_mixture_ops_5_matmul_fast());
    // test_cases.emplace_back(new test_mixture_ops_20_matmul());
    // test_cases.emplace_back(new test_mixture_ops());
    test_cases.emplace_back(new test_mla_preprocess_ds());
    // unary ops
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        for (int v : { 0, 1 }) {
            for (int op = 0; op < GGML_UNARY_OP_COUNT; op++) {
                test_cases.emplace_back(new test_unary((ggml_unary_op) op, type, { 128, 2, 2, 2 }, v));
                test_cases.emplace_back(new test_unary((ggml_unary_op) op, type, { 5, 7, 11, 13 }, v));
            }
        }
    }

    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 1, 8, 2, 1, false));
    for (ggml_type type : all_types) {
        for (int b : { 1, 7 }) {
            for (bool v : { false, true }) {
                test_cases.emplace_back(new test_get_rows(type, 256, 5, 4, b, v));
            }
        }
    }
    for (int b : { 1, 7 }) {
        for (bool v : { false, true }) {
            test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 256, 5, 4, b, v));
        }
    }

    test_cases.emplace_back(new test_get_rows_back(GGML_TYPE_F32, 1, 8, 2, 1, false));
    for (ggml_type type : all_types) {
        for (bool v : { false, true }) {
            test_cases.emplace_back(new test_get_rows_back(type, 256, 5, 4, 1, v));
        }
    }
    for (bool v : { false, true }) {
        test_cases.emplace_back(new test_get_rows_back(GGML_TYPE_I32, 256, 5, 4, 1, v));
    }

    for (ggml_type type_input : { GGML_TYPE_F32 }) {
        for (ggml_op_pool pool_type : { GGML_OP_POOL_AVG, GGML_OP_POOL_MAX }) {
            for (int k0 : { 1, 3 }) {
                for (int k1 : { 1, 3 }) {
                    for (int s0 : { 1, 2 }) {
                        for (int s1 : { 1, 2 }) {
                            for (int p0 : { 0, 1 }) {
                                for (int p1 : { 0, 1 }) {
                                    test_cases.emplace_back(new test_pool2d(pool_type, type_input, { 10, 10, 3, 1 }, k0,
                                                                            k1, s0, s1, p0, p1));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 测试FP32、FP16的moe_fused
    for (ggml_type type_input : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        for (ggml_type type_topk_weight : { GGML_TYPE_F32 }) {
            for (ggml_type type_ids : { GGML_TYPE_I32 }) {
                for (ggml_type type_expert_up_weight : { GGML_TYPE_F32 }) {
                    for (ggml_type type_expert_down_weight : { GGML_TYPE_F32 }) {
                        for (int n_experts : { 64 }) {        //n_expert
                            for (int n_topk : { 3 }) {        //topk
                                for (int n_tokens : { 4 }) {  //n_token
                                    int n_output_dims = 8;
                                    int n_k_dims      = 6;
                                    test_cases.emplace_back(new test_moe_fused(
                                        type_input, type_ids, type_topk_weight, type_input,
                                        type_input, n_experts, n_topk, n_tokens, n_output_dims, n_k_dims));
                                    test_cases.emplace_back(
                                        new test_moe_fused(type_input, type_ids, type_input, type_input, type_input,
                                                           n_experts, n_topk, n_tokens, n_output_dims, n_k_dims));
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    // im2col 1D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, { 3000, 128, 1, 1 },
                                            { 3, 128, 1280, 1 }, 1, 0, 1, 0, 1, 0, false));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, { 3000, 128, 1, 1 },
                                            { 3, 128, 1280, 1 }, 1, 0, 1, 0, 1, 0, false));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 3000, 128, 1, 1 },
                                            { 3, 128, 1280, 1 }, 1, 0, 1, 0, 1, 0, false));
    for (int s0 : { 1, 3 }) {
        for (int p0 : { 0, 3 }) {
            for (int d0 : { 1, 3 }) {
                test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, { 20, 2, 2, 1 },
                                                        { 3, 2, 2, 1 }, s0, 0, p0, 0, d0, 0, false));
            }
        }
    }

    // im2col 2D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    for (int s0 : { 1, 3 }) {
        for (int s1 : { 1, 3 }) {
            for (int p0 : { 0, 3 }) {
                for (int p1 : { 0, 3 }) {
                    for (int d0 : { 1, 3 }) {
                        for (int d1 : { 1, 3 }) {
                            test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32,
                                                                    { 20, 20, 2, 2 }, { 3, 3, 2, 2 }, s0, s1, p0, p1,
                                                                    d0, d1, true));
                        }
                    }
                }
            }
        }
    }

    // extra tests for im2col 2D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 1, 32 },
                                            { 3, 3, 1, 32 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 2, 32 },
                                            { 3, 3, 2, 32 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 1, 1024 },
                                            { 3, 3, 1, 1024 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 2, 1024 },
                                            { 3, 3, 2, 1024 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 1, 2048 },
                                            { 3, 3, 1, 2048 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 2, 2048 },
                                            { 3, 3, 2, 2048 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 1, 2560 },
                                            { 3, 3, 1, 2560 }, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, { 12, 12, 2, 2560 },
                                            { 3, 3, 2, 2560 }, 1, 1, 1, 1, 1, 1, true));

    // sycl backend will limit task global_range < MAX_INT
    // test cases for 2D im2col with large input W and H (occurs in stable-diffusion)
    // however these cases need to alloc more memory which may fail in some devices (Intel Arc770, etc.)
    // these cases are verified (pass) in Intel(R) Data Center GPU Max 1100 (sycl backend) and NV A30 (cuda backend)
    // test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {1024, 1024, 256, 1}, {3, 3, 256, 1}, 1, 1, 1, 1, 1, 1, true));
    // test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, {1024, 1024, 256, 1}, {3, 3, 256, 1}, 1, 1, 1, 1, 1, 1, true));

    test_cases.emplace_back(new test_conv_transpose_1d());
    test_cases.emplace_back(new test_conv_transpose_1d({ 3, 2, 1, 1 }, { 2, 3, 2, 1 }, 3, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 3, 2, 1, 1 }, { 2, 3, 2, 1 }, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 3, 2, 1, 1 }, { 2, 3, 2, 1 }, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 3, 2, 1, 1 }, { 3, 2, 2, 1 }, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 3, 2, 1, 1 }, { 3, 2, 2, 1 }, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 3, 2, 1, 1 }, { 3, 1, 2, 1 }, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({ 2, 1, 1, 1 }, { 3, 1, 1, 1 }, 1, 0, 1));

    test_cases.emplace_back(new test_count_equal(GGML_TYPE_F32, { 4, 500, 1, 1 }));
    test_cases.emplace_back(new test_count_equal(GGML_TYPE_F32, { 4, 5000, 1, 1 }));

    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 32, 1, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 100, 10, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 1024, 10, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 1024, 12, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 2000, 10, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 5438, 3, 1, 1 }));

    for (int ne3 : { 1, 3 }) {  // CUDA backward pass only supports ne3 == 1
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, { 10, 5, 4, ne3 }, { 1, 1, 1, 1 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, { 10, 5, 4, ne3 }, { 2, 1, 1, 1 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, { 10, 5, 4, ne3 }, { 1, 2, 1, 1 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, { 10, 5, 4, ne3 }, { 1, 1, 2, 1 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, { 10, 5, 4, ne3 }, { 1, 1, 1, 2 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_I32, { 10, 5, 4, ne3 }, { 2, 1, 1, 1 }));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_I16, { 10, 5, 4, ne3 }, { 1, 1, 1, 2 }));
    }

    for (bool view : { false, true }) {
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, { 8, 6, 4, 2 }, { 1, 1, 1, 1 }, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, { 8, 6, 4, 2 }, { 2, 1, 1, 1 }, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, { 8, 6, 4, 2 }, { 1, 2, 1, 1 }, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, { 8, 6, 4, 2 }, { 1, 1, 2, 1 }, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, { 8, 6, 4, 2 }, { 1, 1, 1, 2 }, view));
    }

    test_cases.emplace_back(new test_dup(GGML_TYPE_F32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, { 10, 10, 5, 1 }, { 0, 2, 1, 3 }));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, { 10, 10, 5, 1 }, { 0, 2, 1, 3 }));  // dup by rows
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, { 10, 10, 5, 1 }, { 1, 0, 2, 3 }));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, { 10, 10, 5, 1 }, { 1, 0, 2, 3 }));  // dup dst not-contiguous
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, { 10, 8, 3, 1 }, { 0, 2, 1, 3 }));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, { 10, 8, 3, 1 }, { 1, 2, 0, 3 }));

    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        test_cases.emplace_back(new test_set(GGML_TYPE_F32, GGML_TYPE_F32, { 6, 5, 4, 3 }, dim));
    }

    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        test_cases.emplace_back(new test_set(GGML_TYPE_I32, GGML_TYPE_I32, { 6, 5, 4, 3 }, dim));
    }

    for (ggml_type type_src : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        for (ggml_type type_dst : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, { 256, 4, 4, 4 }));
            test_cases.emplace_back(new test_cpy(type_src, type_dst, { 256, 2, 3, 4 }, { 0, 2, 1, 3 }));  // cpy by rows
        }
    }
    for (ggml_type type_dst : { GGML_TYPE_F32 }) {
        for (ggml_type type_src : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, { 256, 4, 4, 4 }));
            test_cases.emplace_back(new test_cpy(type_src, type_dst, { 256, 2, 3, 4 }, { 0, 2, 1, 3 }));  // cpy by rows
        }
    }
    for (ggml_type type_src : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        for (ggml_type type_dst : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            test_cases.emplace_back(
                new test_cpy(type_src, type_dst, { 256, 2, 3, 4 }, { 1, 0, 2, 3 }));  // cpy not-contiguous
        }
    }

    test_cases.emplace_back(new test_reshape());
    test_cases.emplace_back(new test_reshape(GGML_TYPE_F32, { 2, 3, 5, 7 }, { 5, 3, 2, 7 }));
    test_cases.emplace_back(new test_reshape(GGML_TYPE_F32, { 10, 6, 3, 2 }, { 5, 12, 6, 1 }));
    test_cases.emplace_back(new test_reshape(GGML_TYPE_F32, { 10, 10, 10, 1 }, { 10, 1, 10, 10 }));
    test_cases.emplace_back(new test_reshape(GGML_TYPE_F32, { 1000, 1, 1, 1 }, { 10, 10, 10, 1 }));
    test_cases.emplace_back(new test_reshape(GGML_TYPE_F16, { 2, 3, 5, 7 }, { 5, 3, 2, 7 }));
    test_cases.emplace_back(new test_reshape(GGML_TYPE_F16, { 10, 6, 3, 2 }, { 5, 12, 6, 1 }));
    test_cases.emplace_back(new test_reshape(GGML_TYPE_F16, { 10, 10, 10, 1 }, { 10, 1, 10, 10 }));
    test_cases.emplace_back(new test_reshape(GGML_TYPE_F16, { 1000, 1, 1, 1 }, { 10, 10, 10, 1 }));
    test_cases.emplace_back(new test_permute());
    test_cases.emplace_back(new test_permute(GGML_TYPE_F32, { 10, 10, 10, 1 }, { 0, 2, 1, 3 }));
    test_cases.emplace_back(new test_permute(GGML_TYPE_F32, { 2, 3, 5, 7 }, { 0, 1, 3, 2 }));
    test_cases.emplace_back(new test_permute(GGML_TYPE_F32, { 3, 3, 5, 7 }, { 3, 0, 1, 2 }));
    test_cases.emplace_back(new test_permute(GGML_TYPE_F32, { 10, 1, 10, 1 }, { 0, 2, 1, 3 }));
    test_cases.emplace_back(new test_permute(GGML_TYPE_F16, { 10, 10, 10, 1 }, { 0, 2, 1, 3 }));
    test_cases.emplace_back(new test_permute(GGML_TYPE_F16, { 2, 3, 5, 7 }, { 0, 1, 3, 2 }));
    test_cases.emplace_back(new test_permute(GGML_TYPE_F16, { 3, 3, 5, 7 }, { 3, 0, 1, 2 }));
    test_cases.emplace_back(new test_permute(GGML_TYPE_F16, { 10, 1, 10, 1 }, { 0, 2, 1, 3 }));
    test_cases.emplace_back(new test_transpose());
    test_cases.emplace_back(new test_transpose(GGML_TYPE_F32, { 2, 1, 1, 1 }));
    test_cases.emplace_back(new test_transpose(GGML_TYPE_F32, { 2, 1, 3, 5 }));
    test_cases.emplace_back(new test_transpose(GGML_TYPE_F32, { 2, 3, 5, 7 }));
    test_cases.emplace_back(new test_transpose(GGML_TYPE_F16, { 2, 1, 1, 1 }));
    test_cases.emplace_back(new test_transpose(GGML_TYPE_F16, { 2, 1, 3, 5 }));
    test_cases.emplace_back(new test_transpose(GGML_TYPE_F16, { 2, 3, 5, 7 }));
    test_cases.emplace_back(new test_view());
    test_cases.emplace_back(new test_view(GGML_TYPE_F32, { 2, 3, 5, 7 }));
    test_cases.emplace_back(new test_view(GGML_TYPE_F32, { 10, 10, 10, 1 }));
    test_cases.emplace_back(new test_view(GGML_TYPE_F16, { 2, 3, 5, 7 }));
    test_cases.emplace_back(new test_view(GGML_TYPE_F16, { 10, 10, 10, 1 }));

    test_cases.emplace_back(new test_cont());
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, { 2, 1, 1, 1 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, { 2, 1, 3, 5 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, { 2, 3, 5, 7 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, { 2, 1, 1, 1 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, { 2, 1, 3, 5 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, { 2, 3, 5, 7 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_BF16, { 2, 1, 1, 1 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_BF16, { 2, 1, 3, 5 }));
    test_cases.emplace_back(new test_cont(GGML_TYPE_BF16, { 2, 3, 5, 7 }));

    auto add_test_bin_bcast = [&](ggml_type type, std::array<int64_t, 4> ne, std::array<int, 4> nr) {
        for (auto op : { ggml_add, ggml_sub, ggml_mul, ggml_div }) {
            test_cases.emplace_back(new test_bin_bcast(op, type, ne, nr));
        }
    };
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        add_test_bin_bcast(type, { 1, 1, 8, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 1, 1 }, { 32, 1, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 320, 320 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 1, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 2, 1, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 2, 1, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 1, 2, 1 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 1, 1, 2 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 1, 2, 2 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 1, 2, 2, 2 });
        add_test_bin_bcast(type, { 10, 5, 4, 3 }, { 2, 2, 2, 2 });

        // stable diffusion
        add_test_bin_bcast(type, { 1280, 1, 1, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 1280, 1, 1, 1 }, { 1, 16, 16, 1 });
        add_test_bin_bcast(type, { 1280, 16, 16, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 1280, 1, 1, 1 }, { 1, 256, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 1280, 1 }, { 16, 16, 1, 1 });
        add_test_bin_bcast(type, { 16, 16, 1280, 1 }, { 1, 1, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 1920, 1 }, { 16, 16, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 2560, 1 }, { 16, 16, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 1280, 1 }, { 32, 32, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 1920, 1 }, { 32, 32, 1, 1 });
        add_test_bin_bcast(type, { 1, 1, 640, 1 }, { 32, 32, 1, 1 });
        add_test_bin_bcast(type, { 5120, 1, 1, 1 }, { 1, 256, 1, 1 });
        add_test_bin_bcast(type, { 640, 1, 1, 1 }, { 1, 1, 1, 1 });
        //add_test_bin_bcast(type, {3, 3, 2560, 1280}, {1, 1, 1, 1});
        //add_test_bin_bcast(type, {3, 3, 2560, 1280}, {2, 1, 1, 1});
    }

    test_cases.emplace_back(new test_add1());
    test_cases.emplace_back(new test_scale());
    test_cases.emplace_back(new test_silu_back());

    for (float eps : { 0.0f, 1e-6f, 1e-4f, 1e-1f }) {
        for (bool v : { false }) {
            test_cases.emplace_back(new test_norm(GGML_TYPE_F32, { 64, 5, 4, 3 }, v, eps));
            test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, { 64, 5, 4, 3 }, v, eps));
        }
        test_cases.emplace_back(new test_rms_norm_back(GGML_TYPE_F32, { 64, 5, 4, 3 }, eps));
    }

    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, { 4, 1536, 1, 1 }, { 4, 1536, 1, 1 }));
    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, { 8, 1536, 1, 1 }, { 4, 1536, 1, 1 }));
    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, { 4, 1536, 4, 1 }, { 4, 1536, 1, 1 }));

    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 16, 1024, 32, 4));

    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 1, 1));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 32, 1));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 32, 4));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 128, 4));

    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 1, 1));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 32, 1));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 32, 4));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 128, 4));

    for (ggml_type type_a : all_types) {
        for (int i = 1; i < 10; ++i) {
            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 16, i, 256, { 1, 1 }, { 1, 1 }));
        }
    }

#if 1
    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
            // test cases without permutation
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 1, 1 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 1, 1 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 1, 1 }, { 1, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 1 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 1 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 2 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 2 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 2 }, { 1, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 3, 2 }, { 2, 2 }));

            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 1, 1 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 1, 1 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 1, 1 }, { 1, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 1 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 1 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 2 }, { 1, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 2 }, { 2, 1 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 2 }, { 1, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 3, 2 }, { 2, 2 }));

            // test cases with permutation
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 2, 3 }, { 1, 1 }, { 0, 2, 1, 3 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 2, 3 }, { 1, 1 }, { 0, 1, 3, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 2, 3 }, { 1, 1 }, { 0, 3, 2, 1 }));

            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 8, 256, { 2, 3 }, { 1, 1 }, { 0, 2, 1, 3 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 8, 256, { 2, 3 }, { 1, 1 }, { 0, 1, 3, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 8, 256, { 2, 3 }, { 1, 1 }, { 0, 3, 2, 1 }));

            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 2, 3 }, { 1, 1 }, { 0, 2, 1, 3 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 2, 3 }, { 1, 1 }, { 0, 1, 3, 2 }));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 2, 3 }, { 1, 1 }, { 0, 3, 2, 1 }));
        }
    }
    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : { GGML_TYPE_F32 }) {
            if (ggml_blck_size(type_a) != 256) {
                test_cases.emplace_back(
                    new test_mul_mat(type_a, type_b, 16, 1, ggml_blck_size(type_a), { 1, 1 }, { 1, 1 }));
            }
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 1, 1 }, { 1, 1 }));
        }
    }
#else
    // m = a rows
    // n = b rows
    // k = cols
    std::uniform_int_distribution<> dist_m(1, 128);
    std::uniform_int_distribution<> dist_n(16, 128);
    std::uniform_int_distribution<> dist_k(1, 16);
    for (int i = 0; i < 1000; i++) {
        for (ggml_type type_a : all_types) {
            for (ggml_type type_b : { GGML_TYPE_F32 }) {
                int m = dist_m(rng);
                int n = dist_n(rng);
                int k = dist_k(rng) * ggml_blck_size(type_a);
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, m, n, k, { 1, 1 }, { 1, 1 }));
            }
        }
    }
#endif

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 2, 128, { 8, 1 }, { 1, 1 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 83, 2, 128, { 8, 1 }, { 4, 1 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 2, 64, { 8, 1 }, { 4, 1 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 83, 2, 64, { 8, 1 }, { 4, 1 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 45, 128, { 8, 1 }, { 4, 1 }));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 128, 45, 64, { 8, 1 }, { 4, 1 }));

    // sycl backend will limit task global_range < MAX_INT
    // test case for f16-type-convert-to-fp32 kernel with large k under fp32 compute dtype (occurs in stable-diffusion)
    // however this case needs to alloc more memory which may fail in some devices (Intel Arc770, etc.)
    // this case is verified (pass) in Intel(R) Data Center GPU Max 1100 (sycl backend) and NV A30 (cuda backend)
    // test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 512, 262144, 9216, {1, 1}, {1, 1}));

    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : { GGML_TYPE_F32 /*, GGML_TYPE_F16 */ }) {
            for (int n_mats : { 4, 8 }) {
                for (int n_used : { 1, 2, 4 }) {
                    for (bool b : { false, true }) {
                        for (int n : { 1, 32 }) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : { GGML_TYPE_F32 /*, GGML_TYPE_F16 */ }) {
            for (int n_mats : { 4 }) {
                for (int n_used : { 2 }) {
                    for (bool b : { false }) {
                        for (int n : { 1, 32 }) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }
    // test_cases.emplace_back(new test_flash_attn_prompt());

    // test_cases.emplace_back(new test_flash_attn_prompt(2,8,64,64,8,8,4,0.125f));
    // test_cases.emplace_back(new test_flash_attn_prompt(2,8,64,64,8,8,2,0.125f));
    // test_cases.emplace_back(new test_flash_attn_prompt(2,8,64,64,8,8,8,0.125f));
    // test_cases.emplace_back(new test_flash_attn_prompt(2,8,64,64,8,8,1,0.125f));

    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
            for (int n : { 1, 16 }) {
                for (int k : { 1, 16 }) {
                    for (int bs2 : { 1, 3 }) {
                        for (int bs3 : { 1, 3 }) {
                            for (int nr2 : { 1, 2 }) {
                                for (int nr3 : { 1, 2 }) {
                                    test_cases.emplace_back(
                                        new test_out_prod(type_a, type_b, 256, n, k, { bs2, bs3 }, { nr2, nr3 }));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        test_cases.emplace_back(new test_sqr(type));
        test_cases.emplace_back(new test_sqrt(type));
        test_cases.emplace_back(new test_log(type));
        test_cases.emplace_back(new test_sin(type));
        test_cases.emplace_back(new test_cos(type));
        test_cases.emplace_back(new test_clamp(type));
    }

    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, { 10, 10, 1, 1 }, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, { 10, 10, 3, 1 }, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, { 10, 10, 3, 2 }, 5));

#if 0
    std::uniform_int_distribution<> dist_ne1(1, 50);
    int exponent = 1;
    while (exponent < (1 << 17)) {
        std::uniform_int_distribution<> dist_ne0(exponent, 2*exponent);

        for (int n = 0; n < 10; ++n) {
            int64_t ne0 = dist_ne0(rng);
            int64_t ne1 = dist_ne1(rng);
            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, GGML_TYPE_F32, {ne0, ne1, 1, 1}, n/2 == 0, 0.1f, ne0 < 1000 ? 4.0f : 0.0f));
        }

        exponent <<= 1;
    }
#endif
    for (bool mask : { false, true }) {
        for (float max_bias : { 0.0f, 8.0f }) {
            if (!mask && max_bias > 0.0f) {
                continue;
            }
            for (float scale : { 1.0f, 0.1f }) {
                for (int64_t ne0 : { 16, 1024 }) {
                    for (int64_t ne1 : { 16, 1024 }) {
                        if (mask) {
                            for (ggml_type m_prec : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
                                test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { ne0, ne1, 1, 1 }, mask,
                                                                          m_prec, scale, max_bias));
                                test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { ne0 - 1, ne1 - 1, 1, 1 },
                                                                          mask, m_prec, scale, max_bias));
                            }
                        } else {
                            /* The precision of mask here doesn't matter as boolean mask is false */
                            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { ne0, ne1, 1, 1 }, mask,
                                                                      GGML_TYPE_F32, scale, max_bias));
                            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { ne0 - 1, ne1 - 1, 1, 1 }, mask,
                                                                      GGML_TYPE_F32, scale, max_bias));
                        }
                    }
                }
            }
        }
    }
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 16, 2, 32, 1 }, true, GGML_TYPE_F32, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 16, 2, 32, 1 }, true, GGML_TYPE_F16, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 16, 2, 32, 1 }, false, GGML_TYPE_F32, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 32, 2, 32, 1 }, true, GGML_TYPE_F32, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 32, 2, 32, 1 }, true, GGML_TYPE_F16, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 32, 2, 32, 1 }, true, GGML_TYPE_F32, 0.1f, 8.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 32, 2, 32, 1 }, true, GGML_TYPE_F16, 0.1f, 8.0f));

    for (float max_bias : { 0.0f, 8.0f }) {
        for (float scale : { 1.0f, 0.1f }) {
            for (int64_t ne0 : { 16, 1024 }) {
                for (int64_t ne1 : { 16, 1024 }) {
                    test_cases.emplace_back(new test_soft_max_back(GGML_TYPE_F32, { ne0, ne1, 1, 1 }, scale, max_bias));
                    test_cases.emplace_back(
                        new test_soft_max_back(GGML_TYPE_F32, { ne0 - 1, ne1 - 1, 1, 1 }, scale, max_bias));
                }
            }
        }
    }

    for (bool fw : { true }) {  // fw == forward
        bool all = true;

        for (float v : { 0 }) {
            for (float fs : { 0.025f }) {
                for (float ef : { 1.0f }) {
                    for (float af : { 0.730520f }) {
                        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
                            for (bool ff : { false }) {  // freq_factors
                                // test_cases.emplace_back(new test_rope(type, {64,  1, 84, 1}, 64, 0, 4096, fs, ef, af, ff, v, fw)); // llama 7B
                                test_cases.emplace_back(new test_rope(type, { 64, 16, 128, 1 }, 64, 0, 4096, fs, ef, af,
                                    ff, v, fw));  // llama 7B
                                    test_cases.emplace_back(new test_rope(type, { 64, 1, 16, 1 }, 64, 0, 4096, fs, ef, af,
                                    ff, v, fw));  // llama 7B
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

    for (int v : { 0, 1, 2, 3 }) {
        for (int dim : {
                 0,
                 1,
                 2,
                 3,
             }) {
            test_cases.emplace_back(new test_concat(GGML_TYPE_F32, { 11, 12, 13, 14 }, 7, dim, v));
            test_cases.emplace_back(new test_concat(GGML_TYPE_I32, { 11, 12, 13, 14 }, 7, dim, v));
        }
    }

    for (ggml_sort_order order : { GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC }) {
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, { 8, 1, 1, 1 }, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, { 16, 10, 10, 10 }, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, { 60, 10, 10, 10 }, order));  // qwen
    }

    test_cases.emplace_back(new test_sum());
    test_cases.emplace_back(new test_sum_rows());
    test_cases.emplace_back(new test_mean());
    test_cases.emplace_back(new test_upscale());
    test_cases.emplace_back(new test_upscale(GGML_TYPE_F32, { 512, 512, 3, 1 }, 2, true));
    test_cases.emplace_back(new test_upscale_ext());
    test_cases.emplace_back(new test_group_norm(GGML_TYPE_F32, { 64, 64, 320, 1 }));
    test_cases.emplace_back(new test_group_norm(GGML_TYPE_F32, { 9, 9, 1280, 1 }));
    test_cases.emplace_back(new test_acc());
    test_cases.emplace_back(new test_pad());
    test_cases.emplace_back(new test_pad_reflect_1d());
    test_cases.emplace_back(new test_arange());
    test_cases.emplace_back(new test_timestep_embedding());
    test_cases.emplace_back(new test_leaky_relu());

    for (int hs : {
             64,
             80,
             128,
             256,
         }) {
        for (bool mask : { true, false }) {
            for (float max_bias : { 0.0f, 8.0f }) {
                if (!mask && max_bias > 0.0f) {
                    continue;
                }
                for (float logit_softcap : { 0.0f, 10.0f }) {
                    if (hs != 128 && logit_softcap != 0.0f) {
                        continue;
                    }
                    for (int nh : {
                             4,
                         }) {
                        for (int nr : { 1, 4, 16 }) {
                            if (nr == 16 && hs != 128) {
                                continue;
                            }
                            for (int kv : {
                                     512,
                                     1024,
                                 }) {
                                if (nr != 1 && kv != 512) {
                                    continue;
                                }
                                for (int nb : {
                                         1,
                                         3,
                                         32,
                                         35,
                                     }) {
                                    for (ggml_type type_KV :
                                         { GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
                                        test_cases.emplace_back(new test_flash_attn_ext(
                                            hs, nh, nr, kv, nb, mask, max_bias, logit_softcap, type_KV));
                                        // run fewer test cases permuted
                                        if (mask == true && max_bias == 0.0f && logit_softcap == 0 && kv == 512) {
                                            test_cases.emplace_back(new test_flash_attn_ext(hs, nh, nr, kv, nb, mask,
                                                                                            max_bias, logit_softcap,
                                                                                            type_KV, { 0, 2, 1, 3 }));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }



    // these tests are disabled to save execution time, but they can be handy for debugging
    test_cases.emplace_back(new test_llama(1));
#if 0
    test_cases.emplace_back(new test_llama(1));
    test_cases.emplace_back(new test_llama(2));
    test_cases.emplace_back(new test_falcon(1));
    test_cases.emplace_back(new test_falcon(2));
#endif

    return test_cases;
}

// Test cases for performance evaluation: should be representative of real-world use cases
static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {
    std::vector<std::unique_ptr<test_case>> test_cases;

    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, { 4096, 1, 1, 1 }, { 1, 1, 1, 1 }));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, { 4096, 1, 1, 1 }, { 1, 512, 1, 1 }));

    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F16, { 512, 3072, 1, 1 }));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, { 8192, 512, 2, 1 }, { 0, 2, 1, 3 }));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, { 3072, 512, 2, 1 }, { 0, 2, 1, 3 }));

    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 4096, 4096, 5, 1 }, false, GGML_TYPE_F32, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 77, 4096, 5, 1 }, false, GGML_TYPE_F32, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 1024, 1024, 10, 1 }, false, GGML_TYPE_F32, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 77, 1024, 10, 1 }, false, GGML_TYPE_F32, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 256, 256, 20, 1 }, false, GGML_TYPE_F32, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 64, 64, 20, 1 }, false, GGML_TYPE_F32, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 77, 64, 20, 1 }, false, GGML_TYPE_F32, 1.0f, 0.0f));

    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 32, 10, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 1024, 10, 1, 1 }));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, { 32000, 512, 1, 1 }));

    for (int bs : { 1, 2, 3, 4, 5, 8, 512 }) {
        for (ggml_type type_a : all_types) {
            for (ggml_type type_b : { GGML_TYPE_F32 }) {
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 4096, bs, 14336, { 1, 1 }, { 1, 1 }));
            }
        }
    }

    for (int K : { 3, 5 }) {
        for (int IC : { 256, 2560 }) {
            for (int IW_IH : { 32, 64, 256 }) {
                if (IC == 2560 && IW_IH == 256) {
                    // too big
                    continue;
                }
                test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32,
                                                        { IW_IH, IW_IH, IC, 1 }, { K, K, IC, 1 }, 1, 1, 1, 1, 1, 1,
                                                        true));
            }
        }
    }

    // test_cases.emplace_back(new test_flash_attn_prompt());

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
        ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        if (backend_cpu == NULL) {
            printf("  Failed to initialize CPU backend\n");
            return false;
        }

        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->eval(backend, backend_cpu, op_name)) {
                n_ok++;
            }
        }
        printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());

        ggml_backend_free(backend_cpu);

        return n_ok == test_cases.size();
    }

    if (mode == MODE_SPEED) {
        auto test_cases = make_test_cases_eval();
        filter_test_cases(test_cases, params_filter);
        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->eval_speed(backend, op_name)) {
                n_ok++;
            }
        }
        printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());


        return n_ok == test_cases.size();
    }

    if (mode == MODE_GRAD) {
        auto test_cases = make_test_cases_eval();
        filter_test_cases(test_cases, params_filter);
        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->eval_grad(backend, op_name)) {
                n_ok++;
            }
        }
        printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());

        return n_ok == test_cases.size();
    }

    if (mode == MODE_PERF) {
        auto test_cases = make_test_cases_perf();
        filter_test_cases(test_cases, params_filter);
        for (auto & test : test_cases) {
            test->eval_perf(backend, op_name);
        }
        return true;
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