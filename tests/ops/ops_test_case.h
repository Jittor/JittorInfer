#pragma once
#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>
#include <type_traits>
#include <vector>

void init_tensor(ggml_tensor* tensor, float min = -1.0f, float max = 1.0f);

struct test_case{
    ggml_context* ctx;
    std::vector <ggml_tensor*> src;
    int64_t output_size;

    std::string name;
    // 初始化算子源矩阵
    virtual void init_src_size()=0;

    virtual void init_tensor(ggml_tensor* tensor);
    virtual void init_tensors();
    virtual void special_check_cpu(ggml_tensor* output){}
    // 构建计算图关系
    virtual ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const=0;
    virtual ggml_tensor* build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
    // 输出
    void print(){}
    // 初始化矩阵
    
    test_case(std::string name);
    virtual ~test_case();
};

struct test_case_add:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_add(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("ADD"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_mul:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_mul(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("MUL"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_sub:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_sub(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("SUB"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_div:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_div(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("DIV"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_sqr:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_sqr(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("SQR"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_sum_rows:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_sum_rows(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("SUM_ROWS"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_acc:test_case{
    int a_dim1,a_dim2,a_dim3,a_dim4,
    b_dim1,b_dim2,b_dim3,b_dim4,
    nb1,nb2,nb3,offset;
    test_case_acc(int a_dim1 = 256,int a_dim2 = 17,int a_dim3 = 1, int a_dim4 = 1,
        int b_dim1 = 256,int b_dim2 = 16,int b_dim3 = 1, int b_dim4 = 1):
        a_dim1(a_dim1),a_dim2(a_dim2),a_dim3(a_dim3),a_dim4(a_dim4),
        b_dim1(b_dim1),b_dim2(b_dim2),b_dim3(b_dim3),b_dim4(b_dim4),
        test_case("ACC"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_norm:test_case{
    int dim1,dim2,dim3,dim4;
    float eps;
    test_case_norm(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4,float eps = 1e-6f):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),eps(eps),test_case("NORM"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_group_norm:test_case{
    int dim1,dim2,dim3,dim4;
    int32_t num_group;
    float eps;
    test_case_group_norm(int dim1 = 64,int dim2 = 64,int dim3 =320, int dim4 = 1,int32_t n_group = 32,float eps = 1e-6f):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),num_group(n_group),eps(eps),test_case("GROUP_NORM"){}
    void init_src_size();
    void init_tensor(ggml_tensor* tensor) override;
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_concat:test_case{
    std::vector<int> ne_a;
    int ne_b_d;
    int dim;
    test_case_concat(int dim1 = 64,int dim2 = 64,int dim3 =5, int dim4 = 5, int ne_b_d = 6, int dim = 2):
        ne_a({dim1,dim2,dim3,dim4}),dim(dim),ne_b_d(ne_b_d),test_case("CONCAT"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_upscale:test_case{
    std::vector<int> ne_a;
    int scale_factor;
    test_case_upscale(std::vector<int> ne1 = {16, 16, 4, 4},int scale_factor = 2):
        ne_a(std::move(ne1)),scale_factor(scale_factor),test_case("UPSCALE"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_pad:test_case{
    std::vector<int> ne_a;
    std::vector<int> pad;
    test_case_pad(std::vector<int> ne1 = {16, 16, 4, 4},std::vector<int> pad = {5, 5, 5, 5}):
        ne_a(std::move(ne1)),pad(std::move(pad)),test_case("PAD"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_arange:test_case{
    float start, stop, step;
    test_case_arange(float start = 0.0, float stop = 640.0, float step = 5):
        start(start),stop(stop),step(step),test_case("PAD"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_gelu:test_case{
    std::vector<int> ne_a;
    test_case_gelu(std::vector<int> ne1 = {256, 128, 4, 4}):
        ne_a(std::move(ne1)),test_case("GELU"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_gelu_quick:test_case{
    std::vector<int> ne_a;
    test_case_gelu_quick(std::vector<int> ne1 = {256, 128, 4, 4}):
        ne_a(std::move(ne1)),test_case("GELU_QUICK"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_silu:test_case{
    std::vector<int> ne_a;
    test_case_silu(std::vector<int> ne1 = {256, 128, 4, 4}):
        ne_a(std::move(ne1)),test_case("SILU"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_tanh:test_case{
    std::vector<int> ne_a;
    test_case_tanh(std::vector<int> ne1 = {256, 128, 4, 4}):
        ne_a(std::move(ne1)),test_case("TANH"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_relu:test_case{
    std::vector<int> ne_a;
    test_case_relu(std::vector<int> ne1 = {256, 128, 4, 4}):
        ne_a(std::move(ne1)),test_case("RELU"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_hardsigmoid:test_case{
    std::vector<int> ne_a;
    test_case_hardsigmoid(std::vector<int> ne1 = {256, 128, 4, 4}):
        ne_a(std::move(ne1)),test_case("HARDSIGMOID"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_hardswish:test_case{
    std::vector<int> ne_a;
    test_case_hardswish(std::vector<int> ne1 = {256, 128, 4, 4}):
        ne_a(std::move(ne1)),test_case("HARDSWISH"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_timestep_embedding:test_case{
    std::vector<int> ne_a;
    int dim, actual_dim;
    int max_period;
    test_case_timestep_embedding(std::vector<int> ne1 = {32, 1, 1, 1},int dim = 320,int max_period = 10000):
        ne_a(std::move(ne1)),dim(dim),actual_dim(dim),max_period(max_period),test_case("TIMESTEP_EMBEDDING"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_rms_norm_fused:test_case{
    std::vector<int> ne_a;
    float eps;
    test_case_rms_norm_fused(std::vector<int> ne1 = {10, 10, 2, 3},float eps = 1e-6):
        ne_a(std::move(ne1)),eps(eps),test_case("RMS_NORM_FUSED"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
    ggml_tensor* build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const override;
};
struct test_case_leaky_relu:test_case{
    std::vector<int> ne_a;
    float negative_slope;
    test_case_leaky_relu(std::vector<int> ne1 = {256, 128, 4, 4}, float negative_slope = 0.1):
        ne_a(std::move(ne1)),negative_slope(negative_slope),test_case("LEAKY_RELU"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_rms_norm:test_case{
    std::vector<int> ne_a;
    float eps;
    test_case_rms_norm(std::vector<int> ne1 = {256, 128, 4, 4},float eps = 1e-6):
        ne_a(std::move(ne1)),eps(eps),test_case("RMS_NORM"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_mul_mat:test_case{
    int m,k,n;
    std::vector<int> bs;
    std::vector<int> nr;
    test_case_mul_mat(int m = 16,int n = 16,int k = 32,std::vector<int> bs = {10, 10},std::vector<int> nr = {2, 2}):
        m(m),n(n),k(k),bs(std::move(bs)),nr(std::move(nr)),test_case("MUL_MAT"){}
    void init_src_size();
    void special_check_cpu(ggml_tensor* output) override;
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_mul_mat_id:test_case{
    int m,k,n;
    int n_expert,n_used;
    test_case_mul_mat_id(int m = 32,int n = 32,int k = 32,int n_expert = 8, int n_used = 2):
        m(m),n(n),k(k),n_expert(n_expert),n_used(n_used),test_case("MUL_MAT_ID"){}
    void init_src_size();
    void init_tensors() override;
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_scale:test_case{
    std::vector<int> ne;
    float scale_factor;
    test_case_scale(std::vector<int> ne = {64, 64, 5, 5}, float scale_factor= 2.0):
        ne(std::move(ne)),scale_factor(scale_factor),test_case("SCALE"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_clamp:test_case{
    std::vector<int> ne;
    float min,max;
    test_case_clamp(std::vector<int> ne = {64, 64, 5, 5}, float min = -0.5, float max = 0.5):
        ne(std::move(ne)), min(min), max(max), test_case("CLAMP"){}
    void init_src_size();
    void init_tensor(ggml_tensor* tensor) override;
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_cpy:test_case{
    std::vector<int> ne;
    test_case_cpy(std::vector<int> ne = {64, 64, 5, 5}):
        ne(std::move(ne)), test_case("CPY"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_cont:test_case{
    std::vector<int> ne;
    test_case_cont(std::vector<int> ne = {64, 64, 5, 5}):
        ne(std::move(ne)), test_case("CONT"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_diag_mask_inf:test_case{
    std::vector<int> ne;
    int n_past;
    test_case_diag_mask_inf(std::vector<int> ne = {64, 64, 5, 5}, int n_past = 5):
        ne(std::move(ne)), n_past(n_past), test_case("DIAG_MASK_INF"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_soft_max:test_case{
    std::vector<int> ne;
    std::vector<int> nr;
    bool mask;
    float scale;
    float max_bias;
    test_case_soft_max(std::vector<int> ne = {64, 64, 5, 5}, std::vector<int> nr = {1,1}, bool mask = false, float scale = 1.0f,float max_bias = 0.0f):
        ne(std::move(ne)), nr(std::move(nr)), mask(mask), scale(scale), max_bias(max_bias), test_case("SOFT_MASK"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_rope:test_case{
    std::vector<int> ne_a;
    std::vector<int> nr;
    int n_dims, n_ctx, mode;
    bool ff;
    float fs, ef, af;
    test_case_rope(std::vector<int> ne_a = {32, 5, 3, 1}, int mode = 0, bool ff = false,
         int n_dims = 32, int n_ctx = 512, float fs = 1.0f, float ef = 0.0f, float af = 1.0f):
        ne_a(std::move(ne_a)), mode(mode), ff(ff), n_dims(n_dims), n_ctx(n_ctx), fs(fs), ef(ef), af(af), test_case("ROPE"){}
    void init_src_size();
    void init_tensors() override;
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_im2col:test_case{
    std::vector<int64_t> ne_input;
    std::vector<int64_t> ne_kernel;
    int s0,s1,p0,p1,d0,d1;
    bool is_2d;
    ggml_type dst_type;
    test_case_im2col(std::vector<int64_t> ne_input = {32, 32, 5, 1}, std::vector<int64_t> ne_kernel = {3,3,5,1}, ggml_type dst_type = GGML_TYPE_F32,
        int s0 = 1, int s1 = 1, int p0 = 1, int p1 = 1, int d0 = 1, int d1 = 1,
        bool is_2d = true):
        ne_input(std::move(ne_input)), ne_kernel(std::move(ne_kernel)),s0(s0),s1(s1),p0(p0),p1(p1),d0(d0),d1(d1),is_2d(is_2d),dst_type(dst_type),test_case("IM2COL"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_pool_2d:test_case{
    std::vector<int64_t> ne_input;
    int k0,k1,s0,s1;
    float p0,p1;
    ggml_op_pool pool_type;
    test_case_pool_2d(std::vector<int64_t> ne_input = {32, 32, 5, 1}, ggml_op_pool pool_type = GGML_OP_POOL_AVG,
        int k0 = 3, int k1 = 3, int s0 = 1, int s1 = 1, float p0 = 1, float p1 = 1
        ):
        ne_input(std::move(ne_input)),k0(k0),k1(k1),s0(s0),s1(s1),p0(p0),p1(p1),pool_type(pool_type),test_case("POOL_2D"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_argsort:test_case{
    std::vector<int64_t> ne;
    ggml_sort_order order = GGML_SORT_ORDER_ASC;
    test_case_argsort(std::vector<int64_t> ne = {32, 32, 5, 1}, ggml_sort_order order = GGML_SORT_ORDER_ASC):
        ne(std::move(ne)), order(order), test_case("ARGSORT"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_to_zero:test_case{
    std::vector<int64_t> ne;
    test_case_to_zero(std::vector<int64_t> ne = {32, 32, 16, 4}):
        ne(std::move(ne)), test_case("TO_ZERO"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
    ggml_tensor* build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const override;
};
struct test_case_moe_fused:test_case{
    int32_t start_idx;
    int32_t end_idx;
    int32_t batch_size;
    int32_t seq_len;
    int32_t topk;
    int32_t num_experts;
    int32_t hidden_dim;
    int32_t k_dim;
    test_case_moe_fused(int start_idx = 0, int end_idx = 20, int seq_len = 32, int topk = 8, 
        int hidden_dim = 2, int k_dim = 10):
        start_idx(start_idx), end_idx(end_idx), batch_size(1), seq_len(seq_len), topk(topk), 
        num_experts(end_idx - start_idx + 1), hidden_dim(hidden_dim), k_dim(k_dim),test_case("MOE_FUSED"){}
    void init_src_size();
    void init_tensors() override;
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
    ggml_tensor* build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const override;
};
struct test_case_all_reduce_sum:test_case{
    std::vector<int64_t> ne;
    test_case_all_reduce_sum(std::vector<int64_t> ne = {32, 32, 16, 4}):
        ne(std::move(ne)), test_case("ALL_REDUCE_SUM"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
    ggml_tensor* build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const override;
};
struct test_case_get_slice:test_case{
    std::vector<int64_t> ne_a;
    int from,to,axis;
    test_case_get_slice(std::vector<int64_t> ne_a = {10, 10, 16, 5}, int from = 2, int to = 4, int axis = 3):
        ne_a(std::move(ne_a)), from(from), to(to), axis(axis), test_case("GET_SLICE"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
    ggml_tensor* build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const override;
};
struct test_case_scatter_update:test_case{
    std::vector<int64_t> ne_a;
    int n_updates;
    test_case_scatter_update(std::vector<int64_t> ne_a = {2, 5}, int n_updates = 3):
        ne_a(std::move(ne_a)), n_updates(n_updates), test_case("SCATTER_UPDATE"){}
    void init_src_size();
    void init_tensors() override;
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
    ggml_tensor* build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const override;
};
struct test_case_flash_attn_jittor:test_case{
    int64_t batch_size;
    int64_t num_heads;
    int64_t head_dims_kq;
    int64_t head_dims_v;
    int64_t key_num_heads;
    int64_t sequence_lenth_q;
    int64_t sequence_lenth_kv;
    float scalevalue;
    test_case_flash_attn_jittor(int64_t batch_size = 1,int64_t num_heads = 16, int64_t head_dims_kq = 256, 
        int64_t head_dims_v = 256, int64_t key_num_heads = 16, int64_t sequence_lenth_q = 64, int64_t sequence_lenth_kv = 128, float scalevalue = 1.0f):
        batch_size(batch_size), num_heads(num_heads), head_dims_kq(head_dims_kq), head_dims_v(head_dims_v), key_num_heads(key_num_heads), sequence_lenth_kv(sequence_lenth_kv),sequence_lenth_q(sequence_lenth_q), scalevalue(scalevalue/std::sqrt(static_cast<float>(head_dims_kq))),
        test_case("FLASH_ATTEN_JITTOR"){}
    void init_src_size();
    void init_tensors() override;
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
    ggml_tensor* build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const override;
};
struct test_case_flash_attn_prompt:test_case{
    int64_t batch_size;
    int64_t num_heads;
    int64_t head_dims_kq;
    int64_t head_dims_v;
    int64_t key_num_heads;
    int64_t sequence_lenth_q;
    int64_t sequence_lenth_kv;
    float scalevalue;
    test_case_flash_attn_prompt(int64_t batch_size = 1,int64_t num_heads = 16, int64_t head_dims_kq = 256, 
        int64_t head_dims_v = 256, int64_t key_num_heads = 16, int64_t sequence_lenth_q = 64, int64_t sequence_lenth_kv = 128, float scalevalue = 1.0f):
        batch_size(batch_size), num_heads(num_heads), head_dims_kq(head_dims_kq), head_dims_v(head_dims_v), key_num_heads(key_num_heads), sequence_lenth_kv(sequence_lenth_kv),sequence_lenth_q(sequence_lenth_q), scalevalue(scalevalue/std::sqrt(static_cast<float>(head_dims_kq))),
        test_case("FLASH_ATTEN_PROMPT"){}
    void init_src_size();
    void init_tensors() override;
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
    ggml_tensor* build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const override;
};