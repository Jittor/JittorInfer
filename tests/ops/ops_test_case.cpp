#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include "ops_test_case.h"
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>
#include <utility>
#include <vector>
#include <algorithm>

const int max_graph_nodes = 128;
void init_tensor_imp(ggml_tensor* tensor, float min, float max){
    size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    for (auto &val : data){
        val = dis(gen);
    }
    std::cout << "Initializing tensor sample (first few values):" << std::endl;
    for (int i = 0; i < std::min(static_cast<size_t>(10), data.size()); i++) {
        std::cout << std::fixed << std::setprecision(4) << static_cast<float>(data[i]) << " ";
    }
    std::cout << std::endl;
    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        memcpy(tensor->data, data.data(), data.size() * sizeof(float));
    }else{
        // unimplemented
    }
}
void test_case::init_tensor(ggml_tensor* tensor){
    init_tensor_imp(tensor, -1.0, 1.0);
}
void test_case::init_tensors(){
    for(auto &val: src){
        if(val != nullptr){
            init_tensor(val);
        }
    }
}
test_case::test_case(std::string name):name(std::move(name)){
    struct ggml_init_params init_params = {
    /* .mem_size */   ggml_tensor_overhead() * 32 + 1024 * 1024 * 128,  // 分配足够的内存
    /* .mem_buffer */ NULL,
    /* .no_alloc */   false
    };
    ctx = ggml_init(init_params);
    if (!ctx) {
        std::cerr << "Failed to initialize GGML context" << std::endl;
        return;
    }
    output_size = 0;
}
test_case::~test_case(){
    ggml_free(ctx);
}
ggml_tensor* test_case::build_graph_cpu(ggml_context *compute_ctx, std::vector<ggml_tensor *> &src_tensors)const {
    return build_graph(compute_ctx, src_tensors);
}
void test_case_add::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_add::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_add(compute_ctx,src_tensors[0],src_tensors[1]);
}
void test_case_sub::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_sub::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_sub(compute_ctx,src_tensors[0],src_tensors[1]);
}
void test_case_mul::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_mul::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_mul(compute_ctx,src_tensors[0],src_tensors[1]);
}
void test_case_div::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_div::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_div(compute_ctx,src_tensors[0],src_tensors[1]);
}
void test_case_sqr::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_sqr::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_sqr(compute_ctx,src_tensors[0]);
}
void test_case_sum_rows::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = 1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_sum_rows::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_sum_rows(compute_ctx,src_tensors[0]);
}
void test_case_acc::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a_dim1, a_dim2, a_dim3, a_dim4));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b_dim1, b_dim2, b_dim3, b_dim4));
    output_size = a_dim1 * a_dim2 * a_dim3 * a_dim4;
}
ggml_tensor* test_case_acc::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_acc(compute_ctx,src_tensors[0],src_tensors[1],
        src_tensors[0]->nb[1],src_tensors[0]->nb[2],src_tensors[0]->nb[3],src_tensors[1]->nb[1]);
}
void test_case_norm::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_norm::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_norm(compute_ctx,src_tensors[0],eps);
}
void test_case_group_norm::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
void test_case_group_norm::init_tensor(ggml_tensor *tensor){
    init_tensor_imp(tensor, -10.0, 10.0);
}
ggml_tensor* test_case_group_norm::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_group_norm(compute_ctx,src_tensors[0],num_group,eps);
}
void test_case_concat::init_src_size(){
    assert(src.empty());
    std::vector<int> ne_b = ne_a;
    std::vector<int> ne_c = ne_a;
    ne_b[dim] = ne_b_d;
    ne_c[dim] += ne_b_d;
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_b[0],ne_b[1],ne_b[2],ne_b[3]));
    output_size = ne_c[0] * ne_c[1] * ne_c[2] * ne_c[3];
}
ggml_tensor* test_case_concat::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_concat(compute_ctx,src_tensors[0],src_tensors[1],dim);
}
void test_case_upscale::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    output_size = ne_a[0] * scale_factor * ne_a[1] * scale_factor * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_upscale::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_upscale(compute_ctx,src_tensors[0],scale_factor);
}
void test_case_pad::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    output_size = (ne_a[0] + pad[0]) * (ne_a[1] + pad[1]) * (ne_a[2] + pad[2]) * (ne_a[3] + pad[3]);
}
ggml_tensor* test_case_pad::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_pad(compute_ctx,src_tensors[0],pad[0],pad[1],pad[2],pad[3]);
}
void test_case_arange::init_src_size(){
    assert(src.empty());
    output_size = (int64_t) ceilf((stop - start) / step);
}
ggml_tensor* test_case_arange::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_arange(compute_ctx, start, stop, step);
}
void test_case_gelu::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    output_size = ne_a[0] * ne_a[1] * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_gelu::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_gelu(compute_ctx, src_tensors[0]);
}
void test_case_gelu_quick::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    output_size = ne_a[0] * ne_a[1] * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_gelu_quick::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_gelu_quick(compute_ctx, src_tensors[0]);
}
void test_case_silu::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    output_size = ne_a[0] * ne_a[1] * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_silu::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_silu(compute_ctx, src_tensors[0]);
}
void test_case_tanh::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    output_size = ne_a[0] * ne_a[1] * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_tanh::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_tanh(compute_ctx, src_tensors[0]);
}
void test_case_relu::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    output_size = ne_a[0] * ne_a[1] * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_relu::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_relu(compute_ctx, src_tensors[0]);
}
void test_case_hardsigmoid::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    output_size = ne_a[0] * ne_a[1] * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_hardsigmoid::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_hardsigmoid(compute_ctx, src_tensors[0]);
}
void test_case_hardswish::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    output_size = ne_a[0] * ne_a[1] * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_hardswish::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_hardswish(compute_ctx, src_tensors[0]);
}
void test_case_timestep_embedding::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    if (dim % 2 != 0) {
        actual_dim = dim + 1;
    }
    output_size = actual_dim * ne_a[0];
}
ggml_tensor* test_case_timestep_embedding::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_timestep_embedding(compute_ctx, src_tensors[0],dim,max_period);
}
void test_case_rms_norm_fused::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],1,1,1));
    output_size = ne_a[0] * ne_a[1] * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_rms_norm_fused::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_rms_norm_fused(compute_ctx, src_tensors[0],src_tensors[1],eps);
}
ggml_tensor* test_case_rms_norm_fused::build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    // rms norm
    ggml_tensor* rms_norm = ggml_rms_norm(compute_ctx, src_tensors[0], eps);
    // mul gamma
    ggml_tensor* output = ggml_mul(compute_ctx, rms_norm, src_tensors[1]);
    return output;
}
void test_case_leaky_relu::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    output_size = ne_a[0] * ne_a[1] * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_leaky_relu::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_leaky_relu(compute_ctx, src_tensors[0], negative_slope, false);
}
void test_case_rms_norm::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    output_size = ne_a[0] * ne_a[1] * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_rms_norm::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_rms_norm(compute_ctx, src_tensors[0],eps);
}
std::vector<float> mul_mat(ggml_tensor* a, ggml_tensor* b){
    int k = a->ne[0];
    int m = a->ne[1];
    int n = b->ne[1];
    long long nb_a[4] = {1, a->ne[0], a->ne[0] * a->ne[1], a->ne[0] * a->ne[1] * a->ne[2]};
    long long nb_b[4] = {1, b->ne[0], b->ne[0] * b->ne[1], b->ne[0] * b->ne[1] * b->ne[2]};
    long long nb_c[4] = {1, m, m * n, m * n * b->ne[2]};
    std::vector<float> data_a(ggml_nelements(a));
    std::vector<float> data_b(ggml_nelements(b));
    memcpy(data_a.data(), a->data, data_a.size() * sizeof(float));
    memcpy(data_b.data(), b->data, data_b.size() * sizeof(float));
    std::vector<float> ans(m*n*b->ne[2]*b->ne[3]);
    for(int t1_b = 0; t1_b < b->ne[2]; t1_b++){
        int t1_a = t1_b / (b->ne[2] / a->ne[2]);
        for(int t2_b = 0; t2_b < b->ne[3]; t2_b++){
            int t2_a = t2_b / (b->ne[3] / a->ne[3]);
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n;j++){
                    float val = 0.0f;
                    for(int t = 0; t < k; t++){
                        val += data_a[t + i * k + t1_a * nb_a[2] + t2_a * nb_a[3]] * data_b[t + j * k + t1_b * nb_b[2] + t2_b * nb_b[3]];
                    }
                    ans[i + j * m + t1_b * nb_c[2] + t2_b * nb_c[3]] = val;
                }
            }
        }
    }
    return ans;
}
void test_case_mul_mat::special_check_cpu(ggml_tensor* output){
    std::vector<float> ans_host = mul_mat(src[0],src[1]);
    std::vector<float> ans_cpu(output_size);
    memcpy(ans_cpu.data(), output->data, output_size*sizeof(float));
    float max_error = 0.0f;
    for(int i = 0; i < output_size; ++i){
        max_error = std::max(max_error, std::abs(ans_cpu[i] - ans_host[i]));
    }
    std::cout<<"cpu special check, max error :" << max_error << std::endl;
}
void test_case_mul_mat::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k, m, bs[0], bs[1]));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k, n, bs[0] * nr[0], bs[1] * nr[1]));
    output_size = m * n * bs[0] * nr[0] * bs[1] * nr[1];
}
ggml_tensor* test_case_mul_mat::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_mul_mat(compute_ctx, src_tensors[0],src_tensors[1]);
}
void test_case_mul_mat_id::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, n_expert)); // as
    src.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n)); //ids
    src.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n)); // b
    output_size = m * n_used * n;
}
void test_case_mul_mat_id::init_tensors(){
    for(auto &val: src){
        if(val->type == GGML_TYPE_I32){
            // ids
            std::random_device rd;
            std::default_random_engine rng(rd());
            std::vector<int32_t> data(val->ne[0] * val->ne[1]);
            for (int64_t r = 0; r < ggml_nrows(val); r++) {
                for (int i = 0; i < val->ne[0]; i++) {
                    data[i + r * val->ne[0]] = (i + r * val->ne[0]) % n_expert;
                }
                std::shuffle(data.begin() + r * val->ne[0], data.begin() + (r + 1) * val->ne[0], rng);
            }
            memcpy(val->data, data.data(), data.size() * sizeof(int32_t));
            // print to check 
            std::vector<int32_t> data_copy(ggml_nelements(val));
            memcpy(data_copy.data(), val->data, data_copy.size()*sizeof(int32_t));
            std::cout << "Initializing tensor sample (first few values):" << std::endl;
            for (int i = 0; i < std::min(static_cast<size_t>(10), data.size()); i++) {
                std::cout << static_cast<int32_t>(data[i]) << " ";
            }
            std::cout << std::endl;
        }else{
            init_tensor(val);
        }
    }
}
ggml_tensor* test_case_mul_mat_id::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_mul_mat_id(compute_ctx, src_tensors[0],src_tensors[2], src_tensors[1]);
}
void test_case_scale::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]));
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_scale::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_scale(compute_ctx, src_tensors[0],scale_factor);
}
void test_case_clamp::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]));
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_clamp::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_clamp(compute_ctx, src_tensors[0],min,max);
}
void test_case_clamp::init_tensor(ggml_tensor* tensor){
    init_tensor_imp(tensor, -10.0, 10.0);
}
void test_case_cpy::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]));
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_cpy::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_cpy(compute_ctx, src_tensors[0], src_tensors[1]);
}
void test_case_cont::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]));
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_cont::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    ggml_tensor* tanspose = ggml_transpose(compute_ctx, src_tensors[0]); // test with transpose tensor
    return ggml_cont(compute_ctx, tanspose);
}
void test_case_diag_mask_inf::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]));
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_diag_mask_inf::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_diag_mask_inf(compute_ctx, src_tensors[0], n_past);
}
void test_case_soft_max::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2] * nr[0], ne[3] * nr[1]));
    if(mask){
        src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]));
    }
    output_size = ne[0] * ne[1] * ne[2] * nr[0] * ne[3] * nr[1];
}
ggml_tensor* test_case_soft_max::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    if(mask){
        return ggml_soft_max_ext(compute_ctx, src_tensors[0], src_tensors[1],scale,max_bias);
    }
    return ggml_soft_max_ext(compute_ctx, src_tensors[0], nullptr, scale,max_bias);
}
void test_case_rope::init_src_size(){
    assert(src.empty());
    bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0], ne_a[1], ne_a[2], ne_a[3])); // a
    // if(is_mrope || is_vision){
    //     src.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2] * 4)); // pos
    // }else{
        src.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2])); // pos
    // }
    if(ff){
        src.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims/2));
    }
    output_size = ne_a[0] * ne_a[1] * ne_a[2] * ne_a[3];
}
ggml_tensor* test_case_rope::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    ggml_tensor* out = nullptr;
    ggml_tensor* freq = nullptr;
    bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    // if(ff){
    //     freq = src_tensors[2];
    // }
    // if(is_mrope){
    //     if(is_vision){
    //         GGML_ASSERT(n_dims/4 > 0);
    //         int rope_sections[4] = {n_dims/4, n_dims/4, 0, 0}; // Vision-RoPE only use first two dimension for image (x, y) coordinate
    //         out = ggml_rope_multi(compute_ctx, src_tensors[0], src_tensors[1], freq, n_dims/2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 0.0f, 0.0f);
    //     }else{
    //         GGML_ASSERT(n_dims/3 > 0);
    //         int rope_sections[4] = {n_dims/3, n_dims/3, n_dims/3, 0};
    //         out = ggml_rope_multi(compute_ctx, src_tensors[0], src_tensors[1], freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 0.0f, 0.0f);
    //     }
    // }else{
    //     out = ggml_rope_ext(compute_ctx, src_tensors[0], src_tensors[1], NULL, n_dims, mode, 0, 10000.0f, fs, ef, af, 0.0f, 0.0f);
    // }
    out = ggml_rope(compute_ctx, src_tensors[0], src_tensors[1], n_dims, mode); 
    return out;
}
void test_case_rope::init_tensors(){
    for(auto &val: src){
        if(val->type == GGML_TYPE_I32){
            // pose
            std::vector<int32_t> data(ggml_nelements(val));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int32_t> dis(0, n_ctx - 1);
            for (int i = 0; i < data.size(); ++i) {
                data[i] = dis(gen);
            }
            memcpy(val->data, data.data(), data.size() * sizeof(int32_t));
            // print to check 
            std::vector<int32_t> data_copy(ggml_nelements(val));
            memcpy(data_copy.data(), val->data, data_copy.size()*sizeof(int32_t));
            std::cout << "Initializing tensor sample (first few values):" << std::endl;
            for (int i = 0; i < std::min(static_cast<size_t>(10), data.size()); i++) {
                std::cout << static_cast<int32_t>(data[i]) << " ";
            }
            std::cout << std::endl;
        }else{
            if (val->ne[0] == n_dims/2) {
                // frequency factor
                init_tensor_imp(val, 0.9f, 1.1f);
            }else{
                init_tensor(val);
            }
        }
    }
}
int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
    return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}
void test_case_im2col::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_kernel.data()));
    src.push_back(ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data()));
    const int64_t OH = is_2d ? ggml_calc_conv_output_size(ne_input[1], ne_kernel[1], s1, p1, d1) : 0;
    const int64_t OW = ggml_calc_conv_output_size(ne_input[0], ne_kernel[0], s0, p0, d0);
    const int64_t ne[4] = {
        is_2d ? (ne_kernel[2] * ne_kernel[1] * ne_kernel[0]) : ne_kernel[1] * ne_kernel[0],
        OW,
        is_2d ? OH : ne_input[2],
        is_2d ? ne_input[3] : 1,
    };
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_im2col::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_im2col(compute_ctx, src_tensors[0], src_tensors[1], s0,s1,p0,p1,d0,d1,is_2d,dst_type);
}
int64_t ggml_calc_pool_output_size(int64_t ins, int ks, int s, float p) {
    return (ins + 2 * p - ks) / s + 1;
}
void test_case_pool_2d::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data()));
    const int64_t ne[4] = {
        ggml_calc_pool_output_size(ne_input[0], k0, s0, p0),
        ne_input[1],
        ne_input[2],
        ne_input[3],
    };
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_pool_2d::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_pool_2d(compute_ctx, src_tensors[0], pool_type, k0,k1,s0,s1,p0,p1);
}
void test_case_argsort::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data()));
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_argsort::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_argsort(compute_ctx, src_tensors[0], order);
}
void test_case_to_zero::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data()));
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_to_zero::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_to_zero(compute_ctx, src_tensors[0]);
}
ggml_tensor* test_case_to_zero::build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    ggml_tensor* output = ggml_new_tensor(compute_ctx, src_tensors[0]->type, 4, src_tensors[0]->ne);
    init_tensor_imp(output, 0.0, 0.0);
    return output;
}
void test_case_moe_fused::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden_dim, 1, seq_len, 1)); // inputs
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_I32, topk, seq_len, 1, 1)); // ids
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, topk, seq_len, 1)); // topk weights
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden_dim, k_dim, num_experts, 1)); // expert up weights
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k_dim, hidden_dim, num_experts, 1)); // expert dowm weights
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden_dim, k_dim, num_experts, 1)); // expert gate weights
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_I32, seq_len, topk, 1, 1)); // row idx
    output_size = hidden_dim * seq_len * 1 * 1;
}
ggml_tensor* test_case_moe_fused::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_moe_fused(compute_ctx, src_tensors[0], src_tensors[1], src_tensors[2], src_tensors[3], 
        src_tensors[4], src_tensors[5], src_tensors[6], start_idx, end_idx);
}
ggml_tensor* test_case_moe_fused::build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    // id - start_idx
    std::vector<int32_t> ids(ggml_nelements(src_tensors[1]));
    memcpy(ids.data(), src_tensors[1]->data, ids.size() * sizeof(int32_t));
    for(auto &val : ids){
        val -= start_idx;
    }
    ggml_tensor* id = ggml_new_tensor(compute_ctx, GGML_TYPE_I32, 4, src_tensors[1]->ne);
    memcpy(id->data, ids.data(), ids.size() * sizeof(int32_t));
    // expand input to [hidden_dim, topk, seq_len, 1]
    ggml_tensor* repeated_input = ggml_repeat(compute_ctx, src_tensors[0], ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, hidden_dim, topk, seq_len, 1));
    // calculate gate output [k_dim, topk, seq_len, 1]
    ggml_tensor* gate_output = ggml_mul_mat_id(compute_ctx, src_tensors[5], repeated_input, id);
    // calculate silu gate output [k_dim, topk, seq_len, 1]
    ggml_tensor* gate_silu_output = ggml_silu(compute_ctx, gate_output);
    // calculate up output [k_dim, topk, seq_len, 1]
    ggml_tensor* up_output = ggml_mul_mat_id(compute_ctx, src_tensors[3], repeated_input, id);
    // up_ouput * gate_silu_ouput [k_dim, topk, seq_len, 1]
    ggml_tensor* gate_silu_mul_output = ggml_mul(compute_ctx, up_output, gate_silu_output);
    // get expert output [hidden_dim, topk, seq_len, 1]
    ggml_tensor* expert_output = ggml_mul_mat_id(compute_ctx, src_tensors[4], gate_silu_mul_output, id);
    // routing 
    ggml_tensor* expert_output_transpose = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, expert_output));
    ggml_tensor* topk_weight_transpose = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, src_tensors[2]));
    ggml_tensor* output = ggml_mul_mat(compute_ctx, expert_output_transpose, topk_weight_transpose);
    return ggml_cont(compute_ctx, output);
}
void test_case_moe_fused::init_tensors(){
    for(int i = 0; i < src.size(); ++i){
        auto &val = src[i];
        if(i == 1){ // ids
            std::random_device rd;
            std::default_random_engine rng(rd());
            std::vector<int32_t> random_experts(num_experts);
            std::vector<int32_t> data(ggml_nelements(val));
            for (int i = 0; i < num_experts; i++) {
                random_experts[i] = i + start_idx;
            }
            for(int i = 0; i < seq_len; ++i){
                std::shuffle(random_experts.begin(), random_experts.end(), rng);
                for(int j = 0; j < topk; ++j){
                    data[i * topk + j] = random_experts[j];
                }
            }
            memcpy(val->data, data.data(), data.size() * sizeof(int32_t));
            // print to check 
            std::vector<int32_t> data_copy(ggml_nelements(val));
            memcpy(data_copy.data(), val->data, data_copy.size()*sizeof(int32_t));
            std::cout << "Initializing tensor sample (first few values):" << std::endl;
            for (int i = 0; i < std::min(static_cast<size_t>(10), data_copy.size()); i++) {
                std::cout << static_cast<int32_t>(data_copy[i]) << " ";
            }
            std::cout << std::endl;
        }else if(i == 6){ // row ids
            std::vector<int32_t> data(ggml_nelements(val));
            for (int i = 0; i < data.size(); ++i) {
                data[i] = i;
            }
            memcpy(val->data, data.data(), data.size() * sizeof(int32_t));
            // print to check 
            std::vector<int32_t> data_copy(ggml_nelements(val));
            memcpy(data_copy.data(), val->data, data_copy.size()*sizeof(int32_t));
            std::cout << "Initializing tensor sample (first few values):" << std::endl;
            for (int i = 0; i < std::min(static_cast<size_t>(10), data_copy.size()); i++) {
                std::cout << static_cast<int32_t>(data_copy[i]) << " ";
            }
            std::cout << std::endl;
        }else{
            init_tensor(val);
        }
    }
}
void test_case_all_reduce_sum::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data()));
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_all_reduce_sum::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_all_reduce_sum(compute_ctx, src_tensors[0]);
}
ggml_tensor* test_case_all_reduce_sum::build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    ggml_tensor* output = ggml_dup(compute_ctx, src_tensors[0]);
    return output;
}
void test_case_get_slice::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_a.data()));
    auto ne = ne_a;
    ne[axis] = to - from;
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_get_slice::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    ggml_tensor* slice = ggml_get_slice(compute_ctx, src_tensors[0],from,to,axis);
    return ggml_cont(compute_ctx, slice);
}
ggml_tensor* test_case_get_slice::build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    auto ne = ne_a;
    std::vector<int64_t> nb(4);
    nb[0] = src_tensors[0]->nb[0];
    nb[1] = src_tensors[0]->nb[1];
    nb[2] = src_tensors[0]->nb[2];
    nb[3] = src_tensors[0]->nb[3];
    ne[axis] = to - from;
    int64_t offset = nb[axis] * from;
    ggml_tensor* view = ggml_view_4d(compute_ctx, src_tensors[0], ne[0], ne[1], ne[2], ne[3], nb[1], nb[2], nb[3], offset);
    ggml_tensor* output = ggml_cont(compute_ctx, view);
    return output;
}
// scatter_update: with a lot bugs
void test_case_scatter_update::init_src_size(){
    assert(src.empty());
    ne_a.push_back(1);
    ne_a.push_back(1);
    src.push_back(ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_a.data()));
    src.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_updates));
    src.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_a[0], n_updates));
    auto ne = ne_a;
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_scatter_update::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    ggml_tensor* output = ggml_scatter_update(compute_ctx, src_tensors[0], src_tensors[1], src_tensors[2]);
    return ggml_dup(compute_ctx, output);
}
ggml_tensor* test_case_scatter_update::build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_dup(compute_ctx, src_tensors[0]);
}
void test_case_scatter_update::init_tensors(){
    for(auto &val: src){
        if(val->type == GGML_TYPE_I32){
            // index
            std::vector<int32_t> data(ggml_nelements(val));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int32_t> dis(0, ne_a[1] - 1);
            for (int i = 0; i < data.size(); ++i) {
                data[i] = dis(gen);
            }
            memcpy(val->data, data.data(), data.size() * sizeof(int32_t));
            // print to check 
            std::vector<int32_t> data_copy(ggml_nelements(val));
            memcpy(data_copy.data(), val->data, data_copy.size()*sizeof(int32_t));
            std::cout << "Initializing tensor sample (first few values):" << std::endl;
            for (int i = 0; i < std::min(static_cast<size_t>(10), data.size()); i++) {
                std::cout << static_cast<int32_t>(data[i]) << " ";
            }
            std::cout << std::endl;
        }else{
            init_tensor(val);
        }
    }
}
void test_case_flash_attn_jittor::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dims_kq, sequence_lenth_q, num_heads, batch_size)); // q
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dims_kq, sequence_lenth_kv, key_num_heads, batch_size)); // k
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dims_v, sequence_lenth_kv, key_num_heads, batch_size)); // v
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_I8, sequence_lenth_kv, sequence_lenth_q, 1, batch_size));// mask
    std::vector<int64_t> ne = {head_dims_v, sequence_lenth_q, num_heads, batch_size};
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_flash_attn_jittor::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    ggml_tensor* q_tensor_f16 = ggml_cast(compute_ctx, src_tensors[0], GGML_TYPE_F16);
    ggml_tensor* k_tensor_f16 = ggml_cast(compute_ctx, src_tensors[1], GGML_TYPE_F16);
    ggml_tensor* v_tensor_f16 = ggml_cast(compute_ctx, src_tensors[2], GGML_TYPE_F16);
    ggml_tensor* mask_tensor = src_tensors[3];
    ggml_tensor* output = ggml_flash_attn_jittor_v1(compute_ctx, q_tensor_f16, k_tensor_f16, v_tensor_f16, mask_tensor, batch_size, num_heads, head_dims_kq, head_dims_v, key_num_heads, sequence_lenth_q, sequence_lenth_kv, NULL, NULL, scalevalue);
    return ggml_cast(compute_ctx, output, GGML_TYPE_F32);
}
ggml_tensor* test_case_flash_attn_jittor::build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    ggml_tensor* q = src_tensors[0];
    ggml_tensor* k = src_tensors[1];
    ggml_tensor* v = src_tensors[2];
    ggml_tensor* mask = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, sequence_lenth_kv, sequence_lenth_q, 1, batch_size);
    std::vector<int8_t> attn_mask(ggml_nelements(src_tensors[3]));
    memcpy(attn_mask.data(), src_tensors[3]->data, sizeof(int8_t) * ggml_nelements(src_tensors[3]));
    std::vector<float> mask_data(ggml_nelements(src_tensors[3]));
    for (size_t i = 0; i < attn_mask.size(); ++i) {
        mask_data[i] = attn_mask[i] ? -INFINITY : 0.0f;
    }
    memcpy(mask->data, mask_data.data(), attn_mask.size() * sizeof(float));
    ggml_tensor* kq = ggml_mul_mat(compute_ctx, k, q);
    ggml_tensor* kq_soft_max = ggml_soft_max_ext(compute_ctx, kq, mask, scalevalue, 0.0f);
    ggml_tensor* v_cur = ggml_cont(compute_ctx, ggml_permute(compute_ctx, v, 1, 0, 2, 3));
    ggml_tensor* kqv = ggml_mul_mat(compute_ctx, v_cur, kq_soft_max);
    ggml_tensor* output = ggml_cont(compute_ctx, kqv);
    return output;
}
void test_case_flash_attn_jittor::init_tensors(){
    for(auto &val: src){
        if(val->type == GGML_TYPE_I8){
            // mask
            std::vector<int32_t> data(ggml_nelements(val));
            std::random_device mask_rd;
            std::mt19937 mask_gen(mask_rd());
            std::bernoulli_distribution mask_dist(0.3);  // 30% 的概率设置为 1
            for (int i = 0; i < data.size(); ++i) {
                data[i] = mask_dist(mask_gen) ? 1 : 0;
            }
            memcpy(val->data, data.data(), data.size() * sizeof(int8_t));
            // print to check 
            std::vector<int8_t> data_copy(ggml_nelements(val));
            memcpy(data_copy.data(), val->data, data_copy.size()*sizeof(int8_t));
            std::cout << "Initializing tensor sample (first few values):" << std::endl;
            for (int i = 0; i < std::min(static_cast<size_t>(10), data_copy.size()); i++) {
                std::cout << int(data_copy[i]) << " ";
            }
            std::cout << std::endl;
        }else{
            init_tensor(val);
        }
    }
}
void test_case_flash_attn_prompt::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dims_kq, sequence_lenth_q, num_heads, batch_size)); // q
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dims_kq, sequence_lenth_kv, key_num_heads, batch_size)); // k
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dims_v, sequence_lenth_kv, key_num_heads, batch_size)); // v
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_I8, sequence_lenth_kv, sequence_lenth_q, 1, batch_size));// mask
    std::vector<int64_t> ne = {head_dims_v, sequence_lenth_q, num_heads, batch_size};
    output_size = ne[0] * ne[1] * ne[2] * ne[3];
}
ggml_tensor* test_case_flash_attn_prompt::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    ggml_tensor* q_tensor_f16 = ggml_permute(compute_ctx, ggml_cast(compute_ctx, src_tensors[0], GGML_TYPE_F16), 0, 2, 1, 3);
    ggml_tensor* k_tensor_f16 = ggml_permute(compute_ctx, ggml_cast(compute_ctx, src_tensors[1], GGML_TYPE_F16), 0, 2, 1, 3);
    ggml_tensor* v_tensor_f16 = ggml_permute(compute_ctx, ggml_cast(compute_ctx, src_tensors[2], GGML_TYPE_F16), 0, 2, 1, 3);
    ggml_tensor* mask_tensor = src_tensors[3];
    ggml_tensor* output = ggml_flash_attn_prompt(compute_ctx, q_tensor_f16, k_tensor_f16, v_tensor_f16, mask_tensor, batch_size, num_heads, head_dims_kq, head_dims_v, key_num_heads, sequence_lenth_q, sequence_lenth_kv, NULL, NULL, scalevalue);
    ggml_tensor* permuted_output = ggml_permute(compute_ctx, output, 0, 2, 1, 3);
    return ggml_cast(compute_ctx, permuted_output, GGML_TYPE_F32);
}
ggml_tensor* test_case_flash_attn_prompt::build_graph_cpu(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    ggml_tensor* q = src_tensors[0];
    ggml_tensor* k = src_tensors[1];
    ggml_tensor* v = src_tensors[2];
    ggml_tensor* mask = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, sequence_lenth_kv, sequence_lenth_q, 1, batch_size);
    std::vector<int8_t> attn_mask(ggml_nelements(src_tensors[3]));
    memcpy(attn_mask.data(), src_tensors[3]->data, sizeof(int8_t) * ggml_nelements(src_tensors[3]));
    std::vector<float> mask_data(ggml_nelements(src_tensors[3]));
    for (size_t i = 0; i < attn_mask.size(); ++i) {
        mask_data[i] = attn_mask[i] ? -INFINITY : 0.0f;
    }
    memcpy(mask->data, mask_data.data(), attn_mask.size() * sizeof(float));
    ggml_tensor* kq = ggml_mul_mat(compute_ctx, k, q);
    ggml_tensor* kq_soft_max = ggml_soft_max_ext(compute_ctx, kq, mask, scalevalue, 0.0f);
    ggml_tensor* v_cur = ggml_cont(compute_ctx, ggml_permute(compute_ctx, v, 1, 0, 2, 3));
    ggml_tensor* kqv = ggml_mul_mat(compute_ctx, v_cur, kq_soft_max);
    ggml_tensor* output = ggml_cont(compute_ctx, kqv);
    return output;
}
void test_case_flash_attn_prompt::init_tensors(){
    for(auto &val: src){
        if(val->type == GGML_TYPE_I8){
            // mask
            std::vector<int32_t> data(ggml_nelements(val));
            std::random_device mask_rd;
            std::mt19937 mask_gen(mask_rd());
            std::bernoulli_distribution mask_dist(0.3);  // 30% 的概率设置为 1
            for (int i = 0; i < data.size(); ++i) {
                data[i] = mask_dist(mask_gen) ? 1 : 0;
            }
            memcpy(val->data, data.data(), data.size() * sizeof(int8_t));
            // print to check 
            std::vector<int8_t> data_copy(ggml_nelements(val));
            std::cout<<ggml_nelements(val)<<std::endl;
            memcpy(data_copy.data(), val->data, data_copy.size()*sizeof(int8_t));
            std::cout << "Initializing tensor sample (first few values):" << std::endl;
            for (int i = 0; i < std::min(static_cast<size_t>(10), data_copy.size()); i++) {
                std::cout << int(data_copy[i]) << " ";
            }
            std::cout << std::endl;
        }else{
            init_tensor(val);
        }
    }
}