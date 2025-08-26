#include "rope_cache.h"

#include <float.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "all_ops.h"
#include "kernels/ascendc_kernels.h"
#include "op_proto.h"

static const int64_t MAX_KERNELS = 4096;

RopeCache::RopeCache(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];  // input
    ggml_tensor* src2 = dst->src[2];  // freq_factors
    ggml_tensor* src1 = dst->src[1];  // position

    // param
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    // const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims = ((int32_t*)dst->op_params)[1];
    const int mode = ((int32_t*)dst->op_params)[2];
    // const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t*)dst->op_params)[4];

    GGML_TENSOR_UNARY_OP_LOCALS
    memcpy(&freq_base, (int32_t*)dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t*)dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t*)dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t*)dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t*)dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t*)dst->op_params + 10, sizeof(float));

    // TODO: n_dims <= ne0
    GGML_ASSERT(n_dims == ne0);
    GGML_ASSERT(n_dims % 2 == 0);
    // TODO: ext_factor != 0
    // GGML_ASSERT(ext_factor == 0);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    float corr_dims[2];
    const int64_t pos_len = MAX_KERNELS;
    // const int64_t pos_len = src0->ne[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims);

    const float logf_1_freq_scale = logf(1.0f / freq_scale);

    ggml_cann_pool_alloc sin_final_allocator(
        ctx.pool(),
        src0->ne[0] / 2 * ctx.n_ctx * ggml_type_size(ggml_type::GGML_TYPE_F32));
    ggml_cann_pool_alloc cos_final_allocator(
        ctx.pool(),
        src0->ne[0] / 2 * ctx.n_ctx * ggml_type_size(ggml_type::GGML_TYPE_F32));
    void* device_sin_final_buffer = sin_final_allocator.get();
    void* device_cos_final_buffer = cos_final_allocator.get();

    ggml_cann_set_device(ctx.device);
    for (int64_t i = 0; i < ctx.n_ctx; i += MAX_KERNELS) {
        ACLRT_LAUNCH_KERNEL(ascendc_custom_rope_cache_ext)
        (std::min(MAX_KERNELS, ctx.n_ctx - i), ctx.stream(),
         (float*)device_cos_final_buffer + i * src0->ne[0] / 2,
         (float*)device_sin_final_buffer + i * src0->ne[0] / 2, ne0, ne1, s01,
         s02, n_dims, freq_scale, theta_scale, ext_factor, attn_factor,
         corr_dims[0], corr_dims[1], logf_1_freq_scale, i);
    }
    ACL_CHECK(aclrtSynchronizeStream(ctx.stream()));
    final_shape = {1, ctx.n_ctx, 1, src0->ne[0] / 2};
    final_size = std::accumulate(final_shape.begin(), final_shape.end(), 1,
                                 std::multiplies<int64_t>()) *
                 ggml_type_size(ggml_type::GGML_TYPE_F32);
    ACL_CHECK(aclrtMallocHost(&sin_final_buffer, final_size));
    ACL_CHECK(aclrtMallocHost(&cos_final_buffer, final_size));
    ACL_CHECK(aclrtMemcpy((void*)sin_final_buffer, final_size,
                          (void*)device_sin_final_buffer, final_size,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtMemcpy((void*)cos_final_buffer, final_size,
                          (void*)device_cos_final_buffer, final_size,
                          ACL_MEMCPY_DEVICE_TO_HOST));
}

ge::Operator RopeCache::GetCosOp(ge::Graph& graph,
                                 const std::string& name) const {
    std::vector<int64_t> shape(final_shape.begin(), final_shape.end());
    ge::TensorDesc tensor_desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor tensor_ge(tensor_desc, (uint8_t*)cos_final_buffer, final_size);
    ge::op::Const const_op(name.c_str());
    const_op.set_attr_value(tensor_ge);
    graph.AddOp(const_op);
    return const_op;
}

ge::Operator RopeCache::GetSinOp(ge::Graph& graph,
                                 const std::string& name) const {
    std::vector<int64_t> shape(final_shape.begin(), final_shape.end());
    ge::TensorDesc tensor_desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor tensor_ge(tensor_desc, (uint8_t*)sin_final_buffer, final_size);
    ge::op::Const const_op(name.c_str());
    const_op.set_attr_value(tensor_ge);
    graph.AddOp(const_op);
    return const_op;
}
