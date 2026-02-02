/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "aclnn_ops.h"

#include <aclnnop/aclnn_addcdiv.h>
#include <aclnnop/aclnn_avgpool2d.h>
#include <aclnnop/aclnn_batch_matmul.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_constant_pad_nd.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_fill_scalar.h>
#include <aclnnop/aclnn_fused_infer_attention_score.h>
#include <aclnnop/aclnn_gather_v2.h>
#include <aclnnop/aclnn_group_norm.h>
#include <aclnnop/aclnn_index_fill_tensor.h>
#include <aclnnop/aclnn_layer_norm.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_max_pool.h>
#include <aclnnop/aclnn_mm.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_pow_tensor_tensor.h>
#include <aclnnop/aclnn_prompt_flash_attention_v3.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_repeat.h>
#include <aclnnop/aclnn_repeat_interleave.h>
#include <aclnnop/aclnn_roll.h>
#include <aclnnop/aclnn_scatter_update.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_tril.h>
#include <aclnnop/aclnn_triu.h>
#include <aclnnop/aclnn_upsample_nearest_2d.h>
#include <aclnnop/aclnn_weight_quant_batch_matmul_v2.h>
#include <arm_neon.h>
#include <float.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <vector>

#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_grouped_matmul_v4.h"
#include "aclnnop/aclnn_index_copy.h"
#include "aclnnop/aclnn_index_select.h"
#include "aclnnop/aclnn_moe_compute_expert_tokens.h"
#include "aclnnop/aclnn_moe_finalize_routing_v2.h"
#include "aclnnop/aclnn_moe_init_routing.h"
#include "ggml-impl.h"
#include "kernels/ascendc_kernels.h"

#ifdef LLAMA_JITTOR_OPS_SUPPORT
#include "aclnn_jittor_infer_flash_attention_v4.h"
#endif

#define GGML_COMMON_DECL_C

#include "../ggml-common.h"

/**
 * @brief Repeats elements of a tensor along each dimension according to the
 * specified repeat array.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor to be repeated.
 * @param acl_dst The destination tensor after repeating.
 * @param repeat_array The array specifying the number of repetitions along each
 * dimension.
 */
static void aclnn_repeat(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                         aclTensor* acl_dst, int64_t* repeat_array) {
    // repeat tensor along each dim with repeat_array
    aclIntArray* repeats = aclCreateIntArray(repeat_array, GGML_MAX_DIMS);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnRepeatGetWorkspaceSize(acl_src, repeats, acl_dst,
                                          &workspaceSize, &executor));

    if (workspaceSize > 0) {
        // Memory from allocator will "free" immediately, and this memory
        // will be alloced to other pointers, but it won't access before
        // this async task end because all tasks in same stream will execute
        // in queue.
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(
        aclnnRepeat(workspaceAddr, workspaceSize, executor, ctx.stream()));
    ACL_CHECK(aclDestroyIntArray(repeats));
}

void ggml_cann_repeat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src, dst));

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    int64_t repeatsArray[] = {dst->ne[3] / src->ne[3], dst->ne[2] / src->ne[2],
                              dst->ne[1] / src->ne[1], dst->ne[0] / src->ne[0]};

    aclnn_repeat(ctx, acl_src, acl_dst, repeatsArray);
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

/**
 * @brief Adds two tensors element-wise and stores the result in a destination
 * tensor.
 *
 * This function performs the operation:
 * \f[
 *    dst = acl\_src0 + alpha \times acl\_src1
 * \f]
 * where alpha is a scalar value and defaults to 1.0f.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src0 The first source tensor.
 * @param acl_src1 The second source tensor.
 * @param acl_dst The destination tensor where the result will be stored.
 */
static void aclnn_add(ggml_backend_cann_context& ctx, aclTensor* acl_src0,
                      aclTensor* acl_src1, aclTensor* acl_dst) {
    aclScalar* alpha = nullptr;
    float alphaValue = 1.0f;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnAddGetWorkspaceSize(acl_src0, acl_src1, alpha, acl_dst,
                                       &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnAdd(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyScalar(alpha));
}
static void aclnn_adds_int(ggml_backend_cann_context& ctx, aclTensor* acl_src0,
                           int value, aclTensor* acl_dst) {
    aclScalar* alpha = nullptr;
    int alphaValue = 1.0f;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_INT32);
    aclScalar* value_scalar = nullptr;
    value_scalar = aclCreateScalar(&value, aclDataType::ACL_INT32);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    ACL_CHECK(aclnnAddsGetWorkspaceSize(acl_src0, value_scalar, alpha, acl_dst,
                                        &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(aclnnAdds(workspaceAddr, workspaceSize, executor, ctx.stream()));
    ACL_CHECK(aclDestroyScalar(alpha));
    ACL_CHECK(aclDestroyScalar(value_scalar));
}

//  static void aclnn_gather(ggml_backend_cann_context& ctx, aclTensor* acl_src,
//                           aclTensor* acl_index, aclTensor* acl_dst,int64_t
//                           dim) {
//      uint64_t workspaceSize = 0;
//      aclOpExecutor* executor;
//      void* workspaceAddr = nullptr;

//      ACL_CHECK(aclnnGatherV2GetWorkspaceSize(acl_src, dim,acl_index, acl_dst,
//                                            &workspaceSize, &executor));
//      if (workspaceSize > 0) {
//          ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
//          workspaceAddr = workspace_allocator.get();
//      }

//      ACL_CHECK(aclnnGatherV2(workspaceAddr, workspaceSize, executor,
//      ctx.stream()));
//  }
void ggml_cann_add(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    aclTensor* acl_src0;
    aclTensor* acl_src1;
    aclTensor* acl_dst;

    // Need bcast
    if (!ggml_are_same_shape(src0, src1) && ggml_cann_need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        acl_src0 = ggml_cann_create_tensor(src0, BCAST_PARAM(src0));
        acl_src1 = ggml_cann_create_tensor(src1, BCAST_PARAM(src1));
        acl_dst = ggml_cann_create_tensor(dst, BCAST_PARAM(src0));
    } else {
        acl_src0 = ggml_cann_create_tensor(src0);
        acl_src1 = ggml_cann_create_tensor(src1);
        acl_dst = ggml_cann_create_tensor(dst);
    }

    aclnn_add(ctx, acl_src0, acl_src1, acl_dst);

    ACL_CHECK(aclDestroyTensor(acl_src0));
    ACL_CHECK(aclDestroyTensor(acl_src1));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_leaky_relu(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));
    aclScalar* acl_negative_slope =
        aclCreateScalar(&negative_slope, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnLeakyReluGetWorkspaceSize(
        acl_src, acl_negative_slope, acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnLeakyRelu(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyScalar(acl_negative_slope));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

/**
 * @brief Concatenates a list of tensors along a specified dimension and stores
 * the result in a destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param tensorList The list of tensors to be concatenated.
 * @param acl_dst The destination tensor where the concatenated result will be
 * stored.
 * @param concat_dim The dimension along which the tensors will be concatenated.
 */
static void aclnn_concat(ggml_backend_cann_context& ctx,
                         aclTensorList* tensorList, aclTensor* acl_dst,
                         int64_t concat_dim) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnCatGetWorkspaceSize(tensorList, concat_dim, acl_dst,
                                       &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnCat(workspaceAddr, workspaceSize, executor, ctx.stream()));
}

void ggml_cann_concat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclTensor* acl_src0 = ggml_cann_create_tensor(src0);
    aclTensor* acl_src1 = ggml_cann_create_tensor(src1);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    const int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    int32_t acl_dim = 3 - dim;

    aclTensor* tensors[] = {acl_src0, acl_src1};
    aclTensorList* tensorList = aclCreateTensorList(tensors, 2);
    aclnn_concat(ctx, tensorList, acl_dst, acl_dim);

    ACL_CHECK(aclDestroyTensorList(tensorList));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

/**
 * @brief Creates a tensor with values starting from `start`, incremented by
 * `step`, and ending before `stop`.
 *
 * This function performs the operation:
 * \f[
 *    \text {out }_{i+1}=\text {out }_i+\text {step}
 * \f]
 * the range is [start, stop).
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_dst The destination tensor where the values will be stored.
 * @param start The starting value of the range.
 * @param stop The ending value of the range (exclusive).
 * @param step The step size between consecutive values.
 * @param n_elements The number of elements in the destination tensor.
 */
static void aclnn_arange(ggml_backend_cann_context& ctx, aclTensor* acl_dst,
                         float start, float stop, float step,
                         int64_t n_elements) {
    int64_t steps = (int64_t)std::ceil((stop - start) / step);
    GGML_ASSERT(n_elements == steps);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclScalar* acl_start = aclCreateScalar(&start, aclDataType::ACL_FLOAT);
    aclScalar* acl_end = aclCreateScalar(&stop, aclDataType::ACL_FLOAT);
    aclScalar* acl_step = aclCreateScalar(&step, aclDataType::ACL_FLOAT);

    ACL_CHECK(aclnnArangeGetWorkspaceSize(acl_start, acl_end, acl_step, acl_dst,
                                          &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnArange(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyScalar(acl_start));
    ACL_CHECK(aclDestroyScalar(acl_end));
    ACL_CHECK(aclDestroyScalar(acl_step));
}

static void aclnn_arange_int(ggml_backend_cann_context& ctx, aclTensor* acl_dst,
                             int start, int stop, int steps) {
    // int64_t steps = (int64_t)std::ceil((stop - start) / step);
    // GGML_ASSERT(n_elements == steps);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclScalar* acl_start = aclCreateScalar(&start, aclDataType::ACL_INT32);
    aclScalar* acl_end = aclCreateScalar(&stop, aclDataType::ACL_INT32);
    aclScalar* acl_step = aclCreateScalar(&steps, aclDataType::ACL_INT32);

    ACL_CHECK(aclnnArangeGetWorkspaceSize(acl_start, acl_end, acl_step, acl_dst,
                                          &workspaceSize, &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnArange(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyScalar(acl_start));
    ACL_CHECK(aclDestroyScalar(acl_end));
    ACL_CHECK(aclDestroyScalar(acl_step));
}

void ggml_cann_arange(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    int64_t n_elements = ggml_nelements(dst);
    float start;
    float stop;
    float step;
    memcpy(&start, (float*)dst->op_params + 0, sizeof(float));
    memcpy(&stop, (float*)dst->op_params + 1, sizeof(float));
    memcpy(&step, (float*)dst->op_params + 2, sizeof(float));

    aclnn_arange(ctx, acl_dst, start, stop, step, n_elements);
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_sqr(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    dst->src[1] = dst->src[0];
    ggml_cann_mul_div<aclnnMulGetWorkspaceSize, aclnnMul>(ctx, dst);
}

void ggml_cann_clamp(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float*)dst->op_params + 1, sizeof(float));

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    aclScalar* acl_min = aclCreateScalar(&min, aclDataType::ACL_FLOAT);
    aclScalar* acl_max = aclCreateScalar(&max, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnClampGetWorkspaceSize(acl_src, acl_min, acl_max, acl_dst,
                                         &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnClamp(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyScalar(acl_min));
    ACL_CHECK(aclDestroyScalar(acl_max));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_scale(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    // scale factor
    float v;
    memcpy(&v, dst->op_params, sizeof(float));

    aclScalar* scale = aclCreateScalar(&v, aclDataType::ACL_FLOAT);
    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnMulsGetWorkspaceSize(acl_src, scale, acl_dst, &workspaceSize,
                                        &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnMuls(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyScalar(scale));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_argsort(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    enum ggml_sort_order order = (enum ggml_sort_order)dst->op_params[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);
    ggml_cann_pool_alloc temp_buffer_allocator(
        ctx.pool(), ggml_nelements(dst) * sizeof(int64_t));
    void* buffer = temp_buffer_allocator.get();
    aclTensor* tmp_tensor =
        ggml_cann_create_tensor(buffer, ACL_INT64, ggml_type_size(dst->type),
                                dst->ne, dst->nb, GGML_MAX_DIMS);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnArgsortGetWorkspaceSize(
        acl_src, -1, (order == GGML_SORT_ORDER_DESC ? true : false), tmp_tensor,
        &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnArgsort(workspaceAddr, workspaceSize, executor, ctx.stream()));

    workspaceSize = 0;
    ACL_CHECK(aclnnCastGetWorkspaceSize(tmp_tensor,
                                        ggml_cann_type_mapping(dst->type),
                                        acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnCast(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(tmp_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_allreduce_sum(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ctx.initialized);
    void* src_data = dst->src[0]->data;
    void* dst_data = dst->data;
    ggml_tensor* src = dst->src[0];
    int src_size = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
    int dst_size = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    GGML_ASSERT(src_size == dst_size);
    HCCL_CHECK(HcclAllReduce(src_data, dst_data, dst_size, HCCL_DATA_TYPE_FP32,
                             HCCL_REDUCE_SUM, ctx.hccl_comm, ctx.stream()));
    ACL_CHECK(aclrtSynchronizeStream(ctx.stream()));
}

void ggml_cann_set_tensor_to_zero(ggml_backend_cann_context& ctx,
                                  ggml_tensor* dst) {
    size_t nbytes = ggml_nbytes(dst);
    ACL_CHECK(aclrtMemsetAsync(dst->data, nbytes, 0, nbytes, ctx.stream()));
}

void ggml_cann_norm(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    std::vector<int64_t> normData = {dst->ne[0]};
    aclIntArray* norm = aclCreateIntArray(normData.data(), normData.size());
    ACL_CHECK(aclnnLayerNormGetWorkspaceSize(acl_src, norm, nullptr, nullptr,
                                             eps, acl_dst, nullptr, nullptr,
                                             &workspaceSize, &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnLayerNorm(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyIntArray(norm));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_group_norm(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    int n_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    int64_t N = src->ne[3];
    int64_t C = src->ne[2];
    int64_t HxW = src->ne[1] * src->ne[0];

    size_t type_size = ggml_type_size(src->type);
    int64_t ne[] = {n_groups, N};
    size_t nb[] = {type_size, type_size * n_groups};
    size_t n_bytes = N * n_groups;

    ggml_cann_pool_alloc temp_buffer_allocator(ctx.pool(), n_bytes * 2);
    void* buffer = temp_buffer_allocator.get();
    aclTensor* acl_mean_out = ggml_cann_create_tensor(
        buffer, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);
    aclTensor* acl_rstd_out = ggml_cann_create_tensor(
        (char*)buffer + n_bytes, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);

    ACL_CHECK(aclnnGroupNormGetWorkspaceSize(
        acl_src, nullptr, nullptr, N, C, HxW, n_groups, eps, acl_dst,
        acl_mean_out, acl_rstd_out, &workspaceSize, &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnGroupNorm(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyTensor(acl_mean_out));
    ACL_CHECK(aclDestroyTensor(acl_rstd_out));
}

void ggml_cann_acc(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];

    size_t nb1 = ((int32_t*)dst->op_params)[0];
    size_t nb2 = ((int32_t*)dst->op_params)[1];
    size_t nb3 = ((int32_t*)dst->op_params)[2];
    size_t offset = ((int32_t*)dst->op_params)[3];
    bool inplace = (bool)((int32_t*)dst->op_params)[4];

    size_t param_nb[] = {ggml_element_size(src0), nb1, nb2, nb3};

    aclTensor* acl_dst = ggml_cann_create_tensor(
        dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);
    aclTensor* acl_src1 = ggml_cann_create_tensor(src1);

    aclScalar* alpha = nullptr;
    float alphaValue = 1.0f;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    if (!inplace) {
        size_t cpy_size = ggml_nbytes(dst);
        ACL_CHECK(aclrtMemcpyAsync(dst->data, cpy_size, src0->data, cpy_size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
        aclTensor* acl_src0 = ggml_cann_create_tensor(
            src0, src1->ne, src0->nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);
        ACL_CHECK(aclnnAddGetWorkspaceSize(acl_src0, acl_src1, alpha, acl_dst,
                                           &workspaceSize, &executor));
        if (workspaceSize > 0) {
            ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
            workspaceAddr = workspace_allocator.get();
        }
        ACL_CHECK(
            aclnnAdd(workspaceAddr, workspaceSize, executor, ctx.stream()));
        ACL_CHECK(aclDestroyTensor(acl_src0));
    } else {
        ACL_CHECK(aclnnInplaceAddGetWorkspaceSize(acl_dst, acl_src1, alpha,
                                                  &workspaceSize, &executor));
        if (workspaceSize > 0) {
            ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
            workspaceAddr = workspace_allocator.get();
        }
        ACL_CHECK(aclnnInplaceAdd(workspaceAddr, workspaceSize, executor,
                                  ctx.stream()));
    }

    ACL_CHECK(aclDestroyTensor(acl_src1));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void print_device_tensor_elements(
    ggml_backend_cann_context& ctx, void* tensor_data, const char* tensor_name,
    const int64_t* ne, ggml_type type, int64_t elem_count = 0,
    int max_print = 10, aclrtStream stream = nullptr, bool print_all = false,
    bool print_shape = true, bool print_stats = false) {
    if (tensor_data == nullptr) {
        printf("%s: 张量数据为空\n", tensor_name);
        return;
    }

    // 如果未提供元素总数，从ne计算
    if (elem_count <= 0 && ne != nullptr) {
        elem_count = 1;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (ne[i] > 0)
                elem_count *= ne[i];
            else
                break;
        }
    }

    // 打印张量形状
    if (print_shape && ne != nullptr) {
        printf("%s shape: [", tensor_name);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (ne[i] > 0)
                printf("%d%s", (int)ne[i],
                       (i < GGML_MAX_DIMS - 1 && ne[i + 1] > 0) ? ", " : "");
            else
                break;
        }
        printf("]\n");
    }

    // 确定要打印的元素数量（不超过张量大小和max_print）
    int elem_to_print = print_all
                            ? elem_count
                            : (elem_count < max_print ? elem_count : max_print);

    // 如果要打印很多元素，先确认
    if (print_all && elem_count > 1000) {
        printf("警告: 即将打印 %d 个元素，输出可能很长。\n", (int)elem_count);
    }

    int8_t* host_buffer_i8 = nullptr;
    float* host_buffer_f32 = nullptr;
    int32_t* host_buffer_i32 = nullptr;

    // 根据类型分配正确的缓冲区
    if (type == GGML_TYPE_Q8_0) {
        host_buffer_i8 = new int8_t[elem_to_print];
    } else if (type == GGML_TYPE_I32) {
        host_buffer_i32 = new int32_t[elem_to_print];
    } else {
        host_buffer_f32 = new float[elem_to_print];
    }

    // 使用ctx的stream如果未提供stream
    if (stream == nullptr) {
        stream = ctx.stream();
    }

    // 从设备内存复制到主机内存
    if (type == GGML_TYPE_Q8_0) {
        ACL_CHECK(aclrtMemcpyAsync(
            host_buffer_i8, elem_to_print * sizeof(int8_t), tensor_data,
            elem_to_print * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST, stream));
    } else if (type == GGML_TYPE_I32) {
        ACL_CHECK(aclrtMemcpyAsync(host_buffer_i32,
                                   elem_to_print * sizeof(int32_t), tensor_data,
                                   elem_to_print * sizeof(int32_t),
                                   ACL_MEMCPY_DEVICE_TO_HOST, stream));
    } else {
        ACL_CHECK(aclrtMemcpyAsync(
            host_buffer_f32, elem_to_print * sizeof(float), tensor_data,
            elem_to_print * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST, stream));
    }
    // 确保复制完成
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // 打印数据
    if (ne && ne[0] > 0 && ne[1] > 0) {
        // 多维张量，按形状格式化打印
        printf("%s %s:\n", tensor_name,
               print_all ? "all elements" : "first elements");

        // 确定每个维度要打印多少元素
        int dim0_print =
            print_all ? ne[0] : (ne[0] < max_print ? ne[0] : max_print);
        int dim1_print = print_all ? ne[1]
                                   : (elem_to_print / dim0_print > ne[1]
                                          ? ne[1]
                                          : (elem_to_print / dim0_print));

        for (int i1 = 0; i1 < dim1_print; i1++) {
            printf("  [:, %d,:]:\t", i1);
            for (int i0 = 0; i0 < dim0_print; i0++) {
                int idx = i1 * ne[0] + i0;
                if (idx < elem_to_print) {
                    if (type == GGML_TYPE_Q8_0) {
                        printf("%d ", host_buffer_i8[idx]);
                    } else if (type == GGML_TYPE_I32) {
                        printf("%d ", host_buffer_i32[idx]);
                    } else {
                        printf("%.4f ", host_buffer_f32[idx]);
                    }
                }
            }

            if (ne[0] > dim0_print) {
                printf("... (%d more elements)", (int)ne[0] - dim0_print);
            }
            printf("\n");
        }

        if (ne[1] > dim1_print) {
            printf("  ... (%d more rows)\n", (int)ne[1] - dim1_print);
        }
    } else {
        // 一维向量或标量
        printf("%s %s: ", tensor_name,
               print_all ? "all elements" : "first elements");

        // 打印方式处理：全部打印时每行最多打印20个元素
        int elements_per_line = 20;
        for (int i = 0; i < elem_to_print; i++) {
            if (type == GGML_TYPE_Q8_0) {
                printf("%d ", host_buffer_i8[i]);
            } else if (type == GGML_TYPE_I32) {
                printf("%d ", host_buffer_i32[i]);
            } else {
                printf("%.4f ", host_buffer_f32[i]);
            }
            // 全部打印时，每行最多显示指定元素数，提高可读性
            if (print_all && (i + 1) % elements_per_line == 0 &&
                i < elem_to_print - 1) {
                printf("\n");
            }
        }

        // 显示省略信息
        if (!print_all && elem_count > max_print) {
            printf("... (显示前%d个，共%d个元素)", max_print, (int)elem_count);
        }
        printf("\n");
    }

    //  // 计算并打印统计信息
    //  if (print_stats && elem_to_print > 0) {
    //      float min_val = host_buffer[0];
    //      float max_val = host_buffer[0];
    //      float sum = host_buffer[0];
    //      float sum_abs = fabs(host_buffer[0]);
    //      float sum_sq = host_buffer[0] * host_buffer[0];

    //      for (int i = 1; i < elem_to_print; i++) {
    //          if (host_buffer[i] < min_val) min_val = host_buffer[i];
    //          if (host_buffer[i] > max_val) max_val = host_buffer[i];
    //          sum += host_buffer[i];
    //          sum_abs += fabs(host_buffer[i]);
    //          sum_sq += host_buffer[i] * host_buffer[i];
    //      }

    //      float avg = sum / elem_to_print;
    //      float avg_abs = sum_abs / elem_to_print;
    //      float std_dev = sqrtf((sum_sq / elem_to_print) - (avg * avg));
    //      float l2_norm = sqrtf(sum_sq);

    //      printf("%s 统计: 最小值=%.4f, 最大值=%.4f, 平均值=%.4f\n"
    //             "       平均绝对值=%.4f, 标准差=%.4f, L2范数=%.4f\n",
    //             tensor_name, min_val, max_val, avg, avg_abs, std_dev,
    //             l2_norm);
    //  }

    // 释放临时缓冲区
    delete[] host_buffer_i8;
    delete[] host_buffer_f32;
    delete[] host_buffer_i32;
}

void print_acltensor(void* outDeviceAddr, const int64_t* outShape,
                     aclDataType type, char* name, int dims = GGML_MAX_DIMS,
                     const size_t* out_nb = nullptr,
                     ggml_backend_cann_context* ctx = nullptr) {
    return;
    // printf("------------------------\n");
    printf("output %s\n", name);
    // return;
    printf("output shape: ");
    int64_t size = 1;
    int64_t max_print = 100 * 6 * 2048;
    // int64_t max_print = 100;
    for (int i = 0; i < dims; i++) {
        printf("%ld ", outShape[i]);
        size *= outShape[i];
    }
    if (out_nb != nullptr) {
        auto offset = dims - 1;
        if (dims == GGML_MAX_DIMS) {
            offset = dims - 1;
        }

        size = out_nb[offset];
        printf("output nb: ");
        for (int i = 0; i < dims; i++) {
            printf("%ld ", out_nb[i]);
        }
        printf("\n");
    }
    printf("\n");

    printf("size: %ld", size);
    printf("\n");
    // aclrtSynchronizeStream(ctx.stream());
    std::vector<ggml_fp16_t> resultData_fp16(size, 0);
    std::vector<float> resultData_fp(size, 0);
    std::vector<int8_t> resultData_int8(size, 0);
    std::vector<int32_t> resultData_int32(size, 0);
    if (ctx != nullptr) {
        aclrtSynchronizeStream(ctx->stream());
    }
    if (type == ACL_FLOAT) {
        aclrtMemcpy(resultData_fp.data(),
                    resultData_fp.size() * sizeof(resultData_fp[0]),
                    outDeviceAddr, size * sizeof(resultData_fp[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        if (ctx != nullptr) {
            aclrtSynchronizeStream(ctx->stream());
        }
        if (out_nb != nullptr) {
            int print_count = 0;
            for (int64_t i = 0; i < outShape[1]; i++) {
                for (int64_t j = 0; j < outShape[0]; j++) {
                    auto offset = (j * out_nb[0] + i * out_nb[1]) /
                                  sizeof(resultData_fp[0]);
                    if (print_count < max_print) {
                        printf("%f ", resultData_fp[offset]);
                        print_count++;
                    }
                }
            }
        } else {
            for (int64_t i = 0; i < std::min(size, (int64_t)max_print); i++) {
                printf("%f ", resultData_fp[i]);
            }
        }
    } else if (type == ACL_FLOAT16) {
        aclrtMemcpy(resultData_fp16.data(),
                    resultData_fp16.size() * sizeof(resultData_fp16[0]),
                    outDeviceAddr, size * sizeof(resultData_fp16[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        if (ctx != nullptr) {
            aclrtSynchronizeStream(ctx->stream());
        }
        float* data_fp32 = (float*)malloc(size * sizeof(float));

        ggml_fp16_to_fp32_row(resultData_fp16.data(), data_fp32, size);
        if (out_nb != nullptr) {
            int print_count = 0;
            for (int64_t i = 0; i < outShape[1]; i++) {
                for (int64_t j = 0; j < outShape[0]; j++) {
                    auto offset = (j * out_nb[0] + i * out_nb[1]) /
                                  sizeof(resultData_fp16[0]);
                    if (print_count < max_print) {
                        printf("%f ", data_fp32[offset]);
                        print_count++;
                    }
                }
            }
        } else {
            for (int64_t i = 0; i < std::min(size, (int64_t)max_print); i++) {
                printf("%f ", data_fp32[i]);
            }
        }

    } else if (type == ACL_INT8 || type == ACL_INT4) {
        aclrtMemcpy(resultData_int8.data(),
                    resultData_int8.size() * sizeof(resultData_int8[0]),
                    outDeviceAddr, size * sizeof(resultData_int8[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            printf("%d ", resultData_int8[i]);
        }
    } else if (type == ACL_INT32) {
        aclrtMemcpy(resultData_int32.data(),
                    resultData_int32.size() * sizeof(resultData_int32[0]),
                    outDeviceAddr, size * sizeof(resultData_int32[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        if (ctx != nullptr) {
            aclrtSynchronizeStream(ctx->stream());
        }
        if (out_nb != nullptr) {
            int print_count = 0;
            for (int64_t i = 0; i < outShape[1]; i++) {
                for (int64_t j = 0; j < outShape[0]; j++) {
                    auto offset = (j * out_nb[0] + i * out_nb[1]) /
                                  sizeof(resultData_int32[0]);
                    if (print_count < max_print) {
                        printf(" %d %d ", offset, resultData_int32[offset]);
                        print_count++;
                    }
                }
            }
        } else {
            for (int64_t i = 0; i < std::min(size, (int64_t)max_print); i++) {
                printf("%d ", resultData_int32[i]);
            }
        }

    } else {
        printf("not support type\n");
    }
    printf("======================================\n\n");
}

/**
 * @brief Casts the data type of a source tensor to a destination tensor.
 *
 * This function casts the data type of the source tensor `acl_src` to the
 * specified data type `cast_data_type` and stores the result in the destination
 * tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose data type will be casted.
 * @param acl_dst The destination tensor where the casted result will be stored.
 * @param cast_data_type The target data type to which the source tensor will be
 * casted.
 */
static void aclnn_cast(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                       aclTensor* acl_dst, aclDataType cast_data_type) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnCastGetWorkspaceSize(acl_src, cast_data_type, acl_dst,
                                        &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnCast(workspaceAddr, workspaceSize, executor, ctx.stream()));
}

static void cal_mul_mat_quant(ggml_backend_cann_context& ctx, ggml_tensor* src0,
                              ggml_tensor* src1, ggml_tensor* dst,
                              const enum ggml_type type, char* scale_offset) {
    // ggml_tensor* src0 = dst->src[0];  // weight
    // ggml_tensor* src1 = dst->src[1];  // input

    // The shape of the weight is NCHW.
    // Matrix multiplication uses HW dims.
    // HC is regarded as batch.
    // weight need transpose.

    //  print_device_tensor_elements(ctx, src0->data, "src0", src0->ne,
    //  src0->type,ggml_nelements(src0), 10, ctx.stream(), false, true, true);
    //  print_device_tensor_elements(ctx, src1->data, "src1",
    //  src1->ne,src1->type, ggml_nelements(src1), 10, ctx.stream(), false,
    //  true, true);
    float weight_elem_size;
    if (type == GGML_TYPE_Q4_0) {
        weight_elem_size = float(sizeof(uint8_t)) / 2;
    } else if (type == GGML_TYPE_Q8_0) {
        weight_elem_size = float(sizeof(uint8_t));
    } else {
        GGML_ABORT("Only support Q4_0 and Q8_0 MUL_MAT");
    }
    float weight_nb[] = {src0->ne[0] * weight_elem_size, weight_elem_size};
    size_t weight_stride = src0->ne[1] * src0->ne[0] * weight_elem_size;
    //  size_t weight_size = weight_stride * src0->ne[2] * src0->ne[3];

    // scale stored at the end of weight. Also need transpose.
    size_t scale_elem_size = sizeof(uint16_t);
    size_t scale_nb[] = {src0->ne[0] / QK8_0 * scale_elem_size,
                         scale_elem_size};
    size_t scale_stride = src0->ne[1] * src0->ne[0] / QK8_0 * scale_elem_size;
    //  size_t scale_size = scale_stride * src0->ne[2] * src0->ne[3];
    //  ggml_cann_pool_alloc scale_alloctor(ctx.pool());
    //  void* scale_buffer = scale_alloctor.alloc(scale_size);
    //  float weight_all_nb[] = {src0_all->ne[0] * weight_elem_size,
    //  weight_elem_size}; size_t weight_all_stride = src0_all->ne[1] *
    //  src0_all->ne[0] * weight_elem_size; size_t weight_all_size =
    //  weight_all_stride * src0_all->ne[2] * src0_all->ne[3]; char*
    //  scale_offset = (char*)src0_all->data + weight_all_size;

    // input
    size_t input_elem_size = sizeof(uint16_t);
    int64_t input_ne[] = {src1->ne[0], src1->ne[1]};
    size_t input_nb[] = {input_elem_size, input_ne[0] * input_elem_size};
    size_t input_stride = input_ne[0] * input_ne[1] * input_elem_size;
    ggml_cann_pool_alloc input_alloctor(ctx.pool());
    void* input_buffer = src1->data;

    // case in
    if (src1->type != GGML_TYPE_F16) {
        aclTensor* acl_src1_tensor = ggml_cann_create_tensor(src1);
        //  print_acltensor(src1->data,src1->ne,ggml_cann_type_mapping(src1->type),
        //  "src1_tensor");
        input_buffer =
            input_alloctor.alloc(ggml_nelements(src1) * input_elem_size);

        int64_t* input_cast_ne = src1->ne;
        size_t input_cast_nb[GGML_MAX_DIMS];
        input_cast_nb[0] = sizeof(uint16_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            input_cast_nb[i] = input_cast_nb[i - 1] * input_cast_ne[i - 1];
        }

        aclTensor* acl_input_tensor = ggml_cann_create_tensor(
            input_buffer, ACL_FLOAT16, input_elem_size, input_cast_ne,
            input_cast_nb, GGML_MAX_DIMS);
        // print_acltensor(input_buffer,input_cast_ne,ACL_FLOAT16,
        // "input_tensor");

        aclnn_cast(ctx, acl_src1_tensor, acl_input_tensor, ACL_FLOAT16);

        // print_acltensor( input_buffer,input_cast_ne,ACL_FLOAT16,
        // "input_tensor");
        ACL_CHECK(aclDestroyTensor(acl_input_tensor));
        ACL_CHECK(aclDestroyTensor(acl_src1_tensor));
    }

    // output
    size_t output_elem_size = sizeof(uint16_t);
    size_t output_nb[] = {output_elem_size, dst->ne[0] * output_elem_size};
    ggml_cann_pool_alloc output_allocator(ctx.pool());
    void* output_buffer =
        output_allocator.alloc(ggml_nelements(dst) * output_elem_size);

    //  void* output_buffer = dst->data;
    size_t output_stride = dst->ne[0] * dst->ne[1] * output_elem_size;

    // aclnn
    int64_t max_elem_size = 65535;
    int64_t split_size = (src0->ne[1] / max_elem_size) + 1;
    ggml_cann_pool_alloc workspace_allocator(ctx.pool());
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    for (int64_t n1 = 0; n1 < src1->ne[3]; n1++) {
        for (int64_t c1 = 0; c1 < src1->ne[2]; c1++) {
            int64_t n0 = n1 / (src1->ne[3] / src0->ne[3]);
            int64_t c0 = c1 / (src1->ne[2] / src0->ne[2]);

            int64_t batch1 = (n1 * src1->ne[2]) + c1;
            int64_t batch0 = (n0 * src0->ne[2]) + c0;

            aclTensor* acl_input_tensor = ggml_cann_create_tensor(
                (char*)input_buffer + batch1 * input_stride, ACL_FLOAT16,
                input_elem_size, input_ne, input_nb, 2);

            // print_acltensor((char*)input_buffer + batch1 *
            // input_stride,input_ne,ACL_FLOAT16, "input_tensor",2);

            // first split
            int64_t weight_ne_offset = 0;
            int64_t weight_ne[2] = {
                max_elem_size > src0->ne[1] ? src0->ne[1] : max_elem_size,
                src0->ne[0]};
            int64_t scale_ne_offset = 0;
            int64_t scale_ne[2] = {weight_ne[0], weight_ne[1] / QK8_0};
            int64_t output_ne_offset = 0;
            int64_t output_ne[2] = {weight_ne[0], dst->ne[1]};

            aclTensor* acl_weight_tensor = ggml_cann_create_tensor(
                (char*)src0->data + batch0 * weight_stride,
                ggml_cann_type_mapping(type), weight_elem_size, weight_ne,
                weight_nb, 2, ACL_FORMAT_ND, weight_ne_offset);

            // print_acltensor((char*)src0->data + batch0 *
            // weight_stride,weight_ne,ggml_cann_type_mapping(type),
            // "weight_tensor",2);

            aclTensor* acl_scale_tensor = ggml_cann_create_tensor(
                scale_offset + batch0 * scale_stride, ACL_FLOAT16,
                scale_elem_size, scale_ne, scale_nb, 2, ACL_FORMAT_ND,
                scale_ne_offset);

            aclTensor* acl_output_tensor = ggml_cann_create_tensor(
                (char*)output_buffer + batch1 * output_stride, ACL_FLOAT16,
                output_elem_size, output_ne, output_nb, 2, ACL_FORMAT_ND,
                output_ne_offset);

            ACL_CHECK(aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
                acl_input_tensor, acl_weight_tensor, acl_scale_tensor, nullptr,
                nullptr, nullptr, nullptr, QK8_0, acl_output_tensor,
                &workspaceSize, &executor));
            if (workspaceAddr == nullptr) {
                workspaceAddr = workspace_allocator.alloc(workspaceSize);
            }
            ACL_CHECK(aclnnWeightQuantBatchMatmulV2(
                workspaceAddr, workspaceSize, executor, ctx.stream()));

            // print_acltensor(scale_offset + batch0 *
            // scale_stride,scale_ne,ACL_FLOAT16, "scale_tensor",2);
            // print_acltensor((char*)output_buffer + batch1 *
            // output_stride,output_ne,ACL_FLOAT16, "output_tensor",2);
            ACL_CHECK(aclDestroyTensor(acl_weight_tensor));
            ACL_CHECK(aclDestroyTensor(acl_scale_tensor));
            ACL_CHECK(aclDestroyTensor(acl_output_tensor));

            //  printf("split_size:%d\n",split_size);
            // other splits
            for (int64_t split = 1; split < split_size; split++) {
                weight_ne_offset +=
                    weight_elem_size * weight_ne[0] * weight_ne[1];
                weight_ne[0] = max_elem_size * (split + 1) > src0->ne[1]
                                   ? src0->ne[1] - (max_elem_size * split)
                                   : max_elem_size;
                scale_ne_offset += scale_elem_size * scale_ne[0] * scale_ne[1];
                scale_ne[0] = weight_ne[0];
                output_ne_offset +=
                    output_elem_size * output_ne[0] * output_ne[1];
                output_ne[0] = weight_ne[0];

                acl_weight_tensor = ggml_cann_create_tensor(
                    (char*)src0->data + batch0 * weight_stride,
                    ggml_cann_type_mapping(type), weight_elem_size, weight_ne,
                    weight_nb, 2, ACL_FORMAT_ND, weight_ne_offset);
                acl_scale_tensor = ggml_cann_create_tensor(
                    scale_offset + batch0 * scale_stride, ACL_FLOAT16,
                    scale_elem_size, scale_ne, scale_nb, 2, ACL_FORMAT_ND,
                    scale_ne_offset);
                acl_output_tensor = ggml_cann_create_tensor(
                    (char*)output_buffer + batch1 * output_stride, ACL_FLOAT16,
                    output_elem_size, output_ne, output_nb, 2, ACL_FORMAT_ND,
                    output_ne_offset);

                ACL_CHECK(aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
                    acl_input_tensor, acl_weight_tensor, acl_scale_tensor,
                    nullptr, nullptr, nullptr, nullptr, QK8_0,
                    acl_output_tensor, &workspaceSize, &executor));
                ACL_CHECK(aclnnWeightQuantBatchMatmulV2(
                    workspaceAddr, workspaceSize, executor, ctx.stream()));

                ACL_CHECK(aclDestroyTensor(acl_weight_tensor));
                ACL_CHECK(aclDestroyTensor(acl_scale_tensor));
                ACL_CHECK(aclDestroyTensor(acl_output_tensor));
            }

            ACL_CHECK(aclDestroyTensor(acl_input_tensor));
        }
    }

    // cast out
    if (dst->type != GGML_TYPE_F16) {
        int64_t* output_cast_ne = dst->ne;
        size_t output_cast_nb[GGML_MAX_DIMS];
        output_cast_nb[0] = sizeof(uint16_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            output_cast_nb[i] = output_cast_nb[i - 1] * output_cast_ne[i - 1];
        }

        aclTensor* acl_output_tensor = ggml_cann_create_tensor(
            output_buffer, ACL_FLOAT16, output_elem_size, output_cast_ne,
            output_cast_nb, GGML_MAX_DIMS);

        // print_acltensor(output_buffer,output_cast_ne,ACL_FLOAT16,
        // "output_tensor");
        aclTensor* acl_dst_tensor = ggml_cann_create_tensor(dst);
        aclnn_cast(ctx, acl_output_tensor, acl_dst_tensor,
                   ggml_cann_type_mapping(dst->type));
        // print_acltensor(dst->data,dst->ne,ggml_cann_type_mapping(dst->type),
        // "dst_tensor");
        ACL_CHECK(aclDestroyTensor(acl_output_tensor));
        ACL_CHECK(aclDestroyTensor(acl_dst_tensor));
    }
}

void ggml_cann_mul_mat_id_quant(ggml_backend_cann_context& ctx,
                                ggml_tensor* dst, const enum ggml_type type) {
    const ggml_tensor* src0 = dst->src[0];  // 矩阵A
    const ggml_tensor* src1 = dst->src[1];  // 矩阵B
    const ggml_tensor* ids = dst->src[2];   // ID张量

    //  print_device_tensor_elements(ctx, src0->data, "src0", src0->ne,
    //  src0->type,ggml_nelements(src0), 10, ctx.stream(), false, true, true);
    //  print_device_tensor_elements(ctx, src1->data, "src1",
    //  src1->ne,src1->type, ggml_nelements(src1), 10, ctx.stream(), false,
    //  true, true);

    GGML_TENSOR_BINARY_OP_LOCALS

    aclrtStream stream = ctx.stream();

    const int64_t n_as = ne02;         // 源矩阵A的批次数量
    const int64_t n_ids = ids->ne[0];  // ID数量

    // 从设备上获取IDs到主机
    std::vector<char> ids_host(ggml_nbytes(ids));
    const char* ids_dev = (const char*)ids->data;
    ACL_CHECK(aclrtMemcpyAsync(ids_host.data(), ggml_nbytes(ids), ids_dev,
                               ggml_nbytes(ids), ACL_MEMCPY_DEVICE_TO_HOST,
                               stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    char* src0_original = (char*)src0->data;
    char* src1_original = (char*)src1->data;
    char* dst_original = (char*)dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;

    float weight_elem_size;
    if (type == GGML_TYPE_Q4_0) {
        weight_elem_size = float(sizeof(uint8_t)) / 2;
    } else if (type == GGML_TYPE_Q8_0) {
        weight_elem_size = float(sizeof(uint8_t));
    } else {
        GGML_ABORT("Only support Q4_0 and Q8_0 MUL_MAT");
    }

    size_t weight_stride = src0->ne[1] * src0->ne[0] * weight_elem_size;
    size_t weight_size = weight_stride * src0->ne[2] * src0->ne[3];

    // printf src0_row.ne[1] src0_row.ne[0] src0->ne[0] src0->ne[1]
    //  printf("%d %d %d
    //  %d\n",src0_row.ne[1],src0_row.ne[0],src0->ne[0],src0->ne[1]);

    size_t scale_stride =
        src0_row.ne[1] * src0_row.ne[0] / QK8_0 * sizeof(uint16_t);
    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 =
                    *(const int32_t*)(ids_host.data() + iid1 * ids->nb[1] +
                                      id * ids->nb[0]);
                // print ids_host_data iid1 ids nb1 id nb0
                //  printf("%c %d %d %d %d
                //  %d\n",ids_host.data(),iid1,ids->nb[1],id,ids->nb[0],i02);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

                src0_row.data = src0_original + i02 * weight_stride;
                src1_row.data = src1_original + i11 * nb11 + i12 * nb12;
                dst_row.data = dst_original + i1 * nb1 + i2 * nb2;

                //  print_device_tensor_elements(ctx, dst_row.data, "dst",
                //  dst_row.ne, dst_row.type,ggml_nelements(&dst_row), 10,
                //  ctx.stream(), false, true, true);
                cal_mul_mat_quant(
                    ctx, &src0_row, &src1_row, &dst_row, type,
                    src0_original + weight_size + scale_stride * i02);
                //  print_device_tensor_elements(ctx, dst_row.data, "dst",
                //  dst_row.ne, dst_row.type,ggml_nelements(&dst_row), 10,
                //  ctx.stream(), false, true, true);
            }
        }
    } else {
        // 为src1和dst创建连续内存缓冲区
        void* src1_contiguous = nullptr;
        void* dst_contiguous = nullptr;

        // 为src1分配连续内存
        const size_t src1_size = sizeof(float) * ggml_nelements(src1);
        ACL_CHECK(aclrtMalloc(&src1_contiguous, src1_size,
                              ACL_MEM_MALLOC_HUGE_FIRST));

        // 为dst分配连续内存
        const size_t dst_size = sizeof(float) * ggml_nelements(dst);
        ACL_CHECK(
            aclrtMalloc(&dst_contiguous, dst_size, ACL_MEM_MALLOC_HUGE_FIRST));

        src1_row.data = src1_contiguous;
        dst_row.data = dst_contiguous;

        // 临时映射存储结构
        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        // 为每个可能的源矩阵处理
        for (int64_t i02 = 0; i02 < n_as; i02++) {
            int64_t num_src1_rows = 0;

            // 计算当前源矩阵的行数
            for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
                for (int64_t id = 0; id < n_ids; id++) {
                    const int32_t row_id_i =
                        *(const int32_t*)(ids_host.data() + iid1 * ids->nb[1] +
                                          id * ids->nb[0]);

                    GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

                    if (row_id_i != i02) {
                        continue;
                    }

                    num_src1_rows++;
                }
            }

            // 跳过没有行的矩阵
            if (num_src1_rows == 0) {
                continue;
            }

            // 分配映射数组，记录行的原始位置
            std::vector<mmid_row_mapping> row_mappings(num_src1_rows);

            // 将匹配的行复制到连续内存中
            int curr_row = 0;
            for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
                for (int64_t id = 0; id < n_ids; id++) {
                    const int32_t row_id_i =
                        *(const int32_t*)(ids_host.data() + iid1 * ids->nb[1] +
                                          id * ids->nb[0]);

                    if (row_id_i != i02) {
                        continue;
                    }

                    const int64_t i11 = id % ne11;
                    const int64_t i12 = iid1;

                    // 记录原始位置
                    row_mappings[curr_row].i1 = id;
                    row_mappings[curr_row].i2 = i12;

                    // 计算源位置
                    const float* src1_row_original =
                        (const float*)(src1_original + i11 * nb11 + i12 * nb12);
                    // 计算目标位置
                    float* src1_row_contiguous =
                        (float*)src1_contiguous + curr_row * ne10;

                    // 复制行到连续内存
                    ACL_CHECK(aclrtMemcpyAsync(
                        src1_row_contiguous, ne10 * sizeof(float),
                        src1_row_original, ne10 * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
                    ACL_CHECK(aclrtSynchronizeStream(stream));

                    curr_row++;
                }
            }

            // 设置输入矩阵
            src0_row.data = src0_original + i02 * weight_stride;

            GGML_ASSERT(nb11 == sizeof(float) * ne10);
            //  GGML_ASSERT(nb1 == sizeof(float)*ne0);

            src1_row.ne[1] = num_src1_rows;
            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows * nb11;
            src1_row.nb[3] = num_src1_rows * nb11;

            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows * nb1;
            dst_row.nb[3] = num_src1_rows * nb1;

            cal_mul_mat_quant(ctx, &src0_row, &src1_row, &dst_row, type,
                              src0_original + weight_size + scale_stride * i02);
            // 同步流以确保计算完成
            ACL_CHECK(aclrtSynchronizeStream(stream));

            // 将结果从连续内存复制回原始位置
            for (int64_t i = 0; i < num_src1_rows; i++) {
                const int32_t i1 = row_mappings[i].i1;
                const int32_t i2 = row_mappings[i].i2;

                const float* dst_row_contiguous =
                    (const float*)dst_contiguous + i * ne0;
                float* dst_row_original =
                    (float*)(dst_original + i1 * nb1 + i2 * nb2);

                // 复制结果行
                ACL_CHECK(aclrtMemcpyAsync(
                    dst_row_original, ne0 * sizeof(float), dst_row_contiguous,
                    ne0 * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
                ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        }

        // 释放连续内存
        ACL_CHECK(aclrtFree(src1_contiguous));
        ACL_CHECK(aclrtFree(dst_contiguous));
    }
}

void aclnn_moe_init_routing(
    ggml_backend_cann_context& ctx, aclTensor* acl_input_tensor,
    aclTensor* acl_row_index_tensor, aclTensor* acl_expert_index_tensor,
    aclTensor* acl_output_tensor, aclTensor* acl_expand_row_index_tensor,
    aclTensor* acl_expand_expert_index_tensor, int64_t active_num) {
    // 为aclnn_moe_init_routing操作分配工作空间
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    ACL_CHECK(aclnnMoeInitRoutingGetWorkspaceSize(
        acl_input_tensor, acl_row_index_tensor, acl_expert_index_tensor,
        active_num, acl_output_tensor, acl_expand_row_index_tensor,
        acl_expand_expert_index_tensor, &workspaceSize, &executor));
    // workspaceSize ;
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    // 执行aclnn_moe_init_routing操作
    ACL_CHECK(aclnnMoeInitRouting(workspaceAddr, workspaceSize, executor,
                                  ctx.stream()));
}
void aclnn_moe_compute_expert_tokens(ggml_backend_cann_context& ctx,
                                     aclTensor* acl_expand_expert_idx_tensor,
                                     aclTensor* acl_expert_tokens_tensor,
                                     int64_t num_experts) {
    // 为aclnn_moe_compute_expert_tokens操作分配工作空间
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    ACL_CHECK(aclnnMoeComputeExpertTokensGetWorkspaceSize(
        acl_expand_expert_idx_tensor, num_experts, acl_expert_tokens_tensor,
        &workspaceSize, &executor));
    // workspaceSize+= 64;
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    // 执行aclnn_moe_compute_expert_tokens操作
    ACL_CHECK(aclnnMoeComputeExpertTokens(workspaceAddr, workspaceSize,
                                          executor, ctx.stream()));
}
void print_acl_tensor_datatype_with_name(aclTensor* tensor, char* name) {
    aclDataType type;
    aclGetDataType(tensor, &type);
    if (type == ACL_FLOAT16) {
        printf("%s:ACL_FLOAT16\n", name);
    } else if (type == ACL_FLOAT) {
        printf("%s:ACL_FLOAT\n", name);
    } else if (type == ACL_INT32) {
        printf("%s:ACL_INT32\n", name);
    } else if (type == ACL_INT64) {
        printf("%s:ACL_INT64\n", name);
    } else {
        printf("%s:other\n", name);
    }
}
void aclnn_grouped_matmul(ggml_backend_cann_context& ctx,
                          aclTensor* acl_x_tensor,
                          aclTensor* acl_weights_tensor,
                          aclTensor* acl_expert_tokens_tensor,
                          aclTensor* acl_dst_tensor, int64_t active_type = 0) {
    // 为aclnn_grouped_matmul操作分配工作空间
    aclOpExecutor* executor = nullptr;
    // auto x_type = acl_x_tensor->dataType;
    // print_acl_tensor_datatype_with_name(acl_x_tensor,"x_tensor");
    // print_acl_tensor_datatype_with_name(acl_weights_tensor,"weights_tensor");
    // print_acl_tensor_datatype_with_name(acl_expert_tokens_tensor,"expert_tokens_tensor");
    // print_acl_tensor_datatype_with_name(acl_dst_tensor,"dst_tensor");
    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    // acl_expert_tokens_tensor = nullptr;
    // aclTensorList
    std::vector<aclTensor*> x_list{acl_x_tensor};
    std::vector<aclTensor*> weights_list{acl_weights_tensor};
    std::vector<aclTensor*> dst_list{acl_dst_tensor};
    aclTensorList* x_tensorList =
        aclCreateTensorList(x_list.data(), x_list.size());
    aclTensorList* weights_tensorList =
        aclCreateTensorList(weights_list.data(), weights_list.size());
    aclTensorList* dst_tensorList =
        aclCreateTensorList(dst_list.data(), dst_list.size());
    // aclnnStatus aclnnGroupedMatmulV4GetWorkspaceSize(const aclTensorList *x,
    // const aclTensorList *weight, const aclTensorList *biasOptional, const
    // aclTensorList *scaleOptional, const aclTensorList *offsetOptional, const
    // aclTensorList *antiquantScaleOptional, const aclTensorList
    // *antiquantOffsetOptional, const aclTensorList *perTokenScaleOptional,
    // const aclTensor *groupListOptional, const aclTensorList
    // *activationInputOptional, const aclTensorList
    // *activationQuantScaleOptional, const aclTensorList
    // *activationQuantOffsetOptional, int64_t splitItem, int64_t groupType,
    // int64_t groupListType, int64_t actType, aclTensorList *out, aclTensorList
    // *activationFeatureOutOptional, aclTensorList *dynQuantScaleOutOptional,
    // uint64_t *workspaceSize, aclOpExecutor **executor)
    ACL_CHECK(aclnnGroupedMatmulV4GetWorkspaceSize(
        x_tensorList,        // const aclTensorList *x
        weights_tensorList,  // const aclTensorList *weight
        nullptr,             // const aclTensorList *biasOptional
        nullptr,             // const aclTensorList *scaleOptional
        nullptr,             // const aclTensorList *offsetOptional
        nullptr,             // const aclTensorList *antiquantScaleOptional
        nullptr,             // const aclTensorList *antiquantOffsetOptional
        nullptr,             // const aclTensorList *perTokenScaleOptional
        acl_expert_tokens_tensor,  // const aclTensor *groupListOptional
        nullptr,         // const aclTensorList *activationInputOptional
        nullptr,         // const aclTensorList *activationQuantScaleOptional
        nullptr,         // const aclTensorList *activationQuantOffsetOptional
        2,               // int64_t splitItem
        0,               // int64_t groupType
        0,               // int64_t groupListType
        0,               // int64_t actType
        dst_tensorList,  // aclTensorList *out
        nullptr,         // aclTensorList *activationFeatureOutOptional
        nullptr,         // aclTensorList *dynQuantScaleOutOptional
        &workspaceSize,  // uint64_t *workspaceSize
        &executor));     // aclOpExecutor **executor
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    // 执行aclnn_grouped_matmul操作
    ACL_CHECK(aclnnGroupedMatmulV4(workspaceAddr, workspaceSize, executor,
                                   ctx.stream()));
    // 释放TensorList
    // FIXME: 释放TensorList会导致程序崩溃
    // ACL_CHECK(aclDestroyTensorList(x_tensorList));
    // ACL_CHECK(aclDestroyTensorList(weights_tensorList));
    // ACL_CHECK(aclDestroyTensorList(dst_tensorList));
}
static void aclnn_permute(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                          aclTensor* acl_dst, int64_t* new_dim, uint64_t dims);
static void aclnn_index_fill_tensor(ggml_backend_cann_context& ctx,
                                    aclTensor* acl_src, int64_t dim,
                                    int64_t* index, int64_t index_num,
                                    float value);
void aclnn_index_select(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                        aclTensor* acl_dst, aclTensor* acl_index, int64_t dim) {
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    ACL_CHECK(aclnnIndexSelectGetWorkspaceSize(acl_src, dim, acl_index, acl_dst,
                                               &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(
        aclnnIndexSelect(workspaceAddr, workspaceSize, executor, ctx.stream()));
}
void aclnn_index_copy_inplace(ggml_backend_cann_context& ctx,
                              aclTensor* acl_src, aclTensor* acl_dst,
                              aclTensor* acl_index, int64_t dim) {
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    ACL_CHECK(aclnnInplaceIndexCopyGetWorkspaceSize(
        acl_dst, dim, acl_index, acl_src, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(aclnnInplaceIndexCopy(workspaceAddr, workspaceSize, executor,
                                    ctx.stream()));
}

void ggml_cann_mul_mat_id_fp(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];  // 矩阵A
    const ggml_tensor* src1 = dst->src[1];  // 矩阵B
    const ggml_tensor* ids = dst->src[2];   // ID张量

    GGML_TENSOR_BINARY_OP_LOCALS

    aclrtStream stream = ctx.stream();

    const int64_t n_as = ne02;         // 源矩阵A的批次数量
    const int64_t n_ids = ids->ne[0];  // ID数量

    // 从设备上获取IDs到主机
    std::vector<char> ids_host(ggml_nbytes(ids));
    const char* ids_dev = (const char*)ids->data;
    ACL_CHECK(aclrtMemcpyAsync(ids_host.data(), ggml_nbytes(ids), ids_dev,
                               ggml_nbytes(ids), ACL_MEMCPY_DEVICE_TO_HOST,
                               stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    char* src0_original = (char*)src0->data;
    char* src1_original = (char*)src1->data;
    char* dst_original = (char*)dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;

    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 =
                    *(const int32_t*)(ids_host.data() + iid1 * ids->nb[1] +
                                      id * ids->nb[0]);

                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

                src0_row.data = src0_original + i02 * nb02;
                src1_row.data = src1_original + i11 * nb11 + i12 * nb12;
                dst_row.data = dst_original + i1 * nb1 + i2 * nb2;

                // 修改这里的调用，使用正确的参数
                aclTensor* acl_input_tensor = ggml_cann_create_tensor(
                    &src0_row, src0_row.ne, src0_row.nb, 4);
                aclTensor* acl_weight_tensor =
                    ggml_cann_create_tensor_transpose(&src1_row, src1_row.ne,
                                                      src1_row.nb, 4);
                aclTensor* acl_dst = ggml_cann_create_tensor_transpose(
                    &dst_row, dst_row.ne, dst_row.nb, 4);

                aclnn_mat_mul(ctx, acl_input_tensor, acl_weight_tensor,
                              acl_dst);

                // 确保释放这些资源
                ACL_CHECK(aclDestroyTensor(acl_input_tensor));
                ACL_CHECK(aclDestroyTensor(acl_weight_tensor));
                ACL_CHECK(aclDestroyTensor(acl_dst));
            }
        }
    } else {
        // 为src1和dst创建连续内存缓冲区
        void* src1_contiguous = nullptr;
        void* dst_contiguous = nullptr;

        // 为src1分配连续内存
        const size_t src1_size = sizeof(float) * ggml_nelements(src1);
        ACL_CHECK(aclrtMalloc(&src1_contiguous, src1_size,
                              ACL_MEM_MALLOC_HUGE_FIRST));

        // 为dst分配连续内存
        const size_t dst_size = sizeof(float) * ggml_nelements(dst);
        ACL_CHECK(
            aclrtMalloc(&dst_contiguous, dst_size, ACL_MEM_MALLOC_HUGE_FIRST));

        src1_row.data = src1_contiguous;
        dst_row.data = dst_contiguous;

        // 临时映射存储结构
        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        // 为每个可能的源矩阵处理
        for (int64_t i02 = 0; i02 < n_as; i02++) {
            int64_t num_src1_rows = 0;

            // 计算当前源矩阵的行数
            for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
                for (int64_t id = 0; id < n_ids; id++) {
                    const int32_t row_id_i =
                        *(const int32_t*)(ids_host.data() + iid1 * ids->nb[1] +
                                          id * ids->nb[0]);

                    GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

                    if (row_id_i != i02) {
                        continue;
                    }

                    num_src1_rows++;
                }
            }

            // 跳过没有行的矩阵
            if (num_src1_rows == 0) {
                continue;
            }

            // 分配映射数组，记录行的原始位置
            std::vector<mmid_row_mapping> row_mappings(num_src1_rows);

            // 将匹配的行复制到连续内存中
            int curr_row = 0;
            for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
                for (int64_t id = 0; id < n_ids; id++) {
                    const int32_t row_id_i =
                        *(const int32_t*)(ids_host.data() + iid1 * ids->nb[1] +
                                          id * ids->nb[0]);

                    if (row_id_i != i02) {
                        continue;
                    }

                    const int64_t i11 = id % ne11;
                    const int64_t i12 = iid1;

                    // 记录原始位置
                    row_mappings[curr_row].i1 = id;
                    row_mappings[curr_row].i2 = i12;

                    // 计算源位置
                    const float* src1_row_original =
                        (const float*)(src1_original + i11 * nb11 + i12 * nb12);
                    // 计算目标位置
                    float* src1_row_contiguous =
                        (float*)src1_contiguous + curr_row * ne10;

                    // 复制行到连续内存
                    ACL_CHECK(aclrtMemcpyAsync(
                        src1_row_contiguous, ne10 * sizeof(float),
                        src1_row_original, ne10 * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
                    // ACL_CHECK(aclrtSynchronizeStream(stream));

                    curr_row++;
                }
            }

            // 设置输入矩阵
            src0_row.data = src0_original + i02 * nb02;

            GGML_ASSERT(nb11 == sizeof(float) * ne10);
            GGML_ASSERT(nb1 == sizeof(float) * ne0);

            src1_row.ne[1] = num_src1_rows;
            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows * nb11;
            src1_row.nb[3] = num_src1_rows * nb11;

            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows * nb1;
            dst_row.nb[3] = num_src1_rows * nb1;

            // 创建CANN张量
            aclTensor* acl_input_tensor =
                ggml_cann_create_tensor(&src0_row, src0_row.ne, src0_row.nb, 4);
            aclTensor* acl_weight_tensor = ggml_cann_create_tensor_transpose(
                &src1_row, src1_row.ne, src1_row.nb, 4);
            aclTensor* acl_output_tensor = ggml_cann_create_tensor_transpose(
                &dst_row, dst_row.ne, dst_row.nb, 4);

            // 执行矩阵乘法
            aclnn_mat_mul(ctx, acl_input_tensor, acl_weight_tensor,
                          acl_output_tensor);

            // 释放张量
            ACL_CHECK(aclDestroyTensor(acl_input_tensor));
            ACL_CHECK(aclDestroyTensor(acl_weight_tensor));
            ACL_CHECK(aclDestroyTensor(acl_output_tensor));

            // 同步流以确保计算完成
            ACL_CHECK(aclrtSynchronizeStream(stream));

            // 将结果从连续内存复制回原始位置
            for (int64_t i = 0; i < num_src1_rows; i++) {
                const int32_t i1 = row_mappings[i].i1;
                const int32_t i2 = row_mappings[i].i2;

                const float* dst_row_contiguous =
                    (const float*)dst_contiguous + i * ne0;
                float* dst_row_original =
                    (float*)(dst_original + i1 * nb1 + i2 * nb2);

                // 复制结果行
                ACL_CHECK(aclrtMemcpyAsync(
                    dst_row_original, ne0 * sizeof(float), dst_row_contiguous,
                    ne0 * sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
                // ACL_CHECK(aclrtSynchronizeStream(stream));
            }
        }

        // 释放连续内存
        ACL_CHECK(aclrtFree(src1_contiguous));
        ACL_CHECK(aclrtFree(dst_contiguous));
    }
}
void aclnn_silu(ggml_backend_cann_context& ctx, aclTensor* acl_input_tensor,
                aclTensor* acl_output_tensor) {
    aclOpExecutor* executor;
    size_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    ACL_CHECK(aclnnSiluGetWorkspaceSize(acl_input_tensor, acl_output_tensor,
                                        &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(aclnnSilu(workspaceAddr, workspaceSize, executor, ctx.stream()));
}
void ggml_cann_mul_mat_id(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    const enum ggml_type type = dst->src[0]->type;
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            ggml_cann_mul_mat_id_fp(ctx, dst);
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_id_quant(ctx, dst, type);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
void aclnn_moe_finalize_routing(ggml_backend_cann_context& ctx,
                                aclTensor* moe_weights, aclTensor* moe_down,
                                aclTensor* expanded_row_idx,
                                aclTensor* expert_idx, aclTensor* out) {
    aclOpExecutor* executor;
    // aclnnStatus aclnnMoeFinalizeRoutingV2GetWorkspaceSize(const aclTensor*
    // expandedX, const aclTensor* expandedRowIdx, const aclTensor* x1Optional,
    // const aclTensor* x2Optional, const aclTensor* biasOptional, const
    // aclTensor* scalesOptional,const aclTensor* expertIdxOptional, int64_t
    // dropPadMode, const aclTensor* out, uint64_t* workspaceSize,
    // aclOpExecutor** executor)
    size_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    ACL_CHECK(aclnnMoeFinalizeRoutingV2GetWorkspaceSize(
        moe_down, expanded_row_idx, nullptr, nullptr, nullptr, moe_weights,
        expert_idx, 0, out, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(aclnnMoeFinalizeRoutingV2(workspaceAddr, workspaceSize, executor,
                                        ctx.stream()));
}

static void aclnn_mul(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                      aclTensor* acl_other, aclTensor* acl_dst);
static void aclnn_mul_inplace(ggml_backend_cann_context& ctx,
                              aclTensor* acl_src, aclTensor* acl_other);

static void aclnn_fill_scalar(ggml_backend_cann_context& ctx, float scalar,
                              aclTensor* acl_dst);

// 二分查找数组中第一个大于等于target的元素位置
int find_lower_bound(const int32_t* arr, int size, int32_t target) {
    int left = 0;
    int right = size - 1;
    int result = size;  // 如果所有元素都小于target，返回size

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] >= target) {
            result = mid;     // 记录当前找到的位置
            right = mid - 1;  // 继续在左半部分查找更小的索引
        } else {
            left = mid + 1;
        }
    }

    return result;
}

// 二分查找数组中最后一个小于等于target的元素位置
int find_upper_bound(const int32_t* arr, int size, int32_t target) {
    int left = 0;
    int right = size - 1;
    int result = -1;  // 如果所有元素都大于target，返回-1

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] <= target) {
            result = mid;    // 记录当前找到的位置
            left = mid + 1;  // 继续在右半部分查找更大的索引
        } else {
            right = mid - 1;
        }
    }

    return result;
}

void ggml_cann_moe_fused(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* input = dst->src[0];
    ggml_tensor* ids = dst->src[1];
    ggml_tensor* topk_weight = dst->src[2];
    ggml_tensor* expert_up_weights = dst->src[3];
    ggml_tensor* expert_down_weights = dst->src[4];
    ggml_tensor* expert_gate_weights = dst->src[5];
    ggml_tensor* row_idx = dst->src[6];
    int32_t start_idx = dst->op_params[0];
    int32_t end_idx = dst->op_params[1];
    auto batch_size = input->ne[3];
    GGML_ASSERT(batch_size == 1);
    auto seq_len = input->ne[2];
    auto topk = ids->ne[0];
    auto num_experts = expert_up_weights->ne[2];
    auto num_rows = batch_size * seq_len;
    auto hidden_dim = input->ne[0];
    auto k_dim = expert_up_weights->ne[1];
    // printf("seq_len:%d, topk:%d, num_experts:%d, num_rows:%d, hidden_dim:%d,
    // k_dim:%d,start_idx:%d, end_idx:%d\n", seq_len, topk, num_experts,
    // num_rows, hidden_dim, k_dim,start_idx,end_idx);
    GGML_ASSERT(input->ne[0] == hidden_dim);
    GGML_ASSERT(input->ne[1] == 1);
    GGML_ASSERT(input->ne[2] == seq_len);
    GGML_ASSERT(input->ne[3] == 1);
    GGML_ASSERT(ids->ne[0] == topk);
    GGML_ASSERT(ids->ne[1] == seq_len);
    GGML_ASSERT(ids->ne[2] == 1);
    GGML_ASSERT(ids->ne[3] == 1);
    GGML_ASSERT(expert_up_weights->ne[0] == hidden_dim);
    GGML_ASSERT(expert_up_weights->ne[1] == k_dim);
    GGML_ASSERT(expert_up_weights->ne[2] == num_experts);
    GGML_ASSERT(expert_up_weights->ne[3] == 1);
    GGML_ASSERT(expert_down_weights->ne[0] == k_dim);
    GGML_ASSERT(expert_down_weights->ne[1] == hidden_dim);
    GGML_ASSERT(expert_down_weights->ne[2] == num_experts);
    GGML_ASSERT(expert_down_weights->ne[3] == 1);
    GGML_ASSERT(expert_gate_weights->ne[0] == hidden_dim);
    GGML_ASSERT(expert_gate_weights->ne[1] == k_dim);
    GGML_ASSERT(expert_gate_weights->ne[2] == num_experts);
    GGML_ASSERT(expert_gate_weights->ne[3] == 1);
    GGML_ASSERT(row_idx->ne[0] == seq_len);
    GGML_ASSERT(row_idx->ne[1] == topk);
    GGML_ASSERT(row_idx->ne[2] == 1);
    GGML_ASSERT(row_idx->ne[3] == 1);
    GGML_ASSERT(topk_weight->ne[0] == 1);
    GGML_ASSERT(topk_weight->ne[1] == topk);
    GGML_ASSERT(topk_weight->ne[2] == seq_len);
    GGML_ASSERT(topk_weight->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == hidden_dim);
    GGML_ASSERT(dst->ne[1] == seq_len);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == 1);
    GGML_ASSERT(topk_weight->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input));
    // GGML_ASSERT(ggml_is_contiguous(ids));
    GGML_ASSERT(ggml_is_contiguous(row_idx));
    GGML_ASSERT(ggml_is_contiguous(topk_weight));
    GGML_ASSERT(ggml_is_contiguous(expert_up_weights));
    GGML_ASSERT(ggml_is_contiguous(expert_down_weights));
    GGML_ASSERT(ggml_is_contiguous(expert_gate_weights));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(num_experts == end_idx - start_idx + 1);

    int64_t active_num = batch_size * seq_len;
    int64_t num_length = seq_len * topk;

    int64_t input_ne[] = {
        input->ne[0],
        input->ne[1] * input->ne[2],
    };
    size_t input_nb[] = {
        ggml_element_size(input),
        ggml_element_size(input) * input->ne[0],
    };
    aclTensor* acl_input_tensor =
        ggml_cann_create_tensor(input, input_ne, input_nb, 2, ACL_FORMAT_ND);

    // int64_t expert_idx_ne[] = {topk, seq_len};
    // size_t expert_idx_nb[] = {ggml_element_size(ids), ggml_element_size(ids)
    // * expert_idx_ne[0]};
    int64_t expert_idx_ne[] = {ids->ne[0], ids->ne[1]};
    size_t expert_idx_nb[] = {ids->nb[0], ids->nb[1]};
    aclTensor* acl_expert_idx_tensor = ggml_cann_create_tensor(
        ids, expert_idx_ne, expert_idx_nb, 2, ACL_FORMAT_ND);
    // support no-contiguous tensor
    int64_t row_idx_ne[] = {row_idx->ne[0], row_idx->ne[1]};
    size_t row_idx_nb[] = {row_idx->nb[0], row_idx->nb[1]};
    aclTensor* acl_row_idx_tensor = ggml_cann_create_tensor(
        row_idx, row_idx_ne, row_idx_nb, 2, ACL_FORMAT_ND);
    // print_acltensor(row_idx->data,
    // row_idx_ne,ggml_cann_type_mapping(row_idx->type),
    // "input_row_idx_tensor",2);

    int64_t row_idx_permute_ne[] = {row_idx->ne[1], row_idx->ne[0]};
    size_t row_idx_permute_nb[] = {
        ggml_element_size(row_idx),
        ggml_element_size(row_idx) * row_idx_permute_ne[0]};
    ggml_cann_pool_alloc row_idx_permute_allocator(
        ctx.pool(), row_idx_permute_ne[0] * row_idx_permute_ne[1] *
                        ggml_element_size(row_idx));
    void* row_idx_permute_buffer = row_idx_permute_allocator.get();
    aclTensor* acl_row_idx_permute_tensor = ggml_cann_create_tensor(
        row_idx_permute_buffer, ggml_cann_type_mapping(row_idx->type),
        ggml_element_size(row_idx), row_idx_permute_ne, row_idx_permute_nb, 2,
        ACL_FORMAT_ND);
    int64_t permute_dim[] = {1, 0};
    aclnn_permute(ctx, acl_row_idx_tensor, acl_row_idx_permute_tensor,
                  permute_dim, 2);
    // aclrtSynchronizeStream(ctx.stream());
    // print_acltensor(row_idx_permute_buffer,row_idx_permute_ne,ggml_cann_type_mapping(row_idx->type),
    // "row_idx_permute_buffer",2);

    int64_t expand_input_ne[2] = {hidden_dim,
                                  std::min(num_rows, active_num) * topk};
    size_t expand_input_nb[2] = {ggml_element_size(input),
                                 ggml_element_size(input) * expand_input_ne[0]};
    ggml_cann_pool_alloc expand_input_allocator(
        ctx.pool(),
        expand_input_ne[0] * expand_input_ne[1] * ggml_element_size(input));
    void* expand_input_buffer = expand_input_allocator.get();
    aclTensor* acl_expand_input_tensor = ggml_cann_create_tensor(
        expand_input_buffer, ggml_cann_type_mapping(input->type),
        ggml_element_size(input), expand_input_ne, expand_input_nb, 2,
        ACL_FORMAT_ND);

    int64_t expand_expert_idx_ne[] = {num_rows * topk};
    size_t expand_expert_idx_nb[] = {ggml_element_size(ids)};
    ggml_cann_pool_alloc expand_expert_idx_allocator(
        ctx.pool(), expand_expert_idx_ne[0] * ggml_element_size(ids));
    void* expand_expert_idx_buffer = expand_expert_idx_allocator.get();
    aclTensor* acl_expand_expert_idx_tensor = ggml_cann_create_tensor(
        expand_expert_idx_buffer, ggml_cann_type_mapping(ids->type),
        ggml_element_size(ids), expand_expert_idx_ne, expand_expert_idx_nb, 1,
        ACL_FORMAT_ND);

    int64_t expand_row_idx_ne[] = {num_rows * topk};
    size_t expand_row_idx_nb[] = {ggml_element_size(row_idx)};
    ggml_cann_pool_alloc expand_row_idx_allocator(
        ctx.pool(), expand_row_idx_ne[0] * ggml_element_size(row_idx));
    void* expand_row_idx_buffer = expand_row_idx_allocator.get();
    aclTensor* acl_expand_row_idx_tensor = ggml_cann_create_tensor(
        expand_row_idx_buffer, ggml_cann_type_mapping(row_idx->type),
        ggml_element_size(row_idx), expand_row_idx_ne, expand_row_idx_nb, 1,
        ACL_FORMAT_ND);
    // aclrtSynchronizeStream(ctx.stream());

    aclnn_moe_init_routing(ctx, acl_input_tensor, acl_row_idx_permute_tensor,
                           acl_expert_idx_tensor, acl_expand_input_tensor,
                           acl_expand_row_idx_tensor,
                           acl_expand_expert_idx_tensor, active_num);

    num_experts = end_idx - start_idx + 1;
    // num_experts = 64;
    int32_t* expert_idx_host = new int32_t[num_rows * topk];
    aclrtMemcpyAsync(expert_idx_host, num_rows * topk * ggml_element_size(ids),
                     expand_expert_idx_buffer,
                     num_rows * topk * ggml_element_size(ids),
                     ACL_MEMCPY_DEVICE_TO_HOST, ctx.stream());
    aclrtSynchronizeStream(ctx.stream());

    // 查找
    auto range_start =
        find_lower_bound(expert_idx_host, topk * seq_len, start_idx);
    auto range_end = find_upper_bound(expert_idx_host, topk * seq_len, end_idx);
    // printf("range_start:%d range_end:%d\n", range_start, range_end);
    // GGML_ASSERT(range_start >= 0 && range_end < num_rows*topk);
    // GGML_ASSERT(range_start <= range_end);
    int64_t f_dst_ne[2] = {hidden_dim, seq_len};
    size_t f_dst_nb[2] = {ggml_element_size(dst),
                          ggml_element_size(dst) * f_dst_ne[0]};
    aclTensor* acl_f_dst_tensor =
        ggml_cann_create_tensor(dst, f_dst_ne, f_dst_nb, 2, ACL_FORMAT_ND);
    if (range_start > range_end) {
        aclnn_fill_scalar(ctx, 0.0f, acl_f_dst_tensor);
        return;
    }

    int32_t range_size = range_end - range_start + 1;
    num_length = range_size;
    int64_t expand_expert_idx_new_ne[] = {range_size};
    size_t expand_expert_idx_new_nb[] = {ggml_type_size(GGML_TYPE_I32)};
    acl_expand_expert_idx_tensor = ggml_cann_create_tensor(
        (char*)expand_expert_idx_buffer +
            range_start * ggml_type_size(GGML_TYPE_I32),
        ggml_cann_type_mapping(GGML_TYPE_I32), ggml_type_size(GGML_TYPE_I32),
        expand_expert_idx_new_ne, expand_expert_idx_new_nb, 1, ACL_FORMAT_ND);

    int64_t expand_input_new_ne[2] = {hidden_dim, range_size};
    size_t expand_input_new_nb[2] = {
        ggml_element_size(input),
        ggml_element_size(input) * expand_input_new_ne[0]};
    acl_expand_input_tensor = ggml_cann_create_tensor(
        (char*)expand_input_buffer +
            range_start * ggml_element_size(input) * hidden_dim,
        ggml_cann_type_mapping(input->type), ggml_element_size(input),
        expand_input_new_ne, expand_input_new_nb, 2, ACL_FORMAT_ND);
    // TODO:减去
    int64_t expand_row_idx_new_add_ne[1] = {range_size};
    size_t expand_row_idx_new_add_nb[1] = {ggml_type_size(GGML_TYPE_I32)};
    ggml_cann_pool_alloc expand_row_idx_new_add_allocator(
        ctx.pool(),
        expand_row_idx_new_add_ne[0] * ggml_type_size(GGML_TYPE_I32));
    void* expand_row_idx_new_add_buffer =
        expand_row_idx_new_add_allocator.get();
    aclTensor* acl_expand_row_idx_new_add_tensor = ggml_cann_create_tensor(
        expand_row_idx_new_add_buffer, ggml_cann_type_mapping(GGML_TYPE_I32),
        ggml_type_size(GGML_TYPE_I32), expand_row_idx_new_add_ne,
        expand_row_idx_new_add_nb, 1, ACL_FORMAT_ND);

    aclnn_adds_int(ctx, acl_expand_expert_idx_tensor, -start_idx,
                   acl_expand_row_idx_new_add_tensor);
    acl_expand_expert_idx_tensor = acl_expand_row_idx_new_add_tensor;

    // print_acltensor(expand_row_idx_buffer,expand_row_idx_ne,ggml_cann_type_mapping(row_idx->type),
    // "expand_row_idx_tensor",1,nullptr,&ctx);
    // aclrtSynchronizeStream(ctx.stream());
    // print_acltensor(row_idx_permute_buffer,row_idx_permute_ne,ggml_cann_type_mapping(row_idx->type),
    // "row_idx_permute_buffer",2);
    // print_acltensor(expand_row_idx_buffer,expand_row_idx_ne,ggml_cann_type_mapping(GGML_TYPE_I32),
    // "expand_row_idx_tensor",1);
    // print_acltensor(expand_expert_idx_buffer,expand_expert_idx_ne,ggml_cann_type_mapping(ids->type),
    // "expand_expert_idx_tensor",1);
    // print_acltensor(expand_input_buffer,expand_input_ne,ggml_cann_type_mapping(input->type),
    // "expand_input_tensor1",2); return;
    int64_t expert_tokens_ne[] = {num_experts};
    size_t expert_tokens_nb[] = {ggml_type_size(GGML_TYPE_I32)};
    ggml_cann_pool_alloc expert_tokens_allocator(
        ctx.pool(), expert_tokens_ne[0] * ggml_type_size(GGML_TYPE_I32));
    void* expert_tokens_buffer = expert_tokens_allocator.get();
    aclTensor* acl_expert_tokens_tensor = ggml_cann_create_tensor(
        expert_tokens_buffer, ggml_cann_type_mapping(GGML_TYPE_I32),
        ggml_type_size(GGML_TYPE_I32), expert_tokens_ne, expert_tokens_nb, 1,
        ACL_FORMAT_ND);
    // aclrtSynchronizeStream(ctx.stream());
    aclnn_moe_compute_expert_tokens(ctx, acl_expand_expert_idx_tensor,
                                    acl_expert_tokens_tensor, num_experts);
    // aclrtSynchronizeStream(ctx.stream());

    int64_t expert_tokens_int64_ne[1] = {num_experts};
    size_t expert_tokens_int64_nb[1] = {ggml_type_size(GGML_TYPE_I64)};
    ggml_cann_pool_alloc expert_tokens_int64_allocator(
        ctx.pool(), expert_tokens_int64_ne[0] * ggml_type_size(GGML_TYPE_I64));
    void* expert_tokens_int64_buffer = expert_tokens_int64_allocator.get();
    aclTensor* acl_expert_tokens_int64_tensor = ggml_cann_create_tensor(
        expert_tokens_int64_buffer, ggml_cann_type_mapping(GGML_TYPE_I64),
        ggml_type_size(GGML_TYPE_I64), expert_tokens_int64_ne,
        expert_tokens_int64_nb, 1, ACL_FORMAT_ND);
    // print_acltensor(expert_tokens_buffer,expert_tokens_ne,ACL_INT32,
    // "expert_tokens_tensor",1);
    aclnn_cast(ctx, acl_expert_tokens_tensor, acl_expert_tokens_int64_tensor,
               ACL_INT64);
    // aclrtSynchronizeStream(ctx.stream());
    int64_t up_permute_ne[3] = {expert_up_weights->ne[0],
                                expert_up_weights->ne[1],
                                expert_up_weights->ne[2]};
    size_t up_permute_nb[3];
    up_permute_nb[0] = ggml_element_size(expert_up_weights);
    up_permute_nb[1] = up_permute_nb[0] * up_permute_ne[0];
    up_permute_nb[2] = up_permute_nb[1] * up_permute_ne[1];
    aclTensor* acl_expert_up_weights_t_tensor =
        ggml_cann_create_tensor_transpose(expert_up_weights, up_permute_ne,
                                          up_permute_nb, 3);
    int64_t moe_up_ne[2] = {k_dim, num_length};
    size_t moe_up_nb[2];
    moe_up_nb[0] = ggml_element_size(expert_up_weights);
    moe_up_nb[1] = moe_up_nb[0] * moe_up_ne[0];
    // moe_up_nb[2] = moe_up_nb[1] * moe_up_ne[1];
    ggml_cann_pool_alloc moe_up_allocator(
        ctx.pool(),
        moe_up_ne[0] * moe_up_ne[1] * ggml_element_size(expert_up_weights));
    void* moe_up_buffer = moe_up_allocator.get();
    aclTensor* acl_moe_up_tensor = ggml_cann_create_tensor(
        moe_up_buffer, ggml_cann_type_mapping(expert_up_weights->type),
        ggml_element_size(expert_up_weights), moe_up_ne, moe_up_nb, 2,
        ACL_FORMAT_ND);
    aclnn_grouped_matmul(ctx, acl_expand_input_tensor,
                         acl_expert_up_weights_t_tensor,
                         acl_expert_tokens_int64_tensor, acl_moe_up_tensor);

    // print_acltensor(moe_up_buffer,moe_up_ne,ggml_cann_type_mapping(expert_up_weights->type),"moe_up_tensor",2);

    // int64_t tmp_out_ne[2];
    // size_t tmp_out_nb[2];
    // tmp_out_ne[0] = moe_up_ne[0];
    // tmp_out_ne[1] = moe_up_ne[1];
    // tmp_out_nb[0] = moe_up_nb[0];
    // tmp_out_nb[1] = moe_up_nb[1];

    // ggml_cann_pool_alloc
    // tmp_out_allocator(ctx.pool(),tmp_out_ne[0]*tmp_out_ne[1]*ggml_element_size(expert_up_weights));
    // void* tmp_out_buffer = tmp_out_allocator.get();
    // aclTensor* acl_tmp_out_tensor = ggml_cann_create_tensor(tmp_out_buffer,
    // ggml_cann_type_mapping(expert_up_weights->type),
    // ggml_element_size(expert_up_weights), tmp_out_ne, tmp_out_nb, 2,
    // ACL_FORMAT_ND);

    // aclnn_index_select(ctx,acl_moe_up_tensor,acl_tmp_out_tensor,acl_expand_row_idx_tensor,0);
    // print_acltensor(tmp_out_buffer,tmp_out_ne,ggml_cann_type_mapping(expert_up_weights->type),"acl_moe_up_tensor6666666666666666666666666",2,nullptr,&ctx);
    // int64_t
    // gate_permute_ne[3]={expert_gate_weights->ne[0],expert_gate_weights->ne[1],expert_gate_weights->ne[2]};
    // size_t gate_permute_nb[3];
    // gate_permute_nb[0] = ggml_element_size(expert_gate_weights);
    // gate_permute_nb[1] = gate_permute_nb[0] * gate_permute_ne[0];
    // gate_permute_nb[2] = gate_permute_nb[1] * gate_permute_ne[1];
    // aclTensor*
    // ed_matmul(ctx,acl_expand_input_tensor,acl_expert_up_weights_t_tensor,acl_expert_tokens_int64_tensor,acl_moe_up_tensor);

    int64_t gate_permute_ne[3] = {expert_gate_weights->ne[0],
                                  expert_gate_weights->ne[1],
                                  expert_gate_weights->ne[2]};
    size_t gate_permute_nb[3];
    gate_permute_nb[0] = ggml_element_size(expert_gate_weights);
    gate_permute_nb[1] = gate_permute_nb[0] * gate_permute_ne[0];
    gate_permute_nb[2] = gate_permute_nb[1] * gate_permute_ne[1];
    aclTensor* acl_expert_gate_weights_t_tensor =
        ggml_cann_create_tensor_transpose(expert_gate_weights, gate_permute_ne,
                                          gate_permute_nb, 3);

    int64_t moe_gate_ne[2] = {k_dim, num_length};
    size_t moe_gate_nb[2];
    moe_gate_nb[0] = ggml_element_size(expert_gate_weights);
    moe_gate_nb[1] = moe_gate_nb[0] * moe_gate_ne[0];
    // moe_gate_nb[2] = moe_gate_nb[1] * moe_gate_ne[1];
    ggml_cann_pool_alloc moe_gate_allocator(
        ctx.pool(), moe_gate_ne[0] * moe_gate_ne[1] *
                        ggml_element_size(expert_gate_weights));
    void* moe_gate_buffer = moe_gate_allocator.get();
    aclTensor* acl_moe_gate_tensor = ggml_cann_create_tensor(
        moe_gate_buffer, ggml_cann_type_mapping(expert_gate_weights->type),
        ggml_element_size(expert_gate_weights), moe_gate_ne, moe_gate_nb, 2,
        ACL_FORMAT_ND);
    aclnn_grouped_matmul(ctx, acl_expand_input_tensor,
                         acl_expert_gate_weights_t_tensor,
                         acl_expert_tokens_int64_tensor, acl_moe_gate_tensor);
    // int64_t tmp1_out_ne[2];
    // size_t tmp1_out_nb[2];
    // tmp1_out_ne[0] = moe_gate_ne[0];
    // tmp1_out_ne[1] = moe_gate_ne[1];
    // tmp1_out_nb[0] = moe_gate_nb[0];
    // tmp1_out_nb[1] = moe_gate_nb[1];
    // ggml_cann_pool_alloc
    // tmp1_out_allocator(ctx.pool(),tmp1_out_ne[0]*tmp1_out_ne[1]*ggml_element_size(expert_gate_weights));
    // void* tmp1_out_buffer = tmp1_out_allocator.get();
    // aclTensor* acl_tmp1_out_tensor = ggml_cann_create_tensor(tmp1_out_buffer,
    // ggml_cann_type_mapping(expert_gate_weights->type),
    // ggml_element_size(expert_gate_weights), tmp1_out_ne, tmp1_out_nb, 2,
    // ACL_FORMAT_ND);
    // aclnn_index_select(ctx,acl_moe_gate_tensor,acl_tmp1_out_tensor,acl_expand_row_idx_tensor,0);
    // print_acltensor(tmp1_out_buffer,tmp1_out_ne,ggml_cann_type_mapping(expert_gate_weights->type),"acl_tmp1_out_tensor",2,nullptr,&ctx);

    int64_t moe_gate_silu_ne[2] = {k_dim, num_length};
    size_t moe_gate_silu_nb[2];
    moe_gate_silu_nb[0] = ggml_element_size(expert_gate_weights);
    moe_gate_silu_nb[1] = moe_gate_silu_nb[0] * moe_gate_ne[0];
    ggml_cann_pool_alloc moe_gate_silu_allocator(
        ctx.pool(), moe_gate_silu_ne[0] * moe_gate_silu_ne[1] *
                        ggml_element_size(expert_gate_weights));
    void* moe_gate_silu_buffer = moe_gate_silu_allocator.get();
    aclTensor* acl_moe_gate_silu_tensor = ggml_cann_create_tensor(
        moe_gate_silu_buffer, ggml_cann_type_mapping(expert_gate_weights->type),
        ggml_element_size(expert_gate_weights), moe_gate_silu_ne,
        moe_gate_silu_nb, 2, ACL_FORMAT_ND);
    aclnn_silu(ctx, acl_moe_gate_tensor, acl_moe_gate_silu_tensor);
    // int64_t tmp2_out_ne[2];
    // size_t tmp2_out_nb[2];
    // tmp2_out_ne[0] = moe_gate_silu_ne[0];
    // tmp2_out_ne[1] = moe_gate_silu_ne[1];
    // tmp2_out_nb[0] = moe_gate_silu_nb[0];
    // tmp2_out_nb[1] = moe_gate_silu_nb[1];
    // ggml_cann_pool_alloc
    // tmp2_out_allocator(ctx.pool(),tmp2_out_ne[0]*tmp2_out_ne[1]*ggml_element_size(expert_gate_weights));
    // void* tmp2_out_buffer = tmp2_out_allocator.get();
    // aclTensor* acl_tmp2_out_tensor = ggml_cann_create_tensor(tmp2_out_buffer,
    // ggml_cann_type_mapping(expert_gate_weights->type),
    // ggml_element_size(expert_gate_weights), tmp2_out_ne, tmp2_out_nb, 2,
    // ACL_FORMAT_ND);
    // aclnn_index_select(ctx,acl_moe_gate_silu_tensor,acl_tmp2_out_tensor,acl_expand_row_idx_tensor,0);
    // print_acltensor(tmp2_out_buffer,tmp2_out_ne,ggml_cann_type_mapping(expert_gate_weights->type),"acl_tmp2_out_tensor",2,nullptr,&ctx);
    // FIX ME BATCH SIZE
    int64_t moe_pair_ne[2] = {k_dim, num_length};
    size_t moe_pair_nb[2];
    moe_pair_nb[0] = ggml_element_size(expert_up_weights);
    moe_pair_nb[1] = moe_pair_nb[0] * moe_pair_ne[0];
    // moe_pair_nb[2] = moe_pair_nb[1] * moe_pair_ne[1];
    ggml_cann_pool_alloc moe_pair_allocator(
        ctx.pool(),
        moe_pair_ne[0] * moe_pair_ne[1] * ggml_element_size(expert_up_weights));
    void* moe_pair_buffer = moe_pair_allocator.get();
    aclTensor* acl_moe_pair_tensor = ggml_cann_create_tensor(
        moe_pair_buffer, ggml_cann_type_mapping(expert_up_weights->type),
        ggml_element_size(expert_up_weights), moe_pair_ne, moe_pair_nb, 2,
        ACL_FORMAT_ND);
    // print_acltensor(moe_up_buffer,moe_up_ne,ggml_cann_type_mapping(expert_up_weights->type),"moe_up_tensor",2);
    // print_acltensor(moe_gate_silu_buffer,moe_gate_silu_ne,ggml_cann_type_mapping(expert_up_weights->type),"moe_gate_silu_tensor",2);
    aclnn_mul(ctx, acl_moe_up_tensor, acl_moe_gate_silu_tensor,
              acl_moe_pair_tensor);
    // print_acltensor(moe_pair_buffer,moe_pair_ne,ggml_cann_type_mapping(expert_up_weights->type),"moe_pair_tensor",2);
    // int64_t tmp3_out_ne[2];
    // size_t tmp3_out_nb[2];
    // tmp3_out_ne[0] = moe_pair_ne[0];
    // tmp3_out_ne[1] = moe_pair_ne[1];
    // tmp3_out_nb[0] = moe_pair_nb[0];
    // tmp3_out_nb[1] = moe_pair_nb[1];
    // ggml_cann_pool_alloc
    // tmp3_out_allocator(ctx.pool(),tmp3_out_ne[0]*tmp3_out_ne[1]*ggml_element_size(expert_up_weights));
    // void* tmp3_out_buffer = tmp3_out_allocator.get();
    // aclTensor* acl_tmp3_out_tensor = ggml_cann_create_tensor(tmp3_out_buffer,
    // ggml_cann_type_mapping(expert_up_weights->type),
    // ggml_element_size(expert_up_weights), tmp3_out_ne, tmp3_out_nb, 2,
    // ACL_FORMAT_ND);
    // aclnn_index_select(ctx,acl_moe_pair_tensor,acl_tmp3_out_tensor,acl_expand_row_idx_tensor,0);
    // print_acltensor(tmp3_out_buffer,tmp3_out_ne,ggml_cann_type_mapping(expert_up_weights->type),"acl_tmp3_out_tensor",2,nullptr,&ctx);

    int64_t down_permute_ne[3] = {expert_down_weights->ne[0],
                                  expert_down_weights->ne[1],
                                  expert_down_weights->ne[2]};
    size_t down_permute_nb[3];
    down_permute_nb[0] = ggml_element_size(expert_down_weights);
    down_permute_nb[1] = down_permute_nb[0] * down_permute_ne[0];
    down_permute_nb[2] = down_permute_nb[1] * down_permute_ne[1];
    aclTensor* acl_expert_down_weights_t_tensor =
        ggml_cann_create_tensor_transpose(expert_down_weights, down_permute_ne,
                                          down_permute_nb, 3);
    int64_t moe_down_ne[2] = {hidden_dim, num_length};
    size_t moe_down_nb[2];
    moe_down_nb[0] = ggml_element_size(expert_down_weights);
    moe_down_nb[1] = moe_down_nb[0] * moe_down_ne[0];
    // moe_down_nb[2] = moe_down_nb[1] * moe_down_ne[1];
    ggml_cann_pool_alloc moe_down_allocator(
        ctx.pool(), moe_down_ne[0] * moe_down_ne[1] *
                        ggml_element_size(expert_down_weights));
    void* moe_down_buffer = moe_down_allocator.get();
    aclTensor* acl_moe_down_tensor = ggml_cann_create_tensor(
        moe_down_buffer, ggml_cann_type_mapping(expert_down_weights->type),
        ggml_element_size(expert_down_weights), moe_down_ne, moe_down_nb, 2,
        ACL_FORMAT_ND);
    aclnn_grouped_matmul(ctx, acl_moe_pair_tensor,
                         acl_expert_down_weights_t_tensor,
                         acl_expert_tokens_int64_tensor, acl_moe_down_tensor);

    int64_t moe_down_f32_ne[2] = {hidden_dim, num_length};
    size_t moe_down_f32_nb[2];
    moe_down_f32_nb[0] = ggml_type_size(GGML_TYPE_F32);
    moe_down_f32_nb[1] = moe_down_f32_nb[0] * moe_down_f32_ne[0];
    ggml_cann_pool_alloc moe_down_f32_allocator(
        ctx.pool(), moe_down_f32_ne[0] * moe_down_f32_ne[1] *
                        ggml_type_size(GGML_TYPE_F32));
    void* moe_down_f32_buffer = moe_down_f32_allocator.get();
    aclTensor* acl_moe_down_f32_tensor = ggml_cann_create_tensor(
        moe_down_f32_buffer, ggml_cann_type_mapping(GGML_TYPE_F32),
        ggml_type_size(GGML_TYPE_F32), moe_down_f32_ne, moe_down_f32_nb, 2,
        ACL_FORMAT_ND);
    aclnn_cast(ctx, acl_moe_down_tensor, acl_moe_down_f32_tensor, ACL_FLOAT);
    // GGML_ASSERT(topk_weight->type==GGML_TYPE_F32);
    // print_acltensor(moe_down_buffer,moe_down_ne,ggml_cann_type_mapping(expert_down_weights->type),"moe_down_tensor",2,nullptr,&ctx);

    int64_t moe_down_padding_ne[2] = {hidden_dim, seq_len * topk};
    size_t moe_down_padding_nb[2];
    // moe_down_padding_nb[0] = ggml_element_size(expert_down_weights);
    moe_down_padding_nb[0] = ggml_type_size(GGML_TYPE_F32);
    moe_down_padding_nb[1] = moe_down_padding_nb[0] * moe_down_padding_ne[0];
    ggml_cann_pool_alloc moe_down_padding_allocator(
        ctx.pool(), moe_down_padding_ne[0] * moe_down_padding_ne[1] *
                        ggml_type_size(GGML_TYPE_F32));
    void* moe_down_padding_buffer = moe_down_padding_allocator.get();
    aclTensor* acl_moe_down_padding_tensor = ggml_cann_create_tensor(
        moe_down_padding_buffer, ggml_cann_type_mapping(GGML_TYPE_F32),
        ggml_type_size(GGML_TYPE_F32), moe_down_padding_ne, moe_down_padding_nb,
        2, ACL_FORMAT_ND);
    aclnn_fill_scalar(ctx, 0.0f, acl_moe_down_padding_tensor);
    aclrtMemcpyAsync(
        (char*)moe_down_padding_buffer +
            hidden_dim * range_start * ggml_type_size(GGML_TYPE_F32),
        range_size * ggml_type_size(GGML_TYPE_F32) * hidden_dim,
        moe_down_f32_buffer,
        range_size * ggml_type_size(GGML_TYPE_F32) * hidden_dim,
        ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream());

    // int64_t tmp4_out_ne[2];
    // size_t tmp4_out_nb[2];
    // tmp4_out_ne[0] = moe_down_ne[0];
    // tmp4_out_ne[1] = moe_down_ne[1];
    // tmp4_out_nb[0] = moe_down_nb[0];
    // tmp4_out_nb[1] = moe_down_nb[1];
    // ggml_cann_pool_alloc
    // tmp4_out_allocator(ctx.pool(),tmp4_out_ne[0]*tmp4_out_ne[1]*ggml_element_size(expert_down_weights));
    // void* tmp4_out_buffer = tmp4_out_allocator.get();
    // aclTensor* acl_tmp4_out_tensor = ggml_cann_create_tensor(tmp4_out_buffer,
    // ggml_cann_type_mapping(expert_down_weights->type),
    // ggml_element_size(expert_down_weights), tmp4_out_ne, tmp4_out_nb, 2,
    // ACL_FORMAT_ND);
    // aclnn_index_select(ctx,acl_moe_down_tensor,acl_tmp4_out_tensor,acl_expand_row_idx_tensor,0);
    // print_acltensor(tmp4_out_buffer,tmp4_out_ne,ggml_cann_type_mapping(expert_down_weights->type),"acl_tmp4_out_tensor",2,nullptr,&ctx);

    int64_t dst_ne[2] = {hidden_dim, seq_len};
    size_t dst_nb[2];
    dst_nb[0] = ggml_element_size(dst);
    dst_nb[1] = dst_nb[0] * dst_ne[0];
    aclTensor* acl_dst_tensor =
        ggml_cann_create_tensor(dst, dst_ne, dst_nb, 2, ACL_FORMAT_ND);
    int64_t moe_weight_ne[2] = {topk, num_rows};
    size_t moe_weight_nb[2];
    moe_weight_nb[0] = ggml_element_size(topk_weight);
    moe_weight_nb[1] = moe_weight_nb[0] * moe_weight_ne[0];

    // print_acltensor(moe_down_buffer,moe_down_ne,ggml_cann_type_mapping(expert_up_weights->type),"moe_down_tensor",2);
    // aclTensor* acl_moe_weight_tensor =
    // ggml_cann_create_tensor(topk_weight,topk_weight->ne,topk_weight->nb,3,ACL_FORMAT_ND);
    aclTensor* acl_moe_weight_tensor = ggml_cann_create_tensor(
        topk_weight, moe_weight_ne, moe_weight_nb, 2, ACL_FORMAT_ND);
    // print_acltensor(topk_weight->data,moe_weight_ne,ggml_cann_type_mapping(topk_weight->type),"topk_weight_tensor",2,nullptr,&ctx);
    // int64_t moe_down_reshape_ne[3] = {hidden_dim,topk,seq_len};
    // size_t moe_down_reshape_nb[3];
    // moe_down_reshape_nb[0] = ggml_element_size(expert_down_weights);
    // moe_down_reshape_nb[1] = moe_down_reshape_nb[0] * moe_down_reshape_ne[0];
    // moe_down_reshape_nb[2] = moe_down_reshape_nb[1] * moe_down_reshape_ne[1];
    // aclTensor* acl_moe_down_reshape_tensor =
    // ggml_cann_create_tensor(tmp4_out_buffer,
    // ggml_cann_type_mapping(expert_down_weights->type),
    // ggml_element_size(expert_down_weights), moe_down_reshape_ne,
    // moe_down_reshape_nb, 3, ACL_FORMAT_ND);
    // print_acltensor(tmp4_out_buffer,moe_down_reshape_ne,ggml_cann_type_mapping(expert_down_weights->type),"acl_moe_down_reshape_tensor",3,nullptr,&ctx);
    // int64_t dst_ne[]= {hidden_dim,seq_len};
    // size_t dst_nb[]= {ggml_element_size(dst), ggml_element_size(dst) *
    // dst_ne[0]}; aclTensor* acl_dst_tensor =
    // ggml_cann_create_tensor(dst,dst_ne,dst_nb,2,ACL_FORMAT_ND);
    // print_acltensor(moe_down_buffer,moe_down_ne,ggml_cann_type_mapping(expert_down_weights->type),"moe_down_tensor",2,moe_down_nb,&ctx);
    // print_acltensor(topk_weight->data,moe_weight_ne,ggml_cann_type_mapping(topk_weight->type),"moe_weight_tensor",2,moe_weight_nb,&ctx);
    // print_acltensor(ids->data,expert_idx_ne,ggml_cann_type_mapping(ids->type),"expert_idx_tensor",2,nullptr,&ctx);
    // print_acltensor(expand_row_idx_buffer,expand_row_idx_ne,ggml_cann_type_mapping(row_idx->type),"expand_row_idx_tensor",1,nullptr,&ctx);
    // int64_t f_moe_down_ne[2] = {hidden_dim,seq_len*topk};
    // size_t f_moe_down_nb[2];
    // f_moe_down_nb[0] = ggml_element_size(expert_down_weights);
    // f_moe_down_nb[1] = f_moe_down_nb[0] * f_moe_down_ne[0];
    // aclTensor* acl_f_moe_down_tensor =
    // ggml_cann_create_tensor(moe_down_buffer,
    // ggml_cann_type_mapping(expert_down_weights->type),
    // ggml_element_size(expert_down_weights), f_moe_down_ne, f_moe_down_nb, 2,
    // ACL_FORMAT_ND);
    int64_t f_moe_weight_ne[2] = {topk, seq_len};
    size_t f_moe_weight_nb[2];
    f_moe_weight_nb[0] = ggml_element_size(topk_weight);
    f_moe_weight_nb[1] = f_moe_weight_nb[0] * f_moe_weight_ne[0];
    aclTensor* acl_f_moe_weight_tensor = ggml_cann_create_tensor(
        topk_weight, f_moe_weight_ne, f_moe_weight_nb, 2, ACL_FORMAT_ND);
    int64_t f_expand_row_idx_ne[1] = {num_rows * topk};
    size_t f_expand_row_idx_nb[1] = {ggml_element_size(row_idx)};
    aclTensor* acl_f_expand_row_idx_tensor = ggml_cann_create_tensor(
        expand_row_idx_buffer, ggml_cann_type_mapping(row_idx->type),
        ggml_element_size(row_idx), f_expand_row_idx_ne, f_expand_row_idx_nb, 1,
        ACL_FORMAT_ND);
    int64_t f_expert_idx_ne[2] = {topk, seq_len};
    size_t f_expert_idx_nb[2] = {ggml_element_size(ids),
                                 ggml_element_size(ids) * f_expert_idx_ne[0]};
    aclTensor* acl_f_expert_idx_tensor = ggml_cann_create_tensor(
        ids, f_expert_idx_ne, f_expert_idx_nb, 2, ACL_FORMAT_ND);

    aclnn_moe_finalize_routing(
        ctx, acl_f_moe_weight_tensor, acl_moe_down_padding_tensor,
        acl_f_expand_row_idx_tensor, acl_f_expert_idx_tensor, acl_f_dst_tensor);
    // aclnn_moe_finalize_routing(ctx,acl_moe_weight_tensor,acl_moe_down_tensor,acl_expand_row_idx_tensor,acl_expert_idx_tensor,acl_dst_tensor);
    // aclrtSynchronizeStream(ctx.stream());
    // print_acltensor(dst->data,dst_ne,ggml_cann_type_mapping(dst->type),"dst_tensor",2,dst_nb,&ctx);

    // GGML_ASSERT(1==2);
}

void ggml_cann_sum_rows(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src);

    GGML_ASSERT(dst->ne[0] == 1);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    int64_t reduce_dims_host[] = {3};
    aclIntArray* reduce_dims = aclCreateIntArray(reduce_dims_host, 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnReduceSumGetWorkspaceSize(
        acl_src, reduce_dims, true, ggml_cann_type_mapping(src->type), acl_dst,
        &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnReduceSum(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_upsample_nearest2d(ggml_backend_cann_context& ctx,
                                  ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    aclTensor* acl_src =
        ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    aclTensor* acl_dst =
        ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    std::vector<int64_t> output_size{dst->ne[1], dst->ne[0]};
    auto output_size_array = aclCreateIntArray(output_size.data(), 2);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnUpsampleNearest2dGetWorkspaceSize(
        acl_src, output_size_array, acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnUpsampleNearest2d(workspaceAddr, workspaceSize, executor,
                                     ctx.stream()));

    ACL_CHECK(aclDestroyIntArray(output_size_array));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

/**
 * @brief Pads a tensor with a specified value along each dimension.
 *
 * This function performs padding of the source tensor `acl_src` and stores the
 * result in the destination tensor `acl_dst`. The padding values for each
 * dimension are specified in the `paddings` array.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor to be padded.
 * @param acl_dst The destination tensor where the padded result will be stored.
 * @param paddings An array specifying the padding values for each dimension.
 * The size of the array should be twice the number of dimensions of the tensor.
 * @param value The value to be used for padding. The default value is 0.0.
 */
static void aclnn_pad(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                      aclTensor* acl_dst, int64_t* paddings,
                      float value = 0.0f) {
    aclIntArray* acl_pad = aclCreateIntArray(paddings, GGML_MAX_DIMS * 2);
    aclScalar* acl_value = aclCreateScalar(&value, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnConstantPadNdGetWorkspaceSize(
        acl_src, acl_pad, acl_value, acl_dst, &workspaceSize, &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnConstantPadNd(workspaceAddr, workspaceSize, executor,
                                 ctx.stream()));

    ACL_CHECK(aclDestroyIntArray(acl_pad));
    ACL_CHECK(aclDestroyScalar(acl_value));
}

void ggml_cann_pad(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    // padding: value in the array means how much distance will be padding.
    // the position of elements in the array means which dirction to padding,
    // each position means: [dim0.front, dim0.behind, dim1.front, dim1.behind,
    //                       dim2.front, dim2.behind, dim3.front, dim3.behind]
    int64_t paddings[] = {
        0, dst->ne[0] - src->ne[0], 0, dst->ne[1] - src->ne[1],
        0, dst->ne[2] - src->ne[2], 0, dst->ne[3] - src->ne[3]};
    aclnn_pad(ctx, acl_src, acl_dst, paddings);

    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyTensor(acl_src));
}

/**
 * @brief Performs 2D average pooling on the input tensor and stores the result
 * in the destination tensor.
 *
 * This function performs average pooling on the source tensor and stores the
 * result in the destination tensor. The pooling parameters (kernel size,
 * strides, padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
static void ggml_cann_avg_pool2d(ggml_backend_cann_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src =
        ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    aclTensor* acl_dst =
        ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    const int32_t* opts = (const int32_t*)dst->op_params;
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    std::vector<int64_t> kernel_dims = {k1, k0};
    std::vector<int64_t> stride_dims = {s1, s0};
    std::vector<int64_t> padding_avg_dims = {p1, p0};  // (padH, padW)

    auto* kernel_size = aclCreateIntArray(kernel_dims.data(), 2);
    auto* strides = aclCreateIntArray(stride_dims.data(), 2);
    auto* paddings_avg = aclCreateIntArray(padding_avg_dims.data(), 2);

    bool ceil_mode = false;
    bool count_include_pad = true;
    int64_t divisor_override = 0;
    int8_t cube_math_type = 0;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnAvgPool2dGetWorkspaceSize(
        acl_src, kernel_size, strides, paddings_avg, ceil_mode,
        count_include_pad, divisor_override, cube_math_type, acl_dst,
        &workspaceSize, &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(
        aclnnAvgPool2d(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyIntArray(kernel_size));
    ACL_CHECK(aclDestroyIntArray(strides));
    ACL_CHECK(aclDestroyIntArray(paddings_avg));
}

/**
 * @brief Performs 2D max pooling on the input tensor and stores the result in
 * the destination tensor.
 *
 * This function performs max pooling on the source tensor and stores the result
 * in the destination tensor. The pooling parameters (kernel size, strides,
 * padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
static void ggml_cann_max_pool2d(ggml_backend_cann_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src =
        ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    aclTensor* acl_dst =
        ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    const int32_t* opts = (const int32_t*)dst->op_params;
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    int64_t temp_ne[] = {src->ne[0] + p0 * 2, src->ne[1] + p1 * 2, src->ne[2],
                         src->ne[3]};
    size_t temp_nb[GGML_MAX_DIMS];

    temp_nb[0] = ggml_element_size(src);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        temp_nb[i] = temp_nb[i - 1] * temp_ne[i - 1];
    }

    ggml_cann_pool_alloc temp_buffer_allocator(
        ctx.pool(), ggml_nbytes(src) + p0 * 2 + p1 * 2 * src->nb[1]);
    void* buffer = temp_buffer_allocator.get();
    aclTensor* tmp_tensor = ggml_cann_create_tensor(
        buffer, ACL_FLOAT, ggml_element_size(src), temp_ne, temp_nb,
        GGML_MAX_DIMS, ACL_FORMAT_NCHW);

    // pad: see padding in ggml_cann_pad()
    int64_t paddings[] = {p0, p0, p1, p1, 0, 0, 0, 0};
    float value = -FLT_MAX;
    aclnn_pad(ctx, acl_src, tmp_tensor, paddings, value);

    // max_pool
    std::vector<int64_t> kernel_dims = {k1, k0};
    std::vector<int64_t> stride_dims = {s1, s0};
    // padding_max_dims: [dim0_start, dim0_end, dim1_start, dim1_end]
    std::vector<int64_t> padding_max_dims = {0, 0, 0, 0};
    std::vector<int64_t> dilation_size = {1, 1};
    auto* kernel_size = aclCreateIntArray(kernel_dims.data(), 2);
    auto* strides = aclCreateIntArray(stride_dims.data(), 2);
    auto* paddings_max = aclCreateIntArray(padding_max_dims.data(), 4);
    auto* dilations = aclCreateIntArray(dilation_size.data(), 2);

    bool ceil_mode = false;
    int64_t auto_pads = 0;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnMaxPoolGetWorkspaceSize(
        tmp_tensor, kernel_size, strides, auto_pads, paddings_max, dilations,
        ceil_mode, acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnMaxPool(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyTensor(tmp_tensor));
    ACL_CHECK(aclDestroyIntArray(kernel_size));
    ACL_CHECK(aclDestroyIntArray(strides));
    ACL_CHECK(aclDestroyIntArray(paddings_max));
    ACL_CHECK(aclDestroyIntArray(dilations));
}

void ggml_cann_pool2d(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    const int32_t* opts = (const int32_t*)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    switch (op) {
        case GGML_OP_POOL_AVG:
            ggml_cann_avg_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_MAX:
            ggml_cann_max_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_COUNT:
            GGML_ABORT("fatal error");
            break;
    }
}

/**
 * @brief Copies data from the source tensor to the destination tensor.
 *
 * This function copies data from the source tensor `acl_src` to the destination
 * tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor from which data will be copied.
 * @param acl_dst The destination tensor where the data will be copied to.
 */
static void cann_copy(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                      aclTensor* acl_dst) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceCopyGetWorkspaceSize(acl_dst, acl_src, &workspaceSize,
                                               &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, ctx.stream()));
}

void ggml_cann_dup(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    ggml_cann_pool_alloc src_extra_allocator(ctx.pool(), sizeof(ggml_tensor));
    ggml_cann_pool_alloc dst_extra_allocator(ctx.pool(), sizeof(ggml_tensor));
    src->extra = src_extra_allocator.get();
    dst->extra = dst_extra_allocator.get();
    ACL_CHECK(aclrtMemcpyAsync(src->extra, sizeof(ggml_tensor), src,
                               sizeof(ggml_tensor), ACL_MEMCPY_HOST_TO_DEVICE,
                               ctx.stream()));
    ACL_CHECK(aclrtMemcpyAsync(dst->extra, sizeof(ggml_tensor), dst,
                               sizeof(ggml_tensor), ACL_MEMCPY_HOST_TO_DEVICE,
                               ctx.stream()));

    if ((dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32 ||
         dst->type == GGML_TYPE_I32) &&
        ggml_are_same_shape(src, dst)) {
        cann_copy(ctx, acl_src, acl_dst);
        ACL_CHECK(aclDestroyTensor(acl_src));
        ACL_CHECK(aclDestroyTensor(acl_dst));
        return;
    }
    // TODO: simplify
    if (src->type == GGML_TYPE_F16) {
        if (dst->type == GGML_TYPE_Q8_0) {
            aclrtlaunch_ascendc_quantize_f16_q8_0(
                24, ctx.stream(), src->data, dst->data,
                ((ggml_tensor*)src->extra)->ne, ((ggml_tensor*)src->extra)->nb,
                ((ggml_tensor*)dst->extra)->ne);
            return;
        }
        if (dst->type == GGML_TYPE_Q4_0) {
            aclrtlaunch_ascendc_quantize_f16_to_q4_0(
                24, ctx.stream(), src->data, dst->data,
                ((ggml_tensor*)src->extra)->ne, ((ggml_tensor*)src->extra)->nb,
                ((ggml_tensor*)dst->extra)->ne);
            return;
        }
        if (dst->type == GGML_TYPE_F16) {
            if (ggml_are_same_shape(src, dst)) {
                cann_copy(ctx, acl_src, acl_dst);
                ACL_CHECK(aclDestroyTensor(acl_src));
                ACL_CHECK(aclDestroyTensor(acl_dst));
                return;
            }
            if (ggml_is_contiguous(dst)) {
                const size_t src_type_size = ggml_type_size(src->type);
                if (src->nb[0] == src_type_size) {
                    // src0 is contigous on first dimension, copy by rows
                    int64_t rows_num = ggml_nrows(src);

                    aclrtlaunch_ascendc_dup_by_rows_fp16(
                        rows_num, ctx.stream(), src->data, dst->data,
                        ((ggml_tensor*)src->extra)->ne,
                        ((ggml_tensor*)src->extra)->nb,
                        ((ggml_tensor*)dst->extra)->ne,
                        ((ggml_tensor*)dst->extra)->nb);
                    return;
                }
                GGML_ABORT("fatal error");
            }
            GGML_ABORT("fatal error");
        }
        if (dst->type == GGML_TYPE_F32) {
            if (ggml_are_same_shape(src, dst)) {
                cann_copy(ctx, acl_src, acl_dst);
                ACL_CHECK(aclDestroyTensor(acl_src));
                ACL_CHECK(aclDestroyTensor(acl_dst));
                return;
            }
            if (ggml_is_contiguous(dst)) {
                const size_t src_type_size = ggml_type_size(src->type);
                if (src->nb[0] == src_type_size) {
                    // src0 is contigous on first dimension, copy by rows
                    int64_t rows_num = ggml_nrows(src);
                    aclrtlaunch_ascendc_dup_by_rows_fp16_to_fp32(
                        rows_num, ctx.stream(), src->data, dst->data,
                        ((ggml_tensor*)src->extra)->ne,
                        ((ggml_tensor*)src->extra)->nb,
                        ((ggml_tensor*)dst->extra)->ne,
                        ((ggml_tensor*)dst->extra)->nb);
                    return;
                }
                GGML_ABORT("fatal error");
            }
            GGML_ABORT("fatal error");
        }
        // TODO
        GGML_ABORT("fatal error");
    } else if (src->type == GGML_TYPE_F32) {
        // TODO: if (src0->type == dst->type && ne00 == ne0 && nb00 == type_size
        //          && nb0 == type_size)
        if (dst->type == GGML_TYPE_Q8_0) {
            aclrtlaunch_ascendc_quantize_f32_q8_0(
                24, ctx.stream(), src->data, dst->data,
                ((ggml_tensor*)src->extra)->ne, ((ggml_tensor*)src->extra)->nb,
                ((ggml_tensor*)dst->extra)->ne);
            return;
        }
        if (dst->type == GGML_TYPE_Q4_0) {
            aclrtlaunch_ascendc_quantize_f32_to_q4_0(
                24, ctx.stream(), src->data, dst->data,
                ((ggml_tensor*)src->extra)->ne, ((ggml_tensor*)src->extra)->nb,
                ((ggml_tensor*)dst->extra)->ne);
            return;
        }
        if (dst->type == GGML_TYPE_F32) {
            if (ggml_are_same_shape(src, dst)) {
                cann_copy(ctx, acl_src, acl_dst);
                ACL_CHECK(aclDestroyTensor(acl_src));
                ACL_CHECK(aclDestroyTensor(acl_dst));
                return;
            }
            if (ggml_is_contiguous(dst)) {
                const size_t src_type_size = ggml_type_size(src->type);
                if (src->nb[0] == src_type_size) {
                    // src0 is contigous on first dimension, copy by rows
                    int64_t rows_num = ggml_nrows(src);
                    aclrtlaunch_ascendc_dup_by_rows_fp32(
                        rows_num, ctx.stream(), src->data, dst->data,
                        ((ggml_tensor*)src->extra)->ne,
                        ((ggml_tensor*)src->extra)->nb,
                        ((ggml_tensor*)dst->extra)->ne,
                        ((ggml_tensor*)dst->extra)->nb);
                    return;
                }
                GGML_ABORT("fatal error");
            } else {
                // TODO: dst not contiguous
                GGML_ABORT("fatal error");
            }
        }
        if (dst->type == GGML_TYPE_F16) {
            if (ggml_are_same_shape(src, dst)) {
                cann_copy(ctx, acl_src, acl_dst);
                ACL_CHECK(aclDestroyTensor(acl_src));
                ACL_CHECK(aclDestroyTensor(acl_dst));
                return;
            }
            if (ggml_is_contiguous(dst)) {
                const size_t src_type_size = ggml_type_size(src->type);
                if (src->nb[0] == src_type_size) {
                    // src0 is contigous on first dimension, copy by rows
                    int64_t rows_num = ggml_nrows(src);
                    aclrtlaunch_ascendc_dup_by_rows_fp32_to_fp16(
                        rows_num, ctx.stream(), src->data, dst->data,
                        ((ggml_tensor*)src->extra)->ne,
                        ((ggml_tensor*)src->extra)->nb,
                        ((ggml_tensor*)dst->extra)->ne,
                        ((ggml_tensor*)dst->extra)->nb);
                    return;
                }
                GGML_ABORT("fatal error");
            }
        }
        //  if(dst->type == GGML_TYPE_I32){
        //     if (ggml_are_same_shape(src, dst)) {
        //         cann_copy(ctx, acl_src, acl_dst);
        //         ACL_CHECK(aclDestroyTensor(acl_src));
        //         ACL_CHECK(aclDestroyTensor(acl_dst));
        //         return;
        //     }
        //     GGML_ABORT("fatal error");

        //  }
        // TODO
        GGML_ABORT("fatal error");
    } else {
        if (ggml_are_same_shape(src, dst)) {
            cann_copy(ctx, acl_src, acl_dst);
            ACL_CHECK(aclDestroyTensor(acl_src));
            ACL_CHECK(aclDestroyTensor(acl_dst));
            return;
        }
        GGML_ABORT("fatal error");
    }
}

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnRmsNormGetWorkspaceSize(const aclTensor* x,
                                         const aclTensor* gamma, double epsilon,
                                         const aclTensor* yOut,
                                         const aclTensor* rstdOout,
                                         uint64_t* workspaceSize,
                                         aclOpExecutor** executor);
aclnnStatus aclnnRmsNorm(void* workspace, uint64_t workspaceSize,
                         aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

/**
 * @brief Creates an ACL tensor initialized with zeros using a provided buffer.
 *
 * This function initializes a tensor with zeros using the specified buffer and
 * tensor parameters.
 *
 * @param ctx The context for the CANN backend operations.
 * @param buffer The buffer to be used for the tensor data.
 * @param n_bytes The size of the buffer in bytes.
 * @param ne An array specifying the extents (sizes) of each dimension of the
 * tensor.
 * @param dims The number of dimensions of the tensor.
 * @param type The data type of the tensor.
 * @param type_size The size of each element in the tensor data type.
 * @return An ACL tensor initialized with zeros.
 */
static aclTensor* aclnn_zero(ggml_backend_cann_context& ctx, void* buffer,
                             size_t n_bytes, int64_t* ne, int64_t dims,
                             aclDataType type, size_t type_size) {
    size_t nb[GGML_MAX_DIMS];
    nb[0] = type_size;
    for (int i = 1; i < dims; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }

    ACL_CHECK(aclrtMemsetAsync(buffer, n_bytes, 0, n_bytes, ctx.stream()));
    aclTensor* zero =
        ggml_cann_create_tensor(buffer, type, type_size, ne, nb, dims);
    return zero;
}

/**
 * @brief Creates an ACL tensor initialized with value using a provided buffer.
 *
 * This function initializes a tensor with value using the specified buffer and
 * tensor parameters.
 *
 * @param ctx The context for the CANN backend operations.
 * @param buffer The buffer to be used for the tensor data.
 * @param n_bytes The size of the buffer in bytes.
 * @param ne An array specifying the extents (sizes) of each dimension of the
 * tensor.
 * @param dims The number of dimensions of the tensor.
 * @param type The data type of the tensor.
 * @param type_size The size of each element in the tensor data type.
 * @param value The value to be used for initializing the tensor (default
 * is 1.0).
 * @return An ACL tensor initialized with value.
 */
static aclTensor* aclnn_values(ggml_backend_cann_context& ctx, void* buffer,
                               size_t n_bytes, int64_t* ne, int64_t dims,
                               aclDataType type, size_t type_size,
                               float value = 1.0f) {
    aclTensor* acl_tensor =
        aclnn_zero(ctx, buffer, n_bytes, ne, dims, type, type_size);
    float alpha_host = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alpha_host, aclDataType::ACL_FLOAT);
    aclScalar* other = aclCreateScalar(&value, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceAddsGetWorkspaceSize(acl_tensor, other, alpha,
                                               &workspaceSize, &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(
        aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, ctx.stream()));

    return acl_tensor;
}

void ggml_cann_rms_norm(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_ASSERT(eps > 0.0f);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    size_t one_tensor_n_bytes = src->ne[0] * ggml_element_size(src);
    ggml_cann_pool_alloc one_tensor_allocator(ctx.pool(), one_tensor_n_bytes);

    aclTensor* acl_gamma = aclnn_values(
        ctx, one_tensor_allocator.get(), one_tensor_n_bytes, src->ne, 1,
        ggml_cann_type_mapping(src->type), ggml_element_size(src));

    size_t zero_tensor_n_bytes =
        src->ne[1] * src->ne[2] * src->ne[3] * ggml_element_size(src);
    ggml_cann_pool_alloc zero_tensor_allocator(ctx.pool(), zero_tensor_n_bytes);
    aclTensor* acl_rstd =
        aclnn_zero(ctx, zero_tensor_allocator.get(), zero_tensor_n_bytes,
                   src->ne, GGML_MAX_DIMS, ggml_cann_type_mapping(src->type),
                   ggml_element_size(src));

    ACL_CHECK(aclnnRmsNormGetWorkspaceSize(
        acl_src, acl_gamma, eps, acl_dst, acl_rstd, &workspaceSize, &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnRmsNorm(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyTensor(acl_gamma));
    ACL_CHECK(aclDestroyTensor(acl_rstd));
}

void ggml_cann_rms_norm_fused(ggml_backend_cann_context& ctx,
                              ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    ggml_tensor* gamma = dst->src[1];

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_gamma =
        ggml_cann_create_tensor(gamma, gamma->ne, gamma->nb, 1);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_ASSERT(eps > 0.0f);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    size_t zero_tensor_n_bytes =
        src->ne[1] * src->ne[2] * src->ne[3] * ggml_element_size(src);
    ggml_cann_pool_alloc zero_tensor_allocator(ctx.pool(), zero_tensor_n_bytes);
    aclTensor* acl_rstd =
        aclnn_zero(ctx, zero_tensor_allocator.get(), zero_tensor_n_bytes,
                   src->ne, GGML_MAX_DIMS, ggml_cann_type_mapping(src->type),
                   ggml_element_size(src));

    ACL_CHECK(aclnnRmsNormGetWorkspaceSize(
        acl_src, acl_gamma, eps, acl_dst, acl_rstd, &workspaceSize, &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnRmsNorm(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyTensor(acl_gamma));
    ACL_CHECK(aclDestroyTensor(acl_rstd));
}
// TODO: performace is low.
void ggml_cann_diag_mask(ggml_backend_cann_context& ctx, ggml_tensor* dst,
                         float value) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    const int n_past = ((int32_t*)dst->op_params)[0];

    size_t one_tensor_n_bytes = src->ne[0] * src->ne[1] * src->ne[2] *
                                src->ne[3] * ggml_element_size(src);
    ggml_cann_pool_alloc one_tensor_allocator(ctx.pool(), one_tensor_n_bytes);

    aclTensor* mask_tensor =
        aclnn_values(ctx, one_tensor_allocator.get(), one_tensor_n_bytes,
                     src->ne, GGML_MAX_DIMS, ggml_cann_type_mapping(src->type),
                     ggml_element_size(src), value);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceTriuGetWorkspaceSize(mask_tensor, n_past + 1,
                                               &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnInplaceTriu(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclnnTrilGetWorkspaceSize(acl_src, n_past + 1, acl_dst,
                                        &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnTril(workspaceAddr, workspaceSize, executor, ctx.stream()));

    aclScalar* alpha = nullptr;
    float alphaValue = 1.0f;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

    ACL_CHECK(aclnnInplaceAddGetWorkspaceSize(acl_dst, mask_tensor, alpha,
                                              &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(
        aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyScalar(alpha));
    ACL_CHECK(aclDestroyTensor(mask_tensor));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

/**
 * @brief Permutes the dimensions of a tensor according to a specified order.
 *
 * This function permutes the dimensions of the source tensor `acl_src`
 * according to the order specified in the `new_dim` array and stores the result
 * in the destination tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose dimensions will be permuted.
 * @param acl_dst The destination tensor where the permuted result will be
 * stored.
 * @param new_dim An array specifying the new order of dimensions for the
 * tensor.
 * @param dims The number of dimensions in the tensor.
 */
static void aclnn_permute(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                          aclTensor* acl_dst, int64_t* new_dim, uint64_t dims) {
    aclIntArray* acl_dims = aclCreateIntArray(new_dim, dims);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnPermuteGetWorkspaceSize(acl_src, acl_dims, acl_dst,
                                           &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnPermute(workspaceAddr, workspaceSize, executor, ctx.stream()));

    //  ACL_CHECK(aclDestroyIntArray(acl_dims));
}

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnIm2colGetWorkspaceSize(const aclTensor* self,
                                        const aclIntArray* kernelSize,
                                        const aclIntArray* dilation,
                                        const aclIntArray* padding,
                                        const aclIntArray* stride,
                                        aclTensor* out, uint64_t* workspaceSize,
                                        aclOpExecutor** executor);
aclnnStatus aclnnIm2col(void* workspace, uint64_t workspaceSize,
                        aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

static void ggml_cann_im2col_2d_post_process(ggml_backend_cann_context& ctx,
                                             ggml_tensor* dst,
                                             ggml_tensor* src1,
                                             aclTensor* tmp_cast_tensor,
                                             aclTensor* tmp_im2col_tensor) {
    // Permute: [N, IC * KH * KW, OW * OH] -> [N, OW * OH, IC * KH * KW]
    int64_t dst_ne[] = {dst->ne[0], dst->ne[1] * dst->ne[2], dst->ne[3]};
    size_t dst_nb[] = {dst->nb[0], dst->nb[1], dst->nb[3]};
    aclTensor* acl_dst =
        ggml_cann_create_tensor(dst, dst_ne, dst_nb, GGML_MAX_DIMS - 1);

    int64_t permute_dim[] = {0, 2, 1};
    if (src1->type != dst->type) {
        aclnn_permute(ctx, tmp_cast_tensor, acl_dst, permute_dim, 3);
    } else {
        aclnn_permute(ctx, tmp_im2col_tensor, acl_dst, permute_dim, 3);
    }

    // release
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

static void ggml_cann_im2col_1d_post_process(
    ggml_backend_cann_context& ctx, ggml_tensor* dst, ggml_tensor* src1,
    aclTensor* tmp_cast_tensor, aclTensor* tmp_im2col_tensor,
    const std::vector<int64_t>& im2col_op_params) {
    // get params
    const int64_t KH = im2col_op_params[0];
    const int64_t KW = im2col_op_params[1];
    const int64_t IW = im2col_op_params[2];
    const int64_t IC = im2col_op_params[3];
    const int64_t N = im2col_op_params[4];
    const int64_t OH = im2col_op_params[5];
    const int64_t OW = im2col_op_params[6];
    const int64_t s0 = im2col_op_params[7];
    const int64_t p0 = im2col_op_params[8];
    const int64_t d0 = im2col_op_params[9];
    const int64_t n_bytes_factor = im2col_op_params[10];

    // Permute: [N, IC * KH * KW, OW * OH] ->
    // [N, OW * OH * n_bytes_factor, IC * KH * KW]
    aclTensor* tmp_permute_tensor = nullptr;
    ggml_cann_pool_alloc tmp_permute_allocator(ctx.pool());
    tmp_permute_allocator.alloc(ggml_nbytes(dst) * n_bytes_factor);
    void* tmp_permute_buffer = tmp_permute_allocator.get();

    int64_t tmp_permute_ne[] = {IC * KH * KW, OW * OH * n_bytes_factor, N};
    size_t tmp_permute_nb[GGML_MAX_DIMS - 1];
    tmp_permute_nb[0] = ggml_type_size(dst->type);
    for (int i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_permute_nb[i] = tmp_permute_nb[i - 1] * tmp_permute_ne[i - 1];
    }

    tmp_permute_tensor = ggml_cann_create_tensor(
        tmp_permute_buffer, ggml_cann_type_mapping(dst->type),
        ggml_type_size(dst->type), tmp_permute_ne, tmp_permute_nb,
        GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    int64_t permute_dim[] = {0, 2, 1};
    if (src1->type != dst->type) {
        aclnn_permute(ctx, tmp_cast_tensor, tmp_permute_tensor, permute_dim, 3);
    } else {
        aclnn_permute(ctx, tmp_im2col_tensor, tmp_permute_tensor, permute_dim,
                      3);
    }

    // number of times the kernel moves in W dimension
    const int n_step_w = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;
    size_t offset;
    void *cur_dst_buffer = dst->data, *cur_permute_buffer = tmp_permute_buffer;

    // memory copy with offset to restore 1D im2col from 2d
    if (IC > 1) {
        offset = IC * KH * KW * n_step_w * ggml_type_size(dst->type);
        size_t size_cpy = KH * KW * ggml_type_size(dst->type);

        for (int c = 0; c < IC; c++) {
            cur_permute_buffer = (char*)tmp_permute_buffer + offset +
                                 KH * KW * c * ggml_type_size(dst->type);
            cur_dst_buffer = (char*)dst->data +
                             c * KH * KW * n_step_w * ggml_type_size(dst->type);

            for (int i = 0; i < n_step_w; i++) {
                ACL_CHECK(aclrtMemcpyAsync(
                    cur_dst_buffer, size_cpy, cur_permute_buffer, size_cpy,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
                cur_dst_buffer =
                    (char*)cur_dst_buffer + KH * KW * ggml_type_size(dst->type);
                cur_permute_buffer = (char*)cur_permute_buffer +
                                     KH * KW * IC * ggml_type_size(dst->type);
            }
        }
    } else {
        offset = KH * KW * n_step_w *
                 ggml_type_size(dst->type);  // equal to ggml_nbytes(dst)
        ACL_CHECK(aclrtMemcpyAsync(dst->data, offset,
                                   (char*)tmp_permute_buffer + offset, offset,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
    }

    // release
    ACL_CHECK(aclDestroyTensor(tmp_permute_tensor));
}

void ggml_cann_im2col(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];  // kernel
    ggml_tensor* src1 = dst->src[1];  // input

    GGML_TENSOR_BINARY_OP_LOCALS;

    // aclnnIm2col only works on 2D. set s1, p1, d1 to 1 to perform 2D
    // im2col and do post-processing to restore it to 1D.
    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;
    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = is_2D ? ((const int32_t*)(dst->op_params))[1] : 1;
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = is_2D ? ((const int32_t*)(dst->op_params))[3] : 1;
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = is_2D ? ((const int32_t*)(dst->op_params))[5] : 1;

    const int64_t N = ne13;
    const int64_t IC = ne12;
    const int64_t KH = ne01;
    const int64_t KW = ne00;
    const int64_t IW = ne10;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    // memory allocated increased to 3x when is_2D == false
    const int64_t n_bytes_factor = is_2D ? 1 : 3;

    // im2col: [N,C,H,W] -> [N, IC * KH * KW, OW * OH * n_bytes_factor]
    aclTensor* acl_src1 = ggml_cann_create_tensor(src1);
    int64_t tmp_im2col_ne[] = {OW * OH * n_bytes_factor, IC * KH * KW, N};
    size_t tmp_im2col_nb[GGML_MAX_DIMS - 1];

    tmp_im2col_nb[0] = ggml_type_size(src1->type);
    for (int i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_im2col_nb[i] = tmp_im2col_nb[i - 1] * tmp_im2col_ne[i - 1];
    }

    // Calculate im2col.
    // If dst is f16, tmp_buffer is f32, we need alloc src.typesize *
    // dst.elemcount.
    ggml_cann_pool_alloc im2col_allocator(
        ctx.pool(),
        ggml_nelements(dst) * ggml_element_size(src1) * n_bytes_factor);
    void* tmp_im2col_buffer = im2col_allocator.get();

    aclTensor* tmp_im2col_tensor = ggml_cann_create_tensor(
        tmp_im2col_buffer, ggml_cann_type_mapping(src1->type),
        ggml_type_size(src1->type), tmp_im2col_ne, tmp_im2col_nb,
        GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    std::vector<int64_t> kernel_dims = {KH, KW};
    std::vector<int64_t> dilation_size = {d1, d0};
    std::vector<int64_t> padding_dims = {p1, p0};
    std::vector<int64_t> stride_dims = {s1, s0};
    auto* kernel_size = aclCreateIntArray(kernel_dims.data(), 2);
    auto* dilations = aclCreateIntArray(dilation_size.data(), 2);
    auto* paddings = aclCreateIntArray(padding_dims.data(), 2);
    auto* strides = aclCreateIntArray(stride_dims.data(), 2);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnIm2colGetWorkspaceSize(acl_src1, kernel_size, dilations,
                                          paddings, strides, tmp_im2col_tensor,
                                          &workspaceSize, &executor));

    ggml_cann_pool_alloc workspace_allocator(ctx.pool());
    if (workspaceSize > 0) {
        workspace_allocator.alloc(workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnIm2col(workspaceAddr, workspaceSize, executor, ctx.stream()));

    // Cast if dst is f16.
    aclTensor* tmp_cast_tensor = nullptr;
    ggml_cann_pool_alloc tmp_cast_allocator(ctx.pool());
    void* tmp_cast_buffer = nullptr;
    if (src1->type != dst->type) {
        tmp_cast_allocator.alloc(ggml_nbytes(dst) * n_bytes_factor);
        tmp_cast_buffer = tmp_cast_allocator.get();
        size_t temp_cast_nb[GGML_MAX_DIMS - 1];
        temp_cast_nb[0] = ggml_type_size(dst->type);
        for (int i = 1; i < GGML_MAX_DIMS - 1; i++) {
            temp_cast_nb[i] = temp_cast_nb[i - 1] * tmp_im2col_ne[i - 1];
        }

        tmp_cast_tensor = ggml_cann_create_tensor(
            tmp_cast_buffer, ggml_cann_type_mapping(dst->type),
            ggml_type_size(dst->type), tmp_im2col_ne, temp_cast_nb,
            GGML_MAX_DIMS - 1, ACL_FORMAT_ND);
        aclnn_cast(ctx, tmp_im2col_tensor, tmp_cast_tensor,
                   ggml_cann_type_mapping(dst->type));
    }

    // post-processing
    if (is_2D) {
        ggml_cann_im2col_2d_post_process(ctx, dst, src1, tmp_cast_tensor,
                                         tmp_im2col_tensor);
    } else {
        std::vector<int64_t> im2col_op_params = {
            KH, KW, IW, IC, N, OH, OW, s0, p0, d0, n_bytes_factor};
        ggml_cann_im2col_1d_post_process(ctx, dst, src1, tmp_cast_tensor,
                                         tmp_im2col_tensor, im2col_op_params);
    }

    // release
    ACL_CHECK(aclDestroyTensor(acl_src1));
    ACL_CHECK(aclDestroyTensor(tmp_im2col_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_cast_tensor));
    ACL_CHECK(aclDestroyIntArray(kernel_size));
    ACL_CHECK(aclDestroyIntArray(dilations));
    ACL_CHECK(aclDestroyIntArray(paddings));
    ACL_CHECK(aclDestroyIntArray(strides));
}

/**
 * @brief Applies element-wise exponential function to the elements of a tensor.
 *
 * This function computes the exponential of each element in the source tensor
 * `acl_src` and stores the result back into the same tensor.
 * The operation is defined as:
 * \f[
 *     \text {acl_src }_i=e^{acl\_src_i}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The tensor on which the exponential function will be applied.
 */
static void aclnn_exp(ggml_backend_cann_context& ctx, aclTensor* acl_src) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(
        aclnnInplaceExpGetWorkspaceSize(acl_src, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnInplaceExp(workspaceAddr, workspaceSize, executor, ctx.stream()));
}

/**
 * @brief Multiplies elements of a tensor by a scalar value, optionally
 * in-place.
 *
 * This function multiplies each element of the source tensor `acl_src` by the
 * scalar `scale` and stores the result in the destination tensor `acl_dst`. If
 * `inplace` is true, `acl_dst` will not be used and the operation is performed
 *  in-place on `acl_src`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=\text {acl_src }_i \times \text {scale}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose elements will be multiplied.
 * @param scale The scalar value by which each element of `acl_src` will be
 * multiplied.
 * @param acl_dst The destination tensor where the result will be stored if
 * `inplace` is false.
 * @param inplace Flag indicating whether to perform the operation in-place on
 * `acl_src`.
 */
static void aclnn_muls(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                       float scale, aclTensor* acl_dst, bool inplace) {
    aclScalar* acl_scale = aclCreateScalar(&scale, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    if (inplace) {
        ACL_CHECK(aclnnInplaceMulsGetWorkspaceSize(acl_src, acl_scale,
                                                   &workspaceSize, &executor));
        if (workspaceSize > 0) {
            ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
            workspaceAddr = workspace_allocator.get();
        }

        ACL_CHECK(aclnnInplaceMuls(workspaceAddr, workspaceSize, executor,
                                   ctx.stream()));
    } else {
        ACL_CHECK(aclnnMulsGetWorkspaceSize(acl_src, acl_scale, acl_dst,
                                            &workspaceSize, &executor));
        if (workspaceSize > 0) {
            ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
            workspaceAddr = workspace_allocator.get();
        }

        ACL_CHECK(
            aclnnMuls(workspaceAddr, workspaceSize, executor, ctx.stream()));
    }

    ACL_CHECK(aclDestroyScalar(acl_scale));
}

/**
 * @brief Performs an in-place element-wise multiplication of two tensors.
 *
 * This function performs an element-wise multiplication of the tensors
 * `acl_src` and `acl_other` and stores the result in `acl_src`.
 * The operation is defined as:
 * \f[
 *     \text {acl_src }_i=\text {acl_src }_i \times \text {acl_other }_i
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor where the multiplication result will be
 * stored.
 * @param acl_other The tensor whose elements will be multiplied with `acl_src`.
 */
static void aclnn_inplace_mul(ggml_backend_cann_context& ctx,
                              aclTensor* acl_src, aclTensor* acl_other) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceMulGetWorkspaceSize(acl_src, acl_other,
                                              &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnInplaceMul(workspaceAddr, workspaceSize, executor, ctx.stream()));
}

/**
 * @brief Performs element-wise multiplication of two tensors and stores the
 * result in a destination tensor.
 *
 * This function performs element-wise multiplication of the tensors `acl_src`
 * and `acl_other` and stores the result in the destination tensor `acl_dst`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=\text {acl_src }_i \times \text {acl_other }_i
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The first tensor for element-wise multiplication.
 * @param acl_other The second tensor for element-wise multiplication.
 * @param acl_dst The destination tensor where the result will be stored.
 */
static void aclnn_mul(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                      aclTensor* acl_other, aclTensor* acl_dst) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnMulGetWorkspaceSize(acl_src, acl_other, acl_dst,
                                       &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnMul(workspaceAddr, workspaceSize, executor, ctx.stream()));
}
static void aclnn_mul_inplace(ggml_backend_cann_context& ctx,
                              aclTensor* acl_src, aclTensor* acl_other) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceMulGetWorkspaceSize(acl_src, acl_other,
                                              &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnInplaceMul(workspaceAddr, workspaceSize, executor, ctx.stream()));
}

/**
 * @brief Applies element-wise cosine function to the elements of a tensor.
 *
 * This function computes the cosine of each element in the source tensor
 * `acl_src` and stores the result in the destination tensor `acl_dst`. The
 * operation is defined as: \f[ \text {acl_dst }_i=\cos \left(\text {acl_src
 * }_i\right) \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor on which the cosine function will be
 * applied.
 * @param acl_dst The destination tensor where the cosine results will be
 * stored.
 */
static void aclnn_cos(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                      aclTensor* acl_dst) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(
        aclnnCosGetWorkspaceSize(acl_src, acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnCos(workspaceAddr, workspaceSize, executor, ctx.stream()));
}

/**
 * @brief Applies element-wise sine function to the elements of a tensor.
 *
 * This function computes the sine of each element in the source tensor
 `acl_src`
 * and stores the result in the destination tensor `acl_dst`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=\sin \left(\text {acl_src }_i\right)
 * \f]

 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor on which the sine function will be applied.
 * @param acl_dst The destination tensor where the sine results will be stored.
 */
static void aclnn_sin(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                      aclTensor* acl_dst) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(
        aclnnSinGetWorkspaceSize(acl_src, acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnSin(workspaceAddr, workspaceSize, executor, ctx.stream()));
}

/**
 * @brief Performs element-wise division of tensor1 by tensor2 , multiplies the
 result by the scalar value and adds it to self .
 *
 * Performs element-wise division of tensor1 by tensor2,
 * multiplies the result by the scalar value and adds it to self .
 * The operation is defined as:
 * \f[
 *     \text{out}_i = \text{selft}_i + \text{value} \times
 \frac{\text{tensor1}_i}{\text{tensor2}_i}
 * \f]

 * @param ctx The context for the CANN backend operations.
 * @param acl_self The source tensor on which the addcdiv function will be
 applied.
 * @param tensor1 Numerator tensor.
 * @param tensor2 Denominator tensor.
 * @param value The value to be used for coefficient.
 */
static void aclnn_inplace_addcdiv(ggml_backend_cann_context& ctx,
                                  aclTensor* acl_self, aclTensor* tensor1,
                                  aclTensor* tensor2, float value) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    aclScalar* acl_value = aclCreateScalar(&value, aclDataType::ACL_FLOAT);

    ACL_CHECK(aclnnInplaceAddcdivGetWorkspaceSize(
        acl_self, tensor1, tensor2, acl_value, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnInplaceAddcdiv(workspaceAddr, workspaceSize, executor,
                                  ctx.stream()));
}

/**
 * @brief Matrix division, optionally in-place.
 *
 * This function division each element of the source tensor `acl_src` by the
 * tensor `acl_other` and stores the result in the destination tensor `acl_dst`.
 * If `inplace` is true, `acl_dst` will not be used and the operation is
 * performed in-place on `acl_src`. The operation is defined as: \f[
 *     \text{dst}_i = \frac{\text{acl_src}_i}{\text{acl_other}_i}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src Numerator tensor..
 * @param acl_other Denominator tensor.
 * @param acl_dst The destination tensor where the result will be stored if
 * `inplace` is false.
 * @param inplace Flag indicating whether to perform the operation in-place on
 * `acl_src`.
 */
static void aclnn_div_tensor(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                             aclTensor* acl_other, aclTensor* acl_dst,
                             bool inplace) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    if (inplace) {
        ACL_CHECK(aclnnInplaceDivGetWorkspaceSize(acl_src, acl_other,
                                                  &workspaceSize, &executor));
        if (workspaceSize > 0) {
            ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
            workspaceAddr = workspace_allocator.get();
        }

        ACL_CHECK(aclnnInplaceDiv(workspaceAddr, workspaceSize, executor,
                                  ctx.stream()));
    } else {
        ACL_CHECK(aclnnDivGetWorkspaceSize(acl_src, acl_other, acl_dst,
                                           &workspaceSize, &executor));
        if (workspaceSize > 0) {
            ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
            workspaceAddr = workspace_allocator.get();
        }

        ACL_CHECK(
            aclnnDiv(workspaceAddr, workspaceSize, executor, ctx.stream()));
    }
}

void ggml_cann_timestep_embedding(ggml_backend_cann_context& ctx,
                                  ggml_tensor* dst) {
    const ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    const int max_period = dst->op_params[1];
    int half = dim / 2;

    aclTensor* acl_src = ggml_cann_create_tensor(src);

    // arange: [0, ..., half)
    float start = 0;
    float stop = half;
    float step = 1;
    int64_t n_elements_arange = half;
    int64_t tmp_arange_ne[] = {half};
    size_t tmp_arange_nb[] = {sizeof(dst->type)};

    ggml_cann_pool_alloc arange_allocator(ctx.pool(), half * sizeof(dst->type));
    void* tmp_arange_buffer = arange_allocator.get();
    aclTensor* tmp_arange_tensor = ggml_cann_create_tensor(
        tmp_arange_buffer, ggml_cann_type_mapping(dst->type),
        ggml_type_size(dst->type), tmp_arange_ne, tmp_arange_nb,
        GGML_MAX_DIMS - 3, ACL_FORMAT_ND);

    aclnn_arange(ctx, tmp_arange_tensor, start, stop, step, n_elements_arange);

    // freq
    float freq_param = -logf(max_period) / half;
    bool inplace = true;
    aclnn_muls(ctx, tmp_arange_tensor, freq_param, nullptr, inplace);
    aclnn_exp(ctx, tmp_arange_tensor);

    // permute: src [0,1,2,3]->[0,1,3,2]
    int64_t tmp_permute_ne[] = {src->ne[1], src->ne[0], src->ne[2], src->ne[3]};
    size_t tmp_permute_nb[GGML_MAX_DIMS];
    tmp_permute_nb[0] = ggml_type_size(src->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_permute_nb[i] = tmp_permute_nb[i - 1] * tmp_permute_ne[i - 1];
    }

    ggml_cann_pool_alloc permute_allocator(ctx.pool(), ggml_nbytes(src));
    void* tmp_permute_buffer = permute_allocator.get();
    aclTensor* tmp_permute_tenosr = ggml_cann_create_tensor(
        tmp_permute_buffer, ggml_cann_type_mapping(src->type),
        ggml_type_size(src->type), tmp_permute_ne, tmp_permute_nb,
        GGML_MAX_DIMS, ACL_FORMAT_ND);
    int64_t permute_dim[] = {0, 1, 3, 2};
    int64_t num_dims = 4;
    aclnn_permute(ctx, acl_src, tmp_permute_tenosr, permute_dim, num_dims);

    // timestep * freq
    int64_t tmp_mul_ne[] = {src->ne[1] * half, src->ne[0], src->ne[2],
                            src->ne[3]};
    size_t tmp_mul_nb[GGML_MAX_DIMS];
    tmp_mul_nb[0] = ggml_type_size(src->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_mul_nb[i] = tmp_mul_nb[i - 1] * tmp_mul_ne[i - 1];
    }

    int mul_nelements =
        src->ne[1] * half * src->ne[0] * src->ne[2] * src->ne[3];

    ggml_cann_pool_alloc mul_allocator(
        ctx.pool(), mul_nelements * ggml_type_size(src->type));
    void* tmp_mul_buffer = mul_allocator.get();
    aclTensor* tmp_mul_tensor = ggml_cann_create_tensor(
        tmp_mul_buffer, ggml_cann_type_mapping(src->type),
        ggml_type_size(src->type), tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS,
        ACL_FORMAT_ND);
    aclnn_mul(ctx, tmp_permute_tenosr, tmp_arange_tensor, tmp_mul_tensor);

    // cos
    ggml_cann_pool_alloc cos_allocator(
        ctx.pool(), mul_nelements * ggml_type_size(src->type));
    void* tmp_cos_buffer = cos_allocator.get();
    aclTensor* tmp_cos_tensor = ggml_cann_create_tensor(
        tmp_cos_buffer, ggml_cann_type_mapping(dst->type),
        ggml_type_size(dst->type), tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS,
        ACL_FORMAT_ND);

    aclnn_cos(ctx, tmp_mul_tensor, tmp_cos_tensor);

    // sin
    ggml_cann_pool_alloc sin_allocator(
        ctx.pool(), mul_nelements * ggml_type_size(src->type));
    void* tmp_sin_buffer = sin_allocator.get();
    aclTensor* tmp_sin_tensor = ggml_cann_create_tensor(
        tmp_sin_buffer, ggml_cann_type_mapping(dst->type),
        ggml_type_size(dst->type), tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS,
        ACL_FORMAT_ND);

    aclnn_sin(ctx, tmp_mul_tensor, tmp_sin_tensor);

    // concat
    int64_t concat_dim = 3;
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);
    aclTensor* tensors[] = {tmp_cos_tensor, tmp_sin_tensor};
    aclTensorList* tensorList = aclCreateTensorList(tensors, 2);
    aclnn_concat(ctx, tensorList, acl_dst, concat_dim);

    // release
    // segmentation fault when delete both tensorList and his elements.
    ACL_CHECK(aclDestroyTensorList(tensorList));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(tmp_arange_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_permute_tenosr));
    ACL_CHECK(aclDestroyTensor(tmp_mul_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

/**
 * @brief Fills a tensor with a scalar value.
 *
 * This function fills the destination tensor `acl_dst` with the scalar value
 * `scalar`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param scalar The scalar value used to fill the tensor.
 * @param acl_dst The destination tensor to be filled with the scalar value.
 */
static void aclnn_fill_scalar(ggml_backend_cann_context& ctx, float scalar,
                              aclTensor* acl_dst) {
    auto acl_scalar = aclCreateScalar(&scalar, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceFillScalarGetWorkspaceSize(
        acl_dst, acl_scalar, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnInplaceFillScalar(workspaceAddr, workspaceSize, executor,
                                     ctx.stream()));
    ACL_CHECK(aclDestroyScalar(acl_scalar));
}

/**
 * @brief Raises each element of a tensor to the power of the corresponding
 * element in another tensor.
 *
 * This function computes the element-wise power of the destination tensor
 * `acl_dst` raised to the power of the exponent tensor `acl_exp`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=acl\_dst_i^{\text {acl_exp }_i}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_dst The destination tensor, which also serves as the base tensor.
 * @param acl_exp The exponent tensor, each element of which is used to raise
 * the corresponding element in the destination tensor.
 */
static void aclnn_pow_tensor_tensor(ggml_backend_cann_context& ctx,
                                    aclTensor* acl_dst, aclTensor* acl_exp) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplacePowTensorTensorGetWorkspaceSize(
        acl_dst, acl_exp, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize,
                                          executor, ctx.stream()));
}

/**
 * @brief   Applies the Alibi (Attention with Linear Biases) mechanism to the
 * @details This function implements the Alibi mechanism, which introduces
 *          learnable biases into the attention scores to simulate relative
 *          position encoding without the need for explicit positional
 *          embeddings.
 *
 * @param ctx          The backend CANN context for executing operations.
 * @param acl_src      The source tensor representing the query or key.
 * @param acl_position The position tensor containing relative positions.
 * @param acl_dst      The destination tensor where the result will be stored.
 * @param n_head       The number of attention heads.
 * @param src_ne       The dimensions of the source tensor.
 * @param src_nb0      The byte size of the first dimension of the source
 tensor.
 * @param max_bias     The maximum bias value used in the Alibi mechanism.
 * @param dst          The destination tensor object for additional metadata.
 *
 * The function performs the following steps:
 * 1. Calculates the logarithm floor of the number of heads to determine the
      base for bias calculation.
 * 2. Initializes arrays with arithmetic sequences and fills them with bias
      values.
 * 3. Computes the bias tensor based on the calculated biases and arithmetic
      sequences.
 * 4. Reshapes the bias tensor to match the dimensions of the input tensors.
 * 5. Multiplies the position tensor by the bias tensor.
 * 6. Adds the result of the multiplication to the source tensor to produce the
      final output.
 */
static void aclnn_alibi(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                        aclTensor* acl_position, aclTensor* acl_dst,
                        const int n_head, int64_t* src_ne, const size_t src_nb0,
                        float max_bias, ggml_tensor* dst) {
    const int64_t ne2_ne3 = src_ne[2] * src_ne[3];
    GGML_ASSERT(src_nb0 == sizeof(float));
    GGML_ASSERT(n_head == src_ne[2]);

    const int n_heads_log2_floor = 1u << (uint32_t)floor(log2(n_head));

    float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
    float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

    // init arange
    ggml_cann_pool_alloc arange_allocator(ctx.pool(),
                                          ne2_ne3 * ggml_type_size(dst->type));
    void* tmp_arange_buffer = arange_allocator.get();

    // arange1: [1, ..., n_heads_log2_floor+1)
    float start = 1;
    float stop = n_heads_log2_floor + 1;
    float step = 1;
    int64_t n_elements_arange = n_heads_log2_floor;

    int64_t tmp_arange1_ne[] = {n_heads_log2_floor};
    size_t tmp_arange1_nb[] = {sizeof(dst->type)};
    aclTensor* tmp_arange1_tensor = ggml_cann_create_tensor(
        tmp_arange_buffer, ggml_cann_type_mapping(dst->type),
        ggml_type_size(dst->type), tmp_arange1_ne, tmp_arange1_nb,
        GGML_MAX_DIMS - 3, ACL_FORMAT_ND);

    aclnn_arange(ctx, tmp_arange1_tensor, start, stop, step, n_elements_arange);

    aclTensor* tmp_arange2_tensor = nullptr;
    if (n_heads_log2_floor < ne2_ne3) {
        // arange2: [1, ..., 2 * (k - n_heads_log2_floor) + 1)
        start = 1;
        stop = 2 * (ne2_ne3 - n_heads_log2_floor) + 1;
        step = 2;
        n_elements_arange = ne2_ne3 - n_heads_log2_floor;
        int64_t tmp_arange2_ne[] = {ne2_ne3 - n_heads_log2_floor};
        size_t tmp_arange2_nb[] = {sizeof(dst->type)};

        aclTensor* tmp_arange2_tensor = ggml_cann_create_tensor(
            (char*)tmp_arange_buffer +
                n_heads_log2_floor * ggml_type_size(dst->type),
            ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
            tmp_arange2_ne, tmp_arange2_nb, GGML_MAX_DIMS - 3, ACL_FORMAT_ND);
        aclnn_arange(ctx, tmp_arange2_tensor, start, stop, step,
                     n_elements_arange);
    }

    // init mk_base
    ggml_cann_pool_alloc mk_base_allocator(ctx.pool(),
                                           ne2_ne3 * ggml_type_size(dst->type));
    void* tmp_mk_base_buffer = mk_base_allocator.get();
    int64_t tmp_mk_base1_ne[] = {n_heads_log2_floor};
    size_t tmp_mk_base1_nb[] = {sizeof(dst->type)};
    aclTensor* tmp_mk_base1_tensor = ggml_cann_create_tensor(
        tmp_mk_base_buffer, ggml_cann_type_mapping(dst->type),
        ggml_type_size(dst->type), tmp_mk_base1_ne, tmp_mk_base1_nb,
        GGML_MAX_DIMS - 3, ACL_FORMAT_ND);

    aclnn_fill_scalar(ctx, m0, tmp_mk_base1_tensor);

    aclTensor* tmp_mk_base2_tensor = nullptr;
    if (n_heads_log2_floor < ne2_ne3) {
        int64_t tmp_mk_base2_ne[] = {ne2_ne3 - n_heads_log2_floor};
        size_t tmp_mk_base2_nb[] = {sizeof(dst->type)};
        aclTensor* tmp_mk_base2_tensor = ggml_cann_create_tensor(
            (char*)tmp_mk_base_buffer +
                n_heads_log2_floor * ggml_type_size(dst->type),
            ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
            tmp_mk_base2_ne, tmp_mk_base2_nb, GGML_MAX_DIMS - 3, ACL_FORMAT_ND);
        aclnn_fill_scalar(ctx, m1, tmp_mk_base2_tensor);
    }

    // init mk
    int64_t tmp_mk_base_ne[] = {ne2_ne3};
    size_t tmp_mk_base_nb[] = {sizeof(dst->type)};
    aclTensor* tmp_mk_base_tensor = ggml_cann_create_tensor(
        tmp_mk_base_buffer, ggml_cann_type_mapping(dst->type),
        ggml_type_size(dst->type), tmp_mk_base_ne, tmp_mk_base_nb,
        GGML_MAX_DIMS - 3, ACL_FORMAT_ND);
    aclTensor* tmp_arange_tensor = ggml_cann_create_tensor(
        tmp_arange_buffer, ggml_cann_type_mapping(dst->type),
        ggml_type_size(dst->type), tmp_mk_base_ne, tmp_mk_base_nb,
        GGML_MAX_DIMS - 3, ACL_FORMAT_ND);
    aclnn_pow_tensor_tensor(ctx, tmp_mk_base_tensor, tmp_arange_tensor);

    // reshape mk
    int64_t tmp_mk_ne[] = {1, 1, src_ne[2], src_ne[3]};
    size_t tmp_mk_nb[GGML_MAX_DIMS];
    tmp_mk_nb[0] = ggml_type_size(dst->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_mk_nb[i] = tmp_mk_nb[i - 1] * tmp_mk_ne[i - 1];
    }
    aclTensor* tmp_mk_tensor = ggml_cann_create_tensor(
        tmp_mk_base_buffer, ggml_cann_type_mapping(dst->type),
        ggml_type_size(dst->type), tmp_mk_ne, tmp_mk_nb, GGML_MAX_DIMS,
        ACL_FORMAT_ND);

    // acl_position * mk
    int64_t tmp_output_ne[] = {src_ne[0], src_ne[1], src_ne[2], src_ne[3]};
    size_t tmp_output_nb[GGML_MAX_DIMS];
    tmp_output_nb[0] = ggml_type_size(dst->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_output_nb[i] = tmp_output_nb[i - 1] * tmp_output_ne[i - 1];
    }
    ggml_cann_pool_alloc output_allocator(ctx.pool(), ggml_nbytes(dst));
    void* tmp_output_buffer = output_allocator.get();
    aclTensor* tmp_output_tensor = ggml_cann_create_tensor(
        tmp_output_buffer, ggml_cann_type_mapping(dst->type),
        ggml_type_size(dst->type), tmp_output_ne, tmp_output_nb, GGML_MAX_DIMS,
        ACL_FORMAT_ND);
    aclnn_mul(ctx, acl_position, tmp_mk_tensor, tmp_output_tensor);

    // add
    aclnn_add(ctx, tmp_output_tensor, acl_src, acl_dst);

    ACL_CHECK(aclDestroyTensor(tmp_arange1_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_arange2_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_mk_base1_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_mk_base2_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_mk_base_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_arange_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_mk_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_output_tensor));
}

void ggml_cann_cpy(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    // DEBUG
    //  if(dst->src[0] == dst->src[1]){
    //  print_acltensor(dst->src[0]->data,dst->src[0]->ne,ggml_cann_type_mapping(dst->src[0]->type),dst->src[0]->name,4,nullptr,&ctx);
    // }

    ggml_cann_dup(ctx, dst);
}

/**
 * @brief Performs element-wise addition of two tensors in place.
 *
 * This function adds the source tensor `acl_src` to the destination tensor
 * `acl_dst` element-wise and stores the result in the destination tensor
 * `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor to be added.
 * @param acl_dst The destination tensor which will hold the result of the
 * addition.
 */
static void aclnn_inplace_add(ggml_backend_cann_context& ctx,
                              aclTensor* acl_src, aclTensor* acl_dst) {
    aclScalar* alpha = nullptr;
    float alphaValue = 1.0f;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceAddGetWorkspaceSize(acl_dst, acl_src, alpha,
                                              &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyScalar(alpha));
}

/**
 * @brief Applies the softmax function to a tensor along a specified dimension.
 *
 * This function computes the softmax of the source tensor `acl_src` along the
 * specified dimension `dim` and stores the result in the destination tensor
 * `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor on which the softmax function will be
 * applied.
 * @param dim The dimension along which the softmax function will be computed.
 * @param acl_dst The destination tensor where the softmax results will be
 * stored.
 */
static void aclnn_softmax(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                          int64_t dim, aclTensor* acl_dst) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnSoftmaxGetWorkspaceSize(acl_src, dim, acl_dst,
                                           &workspaceSize, &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    aclrtStream stream = ctx.stream();
    ACL_CHECK(aclnnSoftmax(workspaceAddr, workspaceSize, executor, stream));
}

void ggml_cann_softmax(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];  // mask

    aclTensor* acl_src0 = ggml_cann_create_tensor(src0);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    float scale = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale, (float*)dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float*)dst->op_params + 1, sizeof(float));

    // input mul scale
    aclScalar* acl_scale = aclCreateScalar(&scale, aclDataType::ACL_FLOAT);

    size_t n_bytes = ggml_nbytes(src0);
    ggml_cann_pool_alloc mul_scale_allocator(ctx.pool(), n_bytes);
    void* input_mul_scale_buffer = mul_scale_allocator.get();
    aclTensor* acl_input_mul_scale_tensor = ggml_cann_create_tensor(
        input_mul_scale_buffer, ACL_FLOAT, ggml_type_size(src0->type), src0->ne,
        src0->nb, GGML_MAX_DIMS);

    bool inplace = false;
    aclnn_muls(ctx, acl_src0, scale, acl_input_mul_scale_tensor, inplace);

    // mask
    aclTensor* acl_src1_fp32_tensor = nullptr;
    aclTensor* tmp_mask_tensor = nullptr;
    ggml_cann_pool_alloc src1_fp32_allocator(ctx.pool());
    if (src1) {
        const bool use_f16 = src1->type == GGML_TYPE_F16;
        if (use_f16) {
            // cast to fp32
            size_t n_bytes = ggml_nelements(src1) * sizeof(float_t);
            size_t src1_fp32_nb[GGML_MAX_DIMS];
            src1_fp32_nb[0] = sizeof(float_t);
            for (int i = 1; i < GGML_MAX_DIMS; i++) {
                src1_fp32_nb[i] = src1_fp32_nb[i - 1] * src1->ne[i - 1];
            }
            src1_fp32_allocator.alloc(n_bytes);
            void* src1_fp32_buffer = src1_fp32_allocator.get();
            acl_src1_fp32_tensor = ggml_cann_create_tensor(
                src1_fp32_buffer, ACL_FLOAT, sizeof(float), src1->ne,
                src1_fp32_nb, GGML_MAX_DIMS);
            aclTensor* acl_src1 = ggml_cann_create_tensor(src1);
            aclnn_cast(ctx, acl_src1, acl_src1_fp32_tensor, ACL_FLOAT);

            ACL_CHECK(aclDestroyTensor(acl_src1));
        } else {
            acl_src1_fp32_tensor = ggml_cann_create_tensor(src1);
        }

        // broadcast the mask across rows, only use ne11 of ne01 in mask
        if (src1->ne[1] != src0->ne[1]) {
            // mask shape: [1,1,ne11,ne10]
            int64_t tmp_mask_ne[] = {src0->ne[0], src0->ne[1], 1, 1};
            size_t tmp_mask_nb[GGML_MAX_DIMS];
            tmp_mask_nb[0] = sizeof(float_t);
            for (int i = 1; i < GGML_MAX_DIMS; i++) {
                tmp_mask_nb[i] = tmp_mask_nb[i - 1] * tmp_mask_ne[i - 1];
            }
            tmp_mask_tensor = ggml_cann_create_tensor(
                src1->data, ACL_FLOAT, sizeof(float), tmp_mask_ne, tmp_mask_nb,
                GGML_MAX_DIMS, ACL_FORMAT_ND);
        }

        // alibi
        const int n_head = src0->ne[2];
        const size_t src_nb0 = src0->nb[0];

        n_bytes = ggml_nbytes(dst);
        ggml_cann_pool_alloc output_allocator(ctx.pool(), n_bytes);
        void* output_buffer = output_allocator.get();
        aclTensor* alibi_output_tensor = ggml_cann_create_tensor(
            output_buffer, ACL_FLOAT, ggml_type_size(dst->type), dst->ne,
            dst->nb, GGML_MAX_DIMS);
        if (max_bias <= 0.0f) {
            // slope = 1.0
            if (tmp_mask_tensor) {
                aclnn_add(ctx, tmp_mask_tensor, acl_input_mul_scale_tensor,
                          alibi_output_tensor);
            } else {
                aclnn_add(ctx, acl_src1_fp32_tensor, acl_input_mul_scale_tensor,
                          alibi_output_tensor);
            }
        } else {
            // slope != 1.0
            if (tmp_mask_tensor) {
                aclnn_alibi(ctx, acl_input_mul_scale_tensor, tmp_mask_tensor,
                            alibi_output_tensor, n_head, src0->ne, src_nb0,
                            max_bias, dst);
            } else {
                aclnn_alibi(ctx, acl_input_mul_scale_tensor,
                            acl_src1_fp32_tensor, alibi_output_tensor, n_head,
                            src0->ne, src_nb0, max_bias, dst);
            }
        }

        // softmax
        aclnn_softmax(ctx, alibi_output_tensor, 3, acl_dst);
        ACL_CHECK(aclDestroyTensor(alibi_output_tensor));
    } else {
        aclnn_softmax(ctx, acl_input_mul_scale_tensor, 3, acl_dst);
    }

    ACL_CHECK(aclDestroyTensor(acl_src0));
    ACL_CHECK(aclDestroyTensor(acl_src1_fp32_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyScalar(acl_scale));
    ACL_CHECK(aclDestroyTensor(acl_input_mul_scale_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_mask_tensor));
}

void ggml_cann_get_rows(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];

    ggml_cann_pool_alloc src0_extra_allocator(ctx.pool(), sizeof(ggml_tensor));
    ggml_cann_pool_alloc src1_extra_allocator(ctx.pool(), sizeof(ggml_tensor));
    ggml_cann_pool_alloc dst_extra_allocator(ctx.pool(), sizeof(ggml_tensor));
    src0->extra = src0_extra_allocator.get();
    src1->extra = src1_extra_allocator.get();
    dst->extra = dst_extra_allocator.get();
    ACL_CHECK(aclrtMemcpyAsync(src0->extra, sizeof(ggml_tensor), src0,
                               sizeof(ggml_tensor), ACL_MEMCPY_HOST_TO_DEVICE,
                               ctx.stream()));
    ACL_CHECK(aclrtMemcpyAsync(src1->extra, sizeof(ggml_tensor), src1,
                               sizeof(ggml_tensor), ACL_MEMCPY_HOST_TO_DEVICE,
                               ctx.stream()));
    ACL_CHECK(aclrtMemcpyAsync(dst->extra, sizeof(ggml_tensor), dst,
                               sizeof(ggml_tensor), ACL_MEMCPY_HOST_TO_DEVICE,
                               ctx.stream()));

    switch (src0->type) {
        case GGML_TYPE_F32: {
#ifdef ASCEND_310P
            // Special operation for get_row_f32 kernel of 310P: clear the
            // content of dest data buffer when row is not aligned to 32 bytes
            if ((src0->ne[0] % 8) != 0) {
                size_t dst_len = src1->ne[0] * src1->ne[1] * src1->ne[2] *
                                 src0->ne[0] * ggml_type_size(GGML_TYPE_F32);
                ACL_CHECK(aclrtMemset((char*)dst->data, dst_len, 0, dst_len));
            }
#endif
            aclrtlaunch_ascendc_get_row_f32(
                24, ctx.stream(), src0->data, src1->data, dst->data,
                ((ggml_tensor*)src0->extra)->ne,
                ((ggml_tensor*)src0->extra)->nb,
                ((ggml_tensor*)src1->extra)->ne,
                ((ggml_tensor*)src1->extra)->nb, ((ggml_tensor*)dst->extra)->ne,
                ((ggml_tensor*)dst->extra)->nb);
            break;
        }
        case GGML_TYPE_F16: {
#ifdef ASCEND_310P
            // Special operation for get_row_f16 kernel of 310P: clear the
            // content of dest data buffer when row is not aligned to 32 bytes
            if ((src0->ne[0] % 16) != 0) {
                size_t dst_len =
                    src1->ne[0] * src1->ne[1] * src1->ne[2] * src0->ne[0] *
                    ggml_type_size(
                        GGML_TYPE_F32);  // out is also f32, even input is f16
                ACL_CHECK(aclrtMemset((char*)dst->data, dst_len, 0, dst_len));
            }
#endif
            aclrtlaunch_ascendc_get_row_f16(
                24, ctx.stream(), src0->data, src1->data, dst->data,
                ((ggml_tensor*)src0->extra)->ne,
                ((ggml_tensor*)src0->extra)->nb,
                ((ggml_tensor*)src1->extra)->ne,
                ((ggml_tensor*)src1->extra)->nb, ((ggml_tensor*)dst->extra)->ne,
                ((ggml_tensor*)dst->extra)->nb);
            break;
        }
        case GGML_TYPE_Q4_0:
            aclrtlaunch_ascendc_get_row_q4_0(
                24, ctx.stream(), src0->data, src1->data, dst->data,
                ((ggml_tensor*)src0->extra)->ne,
                ((ggml_tensor*)src1->extra)->ne,
                ((ggml_tensor*)src1->extra)->nb, ((ggml_tensor*)dst->extra)->ne,
                ((ggml_tensor*)dst->extra)->nb);
            break;
        case GGML_TYPE_Q8_0:
            aclrtlaunch_ascendc_get_row_q8_0(
                24, ctx.stream(), src0->data, src1->data, dst->data,
                ((ggml_tensor*)src0->extra)->ne,
                ((ggml_tensor*)src1->extra)->ne,
                ((ggml_tensor*)src1->extra)->nb, ((ggml_tensor*)dst->extra)->ne,
                ((ggml_tensor*)dst->extra)->nb);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

/**
 * @brief Repeats elements of a tensor along a specified dimension.
 *
 * This function repeats each element of the source tensor `acl_src` a specified
 * number of times (`repeats`) along the specified dimension `dim` and stores
 * the result in the destination tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose elements will be repeated.
 * @param acl_dst The destination tensor where the repeated elements will be
 * stored.
 * @param dim The dimension along which the elements will be repeated.
 * @param repeats The number of times each element will be repeated.
 * @param output_size The size of the output tensor.
 */
static void aclnn_repeat_interleave(ggml_backend_cann_context& ctx,
                                    aclTensor* acl_src, aclTensor* acl_dst,
                                    int64_t dim, int64_t repeats,
                                    int64_t output_size) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnRepeatInterleaveIntWithDimGetWorkspaceSize(
        acl_src, repeats, dim, output_size, acl_dst, &workspaceSize,
        &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnRepeatInterleaveIntWithDim(workspaceAddr, workspaceSize,
                                              executor, ctx.stream()));
}

/**
 * @brief Performs matrix multiplication of two tensors.
 *
 * This function computes the matrix multiplication of the input tensor
 * `acl_input` and the weight tensor `acl_weight`, and stores the result in the
 * destination tensor `acl_dst`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst}=\text {acl_input@acl_weight}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_input The input tensor for the matrix multiplication.
 * @param acl_weight The weight tensor for the matrix multiplication.
 * @param acl_dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
static void aclnn_mat_mul(ggml_backend_cann_context& ctx, aclTensor* acl_input,
                          aclTensor* acl_weight, aclTensor* acl_dst) {
    int8_t cube_math_type = 1;  // ALLOW_FP32_DOWN_PRECISION, when input is
                                // fp32, atlas a2 will transpose it to HFLOAT32.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnMatmulGetWorkspaceSize(acl_input, acl_weight, acl_dst,
                                          cube_math_type, &workspaceSize,
                                          &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnMatmul(workspaceAddr, workspaceSize, executor, ctx.stream()));
}

/**
 * @brief Performs matrix multiplication of two 2D tensors.
 *
 * This function computes the matrix multiplication of the input tensor
 * `acl_input` and the weight tensor `acl_weight`, and stores the result in the
 * destination tensor `acl_dst`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst}=\text {acl_input@acl_weight}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_input The input tensor for the matrix multiplication.
 * @param acl_weight The weight tensor for the matrix multiplication.
 * @param acl_dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
static void aclnn_mat_mul_2d(ggml_backend_cann_context& ctx,
                             aclTensor* acl_input, aclTensor* acl_weight,
                             aclTensor* acl_dst) {
    int8_t cube_math_type = 2;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnMmGetWorkspaceSize(acl_input, acl_weight, acl_dst,
                                      cube_math_type, &workspaceSize,
                                      &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnMm(workspaceAddr, workspaceSize, executor, ctx.stream()));
}

/**
 * @brief Performs matrix multiplication of two 3D tensors.
 *
 * This function computes the matrix multiplication of the input tensor
 * `acl_input` and the weight tensor `acl_weight`, and stores the result in the
 * destination tensor `acl_dst`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst}=\text {acl_input@acl_weight}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_input The input tensor for the matrix multiplication.
 * @param acl_weight The weight tensor for the matrix multiplication.
 * @param acl_dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
static void aclnn_mat_mul_3d(ggml_backend_cann_context& ctx,
                             aclTensor* acl_input, aclTensor* acl_weight,
                             aclTensor* acl_dst) {
    int8_t cube_math_type = 2;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnBatchMatMulGetWorkspaceSize(acl_input, acl_weight, acl_dst,
                                               cube_math_type, &workspaceSize,
                                               &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnBatchMatMul(workspaceAddr, workspaceSize, executor, ctx.stream()));
}

/**
 * @brief Performs matrix multiplication with floating-point precision on
 * tensors using the CANN backend.
 *
 * This function performs matrix multiplication of the input tensor and the
 * weight tensor, handling broadcasting and transposing as needed, and stores
 * the result in the destination tensor `dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
static void ggml_cann_mat_mul_fp(ggml_backend_cann_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* weight = dst->src[0];  // weight
    ggml_tensor* input = dst->src[1];   // input

    // when weight ne2 or ne3 is 1, aclnnMatmulGetWorkspaceSize will auto
    // broadcast, when weight ne2 or ne3 is not 1, weight need repeat.
    BCAST_MUL_MAT_SHAPE(input, weight, dst);

    int64_t n_dims = bcast_dims;
    if (bcast_input_ne[3] == bcast_weight_ne[3] && bcast_input_ne[3] == 1) {
        if (bcast_input_ne[2] == 1 && bcast_weight_ne[2] == 1) {
            n_dims = 2;
        } else if (bcast_input_ne[2] == 1) {
            n_dims = 3;
        }
    }

    aclTensor* acl_input_tensor =
        ggml_cann_create_tensor(input, bcast_input_ne, bcast_input_nb, n_dims);
    int64_t transpose_ne[] = {bcast_weight_ne[1], bcast_weight_ne[0],
                              bcast_weight_ne[2], bcast_weight_ne[3],
                              bcast_weight_ne[4], bcast_weight_ne[5]};
    size_t transpose_nb[] = {bcast_weight_nb[1], bcast_weight_nb[0],
                             bcast_weight_nb[2], bcast_weight_nb[3],
                             bcast_weight_nb[4], bcast_weight_nb[5]};
    aclTensor* acl_weight_tensor =
        ggml_cann_create_tensor(weight, transpose_ne, transpose_nb, n_dims);
    aclTensor* acl_dst =
        ggml_cann_create_tensor(dst, bcast_dst_ne, bcast_dst_nb, n_dims);

    switch (n_dims) {
        case 2:
            aclnn_mat_mul_2d(ctx, acl_input_tensor, acl_weight_tensor, acl_dst);
            break;
        case 3:
            aclnn_mat_mul_3d(ctx, acl_input_tensor, acl_weight_tensor, acl_dst);
            break;
        default:
            aclnn_mat_mul(ctx, acl_input_tensor, acl_weight_tensor, acl_dst);
            break;
    }

    ACL_CHECK(aclDestroyTensor(acl_weight_tensor));
    ACL_CHECK(aclDestroyTensor(acl_input_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

/**
 * @brief Performs matrix multiplication with quantized weights and
 * floating-point inputs using the CANN backend.
 *
 * This function performs matrix multiplication of the input tensor `src1` and
 * the weight tensor `src0`, handling broadcasting, transposing, and
 * quantization as needed, and stores the result in the destination tensor
 * `dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
static void ggml_cann_mul_mat_quant(ggml_backend_cann_context& ctx,
                                    ggml_tensor* dst,
                                    const enum ggml_type type) {
    ggml_tensor* src0 = dst->src[0];  // weight
    ggml_tensor* src1 = dst->src[1];  // input

    // The shape of the weight is NCHW.
    // Matrix multiplication uses HW dims.
    // HC is regarded as batch.
    // weight need transpose.
    float weight_elem_size;
    if (type == GGML_TYPE_Q4_0) {
        weight_elem_size = float(sizeof(uint8_t)) / 2;
    } else if (type == GGML_TYPE_Q8_0) {
        weight_elem_size = float(sizeof(uint8_t));
    } else {
        GGML_ABORT("Only support Q4_0 and Q8_0 MUL_MAT");
    }
    float weight_nb[] = {src0->ne[0] * weight_elem_size, weight_elem_size};
    size_t weight_stride = src0->ne[1] * src0->ne[0] * weight_elem_size;
    size_t weight_size = weight_stride * src0->ne[2] * src0->ne[3];

    // scale stored at the end of weight. Also need transpose.
    size_t scale_elem_size = sizeof(uint16_t);
    size_t scale_nb[] = {src0->ne[0] / QK8_0 * scale_elem_size,
                         scale_elem_size};
    size_t scale_stride = src0->ne[1] * src0->ne[0] / QK8_0 * scale_elem_size;
    char* scale_offset = (char*)src0->data + weight_size;

    // input
    size_t input_elem_size = sizeof(uint16_t);
    int64_t input_ne[] = {src1->ne[0], src1->ne[1]};
    size_t input_nb[] = {input_elem_size, input_ne[0] * input_elem_size};
    size_t input_stride = input_ne[0] * input_ne[1] * input_elem_size;
    ggml_cann_pool_alloc input_alloctor(ctx.pool());
    void* input_buffer = src1->data;

    // case in
    if (src1->type != GGML_TYPE_F16) {
        aclTensor* acl_src1_tensor = ggml_cann_create_tensor(src1);
        input_buffer =
            input_alloctor.alloc(ggml_nelements(src1) * input_elem_size);

        int64_t* input_cast_ne = src1->ne;
        size_t input_cast_nb[GGML_MAX_DIMS];
        input_cast_nb[0] = sizeof(uint16_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            input_cast_nb[i] = input_cast_nb[i - 1] * input_cast_ne[i - 1];
        }

        aclTensor* acl_input_tensor = ggml_cann_create_tensor(
            input_buffer, ACL_FLOAT16, input_elem_size, input_cast_ne,
            input_cast_nb, GGML_MAX_DIMS);

        aclnn_cast(ctx, acl_src1_tensor, acl_input_tensor, ACL_FLOAT16);

        ACL_CHECK(aclDestroyTensor(acl_input_tensor));
        ACL_CHECK(aclDestroyTensor(acl_src1_tensor));
    }

    // output
    size_t output_elem_size = sizeof(uint16_t);
    size_t output_nb[] = {output_elem_size, dst->ne[0] * output_elem_size};
    ggml_cann_pool_alloc output_allocator(ctx.pool());
    void* output_buffer =
        output_allocator.alloc(ggml_nelements(dst) * output_elem_size);
    size_t output_stride = dst->ne[0] * dst->ne[1] * output_elem_size;

    // aclnn
    int64_t max_elem_size = 65535;
    int64_t split_size = (src0->ne[1] / max_elem_size) + 1;
    ggml_cann_pool_alloc workspace_allocator(ctx.pool());
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    for (int64_t n1 = 0; n1 < src1->ne[3]; n1++) {
        for (int64_t c1 = 0; c1 < src1->ne[2]; c1++) {
            int64_t n0 = n1 / (src1->ne[3] / src0->ne[3]);
            int64_t c0 = c1 / (src1->ne[2] / src0->ne[2]);

            int64_t batch1 = (n1 * src1->ne[2]) + c1;
            int64_t batch0 = (n0 * src0->ne[2]) + c0;

            aclTensor* acl_input_tensor = ggml_cann_create_tensor(
                (char*)input_buffer + batch1 * input_stride, ACL_FLOAT16,
                input_elem_size, input_ne, input_nb, 2);

            // print_acltensor((char*)input_buffer + batch1 *
            // input_stride,input_ne,ACL_FLOAT16, "input_tensor",2);

            // first split
            int64_t weight_ne_offset = 0;
            int64_t weight_ne[2] = {
                max_elem_size > src0->ne[1] ? src0->ne[1] : max_elem_size,
                src0->ne[0]};
            int64_t scale_ne_offset = 0;
            int64_t scale_ne[2] = {weight_ne[0], weight_ne[1] / QK8_0};
            int64_t output_ne_offset = 0;
            int64_t output_ne[2] = {weight_ne[0], dst->ne[1]};

            aclTensor* acl_weight_tensor = ggml_cann_create_tensor(
                (char*)src0->data + batch0 * weight_stride,
                ggml_cann_type_mapping(type), weight_elem_size, weight_ne,
                weight_nb, 2, ACL_FORMAT_ND, weight_ne_offset);

            aclTensor* acl_scale_tensor = ggml_cann_create_tensor(
                scale_offset + batch0 * scale_stride, ACL_FLOAT16,
                scale_elem_size, scale_ne, scale_nb, 2, ACL_FORMAT_ND,
                scale_ne_offset);

            aclTensor* acl_output_tensor = ggml_cann_create_tensor(
                (char*)output_buffer + batch1 * output_stride, ACL_FLOAT16,
                output_elem_size, output_ne, output_nb, 2, ACL_FORMAT_ND,
                output_ne_offset);

            ACL_CHECK(aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
                acl_input_tensor, acl_weight_tensor, acl_scale_tensor, nullptr,
                nullptr, nullptr, nullptr, QK8_0, acl_output_tensor,
                &workspaceSize, &executor));
            if (workspaceAddr == nullptr) {
                workspaceAddr = workspace_allocator.alloc(workspaceSize);
            }
            ACL_CHECK(aclnnWeightQuantBatchMatmulV2(
                workspaceAddr, workspaceSize, executor, ctx.stream()));

            ACL_CHECK(aclDestroyTensor(acl_weight_tensor));
            ACL_CHECK(aclDestroyTensor(acl_scale_tensor));
            ACL_CHECK(aclDestroyTensor(acl_output_tensor));

            // other splits
            for (int64_t split = 1; split < split_size; split++) {
                weight_ne_offset +=
                    weight_elem_size * weight_ne[0] * weight_ne[1];
                weight_ne[0] = max_elem_size * (split + 1) > src0->ne[1]
                                   ? src0->ne[1] - (max_elem_size * split)
                                   : max_elem_size;
                scale_ne_offset += scale_elem_size * scale_ne[0] * scale_ne[1];
                scale_ne[0] = weight_ne[0];
                output_ne_offset +=
                    output_elem_size * output_ne[0] * output_ne[1];
                output_ne[0] = weight_ne[0];

                acl_weight_tensor = ggml_cann_create_tensor(
                    (char*)src0->data + batch0 * weight_stride,
                    ggml_cann_type_mapping(type), weight_elem_size, weight_ne,
                    weight_nb, 2, ACL_FORMAT_ND, weight_ne_offset);
                acl_scale_tensor = ggml_cann_create_tensor(
                    scale_offset + batch0 * scale_stride, ACL_FLOAT16,
                    scale_elem_size, scale_ne, scale_nb, 2, ACL_FORMAT_ND,
                    scale_ne_offset);
                acl_output_tensor = ggml_cann_create_tensor(
                    (char*)output_buffer + batch1 * output_stride, ACL_FLOAT16,
                    output_elem_size, output_ne, output_nb, 2, ACL_FORMAT_ND,
                    output_ne_offset);

                ACL_CHECK(aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
                    acl_input_tensor, acl_weight_tensor, acl_scale_tensor,
                    nullptr, nullptr, nullptr, nullptr, QK8_0,
                    acl_output_tensor, &workspaceSize, &executor));
                ACL_CHECK(aclnnWeightQuantBatchMatmulV2(
                    workspaceAddr, workspaceSize, executor, ctx.stream()));

                ACL_CHECK(aclDestroyTensor(acl_weight_tensor));
                ACL_CHECK(aclDestroyTensor(acl_scale_tensor));
                ACL_CHECK(aclDestroyTensor(acl_output_tensor));
            }

            ACL_CHECK(aclDestroyTensor(acl_input_tensor));
        }
    }

    // cast out
    if (dst->type != GGML_TYPE_F16) {
        int64_t* output_cast_ne = dst->ne;
        size_t output_cast_nb[GGML_MAX_DIMS];
        output_cast_nb[0] = sizeof(uint16_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            output_cast_nb[i] = output_cast_nb[i - 1] * output_cast_ne[i - 1];
        }

        aclTensor* acl_output_tensor = ggml_cann_create_tensor(
            output_buffer, ACL_FLOAT16, output_elem_size, output_cast_ne,
            output_cast_nb, GGML_MAX_DIMS);
        aclTensor* acl_dst_tensor = ggml_cann_create_tensor(dst);
        aclnn_cast(ctx, acl_output_tensor, acl_dst_tensor,
                   ggml_cann_type_mapping(dst->type));
        ACL_CHECK(aclDestroyTensor(acl_output_tensor));
        ACL_CHECK(aclDestroyTensor(acl_dst_tensor));
    }
}

void ggml_cann_mul_mat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    const enum ggml_type type = dst->src[0]->type;
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            ggml_cann_mat_mul_fp(ctx, dst);
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_quant(ctx, dst, type);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

/**
 * @brief Rolls the elements of a tensor along a specified dimension.
 *
 * This function rolls the elements of the source tensor `acl_src` by the
 * specified shifts `shifts` along the specified dimensions `dims`, and stores
 * the result in the destination tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose elements will be rolled.
 * @param acl_dst The destination tensor where the rolled elements will be
 * stored.
 * @param shifts An array specifying the number of positions by which elements
 * are shifted.
 * @param dims An array specifying the dimensions along which elements are
 * shifted.
 */
static void aclnn_roll(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                       aclTensor* acl_dst, int64_t* shifts, int64_t* dims) {
    aclIntArray* acl_shifts = aclCreateIntArray(shifts, 1);
    aclIntArray* acl_dims = aclCreateIntArray(dims, 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnRollGetWorkspaceSize(acl_src, acl_shifts, acl_dims, acl_dst,
                                        &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnRoll(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyIntArray(acl_shifts));
    ACL_CHECK(aclDestroyIntArray(acl_dims));
}

/**
 * @brief Fills specified positions of a tensor with a scalar value.
 *
 * This function fills the positions in the source tensor `acl_src` specified by
 * `index` along the dimension `dim` with the scalar value `value`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor where the positions will be filled.
 * @param dim The dimension along which the positions are specified.
 * @param index An array specifying the positions to be filled.
 * @param index_num The number of positions specified in the index array.
 * @param value The scalar value used to fill the specified positions.
 */
static void aclnn_index_fill_tensor(ggml_backend_cann_context& ctx,
                                    aclTensor* acl_src, int64_t dim,
                                    int64_t* index, int64_t index_num,
                                    float value) {
    aclIntArray* acl_index = aclCreateIntArray(index, index_num);
    aclScalar* acl_value = aclCreateScalar(&value, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceIndexFillTensorGetWorkspaceSize(
        acl_src, dim, acl_index, acl_value, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnInplaceIndexFillTensor(workspaceAddr, workspaceSize,
                                          executor, ctx.stream()));

    ACL_CHECK(aclDestroyIntArray(acl_index));
    ACL_CHECK(aclDestroyScalar(acl_value));
}

static void aclnn_cache_init(ggml_backend_cann_context& ctx, ggml_tensor* dst,
                             aclTensor* acl_cos_repeat_tensor,
                             aclTensor* acl_sin_repeat_tensor,
                             float theta_scale, float freq_scale,
                             float attn_factor, bool is_neox) {
    // int sin/cos cache, cache has different repeat method depond on
    // @param.is_neox

    ggml_tensor* src0 = dst->src[0];  // input
    ggml_tensor* src1 = dst->src[1];  // position
    ggml_tensor* src2 = dst->src[2];  // freq_factors

    // arange, [0,1,...,ne0/2]
    int64_t arange_length = src0->ne[0] / 2;
    ggml_cann_pool_alloc arange_allocator(ctx.pool(),
                                          arange_length * sizeof(float_t));
    void* arange_buffer = arange_allocator.get();
    int64_t arange_ne[] = {arange_length, 1, 1, 1};
    size_t arange_nb[] = {sizeof(float_t), sizeof(float_t), sizeof(float_t),
                          arange_length * sizeof(float_t)};

    aclTensor* acl_arange_tensor =
        ggml_cann_create_tensor(arange_buffer, ACL_FLOAT, sizeof(float_t),
                                arange_ne, arange_nb, GGML_MAX_DIMS);
    float start = 0;
    float step = 1;
    float stop = src0->ne[0] / 2;
    float n_elements = src0->ne[0] / 2;
    aclnn_arange(ctx, acl_arange_tensor, start, stop, step, n_elements);

    // power
    // aclnnPowScalarTensor(): @param self is tensor which should be scalar, so
    // use aclnn_pow_tensor_tensor() until fixed. aclScalar* acl_theta_scale =
    // aclCreateScalar(&theta_scale, aclDataType::ACL_FLOAT);
    // aclnn_power_scalar_tensor(ctx, acl_theta_scale, acl_arange_tensor,
    // acl_power_tensor);
    ggml_cann_pool_alloc theta_scale_allocator(ctx.pool(),
                                               arange_length * sizeof(float_t));
    void* theta_scale_buffer = theta_scale_allocator.get();
    aclTensor* acl_theta_scale_tensor = aclnn_values(
        ctx, theta_scale_buffer, arange_length * sizeof(float_t), arange_ne,
        GGML_MAX_DIMS, ACL_FLOAT, sizeof(float_t), theta_scale);
    aclnn_pow_tensor_tensor(ctx, acl_theta_scale_tensor, acl_arange_tensor);

    // freq_scale
    if (freq_scale != 1) {
        aclnn_muls(ctx, acl_theta_scale_tensor, freq_scale, nullptr, true);
    }

    // freq_factors
    if (src2) {
        aclTensor* acl_freq_factors_tensor = ggml_cann_create_tensor(
            src2->data, ggml_cann_type_mapping(src2->type),
            ggml_type_size(src2->type), arange_ne, arange_nb, GGML_MAX_DIMS);
        aclnn_div_tensor(ctx, acl_theta_scale_tensor, acl_freq_factors_tensor,
                         nullptr, true);
        ACL_CHECK(aclDestroyTensor(acl_freq_factors_tensor));
    }

    // position
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    int64_t position_length = src1->ne[0];
    int64_t position_ne[] = {1, position_length, 1, 1};
    size_t position_nb[] = {sizeof(int32_t), sizeof(int32_t),
                            sizeof(int32_t) * position_length,
                            sizeof(int32_t) * position_length};
    aclTensor* acl_position_tensor = ggml_cann_create_tensor(
        src1->data, ggml_cann_type_mapping(src1->type),
        ggml_type_size(src1->type), position_ne, position_nb, GGML_MAX_DIMS);

    // power * position
    int64_t theta_length = arange_length * position_length;
    ggml_cann_pool_alloc theta_allocator(ctx.pool(),
                                         theta_length * sizeof(float_t));
    void* theta_buffer = theta_allocator.get();
    int64_t theta_ne[] = {arange_length, position_length, 1, 1};
    size_t theta_nb[GGML_MAX_DIMS];
    theta_nb[0] = sizeof(float_t);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        theta_nb[i] = theta_nb[i - 1] * theta_ne[i - 1];
    }
    aclTensor* acl_theta_tensor =
        ggml_cann_create_tensor(theta_buffer, ACL_FLOAT, sizeof(float_t),
                                theta_ne, theta_nb, GGML_MAX_DIMS);
    aclnn_mul(ctx, acl_position_tensor, acl_theta_scale_tensor,
              acl_theta_tensor);

    // permute: [0,1,2,3]->[0,2,1,3]
    int64_t permute_ne[] = {arange_length, 1, position_length, 1};
    size_t permute_nb[GGML_MAX_DIMS];
    permute_nb[0] = sizeof(float_t);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        permute_nb[i] = permute_nb[i - 1] * permute_ne[i - 1];
    }
    ggml_cann_pool_alloc permute_allocator(ctx.pool(),
                                           theta_length * sizeof(float_t));
    void* permute_buffer = permute_allocator.get();
    aclTensor* acl_permute_tensor = ggml_cann_create_tensor(
        permute_buffer, ACL_FLOAT, sizeof(float_t), permute_ne, permute_nb,
        GGML_MAX_DIMS, ACL_FORMAT_ND);
    int64_t permute_dim[] = {0, 2, 1, 3};
    int64_t num_dims = 4;
    aclnn_permute(ctx, acl_theta_tensor, acl_permute_tensor, permute_dim,
                  num_dims);

    // sin/cos
    ggml_cann_pool_alloc sin_allocator(ctx.pool(),
                                       theta_length * sizeof(float_t));
    void* sin_buffer = sin_allocator.get();
    aclTensor* acl_sin_tensor = ggml_cann_create_tensor(
        sin_buffer, ACL_FLOAT, sizeof(float_t), permute_ne, permute_nb,
        GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_sin(ctx, acl_permute_tensor, acl_sin_tensor);

    ggml_cann_pool_alloc cos_allocator(ctx.pool(),
                                       theta_length * sizeof(float_t));
    void* cos_buffer = cos_allocator.get();
    aclTensor* acl_cos_tensor = ggml_cann_create_tensor(
        cos_buffer, ACL_FLOAT, sizeof(float_t), permute_ne, permute_nb,
        GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_cos(ctx, acl_permute_tensor, acl_cos_tensor);

    // attn_factor
    if (attn_factor != 1) {
        aclnn_muls(ctx, acl_sin_tensor, attn_factor, nullptr, true);
        aclnn_muls(ctx, acl_cos_tensor, attn_factor, nullptr, true);
    }

    // repeat
    if (is_neox) {
        int64_t repeatsArray[] = {1, 1, 1, 2};
        aclnn_repeat(ctx, acl_sin_tensor, acl_sin_repeat_tensor, repeatsArray);
        aclnn_repeat(ctx, acl_cos_tensor, acl_cos_repeat_tensor, repeatsArray);
    } else {
        int64_t num_repeats = 2;
        int64_t dim = 3;
        int64_t output_size = arange_length * num_repeats;
        aclnn_repeat_interleave(ctx, acl_sin_tensor, acl_sin_repeat_tensor, dim,
                                num_repeats, output_size);
        aclnn_repeat_interleave(ctx, acl_cos_tensor, acl_cos_repeat_tensor, dim,
                                num_repeats, output_size);
    }

    // release
    ACL_CHECK(aclDestroyTensor(acl_arange_tensor));
    ACL_CHECK(aclDestroyTensor(acl_theta_scale_tensor));
    ACL_CHECK(aclDestroyTensor(acl_position_tensor));
    ACL_CHECK(aclDestroyTensor(acl_theta_tensor));
    ACL_CHECK(aclDestroyTensor(acl_permute_tensor));
    ACL_CHECK(aclDestroyTensor(acl_sin_tensor));
    ACL_CHECK(aclDestroyTensor(acl_cos_tensor));
}

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnRotaryPositionEmbeddingGetWorkspaceSize(
    const aclTensor* x, const aclTensor* cos, const aclTensor* sin,
    int64_t mode, const aclTensor* yOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);
aclnnStatus aclnnRotaryPositionEmbedding(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);
#ifdef __cplusplus
}
#endif

void ggml_cann_rope(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    // TODO: use ascendc
    // Only test with LLAMA model.
    ggml_tensor* src0 = dst->src[0];  // input
    ggml_tensor* src2 = dst->src[2];  // freq_factors

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
    GGML_ASSERT(ext_factor == 0);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast,
                             beta_slow, corr_dims);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    // init cos/sin cache
    ggml_cann_pool_alloc sin_allocator(
        ctx.pool(), src0->ne[0] * src0->ne[2] * sizeof(float_t));
    ggml_cann_pool_alloc cos_allocator(
        ctx.pool(), src0->ne[0] * src0->ne[2] * sizeof(float_t));
    void* sin_buffer = sin_allocator.get();
    void* cos_buffer = cos_allocator.get();

    int64_t sin_reshape_ne[4] = {src0->ne[0], 1, src0->ne[2], 1};
    size_t sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = sizeof(float_t);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    aclTensor* acl_sin_reshape_tensor =
        ggml_cann_create_tensor(sin_buffer, ACL_FLOAT, sizeof(float_t),
                                sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    aclTensor* acl_cos_reshape_tensor =
        ggml_cann_create_tensor(cos_buffer, ACL_FLOAT, sizeof(float_t),
                                sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    aclnn_cache_init(ctx, dst, acl_cos_reshape_tensor, acl_sin_reshape_tensor,
                     theta_scale, freq_scale, attn_factor, is_neox);

    aclTensor* acl_src = ggml_cann_create_tensor(src0);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

#ifdef ASCEND_310P
    // Special ROPE operation for 310P

    // roll input
    void* input_roll_buffer;
    aclTensor* acl_minus_one_tensor;
    void* minus_one_scale_buffer = nullptr;
    ggml_cann_pool_alloc roll_allocator(ctx.pool(), ggml_nbytes(src0));
    ggml_cann_pool_alloc minus_one_scale_allocator(
        ctx.pool(), sizeof(float_t) * src0->ne[0]);
    if (!is_neox) {
        // roll input: [q0,q1,q2,q3,...] -> [q1,q0,q3,q2,...]
        input_roll_buffer = roll_allocator.get();
        int64_t input_roll_ne[4] = {2, src0->ne[1] * (src0->ne[0] / 2),
                                    src0->ne[2], src0->ne[3]};
        size_t input_roll_nb[GGML_MAX_DIMS];
        input_roll_nb[0] = ggml_type_size(src0->type);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            input_roll_nb[i] = input_roll_nb[i - 1] * input_roll_ne[i - 1];
        }
        aclTensor* acl_input_roll_tensor = ggml_cann_create_tensor(
            input_roll_buffer, ggml_cann_type_mapping(src0->type),
            ggml_type_size(src0->type), input_roll_ne, input_roll_nb,
            GGML_MAX_DIMS);
        aclTensor* acl_input_tensor = ggml_cann_create_tensor(
            src0->data, ggml_cann_type_mapping(src0->type),
            ggml_type_size(src0->type), input_roll_ne, input_roll_nb,
            GGML_MAX_DIMS);

        int64_t shifts[] = {1};
        int64_t dims[] = {3};
        aclnn_roll(ctx, acl_input_tensor, acl_input_roll_tensor, shifts, dims);
        ACL_CHECK(aclDestroyTensor(acl_input_roll_tensor));
        ACL_CHECK(aclDestroyTensor(acl_input_tensor));

        // init [-1, 1, -1, 1, ...]
        minus_one_scale_buffer = minus_one_scale_allocator.get();

        int64_t minus_one_ne[4] = {src0->ne[0], 1, 1, 1};
        size_t minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = sizeof(float_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor = aclnn_values(
            ctx, minus_one_scale_buffer, sizeof(float_t) * src0->ne[0],
            minus_one_ne, GGML_MAX_DIMS, ACL_FLOAT, sizeof(float_t), 1);
        int64_t dim = 3;
        int64_t* index = new int64_t[src0->ne[0]];
        for (int i = 0; i < src0->ne[0]; i++) {
            index[i] = i / 2 * 2;
        }
        int64_t index_num = src0->ne[0];
        float value = -1;
        aclnn_index_fill_tensor(ctx, acl_minus_one_tensor, dim, index,
                                index_num, value);
    } else {
        // roll input: [q0,q1,q2,...] ->
        // [q_half,q_half+1,...,q_end,q0,q1,...q_half-1]
        input_roll_buffer = roll_allocator.get();
        aclTensor* acl_input_roll_tensor = ggml_cann_create_tensor(
            input_roll_buffer, ggml_cann_type_mapping(src0->type),
            ggml_type_size(src0->type), src0->ne, src0->nb, GGML_MAX_DIMS);
        aclTensor* acl_input_tensor = ggml_cann_create_tensor(src0);

        int64_t shifts[] = {src0->ne[0] / 2};
        int64_t dims[] = {3};
        aclnn_roll(ctx, acl_input_tensor, acl_input_roll_tensor, shifts, dims);

        ACL_CHECK(aclDestroyTensor(acl_input_roll_tensor));
        ACL_CHECK(aclDestroyTensor(acl_input_tensor));
        // init [-1, -1, -1, 1, 1，1，...]
        minus_one_scale_buffer = minus_one_scale_allocator.get();
        int64_t minus_one_ne[4] = {src0->ne[0], 1, 1, 1};
        size_t minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = sizeof(float_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor = aclnn_values(
            ctx, minus_one_scale_buffer, sizeof(float_t) * src0->ne[0],
            minus_one_ne, GGML_MAX_DIMS, ACL_FLOAT, sizeof(float_t), 1);
        // -1 * first half
        int64_t first_half_ne[4] = {src0->ne[0] / 2, 1, 1, 1};
        size_t first_half_nb[GGML_MAX_DIMS];
        first_half_nb[0] = sizeof(float_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            first_half_nb[i] = first_half_nb[i - 1] * first_half_ne[i - 1];
        }
        aclTensor* acl_first_half_tensor = ggml_cann_create_tensor(
            minus_one_scale_buffer, ACL_FLOAT, sizeof(float_t), first_half_ne,
            first_half_nb, GGML_MAX_DIMS);
        bool inplace = true;
        float scale = -1;
        aclnn_muls(ctx, acl_first_half_tensor, scale, nullptr, inplace);
        ACL_CHECK(aclDestroyTensor(acl_first_half_tensor));
    }

    // TODO: n_dims < ne0
    GGML_ASSERT(n_dims == src0->ne[0]);

    // input * scale
    ggml_cann_pool_alloc roll_mul_scale_allocator(ctx.pool(),
                                                  ggml_nbytes(src0));
    void* input_roll_mul_scale_buffer = roll_mul_scale_allocator.get();
    size_t input_nb[GGML_MAX_DIMS];
    input_nb[0] = ggml_type_size(src0->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        input_nb[i] = input_nb[i - 1] * src0->ne[i - 1];
    }
    aclTensor* acl_input_roll_mul_scale_tensor = ggml_cann_create_tensor(
        input_roll_mul_scale_buffer, ggml_cann_type_mapping(src0->type),
        ggml_type_size(src0->type), src0->ne, input_nb, GGML_MAX_DIMS);
    aclTensor* acl_input_roll_reshape_tensor = ggml_cann_create_tensor(
        input_roll_buffer, ggml_cann_type_mapping(src0->type),
        ggml_type_size(src0->type), src0->ne, input_nb, GGML_MAX_DIMS);

    aclnn_mul(ctx, acl_input_roll_reshape_tensor, acl_minus_one_tensor,
              acl_input_roll_mul_scale_tensor);

    // output
    void* output_fp32_buffer;
    if (src0->type == GGML_TYPE_F32) {
        aclnn_inplace_mul(ctx, acl_src, acl_cos_reshape_tensor);
        aclnn_inplace_mul(ctx, acl_input_roll_mul_scale_tensor,
                          acl_sin_reshape_tensor);
        aclnn_add(ctx, acl_src, acl_input_roll_mul_scale_tensor, acl_dst);
        // TODO: ne0 != n_dims in mode2
    } else if (src0->type == GGML_TYPE_F16) {
        size_t input_fp32_nb[GGML_MAX_DIMS];
        input_fp32_nb[0] = sizeof(float_t);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            input_fp32_nb[i] = input_fp32_nb[i - 1] * dst->ne[i - 1];
        }
        ggml_cann_pool_alloc fp32_allocator1(
            ctx.pool(), ggml_nelements(dst) * sizeof(float_t));
        void* input_fp32_buffer1 = fp32_allocator1.get();
        aclTensor* input_fp32_tensor1 = ggml_cann_create_tensor(
            input_fp32_buffer1, ACL_FLOAT, sizeof(float_t), dst->ne,
            input_fp32_nb, GGML_MAX_DIMS);
        ggml_cann_pool_alloc fp32_allocator2(
            ctx.pool(), ggml_nelements(dst) * sizeof(float_t));
        void* input_fp32_buffer2 = fp32_allocator2.get();
        aclTensor* input_fp32_tensor2 = ggml_cann_create_tensor(
            input_fp32_buffer2, ACL_FLOAT, sizeof(float_t), dst->ne,
            input_fp32_nb, GGML_MAX_DIMS);

        ggml_cann_pool_alloc fp32_allocator(
            ctx.pool(), ggml_nelements(dst) * sizeof(float_t));
        output_fp32_buffer = fp32_allocator.get();
        aclTensor* output_fp32_tensor = ggml_cann_create_tensor(
            output_fp32_buffer, ACL_FLOAT, sizeof(float_t), dst->ne,
            input_fp32_nb, GGML_MAX_DIMS);
        aclnn_mul(ctx, acl_src, acl_cos_reshape_tensor, input_fp32_tensor1);
        aclnn_mul(ctx, acl_input_roll_mul_scale_tensor, acl_sin_reshape_tensor,
                  input_fp32_tensor2);
        aclnn_add(ctx, input_fp32_tensor1, input_fp32_tensor2,
                  output_fp32_tensor);
        aclnn_cast(ctx, output_fp32_tensor, acl_dst, ACL_FLOAT16);

        ACL_CHECK(aclDestroyTensor(input_fp32_tensor1));
        ACL_CHECK(aclDestroyTensor(input_fp32_tensor2));
        ACL_CHECK(aclDestroyTensor(output_fp32_tensor));
        ACL_CHECK(aclDestroyTensor(acl_sin_reshape_tensor));
        ACL_CHECK(aclDestroyTensor(acl_minus_one_tensor));
        ACL_CHECK(aclDestroyTensor(acl_input_roll_mul_scale_tensor));
        ACL_CHECK(aclDestroyTensor(acl_input_roll_reshape_tensor));
        ACL_CHECK(aclDestroyTensor(acl_src));
    }
    return;
#endif

    // src0 == GGML_TYPE_F16
    // TODO: optimization this `if` code
    if (src0->type == GGML_TYPE_F16) {
        ggml_cann_pool_alloc sin_final_allocator(
            ctx.pool(), src0->ne[0] * src0->ne[2] * ggml_type_size(src0->type));
        ggml_cann_pool_alloc cos_final_allocator(
            ctx.pool(), src0->ne[0] * src0->ne[2] * ggml_type_size(src0->type));
        void* sin_final_buffer = sin_final_allocator.get();
        void* cos_final_buffer = cos_final_allocator.get();

        int64_t sin_final_ne[4] = {src0->ne[0], 1, src0->ne[2], 1};
        size_t sin_final_nb[GGML_MAX_DIMS];
        sin_final_nb[0] = ggml_type_size(src0->type);
        for (int i = 1; i < GGML_MAX_DIMS; i++) {
            sin_final_nb[i] = sin_final_nb[i - 1] * sin_final_ne[i - 1];
        }
        aclTensor* acl_sin_final_tensor = ggml_cann_create_tensor(
            sin_final_buffer, ggml_cann_type_mapping(src0->type),
            ggml_type_size(src0->type), sin_final_ne, sin_final_nb,
            GGML_MAX_DIMS);
        aclTensor* acl_cos_final_tensor = ggml_cann_create_tensor(
            cos_final_buffer, ggml_cann_type_mapping(src0->type),
            ggml_type_size(src0->type), sin_final_ne, sin_final_nb,
            GGML_MAX_DIMS);

        aclnn_cast(ctx, acl_sin_reshape_tensor, acl_sin_final_tensor,
                   ggml_cann_type_mapping(src0->type));
        aclnn_cast(ctx, acl_cos_reshape_tensor, acl_cos_final_tensor,
                   ggml_cann_type_mapping(src0->type));
        ACL_CHECK(aclDestroyTensor(acl_cos_reshape_tensor));
        ACL_CHECK(aclDestroyTensor(acl_sin_reshape_tensor));
        acl_sin_reshape_tensor = acl_sin_final_tensor;
        acl_cos_reshape_tensor = acl_cos_final_tensor;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    void* workspaceAddr = nullptr;

    int acl_mode = mode;
    if (mode == 0) {
        acl_mode = 1;
    }

    ACL_CHECK(aclnnRotaryPositionEmbeddingGetWorkspaceSize(
        acl_src, acl_cos_reshape_tensor, acl_sin_reshape_tensor, acl_mode,
        acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnRotaryPositionEmbedding(workspaceAddr, workspaceSize,
                                           executor, ctx.stream()));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_cos_reshape_tensor));
    ACL_CHECK(aclDestroyTensor(acl_sin_reshape_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_flash_attn_prompt(ggml_backend_cann_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* query = dst->src[0];
    ggml_tensor* key = dst->src[1];
    ggml_tensor* value = dst->src[2];
    ggml_tensor* attn_mask = dst->src[3];
    GGML_ASSERT(query->type == GGML_TYPE_F16);
    GGML_ASSERT(key->type == GGML_TYPE_F16);
    GGML_ASSERT(value->type == GGML_TYPE_F16);
    GGML_ASSERT(attn_mask->type == GGML_TYPE_I8);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    struct {
        int batch_size;
        int num_heads;
        int head_dim_kq;
        int head_dim_v;
        int key_num_heads;
        int sequence_lenth_q;
        int64_t sequence_lenth_kv;
        float scaleValue;
    } params;

    memcpy(&params, dst->op_params, sizeof(params));
    int32_t batch_size = params.batch_size;
    int32_t num_heads = params.num_heads;
    int32_t head_dim_kq = params.head_dim_kq;
    int32_t head_dim_v = params.head_dim_v;
    int32_t key_num_heads = params.key_num_heads;
    int32_t sequence_lenth_q = params.sequence_lenth_q;
    int64_t sequence_lenth_kv = params.sequence_lenth_kv;
    float scaleValue = params.scaleValue;

    int64_t numKeyValueHeads = num_heads;
    std::string sLayerOut = "BSND";
    char layerOut[sLayerOut.length()];
    strcpy(layerOut, sLayerOut.c_str());
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 0;
    int64_t sparseMode = 1;  // 拦截，非量化情况不考虑
    int64_t innerPrecise =
        sequence_lenth_q > 1 ? 2 : 0;  // 高精度模式，开启行无效修正。

    aclTensor* acl_query_tensor = ggml_cann_create_tensor(query);
    aclTensor* acl_key_tensor = ggml_cann_create_tensor(key);
    aclTensor* acl_value_tensor = ggml_cann_create_tensor(value);
    aclTensor* acl_attn_mask_tensor = ggml_cann_create_tensor(attn_mask);
    aclTensor* acl_dst_tensor = ggml_cann_create_tensor(dst);

    aclTensor* acl_key_tensor_list[1] = {acl_key_tensor};
    aclTensorList* acl_key_tensors =
        aclCreateTensorList(acl_key_tensor_list, 1);
    aclTensor* acl_value_tensor_list[1] = {acl_value_tensor};
    aclTensorList* acl_value_tensors =
        aclCreateTensorList(acl_value_tensor_list, 1);

    std::vector<int64_t> actualSeqlenVectorq = {sequence_lenth_q};
    auto* actualSeqLengthsq = aclCreateIntArray(actualSeqlenVectorq.data(),
                                                actualSeqlenVectorq.size());

    std::vector<int64_t> actualSeqlenVectorkv = {sequence_lenth_kv};
    auto* actualSeqLengthskv = aclCreateIntArray(actualSeqlenVectorkv.data(),
                                                 actualSeqlenVectorkv.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    ACL_CHECK(aclnnFusedInferAttentionScoreGetWorkspaceSize(
        acl_query_tensor, acl_key_tensors, acl_value_tensors, nullptr,
        acl_attn_mask_tensor, actualSeqLengthsq, actualSeqLengthskv, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, num_heads, scaleValue, preTokens, nextTokens, layerOut,
        numKeyValueHeads, sparseMode, innerPrecise, 0, 0, false, acl_dst_tensor,
        nullptr, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(aclnnFusedInferAttentionScore(workspaceAddr, workspaceSize,
                                            executor, ctx.stream()));
    ACL_CHECK(aclDestroyTensor(acl_query_tensor));
    ACL_CHECK(aclDestroyTensor(acl_key_tensor));
    ACL_CHECK(aclDestroyTensor(acl_value_tensor));
    ACL_CHECK(aclDestroyTensor(acl_attn_mask_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst_tensor));
    ACL_CHECK(aclDestroyIntArray(actualSeqLengthsq));
    ACL_CHECK(aclDestroyIntArray(actualSeqLengthskv));
}

#ifdef LLAMA_JITTOR_OPS_SUPPORT
void ggml_cann_flash_attn_jittor_v1(ggml_backend_cann_context& ctx,
                                    ggml_tensor* dst) {
    ggml_tensor* query = dst->src[0];
    ggml_tensor* key = dst->src[1];
    ggml_tensor* value = dst->src[2];
    ggml_tensor* attn_mask = dst->src[3];
    GGML_ASSERT(query->type == GGML_TYPE_F16);
    GGML_ASSERT(key->type == GGML_TYPE_F16);
    GGML_ASSERT(value->type == GGML_TYPE_F16);
    GGML_ASSERT(attn_mask->type == GGML_TYPE_I8);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    struct {
        int batch_size;
        int num_heads;
        int head_dim_kq;
        int head_dim_v;
        int key_num_heads;
        int sequence_lenth_q;
        int64_t sequence_lenth_kv;
        float scaleValue;
    } params;

    memcpy(&params, dst->op_params, sizeof(params));
    int32_t batch_size = params.batch_size;
    int32_t num_heads = params.num_heads;
    int32_t head_dim_kq = params.head_dim_kq;
    int32_t head_dim_v = params.head_dim_v;
    int32_t key_num_heads = params.key_num_heads;
    int32_t sequence_lenth_q = params.sequence_lenth_q;
    int64_t sequence_lenth_kv = params.sequence_lenth_kv;
    float scaleValue = params.scaleValue;

    int64_t numKeyValueHeads = num_heads;
    std::string sLayerOut = "BNSD";
    char layerOut[sLayerOut.length()];
    strcpy(layerOut, sLayerOut.c_str());
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 0;
    int64_t sparseMode = 1;  // 拦截，非量化情况不考虑
    int64_t innerPrecise =
        sequence_lenth_q > 1 ? 2 : 0;  // 高精度模式，开启行无效修正。

    aclTensor* acl_query_tensor = ggml_cann_create_tensor(query);
    aclTensor* acl_key_tensor = ggml_cann_create_tensor(key);
    aclTensor* acl_value_tensor = ggml_cann_create_tensor(value);
    aclTensor* acl_attn_mask_tensor = ggml_cann_create_tensor(attn_mask);
    aclTensor* acl_dst_tensor = ggml_cann_create_tensor(dst);

    std::vector<int64_t> actualSeqlenVectorq = {sequence_lenth_q};
    auto* actualSeqLengthsq = aclCreateIntArray(actualSeqlenVectorq.data(),
                                                actualSeqlenVectorq.size());

    std::vector<int64_t> actualSeqlenVectorkv = {sequence_lenth_kv};
    auto* actualSeqLengthskv = aclCreateIntArray(actualSeqlenVectorkv.data(),
                                                 actualSeqlenVectorkv.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    ACL_CHECK(aclnnJittorInferFlashAttentionV4GetWorkspaceSize(
        acl_query_tensor, acl_key_tensor, acl_value_tensor, nullptr,
        acl_attn_mask_tensor, actualSeqLengthsq, actualSeqLengthskv, nullptr,
        nullptr, nullptr, nullptr, nullptr, num_heads, scaleValue, preTokens,
        nextTokens, layerOut, numKeyValueHeads, sparseMode, innerPrecise,
        acl_dst_tensor, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(aclnnJittorInferFlashAttentionV4(workspaceAddr, workspaceSize,
                                               executor, ctx.stream()));
    ACL_CHECK(aclDestroyTensor(acl_query_tensor));
    ACL_CHECK(aclDestroyTensor(acl_key_tensor));
    ACL_CHECK(aclDestroyTensor(acl_value_tensor));
    ACL_CHECK(aclDestroyTensor(acl_attn_mask_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst_tensor));
    ACL_CHECK(aclDestroyIntArray(actualSeqLengthsq));
    ACL_CHECK(aclDestroyIntArray(actualSeqLengthskv));
}
#endif

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)
int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
              return ret);
    return 0;
}
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                    const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclBuffer(const std::vector<T>& hostData,
                    const std::vector<int64_t>& shape, void** deviceAddr) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              return ret);

    return 0;
}

template <typename T>
int CreateAclTensorFromBuffer(const std::vector<int64_t>& shape,
                              void** deviceAddr, aclDataType dataType,
                              aclTensor** tensor) {
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
}

void convert_vector_fp32_to_fp16(const std::vector<float>& src,
                                 std::vector<float16_t>& dst) {
    dst.resize(src.size());
    for (uint64_t i = 0; i < src.size(); i++) {
        dst[i] = float16_t(src[i]);
    }
}

void convert_vector_fp16_to_fp32(const std::vector<float16_t>& src,
                                 std::vector<float>& dst) {
    dst.resize(src.size());
    for (uint64_t i = 0; i < src.size(); i++) {
        dst[i] = float(src[i]);
    }
}

int ggml_cann_prompt_flash_attention(
    const std::vector<float>& query_host, const std::vector<float>& key_host,
    const std::vector<float>& value_host, const std::vector<int8_t>& attn_mask,
    std::vector<float>& output_host, int64_t batch_size, int64_t num_heads,
    int64_t head_dim_kq, int64_t head_dim_v, int64_t key_num_heads,
    int64_t sequence_lenth_q, int64_t sequence_lenth_kv, float32_t scaleValue) {
    int64_t numKeyValueHeads = num_heads;
    std::string sLayerOut = "BSND";
    char layerOut[sLayerOut.length()];
    strcpy(layerOut, sLayerOut.c_str());
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 0;
    int64_t sparseMode = 1;    // 拦截，非量化情况不考虑
    int64_t innerPrecise = 2;  // 高精度模式，开启行无效修正。

    std::vector<float16_t> query_host_fp16;
    std::vector<float16_t> key_host_fp16;
    std::vector<float16_t> value_host_fp16;
    std::vector<float16_t> output_host_fp16;

    convert_vector_fp32_to_fp16(query_host, query_host_fp16);
    convert_vector_fp32_to_fp16(key_host, key_host_fp16);
    convert_vector_fp32_to_fp16(value_host, value_host_fp16);
    convert_vector_fp32_to_fp16(output_host, output_host_fp16);

    int deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
              return ret);

    std::vector<int64_t> queryShape = {batch_size, sequence_lenth_q, num_heads,
                                       head_dim_kq};
    std::vector<int64_t> keyShape = {batch_size, sequence_lenth_kv,
                                     key_num_heads, head_dim_kq};
    std::vector<int64_t> valueShape = {batch_size, sequence_lenth_kv,
                                       key_num_heads, head_dim_v};
    std::vector<int64_t> attenShape = {batch_size, 1, sequence_lenth_q,
                                       sequence_lenth_kv};
    std::vector<int64_t> outputShape = {batch_size, sequence_lenth_q, num_heads,
                                        head_dim_v};
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* attenDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* attenTensor = nullptr;
    aclTensor* outTensor = nullptr;
    ret = CreateAclTensor(query_host_fp16, queryShape, &queryDeviceAddr,
                          aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(key_host_fp16, keyShape, &keyDeviceAddr,
                          aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    const int kvTensorNum = 1;
    aclTensor* tensorsOfKey[kvTensorNum];
    tensorsOfKey[0] = keyTensor;
    auto* tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);

    ret = CreateAclTensor(value_host_fp16, valueShape, &valueDeviceAddr,
                          aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    const int valueTensorNum = 1;
    aclTensor* tensorsOfValue[valueTensorNum];
    tensorsOfValue[0] = valueTensor;
    auto* tensorValueList = aclCreateTensorList(tensorsOfValue, valueTensorNum);

    ret = CreateAclTensor(attn_mask, attenShape, &attenDeviceAddr,
                          aclDataType::ACL_INT8, &attenTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(output_host_fp16, outputShape, &outDeviceAddr,
                          aclDataType::ACL_FLOAT16, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> actualSeqlenVectorq = {sequence_lenth_q};
    auto* actualSeqLengthsq = aclCreateIntArray(actualSeqlenVectorq.data(),
                                                actualSeqlenVectorq.size());

    std::vector<int64_t> actualSeqlenVectorkv = {sequence_lenth_kv};
    auto* actualSeqLengthskv = aclCreateIntArray(actualSeqlenVectorkv.data(),
                                                 actualSeqlenVectorkv.size());

    ret = aclrtSynchronizeStream(stream);
    auto start = std::chrono::high_resolution_clock::now();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ACL_CHECK(aclnnFusedInferAttentionScoreGetWorkspaceSize(
        queryTensor, tensorKeyList, tensorValueList, nullptr, attenTensor,
        actualSeqLengthsq, actualSeqLengthskv, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,  // 量化相关
        nullptr, nullptr, nullptr,  // blockTable, qpaddingsize, kvpaddingsize
        num_heads, scaleValue, preTokens, nextTokens, layerOut,
        numKeyValueHeads, sparseMode, innerPrecise,
        0,      // blockSIze
        0,      // antiquantMode
        false,  // softmaxLseFlag
        outTensor, nullptr, &workspaceSize, &executor));
    // ret = aclnnPromptFlashAttentionGetWorkspaceSize(
    //     queryTensor, keyTensor, valueTensor,
    //     nullptr, attenTensor, actualSeqLengths,
    //     num_heads, scaleValue, preTokens, nextTokens,
    //     layerOut, numKeyValueHeads,
    //     outTensor, &workspaceSize, &executor);

    // CHECK_RET(ret == ACL_SUCCESS,
    // LOG_PRINT("aclnnPromptFlashAttentionGetWorkspaceSize failed. ERROR:
    // %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize,
                          ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  return ret);
    }

    ret = aclnnFusedInferAttentionScore(workspaceAddr, workspaceSize, executor,
                                        stream);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("aclnnFusedInferAttentionScore failed. ERROR: %d\n", ret);
        return ret);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "aclnnFusedInferAttentionScore time: " << duration.count()
              << " ms" << std::endl;

    ACL_CHECK(aclrtSynchronizeStream(stream));

    auto size = GetShapeSize(outputShape);
    ret = aclrtMemcpy(output_host_fp16.data(), size * sizeof(float16_t),
                      outDeviceAddr, size * sizeof(float16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret);

    convert_vector_fp16_to_fp32(output_host_fp16, output_host);

    // 6. 释放资源
    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(attenTensor);
    aclDestroyTensor(outTensor);
    aclDestroyIntArray(actualSeqLengthsq);
    aclDestroyIntArray(actualSeqLengthskv);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(attenDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

void ggml_cann_get_slice(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    int64_t params[3];
    memcpy(params, dst->op_params, sizeof(params));
    const int64_t fr = params[0];
    const int64_t to = params[1];
    const int64_t axis = params[2];
    const int64_t off = fr * src0->nb[axis];
    const int64_t ori_ne = src0->ne[axis];
    void* ori_data = src0->data;
    src0->ne[axis] = to - fr;
    src0->data = (void*)((char*)ori_data + off);
    ggml_cann_dup(ctx, dst);
    src0->ne[axis] = ori_ne;
    src0->data = ori_data;
}

static void aclnn_sub(ggml_backend_cann_context& ctx, aclTensor* acl_a_tensor,
                      aclTensor* acl_b_tensor, aclTensor* acl_dst_tensor) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    int64_t alphaValue = 1;
    aclScalar* alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_INT64);

    ACL_CHECK(aclnnSubGetWorkspaceSize(acl_a_tensor, acl_b_tensor, alpha,
                                       acl_dst_tensor, &workspaceSize,
                                       &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(aclnnSub(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyScalar(alpha));
}

static void aclnn_muls_inplace(ggml_backend_cann_context& ctx,
                               aclTensor* acl_a_tensor, int64_t scaleValue) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclScalar* scale = aclCreateScalar(&scaleValue, aclDataType::ACL_INT64);

    ACL_CHECK(aclnnInplaceMulsGetWorkspaceSize(acl_a_tensor, scale,
                                               &workspaceSize, &executor));

    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }

    ACL_CHECK(
        aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyScalar(scale));
}

void ggml_cann_scatter_update(ggml_backend_cann_context& ctx,
                              ggml_tensor* dst) {
    // 不依赖外部参数的实现
    ggml_tensor* indices = dst->src[1];
    ggml_tensor* updates = dst->src[2];

    GGML_ASSERT(dst->ne[2] == dst->ne[3] == 1);
    GGML_ASSERT(updates->ne[2] == updates->ne[3] == 1);
    GGML_ASSERT(indices->ne[1] == indices->ne[2] == indices->ne[3] == 1);
    GGML_ASSERT(indices->ne[0] == updates->ne[1]);
    GGML_ASSERT(updates->ne[0] == dst->ne[0]);

    aclTensor* acl_indices_tensor_ori =
        ggml_cann_create_tensor(indices, nullptr, nullptr, 1, ACL_FORMAT_ND, 0);
    aclTensor* acl_updates_tensor =
        ggml_cann_create_tensor(updates, nullptr, nullptr, 2, ACL_FORMAT_ND, 0);
    aclTensor* acl_dst_tensor =
        ggml_cann_create_tensor(dst, nullptr, nullptr, 2, ACL_FORMAT_ND, 0);

    // arange, [0,1,...,ne0]
    int64_t arange_length = indices->ne[0];
    ggml_cann_pool_alloc arange_allocator(ctx.pool(),
                                          arange_length * sizeof(int64_t));
    void* arange_buffer = arange_allocator.get();
    int64_t arange_ne[] = {arange_length, 1, 1, 1};
    size_t arange_nb[] = {sizeof(int64_t), sizeof(int64_t), sizeof(int64_t),
                          arange_length * sizeof(int64_t)};

    aclTensor* acl_arange_tensor = ggml_cann_create_tensor(
        arange_buffer, ACL_INT64, sizeof(int64_t), arange_ne, arange_nb, 1);

    // copyresult of indices
    ggml_cann_pool_alloc indices_allocator(ctx.pool(),
                                           arange_length * sizeof(int64_t));
    void* indices_buffer = indices_allocator.get();
    aclTensor* acl_indices_tensor =
        ggml_cann_create_tensor(indices_buffer, ACL_INT64, sizeof(int64_t),
                                indices->ne, indices->nb, 1);

    int64_t start = 0;
    int64_t step = 1;
    int64_t stop = arange_length;

    // print_device_tensor_elements(ctx, indices->data, "indices", indices->ne,
    //     indices->type,ggml_nelements(indices), 10, ctx.stream(), false, true,
    //     true);

    // 这里存在一个bug，对于第i位，实际偏移量是 i * ne[0] +
    // indices[i]，但是我们的预期是 indices[i] * ne[0]
    // 为此，我们需要进行如下步骤，以实现预期的偏移量
    aclnn_arange_int(ctx, acl_arange_tensor, start, stop, step);
    aclnn_sub(ctx, acl_indices_tensor_ori, acl_arange_tensor,
              acl_indices_tensor);
    aclnn_muls_inplace(ctx, acl_indices_tensor, dst->ne[0]);

    // print_device_tensor_elements(ctx, indices->data, "indices", indices->ne,
    //     indices->type,ggml_nelements(indices), 10, ctx.stream(), false, true,
    //     true);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceScatterUpdateGetWorkspaceSize(
        acl_dst_tensor, acl_indices_tensor, acl_updates_tensor, 1,
        &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ggml_cann_pool_alloc workspace_allocator(ctx.pool(), workspaceSize);
        workspaceAddr = workspace_allocator.get();
    }
    ACL_CHECK(aclnnInplaceScatterUpdate(workspaceAddr, workspaceSize, executor,
                                        ctx.stream()));
    ACL_CHECK(aclDestroyTensor(acl_indices_tensor));
    ACL_CHECK(aclDestroyTensor(acl_updates_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst_tensor));
    ACL_CHECK(aclDestroyTensor(acl_arange_tensor));
    ACL_CHECK(aclDestroyTensor(acl_indices_tensor_ori));
}
