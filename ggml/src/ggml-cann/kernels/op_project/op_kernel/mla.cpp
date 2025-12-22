/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include <type_traits>

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "mlapo.utils/common.h"
#include "mlapo.utils/common_func.h"
#include "mlapo.utils/fa_common.cpp"
#include "mlapo.utils/iterator.h"
#include "mlapo.utils/mma.h"
#include "mlapo.utils/simd.h"
#include "mlapo.utils/utils.h"

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

// define common const value

// FFTS Flag
constexpr int32_t QK_READY = 0;
constexpr int32_t SOFTMAX_READY = 1;
constexpr int32_t UPDATE_READY = 2;
constexpr int32_t QK_READY_DECODER = 3;
constexpr int32_t SOFTMAX_READY_DECODER = 4;
constexpr int32_t UPDATE_READY_DECODER = 5;

constexpr int32_t BLOCK_SIZE_32 = 32;
constexpr int32_t BLOCK_SIZE_8 = 8;
constexpr int64_t TMP_SIZE = 65536;            // 256 * 256
constexpr int64_t P_TMP_SIZE = 128 * 512 * 2;  // 256 * 256

constexpr int32_t TILING_BATCH = 0;
constexpr int32_t TILING_NUMHEADS = 1;
constexpr int32_t TILING_HEADDIM = 2;
constexpr int32_t TILING_NUMBLOKS = 3;
constexpr int32_t TILING_BLOCKSIZE = 4;
constexpr int32_t TILING_MAXBLOCKS = 5;
constexpr int32_t TILING_TOR = 6;
constexpr int32_t TILING_KVHEADS = 7;
constexpr int32_t TILING_HEADSIZE = 8;
constexpr int32_t TILING_PARASIZE = 9;
constexpr int32_t TILING_MTP_HEAD_SPLIT_SIZE = 10;
constexpr int32_t TILING_TOTAL_BLOCK_NUM = 11;
constexpr int32_t TILING_MASK_TYPE_ND = 12;
constexpr int32_t TILING_TASK_NUM = 13;
constexpr int32_t TILING_MAX_KVSEQLEN = 14;
constexpr int32_t TILING_KVCORENUM = 15;
constexpr int32_t TILING_DECODINGNUM = 16;

constexpr int32_t KV_FLOW_NUM = 4;
constexpr int32_t K_BLOCK_NUM = 4;

template <typename T>
using GlobalT = AscendC::GlobalTensor<T>;

template <typename T>
using LocalT = AscendC::LocalTensor<T>;

enum class TilingKeyType {
    TILING_HALF_DATA = 0,
    TILING_BF16_DATA = 1,
    TILING_INT8_DATA = 2
};

enum class InputFormat { ND_FORMAT = 0, NZ_FORMAT = 1 };

enum class BlockStack { ONE_FLOW = 0, FOUR_FLOW = 1, FIVE_FLOW = 2 };

template <TilingKeyType tilingKeyType>
struct AttentionType {};

template <>
struct AttentionType<TilingKeyType::TILING_HALF_DATA> {
    using mm1OutputType = float;
    using mm1CopyType = float;
    using mmBiasType = float;
    using mmScaleType = float;
    using mm2OutputType = float;
    using mm2CopyType = float;
};

template <>
struct AttentionType<TilingKeyType::TILING_BF16_DATA> {
    using mm1OutputType = float;
    using mm1CopyType = float;
    using mmBiasType = float;
    using mmScaleType = float;
    using mm2OutputType = float;
    using mm2CopyType = float;
};

template <>
struct AttentionType<TilingKeyType::TILING_INT8_DATA> {
    using mm1OutputType = int32_t;
    using mm1CopyType = int32_t;
    using mmBiasType = float;
    using mmScaleType = float;
    using mm2OutputType = int32_t;
    using mm2CopyType = int32_t;
};

#ifdef __DAV_C220_CUBE__
constexpr int32_t L0AB_UINT8_BUF_SIZE = 16384 * 2;
constexpr int32_t L0C_FLOAT_BUF_SIZE = 16384;
constexpr int32_t CUBE_MATRIX_SIZE = 256;  // 16 * 16
constexpr int32_t L1_P_UINT8_BUF_SIZE = 32768;
constexpr int32_t SP_BLOCK_SIZE = 16384;
constexpr int32_t TMP_SIZE_DECODER = 32768;

constexpr int32_t L1_UINT8_BUF_SIZE_DECODER = 16384 * 2;
constexpr int32_t L1_KV_HALF_BUF_SIZE = 73728;  // 2* 128 * 256
constexpr int32_t L1_KV_UINT8_BUF_SIZE = 73728 * 2;
constexpr uint64_t L1_E_UINT8_SIZE = 1024;      // 32 * 32 * 1B
constexpr uint64_t L1_SCALE_UINT8_SIZE = 4096;  // uint64 256 * 8 * 2head
constexpr uint64_t L1_SCALE_UINT64_SIZE = L1_SCALE_UINT8_SIZE / 8;
constexpr uint64_t L1_OFFSET_UINT8_SIZE = 2048;  // int32 256 * 4 8 2head
constexpr uint64_t L1_OFFSET_INT32_SIZE = L1_OFFSET_UINT8_SIZE / 4;
constexpr uint32_t CUBE_MATRIX_SIZE_512 = 16 * 32;  // 16 * 23

#elif __DAV_C220_VEC__
constexpr uint32_t HALF_VECTOR_SIZE = 128;
constexpr int64_t UB_UINT8_BLOCK_SIZE_MLA =
    16384;  // 96 * 128 * 2B // prefill/decoder diff
constexpr int64_t UB_UINT8_LINE_SIZE = 512;  // 64 * 4B，申请两倍空间防踩踏。
constexpr int64_t UB_FLOAT_LINE_SIZE = 64;  // 64，申请两倍空间防踩踏。

constexpr int32_t TMP_SIZE_DECODER = 32768;
constexpr int32_t STAGE2_UB_UINT8_BLOCK_SIZE = 8192;
constexpr int32_t CUBE_MATRIX_SIZE = 256;
constexpr int32_t MASK_COLUMNS = 128;
constexpr int32_t SP_BLOCK_SIZE = 16384;

__aicore__ __attribute__((always_inline)) void inline __set_mask(int32_t len) {
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % FLOAT_VECTOR_SIZE;
    for (int64_t i = 0; i < temp; i++) {
        mask |= one << i;
    }

    if (len == VECTOR_SIZE) {
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else if (len >= FLOAT_VECTOR_SIZE) {
        SetVectorMask<int8_t>(mask, (uint64_t)-1);
    } else {
        SetVectorMask<int8_t>(0x0, mask);
    }
}

template <BlockStack blockStack>
struct UbufAlloc {};

template <>
struct UbufAlloc<BlockStack::ONE_FLOW> {
    static constexpr uint32_t ls32_ubuf_offset = 0;
    static constexpr uint32_t ls32_quant_ubuf_offset =
        2 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lp_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t mask32_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lp32_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t mask_ubuf_offset = 3 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lo_ubuf_ping_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lo_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t ls16_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t hm32_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 1 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t pm32_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 2 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t pm32_ubuf_stage2_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 3 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t descale1_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 4 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t descale2_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 5 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t dm32_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 6 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t dm32_ubuf_stage2_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 7 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t ll_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 9 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t ll_ubuf_stage2_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 11 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t gm32_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 13 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t gl_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 15 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t gl32_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 15 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t p_scale_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 17 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t go_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t go32_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lo_ubuf_pong_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t tv32_ubuf_offset = 10 * UB_UINT8_BLOCK_SIZE_MLA;
};

template <>
struct UbufAlloc<BlockStack::FOUR_FLOW> {
    static constexpr uint32_t ls32_ubuf_offset = 0;
    static constexpr uint32_t ls32_quant_ubuf_offset =
        2 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lp_ubuf_offset = 0 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lp32_ubuf_offset = 0 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t mask_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lo_ubuf_ping_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lo_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t mask32_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t ls16_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t hm32_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 1 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t pm32_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 2 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t pm32_ubuf_stage2_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 3 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t descale1_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 4 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t descale2_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 5 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t dm32_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 6 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t dm32_ubuf_stage2_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 7 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t ll_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 10 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t ll_ubuf_stage2_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 12 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t gm32_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 14 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t gl_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 16 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t gl32_ubuf_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA + 16 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t go_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t go32_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lo_ubuf_pong_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t tv32_ubuf_offset = 10 * UB_UINT8_BLOCK_SIZE_MLA;
};

template <>
struct UbufAlloc<BlockStack::FIVE_FLOW> {
    static constexpr uint32_t ls32_ubuf_offset = 0;
    static constexpr uint32_t ls32_quant_ubuf_offset = 0;
    static constexpr uint32_t lp_ubuf_offset = 0 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lp32_ubuf_offset = 0 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t mask_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lo_ubuf_ping_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lo_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t mask32_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t ls16_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t lm32_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t hm32_ubuf_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + 1 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t pm32_ubuf_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + 2 * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t pm32_ubuf_stage2_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + 3 * UB_UINT8_LINE_SIZE;  // unused
    static constexpr uint32_t descale1_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + (4 + 2) * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t descale2_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + (5 + 2) * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t dm32_ubuf_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + (6 + 2) * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t dm32_ubuf_stage2_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + (7 + 2) * UB_UINT8_LINE_SIZE;  // unused
    static constexpr uint32_t ll_ubuf_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + (10 + 2) * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t ll_ubuf_stage2_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + (12 + 2) * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t gm32_ubuf_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + (14 + 2) * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t gl_ubuf_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + (16 + 2) * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t gl32_ubuf_offset =
        8 * UB_UINT8_BLOCK_SIZE_MLA + (16 + 2) * UB_UINT8_LINE_SIZE;
    static constexpr uint32_t go_ubuf_offset = 0 * UB_UINT8_BLOCK_SIZE_MLA;
    static constexpr uint32_t go32_ubuf_offset =
        0 * UB_UINT8_BLOCK_SIZE_MLA;  // UB_UINT8_BLOCK_SIZE_MLA = 16384;
    static constexpr uint32_t lo_ubuf_pong_offset =
        6 * UB_UINT8_BLOCK_SIZE_MLA;  //
    static constexpr uint32_t tv32_ubuf_offset =
        10 * UB_UINT8_BLOCK_SIZE_MLA;  // 16384 * 10 ~ 11
};

#endif

#ifdef __DAV_C220_CUBE__
template <TilingKeyType tilingKeyType = TilingKeyType::TILING_HALF_DATA,
          typename IN_DTYPE = half, typename OUT_DTYPE = half,
          typename IN_KVDTYPE = half,
          InputFormat KInputType = InputFormat::ND_FORMAT,
          bool flashDecoding = false>
class MLAttentionOp_C {
    // define dtype
    using IN_ROPE_DTYPE = IN_KVDTYPE;
    using mm1OutputType = typename AttentionType<tilingKeyType>::mm1OutputType;
    using mm1CopyType = typename AttentionType<tilingKeyType>::mm1CopyType;
    using mmBiasType = typename AttentionType<tilingKeyType>::mmBiasType;
    using mmScaleType = typename AttentionType<tilingKeyType>::mmScaleType;
    using mm2OutputType = typename AttentionType<tilingKeyType>::mm2OutputType;
    using mm2CopyType = typename AttentionType<tilingKeyType>::mm2CopyType;
    static constexpr uint32_t T_CUBE_MATRIX_SIZE =
        CUBE_MATRIX_SIZE_512 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_SIZE = BLOCK_SIZE_32 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_OFFSET = 2 / sizeof(IN_DTYPE);

   public:
    __aicore__ __attribute__((always_inline)) inline MLAttentionOp_C() {}

    __aicore__ __attribute__((always_inline)) inline void Init(
        __gm__ uint8_t *__restrict__ qNope, __gm__ uint8_t *__restrict__ qRope,
        __gm__ uint8_t *__restrict__ ctKV, __gm__ uint8_t *__restrict__ kRope,
        __gm__ uint8_t *__restrict__ blockTables,
        __gm__ uint8_t *__restrict__ workspace,
        __gm__ const MLATilingData *__restrict__ tiling) {
        SetPadding<uint64_t>(0);
        SetAtomicnone();
        SetNdpara(1, 0, 0);
        SetMasknorm();

        q_gm = reinterpret_cast<__gm__ IN_DTYPE *>(qNope);
        q_rope_gm = reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(qRope);
        k_gm = reinterpret_cast<__gm__ IN_KVDTYPE *>(ctKV);
        k_rope_gm = reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(kRope);
        block_tables_gm = reinterpret_cast<__gm__ int32_t *>(blockTables);
        tiling_gm = const_cast<__gm__ MLATilingData *>(tiling);

        q_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(qNope));
        q_rope_gm_tensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(q_rope_gm));
        k_gm_tensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ IN_KVDTYPE *>(ctKV));
        k_rope_gm_tensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(k_rope_gm));

        __gm__ const MLAHeadParams *__restrict headparams =
            &(tiling_gm->MLAHead);
        __gm__ const NdMLATiling *__restrict ndtiling =
            &(tiling_gm->ndMLATiling);

        num_batches = (uint32_t)(headparams->batch);
        q_heads = (uint32_t)(headparams->numHeads);
        embedding_size = (uint32_t)(headparams->headDim);
        block_size = (uint32_t)(headparams->blockSize);
        max_num_blocks_per_query = (uint32_t)(headparams->maxBlocks);
        kv_heads = (uint32_t)(headparams->kvHeads);
        tiling_head_size = (uint32_t)(headparams->headSize);
        tiling_para_size = (uint32_t)(headparams->paraSize);
        cur_qn_blk_size = (uint32_t)(ndtiling->curQNBlockTile);
        mask_type = (uint32_t)(headparams->maskTypeND);
        totalTaskNum = (uint32_t)(headparams->taskNum);
        maxKVSeqLen = (uint32_t)(ndtiling->maxKVseqlen);
        kv_split_core_num = (uint32_t)(headparams->kvSplitNum);
        flashDecodingTaskNum = (uint32_t)(headparams->splitTaskNum);
        block_dim = (uint32_t)(headparams->blockDim);

        bool mtpTp1Flag = (tiling_para_size == (uint32_t)(7));
        const uint64_t basicWorkSpaceHalf = block_dim * TMP_SIZE * 2;
        const uint64_t basicWorkSpaceFloat = block_dim * TMP_SIZE * 4;
        const uint64_t sWorkSpaceSize =
            mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        const uint64_t pWorkSpaceSize =
            mtpTp1Flag ? basicWorkSpaceHalf * 4 : basicWorkSpaceHalf * 2;
        const uint64_t oTempWorkSpaceSize =
            mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        const uint64_t defaultSysWorkspaceSize910B = 16U * 1024U * 1024U;

        __gm__ mm1CopyType *s_out_gm =
            reinterpret_cast<__gm__ mm1CopyType *>(workspace);
        __gm__ IN_DTYPE *p_out_gm = reinterpret_cast<__gm__ IN_DTYPE *>(
            workspace + sWorkSpaceSize + 512);
        __gm__ mm2CopyType *o_tmp_gm = reinterpret_cast<__gm__ mm2CopyType *>(
            workspace + sWorkSpaceSize + 512 + pWorkSpaceSize);
        s_gm_tensor.SetGlobalBuffer(s_out_gm, sWorkSpaceSize);
        p_gm_tensor.SetGlobalBuffer(p_out_gm, pWorkSpaceSize);
        o_tmp_gm_tensor.SetGlobalBuffer(o_tmp_gm, oTempWorkSpaceSize);

        l1q_buf_addr_tensor =
            buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(l1q_buf_addr_offset);
        l1kv_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(
            l1kv_buf_addr_offset);
        l1p_buf_addr_tensor =
            buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(l1p_buf_addr_offset);

        stride_kv = static_cast<uint64_t>(kv_heads) * 512;
        stride_kv_rope = static_cast<uint64_t>(kv_heads) * 64;
        __v = embedding_size;
        round_v = RoundUp<BLOCK_SIZE>(__v);
    }

    __aicore__ __attribute__((always_inline)) inline void Run() {
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(M, MTE1, EVENT_ID2);
        SET_FLAG(M, MTE1, EVENT_ID3);
        SET_FLAG(M, MTE1, EVENT_ID4);
        SET_FLAG(M, MTE1, EVENT_ID5);
        SET_FLAG(M, MTE1, EVENT_ID6);
        SET_FLAG(M, MTE1, EVENT_ID7);
        SET_FLAG(FIX, M, EVENT_ID0);
        SET_FLAG(FIX, M, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(MTE1, MTE2, EVENT_ID4);
        SET_FLAG(MTE1, MTE2, EVENT_ID5);
        SET_FLAG(MTE1, MTE2, EVENT_ID6);
        SET_FLAG(MTE1, MTE2, EVENT_ID7);
        SET_FLAG(FIX, MTE1, EVENT_ID0);
        SET_FLAG(FIX, MTE1, EVENT_ID1);
        SET_FLAG(FIX, MTE1, EVENT_ID2);
        SET_FLAG(FIX, MTE1, EVENT_ID3);
        SET_FLAG(FIX, MTE1, EVENT_ID4);
        SET_FLAG(FIX, MTE1, EVENT_ID5);
        SET_FLAG(MTE2, FIX, EVENT_ID0);

        uint64_t cur_batch = 0;
        uint32_t q_block_num_per_batch =
            (q_heads + cur_qn_blk_size - 1) / cur_qn_blk_size;
        uint32_t process_num = q_block_num_per_batch * num_batches;

        __gm__ const NdMLATiling *__restrict ndtiling =
            &(tiling_gm->ndMLATiling);
        for (uint32_t process = block_idx; process < process_num;
             process += (uint32_t)block_num) {  // for task

            cur_batch = process / q_block_num_per_batch;
            if (cur_batch >= num_batches) break;
            uint32_t offset_tiling =
                tiling_head_size + tiling_para_size * cur_batch;
            uint32_t start_core_idx =
                (cur_batch * q_block_num_per_batch) % block_num;
            uint32_t q_seqlen = (uint32_t)(ndtiling->qSeqLen[cur_batch]);
            uint32_t kv_seqlen = (uint32_t)(ndtiling->kvSeqlen[cur_batch]);

            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t kv_seqlen_align =
                (kv_seqlen + block_size - 1) / block_size * block_size;
            uint32_t start_head =
                (process % q_block_num_per_batch) * cur_qn_blk_size;
            uint32_t start_kv = 0;
            uint32_t cur_q_seq_len = q_seqlen;
            uint32_t cur_kv_seqlen = kv_seqlen;
            uint32_t cur_head_num = cur_qn_blk_size;
            InnerRunCubeMLA(cur_batch, start_head, cur_head_num, start_kv,
                            cur_q_seq_len, cur_kv_seqlen, ndtiling);
        }
        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(M, MTE1, EVENT_ID2);
        WAIT_FLAG(M, MTE1, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID4);
        WAIT_FLAG(M, MTE1, EVENT_ID5);
        WAIT_FLAG(M, MTE1, EVENT_ID6);
        WAIT_FLAG(M, MTE1, EVENT_ID7);
        WAIT_FLAG(FIX, M, EVENT_ID0);
        WAIT_FLAG(FIX, M, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID6);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
        WAIT_FLAG(FIX, MTE1, EVENT_ID0);
        WAIT_FLAG(FIX, MTE1, EVENT_ID1);
        WAIT_FLAG(FIX, MTE1, EVENT_ID2);
        WAIT_FLAG(FIX, MTE1, EVENT_ID3);
        WAIT_FLAG(FIX, MTE1, EVENT_ID4);
        WAIT_FLAG(FIX, MTE1, EVENT_ID5);
        WAIT_FLAG(MTE2, FIX, EVENT_ID0);
        PIPE_BARRIER(ALL);
    }

    __aicore__ __attribute__((always_inline)) inline void RunTP1() {
        constexpr uint32_t TP1_FP16_L1_Q_SIZE = 256 * 576 * 2;
        constexpr uint32_t TP1_FP16_L1_K_SIZE = 128 * 512 * 2 + 128 * 64 * 4;
        constexpr uint32_t TP1_FP16_Q_OFFSET = 0;
        constexpr uint32_t TP1_FP16_K_OFFSET =
            TP1_FP16_Q_OFFSET + TP1_FP16_L1_Q_SIZE;
        constexpr uint32_t TP1_FP16_P_OFFSET =
            TP1_FP16_K_OFFSET + TP1_FP16_L1_K_SIZE;
        l1q_buf_addr_tensor =
            buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(TP1_FP16_Q_OFFSET);
        l1kv_buf_addr_tensor =
            buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(TP1_FP16_K_OFFSET);
        l1p_buf_addr_tensor =
            buf.GetBuffer<BufferType::ASCEND_CB, IN_DTYPE>(TP1_FP16_P_OFFSET);
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(M, MTE1, EVENT_ID2);
        SET_FLAG(M, MTE1, EVENT_ID3);
        SET_FLAG(M, MTE1, EVENT_ID4);
        SET_FLAG(M, MTE1, EVENT_ID5);
        SET_FLAG(M, MTE1, EVENT_ID6);
        SET_FLAG(M, MTE1, EVENT_ID7);
        SET_FLAG(FIX, M, EVENT_ID0);
        SET_FLAG(FIX, M, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(MTE1, MTE2, EVENT_ID4);
        SET_FLAG(MTE1, MTE2, EVENT_ID5);
        SET_FLAG(MTE1, MTE2, EVENT_ID6);
        SET_FLAG(MTE1, MTE2, EVENT_ID7);
        SET_FLAG(FIX, MTE1, EVENT_ID0);
        SET_FLAG(FIX, MTE1, EVENT_ID1);
        SET_FLAG(FIX, MTE1, EVENT_ID2);
        SET_FLAG(FIX, MTE1, EVENT_ID3);
        SET_FLAG(FIX, MTE1, EVENT_ID4);
        SET_FLAG(FIX, MTE1, EVENT_ID5);
        SET_FLAG(MTE2, FIX, EVENT_ID0);

        __gm__ const NdMLADecodingMtpTilingTP1 *__restrict ndMtpTiling =
            &(tiling_gm->ndMLADecodingMtpTilingTP1);
        for (uint32_t process = block_idx; process < totalTaskNum;
             process += block_num) {
            uint32_t offset_tiling =
                tiling_head_size + tiling_para_size * process;
            uint32_t cur_batch = (uint32_t)(ndMtpTiling->batchIdx[process]);
            uint32_t q_row_id = (uint32_t)(ndMtpTiling->qSeqLen[process]);
            uint32_t kv_seqlen = (uint32_t)(ndMtpTiling->kvBlockIdx[process]);
            uint32_t q_seqlen = (uint32_t)(ndMtpTiling->prevKVSeqlen[process]);

            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t start_head = q_row_id * q_heads;
            uint32_t start_kv = 0;
            uint32_t cur_q_seq_len = q_seqlen;
            uint32_t cur_kv_seqlen = kv_seqlen;
            uint32_t cur_head_num = q_heads;
            InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv,
                               cur_q_seq_len, cur_kv_seqlen, offset_tiling);
        }

        if constexpr (flashDecoding) {
            for (uint32_t process = block_idx; process < flashDecodingTaskNum;
                 process += (uint32_t)block_num) {  // for task
                uint32_t offset_tiling =
                    tiling_head_size +
                    tiling_para_size * (totalTaskNum + process);
                uint32_t cur_batch =
                    (uint32_t)(ndMtpTiling->batchIdx[totalTaskNum + process]);
                uint32_t q_seqlen =
                    (uint32_t)(ndMtpTiling->qSeqLen[totalTaskNum + process]);
                uint32_t kv_seqlen =
                    (uint32_t)(ndMtpTiling->kvBlockIdx[totalTaskNum + process]);
                if (kv_seqlen == 0) {
                    continue;
                }
                uint32_t start_kv =
                    (uint32_t)(ndMtpTiling
                                   ->prevKVSeqlen[totalTaskNum + process]);
                uint32_t prev_task =
                    (uint32_t)(ndMtpTiling->batchSize[totalTaskNum + process]);
                uint32_t split_num =
                    (uint32_t)(ndMtpTiling->splitNum[totalTaskNum + process]);
                uint32_t prev_split_num =
                    (uint32_t)(ndMtpTiling
                                   ->prevSplitNumSum[totalTaskNum + process]);
                uint32_t start_head = prev_task * q_heads;
                uint32_t cur_q_seq_len = q_seqlen;
                if constexpr (tilingKeyType ==
                              TilingKeyType::TILING_INT8_DATA) {
                    cur_q_seq_len = 1;
                }
                uint32_t cur_kv_seqlen = kv_seqlen;
                uint32_t cur_head_num = q_heads;
                uint32_t kvEndFlag =
                    (process - prev_split_num) % split_num == split_num - 1 ? 1
                                                                            : 0;
                InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num,
                                   start_kv, cur_q_seq_len, cur_kv_seqlen,
                                   offset_tiling, kvEndFlag);
            }
        }
        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(M, MTE1, EVENT_ID2);
        WAIT_FLAG(M, MTE1, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID4);
        WAIT_FLAG(M, MTE1, EVENT_ID5);
        WAIT_FLAG(M, MTE1, EVENT_ID6);
        WAIT_FLAG(M, MTE1, EVENT_ID7);
        WAIT_FLAG(FIX, M, EVENT_ID0);
        WAIT_FLAG(FIX, M, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID6);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
        WAIT_FLAG(FIX, MTE1, EVENT_ID0);
        WAIT_FLAG(FIX, MTE1, EVENT_ID1);
        WAIT_FLAG(FIX, MTE1, EVENT_ID2);
        WAIT_FLAG(FIX, MTE1, EVENT_ID3);
        WAIT_FLAG(FIX, MTE1, EVENT_ID4);
        WAIT_FLAG(FIX, MTE1, EVENT_ID5);
        WAIT_FLAG(MTE2, FIX, EVENT_ID0);
        PIPE_BARRIER(ALL);
    }

   private:
    __aicore__ __attribute__((always_inline)) inline void InnerRunCubeMLA(
        uint32_t cur_batch, uint32_t start_head, uint32_t cur_head_num,
        uint32_t start_kv, uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen,
        const __gm__ NdMLATiling *ndtiling) {
        uint32_t addr_q_high32 = (uint32_t)(ndtiling->qSeqHigh[cur_batch]);
        uint32_t addr_q_loww32 = (uint32_t)(ndtiling->qSeqLow[cur_batch]);

        uint64_t addr_q_scalar =
            (uint64_t)(((uint64_t)addr_q_high32) << 32 | addr_q_loww32);
        uint64_t q_offset = addr_q_scalar * 512 + start_head * 512;
        uint64_t q_rope_offset = addr_q_scalar * 64 + start_head * 64;

        uint32_t pp_n_scalar = block_size;

        uint32_t n_loop = (cur_kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;

        uint32_t qk_n = pp_n_scalar;
        uint32_t qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
        uint32_t qk_n_2 = pp_n_scalar;
        uint32_t qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
        uint32_t qk_round_n_l1 = RoundUp<T_BLOCK_SIZE>(qk_n);
        uint32_t qk_round_n_2_l1 = RoundUp<T_BLOCK_SIZE>(qk_n_2);
        uint64_t hidden_size = 576;
        uint64_t k_round_n = qk_round_n;
        uint32_t row_num = cur_head_num * cur_q_seqlen;
        m = RoundUp<16>(row_num);

        if (cur_q_seqlen == 1) {
            gm_to_l1<ArchType::ASCEND_V220, IN_DTYPE, DataFormat::ND,
                     DataFormat::NZ>(
                l1q_buf_addr_tensor, q_gm_tensor[q_offset],
                cur_head_num,               // nValue
                RoundUp<16>(cur_head_num),  // dstNzC0Stride
                0,                          // dstNzMatrixStride, unused
                512,                        // dValue
                0,                          // dstNzMatrixStride, unused
                512                         // srcDValue
            );
        } else {
            if (q_heads < 128) {
                AscendC::DataCopy(
                    l1q_buf_addr_tensor, q_gm_tensor[q_offset],
                    AscendC::Nd2NzParams(
                        cur_q_seqlen,   // ndNum
                        cur_head_num,   // nValue
                        512,            // dValue
                        512 * q_heads,  // srcNdMatrixStride
                        512,            // srcDValue
                        RoundUp<16>(cur_head_num *
                                    cur_q_seqlen),  // dstNzC0Stride
                        cur_q_seqlen,               // dstNzNStride
                        16                          // dstNzMatrixStride
                        ));
            } else {
                for (uint32_t ii = 0; ii < cur_q_seqlen; ii++) {
                    AscendC::DataCopy(
                        l1q_buf_addr_tensor[ii * 16],  // offset one datablock
                        q_gm_tensor[q_offset + ii * q_heads * 512],
                        AscendC::Nd2NzParams(
                            1,             // ndNum
                            cur_head_num,  // nValue
                            512,           // dValue
                            0,             // srcNdMatrixStride
                            512,           // srcDValue
                            RoundUp<16>(cur_q_seqlen *
                                        cur_head_num),  // dstNzC0Stride
                            cur_q_seqlen,               // dstNzNStride
                            16                          // dstNzMatrixStride
                            ));
                }
            }
        }
        AscendC::DataCopy(
            l1q_buf_addr_tensor[RoundUp<16>(cur_head_num * cur_q_seqlen) * 512],
            q_rope_gm_tensor[q_rope_offset],
            AscendC::Nd2NzParams(
                cur_head_num,                              // ndNum, 32
                cur_q_seqlen,                              // nValue, 4
                64,                                        // dValue
                64,                                        // srcNdMatrixStride
                64 * q_heads,                              // srcDValue
                RoundUp<16>(cur_head_num * cur_q_seqlen),  // dstNzC0Stride
                1,                                         // dstNzNStride
                16 * cur_q_seqlen                          // dstNzMatrixStride
                ));
        SET_FLAG(MTE2, MTE1, EVENT_ID0);
        WAIT_FLAG(MTE2, MTE1, EVENT_ID0);

        for (uint32_t n_idx = 0; n_idx < n_loop + 1; n_idx += 1) {
            // cube1 QK^T -> S
            if (n_idx != n_loop) {
                uint32_t l1_kv_pingpong_flag = n_idx % 2;
                if (n_idx == (n_loop - 1)) {
                    qk_n = (cur_kv_seqlen - n_idx * pp_n_scalar);
                    qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
                    qk_round_n_l1 = RoundUp<T_BLOCK_SIZE>(qk_n);
                }

                k_round_n = qk_round_n;
                uint64_t hiddenSize_offset =
                    start_head * cur_q_seqlen * embedding_size;
                uint32_t embed_split_size = 128;
                uint32_t round_embed_split_size =
                    RoundUp<T_BLOCK_SIZE>(embed_split_size);

                uint32_t block_table_id = (uint32_t)(*(
                    block_tables_gm + cur_batch * max_num_blocks_per_query +
                    start_kv / block_size + n_idx));
                int64_t kv_offset =
                    (int64_t)block_table_id * block_size * stride_kv;
                int64_t kv_offset_rope =
                    (int64_t)block_table_id * block_size * stride_kv_rope;
                uint32_t q_load_coeff = 1;
                q_load_coeff = m;
                WAIT_FLAG(MTE1, MTE2, l1_kv_pingpong_flag);

                if constexpr (KInputType == InputFormat::ND_FORMAT) {
                    gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::ND,
                             DataFormat::NZ>(
                        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 576],
                        k_gm_tensor[kv_offset],
                        qk_n,        // nValue
                        qk_round_n,  // dstNzC0Stride
                        0,           // dstNzMatrixStride, unused
                        512,         // dValue
                        0,           // dstNzMatrixStride, unused
                        stride_kv    // srcDValue
                    );
                    SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                    WAIT_FLAG(MTE1, MTE2, l1_kv_pingpong_flag + 2);

                    gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::ND,
                             DataFormat::NZ>(
                        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 576 +
                                             512 * qk_round_n],
                        k_rope_gm_tensor[kv_offset_rope],
                        qk_n,           // nValue
                        qk_round_n,     // dstNzC0Stride
                        0,              // dstNzMatrixStride, unused
                        64,             // dValue
                        0,              // dstNzMatrixStride, unused
                        stride_kv_rope  // srcDValue
                    );
                } else {
                    gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::NZ,
                             DataFormat::NZ>(
                        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 576],
                        k_gm_tensor[kv_offset],
                        qk_round_n,  // nValue
                        qk_round_n,  // nValue
                        block_size,  // dstNzC0Stride
                        512,         // dValue
                        512,         // dValue
                        0            // srcDValue
                    );

                    SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                    WAIT_FLAG(MTE1, MTE2, l1_kv_pingpong_flag + 2);
                    gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE, DataFormat::NZ,
                             DataFormat::NZ>(
                        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 576 +
                                             512 * qk_round_n],
                        k_rope_gm_tensor[kv_offset_rope],
                        qk_round_n,  // nValue
                        qk_round_n,  // nValue
                        block_size,  // dstNzC0Stride
                        64,          // dValue
                        64,          // dstNzMatrixStride, unused
                        0            // srcDValue
                    );
                }

                SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag + 2);
                uint64_t hidden_split_time = (hidden_size + 128 - 1) / 128;
                uint64_t embed_split_idx = 0;
                for (embed_split_idx = 0; embed_split_idx < hidden_split_time;
                     ++embed_split_idx) {
                    if (embed_split_idx == 4) {
                        embed_split_size = 64;
                        round_embed_split_size = 64;
                    }
                    WAIT_FLAG(M, MTE1, embed_split_idx % 2);

                    for (uint64_t loa_load_idx = 0;
                         loa_load_idx < q_load_coeff / BLOCK_SIZE;
                         ++loa_load_idx) {
                        l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE, false,
                                   DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0a_buf_tensor[embed_split_idx % 2 * 16384 +
                                           loa_load_idx *
                                               round_embed_split_size *
                                               BLOCK_SIZE],
                            l1q_buf_addr_tensor[embed_split_idx * m * 128 +
                                                loa_load_idx *
                                                    T_CUBE_MATRIX_SIZE],
                            0,
                            round_embed_split_size / T_BLOCK_SIZE,  // repeat
                            0,
                            q_load_coeff / BLOCK_SIZE,  // srcStride
                            0,
                            0  // dstStride
                        );
                    }

                    SET_FLAG(MTE1, M, embed_split_idx % 2);

                    if (embed_split_idx == 0) {
                        WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                    }
                    if (embed_split_idx == 4) {
                        WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag + 2);
                    }
                    WAIT_FLAG(M, MTE1, embed_split_idx % 2 + 2);

                    l1_to_l0_b<ArchType::ASCEND_V220, IN_DTYPE, false,
                               DataFormat::VECTOR, DataFormat::VECTOR>(
                        l0b_buf_tensor[embed_split_idx % 2 * 16384],
                        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 *
                                                 hidden_size +
                                             embed_split_idx * k_round_n * 128],
                        0,
                        round_embed_split_size * k_round_n /
                            T_CUBE_MATRIX_SIZE,  // repeat
                        0,
                        1,  // srcStride
                        0,
                        0  // dstStride
                    );
                    if (embed_split_idx == 4) {
                        SET_FLAG(MTE1, MTE2, l1_kv_pingpong_flag + 2);
                    }
                    SET_FLAG(MTE1, M, embed_split_idx % 2 + 2);
                    WAIT_FLAG(MTE1, M, embed_split_idx % 2);
                    WAIT_FLAG(MTE1, M, embed_split_idx % 2 + 2);
                    if (embed_split_idx == 0) {
                        WAIT_FLAG(FIX, M, l1_kv_pingpong_flag);
                    }

                    mmad<ArchType::ASCEND_V220, IN_DTYPE, IN_DTYPE,
                         mm1OutputType, false>(
                        mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
                        l0a_buf_tensor[embed_split_idx % 2 * 16384],
                        l0b_buf_tensor[embed_split_idx % 2 * 16384],
                        m,                    // m
                        qk_n,                 // n
                        embed_split_size,     // k
                        embed_split_idx == 0  // cmatrixInitVal
                    );

                    PIPE_BARRIER(M);
                    SET_FLAG(M, MTE1, embed_split_idx % 2);
                    SET_FLAG(M, MTE1, embed_split_idx % 2 + 2);

                    // copy S to gm
                    if (embed_split_idx == 4) {
                        SET_FLAG(M, FIX, l1_kv_pingpong_flag);
                        WAIT_FLAG(M, FIX, l1_kv_pingpong_flag);

                        l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND,
                                  mm1CopyType, mm1OutputType>(
                            s_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER +
                                        (uint64_t)(n_idx % 2) *
                                            TMP_SIZE_DECODER / 2],
                            mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
                            m,               // MSize
                            qk_round_n,      // NSize
                            RoundUp<16>(m),  // srcStride
                            qk_round_n       // dstStride_dst_D
                        );
                        SET_FLAG(FIX, M, l1_kv_pingpong_flag);
                    }
                }
                FftsCrossCoreSync<PIPE_FIX, 2>(QK_READY_DECODER);
            }
            if (n_idx != 0) {
                if (n_idx == n_loop) {
                    qk_n_2 = (cur_kv_seqlen - (n_idx - 1) * pp_n_scalar);
                    qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
                    qk_round_n_2_l1 = RoundUp<T_BLOCK_SIZE>(qk_n_2);
                }
                k_round_n = qk_round_n_2_l1;
                uint32_t l1_kv_pingpong_flag = (n_idx - 1) % 2;
                uint32_t l0_p_pingpong_flag = (n_idx - 1) % 2;
                uint32_t embed_split_size = 128;
                embed_split_loop_v = 4;
                uint32_t round_embed_split_size =
                    RoundUp<T_BLOCK_SIZE>(embed_split_size);
                for (uint32_t embed_split_idx = 0;
                     embed_split_idx < embed_split_loop_v; ++embed_split_idx) {
                    uint32_t l0c_pingpong_flag = (n_idx + embed_split_idx) % 2;
                    uint32_t l0b_pingpong_flag = (embed_split_idx + 1) % 2;
                    uint64_t l1kv_offset =
                        embed_split_idx * k_round_n * round_embed_split_size;
                    WAIT_FLAG(M, MTE1, l0b_pingpong_flag + 2);
                    AscendC::LoadData2dTransposeParams loadDataParams;
                    loadDataParams.dstGap = 0;
                    loadDataParams.startIndex = 0;
                    loadDataParams.dstFracGap = 0;
                    if (k_round_n <= round_embed_split_size) {  // Nz -> nZ
                        loadDataParams.repeatTimes =
                            round_embed_split_size / T_BLOCK_SIZE;
                        loadDataParams.srcStride = k_round_n / T_BLOCK_SIZE;
                        uint16_t dstGap = sizeof(IN_DTYPE) == 1 ? 1 : 0;
                        loadDataParams.dstGap = dstGap;
                        for (uint32_t l0b_load_idx = 0;
                             l0b_load_idx < k_round_n / T_BLOCK_SIZE;
                             ++l0b_load_idx) {
                            AscendC::LoadDataWithTranspose(
                                l0b_buf_tensor[l0b_pingpong_flag * 16384 +
                                               l0b_load_idx *
                                                   RoundUp<16>(
                                                       embed_split_size) *
                                                   T_BLOCK_SIZE],
                                l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 *
                                                         hidden_size +
                                                     l1kv_offset +
                                                     l0b_load_idx *
                                                         T_BLOCK_SIZE *
                                                         T_BLOCK_SIZE],
                                loadDataParams);
                        }
                    } else {
                        for (uint32_t l0b_load_idx = 0;
                             l0b_load_idx <
                             round_embed_split_size / T_BLOCK_SIZE;
                             ++l0b_load_idx) {
                            loadDataParams.repeatTimes =
                                qk_round_n_2 / T_BLOCK_SIZE;
                            loadDataParams.srcStride = 1;
                            loadDataParams.dstGap =
                                round_embed_split_size / BLOCK_SIZE - 1;
                            AscendC::LoadDataWithTranspose(
                                l0b_buf_tensor[l0b_pingpong_flag * 16384 +
                                               l0b_load_idx * T_BLOCK_SIZE *
                                                   T_BLOCK_SIZE],
                                l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 *
                                                         hidden_size +
                                                     l1kv_offset +
                                                     l0b_load_idx *
                                                         qk_round_n_2 *
                                                         T_BLOCK_SIZE],
                                loadDataParams);
                        }
                    }
                    if (embed_split_idx == embed_split_loop_v - 1) {
                        SET_FLAG(MTE1, MTE2, l1_kv_pingpong_flag);
                    }
                    // move p from gm to l1
                    uint32_t p_move_head_num = row_num;
                    if (embed_split_idx == 0) {
                        WaitFlagDev(SOFTMAX_READY_DECODER);

                        WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
                        gm_to_l1<ArchType::ASCEND_V220, IN_DTYPE,
                                 DataFormat::ND, DataFormat::NZ>(
                            l1p_buf_addr_tensor,
                            p_gm_tensor[(uint64_t)block_idx * TMP_SIZE *
                                            T_BLOCK_OFFSET +
                                        ((n_idx - 1) % 2) * TMP_SIZE *
                                            T_BLOCK_OFFSET / 2],
                            p_move_head_num,  // nValue
                            RoundUp<BLOCK_SIZE>(
                                p_move_head_num),  // dstNzC0Stride
                            0,                     // dstNzMatrixStride, unused
                            k_round_n,             // dValue
                            0,                     // dstNzMatrixStride, unused
                            qk_round_n_2 * 2 / sizeof(IN_DTYPE)  // srcDValue
                        );
                        SET_FLAG(MTE2, MTE1, EVENT_ID7);
                        WAIT_FLAG(MTE2, MTE1, EVENT_ID7);
                        // move p from l1 to l0a
                        WAIT_FLAG(M, MTE1, l0_p_pingpong_flag);
                        uint32_t p_load_coeff = RoundUp<16>(p_move_head_num);
                        for (uint64_t loa_load_idx = 0;
                             loa_load_idx < p_load_coeff / BLOCK_SIZE;
                             ++loa_load_idx) {
                            l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE, false,
                                       DataFormat::VECTOR, DataFormat::VECTOR>(
                                l0a_buf_tensor[l0_p_pingpong_flag * 16384 +
                                               loa_load_idx * qk_round_n_2 *
                                                   BLOCK_SIZE],
                                l1p_buf_addr_tensor[loa_load_idx *
                                                    T_CUBE_MATRIX_SIZE],
                                0,
                                qk_round_n_2 / T_BLOCK_SIZE,  // repeat
                                0,
                                p_load_coeff / BLOCK_SIZE,  // srcStride
                                0,
                                0  // dstStride
                            );
                        }
                        SET_FLAG(MTE1, MTE2, EVENT_ID7);
                    }
                    SET_FLAG(MTE1, M, l0b_pingpong_flag);
                    WAIT_FLAG(MTE1, M, l0b_pingpong_flag);
                    WAIT_FLAG(FIX, M, l0c_pingpong_flag);

                    mmad<ArchType::ASCEND_V220, IN_DTYPE, IN_DTYPE,
                         mm2OutputType, false>(
                        mm2_l0c_buf_tensor[l0c_pingpong_flag * 16384],
                        l0a_buf_tensor[l0_p_pingpong_flag * 16384],
                        l0b_buf_tensor[l0b_pingpong_flag * 16384],
                        m,                 // m
                        embed_split_size,  // n
                        qk_n_2,            // k
                        1                  // cmatrixInitVal
                    );
                    PIPE_BARRIER(M);
                    SET_FLAG(M, MTE1, l0b_pingpong_flag + 2);
                    if (embed_split_idx == embed_split_loop_v - 1) {
                        SET_FLAG(M, MTE1, l0_p_pingpong_flag);
                    }
                    SET_FLAG(M, FIX, l0c_pingpong_flag);
                    WAIT_FLAG(M, FIX, l0c_pingpong_flag);

                    // copy O to gm
                    l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND,
                              mm2CopyType, mm2OutputType>(
                        o_tmp_gm_tensor[(uint64_t)block_idx * TMP_SIZE * 2 +
                                        embed_split_idx *
                                            round_embed_split_size +
                                        ((n_idx - 1) % 2) * TMP_SIZE],
                        mm2_l0c_buf_tensor[l0c_pingpong_flag * 16384],
                        m,  // MSize
                        RoundUp<16>(
                            embed_split_size),  // NSize
                                                // 32B对齐，防止workspace补齐的位置中有脏数据
                        RoundUp<16>(m),  // srcStride
                        round_v          // dstStride_dst_D
                    );
                    SET_FLAG(FIX, M, l0c_pingpong_flag);
                }
                FftsCrossCoreSync<PIPE_FIX, 2>(UPDATE_READY_DECODER);
            }
        }
    }

    __aicore__ __attribute__((always_inline)) inline void InnerRunCubeMLATP1(
        uint32_t cur_batch, uint32_t start_head, uint32_t cur_head_num,
        uint32_t start_kv, uint32_t cur_q_seqlen, uint32_t cur_kvs_seqlen,
        uint32_t offset_tiling, uint32_t kvEndFlag = 1) {
        uint32_t cur_kv_seqlen =
            cur_kvs_seqlen + (1 - cur_q_seqlen) * kvEndFlag;
        uint64_t addr_q_scalar = 0;
        uint64_t q_offset = addr_q_scalar * 512 + start_head * 512;
        uint64_t q_rope_offset = addr_q_scalar * 64 + start_head * 64;

        uint32_t pp_n_scalar = block_size;

        uint32_t n_loop =
            (cur_kv_seqlen + (cur_q_seqlen - 1) * kvEndFlag + pp_n_scalar - 1) /
            pp_n_scalar;

        uint32_t qk_n = pp_n_scalar;
        uint32_t qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
        uint32_t qk_n_2 = pp_n_scalar;
        uint32_t qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);

        uint32_t row_num = cur_head_num * cur_q_seqlen;

        uint32_t sv_n = n_loop == 1 ? cur_kv_seqlen : pp_n_scalar;
        m = RoundUp<16>(row_num);
        gm_to_l1<ArchType::ASCEND_V220, IN_DTYPE, DataFormat::ND,
                 DataFormat::NZ>(l1q_buf_addr_tensor, q_gm_tensor[q_offset],
                                 row_num,               // nValue
                                 RoundUp<16>(row_num),  // dstNzC0Stride
                                 0,    // dstNzMatrixStride, unused
                                 512,  // dValue
                                 0,    // dstNzMatrixStride, unused
                                 512   // srcDValue
        );
        gm_to_l1<ArchType::ASCEND_V220, IN_ROPE_DTYPE, DataFormat::ND,
                 DataFormat::NZ>(
            l1q_buf_addr_tensor[RoundUp<16>(row_num) * 512],
            q_rope_gm_tensor[q_rope_offset],
            row_num,               // nValue
            RoundUp<16>(row_num),  // dstNzC0Stride
            0,                     // dstNzMatrixStride, unused
            64,                    // dValue
            0,                     // dstNzMatrixStride, unused
            64                     // srcDValue
        );

        SET_FLAG(MTE2, MTE1, EVENT_ID0);
        WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
        uint32_t q_load_coeff = q_heads;
        uint32_t l0a_pingpong_size = L0AB_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
        uint32_t l0b_pingpong_size = L0AB_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
        uint32_t l0c_pingpong_size = L0C_FLOAT_BUF_SIZE;
        constexpr uint32_t s_block_stack = KV_FLOW_NUM;

        for (uint32_t n_idx = 0; n_idx < n_loop + s_block_stack;
             n_idx += s_block_stack) {
            if (n_idx < n_loop) {
                uint32_t sv_n_triu = n_loop * pp_n_scalar;
                if (n_idx + s_block_stack > n_loop - 1) {
                    sv_n = cur_kv_seqlen - n_idx * pp_n_scalar;  // delete
                } else {
                    sv_n = pp_n_scalar * s_block_stack;
                }
                uint32_t sv_round_n =
                    (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                for (uint32_t split_idx = 0;
                     split_idx < s_block_stack && n_idx + split_idx < n_loop;
                     split_idx++) {
                    uint32_t now_idx = n_idx + split_idx;
                    if (now_idx == (n_loop - 1)) {
                        qk_n = (cur_kv_seqlen - now_idx * pp_n_scalar);
                        qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
                    } else {
                        qk_n = pp_n_scalar;
                        qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
                    }
                    bool last_split =
                        split_idx == s_block_stack - 1 || now_idx == n_loop - 1;

                    uint32_t block_table_id = (uint32_t)(*(
                        block_tables_gm + cur_batch * max_num_blocks_per_query +
                        start_kv / block_size + now_idx));
                    int64_t kv_offset =
                        (int64_t)block_table_id * block_size * stride_kv;
                    int64_t kv_offset_rope =
                        (int64_t)block_table_id * block_size * stride_kv_rope;
                    int64_t now_l1_offset = 0;
                    uint32_t embed_split_size = 256 / sizeof(IN_DTYPE);
                    uint32_t round_embed_split_size =
                        RoundUp<T_BLOCK_SIZE>(embed_split_size);
                    uint32_t embed_split_loop_v =
                        (512 + round_embed_split_size - 1) /
                        round_embed_split_size;
                    uint32_t embed_split_loop_total =
                        embed_split_loop_v + 1;  // add 1 rope round
                    uint32_t embed_split_idx;
                    for (embed_split_idx = 0;
                         embed_split_idx < embed_split_loop_v;
                         ++embed_split_idx) {
                        uint32_t qk_n_q1 =
                            now_idx == n_loop - 1
                                ? qk_n + (cur_q_seqlen - 1) * kvEndFlag
                                : qk_n;
                        uint32_t qk_round_n_q1 = RoundUp<BLOCK_SIZE>(qk_n_q1);
                        auto l1kv_pingpong_flag =
                            (now_idx * embed_split_loop_total +
                             embed_split_idx) %
                            2;
                        WAIT_FLAG(MTE1, MTE2, l1kv_pingpong_flag);
                        now_l1_offset = l1kv_pingpong_flag % 2 * 128 * 128;

                        if constexpr (KInputType == InputFormat::ND_FORMAT) {
                            gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE,
                                     DataFormat::ND, DataFormat::NZ>(
                                l1kv_buf_addr_tensor[now_l1_offset],
                                k_gm_tensor[kv_offset + embed_split_idx * 128],
                                qk_n_q1,        // nValue
                                qk_round_n_q1,  // dstNzC0Stride
                                0,              // dstNzMatrixStride, unused
                                128,            // dValue
                                0,              // dstNzMatrixStride, unused
                                stride_kv       // srcDValue
                            );
                        } else {
                            gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE,
                                     DataFormat::NZ, DataFormat::NZ>(
                                l1kv_buf_addr_tensor[now_l1_offset],
                                k_gm_tensor[kv_offset +
                                            block_size * 128 * embed_split_idx],
                                qk_round_n_q1, qk_round_n_q1, block_size, 128,
                                128, 128);
                        }
                        SET_FLAG(MTE2, MTE1, l1kv_pingpong_flag);
                        WAIT_FLAG(MTE2, MTE1, l1kv_pingpong_flag);

                        auto l0kv_pingpong_flag =
                            (now_idx * embed_split_loop_total +
                             embed_split_idx) %
                                2 +
                            2;
                        auto now_l0_offset =
                            l0kv_pingpong_flag % 2 * l0b_pingpong_size;
                        WAIT_FLAG(M, MTE1, l0kv_pingpong_flag);

                        l1_to_l0_b<ArchType::ASCEND_V220, IN_DTYPE, false,
                                   DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0b_buf_tensor[now_l0_offset],
                            l1kv_buf_addr_tensor[now_l1_offset], 0,
                            round_embed_split_size * qk_round_n_q1 /
                                T_CUBE_MATRIX_SIZE,  // repeat
                            0,
                            1,  // srcStride
                            0,
                            0  // dstStride
                        );
                        SET_FLAG(MTE1, M, l0kv_pingpong_flag);
                        WAIT_FLAG(MTE1, M, l0kv_pingpong_flag);
                        for (int q_i = 0; q_i < cur_q_seqlen; q_i++) {
                            uint32_t qk_n_q =
                                now_idx == n_loop - 1
                                    ? qk_n + (cur_q_seqlen - 1) * kvEndFlag
                                    : qk_n;
                            uint32_t qk_round_n_q = RoundUp<BLOCK_SIZE>(qk_n_q);
                            uint32_t qk_c_pingpong_flag =
                                (now_idx * cur_q_seqlen + q_i) % 2;
                            uint32_t qk_0_pingpong_flag =
                                (embed_split_idx * cur_q_seqlen + q_i) % 2;
                            WAIT_FLAG(M, MTE1, qk_0_pingpong_flag);

                            for (uint64_t loa_load_idx = 0;
                                 loa_load_idx < q_load_coeff / BLOCK_SIZE;
                                 ++loa_load_idx) {
                                l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE,
                                           false, DataFormat::VECTOR,
                                           DataFormat::VECTOR>(
                                    l0a_buf_tensor[qk_0_pingpong_flag *
                                                       l0a_pingpong_size +
                                                   loa_load_idx *
                                                       round_embed_split_size *
                                                       BLOCK_SIZE],
                                    l1q_buf_addr_tensor
                                        [q_i * q_heads * BLOCK_SIZE +
                                         embed_split_idx * m *
                                             round_embed_split_size +
                                         loa_load_idx * T_CUBE_MATRIX_SIZE],
                                    0,
                                    round_embed_split_size /
                                        T_BLOCK_SIZE,  // repeat
                                    0,
                                    m / BLOCK_SIZE,  // srcStride
                                    0,
                                    0  // dstStride
                                );
                            }
                            SET_FLAG(MTE1, M, qk_0_pingpong_flag);
                            WAIT_FLAG(MTE1, M, qk_0_pingpong_flag);

                            mmad<ArchType::ASCEND_V220, IN_DTYPE, IN_DTYPE,
                                 mm1OutputType, false>(
                                mm1_l0c_buf_tensor[qk_c_pingpong_flag *
                                                   l0c_pingpong_size],
                                l0a_buf_tensor[qk_0_pingpong_flag *
                                               l0a_pingpong_size],
                                l0b_buf_tensor[now_l0_offset],
                                q_load_coeff,          // m
                                qk_n_q,                // n
                                embed_split_size,      // k
                                embed_split_idx == 0,  // cmatrixInitVal
                                0b10);
                            PIPE_BARRIER(M);
                            SET_FLAG(M, MTE1, qk_0_pingpong_flag);
                        }
                        SET_FLAG(MTE1, MTE2, l1kv_pingpong_flag);
                        SET_FLAG(M, MTE1, l0kv_pingpong_flag);
                    }
                    embed_split_idx = 4;
                    embed_split_size = 64;
                    round_embed_split_size = 64;
                    uint32_t qk_n_q1 =
                        now_idx == n_loop - 1
                            ? qk_n + (cur_q_seqlen - 1) * kvEndFlag
                            : qk_n;
                    uint32_t qk_round_n_q1 = RoundUp<BLOCK_SIZE>(qk_n_q1);
                    auto l1kv_pingpong_flag =
                        (now_idx * embed_split_loop_total + embed_split_idx) %
                        2;
                    WAIT_FLAG(MTE1, MTE2, l1kv_pingpong_flag);
                    now_l1_offset =
                        l1kv_pingpong_flag * 128 * 64 + 2 * 128 * 128;

                    if constexpr (KInputType == InputFormat::ND_FORMAT) {
                        gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE,
                                 DataFormat::ND, DataFormat::NZ>(
                            l1kv_buf_addr_tensor[now_l1_offset],
                            k_rope_gm_tensor[kv_offset_rope],
                            qk_n_q1,        // nValue
                            qk_round_n_q1,  // dstNzC0Stride
                            0,              // dstNzMatrixStride, unused
                            64,             // dValue
                            0,              // dstNzMatrixStride, unused
                            stride_kv_rope  // srcDValue
                        );
                    } else {
                        gm_to_l1<ArchType::ASCEND_V220, IN_KVDTYPE,
                                 DataFormat::NZ, DataFormat::NZ>(
                            l1kv_buf_addr_tensor[now_l1_offset],
                            k_rope_gm_tensor[kv_offset_rope], qk_round_n_q1,
                            qk_round_n_q1, block_size, 64, 64, 64);
                    }
                    SET_FLAG(MTE2, MTE1, l1kv_pingpong_flag);
                    WAIT_FLAG(MTE2, MTE1, l1kv_pingpong_flag);
                    auto l0kv_pingpong_flag =
                        (now_idx * embed_split_loop_total + embed_split_idx) %
                            2 +
                        2;
                    auto now_l0_offset =
                        l0kv_pingpong_flag % 2 * l0b_pingpong_size;
                    WAIT_FLAG(M, MTE1, l0kv_pingpong_flag);
                    l1_to_l0_b<ArchType::ASCEND_V220, IN_DTYPE, false,
                               DataFormat::VECTOR, DataFormat::VECTOR>(
                        l0b_buf_tensor[now_l0_offset],
                        l1kv_buf_addr_tensor[now_l1_offset], 0,
                        round_embed_split_size * qk_round_n_q1 /
                            T_CUBE_MATRIX_SIZE,  // repeat
                        0,
                        1,  // srcStride
                        0,
                        0  // dstStride
                    );
                    SET_FLAG(MTE1, M, l0kv_pingpong_flag);
                    WAIT_FLAG(MTE1, M, l0kv_pingpong_flag);
                    for (int q_i = 0; q_i < cur_q_seqlen; q_i++) {
                        uint32_t qk_n_q =
                            now_idx == n_loop - 1
                                ? qk_n + (cur_q_seqlen - 1) * kvEndFlag
                                : qk_n;
                        uint32_t qk_round_n_q = RoundUp<BLOCK_SIZE>(qk_n_q);
                        uint32_t sv_round_n =
                            n_idx + s_block_stack > n_loop - 1
                                ? (sv_n + (cur_q_seqlen - 1) * kvEndFlag +
                                   BLOCK_SIZE - 1) /
                                      BLOCK_SIZE * BLOCK_SIZE
                                : (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE *
                                      BLOCK_SIZE;
                        uint32_t qk_c_pingpong_flag =
                            (now_idx * cur_q_seqlen + q_i) % 2;
                        uint32_t qk_0_pingpong_flag =
                            (embed_split_idx * cur_q_seqlen + q_i) % 2;
                        WAIT_FLAG(M, MTE1, qk_0_pingpong_flag);

                        for (uint64_t loa_load_idx = 0;
                             loa_load_idx < q_load_coeff / BLOCK_SIZE;
                             ++loa_load_idx) {
                            l1_to_l0_a<ArchType::ASCEND_V220, IN_ROPE_DTYPE,
                                       false, DataFormat::VECTOR,
                                       DataFormat::VECTOR>(
                                l0a_buf_tensor[qk_0_pingpong_flag * 16384 +
                                               loa_load_idx *
                                                   round_embed_split_size *
                                                   BLOCK_SIZE],
                                l1q_buf_addr_tensor[q_i * q_heads * BLOCK_SIZE +
                                                    embed_split_idx * m * 128 +
                                                    loa_load_idx *
                                                        CUBE_MATRIX_SIZE],
                                0,
                                round_embed_split_size / BLOCK_SIZE,  // repeat
                                0,
                                m / BLOCK_SIZE,  // srcStride
                                0,
                                0  // dstStride
                            );
                        }
                        SET_FLAG(MTE1, M, qk_0_pingpong_flag);
                        WAIT_FLAG(MTE1, M, qk_0_pingpong_flag);

                        mmad<ArchType::ASCEND_V220, IN_ROPE_DTYPE,
                             IN_ROPE_DTYPE, float, false>(
                            mm1_l0c_buf_tensor.template ReinterpretCast<
                                float>()[qk_c_pingpong_flag * 16384],
                            l0a_buf_tensor.template ReinterpretCast<
                                IN_ROPE_DTYPE>()[qk_0_pingpong_flag * 16384],
                            l0b_buf_tensor.template ReinterpretCast<
                                IN_ROPE_DTYPE>()[now_l0_offset],
                            q_load_coeff,      // m
                            qk_n_q,            // n
                            embed_split_size,  // k
                            0,                 // cmatrixInitVal
                            0b11);
                        PIPE_BARRIER(M);
                        SET_FLAG(M, MTE1, qk_0_pingpong_flag);
                        l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND,
                                  mm1CopyType, mm1OutputType>(
                            s_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER *
                                            4 * 2 +
                                        (uint64_t)((n_idx / s_block_stack) %
                                                   2) *
                                            TMP_SIZE_DECODER * 2 * 2 +
                                        q_i * 128 * 512 +
                                        split_idx * pp_n_scalar * q_heads],
                            mm1_l0c_buf_tensor[qk_c_pingpong_flag * 16384],
                            q_load_coeff,               // MSize
                            qk_round_n_q,               // NSize
                            RoundUp<16>(q_load_coeff),  // srcStride
                            128,                        // dstStride_dst_D
                            0b11);
                    }
                    SET_FLAG(MTE1, MTE2, l1kv_pingpong_flag);
                    SET_FLAG(M, MTE1, l0kv_pingpong_flag);
                }
                FftsCrossCoreSync<PIPE_FIX, 2>(QK_READY_DECODER);
            }
            if (n_idx >= s_block_stack) {
                if (n_idx + s_block_stack > n_loop + s_block_stack - 1) {
                    sv_n = cur_kv_seqlen -
                           (n_idx - s_block_stack) * pp_n_scalar;  // delete
                } else {
                    sv_n = pp_n_scalar * s_block_stack;
                }
                uint32_t sv_round_n =
                    (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                uint32_t embed_split_size = 128;
                embed_split_loop_v = 4;
                uint32_t round_embed_split_size =
                    RoundUp<T_BLOCK_SIZE>(embed_split_size);
                for (uint32_t embed_split_idx = 0;
                     embed_split_idx < embed_split_loop_v; ++embed_split_idx) {
                    for (uint32_t split_idx = 0;
                         split_idx < s_block_stack &&
                         n_idx + split_idx < n_loop + s_block_stack;
                         split_idx++) {
                        uint32_t now_idx = n_idx + split_idx;
                        uint32_t l1_kv_pingpong_flag = split_idx % 2;
                        uint32_t l0b_pingpong_flag = split_idx % 2;
                        uint32_t block_table_id = (uint32_t)(*(
                            block_tables_gm +
                            cur_batch * max_num_blocks_per_query +
                            start_kv / block_size + now_idx - s_block_stack));
                        int64_t kv_offset =
                            (int64_t)block_table_id * block_size * stride_kv +
                            block_size * embed_split_idx * embed_split_size;
                        for (int q_i = cur_q_seqlen - 1; q_i >= 0; q_i--) {
                            uint32_t qk_n_v_round = pp_n_scalar;
                            if (now_idx == (n_loop + s_block_stack - 1)) {
                                qk_n_2 =
                                    (cur_kv_seqlen + q_i * kvEndFlag -
                                     (now_idx - s_block_stack) * pp_n_scalar);
                                qk_n_v_round = RoundUp<BLOCK_SIZE>(
                                    cur_kv_seqlen +
                                    (cur_q_seqlen - 1) * kvEndFlag -
                                    (now_idx - s_block_stack) * pp_n_scalar);
                            } else {
                                qk_n_2 = pp_n_scalar;
                            }
                            qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
                            uint32_t l0_p_pingpong_flag =
                                (q_i + split_idx * cur_q_seqlen) % 2;
                            ;
                            uint32_t l0c_pingpong_flag =
                                (q_i + embed_split_idx * cur_q_seqlen) % 2;
                            uint32_t sv_round_n =
                                n_idx + s_block_stack >
                                        n_loop + s_block_stack - 1
                                    ? (sv_n + (cur_q_seqlen - 1) * kvEndFlag +
                                       BLOCK_SIZE - 1) /
                                          BLOCK_SIZE * BLOCK_SIZE
                                    : (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE *
                                          BLOCK_SIZE;
                            if (qk_n_2 != 0) {
                                if (q_i == cur_q_seqlen - 1) {
                                    WAIT_FLAG(
                                        MTE1, MTE2,
                                        4 + l1_kv_pingpong_flag);  // 等待V全部搬入L0B
                                    if constexpr (KInputType ==
                                                  InputFormat::ND_FORMAT) {
                                        gm_to_l1<ArchType::ASCEND_V220,
                                                 IN_KVDTYPE, DataFormat::ND,
                                                 DataFormat::NZ>(
                                            l1kv_buf_addr_tensor
                                                [128 * 128 * 2 + 128 * 64 * 2 +
                                                 l1_kv_pingpong_flag * 128 *
                                                     128],
                                            k_gm_tensor
                                                [(int64_t)block_table_id *
                                                     block_size * stride_kv +
                                                 embed_split_idx *
                                                     embed_split_size],
                                            qk_n_2,        // nValue
                                            qk_round_n_2,  // dstNzC0Stride
                                            0,    // dstNzMatrixStride, unused
                                            128,  // dValue
                                            0,    // dstNzMatrixStride, unused
                                            stride_kv  // srcDValue
                                        );
                                    } else {
                                        gm_to_l1<ArchType::ASCEND_V220,
                                                 IN_KVDTYPE, DataFormat::NZ,
                                                 DataFormat::NZ>(
                                            l1kv_buf_addr_tensor
                                                [128 * 128 * 2 + 128 * 64 * 2 +
                                                 l1_kv_pingpong_flag * 128 *
                                                     128],
                                            k_gm_tensor[kv_offset],
                                            qk_round_n_2, qk_round_n_2,
                                            block_size, 128, 128, 128);
                                    }
                                    SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                                    WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
                                    WAIT_FLAG(M, MTE1, l0b_pingpong_flag + 2);
                                    AscendC::LoadData2dTransposeParams
                                        loadDataParams;
                                    loadDataParams.dstGap = 0;
                                    loadDataParams.startIndex = 0;
                                    loadDataParams.dstFracGap = 0;
                                    loadDataParams.repeatTimes =
                                        round_embed_split_size / T_BLOCK_SIZE;
                                    loadDataParams.srcStride =
                                        qk_n_v_round / T_BLOCK_SIZE;
                                    uint16_t dstGap =
                                        sizeof(IN_DTYPE) == 1 ? 1 : 0;
                                    loadDataParams.dstGap = dstGap;
                                    for (uint32_t l0b_load_idx = 0;
                                         l0b_load_idx <
                                         qk_round_n_2 / T_BLOCK_SIZE;
                                         ++l0b_load_idx) {
                                        AscendC::LoadDataWithTranspose(
                                            l0b_buf_tensor
                                                [l0b_pingpong_flag * 16384 +
                                                 l0b_load_idx *
                                                     RoundUp<16>(
                                                         embed_split_size) *
                                                     T_BLOCK_SIZE],
                                            l1kv_buf_addr_tensor
                                                [128 * 128 * 2 + 128 * 64 * 2 +
                                                 l1_kv_pingpong_flag * 128 *
                                                     128 +
                                                 l0b_load_idx * T_BLOCK_SIZE *
                                                     T_BLOCK_SIZE],
                                            loadDataParams);
                                    }
                                    SET_FLAG(MTE1, MTE2,
                                             4 + l1_kv_pingpong_flag);
                                }
                                // move p from gm to l1
                                uint32_t p_move_head_num = q_heads;
                                if (embed_split_idx == 0 && split_idx == 0 &&
                                    (q_i == cur_q_seqlen - 1)) {
                                    WaitFlagDev(SOFTMAX_READY_DECODER);
                                }

                                WAIT_FLAG(MTE1, MTE2, l0_p_pingpong_flag + 6);
                                gm_to_l1<ArchType::ASCEND_V220, IN_DTYPE,
                                         DataFormat::ND, DataFormat::NZ>(
                                    l1p_buf_addr_tensor[l0_p_pingpong_flag *
                                                        128 * 128],
                                    p_gm_tensor[(uint64_t)block_idx * TMP_SIZE *
                                                    2 * 2 +
                                                q_i * 128 * 512 +
                                                ((n_idx / s_block_stack - 1) %
                                                 2) *
                                                    TMP_SIZE * 2 +
                                                split_idx * 128 * q_heads],
                                    p_move_head_num,  // nValue
                                    (p_move_head_num + 15) / 16 *
                                        16,        // dstNzC0Stride
                                    0,             // dstNzMatrixStride, unused
                                    qk_round_n_2,  // dValue
                                    0,             // dstNzMatrixStride, unused
                                    128            // srcDValue
                                );
                                SET_FLAG(MTE2, MTE1, l0_p_pingpong_flag + 6);
                                WAIT_FLAG(MTE2, MTE1, l0_p_pingpong_flag + 6);

                                // move p from l1 to l0a

                                WAIT_FLAG(M, MTE1, l0_p_pingpong_flag);
                                uint32_t p_load_coeff =
                                    RoundUp<16>(p_move_head_num);
                                for (uint64_t loa_load_idx = 0;
                                     loa_load_idx < p_load_coeff / BLOCK_SIZE;
                                     ++loa_load_idx) {
                                    l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE,
                                               false, DataFormat::VECTOR,
                                               DataFormat::VECTOR>(
                                        l0a_buf_tensor[l0_p_pingpong_flag *
                                                           16384 +
                                                       loa_load_idx *
                                                           qk_round_n_2 *
                                                           BLOCK_SIZE],
                                        l1p_buf_addr_tensor
                                            [l0_p_pingpong_flag * 128 * 128 +
                                             loa_load_idx * T_CUBE_MATRIX_SIZE],
                                        0,
                                        qk_round_n_2 / T_BLOCK_SIZE,  // repeat
                                        0,
                                        p_load_coeff / BLOCK_SIZE,  // srcStride
                                        0,
                                        0  // dstStride
                                    );
                                }
                                SET_FLAG(MTE1, MTE2, l0_p_pingpong_flag + 6);

                                SET_FLAG(MTE1, M, l0_p_pingpong_flag);
                                WAIT_FLAG(MTE1, M, l0_p_pingpong_flag);
                                bool use_unit_flag =
                                    ((sv_n + q_i * kvEndFlag) != 0 &&
                                     (split_idx == s_block_stack - 1 ||
                                      n_idx + split_idx ==
                                          n_loop + s_block_stack - 1)) ||
                                    (cur_q_seqlen == 2 && qk_n_2 == 1 &&
                                     q_i == 1) ||
                                    (n_idx + split_idx ==
                                         n_loop + s_block_stack - 2 &&
                                     q_i == 0 && (cur_kv_seqlen % 128 == 0));
                                auto unit_flag = use_unit_flag ? 0b11 : 0b10;

                                mmad<ArchType::ASCEND_V220, IN_DTYPE, IN_DTYPE,
                                     mm2OutputType, false>(
                                    mm2_l0c_buf_tensor[l0c_pingpong_flag *
                                                       16384],
                                    l0a_buf_tensor[l0_p_pingpong_flag * 16384],
                                    l0b_buf_tensor[l0b_pingpong_flag * 16384],
                                    q_load_coeff,      // m
                                    embed_split_size,  // n
                                    qk_n_2,            // k
                                    split_idx == 0,    // cmatrixInitVal
                                    unit_flag);
                                PIPE_BARRIER(M);
                                if (q_i == 0) {
                                    SET_FLAG(M, MTE1, l0b_pingpong_flag + 2);
                                }
                                SET_FLAG(M, MTE1, l0_p_pingpong_flag);
                                if (use_unit_flag) {
                                    // copy O to gm
                                    l0c_to_gm<ArchType::ASCEND_V220,
                                              DataFormat::ND, mm2CopyType,
                                              mm2OutputType>(
                                        o_tmp_gm_tensor
                                            [(uint64_t)block_idx * TMP_SIZE *
                                                 2 * 2 +
                                             q_i * 128 * 512 +
                                             embed_split_idx *
                                                 round_embed_split_size +
                                             ((n_idx / s_block_stack - 1) % 2) *
                                                 TMP_SIZE * 2],
                                        mm2_l0c_buf_tensor[l0c_pingpong_flag *
                                                           16384],
                                        q_load_coeff,  // MSize
                                        RoundUp<16>(
                                            embed_split_size),  // NSize
                                                                // 32B对齐，防止workspace补齐的位置中有脏数据
                                        RoundUp<16>(q_load_coeff),  // srcStride
                                        round_v,  // dstStride_dst_D
                                        0b11);
                                }
                            } else {
                                SET_FLAG(M, MTE1, l0b_pingpong_flag + 2);
                            }
                        }
                    }
                }
                FftsCrossCoreSync<PIPE_FIX, 2>(UPDATE_READY_DECODER);
            }
        }
    }

   private:
    __gm__ IN_DTYPE *__restrict__ q_gm{nullptr};
    __gm__ IN_ROPE_DTYPE *__restrict__ q_rope_gm{nullptr};
    __gm__ IN_KVDTYPE *__restrict__ k_gm{nullptr};
    __gm__ IN_ROPE_DTYPE *__restrict__ k_rope_gm{nullptr};
    __gm__ OUT_DTYPE *__restrict__ o_gm{nullptr};

    __gm__ int32_t *__restrict__ block_tables_gm{nullptr};
    __gm__ MLATilingData *__restrict__ tiling_gm{nullptr};

    AscendC::GlobalTensor<IN_DTYPE> q_gm_tensor;
    AscendC::GlobalTensor<IN_ROPE_DTYPE> q_rope_gm_tensor;
    AscendC::GlobalTensor<IN_KVDTYPE> k_gm_tensor;
    AscendC::GlobalTensor<IN_ROPE_DTYPE> k_rope_gm_tensor;
    AscendC::GlobalTensor<mm1CopyType> s_gm_tensor;
    AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor;
    AscendC::GlobalTensor<mm2CopyType> o_tmp_gm_tensor;

    const uint32_t l1q_buf_addr_offset = 0;
    const uint32_t l1kv_buf_addr_offset = 147456;
    const uint32_t l1p_buf_addr_offset = 442368;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<IN_DTYPE> l1q_buf_addr_tensor;
    AscendC::LocalTensor<IN_DTYPE> l1kv_buf_addr_tensor;
    AscendC::LocalTensor<IN_DTYPE> l1p_buf_addr_tensor;
    AscendC::LocalTensor<IN_DTYPE> l1v_buf_addr_tensor;
    AscendC::LocalTensor<IN_DTYPE> l0a_buf_tensor =
        buf.GetBuffer<BufferType::ASCEND_L0A, IN_DTYPE>(0);
    AscendC::LocalTensor<IN_DTYPE> l0b_buf_tensor =
        buf.GetBuffer<BufferType::ASCEND_L0B, IN_DTYPE>(0);
    AscendC::LocalTensor<mm1OutputType> mm1_l0c_buf_tensor =
        buf.GetBuffer<BufferType::ASCEND_L0C, mm1OutputType>(0);
    AscendC::LocalTensor<mm2OutputType> mm2_l0c_buf_tensor =
        buf.GetBuffer<BufferType::ASCEND_L0C, mm2OutputType>(0);

    uint32_t num_batches{0};
    uint32_t q_heads{0};
    uint32_t kv_heads{0};
    uint32_t embedding_size{0};
    uint32_t block_size{0};
    uint32_t max_num_blocks_per_query{0};
    uint32_t group_num{0};
    uint32_t stride_kv{0};
    uint32_t stride_kv_rope{0};
    uint32_t m{0};
    uint32_t __v{0};
    uint32_t round_v{0};
    uint32_t process_num{0};
    uint32_t tiling_head_size{0};
    uint32_t tiling_para_size{0};
    uint32_t mask_type{0};
    uint32_t kv_split_core_num{0};
    uint32_t totalTaskNum{0};
    uint32_t flashDecodingTaskNum{0};
    uint32_t maxKVSeqLen{0};
    uint32_t block_dim{0};

    uint32_t cur_qn_blk_size{0};
    uint32_t embed_split_size_v{0};
    uint32_t embed_split_loop_v{1};

    uint32_t l1_pingpong_flag = 0;
    uint32_t l1b_pingpong_flag = 0;
    uint32_t l0_pingpong_flag = 0;
    uint32_t l0b_pingpong_flag = 0;
    uint32_t l0c_pingpong_flag = 0;
    uint32_t l1p_pingpong_flag = 0;
};
#endif

#ifdef __DAV_C220_VEC__
template <TilingKeyType tilingKeyType = TilingKeyType::TILING_HALF_DATA,
          typename IN_DTYPE = half, typename OUT_DTYPE = half,
          bool IS_RING = false, BlockStack blockStack = BlockStack::ONE_FLOW,
          bool flashDecoding = false>
class MLAttentionOp_V {
   public:
    using mm1OutputType = typename AttentionType<tilingKeyType>::mm1OutputType;
    using mm1CopyType = typename AttentionType<tilingKeyType>::mm1CopyType;
    using mmScaleType = typename AttentionType<tilingKeyType>::mmScaleType;
    using mm2OutputType = typename AttentionType<tilingKeyType>::mm2OutputType;
    using mm2CopyType = typename AttentionType<tilingKeyType>::mm2CopyType;
    static constexpr uint32_t T_BLOCK_SIZE = BLOCK_SIZE_32 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_OFFSET = 2 / sizeof(IN_DTYPE);

    __aicore__ __attribute__((always_inline)) inline MLAttentionOp_V() {}

    __aicore__ __attribute__((always_inline)) inline void Init(
        __gm__ uint8_t *__restrict__ qkDescale,
        __gm__ uint8_t *__restrict__ pvDescale,
        __gm__ uint8_t *__restrict__ mask,
        __gm__ uint8_t *__restrict__ attenOut,
        __gm__ uint8_t *__restrict__ workspace,
        __gm__ const MLATilingData *__restrict__ tiling,
        __gm__ uint8_t *__restrict__ o_core_tmp_gm = nullptr,
        __gm__ uint8_t *__restrict__ l_gm = nullptr,
        __gm__ uint8_t *__restrict__ s_rope_out_gm = nullptr) {
        sub_block_idx = static_cast<uint64_t>(GetSubBlockidx());
        SetAtomicnone();
        SetMasknorm();
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        tiling_gm = const_cast<__gm__ MLATilingData *>(tiling);

        __gm__ const MLAHeadParams *__restrict headparams =
            &(tiling_gm->MLAHead);
        __gm__ const NdMLATiling *__restrict ndtiling =
            &(tiling_gm->ndMLATiling);

        num_batches = (uint32_t)(headparams->batch);
        q_heads = (uint32_t)(headparams->numHeads);
        embedding_size = (uint32_t)(headparams->headDim);
        block_size = (uint32_t)(headparams->blockSize);
        max_num_blocks_per_query = (uint32_t)(headparams->maxBlocks);
        tor = (float)(headparams->tor);
        num_kv_heads = (uint32_t)(headparams->kvHeads);
        tiling_head_size = (uint32_t)(headparams->headSize);
        tiling_para_size = (uint32_t)(headparams->paraSize);
        cur_qn_blk_size = (uint32_t)(ndtiling->curQNBlockTile);
        mask_type = (uint32_t)(headparams->maskTypeND);
        totalTaskNum = (uint32_t)(headparams->taskNum);
        maxKVSeqLen = (uint32_t)(ndtiling->maxKVseqlen);
        kv_split_core_num = (uint32_t)(headparams->kvSplitNum);
        flashDecodingTaskNum = (uint32_t)(headparams->splitTaskNum);
        block_dim = (uint32_t)(headparams->blockDim);

        bool mtpTp1Flag = (tiling_para_size == (uint32_t)(7));
        const uint64_t basicWorkSpaceHalf = block_dim * TMP_SIZE * 2;
        const uint64_t basicWorkSpaceFloat = block_dim * TMP_SIZE * 4;
        const uint64_t sWorkSpaceSize =
            mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        const uint64_t pWorkSpaceSize =
            mtpTp1Flag ? basicWorkSpaceHalf * 4 : basicWorkSpaceHalf * 2;
        const uint64_t oTempWorkSpaceSize =
            mtpTp1Flag ? basicWorkSpaceFloat * 4 : basicWorkSpaceFloat * 2;
        const uint64_t goWorkSpaceSize =
            mtpTp1Flag ? basicWorkSpaceFloat * 2 : basicWorkSpaceFloat;
        const uint64_t defaultSysWorkspaceSize910B = 16U * 1024U * 1024U;

        __gm__ mm1CopyType *s_gm =
            reinterpret_cast<__gm__ mm1CopyType *>(workspace);
        __gm__ IN_DTYPE *p_gm = reinterpret_cast<__gm__ IN_DTYPE *>(
            workspace + sWorkSpaceSize + 512);
        __gm__ mm2CopyType *o_tmp_gm = reinterpret_cast<__gm__ mm2CopyType *>(
            workspace + sWorkSpaceSize + 512 + pWorkSpaceSize);
        __gm__ float *go_gm = reinterpret_cast<__gm__ float *>(
            workspace + sWorkSpaceSize + 512 + pWorkSpaceSize +
            oTempWorkSpaceSize);

        o_gm = reinterpret_cast<__gm__ OUT_DTYPE *>(attenOut);
        o_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_DTYPE *>(o_gm));
        mask_gm_tensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ OUT_DTYPE *>(mask));

        s_gm_tensor.SetGlobalBuffer(s_gm, sWorkSpaceSize);
        p_gm_tensor.SetGlobalBuffer(p_gm, pWorkSpaceSize);
        o_tmp_gm_tensor.SetGlobalBuffer(o_tmp_gm, oTempWorkSpaceSize);
        go_gm_tensor.SetGlobalBuffer(go_gm, goWorkSpaceSize);

        o_core_tmp_gm_tensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(o_core_tmp_gm));
        l_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(l_gm));
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            deq_scale_gm_tensor_q1.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(qkDescale));
            deq_scale_gm_tensor_k1.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(pvDescale));
            s_rope_gm_tensor.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(s_rope_out_gm));
        }

        go_flag_scalar = 1;
        gl_flag_scalar = 1;

        __k = embedding_size;
        round_k = RoundUp<T_BLOCK_SIZE>(__k);
        __v = embedding_size;
        round_v = RoundUp<BLOCK_SIZE>(__v);
    }

    __aicore__ __attribute__((always_inline)) inline void Run() {
        SET_FLAG(MTE3, V, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID2);
        SET_FLAG(MTE3, MTE2, EVENT_ID3);
        SET_FLAG(MTE3, MTE2, EVENT_ID4);
        SET_FLAG(V, MTE2, EVENT_ID4);
        SET_FLAG(V, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, V, EVENT_ID2);
        SET_FLAG(V, MTE2, EVENT_ID2);

        uint64_t cur_batch = 0;

        uint32_t q_block_num_per_batch =
            (q_heads + cur_qn_blk_size - 1) / cur_qn_blk_size;
        uint32_t process_num = q_block_num_per_batch * num_batches;

        __gm__ const NdMLATiling *__restrict ndtiling =
            &(tiling_gm->ndMLATiling);
        for (uint32_t process = block_idx; process < process_num;
             process += (uint32_t)block_num) {  // for task
            cur_batch = process / q_block_num_per_batch;
            if (cur_batch >= num_batches) break;

            uint32_t start_core_idx =
                (cur_batch * q_block_num_per_batch) % block_num;
            uint32_t q_seqlen = (uint32_t)(ndtiling->qSeqLen[cur_batch]);
            uint32_t kv_seqlen = (uint32_t)(ndtiling->kvSeqlen[cur_batch]);
            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t kv_seqlen_align =
                (kv_seqlen + block_size - 1) / block_size * block_size;

            uint32_t start_head =
                (process % q_block_num_per_batch) * cur_qn_blk_size;
            uint32_t start_kv = 0;
            uint32_t cur_q_seq_len = q_seqlen;
            uint32_t cur_kv_seqlen = kv_seqlen;
            uint32_t cur_head_num = cur_qn_blk_size;
            uint32_t cur_nIndx = 0;
            InnerRunVectorChange(cur_batch, start_head, cur_nIndx,
                                 cur_q_seq_len, cur_kv_seqlen, cur_head_num,
                                 512, embed_split_loop_v_former);
        }

        WAIT_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID3);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID4);
        WAIT_FLAG(MTE3, V, EVENT_ID2);
        WAIT_FLAG(V, MTE2, EVENT_ID2);
    }

    __aicore__ __attribute__((always_inline)) inline void RunTP1() {
        SET_FLAG(MTE3, V, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID1);
        SET_FLAG(MTE3, MTE2, EVENT_ID2);
        SET_FLAG(MTE3, MTE2, EVENT_ID3);
        SET_FLAG(MTE3, MTE2, EVENT_ID4);
        SET_FLAG(MTE3, MTE2, EVENT_ID5);
        SET_FLAG(V, MTE2, EVENT_ID0);
        SET_FLAG(V, MTE2, EVENT_ID1);
        SET_FLAG(V, MTE2, EVENT_ID2);
        SET_FLAG(V, MTE2, EVENT_ID4);
        SET_FLAG(V, MTE2, EVENT_ID7);
        SET_FLAG(MTE3, V, EVENT_ID2);

        __gm__ const NdMLADecodingMtpTilingTP1 *__restrict ndMtpTiling =
            &(tiling_gm->ndMLADecodingMtpTilingTP1);
        for (uint32_t process = block_idx; process < totalTaskNum;
             process += block_num) {
            uint32_t offset_tiling =
                tiling_head_size + tiling_para_size * process;
            uint32_t cur_batch = (uint32_t)(ndMtpTiling->batchIdx[process]);
            uint32_t q_row_id = (uint32_t)(ndMtpTiling->qSeqLen[process]);
            uint32_t kv_seqlen = (uint32_t)(ndMtpTiling->kvBlockIdx[process]);
            uint32_t q_seqlen = (uint32_t)(ndMtpTiling->prevKVSeqlen[process]);
            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t kv_seqlen_align =
                (kv_seqlen + block_size - 1) / block_size * block_size;
            uint32_t start_head = q_row_id * q_heads;
            uint32_t start_kv = 0;
            uint32_t cur_q_seq_len = q_seqlen;
            uint32_t cur_kv_seqlen = kv_seqlen;
            uint32_t cur_head_num = q_heads;
            uint32_t cur_nIndx = 0;
            InnerRunVectorChangeTP1(
                cur_batch, start_head, cur_nIndx, cur_q_seq_len, cur_kv_seqlen,
                cur_head_num, offset_tiling, 512, embed_split_loop_v_former);
        }

        if constexpr (flashDecoding) {
            for (uint32_t process = block_idx; process < flashDecodingTaskNum;
                 process += (uint32_t)block_num) {
                uint32_t offset_tiling =
                    tiling_head_size +
                    tiling_para_size * (totalTaskNum + process);
                uint32_t cur_batch =
                    (uint32_t)(ndMtpTiling->batchIdx[totalTaskNum + process]);
                uint32_t q_seqlen =
                    (uint32_t)(ndMtpTiling->qSeqLen[totalTaskNum + process]);
                uint32_t kv_seqlen =
                    (uint32_t)(ndMtpTiling->kvBlockIdx[totalTaskNum + process]);
                uint32_t prev_task =
                    (uint32_t)(ndMtpTiling->batchSize[totalTaskNum + process]);
                uint32_t split_num =
                    (uint32_t)(ndMtpTiling->splitNum[totalTaskNum + process]);
                uint32_t prev_split_num =
                    (uint32_t)(ndMtpTiling
                                   ->prevSplitNumSum[totalTaskNum + process]);

                if (kv_seqlen == 0) {
                    continue;
                }
                uint32_t kv_seqlen_align =
                    (kv_seqlen + block_size - 1) / block_size * block_size;
                uint32_t start_head = prev_task * q_heads;
                uint32_t cur_q_seq_len = q_seqlen;
                if constexpr (tilingKeyType ==
                              TilingKeyType::TILING_INT8_DATA) {
                    cur_q_seq_len = 1;
                }
                uint32_t cur_kv_seqlen = kv_seqlen;
                uint32_t cur_head_num = q_heads;
                uint32_t cur_nIndx = process - prev_split_num;
                uint32_t kvEndFlag =
                    (process - prev_split_num) % split_num == split_num - 1 ? 1
                                                                            : 0;
                InnerRunVectorChangeTP1<true>(
                    cur_batch, start_head, cur_nIndx, cur_q_seq_len,
                    cur_kv_seqlen, cur_head_num, offset_tiling, 512,
                    embed_split_loop_v_former, prev_split_num, split_num,
                    kvEndFlag);
            }
        }
        WAIT_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID3);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID5);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID1);
        WAIT_FLAG(V, MTE2, EVENT_ID2);
        WAIT_FLAG(V, MTE2, EVENT_ID4);
        WAIT_FLAG(V, MTE2, EVENT_ID7);
        WAIT_FLAG(MTE3, V, EVENT_ID2);

        if constexpr (flashDecoding) {
            int reduce_flag_id = 3;
            FftsCrossCoreSync<PIPE_MTE3, 0>(reduce_flag_id);
            WaitFlagDev(reduce_flag_id);
            CombineScaleBlock(num_batches, q_heads, kv_split_core_num,
                              embedding_size);
        }
    }

   private:
    __aicore__ __attribute__((always_inline)) inline void ReduceMaxRepeatM(
        const AscendC::LocalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src,
        const AscendC::LocalTensor<float> &tempTensor, uint32_t sub_m,
        uint32_t qk_n, uint32_t qk_round_n) {
        if (qk_n <= FLOAT_VECTOR_SIZE) {
            __set_mask(qk_n);
            cmax_v<ArchType::ASCEND_V220, float,
                   AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
                dst, src,
                sub_m,                         // repeat
                1,                             // dstRepeatStride
                1,                             // srcBlockStride
                qk_round_n / FLOAT_BLOCK_SIZE  // srcRepeatStride
            );
        } else {
            ub_to_ub<ArchType::ASCEND_V220, float>(
                tempTensor, src,
                0,                              // sid
                sub_m,                          // nBurst
                HALF_VECTOR_SIZE / BLOCK_SIZE,  // lenBurst
                (qk_round_n - FLOAT_VECTOR_SIZE) / FLOAT_BLOCK_SIZE,  // srcGap
                0                                                     // dstGap
            );
            PIPE_BARRIER(V);
            for (uint32_t rowmax_idx = 1; rowmax_idx < qk_n / FLOAT_VECTOR_SIZE;
                 ++rowmax_idx) {
                max_v<ArchType::ASCEND_V220, float>(
                    tempTensor, tempTensor, src[rowmax_idx * FLOAT_VECTOR_SIZE],
                    sub_m,                         // repeat
                    1,                             // dstBlockStride
                    1,                             // src0BlockStride
                    1,                             // src1BlockStride
                    8,                             // dstRepeatStride
                    8,                             // src0RepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE  // src1RepeatStride
                );
                PIPE_BARRIER(V);
            }
            if (qk_n % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(qk_n % FLOAT_VECTOR_SIZE);
                max_v<ArchType::ASCEND_V220, float>(
                    tempTensor, tempTensor,
                    src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    sub_m,                         // repeat
                    1,                             // dstBlockStride
                    1,                             // src0BlockStride
                    1,                             // src1BlockStride
                    8,                             // dstRepeatStride
                    8,                             // src0RepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE  // src1RepeatStride
                );
            }
            PIPE_BARRIER(V);
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            cmax_v<ArchType::ASCEND_V220, float,
                   AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
                dst, tempTensor,
                sub_m,  // repeat
                1,      // dstRepeatStride
                1,      // srcBlockStride
                8       // srcRepeatStride
            );
        }
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        PIPE_BARRIER(V);
    }

    __aicore__ __attribute__((always_inline)) inline void ReduceSumRepeatM(
        const AscendC::LocalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src, uint32_t sub_m, uint32_t qk_n,
        uint32_t qk_round_n) {
        if (qk_n <= FLOAT_VECTOR_SIZE) {
            __set_mask(qk_n);
            cadd_v<ArchType::ASCEND_V220, float>(
                dst, src,
                sub_m,                         // repeat
                1,                             // dstRepeatStride
                1,                             // srcBlockStride
                qk_round_n / FLOAT_BLOCK_SIZE  // srcRepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else {
            for (uint32_t rowsum_idx = 1; rowsum_idx < qk_n / FLOAT_VECTOR_SIZE;
                 ++rowsum_idx) {
                add_v<ArchType::ASCEND_V220, float>(
                    src, src, src[rowsum_idx * FLOAT_VECTOR_SIZE],
                    sub_m,                          // repeat
                    1,                              // dstBlockStride
                    1,                              // src0BlockStride
                    1,                              // src1BlockStride
                    qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE   // src1RepeatStride
                );
                PIPE_BARRIER(V);
            }
            if (qk_n % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(qk_n % FLOAT_VECTOR_SIZE);
                add_v<ArchType::ASCEND_V220, float>(
                    src, src, src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    sub_m,                          // repeat
                    1,                              // dstBlockStride
                    1,                              // src0BlockStride
                    1,                              // src1BlockStride
                    qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    qk_round_n / FLOAT_BLOCK_SIZE   // src1RepeatStride
                );
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            PIPE_BARRIER(V);

            cadd_v<ArchType::ASCEND_V220, float>(
                dst, src,
                sub_m,                         // repeat
                1,                             // dstRepeatStride
                1,                             // srcBlockStride
                qk_round_n / FLOAT_BLOCK_SIZE  // srcRepeatStride
            );
        }
    }

    __aicore__ __attribute__((always_inline)) inline void TensorSubValueRepeatM(
        const AscendC::LocalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src,
        const AscendC::LocalTensor<float> &MaxTensor,
        const AscendC::LocalTensor<float> &tempMaxTensor, uint32_t sub_m,
        uint32_t round_sub_m, uint32_t qk_n, uint32_t qk_round_n) {
        brcb_v<ArchType::ASCEND_V220, uint32_t>(
            tempMaxTensor.ReinterpretCast<uint32_t>(),
            MaxTensor.ReinterpretCast<uint32_t>(),
            1,                              // dstBlockStride
            8,                              // dstRepeatStride
            round_sub_m / FLOAT_BLOCK_SIZE  // repeat
        );
        PIPE_BARRIER(V);
        for (uint32_t sub_v_idx = 0; sub_v_idx < qk_n / FLOAT_VECTOR_SIZE;
             ++sub_v_idx) {
            sub_v<ArchType::ASCEND_V220, float>(
                dst[sub_v_idx * FLOAT_VECTOR_SIZE],
                src[sub_v_idx * FLOAT_VECTOR_SIZE], tempMaxTensor,
                sub_m,                          // repeat
                1,                              // dstBlockStride
                1,                              // src0BlockStride
                0,                              // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1                               // src1RepeatStride
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            sub_v<ArchType::ASCEND_V220, float>(
                dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tempMaxTensor,
                sub_m,                          // repeat
                1,                              // dstBlockStride
                1,                              // src0BlockStride
                0,                              // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1                               // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
    }

    __aicore__ __attribute__((always_inline)) inline void TensorDivRepeatM(
        const AscendC::LocalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src,
        const AscendC::LocalTensor<float> &src1, uint32_t sub_m, uint32_t qk_n,
        uint32_t qk_round_n) {
        PIPE_BARRIER(V);
        for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE;
             ++vadd_idx) {
            div_v<ArchType::ASCEND_V220, float>(
                dst[vadd_idx * FLOAT_VECTOR_SIZE],
                src[vadd_idx * FLOAT_VECTOR_SIZE], src1,
                sub_m,                          // repeat
                1,                              // dstBlockStride
                1,                              // src0BlockStride
                0,                              // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1                               // src1RepeatStride
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            div_v<ArchType::ASCEND_V220, float>(
                dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE], src1,
                sub_m,                          // repeat
                1,                              // dstBlockStride
                1,                              // src0BlockStride
                0,                              // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1                               // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
    }

    __aicore__ __attribute__((always_inline)) inline void TensorMulRepeatM(
        const AscendC::LocalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src,
        const AscendC::LocalTensor<float> &src1, uint32_t sub_m, uint32_t qk_n,
        uint32_t qk_round_n, uint32_t src1BlockStride) {
        PIPE_BARRIER(V);
        for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE;
             ++vadd_idx) {
            mul_v<ArchType::ASCEND_V220, float>(
                dst[vadd_idx * FLOAT_VECTOR_SIZE],
                src[vadd_idx * FLOAT_VECTOR_SIZE], src1,
                sub_m,                          // repeat
                1,                              // dstBlockStride
                1,                              // src0BlockStride
                src1BlockStride,                // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1                               // src1RepeatStride
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            mul_v<ArchType::ASCEND_V220, float>(
                dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE], src1,
                sub_m,                          // repeat
                1,                              // dstBlockStride
                1,                              // src0BlockStride
                src1BlockStride,                // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1                               // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
    }

    __aicore__ __attribute__((always_inline)) inline void DeQuantPerHeadProc(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<int32_t> temp,
        AscendC::LocalTensor<float> deScaleUb,
        AscendC::LocalTensor<float> tempScale,
        AscendC::LocalTensor<float> quantScale, uint32_t sub_m, uint32_t qk_n,
        uint32_t qk_round_n, bool online) {
        if (online) {
            // if dequant online need mul p quant scale
            // 理论上这里repeat不会超过1
            __set_mask(sub_m % FLOAT_VECTOR_SIZE);
            mul_v<ArchType::ASCEND_V220, float>(
                quantScale, deScaleUb, quantScale,
                CeilDiv<FLOAT_VECTOR_SIZE>(sub_m),  // repeat
                1,                                  // dstBlockStride
                1,                                  // src0BlockStride
                1,                                  // src1BlockStride
                8,                                  // dstRepeatStride
                8,                                  // src0RepeatStride
                8                                   // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            PIPE_BARRIER(V);
        } else {
            ub_to_ub<ArchType::ASCEND_V220, float>(
                quantScale, deScaleUb,
                0,  // sid
                1,  // nBurst
                RoundUp<FLOAT_BLOCK_SIZE>(sub_m) /
                    FLOAT_BLOCK_SIZE,  // lenBurst
                0,                     // srcGap
                0                      // dstGap
            );
            PIPE_BARRIER(V);
        }
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        brcb_v<ArchType::ASCEND_V220, uint32_t>(
            tempScale.template ReinterpretCast<uint32_t>(),
            quantScale.template ReinterpretCast<uint32_t>(),
            1,                                     // dstBlockStrides
            8,                                     // dstRepeatStride
            RoundUp<16>(sub_m) / FLOAT_BLOCK_SIZE  // repeat
        );
        PIPE_BARRIER(V);
        uint32_t count = sub_m * qk_round_n;
        uint32_t repeat_times =
            (count + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
        if (repeat_times < 255) {
            conv_v<ArchType::ASCEND_V220, int32_t, float>(
                dst,           // dst
                temp,          // src
                repeat_times,  // repeat_times
                1,             // dstBlockStride
                1,             // srcBlockStride
                8,             // dstRepeatStride
                8              // srcRepeatStride
            );
        } else {
            for (uint64_t vconv_idx = 0; vconv_idx < 2;
                 ++vconv_idx) {  // 一次迭代做一半，循环防止超出 repeat
                                 // 范围（<=255)
                conv_v<ArchType::ASCEND_V220, int32_t, float>(
                    dst[vconv_idx * count / 2],   // dst
                    temp[vconv_idx * count / 2],  // src
                    (count / 2 + FLOAT_VECTOR_SIZE - 1) /
                        FLOAT_VECTOR_SIZE,  // repeat_times
                    1,                      // dstBlockStride
                    1,                      // srcBlockStride
                    8,                      // dstRepeatStride
                    8                       // srcRepeatStride
                );
            }
        }
        TensorMulRepeatM(dst, dst, tempScale, sub_m, qk_n, qk_round_n, 0);
        PIPE_BARRIER(V);
    }
    __aicore__ __attribute__((always_inline)) inline void DeQuantPerHeadImpl(
        const AscendC::GlobalTensor<mmScaleType> &deScaleGm,
        const AscendC::GlobalTensor<int32_t> &src,
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<int32_t> temp,
        AscendC::LocalTensor<mmScaleType> deScaleUb,
        AscendC::LocalTensor<mmScaleType> tempScale,
        AscendC::LocalTensor<float> quantScale, uint32_t sub_m, uint32_t qk_n,
        uint32_t qk_round_n, bool online, bool move_tensor) {
        gm_to_ub_align<ArchType::ASCEND_V220, mmScaleType>(
            deScaleUb, deScaleGm,
            0,                            // sid
            1,                            // nBurst
            sub_m * sizeof(mmScaleType),  // lenBurst
            0,                            // leftPaddingNum
            0,                            // rightPaddingNum
            0,                            // srcGap
            0                             // dstGap
        );
        if (move_tensor) {
            gm_to_ub<ArchType::ASCEND_V220, int32_t>(
                temp, src,
                0,                                              // sid
                1,                                              // nBurst
                CeilDiv<FLOAT_BLOCK_SIZE>(sub_m * qk_round_n),  // lenBurst
                0,                                              // srcGap
                0                                               // dstGap
            );
        }
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);

        DeQuantPerHeadProc(dst, temp, deScaleUb, tempScale, quantScale, sub_m,
                           qk_n, qk_round_n, online);
    }
    __aicore__ __attribute__((always_inline)) inline void QuantScaleCal(
        const AscendC::LocalTensor<float> &quant_online_scale_ub,
        const AscendC::LocalTensor<float> &local_rowmax_ub,
        const AscendC::LocalTensor<float> &hat_rowmax_ub,
        const AscendC::LocalTensor<float> &tmp_ub, uint32_t m,
        uint32_t round_m) {
        float quantMax = (float)1 / (float)127;
        if (m < 64) __set_mask(m % FLOAT_VECTOR_SIZE);
        sub_v<ArchType::ASCEND_V220, float>(
            quant_online_scale_ub, local_rowmax_ub, hat_rowmax_ub,
            CeilDiv<FLOAT_VECTOR_SIZE>(round_m),  // repeat
            1,                                    // dstBlockStride
            1,                                    // src0BlockStride
            1,                                    // src1BlockStride
            8,                                    // dstRepeatStride
            8,                                    // src0RepeatStride
            8                                     // src1RepeatStride
        );
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V220, float>(
            quant_online_scale_ub, quant_online_scale_ub,
            CeilDiv<FLOAT_VECTOR_SIZE>(round_m),  // repeat
            1,                                    // dstBlockStride
            1,                                    // srcBlockStride
            8,                                    // dstRepeatStride
            8                                     // srcRepeatStride
        );
        PIPE_BARRIER(V);
        muls_v<ArchType::ASCEND_V220, float>(
            quant_online_scale_ub, quant_online_scale_ub, quantMax,
            CeilDiv<FLOAT_VECTOR_SIZE>(round_m),  // repeat
            1,                                    // dstBlockStride
            1,                                    // srcBlockStride
            8,                                    // dstRepeatStride
            8                                     // srcRepeatStride
        );
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        brcb_v<ArchType::ASCEND_V220, uint32_t>(
            tmp_ub.ReinterpretCast<uint32_t>(),
            quant_online_scale_ub.ReinterpretCast<uint32_t>(),
            1,                          // dstBlockStride
            8,                          // dstRepeatStride
            round_m / FLOAT_BLOCK_SIZE  // repeat
        );
        PIPE_BARRIER(V);
    }
    __aicore__ __attribute__((always_inline)) inline void QuantPerTokenImpl(
        const AscendC::LocalTensor<IN_DTYPE> &dst,
        const AscendC::LocalTensor<float> &src,
        const AscendC::LocalTensor<float> &scale, uint32_t sub_m, uint32_t qk_n,
        uint32_t qk_round_n, uint32_t pQuantOnline) {
        if (pQuantOnline) {
            // scr / scale 提函数
            TensorDivRepeatM(dst.template ReinterpretCast<float>(), src, scale,
                             sub_m, qk_n, qk_round_n);
        } else {
            // scr * scale
            TensorMulRepeatM(dst.template ReinterpretCast<float>(), src, scale,
                             sub_m, qk_n, qk_round_n, 0);
        }
        // src fp32 -> casttofp16 -> casttoint8
        uint32_t count = sub_m * qk_round_n;
        uint32_t repeat_times =
            (count + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
        if (repeat_times < 255) {
            conv_v<ArchType::ASCEND_V220, float, half>(
                dst.template ReinterpretCast<half>(),   // dst
                dst.template ReinterpretCast<float>(),  // src
                repeat_times,                           // repeat_times
                1,                                      // dstBlockStride
                1,                                      // srcBlockStride
                4,                                      // dstRepeatStride
                8                                       // srcRepeatStride
            );
        } else {
            for (uint64_t vconv_idx = 0; vconv_idx < 2;
                 ++vconv_idx) {  // 一次迭代做一半，循环防止超出 repeat
                                 // 范围（<=255)
                conv_v<ArchType::ASCEND_V220, float, half>(
                    dst.template ReinterpretCast<half>()[vconv_idx * count /
                                                         2],  // dst
                    dst.template ReinterpretCast<float>()[vconv_idx * count /
                                                          2],  // src
                    (count / 2 + FLOAT_VECTOR_SIZE - 1) /
                        FLOAT_VECTOR_SIZE,  // repeat_times
                    1,                      // dstBlockStride
                    1,                      // srcBlockStride
                    4,                      // dstRepeatStride
                    8                       // srcRepeatStride
                );
            }
        }
        PIPE_BARRIER(V);
        for (uint32_t row_idx = 0; row_idx < qk_n / HALF_VECTOR_SIZE;
             ++row_idx) {
            AscendC::Cast<int8_t, half, false>(
                dst.template ReinterpretCast<int8_t>()[row_idx *
                                                       HALF_VECTOR_SIZE],
                dst.template ReinterpretCast<half>()[row_idx *
                                                     HALF_VECTOR_SIZE],
                AscendC::RoundMode::CAST_RINT, (uint64_t)0, sub_m,
                {1, 1, (uint8_t)((qk_round_n) / BLOCK_SIZE),
                 (uint8_t)(qk_round_n / BLOCK_SIZE)});
        }
        if (qk_n % HALF_VECTOR_SIZE > 0) {
            __set_mask(qk_n % HALF_VECTOR_SIZE);
            AscendC::Cast<int8_t, half, false>(
                dst.template ReinterpretCast<int8_t>()[qk_n / HALF_VECTOR_SIZE *
                                                       HALF_VECTOR_SIZE],
                dst.template ReinterpretCast<half>()[qk_n / HALF_VECTOR_SIZE *
                                                     HALF_VECTOR_SIZE],
                AscendC::RoundMode::CAST_RINT, (uint64_t)0, sub_m,
                {1, 1, (uint8_t)((qk_round_n) / BLOCK_SIZE),
                 (uint8_t)(qk_round_n / BLOCK_SIZE)});
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
    }

    __aicore__ __attribute__((always_inline)) inline void CalcScale(
        AscendC::LocalTensor<float> &lTensor,
        AscendC::LocalTensor<float> &lTmptensor,
        AscendC::LocalTensor<float> &lMaxTensor, uint32_t &headNum,
        uint32_t &cur_kv_split_core_num) {
        uint64_t mask = 64;
        uint32_t headNumAlign8 = (headNum + 8 - 1) / 8 * 8;
        uint32_t rowLength = headNumAlign8 * FLOAT_BLOCK_SIZE;
        // 转置，64 对齐 : headNum * (splitKV * 8) -> splitKV * rowLength
        AscendC::Copy(lTensor, lTmptensor, mask, cur_kv_split_core_num,
                      {1, static_cast<uint16_t>(cur_kv_split_core_num), 8, 1});
        PIPE_BARRIER(V);
        /*
            l_max = np.max(l, axis=1, keepdims=True)
            l_tmp = np.exp(l - l_max)
            l_tmp = np.sum(l_tmp, axis=1, keepdims=True)
            l_tmp = np.log(l_tmp) + l_max
            scale = np.exp(l - l_tmp)
        */
        max_v<ArchType::ASCEND_V220, float>(lMaxTensor, lTensor,
                                            lTensor[rowLength],
                                            1,  // repeat
                                            1,  // dstBlockStride
                                            1,  // src0BlockStride
                                            1,  // src1BlockStride
                                            8,  // dstRepeatStride
                                            8,  // src0RepeatStride
                                            8   // src1RepeatStride
        );
        PIPE_BARRIER(V);
        for (uint32_t i = 2; i < cur_kv_split_core_num; i++) {
            max_v<ArchType::ASCEND_V220, float>(lMaxTensor, lMaxTensor,
                                                lTensor[i * rowLength],
                                                1,  // repeat
                                                1,  // dstBlockStride
                                                1,  // src0BlockStride
                                                1,  // src1BlockStride
                                                8,  // dstRepeatStride
                                                8,  // src0RepeatStride
                                                8   // src1RepeatStride
            );
            PIPE_BARRIER(V);
        }
        sub_v<ArchType::ASCEND_V220, float>(lTmptensor, lTensor, lMaxTensor,
                                            cur_kv_split_core_num,  // repeat
                                            1,  // dstBlockStride
                                            1,  // src0BlockStride
                                            1,  // src1BlockStride
                                            8,  // dstRepeatStride
                                            8,  // src0RepeatStride
                                            0   // src1RepeatStride
        );
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V220, float>(lTmptensor, lTmptensor,
                                            cur_kv_split_core_num,  // repeat
                                            1,  // dstBlockStride
                                            1,  // srcBlockStride
                                            8,  // dstRepeatStride
                                            8   // srcRepeatStride
        );
        PIPE_BARRIER(V);
        for (uint32_t i = 1; i < cur_kv_split_core_num; i++) {
            add_v<ArchType::ASCEND_V220, float>(lTmptensor, lTmptensor,
                                                lTmptensor[i * rowLength],
                                                1,  // repeat
                                                1,  // dstBlockStride
                                                1,  // src0BlockStride
                                                1,  // src1BlockStride
                                                8,  // dstRepeatStride
                                                8,  // src0RepeatStride
                                                8   // src1RepeatStride
            );
            PIPE_BARRIER(V);
        }
        ln_v<ArchType::ASCEND_V220, float>(lTmptensor, lTmptensor,
                                           1,  // repeat
                                           1,  // dstBlockStride
                                           1,  // srcBlockStride
                                           8,  // dstRepeatStride
                                           8   // srcRepeatStride
        );
        PIPE_BARRIER(V);
        add_v<ArchType::ASCEND_V220, float>(lTmptensor, lTmptensor, lMaxTensor,
                                            1,  // repeat
                                            1,  // dstBlockStride
                                            1,  // src0BlockStride
                                            1,  // src1BlockStride
                                            8,  // dstRepeatStride
                                            8,  // src0RepeatStride
                                            8   // src1RepeatStride
        );
        PIPE_BARRIER(V);
        sub_v<ArchType::ASCEND_V220, float>(lTensor, lTensor, lTmptensor,
                                            cur_kv_split_core_num,  // repeat
                                            1,  // dstBlockStride
                                            1,  // src0BlockStride
                                            1,  // src1BlockStride
                                            8,  // dstRepeatStride
                                            8,  // src0RepeatStride
                                            0   // src1RepeatStride
        );
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V220, float>(lTensor, lTensor,
                                            cur_kv_split_core_num,  // repeat
                                            1,  // dstBlockStride
                                            1,  // srcBlockStride
                                            8,  // dstRepeatStride
                                            8   // srcRepeatStride
        );
        PIPE_BARRIER(V);
    }

    __aicore__ __attribute__((always_inline)) inline void CombineOutByMultiHead(
        AscendC::GlobalTensor<float> gmLoTensor,
        AscendC::GlobalTensor<OUT_DTYPE> gmOutTensor,
        AscendC::LocalTensor<float> scaleTensor, uint32_t headProceesNum,
        uint32_t scaleRowLenth, uint32_t cur_kv_split_core_num, uint32_t i) {
        constexpr uint32_t ubuf_offset = 3 * STAGE2_UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t lo_ubuf_size =
            ubuf_offset + STAGE2_UB_UINT8_BLOCK_SIZE *
                              4;  // 8 * 512 * doublebuffer * sizeof(float)
        constexpr uint32_t to_ubuf_size =
            lo_ubuf_size + STAGE2_UB_UINT8_BLOCK_SIZE *
                               4;  // 8 * 512 * doublebuffer * sizeof(float)
        constexpr uint32_t go_ubuf_size =
            to_ubuf_size +
            STAGE2_UB_UINT8_BLOCK_SIZE * 2;  // 8 * 512 * sizeof(float)
        constexpr uint32_t go16_ubuf_size =
            go_ubuf_size +
            STAGE2_UB_UINT8_BLOCK_SIZE;  // 8 * 512 * sizeof(float) / 2
        constexpr uint32_t lBrcb_size = FLOAT_VECTOR_SIZE * 8;
        AscendC::LocalTensor<float> loTensor =
            buf.GetBuffer<BufferType::ASCEND_UB, float>(ubuf_offset);
        AscendC::LocalTensor<float> toTensor =
            buf.GetBuffer<BufferType::ASCEND_UB, float>(lo_ubuf_size);
        AscendC::LocalTensor<float> goTensor =
            buf.GetBuffer<BufferType::ASCEND_UB, float>(to_ubuf_size);
        AscendC::LocalTensor<OUT_DTYPE> go16tensor =
            buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(go_ubuf_size);
        AscendC::LocalTensor<float> lBrcb =
            buf.GetBuffer<BufferType::ASCEND_UB, float>(go16_ubuf_size);

        uint32_t __k0 = embedding_size;
        uint32_t dataCount = headProceesNum * __k0;
        uint32_t repeat = dataCount / FLOAT_VECTOR_SIZE;
        uint32_t pingpong_flag = i % 2;

        AscendC::Copy(lBrcb[pingpong_flag * lBrcb_size], scaleTensor, 64,
                      headProceesNum, {1, 0, 8, 1});
        PIPE_BARRIER(V);

        WAIT_FLAG(V, MTE2, pingpong_flag);
        gm_to_ub<ArchType::ASCEND_V220, float>(
            loTensor[pingpong_flag * dataCount], gmLoTensor,
            0,                                                      // sid
            headProceesNum,                                         // nBurst
            __k0 / FLOAT_BLOCK_SIZE,                                // lenBurst
            __k0 / FLOAT_BLOCK_SIZE * (cur_kv_split_core_num - 1),  // srcGap
            0                                                       // dstGap
        );
        SET_FLAG(MTE2, V, pingpong_flag);
        WAIT_FLAG(MTE2, V, pingpong_flag);
        mul_v<ArchType::ASCEND_V220, float>(goTensor,
                                            loTensor[pingpong_flag * dataCount],
                                            lBrcb[pingpong_flag * lBrcb_size],
                                            repeat,  // repeat
                                            1,       // dstBlockStride
                                            1,       // src0BlockStride
                                            0,       // src1BlockStride
                                            8,       // dstRepeatStride
                                            8,       // src0RepeatStride
                                            1        // src1RepeatStride
        );
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE2, pingpong_flag);
        pingpong_flag = 1 - pingpong_flag;
        for (uint32_t i = 1; i < cur_kv_split_core_num; i++) {
            AscendC::Copy(lBrcb[pingpong_flag * lBrcb_size],
                          scaleTensor[i * scaleRowLenth], 64, headProceesNum,
                          {1, 0, 8, 1});
            PIPE_BARRIER(V);

            WAIT_FLAG(V, MTE2, pingpong_flag);
            gm_to_ub<ArchType::ASCEND_V220, float>(
                loTensor[pingpong_flag * dataCount], gmLoTensor[i * __k0],
                0,                        // sid
                headProceesNum,           // nBurst
                __k0 / FLOAT_BLOCK_SIZE,  // lenBurst
                __k0 / FLOAT_BLOCK_SIZE *
                    (cur_kv_split_core_num - 1),  // srcGap
                0                                 // dstGap
            );
            SET_FLAG(MTE2, V, pingpong_flag);
            WAIT_FLAG(MTE2, V, pingpong_flag);
            mul_v<ArchType::ASCEND_V220, float>(
                toTensor[pingpong_flag * dataCount],
                loTensor[pingpong_flag * dataCount],
                lBrcb[pingpong_flag * lBrcb_size],
                repeat,  // repeat
                1,       // dstBlockStride
                1,       // src0BlockStride
                0,       // src1BlockStride
                8,       // dstRepeatStride
                8,       // src0RepeatStride
                1        // src1RepeatStride
            );
            PIPE_BARRIER(V);
            add_v<ArchType::ASCEND_V220, float>(
                goTensor, toTensor[pingpong_flag * dataCount], goTensor,
                repeat,  // repeat
                1,       // dstBlockStride
                1,       // src0BlockStride
                1,       // src1BlockStride
                8,       // dstRepeatStride
                8,       // src0RepeatStride
                8        // src1RepeatStride
            );
            PIPE_BARRIER(V);
            SET_FLAG(V, MTE2, pingpong_flag);
            pingpong_flag = 1 - pingpong_flag;
        }
        conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(go16tensor, goTensor,
                                                        repeat,  // repeat
                                                        1,  // dstBlockStride
                                                        1,  // srcBlockStride
                                                        4,  // dstRepeatStride
                                                        8   // srcRepeatStride
        );
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE3, EVENT_ID2);
        WAIT_FLAG(V, MTE3, EVENT_ID2);
        ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
            gmOutTensor, go16tensor,
            0,              // sid
            1,              // nBurst
            dataCount * 2,  // lenBurst
            0,              // leftPaddingNum
            0,              // rightPaddingNum
            0,              // srcGap
            0               // dstGap
        );
    }

    __aicore__ __attribute__((always_inline)) inline void CopyScaleBlock(
        uint32_t sub_m, uint32_t head_loop_idx, uint32_t l_offset,
        uint32_t o_offset, uint32_t split_num, uint32_t oUbOffset) {
        SET_FLAG(V, MTE3, EVENT_ID2);
        WAIT_FLAG(V, MTE3, EVENT_ID2);
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            l_gm_tensor[(int64_t)l_offset], tv32_ubuf_tensor,
            0,                    // sid
            sub_m,                // nBurst
            32,                   // lenBurst
            0,                    // leftPaddingNum
            0,                    // rightPaddingNum
            0,                    // srcGap
            (split_num - 1) * 32  // dstGap
        );
        if (gl_flag_scalar == 0) {
            SET_FLAG(MTE3, V, EVENT_ID2);
            gl_flag_scalar = 1;
        }
        uint32_t src_gap = ((__k % 16 <= 8) && (__k % 16 > 0)) ? 1 : 0;
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            o_core_tmp_gm_tensor[(int64_t)o_offset],
            go32_ubuf_tensor[oUbOffset],
            0,                         // sid
            sub_m,                     // nBurst
            __k * 4,                   // lenBurst
            0,                         // leftPaddingNum
            0,                         // rightPaddingNum
            src_gap,                   // srcGap
            (split_num - 1) * __k * 4  // dstGap
        );

        WAIT_FLAG(MTE3, V, EVENT_ID3);
        SET_FLAG(MTE3, V, EVENT_ID3);

        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        SET_FLAG(MTE3, MTE2, EVENT_ID1);
    }

    __aicore__ __attribute__((always_inline)) inline void CombineScaleBlock(
        uint32_t num_tokens, uint32_t q_heads, uint32_t kv_split_core_num,
        uint32_t embedding_size) {
        SetAtomicnone();
        SetMasknorm();
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        __gm__ const NdMLADecodingMtpTilingTP1 *__restrict ndMtpTiling =
            &(tiling_gm->ndMLADecodingMtpTilingTP1);
        uint32_t batchTaskNum = (uint32_t)(ndMtpTiling->tailBatch);
        uint32_t q_seqlen = (uint32_t)(ndMtpTiling->qSeqLen[totalTaskNum]);
        uint32_t decodingQSeqLens = batchTaskNum * q_seqlen;
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            q_seqlen = 1;
        }
        uint32_t vectorBlockIdx = block_idx * 2 + sub_block_idx;
        uint32_t coreNumPreBatch = (block_num * 2) / decodingQSeqLens;
        if (vectorBlockIdx >= coreNumPreBatch * decodingQSeqLens) {
            return;
        }
        uint32_t coreIdInBatch = vectorBlockIdx % coreNumPreBatch;
        uint32_t curDecodingBatch = vectorBlockIdx / coreNumPreBatch;
        uint32_t processNumOneBatch = q_heads;
        uint32_t headNumPreCore = processNumOneBatch / coreNumPreBatch;
        uint32_t headTails = processNumOneBatch % coreNumPreBatch;
        uint32_t headOffset = headNumPreCore * coreIdInBatch;
        if (coreIdInBatch < headTails) {
            headNumPreCore += 1;
            headOffset += coreIdInBatch;
        } else {
            headOffset += headTails;
        }
        uint32_t offset_tiling =
            tiling_head_size +
            tiling_para_size * (totalTaskNum + flashDecodingTaskNum);

        uint32_t batchTilingOffset;
        uint32_t qOffset;
        if (curDecodingBatch == 0) {
            batchTilingOffset =
                (uint32_t)(ndMtpTiling
                               ->batchIdx[totalTaskNum + flashDecodingTaskNum]);
            qOffset =
                (uint32_t)(ndMtpTiling
                               ->qSeqLen[totalTaskNum + flashDecodingTaskNum]);
        } else if (curDecodingBatch == 1) {
            batchTilingOffset =
                (uint32_t)(ndMtpTiling->kvBlockIdx[totalTaskNum +
                                                   flashDecodingTaskNum]);
            qOffset =
                (uint32_t)(ndMtpTiling->prevKVSeqlen[totalTaskNum +
                                                     flashDecodingTaskNum]);
        } else if (curDecodingBatch == 2) {
            batchTilingOffset =
                (uint32_t)(ndMtpTiling->batchSize[totalTaskNum +
                                                  flashDecodingTaskNum]);
            qOffset =
                (uint32_t)(ndMtpTiling
                               ->splitNum[totalTaskNum + flashDecodingTaskNum]);
        }

        CombineScaleBlockInBatch(batchTilingOffset, headOffset, headNumPreCore,
                                 q_seqlen, qOffset);
    }

    __aicore__ __attribute__((always_inline)) inline void
    CombineScaleBlockInBatch(uint32_t batchTilingOffset, uint32_t startHead,
                             uint32_t headNum, uint32_t q_seqlen,
                             uint32_t qOffset) {
        __gm__ const NdMLADecodingMtpTilingTP1 *__restrict ndMtpTiling =
            &(tiling_gm->ndMLADecodingMtpTilingTP1);
        uint32_t prev_task =
            (uint32_t)(ndMtpTiling->batchSize[batchTilingOffset]);
        uint32_t cur_kv_split_core_num =
            (uint32_t)(ndMtpTiling->splitNum[batchTilingOffset]);
        uint32_t prev_split_core_num =
            (uint32_t)(ndMtpTiling->prevSplitNumSum[batchTilingOffset]);

        uint64_t addr_l_offset =
            prev_split_core_num * q_heads * FLOAT_BLOCK_SIZE * q_seqlen +
            qOffset * q_heads * FLOAT_BLOCK_SIZE * cur_kv_split_core_num;
        uint64_t addr_o_fd_offset =
            prev_split_core_num * q_heads * embedding_size * q_seqlen +
            qOffset * q_heads * embedding_size * cur_kv_split_core_num;

        uint64_t addr_o_offset =
            (prev_task + qOffset) * q_heads * embedding_size;

        constexpr uint32_t HEADS_PROCESS = 8;
        uint32_t loops = (headNum + HEADS_PROCESS - 1) / HEADS_PROCESS;
        uint32_t tail = headNum % HEADS_PROCESS;
        if (tail == 0) {
            tail = HEADS_PROCESS;
        }
        uint32_t cur_head_num = HEADS_PROCESS;

        SET_FLAG(V, MTE2, EVENT_ID0);
        SET_FLAG(V, MTE2, EVENT_ID1);
        SET_FLAG(MTE3, MTE2, EVENT_ID2);
        for (int i = 0; i < loops; i++) {
            cur_head_num = (i == loops - 1) ? tail : HEADS_PROCESS;
            WAIT_FLAG(MTE3, MTE2, EVENT_ID2);
            gm_to_ub<ArchType::ASCEND_V220, float>(
                lTmptensor,
                l_gm_tensor[addr_l_offset + startHead * FLOAT_BLOCK_SIZE *
                                                cur_kv_split_core_num],
                0,                                     // sid
                1,                                     // nBurst
                cur_head_num * cur_kv_split_core_num,  // lenBurst
                0,                                     // srcGap
                0                                      // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID2);
            WAIT_FLAG(MTE2, V, EVENT_ID2);
            CalcScale(lTensor, lTmptensor, lMaxTensor, cur_head_num,
                      cur_kv_split_core_num);
            uint32_t rowLength =
                ((cur_head_num + 8 - 1) / 8 * 8) * FLOAT_BLOCK_SIZE;
            CombineOutByMultiHead(
                o_core_tmp_gm_tensor[addr_o_fd_offset +
                                     startHead * cur_kv_split_core_num *
                                         embedding_size],
                o_gm_tensor[addr_o_offset + startHead * embedding_size],
                lTensor, cur_head_num, rowLength, cur_kv_split_core_num, i);
            SET_FLAG(MTE3, MTE2, EVENT_ID2);
            startHead += cur_head_num;
        }
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID2);
    }

    __aicore__ __attribute__((always_inline)) inline void SoftmaxStage1(
        AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor,
        AscendC::GlobalTensor<mm1CopyType> s_gm_tensor,
        AscendC::GlobalTensor<float> s_rope_gm_tensor,
        AscendC::GlobalTensor<OUT_DTYPE> mask_gm_tensor,
        AscendC::LocalTensor<float> dm32_ubuf_tensor,
        AscendC::LocalTensor<float> ll_ubuf_tensor,
        AscendC::LocalTensor<float> pm32_ubuf_tensor, uint32_t n_idx,
        uint32_t qk_n, uint32_t qk_round_n, uint32_t sub_m,
        uint32_t mask_offset, const uint32_t sub_n_loop,
        const uint32_t cur_batch, const uint32_t start_kv,
        const uint32_t real_n_loop, const uint32_t head_idx,
        const uint32_t pm_flag_scalar, uint32_t cur_q_seqlen,
        uint32_t cur_kv_seqlen, bool need_mask) {
        uint32_t sub_m_d128 = (sub_m + 127) / 128;  // up aligned to 128
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
        float quantMax = (float)1 / (float)127;
        WAIT_FLAG(V, MTE2, EVENT_ID2);
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            DeQuantPerHeadImpl(
                deq_scale_gm_tensor_q1[head_idx],  // 这里待修改
                s_gm_tensor, ls32_quant_ubuf_tensor,
                ls32_quant_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                descale_q1_ubuf_tensor, tv32_ubuf_tensor, pm32_ubuf_tensor,
                sub_m, qk_n, qk_round_n, 0, 1);
            gm_to_ub<ArchType::ASCEND_V220, float>(
                ls32_ubuf_tensor.template ReinterpretCast<float>(),
                s_rope_gm_tensor,
                0,  // sid
                1,  // nBurst
                sub_m * qk_round_n / FLOAT_BLOCK_SIZE,
                0,  // srcGap
                0   // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            AscendC::Add(ls32_ubuf_tensor, ls32_ubuf_tensor,
                         ls32_quant_ubuf_tensor, sub_m * qk_round_n);  // float
            PIPE_BARRIER(V);
        } else {
            gm_to_ub<ArchType::ASCEND_V220, mm1CopyType>(
                ls32_ubuf_tensor.template ReinterpretCast<mm1CopyType>(),
                s_gm_tensor,
                0,                                      // sid
                1,                                      // nBurst
                sub_m * qk_round_n / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                                      // srcGap
                0                                       // dstGap
            );

            if (mask_type == 3) {
                uint32_t aligned_mask_copy_len =
                    RoundUp<BLOCK_SIZE>(qk_n);  // 16
                uint32_t mask_dst_stride =
                    (qk_round_n - aligned_mask_copy_len) / BLOCK_SIZE;  // 0

                AscendC::DataCopyPad(
                    mask_ubuf_tensor, mask_gm_tensor,
                    AscendC::DataCopyExtParams(cur_q_seqlen, qk_n * 2,
                                               maxKVSeqLen * 2 - qk_n * 2,
                                               mask_dst_stride, 0),
                    AscendC::DataCopyPadExtParams<OUT_DTYPE>(false, 0, 0, 0));
            } else if (need_mask && mask_type == 4) {
                AscendC::DataCopy(
                    mask_ubuf_tensor, mask_gm_tensor,
                    AscendC::DataCopyParams(
                        cur_q_seqlen,         // blockCount
                        qk_round_n * 2 / 32,  // blockLen, 2 is sizeof(half)
                        MASK_COLUMNS * 2 / 32 -
                            qk_round_n * 2 / 32,  // srcStride
                        0                         // dstStride
                        ));
            }

            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);

            if (mask_type == 3 || (need_mask && mask_type == 4)) {
                AscendC::Cast(mask32_ubuf_tensor, mask_ubuf_tensor,
                              AscendC::RoundMode::CAST_NONE,
                              cur_q_seqlen * qk_round_n);
            }
        }
        AscendC::LocalTensor<float> res32_ubuf_tensor = ls32_ubuf_tensor;
        if constexpr (tilingKeyType != TilingKeyType::TILING_INT8_DATA) {
            res32_ubuf_tensor =
                cur_q_seqlen > 1 ? tv32_ubuf_tensor : ls32_ubuf_tensor;
        }
        for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE;
             ++vadd_idx) {
            muls_v<ArchType::ASCEND_V220, float>(
                res32_ubuf_tensor[vadd_idx * FLOAT_VECTOR_SIZE],
                ls32_ubuf_tensor[vadd_idx * FLOAT_VECTOR_SIZE], tor,
                sub_m,                          // repeat
                1,                              // dstBlockStride
                1,                              // srcBlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE   // srcRepeatStride
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            muls_v<ArchType::ASCEND_V220, float>(
                res32_ubuf_tensor[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                ls32_ubuf_tensor[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tor,
                sub_m,                          // repeat
                1,                              // dstBlockStride
                1,                              // srcBlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE   // srcRepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
        if constexpr (tilingKeyType != TilingKeyType::TILING_INT8_DATA) {
            uint32_t cur_compute_head_num = sub_m / cur_q_seqlen;
            if (mask_type == 3 || (need_mask && mask_type == 4)) {
                for (uint32_t i = 0; i < cur_compute_head_num; i++) {
                    Add(res32_ubuf_tensor[cur_q_seqlen * qk_round_n * i],
                        res32_ubuf_tensor[cur_q_seqlen * qk_round_n * i],
                        mask32_ubuf_tensor, cur_q_seqlen * qk_round_n);
                }
                PIPE_BARRIER(V);
            }
            if (cur_q_seqlen > 1) {
                for (uint32_t i = 0; i < cur_q_seqlen; i++) {
                    uint16_t lenBurst = qk_round_n / FLOAT_BLOCK_SIZE;
                    uint16_t srcGap =
                        qk_round_n * (cur_q_seqlen - 1) / FLOAT_BLOCK_SIZE;
                    ub_to_ub<ArchType::ASCEND_V220, float>(
                        ls32_ubuf_tensor[i * cur_compute_head_num * qk_round_n],
                        res32_ubuf_tensor[i * qk_round_n],
                        0,                     // sid
                        cur_compute_head_num,  // nBurst
                        lenBurst,              // lenBurst
                        srcGap,                // srcGap
                        0                      // dstGap
                    );
                }
                PIPE_BARRIER(V);
            }
        }
        // *** lm = rowmax(ls)
        ReduceMaxRepeatM(lm32_ubuf_tensor, ls32_ubuf_tensor, lp32_ubuf_tensor,
                         sub_m, qk_n, qk_round_n);
        // ReduceMaxChange(lm32_ubuf_tensor, ls32_ubuf_tensor, tv32_ubuf_tensor,
        // round_sub_m, qk_n, qk_round_n);
        if (n_idx != 0) {
            // *** hm = vmax(lm, gm)
            max_v<ArchType::ASCEND_V220, float>(
                hm32_ubuf_tensor, lm32_ubuf_tensor, gm32_ubuf_tensor,
                sub_m_d64,  // repeat
                1,          // dstBlockStride
                1,          // src0BlockStride
                1,          // src1BlockStride
                8,          // dstRepeatStride
                8,          // src0RepeatStride
                8           // src1RepeatStride
            );
            PIPE_BARRIER(V);
            // *** dm = gm - hm
            sub_v<ArchType::ASCEND_V220, float>(
                dm32_ubuf_tensor, gm32_ubuf_tensor, hm32_ubuf_tensor,
                sub_m_d64,  // repeat
                1,          // dstBlockStride
                1,          // src0BlockStride
                1,          // src1BlockStride
                8,          // dstRepeatStride
                8,          // src0RepeatStride
                8           // src1RepeatStride
            );
            PIPE_BARRIER(V);
        } else {
            // *** hm = lm
            ub_to_ub<ArchType::ASCEND_V220, float>(
                hm32_ubuf_tensor, lm32_ubuf_tensor,
                0,                               // sid
                1,                               // nBurst
                round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                               // srcGap
                0                                // dstGap
            );
            PIPE_BARRIER(V);
        }
        // *** gm = hm
        ub_to_ub<ArchType::ASCEND_V220, float>(
            gm32_ubuf_tensor, hm32_ubuf_tensor,
            0,                               // sid
            1,                               // nBurst
            round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                               // srcGap
            0                                // dstGap
        );
        PIPE_BARRIER(V);
        // *** hm_block = expand_to_block(hm), 存放于 tv

        // *** ls = ls - hm_block
        TensorSubValueRepeatM(ls32_ubuf_tensor, ls32_ubuf_tensor,
                              hm32_ubuf_tensor, tv32_ubuf_tensor, sub_m,
                              round_sub_m, qk_n, qk_round_n);
        // *** ls = exp(ls)
        exp_v<ArchType::ASCEND_V220, float>(
            ls32_ubuf_tensor, ls32_ubuf_tensor,
            (sub_m * qk_round_n + FLOAT_VECTOR_SIZE - 1) /
                FLOAT_VECTOR_SIZE,  // repeat
            1,                      // dstBlockStride
            1,                      // srcBlockStride
            8,                      // dstRepeatStride
            8                       // srcRepeatStride
        );
        PIPE_BARRIER(V);
        // *** lp = castfp32to16(ls)
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            QuantScaleCal(pm32_ubuf_tensor, lm32_ubuf_tensor, hm32_ubuf_tensor,
                          tv32_ubuf_tensor, round_sub_m, round_sub_m);
            QuantPerTokenImpl(lp_ubuf_tensor, ls32_ubuf_tensor,
                              tv32_ubuf_tensor, sub_m, qk_n, qk_round_n, 1);
        } else {
            conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(
                lp_ubuf_tensor, ls32_ubuf_tensor,
                (sub_m * qk_round_n + FLOAT_VECTOR_SIZE - 1) /
                    FLOAT_VECTOR_SIZE,  // repeat
                1,                      // dstBlockStride
                1,                      // srcBlockStride
                4,                      // dstRepeatStride
                8                       // srcRepeatStride
            );
            PIPE_BARRIER(V);
        }
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        ub_to_gm<ArchType::ASCEND_V220, IN_DTYPE>(
            p_gm_tensor, lp_ubuf_tensor,
            0,                                                   // sid
            1,                                                   // nBurst
            sub_m * qk_round_n * T_BLOCK_OFFSET / T_BLOCK_SIZE,  // lenBurst
            0,                                                   // srcGap
            0                                                    // dstGap
        );

        // *** ll = rowsum(ls32)
        ReduceSumRepeatM(ll_ubuf_tensor, ls32_ubuf_tensor, sub_m, qk_n,
                         qk_round_n);
        SET_FLAG(V, MTE2, EVENT_ID2);
        PIPE_BARRIER(V);
    }
    template <typename P_DTYPE, typename MASK_DTYPE>
    __aicore__ __attribute__((always_inline)) inline void
    OnlineSoftmaxStage1New(const AscendC::LocalTensor<float> &s_ub,
                           const AscendC::LocalTensor<MASK_DTYPE> &mask_orig_ub,
                           const AscendC::LocalTensor<float> &mask_processed_ub,
                           const AscendC::LocalTensor<float> &local_rowmax_ub,
                           const AscendC::LocalTensor<float> &hat_rowmax_ub,
                           const AscendC::LocalTensor<float> &global_rowmax_ub,
                           const AscendC::LocalTensor<float> &diff_rowmax_ub,
                           const AscendC::LocalTensor<float> &s_exp_ub,
                           const AscendC::LocalTensor<float> &local_rowsum_ub,
                           const AscendC::LocalTensor<float> &global_rowsum_ub,
                           const AscendC::LocalTensor<P_DTYPE> &p_ub,
                           const AscendC::LocalTensor<float> &tmp_ub,
                           const AscendC::GlobalTensor<float> &s_gm,
                           const AscendC::GlobalTensor<P_DTYPE> &p_gm,
                           bool first_n_iter, float tor, uint32_t m,
                           uint32_t n_real, uint32_t n_stride,
                           uint32_t gm_stride, uint32_t pingpong_flag) {
        uint32_t round_m =
            (m + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
        WAIT_FLAG(MTE3, MTE2, pingpong_flag);
        // input QK
        uint32_t block_cnt = CeilDiv<128>(n_stride);
        for (int idx = 0; idx < block_cnt; idx++) {
            uint32_t ub_offset = 128 * idx;
            uint32_t block_offset = q_heads * 128 * idx;
            uint32_t n_stride_real =
                (idx == block_cnt - 1) ? n_stride - 128 * idx : 128;
            uint32_t src_gap = 128 - n_stride_real;
            uint32_t dst_gap = n_stride - n_stride_real;
            gm_to_ub<ArchType::ASCEND_V220, float>(
                s_ub[ub_offset], s_gm[block_offset],
                0,                                         // sid
                m,                                         // nBurst
                CeilDiv<FLOAT_BLOCK_SIZE>(n_stride_real),  // lenBurst
                CeilDiv<FLOAT_BLOCK_SIZE>(src_gap),        // srcGap
                CeilDiv<FLOAT_BLOCK_SIZE>(dst_gap)         // dstGap
            );
        }
        SET_FLAG(MTE2, V, pingpong_flag);

        WAIT_FLAG(MTE2, V, pingpong_flag);
        OnlineSoftmaxStage1Step1(
            s_ub.template ReinterpretCast<float>(), local_rowmax_ub,
            hat_rowmax_ub, global_rowmax_ub, diff_rowmax_ub, s_exp_ub,
            local_rowsum_ub, tmp_ub, first_n_iter, tor, m, n_real, n_stride);

        conv_v<ArchType::ASCEND_V220, float, P_DTYPE>(
            p_ub, s_exp_ub,
            (m * n_stride + FLOAT_VECTOR_SIZE - 1) /
                FLOAT_VECTOR_SIZE,  // repeat
            1,                      // dstBlockStride
            1,                      // srcBlockStride
            4,                      // dstRepeatStride
            8                       // srcRepeatStride
        );
        SET_FLAG(V, MTE3, pingpong_flag);

        WAIT_FLAG(V, MTE3, pingpong_flag);
        for (int idx = 0; idx < block_cnt; idx++) {
            uint32_t ub_offset = 128 * idx;
            uint32_t block_offset = q_heads * 128 * idx;
            uint32_t n_stride_real =
                (idx == block_cnt - 1) ? n_stride - 128 * idx : 128;
            uint32_t dst_gap = 128 - n_stride_real;
            uint32_t src_gap = n_stride - n_stride_real;
            ub_to_gm<ArchType::ASCEND_V220, P_DTYPE>(
                p_gm[block_offset], p_ub[ub_offset],
                0,                                   // sid
                m,                                   // nBurst
                CeilDiv<BLOCK_SIZE>(n_stride_real),  // lenBurst
                CeilDiv<BLOCK_SIZE>(src_gap),        // srcGap
                CeilDiv<BLOCK_SIZE>(dst_gap)         // dstGap
            );
        }
        SET_FLAG(MTE3, MTE2, pingpong_flag);

        OnlineSoftmaxStage1Step2(local_rowsum_ub, global_rowsum_ub,
                                 diff_rowmax_ub, first_n_iter, m);
    }
    __aicore__ __attribute__((always_inline)) inline void
    OnlineSoftmaxStage1Quant(
        const AscendC::LocalTensor<int32_t> &s_ub,
        const AscendC::LocalTensor<float> &s_rope_ub,
        const AscendC::LocalTensor<float> &mask_processed_ub,
        const AscendC::LocalTensor<float> &local_rowmax_ub,
        const AscendC::LocalTensor<float> &hat_rowmax_ub,
        const AscendC::LocalTensor<float> &global_rowmax_ub,
        const AscendC::LocalTensor<float> &diff_rowmax_ub,
        const AscendC::LocalTensor<float> &s_exp_ub,
        const AscendC::LocalTensor<float> &local_rowsum_ub,
        const AscendC::LocalTensor<float> &global_rowsum_ub,
        const AscendC::LocalTensor<float> &dequant_scale_ub,
        const AscendC::LocalTensor<float> &quant_online_scale_ub,
        const AscendC::LocalTensor<int8_t> &p_ub,
        const AscendC::LocalTensor<float> &tmp_ub,
        const AscendC::GlobalTensor<int32_t> &s_gm,
        const AscendC::GlobalTensor<float> &s_rope_gm,
        const AscendC::GlobalTensor<int8_t> &p_gm, bool first_n_iter, float tor,
        uint32_t m, uint32_t n_real, uint32_t n_stride, uint32_t gm_stride,
        uint32_t pingpong_flag) {
        uint32_t round_m =
            (m + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
        float quantMax = (float)1 / (float)127;

        WAIT_FLAG(MTE3, MTE2, pingpong_flag);
        // input QK
        gm_to_ub<ArchType::ASCEND_V220, int32_t>(
            s_ub, s_gm,
            0,                                                // sid
            m,                                                // nBurst
            CeilDiv<FLOAT_BLOCK_SIZE>(n_stride),              // lenBurst
            CeilDiv<FLOAT_BLOCK_SIZE>(gm_stride - n_stride),  // srcGap
            0                                                 // dstGap
        );
        SET_FLAG(MTE2, V, pingpong_flag);

        WAIT_FLAG(MTE3, MTE2, pingpong_flag + 2);
        // input QK rope
        gm_to_ub<ArchType::ASCEND_V220, float>(
            s_rope_ub, s_rope_gm,
            0,                                                // sid
            m,                                                // nBurst
            CeilDiv<FLOAT_BLOCK_SIZE>(n_stride),              // lenBurst
            CeilDiv<FLOAT_BLOCK_SIZE>(gm_stride - n_stride),  // srcGap
            0                                                 // dstGap
        );
        SET_FLAG(MTE2, V, pingpong_flag + 2);

        WAIT_FLAG(MTE2, V, pingpong_flag);
        // dequant
        DeQuantPerHeadProc(s_ub.template ReinterpretCast<float>(), s_ub,
                           dequant_scale_ub, tmp_ub, quant_online_scale_ub, m,
                           n_real, n_stride, 0);
        WAIT_FLAG(MTE2, V, pingpong_flag + 2);
        AscendC::Add(s_ub.template ReinterpretCast<float>(),
                     s_ub.template ReinterpretCast<float>(), s_rope_ub,
                     m * n_stride);  // float
        PIPE_BARRIER(V);

        OnlineSoftmaxStage1Step1(
            s_ub.template ReinterpretCast<float>(), local_rowmax_ub,
            hat_rowmax_ub, global_rowmax_ub, diff_rowmax_ub, s_exp_ub,
            local_rowsum_ub, tmp_ub, first_n_iter, tor, m, n_real, n_stride);

        QuantScaleCal(quant_online_scale_ub, local_rowmax_ub, hat_rowmax_ub,
                      tmp_ub, m, round_m);
        QuantPerTokenImpl(p_ub, s_exp_ub, tmp_ub, m, n_real, n_stride, 1);
        SET_FLAG(V, MTE3, pingpong_flag);

        WAIT_FLAG(V, MTE3, pingpong_flag);
        ub_to_gm<ArchType::ASCEND_V220, int8_t>(
            p_gm, p_ub,
            0,                                            // sid
            m,                                            // nBurst
            CeilDiv<BLOCK_SIZE_32>(n_stride),             // lenBurst
            CeilDiv<BLOCK_SIZE_32>(n_stride),             // srcGap
            CeilDiv<BLOCK_SIZE_32>(gm_stride - n_stride)  // dstGap
        );
        // 由于p_ub，s_exp_ub可能复用了s_ub，此处再释放
        SET_FLAG(MTE3, MTE2, pingpong_flag);
        SET_FLAG(MTE3, MTE2, pingpong_flag + 2);

        OnlineSoftmaxStage1Step2(local_rowsum_ub, global_rowsum_ub,
                                 diff_rowmax_ub, first_n_iter, m);
    }

    __aicore__ __attribute__((always_inline)) inline void
    SoftmaxStage2MLAHeadLoop(AscendC::GlobalTensor<mm2CopyType> o_tmp_gm_tensor,
                             AscendC::GlobalTensor<float> go_gm_tensor,
                             AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor,
                             AscendC::LocalTensor<float> dm32_ubuf_tensor,
                             AscendC::LocalTensor<float> ll_ubuf_tensor,
                             AscendC::LocalTensor<float> pm32_ubuf_tensor,
                             uint32_t n_idx, uint32_t n_loop, uint32_t qk_n,
                             uint32_t qk_round_n, uint32_t sub_m,
                             uint64_t o_offset, uint32_t head_idx,
                             uint32_t pm_flag_scalar, uint32_t head_loop,
                             uint32_t head_loop_idx, uint32_t q_seq_len,
                             uint32_t sub_head_num, uint32_t cur_head_num,
                             uint32_t head_used_col_num,
                             uint32_t one_process_copy_qseqlen_num,
                             uint32_t tail_used_col_num) {
        uint32_t sub_m_d64 = (sub_m + 63) / 64;  // up aligned to 64
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        if (n_idx != 0) {
            gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
                lo_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                o_tmp_gm_tensor,
                0,                                   // sid
                1,                                   // nBurst
                sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                                   // srcGap
                0                                    // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
                DeQuantPerHeadImpl(
                    deq_scale_gm_tensor_k1[head_idx], o_tmp_gm_tensor,
                    lo_ubuf_tensor,
                    lo_ubuf_tensor.template ReinterpretCast<
                        mm2CopyType>(),  // lo_ubuf_tensor use the same ptr
                    descale_k1_ubuf_tensor, tv32_ubuf_tensor, pm32_ubuf_tensor,
                    sub_m, round_v, round_v, 1, 0);
            }
        }
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
        if (n_idx != 0) {
            // *** dm = exp(dm)
            if (head_loop_idx == 0) {
                exp_v<ArchType::ASCEND_V220, float>(dm32_ubuf_tensor,
                                                    dm32_ubuf_tensor,
                                                    sub_m_d64,  // repeat
                                                    1,  // dstBlockStride
                                                    1,  // srcBlockStride
                                                    8,  // dstRepeatStride
                                                    8   // srcRepeatStride
                );
                PIPE_BARRIER(V);
                // *** gl = dm * gl
                mul_v<ArchType::ASCEND_V220, float>(
                    gl32_ubuf_tensor, dm32_ubuf_tensor, gl32_ubuf_tensor,
                    sub_m_d64,  // repeat
                    1,          // dstBlockStride
                    1,          // src0BlockStride
                    1,          // src1BlockStride
                    8,          // dstRepeatStride
                    8,          // src0RepeatStride
                    8           // src1RepeatStride
                );
                PIPE_BARRIER(V);
                // *** gl = ll + gl
                add_v<ArchType::ASCEND_V220, float>(
                    gl32_ubuf_tensor, gl32_ubuf_tensor, ll_ubuf_tensor,
                    sub_m_d64,  // repeat
                    1,          // dstBlockStride
                    1,          // src0BlockStride
                    1,          // src1BlockStride
                    8,          // dstRepeatStride
                    8,          // src0RepeatStride
                    8           // src1RepeatStride
                );
                PIPE_BARRIER(V);
            }
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            brcb_v<ArchType::ASCEND_V220, uint32_t>(
                tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                dm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                1,                              // dstBlockStride
                8,                              // dstRepeatStride
                round_sub_m / FLOAT_BLOCK_SIZE  // repeat
            );
            PIPE_BARRIER(V);
            if (head_loop > 1) {
                gm_to_ub<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor, go_gm_tensor, 0, 1,
                    sub_m * round_v / FLOAT_BLOCK_SIZE, 0, 0);
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
            }

            // *** go = go * dm_block
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE;
                 ++vmul_idx) {
                mul_v<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                       // repeat
                    1,                           // dstBlockStride
                    1,                           // src0BlockStride
                    0,                           // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                            // src1RepeatStride
                );
            }
            if (__v % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(__v % FLOAT_VECTOR_SIZE);
                mul_v<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE *
                                     FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE *
                                     FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                       // repeat
                    1,                           // dstBlockStride
                    1,                           // src0BlockStride
                    0,                           // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                            // src1RepeatStride
                );
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            PIPE_BARRIER(V);
            // *** go = lo + go
            add_v<ArchType::ASCEND_V220, float>(
                go32_ubuf_tensor, go32_ubuf_tensor, lo_ubuf_tensor,
                (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) /
                    FLOAT_VECTOR_SIZE,  // repeat
                1,                      // dstBlockStride
                1,                      // src0BlockStride
                1,                      // src1BlockStride
                8,                      // dstRepeatStride
                8,                      // src0RepeatStride
                8                       // src1RepeatStride
            );
            PIPE_BARRIER(V);
        } else {
            // *** gl = ll
            if (head_loop_idx == 0) {
                ub_to_ub<ArchType::ASCEND_V220, float>(
                    gl32_ubuf_tensor, ll_ubuf_tensor,
                    0,                      // sid
                    1,                      // nBurst
                    64 / FLOAT_BLOCK_SIZE,  // lenBurst
                    // round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
                    0,  // srcGap
                    0   // dstGap
                );
                PIPE_BARRIER(V);
            }

            gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
                go32_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                o_tmp_gm_tensor,
                0,                                   // sid
                1,                                   // nBurst
                sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                                   // srcGap
                0                                    // dstGap
            );
            if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
                DeQuantPerHeadImpl(
                    deq_scale_gm_tensor_k1[head_idx], o_tmp_gm_tensor,
                    go32_ubuf_tensor,
                    go32_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                    descale_k1_ubuf_tensor, tv32_ubuf_tensor, pm32_ubuf_tensor,
                    sub_m, round_v, round_v, 1, 0);
            } else {
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
            }
        }
        SET_FLAG(V, MTE2, EVENT_ID0);

        if (n_idx == n_loop - 1) {
            // *** gl_block = expand_to_block(gl), 存放于 tv
            brcb_v<ArchType::ASCEND_V220, uint32_t>(
                tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                gl32_ubuf_tensor
                    .ReinterpretCast<uint32_t>()[head_loop_idx * 16],
                1,                              // dstBlockStride
                8,                              // dstRepeatStride
                round_sub_m / FLOAT_BLOCK_SIZE  // repeat
            );
            PIPE_BARRIER(V);
            // *** go = go / gl_block
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < __v / FLOAT_VECTOR_SIZE;
                 ++vdiv_idx) {
                div_v<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                       // repeat
                    1,                           // dstBlockStride
                    1,                           // src0BlockStride
                    0,                           // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                            // src1RepeatStride
                );
            }
            if (__v % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(__v % FLOAT_VECTOR_SIZE);
                div_v<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE *
                                     FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE *
                                     FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                       // repeat
                    1,                           // dstBlockStride
                    1,                           // src0BlockStride
                    0,                           // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                            // src1RepeatStride
                );
                SetVectorMask<int8_t>((uint64_t)-1,
                                      (uint64_t)-1);  // fix hidden_size=96
            }
            PIPE_BARRIER(V);

            // *** go = castfp32to16(go)
            conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(
                go_ubuf_tensor, go32_ubuf_tensor,
                (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) /
                    FLOAT_VECTOR_SIZE,  // repeat
                1,                      // dstBlockStride
                1,                      // srcBlockStride
                4,                      // dstRepeatStride
                8                       // srcRepeatStride
            );
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);

            uint32_t outBytes = sizeof(OUT_DTYPE);
            if (q_seq_len > 1) {
                uint32_t inner_o_gm_offset = 0;
                uint32_t inner_go_ubuf_offset = 0;
                if (head_used_col_num != 0) {
                    ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                        o_gm_tensor[inner_o_gm_offset],
                        go_ubuf_tensor[inner_go_ubuf_offset],
                        0,                                   // sid
                        1,                                   // nBurst
                        head_used_col_num * __v * outBytes,  // lenBurst
                        0,                                   // leftPaddingNum
                        0,                                   // rightPaddingNum
                        0,                                   // srcGap
                        0                                    // dstGap
                    );
                    inner_o_gm_offset +=
                        (q_heads + head_used_col_num - sub_head_num) * __v;
                    inner_go_ubuf_offset += head_used_col_num * __v;
                }
                for (uint32_t i = 0; i < one_process_copy_qseqlen_num; ++i) {
                    ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                        o_gm_tensor[inner_o_gm_offset],
                        go_ubuf_tensor[inner_go_ubuf_offset],
                        0,                              // sid
                        1,                              // nBurst
                        sub_head_num * __v * outBytes,  // lenBurst
                        0,                              // leftPaddingNum
                        0,                              // rightPaddingNum
                        0,                              // srcGap
                        0                               // dstGap
                    );
                    inner_o_gm_offset += q_heads * __v;
                    inner_go_ubuf_offset += sub_head_num * __v;
                }
                if (tail_used_col_num != 0) {
                    ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                        o_gm_tensor[inner_o_gm_offset],
                        go_ubuf_tensor[inner_go_ubuf_offset],
                        0,                                   // sid
                        1,                                   // nBurst
                        tail_used_col_num * __v * outBytes,  // lenBurst
                        0,                                   // leftPaddingNum
                        0,                                   // rightPaddingNum
                        0,                                   // srcGap
                        0                                    // dstGap
                    );
                }
            } else {
                ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                    o_gm_tensor, go_ubuf_tensor,
                    0,                       // sid
                    1,                       // nBurst
                    sub_m * __v * outBytes,  // lenBurst
                    0,                       // leftPaddingNum
                    0,                       // rightPaddingNum
                    0,                       // srcGap
                    0                        // dstGap
                );
            }
            // ********************* move O to GM ************************
            if constexpr (IS_RING) {
                if (head_loop_idx == head_loop - 1) {
                    uint32_t cal_head_num = q_seq_len * sub_head_num;
                    uint32_t cal_head_num_d64 = (cal_head_num + 63) / 64;
                    ln_v<ArchType::ASCEND_V220, float>(
                        lse32_ubuf_tensor, gl32_ubuf_tensor,
                        cal_head_num_d64,  // repeat
                        1,                 // dstBlockStride
                        1,                 // srcBlockStride
                        8,                 // dstRepeatStride
                        8                  // srcRepeatStride
                    );
                    PIPE_BARRIER(V);
                    add_v<ArchType::ASCEND_V220, float>(
                        lse32_ubuf_tensor, lse32_ubuf_tensor, gm32_ubuf_tensor,
                        cal_head_num_d64,  // repeat
                        1,                 // dstBlockStride
                        1,                 // src0BlockStride
                        1,                 // src1BlockStride
                        8,                 // dstRepeatStride
                        8,                 // src0RepeatStride
                        8                  // src1RepeatStride
                    );
                    PIPE_BARRIER(V);
                    conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(
                        lse_conv_ubuf_tensor, lse32_ubuf_tensor,
                        cal_head_num_d64,  // repeat
                        1,                 // dstBlockStride
                        1,                 // srcBlockStride
                        4,                 // dstRepeatStride
                        8                  // srcRepeatStride
                    );
                    if (outBytes * sub_head_num < 32) {
                        PIPE_BARRIER(V);
                        uint32_t cal_head_num_up32 =
                            (cal_head_num + BLOCK_SIZE - 1) / BLOCK_SIZE *
                            BLOCK_SIZE;
                        brcb_v<ArchType::ASCEND_V220, OUT_DTYPE>(
                            lse_conv_ubuf_tensor[cal_head_num_up32],
                            lse_conv_ubuf_tensor,
                            1,                                // dstBlockStride
                            8,                                // dstRepeatStride
                            cal_head_num_up32 / BLOCK_SIZE_8  // repeat
                        );
                        SET_FLAG(V, MTE3, EVENT_ID1);
                        WAIT_FLAG(V, MTE3, EVENT_ID1);
                        int64_t lse_gm_offset = o_offset / __k;
                        uint32_t srcGap = sub_head_num - 1;
                        uint32_t dstGap = (q_heads - 1) * outBytes;
                        for (uint32_t i = 0; i < sub_head_num; ++i) {
                            ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                                lse_gm_tensor[(int64_t)(lse_gm_offset + i)],
                                lse_conv_ubuf_tensor[cal_head_num_up32 +
                                                     i * BLOCK_SIZE],
                                0,          // sid
                                q_seq_len,  // nBurst
                                outBytes,   // lenBurst
                                0,          // leftPaddingNum
                                0,          // rightPaddingNum
                                srcGap,     // srcGap
                                dstGap      // dstGap
                            );
                        }
                    } else {
                        SET_FLAG(V, MTE3, EVENT_ID1);
                        WAIT_FLAG(V, MTE3, EVENT_ID1);
                        uint32_t dstGap = (q_heads - sub_head_num) * outBytes;
                        ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                            lse_gm_tensor[(int64_t)(o_offset / __k)],
                            lse_conv_ubuf_tensor,
                            0,                        // sid
                            q_seq_len,                // nBurst
                            outBytes * sub_head_num,  // lenBurst
                            0,                        // leftPaddingNum
                            0,                        // rightPaddingNum
                            0,                        // srcGap
                            dstGap                    // dstGap
                        );
                    }
                    SET_FLAG(MTE3, V, EVENT_ID1);
                    WAIT_FLAG(MTE3, V, EVENT_ID1);
                }
            }

        } else if (head_loop > 1) {
            SET_FLAG(V, MTE3, EVENT_ID5);
            WAIT_FLAG(V, MTE3, EVENT_ID5);
            ub_to_gm<ArchType::ASCEND_V220, float>(
                go_gm_tensor, go32_ubuf_tensor, 0, 1,
                sub_m * round_v / FLOAT_BLOCK_SIZE, 0, 0);
        }
        SET_FLAG(MTE3, MTE2, EVENT_ID4);
    }

    template <bool flashDecodingVec = false>
    __aicore__ __attribute__((always_inline)) inline void
    SoftmaxStage2MLAHeadLoopTP1(
        AscendC::GlobalTensor<mm2CopyType> o_tmp_gm_tensor,
        AscendC::GlobalTensor<float> go_gm_tensor,
        AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor,
        AscendC::LocalTensor<float> dm32_ubuf_tensor,
        AscendC::LocalTensor<float> gl32_cur_ubuf_tensor,
        AscendC::LocalTensor<float> pm32_ubuf_tensor,
        AscendC::LocalTensor<float> descale_k1_ubuf_tensor, uint32_t n_idx,
        uint32_t n_loop, uint32_t sub_m, uint64_t o_offset, uint32_t head_idx,
        uint32_t head_loop, uint32_t head_loop_idx, uint32_t q_seq_len,
        int32_t q_i, uint32_t sub_head_num, uint32_t cur_head_idx,
        uint32_t cur_nIndx, uint32_t prev_split_num, uint32_t split_num) {
        uint32_t oPingPongFlag = head_loop_idx % 2;
        uint32_t sub_m_d64 = (sub_m + 63) / 64;  // up aligned to 64
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
        if constexpr (std::is_same_v<IN_DTYPE, int8_t>) {
            oPingPongFlag = 0;
        }
        uint32_t oUbOffset = oPingPongFlag * 4096;  // oPingPongFlag * 8 * 512
        WAIT_FLAG(V, MTE2, oPingPongFlag);
        if (n_idx != 4) {
            gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
                lo_ubuf_tensor
                    .template ReinterpretCast<mm2CopyType>()[oUbOffset],
                o_tmp_gm_tensor,
                0,                                   // sid
                1,                                   // nBurst
                sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                                   // srcGap
                0                                    // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            if constexpr (std::is_same_v<IN_DTYPE, int8_t>) {
                DeQuantPerHeadProc(
                    lo_ubuf_tensor[oUbOffset],
                    lo_ubuf_tensor.template ReinterpretCast<mm2CopyType>()
                        [oUbOffset],  // lo_ubuf_tensor use the same ptr
                    descale_k1_ubuf_tensor, tv32_ubuf_tensor, pm32_ubuf_tensor,
                    sub_m, round_v, round_v, 1);
            }
        }
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        WAIT_FLAG(MTE3, MTE2, oPingPongFlag + 4);
        if (n_idx != 4) {
            // *** dm = exp(dm)
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            brcb_v<ArchType::ASCEND_V220, uint32_t>(
                tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                dm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                1,                              // dstBlockStride
                8,                              // dstRepeatStride
                round_sub_m / FLOAT_BLOCK_SIZE  // repeat
            );
            PIPE_BARRIER(V);
            if (head_loop > 1) {
                gm_to_ub<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor[oUbOffset], go_gm_tensor, 0, 1,
                    sub_m * round_v / FLOAT_BLOCK_SIZE, 0, 0);
                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
            }

            // *** go = go * dm_block
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE;
                 ++vmul_idx) {
                mul_v<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor[oUbOffset + vmul_idx * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[oUbOffset + vmul_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                       // repeat
                    1,                           // dstBlockStride
                    1,                           // src0BlockStride
                    0,                           // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                            // src1RepeatStride
                );
            }
            if (__v % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(__v % FLOAT_VECTOR_SIZE);
                mul_v<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor[oUbOffset + __v / FLOAT_VECTOR_SIZE *
                                                     FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[oUbOffset + __v / FLOAT_VECTOR_SIZE *
                                                     FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                       // repeat
                    1,                           // dstBlockStride
                    1,                           // src0BlockStride
                    0,                           // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                            // src1RepeatStride
                );
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            PIPE_BARRIER(V);
            // *** go = lo + go
            add_v<ArchType::ASCEND_V220, float>(
                go32_ubuf_tensor[oUbOffset], go32_ubuf_tensor[oUbOffset],
                lo_ubuf_tensor[oUbOffset],
                (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) /
                    FLOAT_VECTOR_SIZE,  // repeat
                1,                      // dstBlockStride
                1,                      // src0BlockStride
                1,                      // src1BlockStride
                8,                      // dstRepeatStride
                8,                      // src0RepeatStride
                8                       // src1RepeatStride
            );
            PIPE_BARRIER(V);
        } else {
            // *** gl = ll
            gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
                go32_ubuf_tensor
                    .template ReinterpretCast<mm2CopyType>()[oUbOffset],
                o_tmp_gm_tensor,
                0,                                   // sid
                1,                                   // nBurst
                sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                                   // srcGap
                0                                    // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            if constexpr (std::is_same_v<IN_DTYPE, int8_t>) {
                DeQuantPerHeadProc(
                    go32_ubuf_tensor[oUbOffset],
                    go32_ubuf_tensor.template ReinterpretCast<mm2CopyType>()
                        [oUbOffset],  // lo_ubuf_tensor use the same ptr
                    descale_k1_ubuf_tensor, tv32_ubuf_tensor, pm32_ubuf_tensor,
                    sub_m, round_v, round_v, 1);
            }
        }
        SET_FLAG(V, MTE2, oPingPongFlag);

        if (n_idx + 4 > n_loop + 4 - 1) {
            // *** gl_block = expand_to_block(gl), 存放于 tv
            brcb_v<ArchType::ASCEND_V220, uint32_t>(
                tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                gl32_cur_ubuf_tensor.ReinterpretCast<uint32_t>(),
                1,                              // dstBlockStride
                8,                              // dstRepeatStride
                round_sub_m / FLOAT_BLOCK_SIZE  // repeat
            );
            PIPE_BARRIER(V);
            // *** go = go / gl_block
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < __v / FLOAT_VECTOR_SIZE;
                 ++vdiv_idx) {
                div_v<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor[oUbOffset + vdiv_idx * FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[oUbOffset + vdiv_idx * FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                       // repeat
                    1,                           // dstBlockStride
                    1,                           // src0BlockStride
                    0,                           // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                            // src1RepeatStride
                );
            }
            if (__v % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(__v % FLOAT_VECTOR_SIZE);
                div_v<ArchType::ASCEND_V220, float>(
                    go32_ubuf_tensor[oUbOffset + __v / FLOAT_VECTOR_SIZE *
                                                     FLOAT_VECTOR_SIZE],
                    go32_ubuf_tensor[oUbOffset + __v / FLOAT_VECTOR_SIZE *
                                                     FLOAT_VECTOR_SIZE],
                    tv32_ubuf_tensor,
                    sub_m,                       // repeat
                    1,                           // dstBlockStride
                    1,                           // src0BlockStride
                    0,                           // src1BlockStride
                    round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                    round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                    1                            // src1RepeatStride
                );
                SetVectorMask<int8_t>((uint64_t)-1,
                                      (uint64_t)-1);  // fix hidden_size=96
            }
            PIPE_BARRIER(V);

            if constexpr (flashDecodingVec) {
                ln_v<ArchType::ASCEND_V220, float>(tv32_ubuf_tensor,
                                                   tv32_ubuf_tensor,
                                                   sub_m,  // repeat
                                                   1,      // dstBlockStride
                                                   1,      // srcBlockStride
                                                   8,      // dstRepeatStride
                                                   8       // srcRepeatStride
                );
                PIPE_BARRIER(V);
                brcb_v<ArchType::ASCEND_V220, uint32_t>(
                    hm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                    gm32_ubuf_tensor.ReinterpretCast<uint32_t>()[cur_head_idx],
                    1,                              // dstBlockStride
                    8,                              // dstRepeatStride
                    round_sub_m / FLOAT_BLOCK_SIZE  // repeat
                );
                PIPE_BARRIER(V);
                // logf(lse_sum) + lse_max
                add_v<ArchType::ASCEND_V220, float>(
                    tv32_ubuf_tensor, tv32_ubuf_tensor, hm32_ubuf_tensor,
                    sub_m,  // repeat
                    1,      // dstBlockStride
                    1,      // src0BlockStride
                    1,      // src1BlockStride
                    8,      // dstRepeatStride
                    8,      // src0RepeatStride
                    8       // src1RepeatStride
                );
                PIPE_BARRIER(V);

                uint32_t o_fd_offset =
                    prev_split_num * q_heads * 512 * q_seq_len +
                    cur_nIndx * __k + head_idx * __k * split_num;
                uint32_t l_offset = prev_split_num * q_heads * q_seq_len +
                                    cur_nIndx + head_idx * split_num;
                CopyScaleBlock(sub_m, head_loop_idx, l_offset * 8, o_fd_offset,
                               split_num, oUbOffset);
            } else {
                // *** go = castfp32to16(go)
                conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(
                    go_ubuf_tensor[oUbOffset * 2], go32_ubuf_tensor[oUbOffset],
                    (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) /
                        FLOAT_VECTOR_SIZE,  // repeat
                    1,                      // dstBlockStride
                    1,                      // srcBlockStride
                    4,                      // dstRepeatStride
                    8                       // srcRepeatStride
                );
                SET_FLAG(V, MTE3, EVENT_ID0);
                WAIT_FLAG(V, MTE3, EVENT_ID0);

                ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                    o_gm_tensor, go_ubuf_tensor[oUbOffset * 2],
                    0,        // sid
                    sub_m,    // nBurst
                    __v * 2,  // lenBurst
                    0,        // leftPaddingNum
                    0,        // rightPaddingNum
                    0,        // srcGap
                    0         // dstGap
                );
                // ********************* move O to GM ************************
            }
            if constexpr (IS_RING) {
                if (head_loop_idx == head_loop - 1 &&
                    (uint32_t)q_i == q_seq_len - 1) {
                    uint32_t outBytes = sizeof(OUT_DTYPE);
                    uint32_t cal_head_num = q_seq_len * sub_head_num;
                    uint32_t cal_head_num_d64 = (cal_head_num + 63) / 64;
                    ln_v<ArchType::ASCEND_V220, float>(
                        lse32_ubuf_tensor, gl32_ubuf_tensor,
                        cal_head_num_d64,  // repeat
                        1,                 // dstBlockStride
                        1,                 // srcBlockStride
                        8,                 // dstRepeatStride
                        8                  // srcRepeatStride
                    );
                    PIPE_BARRIER(V);
                    add_v<ArchType::ASCEND_V220, float>(
                        lse32_ubuf_tensor, lse32_ubuf_tensor, gm32_ubuf_tensor,
                        cal_head_num_d64,  // repeat
                        1,                 // dstBlockStride
                        1,                 // src0BlockStride
                        1,                 // src1BlockStride
                        8,                 // dstRepeatStride
                        8,                 // src0RepeatStride
                        8                  // src1RepeatStride
                    );
                    PIPE_BARRIER(V);
                    conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(
                        lse_conv_ubuf_tensor, lse32_ubuf_tensor,
                        cal_head_num_d64,  // repeat
                        1,                 // dstBlockStride
                        1,                 // srcBlockStride
                        4,                 // dstRepeatStride
                        8                  // srcRepeatStride
                    );
                    SET_FLAG(V, MTE3, EVENT_ID1);
                    WAIT_FLAG(V, MTE3, EVENT_ID1);
                    // 搬出lse
                    uint32_t dstGap = (q_heads - sub_head_num) * outBytes;
                    ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                        lse_gm_tensor[(int64_t)(o_offset / __k)],
                        lse_conv_ubuf_tensor,
                        0,                        // sid
                        q_seq_len,                // nBurst
                        outBytes * sub_head_num,  // lenBurst
                        0,                        // leftPaddingNum
                        0,                        // rightPaddingNum
                        0,                        // srcGap
                        dstGap                    // dstGap
                    );
                    SET_FLAG(MTE3, V, EVENT_ID1);
                    WAIT_FLAG(MTE3, V, EVENT_ID1);
                }
            }
        } else if (head_loop > 1) {
            SET_FLAG(V, MTE3, EVENT_ID5);
            WAIT_FLAG(V, MTE3, EVENT_ID5);
            ub_to_gm<ArchType::ASCEND_V220, float>(
                go_gm_tensor, go32_ubuf_tensor[oUbOffset], 0, 1,
                sub_m * round_v / FLOAT_BLOCK_SIZE, 0, 0);
        }
        SET_FLAG(MTE3, MTE2, oPingPongFlag + 4);
    }

    __aicore__ __attribute__((always_inline)) inline void InnerRunVectorChange(
        uint32_t cur_batch, uint32_t start_head, uint32_t cur_nIndx,
        uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
        uint32_t embed_split_size_v, uint32_t embed_split_loop_v) {
        __gm__ const NdMLATiling *__restrict ndtiling =
            &(tiling_gm->ndMLATiling);

        uint32_t addr_o_high32 = (uint32_t)(ndtiling->oSeqHigh[cur_batch]);
        uint32_t addr_o_loww32 = (uint32_t)(ndtiling->oSeqLow[cur_batch]);
        uint64_t addr_o_scalar =
            (uint64_t)(((uint64_t)addr_o_high32) << 32 | addr_o_loww32);

        uint32_t addr_mask_high32 = (uint32_t)(ndtiling->maskHigh[cur_batch]);
        uint32_t addr_mask_loww32 = (uint32_t)(ndtiling->maskLow[cur_batch]);
        uint64_t addr_mask_scalar =
            (uint64_t)(((uint64_t)addr_mask_high32) << 32 | addr_mask_loww32);

        uint32_t mask_offset = addr_mask_scalar;

        uint32_t pp_n_scalar = block_size;  // 64
        uint32_t sub_n_loop = pp_n_scalar / block_size;
        uint32_t real_n_loop = (cur_kv_seqlen + block_size - 1) / block_size;

        uint32_t n_loop = (cur_kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;

        uint32_t qk_n = pp_n_scalar;
        uint32_t qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);

        uint32_t qk_n_2 = pp_n_scalar;
        uint32_t qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);

        // split head num to two vectors
        uint32_t sub_head_num = (sub_block_idx == 1)
                                    ? (cur_head_num - cur_head_num / 2)
                                    : cur_head_num / 2;  // 16
        uint32_t sub_m = sub_head_num * cur_q_seqlen;    // 16 * 3 = 48

        uint32_t head_idx =
            (sub_block_idx == 0)
                ? start_head
                : start_head + cur_head_num / 2 * cur_q_seqlen;  // not used

        o_offset = addr_o_scalar + start_head * embedding_size +
                   sub_block_idx * cur_head_num / 2 *
                       embedding_size;  // for NSD -> SND

        uint32_t sub_m_d128 = (sub_m + 127) / 128;  // up aligned to 128
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;

        uint32_t start_kv = 0;
        /* if tail length smalller than q_len - 1, then need to mask the last
         * two tile*/
        uint32_t tail_len = cur_kv_seqlen - (n_loop - 1) * pp_n_scalar;
        bool prev_tail_mask = (n_loop > 1 && tail_len < cur_q_seqlen - 1);
        for (uint32_t n_idx = 0; n_idx < n_loop + 1; n_idx++) {
            if (n_idx != n_loop) {
                bool need_mask = false;
                uint32_t mask_start_offset = 0;
                if (n_idx == (n_loop - 2)) {
                    need_mask = prev_tail_mask;
                    mask_start_offset =
                        need_mask ? (tail_len + MASK_COLUMNS - 1) * MASK_COLUMNS
                                  : 0;
                }
                if (n_idx == (n_loop - 1)) {
                    qk_n = (cur_kv_seqlen - n_idx * pp_n_scalar);
                    qk_round_n = RoundUp<16>(qk_n);
                    need_mask = true;
                    mask_start_offset = (qk_n - 1) * MASK_COLUMNS;
                }
                WaitFlagDev(QK_READY_DECODER);
                /* ************ softmax1 stage1  ************* */
                WAIT_FLAG(MTE3, MTE2, EVENT_ID3);
                if (sub_m > 0) {
                    if (mask_type == 3) {
                        mask_start_offset = mask_offset + n_idx * pp_n_scalar;
                    }
                    // input QK shape (sub_m, qk_round_n)
                    if (n_idx % 2 == 0) {
                        SoftmaxStage1(
                            p_gm_tensor[(uint64_t)block_idx * TMP_SIZE *
                                            T_BLOCK_OFFSET +
                                        (uint64_t)sub_block_idx * cur_head_num *
                                            cur_q_seqlen / 2 * qk_round_n *
                                            T_BLOCK_OFFSET +
                                        (uint64_t)(n_idx % 2) * TMP_SIZE *
                                            T_BLOCK_OFFSET / 2],
                            s_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER +
                                        (int64_t)sub_block_idx * cur_head_num *
                                            cur_q_seqlen / 2 * qk_round_n +
                                        (uint64_t)(n_idx % 2) *
                                            TMP_SIZE_DECODER / 2],
                            s_rope_gm_tensor
                                [(int64_t)block_idx * TMP_SIZE_DECODER +
                                 (int64_t)sub_block_idx * cur_head_num *
                                     cur_q_seqlen / 2 * qk_round_n +
                                 (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                            mask_gm_tensor[mask_start_offset], dm32_ubuf_tensor,
                            ll_ubuf_tensor, pm32_ubuf_tensor, n_idx, qk_n,
                            qk_round_n, sub_m, 0, sub_n_loop, cur_batch,
                            start_kv, real_n_loop, head_idx, pm_flag_scalar1,
                            cur_q_seqlen, cur_kv_seqlen, need_mask);
                    } else {
                        SoftmaxStage1(
                            p_gm_tensor[(uint64_t)block_idx * TMP_SIZE *
                                            T_BLOCK_OFFSET +
                                        (uint64_t)sub_block_idx * cur_head_num *
                                            cur_q_seqlen / 2 * qk_round_n *
                                            T_BLOCK_OFFSET +
                                        TMP_SIZE * T_BLOCK_OFFSET / 2],
                            s_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER +
                                        (int64_t)sub_block_idx * cur_head_num *
                                            cur_q_seqlen / 2 * qk_round_n +
                                        TMP_SIZE_DECODER / 2],
                            s_rope_gm_tensor[(int64_t)block_idx *
                                                 TMP_SIZE_DECODER +
                                             (int64_t)sub_block_idx *
                                                 cur_head_num * cur_q_seqlen /
                                                 2 * qk_round_n +
                                             TMP_SIZE_DECODER / 2],
                            mask_gm_tensor[mask_start_offset],
                            dm32_stage2_ubuf_tensor, ll_stage2_ubuf_tensor,
                            pm32_ubuf_stage2_tensor, n_idx, qk_n, qk_round_n,
                            sub_m, 0, sub_n_loop, cur_batch, start_kv,
                            real_n_loop, head_idx, pm_flag_scalar2,
                            cur_q_seqlen, cur_kv_seqlen, need_mask);
                    }
                }
                FftsCrossCoreSync<PIPE_MTE3, 2>(SOFTMAX_READY_DECODER);

                SET_FLAG(MTE3, MTE2, EVENT_ID3);
            }
            /* ************ softmax2 stage1  ************* */

            uint32_t process_row_num = 16;
            if (n_idx != 0) {
                if (n_idx == n_loop) {
                    qk_n_2 = (cur_kv_seqlen - (n_idx - 1) * pp_n_scalar);
                    qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
                }
                WaitFlagDev(UPDATE_READY_DECODER);
                if (sub_m > 0) {
                    uint32_t head_loop =
                        (sub_m + process_row_num - 1) / process_row_num;

                    uint32_t one_process_copy_qseqlen_num = 0;
                    uint32_t head_used_col_num = 0;
                    uint32_t tail_used_col_num = 0;

                    for (uint32_t head_loop_idx = 0; head_loop_idx < head_loop;
                         ++head_loop_idx) {
                        uint32_t head_offset =
                            head_loop_idx * process_row_num * round_v;
                        uint32_t cur_sub_m =
                            head_loop_idx == (head_loop - 1)
                                ? sub_m - head_loop_idx * process_row_num
                                : process_row_num;  // 15 or 3

                        if (cur_q_seqlen > 1) {
                            tail_used_col_num =
                                head_loop_idx * process_row_num % sub_head_num;
                            head_used_col_num =
                                (sub_head_num - tail_used_col_num) %
                                sub_head_num;
                            if (head_used_col_num < cur_sub_m) {
                                one_process_copy_qseqlen_num =
                                    (cur_sub_m - head_used_col_num) /
                                    sub_head_num;
                                tail_used_col_num =
                                    (cur_sub_m - head_used_col_num) %
                                    sub_head_num;
                            } else {
                                head_used_col_num = cur_sub_m;
                                one_process_copy_qseqlen_num = 0;
                                tail_used_col_num = 0;
                            }
                        }
                        uint32_t out_o_offset =
                            (head_loop_idx * process_row_num / sub_head_num *
                                 q_heads +
                             head_loop_idx * process_row_num % sub_head_num) *
                            round_v;  // modified, round_v = 512

                        SoftmaxStage2MLAHeadLoop(
                            o_tmp_gm_tensor[(
                                uint64_t)(block_idx * TMP_SIZE * 2 +
                                          sub_block_idx * cur_head_num *
                                              cur_q_seqlen / 2 * round_v +
                                          head_offset +
                                          ((n_idx - 1) % 2) * TMP_SIZE)],
                            go_gm_tensor[(
                                uint64_t)(block_idx * TMP_SIZE +
                                          sub_block_idx * cur_head_num *
                                              cur_q_seqlen / 2 * round_v +
                                          head_offset)],
                            o_gm_tensor[(uint64_t)(o_offset + out_o_offset)],
                            dm32_ubuf_tensor[(uint64_t)((n_idx - 1) % 2 * 128 +
                                                        head_loop_idx *
                                                            process_row_num)],
                            ll_ubuf_tensor[(uint64_t)((n_idx - 1) % 2 * 256 +
                                                      head_loop_idx *
                                                          process_row_num)],
                            pm32_ubuf_tensor[(uint64_t)((n_idx - 1) % 2 * 128 +
                                                        head_loop_idx *
                                                            process_row_num)],
                            n_idx - 1, n_loop, qk_n_2,
                            RoundUp<T_BLOCK_SIZE>(qk_round_n_2), cur_sub_m,
                            o_offset,
                            head_idx + head_loop_idx * process_row_num,
                            pm_flag_scalar1, head_loop, head_loop_idx,
                            cur_q_seqlen, sub_head_num, cur_head_num,
                            head_used_col_num, one_process_copy_qseqlen_num,
                            tail_used_col_num);
                    }
                }
            }
        }
    }

    template <bool flashDecodingVec = false>
    __aicore__ __attribute__((always_inline)) inline void
    InnerRunVectorChangeTP1(uint32_t cur_batch, uint32_t start_head,
                            uint32_t cur_nIndx, uint32_t cur_q_seqlen,
                            uint32_t cur_kvs_seqlen, uint32_t cur_head_num,
                            uint32_t offset_tiling, uint32_t embed_split_size_v,
                            uint32_t embed_split_loop_v,
                            uint32_t prev_split_num = 0, uint32_t split_num = 1,
                            uint32_t kvEndFlag = 1) {
        uint32_t cur_kv_seqlen =
            cur_kvs_seqlen + (-cur_q_seqlen + 1) * kvEndFlag;
        uint64_t addr_o_scalar = 0;
        uint64_t addr_mask_scalar = 0;
        uint32_t mask_offset = addr_mask_scalar;

        uint32_t pp_n_scalar = block_size;  // 64
        uint32_t sub_n_loop = pp_n_scalar / block_size;

        uint32_t n_loop =
            (cur_kv_seqlen + (cur_q_seqlen - 1) * kvEndFlag + pp_n_scalar - 1) /
            pp_n_scalar;

        uint32_t qk_n = pp_n_scalar;
        uint32_t qk_round_n = RoundUp<T_BLOCK_SIZE>(qk_n);

        uint32_t qk_n_2 = pp_n_scalar;
        uint32_t qk_round_n_2 = RoundUp<T_BLOCK_SIZE>(qk_n_2);

        // split head num to two vectors
        uint32_t sub_head_num = (sub_block_idx == 1)
                                    ? (cur_head_num - cur_head_num / 2)
                                    : cur_head_num / 2;  // 16
        uint32_t sub_m = sub_head_num;                   // 16 * 3 = 48

        uint32_t head_idx =
            (sub_block_idx == 0)
                ? 0
                : 0 + cur_head_num / 2 * cur_q_seqlen;  // not used

        o_offset = addr_o_scalar + start_head * embedding_size +
                   sub_block_idx * cur_head_num / 2 *
                       embedding_size;  // for NSD -> SND

        uint32_t sub_m_d128 = (sub_m + 127) / 128;  // up aligned to 128
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;

        uint32_t start_kv = 0;
        constexpr uint32_t s_block_stack = KV_FLOW_NUM;
        uint32_t qk_gm_stride = pp_n_scalar * s_block_stack;
        uint32_t m_slice = FLOAT_VECTOR_SIZE / s_block_stack;
        if constexpr (std::is_same_v<IN_DTYPE, int8_t>) {
            m_slice = 8;
            WAIT_FLAG(V, MTE2, EVENT_ID7);
            gm_to_ub<ArchType::ASCEND_V220, mmScaleType>(
                descale_q1_ubuf_tensor, deq_scale_gm_tensor_q1[head_idx],
                0,                                // sid
                1,                                // nBurst
                cur_head_num / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                                // srcGap
                0                                 // dstGap
            );
            gm_to_ub<ArchType::ASCEND_V220, mmScaleType>(
                descale_k1_ubuf_tensor, deq_scale_gm_tensor_k1[head_idx],
                0,                                // sid
                1,                                // nBurst
                cur_head_num / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                                // srcGap
                0                                 // dstGap
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }

        uint32_t m_end = (sub_m + m_slice - 1) / m_slice;
        for (uint32_t n_idx = 0; n_idx < n_loop + s_block_stack;
             n_idx += s_block_stack) {
            if (n_idx < n_loop) {
                if (sub_m == 0) {
                    WaitFlagDev(QK_READY_DECODER);
                }
                for (int q_i = 0; q_i < cur_q_seqlen; q_i++) {
                    if (n_idx + s_block_stack > n_loop - 1) {
                        qk_n = (cur_kv_seqlen + q_i * kvEndFlag -
                                n_idx * pp_n_scalar);
                        qk_gm_stride = RoundUp<T_BLOCK_SIZE>(
                            cur_kv_seqlen + (cur_q_seqlen - 1) * kvEndFlag -
                            n_idx * pp_n_scalar);
                    } else {
                        qk_n = pp_n_scalar * s_block_stack;
                        qk_gm_stride = RoundUp<T_BLOCK_SIZE>(qk_n);
                    }
                    qk_round_n = RoundUp<T_BLOCK_SIZE>(qk_n);
                    uint32_t pingpong_flag = 0;
                    for (uint32_t m_ind = 0; m_ind < m_end; m_ind++) {
                        uint32_t row_offset = m_ind * m_slice;
                        uint32_t curr_m =
                            m_ind == m_end - 1 ? sub_m - row_offset : m_slice;
                        uint32_t s_ub_offset, p_gm_offset, s_gm_offset;

                        if (m_ind == 0 && q_i == 0) {
                            WaitFlagDev(QK_READY_DECODER);
                        }
                        if (curr_m == 0 || qk_n == 0) {
                            continue;
                        }
                        if constexpr (flashDecodingVec) {
                            if (n_idx == 0) {
                                if (gl_flag_scalar == 1) {
                                    WAIT_FLAG(MTE3, V, EVENT_ID2);
                                    gl_flag_scalar = 0;
                                }
                            }
                        }
                        if constexpr (std::is_same_v<IN_DTYPE, int8_t>) {
                            qk_gm_stride = pp_n_scalar * s_block_stack;
                            s_ub_offset = pingpong_flag * 4096;
                            uint32_t s_rope_offset = 2 * 4096;
                            p_gm_offset =
                                (uint64_t)block_idx * P_TMP_SIZE +
                                (uint64_t)sub_block_idx * cur_head_num / 2 *
                                    qk_gm_stride +
                                row_offset * qk_gm_stride +
                                (uint64_t)((n_idx / s_block_stack) % 2) *
                                    P_TMP_SIZE / 2;
                            s_gm_offset =
                                (int64_t)block_idx * TMP_SIZE_DECODER * 4 +
                                (int64_t)sub_block_idx * cur_head_num / 2 *
                                    qk_gm_stride +
                                row_offset * qk_gm_stride +
                                (uint64_t)((n_idx / s_block_stack) % 2) *
                                    TMP_SIZE_DECODER * 4 / 2;
                            OnlineSoftmaxStage1Quant(
                                ls32_ubuf_tensor[s_ub_offset]
                                    .template ReinterpretCast<int32_t>(),  // s
                                ls32_ubuf_tensor[s_ub_offset +
                                                 s_rope_offset],  // s rope
                                mask_ubuf_tensor
                                    .template ReinterpretCast<float>(),
                                lm32_ubuf_tensor[row_offset],
                                hm32_ubuf_tensor[row_offset],
                                gm32_ubuf_tensor[row_offset],
                                dm32_ubuf_tensor[((n_idx / s_block_stack) % 2) *
                                                     UB_FLOAT_LINE_SIZE * 2 +
                                                 row_offset],
                                ls32_ubuf_tensor[s_ub_offset],
                                ll_ubuf_tensor[row_offset],
                                gl32_ubuf_tensor[q_i * FLOAT_VECTOR_SIZE +
                                                 row_offset],
                                descale_q1_ubuf_tensor[row_offset],
                                pm32_ubuf_tensor[((n_idx / s_block_stack) % 2) *
                                                     UB_FLOAT_LINE_SIZE * 2 +
                                                 row_offset],
                                lp_ubuf_tensor
                                    [s_ub_offset * 4 /
                                     sizeof(
                                         IN_DTYPE)],  // lp_ubuf_tensor复用了ls32_ubuf_tensor，且为half型需要*2
                                tv32_ubuf_tensor, s_gm_tensor[s_gm_offset],
                                s_rope_gm_tensor[s_gm_offset],
                                p_gm_tensor[p_gm_offset],
                                (n_idx / s_block_stack) == 0, this->tor, curr_m,
                                qk_n, qk_round_n, qk_gm_stride, pingpong_flag);
                        } else {
                            qk_gm_stride = pp_n_scalar;
                            s_ub_offset = pingpong_flag * 8192;
                            p_gm_offset =
                                (uint64_t)block_idx * TMP_SIZE * 2 * 2 +
                                q_i * TMP_SIZE +
                                (uint64_t)sub_block_idx * cur_head_num / 2 *
                                    qk_gm_stride +
                                row_offset * qk_gm_stride +
                                (uint64_t)((n_idx / s_block_stack) % 2) *
                                    TMP_SIZE * 2;

                            s_gm_offset =
                                (int64_t)block_idx * TMP_SIZE_DECODER * 4 * 2 +
                                q_i * TMP_SIZE_DECODER * 2 +
                                (int64_t)sub_block_idx * cur_head_num / 2 *
                                    qk_gm_stride +
                                row_offset * qk_gm_stride +
                                (uint64_t)((n_idx / s_block_stack) % 2) *
                                    TMP_SIZE_DECODER * 2 * 2;
                            OnlineSoftmaxStage1New<IN_DTYPE, IN_DTYPE>(
                                ls32_ubuf_tensor[s_ub_offset], mask_ubuf_tensor,
                                mask_ubuf_tensor
                                    .template ReinterpretCast<float>(),
                                lm32_ubuf_tensor[0], hm32_ubuf_tensor[0],
                                gm32_ubuf_tensor[q_i * FLOAT_VECTOR_SIZE +
                                                 row_offset],
                                dm32_ubuf_tensor[((n_idx / s_block_stack) % 2) *
                                                     UB_FLOAT_LINE_SIZE * 2 +
                                                 q_i * FLOAT_VECTOR_SIZE +
                                                 row_offset],
                                ls32_ubuf_tensor[s_ub_offset],
                                ll_ubuf_tensor[0],
                                gl32_ubuf_tensor[q_i * FLOAT_VECTOR_SIZE +
                                                 row_offset],
                                lp_ubuf_tensor[s_ub_offset * 2],
                                tv32_ubuf_tensor, s_gm_tensor[s_gm_offset],
                                p_gm_tensor[p_gm_offset], n_idx == 0, this->tor,
                                curr_m, qk_n, qk_round_n, qk_gm_stride,
                                pingpong_flag);
                        }
                        pingpong_flag = 1 - pingpong_flag;
                    }
                }
                FftsCrossCoreSync<PIPE_MTE3, 2>(SOFTMAX_READY_DECODER);
            }
            /* ************ softmax2 stage1  ************* */
            uint32_t process_row_num = 8;
            if constexpr (std::is_same_v<IN_DTYPE, int8_t>) {
                process_row_num = 16;
            }
            // stage2也被删掉 head32的时候可不改sm2
            if (n_idx >= s_block_stack) {
                WaitFlagDev(UPDATE_READY_DECODER);
                for (int32_t q_i = 0; q_i < cur_q_seqlen; q_i++) {
                    int32_t real_n_loop =
                        (cur_kv_seqlen + q_i * kvEndFlag + pp_n_scalar - 1) /
                        pp_n_scalar;
                    int32_t real_kv_seqlen_left =
                        cur_kv_seqlen + q_i * kvEndFlag -
                        (n_idx - s_block_stack) * pp_n_scalar;
                    if (real_kv_seqlen_left <= 0) {
                        continue;
                    }

                    if (sub_m > 0) {
                        uint32_t head_loop =
                            (sub_m + process_row_num - 1) / process_row_num;

                        for (uint32_t head_loop_idx = 0;
                             head_loop_idx < head_loop; ++head_loop_idx) {
                            uint32_t head_offset =
                                head_loop_idx * process_row_num * round_v;
                            uint32_t cur_sub_m =
                                head_loop_idx == (head_loop - 1)
                                    ? sub_m - head_loop_idx * process_row_num
                                    : process_row_num;

                            uint32_t out_o_offset =
                                head_loop_idx * process_row_num * round_v +
                                q_i * q_heads * 512;
                            uint32_t o_tmp_gm_offset, go_gm_offset;
                            if constexpr (std::is_same_v<IN_DTYPE, int8_t>) {
                                o_tmp_gm_offset =
                                    block_idx * TMP_SIZE * 2 +
                                    ((n_idx / s_block_stack - 1) % 2) *
                                        TMP_SIZE +
                                    sub_block_idx * cur_head_num / 2 * round_v +
                                    head_offset;
                                go_gm_offset =
                                    block_idx * TMP_SIZE +
                                    sub_block_idx * cur_head_num / 2 * round_v +
                                    head_offset;
                            } else {
                                o_tmp_gm_offset =
                                    block_idx * TMP_SIZE * 2 * 2 +
                                    ((n_idx / s_block_stack - 1) % 2) *
                                        TMP_SIZE * 2 +
                                    q_i * 128 * 512 +
                                    sub_block_idx * cur_head_num / 2 * round_v +
                                    head_offset;
                                go_gm_offset =
                                    block_idx * TMP_SIZE * 2 + q_i * 128 * 512 +
                                    sub_block_idx * cur_head_num / 2 * round_v +
                                    head_offset;
                            }
                            SoftmaxStage2MLAHeadLoopTP1<flashDecodingVec>(
                                o_tmp_gm_tensor[o_tmp_gm_offset],
                                go_gm_tensor[go_gm_offset],
                                o_gm_tensor[(uint64_t)(o_offset +
                                                       out_o_offset)],
                                dm32_ubuf_tensor[(
                                    uint64_t)((n_idx / s_block_stack - 1) % 2 *
                                                  UB_FLOAT_LINE_SIZE * 2 +
                                              q_i * FLOAT_VECTOR_SIZE +
                                              head_loop_idx * process_row_num)],
                                gl32_ubuf_tensor[q_i * FLOAT_VECTOR_SIZE +
                                                 head_loop_idx *
                                                     process_row_num],
                                pm32_ubuf_tensor[(
                                    uint64_t)((n_idx / s_block_stack - 1) % 2 *
                                                  UB_FLOAT_LINE_SIZE * 2 +
                                              q_i * FLOAT_VECTOR_SIZE +
                                              head_loop_idx * process_row_num)],
                                descale_k1_ubuf_tensor[head_loop_idx *
                                                       process_row_num],
                                n_idx, real_n_loop, cur_sub_m, o_offset,
                                q_i * q_heads +
                                    cur_head_num / 2 * sub_block_idx +
                                    head_loop_idx * process_row_num,
                                head_loop, head_loop_idx, cur_q_seqlen, q_i,
                                sub_head_num,
                                q_i * FLOAT_VECTOR_SIZE +
                                    head_loop_idx * process_row_num,
                                cur_nIndx, prev_split_num, split_num);
                        }
                    }
                }
            }
        }
        if constexpr (std::is_same_v<IN_DTYPE, int8_t>) {
            SET_FLAG(V, MTE2, EVENT_ID7);
        }
    }

   private:
    __gm__ mm1CopyType *__restrict__ s_gm{nullptr};
    __gm__ IN_DTYPE *__restrict__ p_gm{nullptr};
    __gm__ mm2CopyType *__restrict__ o_tmp_gm{nullptr};
    __gm__ float *__restrict__ go_gm{nullptr};
    __gm__ OUT_DTYPE *__restrict__ lse_gm{nullptr};
    __gm__ OUT_DTYPE *__restrict__ o_gm{nullptr};
    __gm__ OUT_DTYPE *__restrict__ mask_gm{nullptr};
    __gm__ MLATilingData *__restrict__ tiling_gm{nullptr};

    UbufAlloc<blockStack> UbAllocator;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> ls32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.ls32_ubuf_offset);
    AscendC::LocalTensor<float> ls32_quant_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.ls32_quant_ubuf_offset);
    AscendC::LocalTensor<half> ls16_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, half>(
            UbAllocator.ls32_ubuf_offset);
    AscendC::LocalTensor<IN_DTYPE> lp_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, IN_DTYPE>(
            UbAllocator.lp_ubuf_offset);
    AscendC::LocalTensor<float> lp32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.lp32_ubuf_offset);
    AscendC::LocalTensor<OUT_DTYPE> mask_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(
            UbAllocator.mask_ubuf_offset);
    AscendC::LocalTensor<float> lo_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.lo_ubuf_offset);
    AscendC::LocalTensor<float> mask32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.mask32_ubuf_offset);
    AscendC::LocalTensor<float> lm32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.lm32_ubuf_offset);
    AscendC::LocalTensor<float> hm32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.hm32_ubuf_offset);
    AscendC::LocalTensor<float> pm32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.pm32_ubuf_offset);
    AscendC::LocalTensor<float> pm32_ubuf_stage2_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.pm32_ubuf_stage2_offset);
    AscendC::LocalTensor<float> gm32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.gm32_ubuf_offset);
    AscendC::LocalTensor<float> dm32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.dm32_ubuf_offset);
    AscendC::LocalTensor<float> descale_q1_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.descale1_offset);
    AscendC::LocalTensor<float> descale_k1_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.descale2_offset);

    AscendC::LocalTensor<OUT_DTYPE> lse_conv_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(
            UbAllocator.tv32_ubuf_offset);
    AscendC::LocalTensor<float> lse32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.tv32_ubuf_offset);

    AscendC::LocalTensor<float> dm32_stage2_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.dm32_ubuf_stage2_offset);
    AscendC::LocalTensor<float> ll_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(UbAllocator.ll_ubuf_offset);
    AscendC::LocalTensor<float> ll_stage2_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.ll_ubuf_stage2_offset);
    AscendC::LocalTensor<OUT_DTYPE> gl_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(
            UbAllocator.gl_ubuf_offset);
    AscendC::LocalTensor<float> gl32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.gl32_ubuf_offset);
    AscendC::LocalTensor<float> tv32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.tv32_ubuf_offset);
    AscendC::LocalTensor<OUT_DTYPE> go_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(
            UbAllocator.go_ubuf_offset);
    AscendC::LocalTensor<float> go32_ubuf_tensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            UbAllocator.go32_ubuf_offset);

    const uint32_t ll_ubuf_offset = 0;
    const uint32_t lm_ubuf_offset = 1 * STAGE2_UB_UINT8_BLOCK_SIZE;
    const uint32_t tl_ubuf_offset = 2 * STAGE2_UB_UINT8_BLOCK_SIZE;
    AscendC::LocalTensor<float> lTensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            ll_ubuf_offset);  // L，Scale复用
    AscendC::LocalTensor<float> lMaxTensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(lm_ubuf_offset);  // L max
    AscendC::LocalTensor<float> lTmptensor =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(
            tl_ubuf_offset);  // 中间结果 tmp l

    AscendC::GlobalTensor<OUT_DTYPE> mask_gm_tensor;
    AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor;
    AscendC::GlobalTensor<OUT_DTYPE> lse_gm_tensor;
    AscendC::GlobalTensor<mm1CopyType> s_gm_tensor;
    AscendC::GlobalTensor<float> s_rope_gm_tensor;
    AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor;
    AscendC::GlobalTensor<mm2OutputType> o_tmp_gm_tensor;
    AscendC::GlobalTensor<float> go_gm_tensor;
    AscendC::GlobalTensor<float> o_core_tmp_gm_tensor;
    AscendC::GlobalTensor<float> l_gm_tensor;
    AscendC::GlobalTensor<float> deq_scale_gm_tensor_q1;
    AscendC::GlobalTensor<float> deq_scale_gm_tensor_k1;

    uint32_t go_flag_scalar{1};
    uint32_t gl_flag_scalar{1};
    uint32_t pm_flag_scalar1{1};
    uint32_t pm_flag_scalar2{0};
    uint32_t num_batches{0};
    uint32_t q_heads{0};
    uint32_t num_kv_heads{0};
    uint32_t embedding_size{0};
    uint32_t block_size{0};
    uint32_t max_context_len{0};
    uint32_t start_head{0};
    uint32_t cur_head_num{0};
    uint32_t __k{0};
    uint32_t round_k{0};
    uint32_t __v{0};
    uint32_t round_v{0};
    uint32_t cur_batch{0};
    float tor{0};
    uint64_t sub_block_idx{0};
    uint32_t batch_stride{0};
    uint32_t core_per_batch{0};
    uint32_t process_num{0};
    uint32_t tiling_head_size{0};
    uint32_t tiling_para_size{0};
    uint32_t mask_type{0};
    uint32_t kv_split_core_num{0};
    uint32_t max_kv_seq_len{0};
    uint32_t embed_split_size_v_former{0};
    uint32_t embed_split_loop_v_former{1};
    uint32_t embed_split_size_v_tail{0};
    uint32_t embed_split_loop_v_tail{1};
    uint32_t max_num_blocks_per_query{0};
    uint64_t o_offset{0};
    uint32_t totalTaskNum{0};
    uint32_t flashDecodingTaskNum{0};
    uint32_t maxKVSeqLen{0};
    uint32_t cur_qn_blk_size{0};
    uint32_t block_dim{0};
};
#endif

// 这里是kernel侧的启动函数，选择合适的kernel执行,Run()
extern "C" __global__ __aicore__ void mla(
    __gm__ uint8_t *__restrict__ qNope, __gm__ uint8_t *__restrict__ qRope,
    __gm__ uint8_t *__restrict__ ctKV, __gm__ uint8_t *__restrict__ kRope,
    __gm__ uint8_t *__restrict__ blockTables,
    __gm__ uint8_t *__restrict__ contextLens, __gm__ uint8_t *__restrict__ mask,
    __gm__ uint8_t *__restrict__ qseqlen,
    __gm__ uint8_t *__restrict__ qkDescale,
    __gm__ uint8_t *__restrict__ pvDescale,
    __gm__ uint8_t *__restrict__ attenOut,
    __gm__ uint8_t *__restrict__ workspace,
    __gm__ uint8_t *__restrict__ tiling) {
    GET_TILING_DATA_WITH_STRUCT(MLATilingData, tiling_data_in, tiling);

    __gm__ const MLATilingData *__restrict tiling_data =
        reinterpret_cast<__gm__ const MLATilingData *>(tiling);

    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);

    SetAtomicnone();
    SetMasknorm();
#ifdef __DAV_C220_VEC__
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
#elif __DAV_C220_CUBE__
    SetPadding<uint64_t>(0);
    SetNdpara(1, 0, 0);
#endif

    if (TILING_KEY_IS(0)) {  // fp16
#ifdef __DAV_C220_CUBE__
        MLAttentionOp_C<TilingKeyType::TILING_HALF_DATA, half, half, half,
                        InputFormat::ND_FORMAT>
            pa_aic_fp16{};
        pa_aic_fp16.Init(qNope, qRope, ctKV, kRope, blockTables, user,
                         tiling_data);
        pa_aic_fp16.Run();
#elif __DAV_C220_VEC__
        MLAttentionOp_V<TilingKeyType::TILING_HALF_DATA, half, half> pa_aiv{};
        pa_aiv.Init(qkDescale, pvDescale, mask, attenOut, user, tiling_data);
        pa_aiv.Run();
#endif
    } else if (TILING_KEY_IS(16)) {  // fp16
#ifdef __DAV_C220_CUBE__
        MLAttentionOp_C<TilingKeyType::TILING_HALF_DATA, half, half, half,
                        InputFormat::NZ_FORMAT>
            pa_aic_fp16{};
        pa_aic_fp16.Init(qNope, qRope, ctKV, kRope, blockTables, user,
                         tiling_data);
        pa_aic_fp16.Run();
#elif __DAV_C220_VEC__
        MLAttentionOp_V<TilingKeyType::TILING_HALF_DATA, half, half> pa_aiv{};
        pa_aiv.Init(qkDescale, pvDescale, mask, attenOut, user, tiling_data);
        pa_aiv.Run();
#endif
    } else if (TILING_KEY_IS(4)) {  // fp16
#ifdef __DAV_C220_CUBE__
        MLAttentionOp_C<TilingKeyType::TILING_HALF_DATA, half, half, half,
                        InputFormat::ND_FORMAT>
            pa_aic_fp16{};
        pa_aic_fp16.Init(qNope, qRope, ctKV, kRope, blockTables, user,
                         tiling_data);
        pa_aic_fp16.RunTP1();
#elif __DAV_C220_VEC__
        MLAttentionOp_V<TilingKeyType::TILING_HALF_DATA, half, half, false,
                        BlockStack::FOUR_FLOW>
            pa_aiv{};
        pa_aiv.Init(qkDescale, pvDescale, mask, attenOut, user, tiling_data);
        pa_aiv.RunTP1();
#endif
    } else if (TILING_KEY_IS(20)) {  // fp16
#ifdef __DAV_C220_CUBE__
        MLAttentionOp_C<TilingKeyType::TILING_HALF_DATA, half, half, half,
                        InputFormat::NZ_FORMAT>
            pa_aic_fp16{};
        pa_aic_fp16.Init(qNope, qRope, ctKV, kRope, blockTables, user,
                         tiling_data);
        pa_aic_fp16.RunTP1();
#elif __DAV_C220_VEC__
        MLAttentionOp_V<TilingKeyType::TILING_HALF_DATA, half, half, false,
                        BlockStack::FOUR_FLOW>
            pa_aiv{};
        pa_aiv.Init(qkDescale, pvDescale, mask, attenOut, user, tiling_data);
        pa_aiv.RunTP1();
#endif
    } else if (TILING_KEY_IS(
                   68)) {  // fp16 + nd + FOUR_FLOW + flashdecoding 0b1000100
#ifdef __DAV_C220_CUBE__
        MLAttentionOp_C<TilingKeyType::TILING_HALF_DATA, half, half, half,
                        InputFormat::ND_FORMAT, true>
            pa_aic_fp16{};
        pa_aic_fp16.Init(qNope, qRope, ctKV, kRope, blockTables, user,
                         tiling_data);
        pa_aic_fp16.RunTP1();
#elif __DAV_C220_VEC__
        MLAttentionOp_V<TilingKeyType::TILING_HALF_DATA, half, half, false,
                        BlockStack::FOUR_FLOW, true>
            pa_aiv{};
        pa_aiv.Init(qkDescale, pvDescale, mask, attenOut, user, tiling_data);
        pa_aiv.RunTP1();
#endif
    } else if (TILING_KEY_IS(
                   84)) {  // fp16 + nz + FOUR_FLOW + flashdecoding 0b1010100
#ifdef __DAV_C220_CUBE__
        MLAttentionOp_C<TilingKeyType::TILING_HALF_DATA, half, half, half,
                        InputFormat::NZ_FORMAT, true>
            pa_aic_fp16{};
        pa_aic_fp16.Init(qNope, qRope, ctKV, kRope, blockTables, user,
                         tiling_data);
        pa_aic_fp16.RunTP1();
#elif __DAV_C220_VEC__
        MLAttentionOp_V<TilingKeyType::TILING_HALF_DATA, half, half, false,
                        BlockStack::FOUR_FLOW, true>
            pa_aiv{};
        pa_aiv.Init(qkDescale, pvDescale, mask, attenOut, user, tiling_data);
        pa_aiv.RunTP1();
#endif
    }
}