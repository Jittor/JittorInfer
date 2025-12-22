/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#include "common.h"
#include "common_func.h"
#include "simd.h"
#include "iterator.h"
#include "mma.h"
#include "utils.h"
#include "kernel_operator.h"

constexpr int32_t ROW_OPS_SPEC_MASK_32 = 32;
constexpr int32_t ROW_OPS_SPEC_MASK_8 = 8;
constexpr int32_t ROW_OPS_SPEC_MASK_4 = 4;
constexpr int32_t REDUCE_UB_SIZE = 1024;
constexpr int32_t FLOAT_VECTOR_SIZE = 64;
constexpr int32_t VECTOR_SIZE = 128;
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t FLOAT_BLOCK_SIZE = 8;
constexpr int32_t S_DB_SIZE = 8192;
constexpr float HALF_MASK_COFF = -10000.0;
constexpr float BF16_MASK_COFF = -3e38;

enum class RowCalcTile {
    TAIL_TILE = 0,
    SPEC_TILE_256,
    SPEC_TILE_512
};

enum ScaleType {
    SCALE_TOR = 0,
    SCALE_LOGN = 1,
    SCALE_LOGN_FP32 = 2
};

enum class MaskType {
    MASK_TYPE_NONE = 0,
    MASK_TYPE_TRIU = 1,
    MASK_TYPE_ALIBI = 2,
    MASK_TYPE_ALIBI_COMPRESS = 6,
    MASK_TYPE_ALIBI_COMPRESS_SQRT = 7,
    MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN = 8,
    MASK_TYPE_ALIBI_COMPRESS_128 = 9,
    MASK_TYPE_CAUSAL_MASK = 10,
    MASK_TYPE_MASK_FREE = 11,
    MASK_TYPE_MASK_SWA = 12
};

__aicore__ __attribute__((always_inline)) inline void SetVecMask(int32_t len)
{
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % FLOAT_VECTOR_SIZE;
    for (int64_t i = 0; i < temp; i++) {
        mask |= one << i;
    }

    if (len == VECTOR_SIZE || len == 0) {
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else if (len >= FLOAT_VECTOR_SIZE) {
        SetVectorMask<int8_t>(mask, (uint64_t)-1);
    } else {
        SetVectorMask<int8_t>(0x0, mask);
    }
}

__aicore__ __attribute__((always_inline)) inline void GenMaskMatrix(AscendC::LocalTensor<float> mask_tensor, 
                                                                    float mask_coff) 
{
    // Format ND 32 * 32 MaskMatrix
    SetVecMask(FLOAT_VECTOR_SIZE);
    for (int i = 0; i < 3; i++){
        vector_dup(
            (__ubuf__ float*)mask_tensor[(i + 1) * 8 * 32].GetPhyAddr(), 
            (float)0, 
            3 - i, 
            4,
            0,
            33,
            0
        );
        vector_dup(
            (__ubuf__ float*)mask_tensor[(i + 1) * 8].GetPhyAddr(), 
            (float)mask_coff, 
            3 - i, 
            4,
            0,
            33,
            0
        );
    }
    uint64_t diagMask = ((uint64_t)2 << 54) + ((uint64_t)6 << 45) +
    ((uint64_t)14 << 36) + ((uint64_t)30 << 27) + ((uint64_t)62 << 18) +
    ((uint64_t)126 << 9) + 254;
    SetVectorMask<int8_t>(0x0, diagMask);
    vector_dup(
        (__ubuf__ float*)mask_tensor.GetPhyAddr(), 
        (float)mask_coff, 
        4, 
        4,
        0,
        33,
        0
    );
    SetVectorMask<int8_t>(0x0, ~diagMask);
    vector_dup(
        (__ubuf__ float*)mask_tensor.GetPhyAddr(), 
        (float)0, 
        4, 
        4,
        0,
        33,
        0
    );
    SetVecMask(FLOAT_VECTOR_SIZE);
}

template<typename T>
__aicore__ __attribute__((always_inline)) inline void SetBlockReduceMask(int32_t len);

template<typename T, RowCalcTile TILE_MODE>
struct Rowsum {
    __aicore__ __attribute__((always_inline)) inline Rowsum(
        const AscendC::LocalTensor<T> &src_ub,
        const AscendC::LocalTensor<T> &rowsum_ub,
        const AscendC::LocalTensor<T> &tmp_ub,
        uint32_t num_rows_round, uint32_t num_elems, uint32_t num_elems_aligned);
};

template<typename T, RowCalcTile TILE_MODE>
struct Rowmax {
    __aicore__ __attribute__((always_inline)) inline Rowmax(
        const AscendC::LocalTensor<T> &src_ub,
        const AscendC::LocalTensor<T> &rowmax_ub,
        const AscendC::LocalTensor<T> &tmp_ub,
        uint32_t num_rows_round, uint32_t num_elems, uint32_t num_elems_aligned);
};


template<typename S_DTYPE, typename EXP_DTYPE, typename P_DTYPE, typename MASK_DTYPE, MaskType MASK_TYPE>
struct OnlineSoftmaxStage1 {
    __aicore__ __attribute__((always_inline)) inline OnlineSoftmaxStage1(
        const AscendC::LocalTensor<S_DTYPE> &s_ub,
        const AscendC::LocalTensor<MASK_DTYPE> &mask_orig_ub,
        const AscendC::LocalTensor<S_DTYPE> &mask_processed_ub,
        const AscendC::LocalTensor<S_DTYPE> &local_rowmax_ub,
        const AscendC::LocalTensor<S_DTYPE> &hat_rowmax_ub,
        const AscendC::LocalTensor<S_DTYPE> &global_rowmax_ub,
        const AscendC::LocalTensor<S_DTYPE> &diff_rowmax_ub,
        const AscendC::LocalTensor<EXP_DTYPE> &s_exp_ub,
        const AscendC::LocalTensor<EXP_DTYPE> &local_rowsum_ub,
        const AscendC::LocalTensor<EXP_DTYPE> &global_rowsum_ub,
        const AscendC::LocalTensor<P_DTYPE> &p_ub,
        const AscendC::LocalTensor<EXP_DTYPE> &tmp_ub,
        const AscendC::GlobalTensor<S_DTYPE> &s_gm,
        const AscendC::GlobalTensor<P_DTYPE> &p_gm,
        bool first_n_iter, S_DTYPE tor,
        uint32_t m, uint32_t n_real, uint32_t n_stride, uint32_t pingpong_flag);
};


template<>
__aicore__ __attribute__((always_inline)) inline void SetBlockReduceMask<float>(int32_t len)
{
    if (len > 8 || len < 1) {
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    uint64_t subMask = ((uint64_t) 1 << len) - 1;
    uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask +
                            (subMask << 56) + (subMask << 40) + (subMask << 24) + (subMask << 8);
    SetVectorMask<int8_t>(maskValue, maskValue);
}

template<>
__aicore__ __attribute__((always_inline)) inline void SetBlockReduceMask<half>(int32_t len)
{
    if (len > 16 || len < 1) {
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    uint64_t subMask = ((uint64_t) 1 << len) - 1;
    uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
    SetVectorMask<int8_t>(maskValue, maskValue);
}

template<>
struct Rowmax<half, RowCalcTile::SPEC_TILE_512> {
    __aicore__ __attribute__((always_inline)) Rowmax(
        const AscendC::LocalTensor<half> &s_ub,
        const AscendC::LocalTensor<half> &local_rowmax_ub,
        const AscendC::LocalTensor<half> &tmp_ub,
        uint32_t m, uint32_t n_real, uint32_t n_stride)
    {
        cgmax_v<ArchType::ASCEND_V220, half>(
            tmp_ub,
            s_ub,
            m * n_stride / VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
        max_v<ArchType::ASCEND_V220, half>(
            tmp_ub[REDUCE_UB_SIZE],
            tmp_ub,
            tmp_ub[16],
            (m * BLOCK_SIZE + VECTOR_SIZE - 1) / VECTOR_SIZE,   // repeat
            1,                            // dstBlockStride
            2,                            // src0BlockStride
            2,                            // src1BlockStride
            8,                            // dstRepeatStride
            16,                            // src0RepeatStride
            16 // src1RepeatStride
        );
        PIPE_BARRIER(V);
        cgmax_v<ArchType::ASCEND_V220, half>(
            local_rowmax_ub,
            tmp_ub[REDUCE_UB_SIZE],
            (m * BLOCK_SIZE + VECTOR_SIZE - 1) / VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
    }
};

template<>
struct Rowmax<half, RowCalcTile::SPEC_TILE_256>{
    __aicore__ __attribute__((always_inline)) Rowmax(
        const AscendC::LocalTensor<half> &s_ub,
        const AscendC::LocalTensor<half> &local_rowmax_ub,
        const AscendC::LocalTensor<half> &tmp_ub,
        uint32_t m, uint32_t n_real, uint32_t n_stride)
    {
        cgmax_v<ArchType::ASCEND_V220, half>(
            tmp_ub,
            s_ub,
            m * n_stride / VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
        cgmax_v<ArchType::ASCEND_V220, half>(
            local_rowmax_ub,
            tmp_ub,
            (m * BLOCK_SIZE + VECTOR_SIZE - 1) / VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
    }
};

template<>
struct Rowmax<half, RowCalcTile::TAIL_TILE>{
    __aicore__ __attribute__((always_inline)) Rowmax(
        const AscendC::LocalTensor<half> &s_ub,
        const AscendC::LocalTensor<half> &local_rowmax_ub,
        const AscendC::LocalTensor<half> &tmp_ub,
        uint32_t m, uint32_t n_real, uint32_t n_stride)
    {
        if (n_real >= VECTOR_SIZE) {
            // 128 to 8 each row
            cgmax_v<ArchType::ASCEND_V220, half>(
                tmp_ub,
                s_ub,
                m,
                2,
                1,
                n_stride / BLOCK_SIZE
            );
            PIPE_BARRIER(V);
            // 8 to 1 each row
            SetBlockReduceMask<half>(VECTOR_SIZE / 16);
            cgmax_v<ArchType::ASCEND_V220, half>(
                local_rowmax_ub,
                tmp_ub,
                (m * 16 + VECTOR_SIZE - 1) / VECTOR_SIZE,
                1,
                1,
                8
            );
            PIPE_BARRIER(V);
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint64_t rowmax_idx = 1; rowmax_idx < (uint64_t)n_real / VECTOR_SIZE; ++rowmax_idx) {
                // kvseq 方向每128做一次blk max
                cgmax_v<ArchType::ASCEND_V220, half>(
                    tmp_ub,
                    s_ub[rowmax_idx * VECTOR_SIZE],
                    m,
                    2,
                    1,
                    n_stride / BLOCK_SIZE
                );
                PIPE_BARRIER(V);
                SetBlockReduceMask<half>(VECTOR_SIZE / 16);
                cgmax_v<ArchType::ASCEND_V220, half>(
                    tmp_ub[REDUCE_UB_SIZE],
                    tmp_ub,
                    (m * BLOCK_SIZE + VECTOR_SIZE - 1) / VECTOR_SIZE,
                    1,
                    1,
                    8
                );
                PIPE_BARRIER(V);
                SetVecMask(m);
                max_v<ArchType::ASCEND_V220, half>(
                    local_rowmax_ub,
                    local_rowmax_ub,
                    tmp_ub[REDUCE_UB_SIZE],
                    1,                        // repeat
                    1,                            // dstBlockStride
                    1,                            // src0BlockStride
                    1,                            // src1BlockStride
                    8,                            // dstRepeatStride
                    8,                            // src0RepeatStride
                    8 // src1RepeatStride
                );
                PIPE_BARRIER(V);
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (n_real % VECTOR_SIZE > 0) {
            SetVecMask(n_real % VECTOR_SIZE);
            cgmax_v<ArchType::ASCEND_V220, half>(
                tmp_ub,
                s_ub[n_real / VECTOR_SIZE * VECTOR_SIZE],
                m,
                2,
                1,
                n_stride / BLOCK_SIZE
            );
            PIPE_BARRIER(V);
            SetBlockReduceMask<half>((n_real % VECTOR_SIZE + BLOCK_SIZE - 1)/ BLOCK_SIZE);
            if (n_real < VECTOR_SIZE) {
                cgmax_v<ArchType::ASCEND_V220, half>(
                    local_rowmax_ub,
                    tmp_ub,
                    (m * BLOCK_SIZE + VECTOR_SIZE - 1) / VECTOR_SIZE,
                    1,
                    1,
                    8
                );
                PIPE_BARRIER(V);
            } else {
                cgmax_v<ArchType::ASCEND_V220, half>(
                    tmp_ub[REDUCE_UB_SIZE],
                    tmp_ub,
                    (m * BLOCK_SIZE + VECTOR_SIZE - 1) / VECTOR_SIZE,
                    1,
                    1,
                    8
                );
                PIPE_BARRIER(V);
                SetVecMask(m);
                max_v<ArchType::ASCEND_V220, half>(
                    local_rowmax_ub,
                    local_rowmax_ub,
                    tmp_ub[REDUCE_UB_SIZE],
                    1,                        // repeat
                    1,                            // dstBlockStride
                    1,                            // src0BlockStride
                    1,                            // src1BlockStride
                    8,                            // dstRepeatStride
                    8,                            // src0RepeatStride
                    8 // src1RepeatStride
                );
                PIPE_BARRIER(V);
            }
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }
};


template<>
struct Rowsum<float, RowCalcTile::SPEC_TILE_512>{
    __aicore__ __attribute__((always_inline)) inline Rowsum(
        const AscendC::LocalTensor<float> &src_ub,
        const AscendC::LocalTensor<float> &rowsum_ub,
        const AscendC::LocalTensor<float> &tmp_ub,
        uint32_t num_rows_round , uint32_t num_elems, uint32_t num_elems_aligned)
    {
        cgadd_v<ArchType::ASCEND_V220, float>(
            tmp_ub,
            src_ub,
            num_rows_round * num_elems_aligned / FLOAT_VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
        cgadd_v<ArchType::ASCEND_V220, float>(
            tmp_ub[REDUCE_UB_SIZE],
            tmp_ub,
            num_rows_round * num_elems_aligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
        cgadd_v<ArchType::ASCEND_V220, float>(
            rowsum_ub,
            tmp_ub[REDUCE_UB_SIZE],
            num_rows_round * num_elems_aligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
    }
};

template<>
struct Rowsum<float, RowCalcTile::SPEC_TILE_256>{
    __aicore__ __attribute__((always_inline)) inline Rowsum(
        const AscendC::LocalTensor<float> &src_ub,
        const AscendC::LocalTensor<float> &rowsum_ub,
        const AscendC::LocalTensor<float> &tmp_ub,
        uint32_t num_rows_round, uint32_t num_elems, uint32_t num_elems_aligned)
    {
        cgadd_v<ArchType::ASCEND_V220, float>(
            tmp_ub,
            src_ub,
            num_rows_round * num_elems_aligned / FLOAT_VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        cgadd_v<ArchType::ASCEND_V220, float>(
            tmp_ub[REDUCE_UB_SIZE],
            tmp_ub,
            num_rows_round,
            1,
            1,
            4
        );
        PIPE_BARRIER(V);
        SetBlockReduceMask<float>(ROW_OPS_SPEC_MASK_4);
        cgadd_v<ArchType::ASCEND_V220, float>(
            rowsum_ub,
            tmp_ub[REDUCE_UB_SIZE],
            (num_rows_round * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
};

template<>
struct Rowsum<float, RowCalcTile::TAIL_TILE>{
    __aicore__ __attribute__((always_inline)) inline Rowsum(
        const AscendC::LocalTensor<float> &src_ub,
        const AscendC::LocalTensor<float> &rowsum_ub,
        const AscendC::LocalTensor<float> &tmp_ub,
        uint32_t num_rows_round, uint32_t num_elems, uint32_t num_elems_aligned)
    {
        if (num_elems >= FLOAT_VECTOR_SIZE) {
            cgadd_v<ArchType::ASCEND_V220, float>(
                tmp_ub,
                src_ub,
                num_rows_round,
                1,
                1,
                num_elems_aligned / FLOAT_BLOCK_SIZE
            );
            PIPE_BARRIER(V);
            cgadd_v<ArchType::ASCEND_V220, float>(
                rowsum_ub,
                tmp_ub,
                (num_rows_round * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                8
            );
            PIPE_BARRIER(V);
            for (uint64_t rowsum_idx = 1; rowsum_idx < (uint64_t)num_elems / FLOAT_VECTOR_SIZE; ++rowsum_idx) {
                cgadd_v<ArchType::ASCEND_V220, float>(
                    tmp_ub,
                    src_ub[rowsum_idx * FLOAT_VECTOR_SIZE],
                    num_rows_round,
                    1,
                    1,
                    num_elems_aligned / FLOAT_BLOCK_SIZE
                );
                PIPE_BARRIER(V);
                cgadd_v<ArchType::ASCEND_V220, float>(
                    tmp_ub[REDUCE_UB_SIZE],
                    tmp_ub,
                    (num_rows_round * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    1,
                    1,
                    8
                );
                PIPE_BARRIER(V);
                SetVecMask(num_rows_round);
                add_v<ArchType::ASCEND_V220, float>(
                    rowsum_ub,
                    rowsum_ub,
                    tmp_ub[REDUCE_UB_SIZE],
                    1,                        // repeat
                    1,                            // dstBlockStride
                    1,                            // src0BlockStride
                    1,                            // src1BlockStride
                    8,                            // dstRepeatStride
                    8,                            // src0RepeatStride
                    8 // src1RepeatStride
                );
                PIPE_BARRIER(V);
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (num_elems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(num_elems % FLOAT_VECTOR_SIZE);
            cgadd_v<ArchType::ASCEND_V220, float>(
                tmp_ub,
                src_ub[num_elems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                num_rows_round,
                1,
                1,
                num_elems_aligned / FLOAT_BLOCK_SIZE
            );
            PIPE_BARRIER(V);
            SetBlockReduceMask<float>((num_elems % FLOAT_VECTOR_SIZE + FLOAT_BLOCK_SIZE - 1)/ FLOAT_BLOCK_SIZE);
            if (num_elems < FLOAT_VECTOR_SIZE) {
                cgadd_v<ArchType::ASCEND_V220, float>(
                    rowsum_ub,
                    tmp_ub,
                    (num_rows_round * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    1,
                    1,
                    8
                );
                PIPE_BARRIER(V);
            } else {
                cgadd_v<ArchType::ASCEND_V220, float>(
                    tmp_ub[REDUCE_UB_SIZE],
                    tmp_ub,
                    (num_rows_round * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    1,
                    1,
                    8
                );
                PIPE_BARRIER(V);
                SetVecMask(num_rows_round);
                add_v<ArchType::ASCEND_V220, float>(
                    rowsum_ub,
                    rowsum_ub,
                    tmp_ub[REDUCE_UB_SIZE],
                    1,                        // repeat
                    1,                            // dstBlockStride
                    1,                            // src0BlockStride
                    1,                            // src1BlockStride
                    8,                            // dstRepeatStride
                    8,                            // src0RepeatStride
                    8 // src1RepeatStride
                );
                PIPE_BARRIER(V);
            }
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }
};

template<>
struct Rowmax<float, RowCalcTile::SPEC_TILE_512> {
    __aicore__ __attribute__((always_inline)) inline Rowmax(
        const AscendC::LocalTensor<float> &src_ub,
        const AscendC::LocalTensor<float> &rowmax_ub,
        const AscendC::LocalTensor<float> &tmp_ub,
        uint32_t num_rows_round, uint32_t num_elems, uint32_t num_elems_aligned)
    {
        cgmax_v<ArchType::ASCEND_V220, float>(
            tmp_ub,
            src_ub,
            num_rows_round * num_elems_aligned / FLOAT_VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
        cgmax_v<ArchType::ASCEND_V220, float>(
            tmp_ub[REDUCE_UB_SIZE],
            tmp_ub,
            num_rows_round * num_elems_aligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
        cgmax_v<ArchType::ASCEND_V220, float>(
            rowmax_ub,
            tmp_ub[REDUCE_UB_SIZE],
            num_rows_round * num_elems_aligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
    }
};

template<>
struct Rowmax<float, RowCalcTile::SPEC_TILE_256>{
    __aicore__ __attribute__((always_inline)) inline Rowmax(
        const AscendC::LocalTensor<float> &src_ub,
        const AscendC::LocalTensor<float> &rowmax_ub,
        const AscendC::LocalTensor<float> &tmp_ub,
        uint32_t num_rows_round, uint32_t num_elems, uint32_t num_elems_aligned)
    {
        cgmax_v<ArchType::ASCEND_V220, float>(
            tmp_ub,
            src_ub,
            num_rows_round * num_elems_aligned / FLOAT_VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
        SetVecMask(ROW_OPS_SPEC_MASK_32);
        cgmax_v<ArchType::ASCEND_V220, float>(
            tmp_ub[REDUCE_UB_SIZE],
            tmp_ub,
            num_rows_round,
            1,
            1,
            4
        );
        PIPE_BARRIER(V);
        SetBlockReduceMask<float>(ROW_OPS_SPEC_MASK_4);
        cgmax_v<ArchType::ASCEND_V220, float>(
            rowmax_ub,
            tmp_ub[REDUCE_UB_SIZE],
            (num_rows_round * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
            1,
            1,
            8
        );
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
};

template<>
struct Rowmax<float, RowCalcTile::TAIL_TILE>{
    __aicore__ __attribute__((always_inline)) inline Rowmax(
        const AscendC::LocalTensor<float> &src_ub,
        const AscendC::LocalTensor<float> &rowmax_ub,
        const AscendC::LocalTensor<float> &tmp_ub,
        uint32_t num_rows_round, uint32_t num_elems, uint32_t num_elems_aligned)
    {
        if (num_elems >= FLOAT_VECTOR_SIZE) {
            cgmax_v<ArchType::ASCEND_V220, float>(
                tmp_ub,
                src_ub,
                num_rows_round,
                1,
                1,
                num_elems_aligned / FLOAT_BLOCK_SIZE
            );
            PIPE_BARRIER(V);
            cgmax_v<ArchType::ASCEND_V220, float>(
                rowmax_ub,
                tmp_ub,
                (num_rows_round * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                8
            );
            PIPE_BARRIER(V);
            for (uint64_t rowmax_idx = 1; rowmax_idx < (uint64_t)num_elems / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
                cgmax_v<ArchType::ASCEND_V220, float>(
                    tmp_ub,
                    src_ub[rowmax_idx * FLOAT_VECTOR_SIZE],
                    num_rows_round,
                    1,
                    1,
                    num_elems_aligned / FLOAT_BLOCK_SIZE
                );
                PIPE_BARRIER(V);
                cgmax_v<ArchType::ASCEND_V220, float>(
                    tmp_ub[REDUCE_UB_SIZE],
                    tmp_ub,
                    (num_rows_round * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    1,
                    1,
                    8
                );
                PIPE_BARRIER(V);
                SetVecMask(num_rows_round);
                max_v<ArchType::ASCEND_V220, float>(
                    rowmax_ub,
                    rowmax_ub,
                    tmp_ub[REDUCE_UB_SIZE],
                    1,                        // repeat
                    1,                            // dstBlockStride
                    1,                            // src0BlockStride
                    1,                            // src1BlockStride
                    8,                            // dstRepeatStride
                    8,                            // src0RepeatStride
                    8 // src1RepeatStride
                );
                PIPE_BARRIER(V);
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (num_elems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(num_elems % FLOAT_VECTOR_SIZE);
            cgmax_v<ArchType::ASCEND_V220, float>(
                tmp_ub,
                src_ub[num_elems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                num_rows_round,
                1,
                1,
                num_elems_aligned / FLOAT_BLOCK_SIZE
            );
            PIPE_BARRIER(V);
            SetBlockReduceMask<float>((num_elems % FLOAT_VECTOR_SIZE + FLOAT_BLOCK_SIZE - 1)/ FLOAT_BLOCK_SIZE);
            if (num_elems < FLOAT_VECTOR_SIZE) {
                cgmax_v<ArchType::ASCEND_V220, float>(
                    rowmax_ub,
                    tmp_ub,
                    (num_rows_round * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    1,
                    1,
                    8
                );
                PIPE_BARRIER(V);
            } else {
                cgmax_v<ArchType::ASCEND_V220, float>(
                    tmp_ub[REDUCE_UB_SIZE],
                    tmp_ub,
                    (num_rows_round * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    1,
                    1,
                    8
                );
                PIPE_BARRIER(V);
                SetVecMask(num_rows_round);
                max_v<ArchType::ASCEND_V220, float>(
                    rowmax_ub,
                    rowmax_ub,
                    tmp_ub[REDUCE_UB_SIZE],
                    1,                        // repeat
                    1,                            // dstBlockStride
                    1,                            // src0BlockStride
                    1,                            // src1BlockStride
                    8,                            // dstRepeatStride
                    8,                            // src0RepeatStride
                    8 // src1RepeatStride
                );
                PIPE_BARRIER(V);
            }
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }
};

template<MaskType MASK_TYPE = MaskType::MASK_TYPE_NONE>
struct OnlineSoftmaxStage1Step1 {
     __aicore__ __attribute__((always_inline)) inline OnlineSoftmaxStage1Step1(
    const AscendC::LocalTensor<float> &s_ub,
    const AscendC::LocalTensor<float> &local_rowmax_ub,
    const AscendC::LocalTensor<float> &hat_rowmax_ub,
    const AscendC::LocalTensor<float> &global_rowmax_ub,
    const AscendC::LocalTensor<float> &diff_rowmax_ub,
    const AscendC::LocalTensor<float> &s_exp_ub,
    const AscendC::LocalTensor<float> &local_rowsum_ub,
    const AscendC::LocalTensor<float> &tmp_ub,
    bool first_n_iter, float tor,
    uint32_t m, uint32_t n_real, uint32_t n_stride)
{
    uint32_t round_m = (m + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
    // *** ls = tor * ls
    muls_v<ArchType::ASCEND_V220, float>(
        s_ub,
        s_ub,
        tor,
        (m * n_stride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, // repeat
        1,                                                          // dstBlockStride
        1,                                                          // srcBlockStride
        8,                                                          // dstRepeatStride
        8                                                           // srcRepeatStride
    );
    PIPE_BARRIER(V);
    if (n_real == 512) {
        Rowmax<float, RowCalcTile::SPEC_TILE_512>(
            s_ub,
            local_rowmax_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    } else if (n_real == 256) {
        Rowmax<float, RowCalcTile::SPEC_TILE_256>(
            s_ub,
            local_rowmax_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    } else {
        Rowmax<float, RowCalcTile::TAIL_TILE>(
            s_ub,
            local_rowmax_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    }

    if (first_n_iter) {
        // *** hm = lm
        ub_to_ub<ArchType::ASCEND_V220, float>(
            hat_rowmax_ub,
            local_rowmax_ub,
            0,                          // sid
            1,                          // nBurst
            round_m / FLOAT_BLOCK_SIZE, // lenBurst
            0,                          // srcGap
            0                           // dstGap
        );
        PIPE_BARRIER(V);
    } else {
        SetVecMask(m);
        // *** hm = vmax(lm, gm)
        max_v<ArchType::ASCEND_V220, float>(
            hat_rowmax_ub,
            local_rowmax_ub,
            global_rowmax_ub,
            1,         // repeat
            1,         // dstBlockStride
            1,         // src0BlockStride
            1,         // src1BlockStride
            8,         // dstRepeatStride
            8,         // src0RepeatStride
            8          // src1RepeatStride
        );
        PIPE_BARRIER(V);
        // *** dm = gm - hm
        sub_v<ArchType::ASCEND_V220, float>(
            diff_rowmax_ub,
            global_rowmax_ub,
            hat_rowmax_ub,
            1,         // repeat
            1,         // dstBlockStride
            1,         // src0BlockStride
            1,         // src1BlockStride
            8,         // dstRepeatStride
            8,         // src0RepeatStride
            8          // src1RepeatStride
        );
        PIPE_BARRIER(V);
        // *** dm = exp(dm)
        exp_v<ArchType::ASCEND_V220, float>(
            diff_rowmax_ub,
            diff_rowmax_ub,
            1,         // repeat
            1,         // dstBlockStride
            1,         // srcBlockStride
            8,         // dstRepeatStride
            8          // srcRepeatStride
        );
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    PIPE_BARRIER(V);
    // *** gm = hm
    ub_to_ub<ArchType::ASCEND_V220, float>(
        global_rowmax_ub,
        hat_rowmax_ub,
        0,                          // sid
        1,                          // nBurst
        round_m / FLOAT_BLOCK_SIZE, // lenBurst
        0,                          // srcGap
        0                           // dstGap
    );
    PIPE_BARRIER(V);
    // *** hm_block = expand_to_block(hm), 存放于 tv
    brcb_v<ArchType::ASCEND_V220, uint32_t>(
        tmp_ub.template ReinterpretCast<uint32_t>(),
        hat_rowmax_ub.template ReinterpretCast<uint32_t>(),
        1,                         // dstBlockStride
        8,                         // dstRepeatStride
        round_m / FLOAT_BLOCK_SIZE // repeat
    );
    PIPE_BARRIER(V);
    // *** ls = ls - hm_block
    for (uint32_t vsub_idx = 0; vsub_idx < n_real / FLOAT_VECTOR_SIZE; ++vsub_idx) {
        sub_v<ArchType::ASCEND_V220, float>(
            s_ub[vsub_idx * FLOAT_VECTOR_SIZE],
            s_ub[vsub_idx * FLOAT_VECTOR_SIZE],
            tmp_ub,
            m,                           // repeat
            1,                           // dstBlockStride
            1,                           // src0BlockStride
            0,                           // src1BlockStride
            n_stride / FLOAT_BLOCK_SIZE, // dstRepeatStride
            n_stride / FLOAT_BLOCK_SIZE, // src0RepeatStride
            1                            // src1RepeatStride
        );
    }
    if (n_real % FLOAT_VECTOR_SIZE > 0) {
        SetVecMask(n_real % FLOAT_VECTOR_SIZE);
        sub_v<ArchType::ASCEND_V220, float>(
            s_ub[n_real / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            s_ub[n_real / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            tmp_ub,
            m,                           // repeat
            1,                           // dstBlockStride
            1,                           // src0BlockStride
            0,                           // src1BlockStride
            n_stride / FLOAT_BLOCK_SIZE, // dstRepeatStride
            n_stride / FLOAT_BLOCK_SIZE, // src0RepeatStride
            1                            // src1RepeatStride
        );
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);

    // *** ls = exp(ls)
    exp_v<ArchType::ASCEND_V220, float>(
        s_exp_ub,
        s_ub,
        (m * n_stride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, // repeat
        1,                                                          // dstBlockStride
        1,                                                          // srcBlockStride
        8,                                                          // dstRepeatStride
        8                                                           // srcRepeatStride
    );
    PIPE_BARRIER(V);
    // *** ll = rowsum(ls32)
    if (n_real == 512) {
        Rowsum<float, RowCalcTile::SPEC_TILE_512>(
            s_exp_ub,
            local_rowsum_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    } else if (n_real == 256) {
        Rowsum<float, RowCalcTile::SPEC_TILE_256>(
            s_exp_ub,
            local_rowsum_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    } else {
        Rowsum<float, RowCalcTile::TAIL_TILE>(
            s_exp_ub,
            local_rowsum_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    }
}
};

template<>
struct OnlineSoftmaxStage1Step1<MaskType::MASK_TYPE_CAUSAL_MASK>{
    __aicore__ __attribute__((always_inline)) inline OnlineSoftmaxStage1Step1(
    const AscendC::LocalTensor<float> &s_ub,
    const AscendC::LocalTensor<float> &mask_ub_tensor,
    const AscendC::LocalTensor<float> &local_rowmax_ub,
    const AscendC::LocalTensor<float> &hat_rowmax_ub,
    const AscendC::LocalTensor<float> &global_rowmax_ub,
    const AscendC::LocalTensor<float> &diff_rowmax_ub,
    const AscendC::LocalTensor<float> &s_exp_ub,
    const AscendC::LocalTensor<float> &local_rowsum_ub,
    const AscendC::LocalTensor<float> &tmp_ub,
    bool first_n_iter, float tor, float mask_coff,
    uint32_t m, uint32_t n_real, uint32_t n_stride, uint32_t pp_n_scalar,
    bool need_mask, uint32_t slice_row_offset, uint32_t sub_m1)
{
    uint32_t round_m = (m + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
    // *** ls = tor * ls
    muls_v<ArchType::ASCEND_V220, float>(
        s_ub,
        s_ub,
        tor,
        (m * n_stride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, // repeat
        1,                                                          // dstBlockStride
        1,                                                          // srcBlockStride
        8,                                                          // dstRepeatStride
        8                                                           // srcRepeatStride
    );
    PIPE_BARRIER(V);

    // *** ls = ls + mask
    uint32_t s_mask_blk_start = (n_real - 1) / pp_n_scalar * pp_n_scalar;
    uint32_t mask_offset = 0;
    if (GetSubBlockidx() == 1) {
        uint32_t sub_m1_d8 = sub_m1 / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
        s_mask_blk_start += sub_m1_d8;
        mask_offset = (sub_m1 - sub_m1_d8) * ROW_OPS_SPEC_MASK_32;
    }

    uint32_t last_block_n = n_stride - s_mask_blk_start;
    uint32_t s_col_offset = slice_row_offset;
    uint32_t s_ub_start_offset = s_mask_blk_start + s_col_offset;
    uint32_t rep_times = m / FLOAT_BLOCK_SIZE;
    uint32_t m_tail = m % FLOAT_BLOCK_SIZE;
    int32_t dup_blk_num = (last_block_n - BLOCK_SIZE - s_col_offset) / FLOAT_BLOCK_SIZE;
    if (need_mask) {
        for (size_t i = 0; i < rep_times; ++i, dup_blk_num--) {
            add_v<ArchType::ASCEND_V220, float>(
                s_ub[s_ub_start_offset + i * (n_stride + 1) * FLOAT_BLOCK_SIZE],
                s_ub[s_ub_start_offset + i * (n_stride + 1) * FLOAT_BLOCK_SIZE],
                mask_ub_tensor[mask_offset],
                dup_blk_num >= 0 ? 2 : 1,
                n_stride / FLOAT_BLOCK_SIZE,                     
                n_stride / FLOAT_BLOCK_SIZE,                     
                4,                           // 32 / 8
                1,           
                1,
                1                       
            );
            if (dup_blk_num > 0) {
                vector_dup(
                    (__ubuf__ float*)
                    s_ub[s_ub_start_offset + i * (n_stride + 1) * FLOAT_BLOCK_SIZE + BLOCK_SIZE].GetPhyAddr(),
                    (float)mask_coff,
                    dup_blk_num,
                    n_stride / 8,
                    0,
                    1,
                    0
                );
            }
        }
        if (m_tail) {
            SetVecMask(m_tail * FLOAT_BLOCK_SIZE);
            add_v<ArchType::ASCEND_V220, float>(
                s_ub[s_ub_start_offset + rep_times * (n_stride + 1) * FLOAT_BLOCK_SIZE],
                s_ub[s_ub_start_offset + rep_times * (n_stride + 1) * FLOAT_BLOCK_SIZE],
                mask_ub_tensor[mask_offset],
                dup_blk_num >=0 ? 2 : 1,
                n_stride / FLOAT_BLOCK_SIZE,                    
                n_stride / FLOAT_BLOCK_SIZE,                    
                4,                                
                1,
                1,
                1
            );
            if (dup_blk_num > 0) {
                vector_dup(
                    (__ubuf__ float*)
                    s_ub[s_ub_start_offset + rep_times * (n_stride + 1) * FLOAT_BLOCK_SIZE + BLOCK_SIZE]
                    .GetPhyAddr(),
                    (float)mask_coff,
                    dup_blk_num,
                    n_stride / 8,
                    0,
                    1,
                    0
                );
            }
        }
        SetVecMask(VECTOR_SIZE);
    }

    PIPE_BARRIER(V);

    if (n_real == 512) {
        Rowmax<float, RowCalcTile::SPEC_TILE_512>(
            s_ub,
            local_rowmax_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    } else if (n_real == 256) {
        Rowmax<float, RowCalcTile::SPEC_TILE_256>(
            s_ub,
            local_rowmax_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    } else {
        Rowmax<float, RowCalcTile::TAIL_TILE>(
            s_ub,
            local_rowmax_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    }

    if (first_n_iter) {
        // *** hm = lm
        ub_to_ub<ArchType::ASCEND_V220, float>(
            hat_rowmax_ub,
            local_rowmax_ub,
            0,                          // sid
            1,                          // nBurst
            round_m / FLOAT_BLOCK_SIZE, // lenBurst
            0,                          // srcGap
            0                           // dstGap
        );
        PIPE_BARRIER(V);
    } else {
        SetVecMask(m);
        // *** hm = vmax(lm, gm)
        max_v<ArchType::ASCEND_V220, float>(
            hat_rowmax_ub,
            local_rowmax_ub,
            global_rowmax_ub,
            1,         // repeat
            1,         // dstBlockStride
            1,         // src0BlockStride
            1,         // src1BlockStride
            8,         // dstRepeatStride
            8,         // src0RepeatStride
            8          // src1RepeatStride
        );
        PIPE_BARRIER(V);
        // *** dm = gm - hm
        sub_v<ArchType::ASCEND_V220, float>(
            diff_rowmax_ub,
            global_rowmax_ub,
            hat_rowmax_ub,
            1,         // repeat
            1,         // dstBlockStride
            1,         // src0BlockStride
            1,         // src1BlockStride
            8,         // dstRepeatStride
            8,         // src0RepeatStride
            8          // src1RepeatStride
        );
        PIPE_BARRIER(V);
        // *** dm = exp(dm)
        exp_v<ArchType::ASCEND_V220, float>(
            diff_rowmax_ub,
            diff_rowmax_ub,
            1,         // repeat
            1,         // dstBlockStride
            1,         // srcBlockStride
            8,         // dstRepeatStride
            8          // srcRepeatStride
        );
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    PIPE_BARRIER(V);
    // *** gm = hm
    ub_to_ub<ArchType::ASCEND_V220, float>(
        global_rowmax_ub,
        hat_rowmax_ub,
        0,                          // sid
        1,                          // nBurst
        round_m / FLOAT_BLOCK_SIZE, // lenBurst
        0,                          // srcGap
        0                           // dstGap
    );
    PIPE_BARRIER(V);
    // *** hm_block = expand_to_block(hm), 存放于 tv
    brcb_v<ArchType::ASCEND_V220, uint32_t>(
        tmp_ub.template ReinterpretCast<uint32_t>(),
        hat_rowmax_ub.template ReinterpretCast<uint32_t>(),
        1,                         // dstBlockStride
        8,                         // dstRepeatStride
        round_m / FLOAT_BLOCK_SIZE // repeat
    );
    PIPE_BARRIER(V);
    // *** ls = ls - hm_block
    for (uint32_t vsub_idx = 0; vsub_idx < n_real / FLOAT_VECTOR_SIZE; ++vsub_idx) {
        sub_v<ArchType::ASCEND_V220, float>(
            s_ub[vsub_idx * FLOAT_VECTOR_SIZE],
            s_ub[vsub_idx * FLOAT_VECTOR_SIZE],
            tmp_ub,
            m,                           // repeat
            1,                           // dstBlockStride
            1,                           // src0BlockStride
            0,                           // src1BlockStride
            n_stride / FLOAT_BLOCK_SIZE, // dstRepeatStride
            n_stride / FLOAT_BLOCK_SIZE, // src0RepeatStride
            1                            // src1RepeatStride
        );
    }
    if (n_real % FLOAT_VECTOR_SIZE > 0) {
        SetVecMask(n_real % FLOAT_VECTOR_SIZE);
        sub_v<ArchType::ASCEND_V220, float>(
            s_ub[n_real / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            s_ub[n_real / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            tmp_ub,
            m,                           // repeat
            1,                           // dstBlockStride
            1,                           // src0BlockStride
            0,                           // src1BlockStride
            n_stride / FLOAT_BLOCK_SIZE, // dstRepeatStride
            n_stride / FLOAT_BLOCK_SIZE, // src0RepeatStride
            1                            // src1RepeatStride
        );
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);

    // *** ls = exp(ls)
    exp_v<ArchType::ASCEND_V220, float>(
        s_exp_ub,
        s_ub,
        (m * n_stride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, // repeat
        1,                                                          // dstBlockStride
        1,                                                          // srcBlockStride
        8,                                                          // dstRepeatStride
        8                                                           // srcRepeatStride
    );
    PIPE_BARRIER(V);
    // *** ll = rowsum(ls32)
    if (n_real == 512) {
        Rowsum<float, RowCalcTile::SPEC_TILE_512>(
            s_exp_ub,
            local_rowsum_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    } else if (n_real == 256) {
        Rowsum<float, RowCalcTile::SPEC_TILE_256>(
            s_exp_ub,
            local_rowsum_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    } else {
        Rowsum<float, RowCalcTile::TAIL_TILE>(
            s_exp_ub,
            local_rowsum_ub,
            tmp_ub,
            round_m, n_real, n_stride
        );
    }
}
};

__aicore__ __attribute__((always_inline)) inline void OnlineSoftmaxStage1Step2(
    const AscendC::LocalTensor<float> &local_rowsum_ub,
    const AscendC::LocalTensor<float> &global_rowsum_ub,
    const AscendC::LocalTensor<float> &diff_rowmax_ub,
    bool first_n_iter,
    uint32_t m
)
{
    uint32_t round_m = (m + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
    if (first_n_iter) {
        // *** gl = ll
        ub_to_ub<ArchType::ASCEND_V220, float>(
            global_rowsum_ub,
            local_rowsum_ub,
            0,                              // sid
            1,                              // nBurst
            round_m / FLOAT_BLOCK_SIZE, // lenBurst
            0,                              // srcGap
            0                               // dstGap
        );
        PIPE_BARRIER(V);
    } else {
        SetVecMask(m);
        // *** gl = dm * gl
        mul_v<ArchType::ASCEND_V220, float>(
            global_rowsum_ub,
            diff_rowmax_ub,
            global_rowsum_ub,
            1, // repeat
            1,         // dstBlockStride
            1,         // src0BlockStride
            1,         // src1BlockStride
            8,         // dstRepeatStride
            8,         // src0RepeatStride
            8          // src1RepeatStride
        );
        PIPE_BARRIER(V);
        // *** gl = ll + gl
        add_v<ArchType::ASCEND_V220, float>(
            global_rowsum_ub,
            global_rowsum_ub,
            local_rowsum_ub,
            1, // repeat
            1,         // dstBlockStride
            1,         // src0BlockStride
            1,         // src1BlockStride
            8,         // dstRepeatStride
            8,         // src0RepeatStride
            8          // src1RepeatStride
        );
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
}


template<typename P_DTYPE, typename MASK_DTYPE>
struct OnlineSoftmaxStage1<float, float, P_DTYPE, MASK_DTYPE, MaskType::MASK_TYPE_NONE> {
    __aicore__ __attribute__((always_inline)) inline OnlineSoftmaxStage1(
        const AscendC::LocalTensor<float> &s_ub,
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
        bool first_n_iter, float tor,
        uint32_t m, uint32_t n_real, uint32_t n_stride, uint32_t pingpong_flag)
    {
        uint32_t round_m = (m + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
        WAIT_FLAG(MTE3, MTE2, pingpong_flag);
        // input QK
        gm_to_ub<ArchType::ASCEND_V220, float>(
            s_ub,
            s_gm,
            0,                            // sid
            m,                            // nBurst
            n_stride / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                            // srcGap
            0                             // dstGap
        );
        SET_FLAG(MTE2, V, pingpong_flag);
        WAIT_FLAG(MTE2, V, pingpong_flag);

        OnlineSoftmaxStage1Step1<MaskType::MASK_TYPE_NONE>(
            s_ub.template ReinterpretCast<float>(),
            local_rowmax_ub,
            hat_rowmax_ub,
            global_rowmax_ub,
            diff_rowmax_ub,
            s_exp_ub,
            local_rowsum_ub,
            tmp_ub,
            first_n_iter,tor,m,n_real,n_stride
        );

        // *** lp = castfp32to16(ls)
        conv_v<ArchType::ASCEND_V220, float, P_DTYPE>(
            p_ub, s_exp_ub,
            (m * n_stride + FLOAT_VECTOR_SIZE - 1) /
                FLOAT_VECTOR_SIZE, // repeat
            1,                     // dstBlockStride
            1,                     // srcBlockStride
            4,                     // dstRepeatStride
            8                      // srcRepeatStride
        );
        SET_FLAG(V, MTE3, pingpong_flag);
        WAIT_FLAG(V, MTE3, pingpong_flag);
        ub_to_gm<ArchType::ASCEND_V220, P_DTYPE>(
            p_gm,
            p_ub,
            0,                                    // sid
            m,                              // nBurst
            n_stride * 2 / BlockSize<int8_t>(), // lenBurst
            0,                                    // srcGap
            0                                     // dstGap
        );
        SET_FLAG(MTE3, MTE2, pingpong_flag);

        OnlineSoftmaxStage1Step2(
            local_rowsum_ub,
            global_rowsum_ub,
            diff_rowmax_ub,
            first_n_iter,
            m
        );
    }
};

template <typename P_DTYPE, typename MASK_DTYPE>
struct OnlineSoftmaxStage1<float, float, P_DTYPE, MASK_DTYPE, MaskType::MASK_TYPE_CAUSAL_MASK> {
    __aicore__ __attribute__((always_inline)) inline OnlineSoftmaxStage1(
        const AscendC::LocalTensor<float> &s_ub,
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
        bool first_n_iter, float tor, float mask_coff,
        uint32_t m, uint32_t n_real, uint32_t n_stride, uint32_t pp_n_scalar,
        bool last_n_loop, uint32_t slice_row_offset, uint32_t sub_m1, uint32_t pingpong_flag)
    {
        uint32_t round_m = (m + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
        WAIT_FLAG(MTE3, MTE2, pingpong_flag);
        // input QK
        gm_to_ub<ArchType::ASCEND_V220, float>(
            s_ub,
            s_gm,
            0,                            // sid
            m,                            // nBurst
            n_stride / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                            // srcGap
            0                             // dstGap
        );
        SET_FLAG(MTE2, V, pingpong_flag);
        WAIT_FLAG(MTE2, V, pingpong_flag);
        
        OnlineSoftmaxStage1Step1<MaskType::MASK_TYPE_CAUSAL_MASK>(
            s_ub.template ReinterpretCast<float>(),
            mask_processed_ub,
            local_rowmax_ub,
            hat_rowmax_ub,
            global_rowmax_ub,
            diff_rowmax_ub,
            s_exp_ub,
            local_rowsum_ub,
            tmp_ub,
            first_n_iter, tor, mask_coff, m, n_real, n_stride, pp_n_scalar,
            last_n_loop, slice_row_offset, sub_m1
        );

        // *** lp = castfp32to16(ls)
        conv_v<ArchType::ASCEND_V220, float, P_DTYPE>(
            p_ub, s_exp_ub,
            (m * n_stride + FLOAT_VECTOR_SIZE - 1) /
                FLOAT_VECTOR_SIZE, // repeat
            1,                     // dstBlockStride
            1,                     // srcBlockStride
            4,                     // dstRepeatStride
            8                      // srcRepeatStride
        );
        SET_FLAG(V, MTE3, pingpong_flag);
        WAIT_FLAG(V, MTE3, pingpong_flag);
        ub_to_gm<ArchType::ASCEND_V220, P_DTYPE>(
            p_gm,
            p_ub,
            0,                                    // sid
            m,                              // nBurst
            n_stride * 2 / BlockSize<int8_t>(), // lenBurst
            0,                                    // srcGap
            0                                     // dstGap
        );
        SET_FLAG(MTE3, MTE2, pingpong_flag);

        OnlineSoftmaxStage1Step2(
            local_rowsum_ub,
            global_rowsum_ub,
            diff_rowmax_ub,
            first_n_iter,
            m
        );
    }
};