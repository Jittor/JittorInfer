/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file jittor_infer_flash_attention_s1s2_bns1_x910_base.h
 * \brief
 */
#ifndef JITTOR_INFER_FLASH_ATTENTION_S1S2_BNS1_X910_BASE_H
#define JITTOR_INFER_FLASH_ATTENTION_S1S2_BNS1_X910_BASE_H
#include <type_traits>

#include "jittor_infer_flash_attention_base.h"
#include "kernel_data_copy_transpose.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

constexpr uint32_t FP32_ONE_BLOCK_SIZE_PFA = 8;
constexpr uint32_t BYTE_BLOCK_PFA =
    32;  // datacopy的block块大小，datacopy按block块粒度搬移数据
constexpr uint32_t REPEAT_BLOCK_BYTE_PFA = 256;

using namespace matmul;

#define MSD_MUL_TYPE_BY_ROW 1

#define PFA_InitBuffer(para1, para2)    \
    do {                                \
        pipe->InitBuffer(para1, para2); \
    } while (0)
#define PFA_InitQueue(para1, para2, para3)     \
    do {                                       \
        pipe->InitBuffer(para1, para2, para3); \
    } while (0)

enum class PFALayout {
    BSH = 0,
    BNSD,
};

enum class MatMulType {
    MM_MDL = 0,
    MM_NORM,
    MM_IBSHARE_NORM,
    MM_PA,
};

enum class MsdMode {
    MSD_OFF = 0,
    MSD_ON,
};

template <const MatMulType MM_TYPE>
struct GetMatmulConfig {
    static constexpr MatmulConfig mmcfg_value = CFG_MDL;
    static constexpr bool ibshare_value = false;
};

template <>
struct GetMatmulConfig<MatMulType::MM_MDL> {
    static constexpr MatmulConfig mmcfg_value = CFG_MDL;
    static constexpr bool ibshare_value = false;
};

template <>
struct GetMatmulConfig<MatMulType::MM_NORM> {
    static constexpr MatmulConfig mmcfg_value = CFG_NORM;
    static constexpr bool ibshare_value = false;
};

template <>
struct GetMatmulConfig<MatMulType::MM_IBSHARE_NORM> {
    static constexpr MatmulConfig mmcfg_value = CFG_IBSHARE_NORM;
    static constexpr bool ibshare_value = true;
};

template <>
struct GetMatmulConfig<MatMulType::MM_PA> {
    static constexpr MatmulConfig mmcfg_value = GetNormalConfig(
        false, false, false, BatchMode::BATCH_LESS_THAN_L1, false);
    static constexpr bool ibshare_value = false;
};

template <PFALayout L, typename T, typename U, typename O = T,
          typename KV_T = T, Mode M = Mode::HighPerformance,
          const MatMulType MM_TYPE_TMP = MatMulType::MM_MDL,
          const bool F = false, const MsdMode MSD_MODE = MsdMode::MSD_OFF,
          typename... Args>
struct PFAType {
    using inputType = T;
    using maskType = U;
    using outputType = O;
    using kvInputType = KV_T;
    static constexpr PFALayout layout = L;
    static constexpr Mode calcMode = M;
    static constexpr MatMulType MM_TYPE = MM_TYPE_TMP;
    static constexpr MatmulConfig mmCFG = GetMatmulConfig<MM_TYPE>::mmcfg_value;
    static constexpr bool ibShare = GetMatmulConfig<MM_TYPE>::ibshare_value;
    static constexpr MsdMode msdMode = MSD_MODE;
};

constexpr static uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP16 = 0xC77FE000;

constexpr static uint32_t MM2_SINGLE_K_ALIGN_SIZE = 32;
constexpr static uint32_t SINGLE_PROCESS_SINNER_BMMTAIL_LIMIT = 32;
constexpr static uint32_t PFA_BUFFER_SIZE_BYTE_256B = 256;

#define TEMPLATE_LAYOUT template <PFALayout layout = PFAT::layout>
#define TYPENAME_BSH_VOID \
    typename std::enable_if<layout == PFALayout::BSH, void>::type
#define TYPENAME_BNSD_VOID \
    typename std::enable_if<layout == PFALayout::BNSD, void>::type
#define TYPENAME_BSH_INT64 \
    typename std::enable_if<layout == PFALayout::BSH, int64_t>::type
#define TYPENAME_BNSD_INT64 \
    typename std::enable_if<layout == PFALayout::BNSD, int64_t>::type

#define TEMPLATE_MASKTYPE template <typename _maskType>
#define TYPENAME_MASKTYPE_BOOL_VOID \
    typename std::enable_if<std::is_same_v<_maskType, bool>, void>::type
#define TYPENAME_MASKTYPE_INT8_VOID \
    typename std::enable_if<std::is_same_v<_maskType, uint8_t>, void>::type
#define TYPENAME_MASKTYPE_HALF_VOID \
    typename std::enable_if<std::is_same_v<_maskType, half>, void>::type

struct PFAComputeParam {
    bool isFirstInnerIter;
    bool isSecondInnerIter;
    bool isLastInnerIter;
    bool isInnerTail;
    bool useMask;
    bool usePseShift;
    bool kernelInvalidRow;

    uint32_t singleProcessSOuterSize;
    uint32_t singleProcessSInnerSize;
    uint32_t singleProcessSInnerSizeTail;
    uint32_t singleProcessSInnerPrefixSizeTail;
    uint32_t singleProcessSInnerSizeNow;
    uint32_t singleProcessSInnerBmmTail;
    uint32_t padSize;
    uint32_t padPrefixSize;
    uint32_t pseShiftPadSize;
    uint32_t pseShiftPadPrefixSize;
    uint32_t unalignSInner;
    uint32_t unalignSInnerPrefix;
    uint32_t maskCopyInCol;
    uint32_t pseShiftCopyInCol;
    uint32_t maskInnerTailAlign;
    uint32_t maskInnerPrefixTailAlign;
    uint32_t pseShiftInnerTailAlign;
    uint32_t pseShiftInnerPrefixTailAlign;
    uint32_t mm1SingleCoreN;
    uint32_t mm2SingleKAlign;
    int64_t tensorAOffset;
    int64_t tensorBOffset;
    int64_t attenMaskOffset;
    int64_t attenMaskOffsetPre;
    uint64_t pseShiftOffset;
    int64_t valueOffset;
    int64_t attentionOutOffset;
    int64_t sInnerOffsetDataSize;

    int64_t sOuterOffset;
    int64_t batchNOffset;
    uint32_t sInnerLoopOffset;
    int64_t multiSeqOffset;
    int64_t multiSeqOffsetBSNDOut;
    int gmPingpong;
    int64_t SoftMaxOffset;
    int taskBatch;

    bool sparseBandSelect0;  // Sign position，sparse band mode，select 0 to
                             // determine whether to perform this operation
    bool sparseBandSelect1;  // Sign position，sparse band mode，select 1 to
                             // determine whether to perform this operation
    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;
    int64_t actualSeqLengthPerBatch = 0;
    int64_t actualSeqLengthKVPerBatch = 0;
    bool fakeMsg = false;

    // MSD
    uint64_t antiqParamOffsetPerToken = 0ULL;
};
constexpr int32_t PFA_PARAMS_QUEUE_CAPBABILITY = 4;
constexpr uint32_t SPARSE_ATTENTION_MASK_SIZE = 2048;
constexpr event_t NULL_EVENT = static_cast<event_t>(INVALID_TEVENTID);
constexpr uint32_t MAX_SUBSOUTER_NUM =
    64;  // Souter maximum is 512，softmax cut into 8 blocks, not exceeding 64.

template <typename PFAT>
class JittorInferFlashAttentionS1s2Bns1X910Base {
   public:
    __aicore__ inline JittorInferFlashAttentionS1s2Bns1X910Base(){};
    __aicore__ inline void Init(
        __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
        __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
        __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
        __gm__ uint8_t* blocktable, __gm__ uint8_t* attentionOut,
        __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
        const JittorInferFlashAttentionTilingData* __restrict tiling,
        __gm__ uint8_t* gmTiling, TPipe* tPipe);
    __aicore__ inline void Process();

    using FT = float;
    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using U = typename PFAT::maskType;
    using O = typename PFAT::outputType;
    using mmBiasType = typename JittorInferFlashAttentionTypeTraits<
        T, PFAT::calcMode>::mmBiasType;
    using mmOutputTypeTmp = typename JittorInferFlashAttentionTypeTraits<
        T, PFAT::calcMode>::mmOutputType;
    using computeType = typename JittorInferFlashAttentionTypeTraits<
        T, PFAT::calcMode>::softmaxType;
    using pseShiftType = typename JittorInferFlashAttentionTypeTraits<
        T, PFAT::calcMode>::pseShiftType;
    using pseShiftCastType = typename JittorInferFlashAttentionTypeTraits<
        T, PFAT::calcMode>::pseShiftCastType;

    using mmOutputType =
        typename AscendC::Conditional<PFAT::msdMode == MsdMode::MSD_ON, int32_t,
                                      mmOutputTypeTmp>::type;

    template <class SRC_T>
    static __aicore__ void CopyND2NZ(
        const LocalTensor<SRC_T>& dst, const GlobalTensor<SRC_T>& src,
        const int row, const int col, const int height, const int width,
        const int gCol, const int ndNum = 1, const int srcNdMatrixStride = 0,
        const int dstNzMatrixStride = 1, const bool kAlignToC0Size = false,
        const int dstNzC0Stride =
            0) {  // The minimum range of parameter values is 1.
        int64_t srcOffset = (int64_t)row * (int64_t)gCol + (int64_t)col;
        int32_t alignNum = 16;
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = ndNum;
        nd2nzParams.nValue = height;
        nd2nzParams.dValue = width;
        nd2nzParams.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzParams.srcDValue = gCol;
        if (kAlignToC0Size) {
            if constexpr (IsSameType<SRC_T, int8_t>::value) {
                alignNum = 32;
            } else if constexpr (IsSameType<SRC_T, float>::value) {
                alignNum = 8;
            }
        }
        nd2nzParams.dstNzC0Stride = Ceil(dstNzC0Stride, alignNum) * alignNum;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = dstNzMatrixStride;
        DataCopy(dst, src[srcOffset], nd2nzParams);
    }

    static __aicore__ void bmm1CopyB1(const LocalTensor<int8_t>& bMatrix,
                                      const __gm__ void* gm, int row, int col,
                                      int useK, int useN,
                                      const uint64_t tilingPtr,
                                      const uint64_t dataPtr) {
        GlobalTensor<uint32_t> bmm1LocalInfo;
        bmm1LocalInfo.SetGlobalBuffer((__gm__ uint32_t*)dataPtr,
                                      8);  // Align to 8

        uint32_t bmm1BIdx = bmm1LocalInfo.GetValue(0);
        uint32_t bmm1NIdx = bmm1LocalInfo.GetValue(1);
        uint32_t s2SingleOffset = bmm1LocalInfo.GetValue(2);
        uint32_t bmm1TensorBAddr_high = bmm1LocalInfo.GetValue(3);
        uint32_t bmm1TensorBAddr_low = bmm1LocalInfo.GetValue(4);
        uint32_t bmm1BlockTableAddr_high = bmm1LocalInfo.GetValue(5);
        uint32_t bmm1BlockTableAddr_low = bmm1LocalInfo.GetValue(6);
        uint64_t bmm1TensorBAddr =
            (static_cast<uint64_t>(bmm1TensorBAddr_high) << 32U) |
            static_cast<uint64_t>(bmm1TensorBAddr_low);
        uint64_t bmm1BlockTableAddr =
            (static_cast<uint64_t>(bmm1BlockTableAddr_high) << 32U) |
            static_cast<uint64_t>(bmm1BlockTableAddr_low);

        // In other scenarios, the tiling result in the tilingptr configured on
        // the V side is used.
        __gm__ JittorInferFlashAttentionTilingData* tilingDataPtr =
            reinterpret_cast<__gm__ JittorInferFlashAttentionTilingData*>(
                tilingPtr);
        // Directly using fixed TilingData on the stack during static
        // compilation.
        JittorInferFlashAttentionTilingData allTilingData;

        uint32_t blockTableDim2;
        uint32_t blockSize;
        uint32_t isLayoutBSH;  // BSH:1  BNSD:0
        uint32_t headNumSize;
        uint32_t headNumRatio;
        uint32_t kvD;
        uint32_t PABlockNumSum;
        uint32_t baseK;
        uint32_t baseN;
        uint32_t Kb;

        if (tilingDataPtr != nullptr) {
            blockTableDim2 =
                tilingDataPtr->promptAttentionBaseParams.blockTableDim2;
            blockSize = tilingDataPtr->promptAttentionBaseParams.blockSize;
            isLayoutBSH = tilingDataPtr->promptAttentionBaseParams
                              .PAlayoutType;  // BSH:1 BNSD:0
            headNumSize = tilingDataPtr->promptAttentionBaseParams.headNumSize;
            headNumRatio =
                tilingDataPtr->promptAttentionBaseParams.headNumRatio;
            kvD = tilingDataPtr->promptAttentionBaseParams.headSize;
            PABlockNumSum =
                tilingDataPtr->promptAttentionBaseParams.PABlockNumSum;
            baseK = tilingDataPtr->bmm1TilingDataRect.baseK;
            baseN = tilingDataPtr->bmm1TilingDataRect.baseN;
            Kb = tilingDataPtr->bmm1TilingDataRect.Kb;
        } else {
            blockTableDim2 =
                allTilingData.promptAttentionBaseParams.blockTableDim2;
            blockSize = allTilingData.promptAttentionBaseParams.blockSize;
            isLayoutBSH = allTilingData.promptAttentionBaseParams
                              .PAlayoutType;  // BSH:1 BNSD:0
            headNumSize = allTilingData.promptAttentionBaseParams.headNumSize;
            headNumRatio = allTilingData.promptAttentionBaseParams.headNumRatio;
            kvD = allTilingData.promptAttentionBaseParams.headSize;
            PABlockNumSum =
                allTilingData.promptAttentionBaseParams.PABlockNumSum;
            baseK = allTilingData.bmm1TilingDataRect.baseK;
            baseN = allTilingData.bmm1TilingDataRect.baseN;
            Kb = allTilingData.bmm1TilingDataRect.Kb;
        }

        // bmm1 row direction corresponds to the k axis and D axis; col
        // direction corresponds to the N axis and S2 axis.
        uint32_t s2BaseOffset =
            col * baseN;  // The offset of the current useN block in the single
                          // block in the S2 direction.
        uint32_t s2AllOffset =
            s2SingleOffset + s2BaseOffset;  // S2 direction total offset
        uint32_t copyFinishRowCnt = 0;
        uint64_t blockTableIdx = 0;
        uint64_t offsetInBlock = 0;
        int64_t blockRowOffsetInSingle = 0;
        uint32_t blockId = 0;  // Initialize blockId
        uint32_t copyRowCnt = 0;
        uint64_t curOffset = 0;
        uint32_t baseRowOffsetInSingle = 0;
        uint32_t baseColOffsetInSingle = 0;
        uint32_t colElementCnt = 0;
        uint32_t ndNum = 0;
        uint32_t kvHeadNum = headNumSize / headNumRatio;

        colElementCnt = 32 / sizeof(T);
        int32_t alignUseN =
            (useN + colElementCnt - 1) / colElementCnt * colElementCnt;

        while (copyFinishRowCnt <
               useN) {  // 1. useN <= blockSize : copy part of the blockSize 2.
                        // useN > blockSize: Multiple copies of a callback 3.
                        // Tail block
            blockTableIdx =
                s2AllOffset / blockSize;  // Index on the block table,
                                          // s2AllOffset / blockSize
            offsetInBlock =
                s2AllOffset % blockSize;  // Offset on the last block

            blockRowOffsetInSingle = blockTableIdx * blockSize - s2SingleOffset;
            blockId = *(reinterpret_cast<__gm__ int32_t*>(bmm1BlockTableAddr) +
                        bmm1BIdx * blockTableDim2 + blockTableIdx);
            copyRowCnt = blockSize - offsetInBlock;
            if (copyFinishRowCnt + copyRowCnt >
                useN) {  // Copy more than needed
                copyRowCnt = useN - copyFinishRowCnt;
            }

            if (isLayoutBSH == 1) {
                curOffset =
                    blockId * blockSize * kvHeadNum * kvD + bmm1NIdx * kvD;
            } else {
                curOffset = blockId * blockSize * kvHeadNum * kvD +
                            bmm1NIdx * blockSize * kvD;
            }

            GlobalTensor<T> src;  // Pseudo quantization scenarios are also
                                  // dequantized for fp16, and storage to GM.
            src.SetGlobalBuffer((__gm__ T*)bmm1TensorBAddr,
                                PABlockNumSum * blockSize * kvHeadNum * kvD);
            LocalTensor<T> dst = bMatrix.template ReinterpretCast<T>();

            baseRowOffsetInSingle =
                col * baseN;  // The offset of the current base starting point
                              // in single.
            baseColOffsetInSingle = row * baseK;

            if (blockRowOffsetInSingle > baseRowOffsetInSingle) {
                baseRowOffsetInSingle = 0;
            } else {
                baseRowOffsetInSingle -= blockRowOffsetInSingle;
            }

            CopyND2NZ(dst[copyFinishRowCnt * colElementCnt], src[curOffset],
                      baseRowOffsetInSingle, baseColOffsetInSingle, copyRowCnt,
                      useK, Kb, 1, 0, 1, true, useN);

            // Update loop variables
            copyFinishRowCnt += copyRowCnt;
            s2AllOffset += copyRowCnt;
        }
    }

    static __aicore__ void bmm2CopyB1(const LocalTensor<int8_t>& bMatrix,
                                      const __gm__ void* gm, int row, int col,
                                      int useK, int useN,
                                      const uint64_t tilingPtr,
                                      const uint64_t dataPtr) {
        GlobalTensor<uint32_t> bmm2LocalInfo;
        bmm2LocalInfo.SetGlobalBuffer((__gm__ uint32_t*)dataPtr,
                                      8);  // Align to 8

        uint32_t bmm2BIdx = bmm2LocalInfo.GetValue(0);
        uint32_t bmm2NIdx = bmm2LocalInfo.GetValue(1);
        uint32_t s2SingleOffset = bmm2LocalInfo.GetValue(2);
        uint32_t bmm2TensorBAddr_high = bmm2LocalInfo.GetValue(3);
        uint32_t bmm2TensorBAddr_low = bmm2LocalInfo.GetValue(4);
        uint32_t bmm2BlockTableAddr_high = bmm2LocalInfo.GetValue(5);
        uint32_t bmm2BlockTableAddr_low = bmm2LocalInfo.GetValue(6);
        uint64_t bmm2TensorBAddr =
            (static_cast<uint64_t>(bmm2TensorBAddr_high) << 32U) |
            static_cast<uint64_t>(bmm2TensorBAddr_low);
        uint64_t bmm2BlockTableAddr =
            (static_cast<uint64_t>(bmm2BlockTableAddr_high) << 32U) |
            static_cast<uint64_t>(bmm2BlockTableAddr_low);

        // In other scenarios, the tiling result in the tilingptr configured on
        // the V side is used.
        __gm__ JittorInferFlashAttentionTilingData* tilingDataPtr =
            reinterpret_cast<__gm__ JittorInferFlashAttentionTilingData*>(
                tilingPtr);
        JittorInferFlashAttentionTilingData allTilingData;

        uint32_t kvD;
        uint32_t PABlockNumSum;
        uint32_t baseK;
        uint32_t baseN;
        uint32_t N;
        uint32_t blockTableDim2;
        uint32_t blockSize;
        uint32_t isLayoutBSH;  // BSH:1  BNSD:0
        uint32_t headNumSize;
        uint32_t headNumRatio;

        if (tilingDataPtr != nullptr) {
            blockTableDim2 =
                tilingDataPtr->promptAttentionBaseParams.blockTableDim2;
            blockSize = tilingDataPtr->promptAttentionBaseParams.blockSize;
            isLayoutBSH = tilingDataPtr->promptAttentionBaseParams
                              .PAlayoutType;  // BSH:1  BNSD:0
            headNumSize = tilingDataPtr->promptAttentionBaseParams.headNumSize;
            headNumRatio =
                tilingDataPtr->promptAttentionBaseParams.headNumRatio;
            kvD = tilingDataPtr->promptAttentionBaseParams.headSize;
            PABlockNumSum =
                tilingDataPtr->promptAttentionBaseParams.PABlockNumSum;
            baseK = tilingDataPtr->bmm2TilingDataRect.baseK;
            baseN = tilingDataPtr->bmm2TilingDataRect.baseN;
            N = tilingDataPtr->bmm2TilingDataRect.N;

        } else {
            blockTableDim2 =
                allTilingData.promptAttentionBaseParams.blockTableDim2;
            blockSize = allTilingData.promptAttentionBaseParams.blockSize;
            isLayoutBSH = allTilingData.promptAttentionBaseParams
                              .PAlayoutType;  // BSH:1  BNSD:0
            headNumSize = allTilingData.promptAttentionBaseParams.headNumSize;
            headNumRatio = allTilingData.promptAttentionBaseParams.headNumRatio;
            kvD = allTilingData.promptAttentionBaseParams.headSize;
            PABlockNumSum =
                allTilingData.promptAttentionBaseParams.PABlockNumSum;
            baseK = allTilingData.bmm2TilingDataRect.baseK;
            baseN = allTilingData.bmm2TilingDataRect.baseN;
            N = allTilingData.bmm2TilingDataRect.N;
        }
        // bmm2 row direction corresponds to the k axis and S2 axis; col
        // direction corresponds to the N axis and D axis.
        uint32_t s2BaseOffset = row * baseK;
        uint32_t s2AllOffset = s2SingleOffset + s2BaseOffset;
        uint32_t kvHeadNum = headNumSize / headNumRatio;
        uint32_t copyFinishRowCnt = 0;
        uint64_t blockTableIdx = 0;
        uint64_t offsetInBlock = 0;
        int64_t blockRowOffsetInSingle = 0;
        uint32_t blockId = 0;
        uint32_t copyRowCnt = 0;
        uint64_t curOffset = 0;
        uint32_t baseRowOffsetInSingle = 0;
        uint32_t baseColOffsetInSingle = 0;
        uint32_t colElementCnt = 0;
        uint32_t ndNum = 0;

        colElementCnt = 32 / sizeof(T);
        int32_t alignUseK =
            (useK + colElementCnt - 1) / colElementCnt * colElementCnt;

        while (copyFinishRowCnt < useK) {
            blockTableIdx = s2AllOffset / blockSize;
            offsetInBlock = s2AllOffset % blockSize;

            blockRowOffsetInSingle = blockTableIdx * blockSize - s2SingleOffset;
            blockId = *(reinterpret_cast<__gm__ int32_t*>(bmm2BlockTableAddr) +
                        bmm2BIdx * blockTableDim2 + blockTableIdx);

            copyRowCnt = blockSize - offsetInBlock;
            if (copyFinishRowCnt + copyRowCnt > useK) {
                copyRowCnt = useK - copyFinishRowCnt;
            }

            if (isLayoutBSH == 1) {
                curOffset =
                    blockId * blockSize * kvHeadNum * kvD + bmm2NIdx * kvD;
            } else {
                curOffset = blockId * blockSize * kvHeadNum * kvD +
                            bmm2NIdx * blockSize * kvD;
            }

            GlobalTensor<T> src;
            src.SetGlobalBuffer((__gm__ T*)bmm2TensorBAddr,
                                PABlockNumSum * blockSize * kvHeadNum * kvD);

            LocalTensor<T> dst = bMatrix.template ReinterpretCast<T>();

            baseRowOffsetInSingle = row * baseK;
            baseColOffsetInSingle = col * baseN;

            if (blockRowOffsetInSingle > baseRowOffsetInSingle) {
                baseRowOffsetInSingle = 0;
            } else {
                baseRowOffsetInSingle -= blockRowOffsetInSingle;
            }

            CopyND2NZ(dst[copyFinishRowCnt * colElementCnt], src[curOffset],
                      baseRowOffsetInSingle, baseColOffsetInSingle, copyRowCnt,
                      useN, N, 1, 0, 1, true, useK);

            // Update loop variables
            copyFinishRowCnt += copyRowCnt;
            s2AllOffset += copyRowCnt;
        }
    }
    // define matmul
    using MM_IN_T =
        typename AscendC::Conditional<PFAT::msdMode == MsdMode::MSD_ON, KV_T,
                                      T>::type;
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, MM_IN_T, false>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, MM_IN_T, true,
                              LayoutMode::NONE, PFAT::ibShare>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c1Type =
        MatmulType<TPosition::GM, CubeFormat::ND_ALIGN, mmOutputType>;
    using PACBmm1 = typename AscendC::Conditional<
        PFAT::MM_TYPE == MatMulType::MM_PA,
        Matmul<a1Type, b1Type, c1Type, bias1Type, PFAT::mmCFG,
               matmul::MatmulCallBackFunc<nullptr, nullptr, bmm1CopyB1>>,
        Matmul<a1Type, b1Type, c1Type, bias1Type, PFAT::mmCFG>>::
        type;  // PA doesn't need to carry in large packages temporarily.
    PACBmm1 mm;
    // define batchmatmul
    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, MM_IN_T, false>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, MM_IN_T, false,
                              LayoutMode::NONE, PFAT::ibShare>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmOutputType>;
    using PACBmm2 = typename AscendC::Conditional<
        PFAT::MM_TYPE == MatMulType::MM_PA,
        Matmul<a2Type, b2Type, c2Type, bias2Type, PFAT::mmCFG,
               matmul::MatmulCallBackFunc<nullptr, nullptr, bmm2CopyB1>>,
        Matmul<a2Type, b2Type, c2Type, bias2Type, PFAT::mmCFG>>::
        type;  // PA doesn't need to carry in large packages temporarily.
    PACBmm2 bmm2;

   protected:
    const JittorInferFlashAttentionTilingData* __restrict tilingData;
    TPipe* pipe;
    // define the que
    TQue<QuePosition::VECIN, 1> tempBmm2Queue;
    TQue<QuePosition::VECOUT, 1> Bmm1Queue;
    TQue<QuePosition::VECOUT, 1> softmaxOutQueue;
    TQue<QuePosition::VECIN, 1> msdInQueue;
    TQue<QuePosition::VECOUT, 1> msdOutQueue;

    TBuf<> PABmm1UB;
    TBuf<> PABmm2UB;
    TBuf<> selectSpaceUb;
    TBuf<> pseShiftCastUb;
    TBuf<> softmaxExpUb_;
    TBuf<> tempBmm2Ub;

    // ub for msd
    TBuf<> msdAMaxTmpBuff;
    TBuf<> msdAMaxResBuff[2];
    TBuf<> msdQRowSumBuff[2];

    TBuf<> msdSoftmaxResAmaxBuff[2];
    TBuf<> msdSoftmaxRowSumScaleBuff[2];

    TBuf<> msdScaleBuff;
    TBuf<> msdOffsetBuff;
    TBuf<> msdTmpMm1Buff;
    TBuf<> msdTmpMm2Buff;

    // ub for lse
    TBuf<> lseReduceGMask;

    event_t pseShiftEvent = NULL_EVENT;
    event_t bmm1ResCopyInEvent[2];
    event_t bmm2ResCopyInEvent[2];
    event_t bmm1ResCopyOutEvent[2];
    event_t attenOutCopyOut;
    DataCopyParams mm1GmUbCopyParam[2];

    bool copyOutPrevIter = false;
    uint32_t softmaxSouterStepLen = 0;
    bool needAdd;

    LocalTensor<uint32_t> bmm1LocalInfo;
    LocalTensor<uint32_t> bmm2LocalInfo;
    LocalTensor<computeType> mmResUb[2];
    LocalTensor<float> softmaxMaxUb;
    LocalTensor<float> softmaxSumUb;
    LocalTensor<computeType> softmaxExpUb;
    LocalTensor<U> attenMaskUb;
    LocalTensor<pseShiftType> pseShiftUb;

    __gm__ uint8_t* key_ptr;
    __gm__ uint8_t* value_ptr;
    __gm__ uint8_t* currentKey;
    __gm__ uint8_t* currentValue;
    __gm__ uint8_t* blocktable_ptr;

    __gm__ uint32_t* bmm1CBDataPtr[2];
    __gm__ uint32_t* bmm2CBDataPtr[2];

    GlobalTensor<T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<U> attenMaskGm;
    GlobalTensor<O> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    GlobalTensor<pseShiftType> pseShiftGm;
    GlobalTensor<mmOutputType> workspaceGm;

    GlobalTensor<mmOutputType> bmm1ResGmDb[2];
    GlobalTensor<int8_t> quant1ResGmDb[2];
    GlobalTensor<mmOutputType> bmm2ResGmDb[2];

    GlobalTensor<uint32_t> bmm1CBDataGm[2];
    GlobalTensor<uint32_t> bmm2CBDataGm[2];

    int64_t actualKVPrefixLen = 0;

    GlobalTensor<KV_T> queryMsdExpandGm;
    GlobalTensor<KV_T> bmm1ExpandGm[2];  // using same address as bmm1ResGmDb

    // quant: define quant variable
    uint64_t dequantScale1 = 0;
    float quantScale1 = 0;
    uint64_t dequantScale2 = 0;
    float quantScale2 = 0;
    float quantOffset2 = 0;

    GlobalTensor<uint32_t> deqScale1Fp32Gm;
    GlobalTensor<uint32_t> deqScale2Fp32Gm;

    // quant bf16 per-channel
    bool isQuant2PerChn = false;
    bool isQuantOffset2Exist = false;
    uint32_t perChannelQuantUBSize = 0;
    float quant2ScaleValue = 0;
    float quant2OffsetValue = 0;
    GlobalTensor<bfloat16_t> quantScale2BF16Gm;
    GlobalTensor<bfloat16_t> quantOffset2BF16Gm;

    GlobalTensor<half> quantScale2FP16Gm;
    GlobalTensor<half> quantOffset2FP16Gm;

    GlobalTensor<float> quantScale2FP32Gm;
    GlobalTensor<float> quantOffset2FP32Gm;

    TBuf<> quantScale2Size16Ub;
    TBuf<> quantOffset2Size16Ub;
    TBuf<> quantScale2FloatUb;
    TBuf<> quantOffset2FloatUb;

    // kv antiquant
    bool isAntiquantSymmetric = false;
    TQue<QuePosition::VECIN, 1> kvAntiquantSrcQueue;
    TQue<QuePosition::VECOUT, 1> kvAntiquantDstQueue;
    GlobalTensor<T> keyGmAntiquant;
    GlobalTensor<T> valueGmAntiquant;

    PFAComputeParam pfaParamsQueue[PFA_PARAMS_QUEUE_CAPBABILITY];
    PFAComputeParam* tailParams;
    PFAComputeParam* headParams;
    PFAComputeParam* preHeadParams;
    int32_t headId = 0;
    int32_t tailId = 0;
    int32_t queSize = 0;
    int32_t queSizeLimit = PFA_PARAMS_QUEUE_CAPBABILITY - 2;

    int64_t tmp_block_idx = 0;
    int64_t maskOffset = 0;
    int64_t maskCoreOffset = 0;
    uint64_t attenMaskCoreOffset = 0;
    int64_t valueCoreOffset = 0;
    int64_t valuePrefixCoreOffset = 0;
    int64_t tensorACoreOffset = 0;
    int64_t tensorBCoreOffset = 0;
    int64_t tensorBPrefixCoreOffset = 0;
    int64_t offsetSS = 0;
    int64_t offsetSH = 0;
    int64_t offsetSTypeNum = 0;
    int64_t offsetNSTypeNum = 0;
    int64_t offsetNSS = 0;
    int64_t offsetNSH = 0;
    uint32_t maskDataType = 0;
    uint32_t attenMaskBatch = 0;
    uint32_t s2InCurrentBatch = 0;
    AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;

    uint32_t mm1SingleCoreNPrev = 0;
    uint32_t mm2MStridePrev = 0;
    uint32_t mm2KaStridePrev = 0;
    uint64_t pseShiftCoreOffset = 0;
    uint32_t pseShiftBatch = 0;

    // tilingdata
    uint32_t singleProcessSOuterSizeWhole = 0;
    uint32_t singleProcessSOuterSizeTail = 0;
    uint32_t mmResUbSize = 0;
    uint32_t attenMaskUbSize = 0;
    uint32_t maskSize = 0;
    uint32_t pseShiftUbSize = 0;
    uint32_t pseShiftTypeByteNum = 0;
    uint32_t pseShiftStride = 0;
    uint32_t softmaxMaxSize = 0;
    uint32_t softmaxSumSize = 0;
    uint32_t softmaxExpSize = 0;
    uint32_t spmTmpSize = 0;
    uint32_t scmTmpSize = 0;
    uint32_t bmm2ResUbSize = 0;
    uint32_t tmpMMResBmm2PreUbSize = 0;
    uint32_t tmpSoftmaxBmm2UbSize = 0;
    uint32_t typeByteNum = 0;
    uint32_t outputTypeByteNum = 0;
    uint32_t softmaxTypeByteNum = 0;
    uint32_t headNumRatio = 0;
    uint32_t maskTypeByteNum = 0;
    uint32_t selectSpaceUbSize = 0;
    uint32_t maskBmm2ShareSize = 0;

    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxFlashTilingData;
    CopyTransposeTiling transposeTilingData;
    uint32_t MultiHeadQ = 0;
    uint32_t MultiHeadKV = 0;
    uint32_t maxInnerLoopPrefixTimes = 0;
    uint32_t maxInnerLoopTimes = 0;
    int64_t seqListOffset = 0;

    uint32_t attentionMaskStride = 0;
    int32_t attentionMaskType = 0;
    uint32_t negativeScalar = NEGATIVE_MIN_VAULE_FP32;
    bool isSoftmaxResNeedUpdate;
    bool isSoftmaxLseNeedUpdate = false;
    bool isSoftmaxNeedUpdate[MAX_SUBSOUTER_NUM];

    bool isGlobalFirstCompute;

    bool isActualLenDimsNull;
    bool isActualLenDimsKVNull;
    uint64_t actualSeqOffsets[BATCH_NUM_MAX];

    LocalTensor<FT> msdMaxBmm1Ub[2];
    LocalTensor<FT> msdRowSumUb[2];

    LocalTensor<FT> msdMaxBmm2Ub[2];
    LocalTensor<FT> msdSoftmaxScaleResRowSumUb[2];

    static constexpr uint32_t MSD_BLOCK_ELEMENT_NUM =
        BYTE_BLOCK_PFA / sizeof(FT);
    static constexpr uint32_t MSD_REPEAT_ELEMENT_NUM =
        REPEAT_BLOCK_BYTE_PFA / sizeof(FT);
    static constexpr FT msdAntiqExpandCoeff = 254.98;
    static constexpr FT msdAntiqCoeff1 = 127.49;
    static constexpr FT msdAntiqCoeff2 = 1 / msdAntiqCoeff1;
    uint32_t msdIterNum = 1U;
    bool msdIsKOffsetExist = false;
    bool msdIsVOffsetExist = false;
    uint64_t antiqParamBatchOffsetPerToken = 0ULL;

    __aicore__ inline void SoftmaxBasicComputeFirstNoTail(
        LocalTensor<computeType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
        LocalTensor<float>& softmaxSumUb, uint32_t souterSize);

    __aicore__ inline void SoftmaxBasicComputeNoTail(
        LocalTensor<computeType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
        LocalTensor<float>& softmaxSumUb,
        LocalTensor<computeType>& softmaxExpUb, uint32_t souterSize);

    __aicore__ inline void SoftmaxComputeFirstTail(
        LocalTensor<computeType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
        LocalTensor<float>& softmaxSumUb, uint32_t souterSize);

    __aicore__ inline void SoftmaxComputeTail(
        LocalTensor<computeType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
        LocalTensor<float>& softmaxSumUb,
        LocalTensor<computeType>& softmaxExpUb, uint32_t souterSize);

    __aicore__ inline void Bmm2UpdateDivNoTail(
        LocalTensor<computeType>& bmm2ResPreUb,
        LocalTensor<float>& softmaxSumUb);

    __aicore__ inline void UpdateVmul(LocalTensor<computeType>& softmaxExpUb);

    __aicore__ inline void Bmm2UpdateAdd(LocalTensor<computeType>& bmm2ResUb);

    __aicore__ inline void QuantCompute(LocalTensor<int8_t> quantResUb,
                                        LocalTensor<computeType> mmResUb,
                                        float scale, float offset,
                                        uint32_t computeSize);

    __aicore__ inline void CalPseShiftOffset(int sIdx);

    TEMPLATE_LAYOUT
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float>& softmaxSumUb,
                                             LocalTensor<float>& softmaxMaxUb) {
        uint32_t souterSize = this->headParams->singleProcessSOuterSize;

        LocalTensor<float> lseUb =
            this->softmaxExpUb_.template Get<float>(this->softmaxMaxSize);
        Log(lseUb, softmaxSumUb,
            souterSize * 8);  // 8 : second dimension of softmax
        pipe_barrier(PIPE_V);
        Add(lseUb, lseUb, softmaxMaxUb,
            souterSize * 8);  // 8 : second dimension of softmax
        pipe_barrier(PIPE_V);

        if (this->isSoftmaxLseNeedUpdate) {
            SoftMaxShapeInfo softmaxShapeInfo = {
                static_cast<uint32_t>(souterSize), static_cast<uint32_t>(8),
                static_cast<uint32_t>(souterSize), static_cast<uint32_t>(8)};
            AdjustSoftMaxRes<float, float>(lseUb, softmaxMaxUb,
                                           this->negativeScalar, 3e+99,
                                           softmaxShapeInfo);
            pipe_barrier(PIPE_V);
        }

        if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
            if (this->msdIsKOffsetExist) {
                LocalTensor<float>& qRowSumUb =
                    this->msdRowSumUb[this->headParams->gmPingpong];
                Muls(
                    qRowSumUb, qRowSumUb,
                    static_cast<float>(
                        this->tilingData->promptAttentionBaseParams.scaleValue),
                    souterSize * FP32_ONE_BLOCK_SIZE_PFA);
                pipe_barrier(PIPE_V);
                Add(lseUb, lseUb, qRowSumUb,
                    souterSize * FP32_ONE_BLOCK_SIZE_PFA);
                pipe_barrier(PIPE_V);
            }
        }

        if (likely(souterSize % 8 ==
                   0)) {  // copyout element num is 32B aligned, can use
                          // DataCopy to acquire high performance
            GatherMaskParams gatherMaskParams{
                1, 0, 8, 0};  // src0BlockStride = 1, repeatTimes = 0,
                              // src0RepeatStride = 8, src1RepeatStride = 0
            gatherMaskParams.repeatTimes =
                souterSize * 8 / 64;  // single repeat process 256B data
            uint64_t rsvdCnt = 0;
            LocalTensor<uint32_t> tmpLseGMask =
                this->lseReduceGMask.template Get<uint32_t>(64);
            GatherMask(lseUb, lseUb, tmpLseGMask, false, 0, gatherMaskParams,
                       rsvdCnt);

            event_t enQueEvtID = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);

            DataCopy(softmaxLseGm[this->headParams->SoftMaxOffset], lseUb,
                     souterSize);
        } else {  // copyout element num is not 32B aligned, must use
                  // DataCopyPad to ensure precision
            DataCopyParams dataCopyParams{
                static_cast<uint16_t>(souterSize), 4, 0,
                0};  // blockCount = souterSize, blockLen = 4, srcStride = 0,
                     // dstStride = 0

            event_t enQueEvtID = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);

            DataCopyPad(softmaxLseGm[this->headParams->SoftMaxOffset], lseUb,
                        dataCopyParams);  // 8 : second dimension of softmax
        }
        event_t enQueEvtID0 = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(enQueEvtID0);
        WaitFlag<HardEvent::MTE3_V>(enQueEvtID0);
    }

    __aicore__ inline void PostQuant2PerChannelBF16(
        LocalTensor<computeType>& bmm2ResUb,
        LocalTensor<int8_t>& outputQuantRes);

    __aicore__ inline void PostQuant2PerChannelFP16(
        LocalTensor<computeType>& bmm2ResUb,
        LocalTensor<int8_t>& outputQuantRes);

    __aicore__ inline void PostQuant2PerChannelFP32(
        LocalTensor<computeType>& bmm2ResUb,
        LocalTensor<int8_t>& outputQuantRes);

    TEMPLATE_LAYOUT
    __aicore__ inline void DataCopyTransposeOutBSH(
        LocalTensor<computeType>& bmm2ResUb) {
        TransposeParams transposeParams;
        transposeParams.bIndex = 0;
        transposeParams.nIndex = this->preHeadParams->batchNOffset;
        transposeParams.sIndex = this->preHeadParams->sOuterOffset;
        transposeParams.hNIndex = 0;
        int64_t preTokensOffset = 0;
        int64_t nextTokensOffset = 0;
        if (this->preHeadParams->preTokensPerBatch < 0) {
            int64_t preTokenLength =
                this->preHeadParams->actualSeqLengthKVPerBatch +
                actualKVPrefixLen + this->preHeadParams->preTokensPerBatch;
            if (this->preHeadParams->sOuterOffset < preTokenLength &&
                (this->preHeadParams->sOuterOffset +
                 this->preHeadParams->singleProcessSOuterSize) >
                    preTokenLength) {
                preTokensOffset = this->preHeadParams->sOuterOffset +
                                  this->preHeadParams->singleProcessSOuterSize -
                                  preTokenLength;
            } else {
                preTokensOffset = 0;
            }
        }

        if (this->preHeadParams->sOuterOffset <
                this->preHeadParams->nextTokensPerBatch * (-1) &&
            (this->preHeadParams->sOuterOffset +
             this->preHeadParams->singleProcessSOuterSize) >
                this->preHeadParams->nextTokensPerBatch * (-1)) {
            nextTokensOffset = this->preHeadParams->nextTokensPerBatch * (-1) -
                               this->preHeadParams->sOuterOffset;
        } else {
            nextTokensOffset = 0;
        }

        CopyTransposeTiling transposeTilingData22 =
            tilingData->transposeTilingDataRect;
        transposeTilingData22.srcShapeS =
            this->preHeadParams->singleProcessSOuterSize - preTokensOffset -
            nextTokensOffset;
        transposeTilingData22.invalidParamCopyTransposeTiling = 0;
        transposeParams.sIndex = transposeParams.sIndex + nextTokensOffset;

        int64_t multiSeqOffset = this->preHeadParams->multiSeqOffset;
        if constexpr (IsSameType<O, int8_t>::value) {
            LocalTensor<int8_t> outputQuantRes;
            outputQuantRes = bmm2ResUb.template ReinterpretCast<int8_t>();
            // Set the output size
            outputQuantRes.SetSize(bmm2ResUb.GetSize());
            if (isQuant2PerChn) {
                PostQuant2PerChannelFP32(bmm2ResUb, outputQuantRes);
            } else {  // Perform per-channel quantiation, high performance
                      // requires conversion to fp16, high precision or bf16
                      // directly quantizes.
                QuantCompute(outputQuantRes, bmm2ResUb, quantScale2,
                             quantOffset2, bmm2ResUbSize);
            }
            event_t enQueEvtID = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
            DataCopyTranspose2<O>(
                attentionOutGm,
                outputQuantRes[nextTokensOffset *
                               tilingData->promptAttentionBaseParams.headSize],
                CopyTransposeType::TRANSPOSE_ND_UB_GM, transposeParams,
                transposeTilingData22, multiSeqOffset);
        } else if constexpr (PFAT::calcMode == Mode::HighPrecision ||
                             IsSameType<T, bfloat16_t>::value) {
            LocalTensor<T> FinalResUb = bmm2ResUb.template ReinterpretCast<T>();

            pipe_barrier(PIPE_V);
            Cast(FinalResUb, bmm2ResUb, RoundMode::CAST_ROUND,
                 bmm2ResUb.GetSize());

            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopyTranspose2<O>(
                attentionOutGm,
                FinalResUb[nextTokensOffset *
                           tilingData->promptAttentionBaseParams.headSize],
                CopyTransposeType::TRANSPOSE_ND_UB_GM, transposeParams,
                transposeTilingData22, multiSeqOffset);
        } else {
            // Before copyOut, synchronous calculation.
            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopyTranspose2<O>(
                attentionOutGm,
                bmm2ResUb[nextTokensOffset *
                          tilingData->promptAttentionBaseParams.headSize],
                CopyTransposeType::TRANSPOSE_ND_UB_GM, transposeParams,
                transposeTilingData22, multiSeqOffset);
        }
    }

    TEMPLATE_LAYOUT
    __aicore__ inline void DataCopyTransposeOutBNSD(
        LocalTensor<computeType>& bmm2ResUb) {
        int64_t preTokensOffset = 0;
        int64_t nextTokensOffset = 0;
        uint64_t copySize = this->preHeadParams->singleProcessSOuterSize *
                            tilingData->promptAttentionBaseParams.headSize;
        if (this->preHeadParams->preTokensPerBatch < 0) {
            int64_t preTokenLength =
                this->preHeadParams->actualSeqLengthKVPerBatch +
                actualKVPrefixLen + this->preHeadParams->preTokensPerBatch;
            if (this->preHeadParams->sOuterOffset < preTokenLength &&
                (this->preHeadParams->sOuterOffset +
                 this->preHeadParams->singleProcessSOuterSize) >
                    preTokenLength) {
                preTokensOffset = this->preHeadParams->sOuterOffset +
                                  this->preHeadParams->singleProcessSOuterSize -
                                  preTokenLength;
                copySize = copySize -
                           preTokensOffset *
                               tilingData->promptAttentionBaseParams.headSize;
            } else {
                preTokensOffset = 0;
            }
        }

        if (this->preHeadParams->sOuterOffset <
                this->preHeadParams->nextTokensPerBatch * (-1) &&
            (this->preHeadParams->sOuterOffset +
             this->preHeadParams->singleProcessSOuterSize) >
                this->preHeadParams->nextTokensPerBatch * (-1)) {
            nextTokensOffset = this->preHeadParams->nextTokensPerBatch * (-1) -
                               this->preHeadParams->sOuterOffset;
            copySize =
                copySize - nextTokensOffset *
                               tilingData->promptAttentionBaseParams.headSize;
        } else {
            nextTokensOffset = 0;
        }
        int64_t attentionOutTokenOffset =
            nextTokensOffset * tilingData->promptAttentionBaseParams.headSize;

        struct DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copySize / outputTypeByteNum;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        if constexpr (IsSameType<O, int8_t>::value) {
            LocalTensor<int8_t> outputQuantRes;
            outputQuantRes = bmm2ResUb.template ReinterpretCast<int8_t>();
            outputQuantRes.SetSize(bmm2ResUb.GetSize());
            if (isQuant2PerChn) {
                PostQuant2PerChannelFP32(bmm2ResUb, outputQuantRes);
            } else {
                QuantCompute(outputQuantRes, bmm2ResUb, quantScale2,
                             quantOffset2, bmm2ResUbSize);
            }
            event_t enQueEvtID = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
            DataCopy(attentionOutGm[this->preHeadParams->attentionOutOffset +
                                    attentionOutTokenOffset],
                     outputQuantRes[attentionOutTokenOffset], dataCopyParams);
        } else if constexpr (PFAT::calcMode == Mode::HighPrecision ||
                             IsSameType<T, bfloat16_t>::value) {
            LocalTensor<T> FinalResUb = bmm2ResUb.template ReinterpretCast<T>();

            pipe_barrier(PIPE_V);
            Cast(FinalResUb, bmm2ResUb, RoundMode::CAST_ROUND,
                 bmm2ResUb.GetSize());

            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopy(attentionOutGm[this->preHeadParams->attentionOutOffset +
                                    attentionOutTokenOffset],
                     FinalResUb[attentionOutTokenOffset], dataCopyParams);
        } else {
            // Before copyOut, synchronous calculation.
            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopy(attentionOutGm[this->preHeadParams->attentionOutOffset +
                                    attentionOutTokenOffset],
                     bmm2ResUb[attentionOutTokenOffset], dataCopyParams);
        }
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BSH_VOID ComputeOffset(
        PFAComputeParam* params, uint32_t sInnerLoopIdx,
        int32_t firstInnerMargin) {
        int64_t sInnerOffsetDataSize = 0;
        int64_t computeOffset = 0;
        sInnerOffsetDataSize =
            (int64_t)sInnerLoopIdx * this->tailParams->singleProcessSInnerSize +
            firstInnerMargin;
        computeOffset = sInnerOffsetDataSize;
        this->tailParams->tensorBOffset =
            tensorBCoreOffset +
            (int64_t)sInnerOffsetDataSize * (int64_t)MultiHeadKV;
        this->tailParams->valueOffset =
            valueCoreOffset +
            (int64_t)sInnerOffsetDataSize * (int64_t)MultiHeadKV;
        ComputePseShiftOffset(computeOffset);
        ComputeAttenMaskOffset(computeOffset);
        ComputeAttenMaskOffsetPre(computeOffset);

        this->tailParams->sInnerOffsetDataSize = sInnerOffsetDataSize;
        this->tailParams->tensorAOffset = tensorACoreOffset;

        // msd K/V's Scaler/Offset offset
        this->tailParams->antiqParamOffsetPerToken =
            this->antiqParamBatchOffsetPerToken + computeOffset;
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BNSD_VOID ComputeOffset(
        PFAComputeParam* params, uint32_t sInnerLoopIdx,
        int32_t firstInnerMargin) {
        int64_t sInnerOffsetDataSize = 0;
        int64_t computeOffset = 0;
        sInnerOffsetDataSize =
            (int64_t)sInnerLoopIdx * this->tailParams->singleProcessSInnerSize +
            firstInnerMargin;
        computeOffset = sInnerOffsetDataSize;
        this->tailParams->valueOffset =
            valueCoreOffset +
            (int64_t)sInnerOffsetDataSize *
                (int64_t)this->tilingData->promptAttentionBaseParams.headSize;
        ComputePseShiftOffset(computeOffset);
        ComputeAttenMaskOffset(computeOffset);
        ComputeAttenMaskOffsetPre(computeOffset);

        this->tailParams->sInnerOffsetDataSize = sInnerOffsetDataSize;

        // msd k/v offset
        this->tailParams->antiqParamOffsetPerToken =
            this->antiqParamBatchOffsetPerToken + computeOffset;
    }

    __aicore__ inline int64_t CalMultiSeqOffset(int sIdx) {
        int64_t multiSeqOffset = 0;
        if constexpr (PFAT::layout == PFALayout::BNSD) {
            multiSeqOffset =
                (int64_t)sIdx *
                this->tilingData->promptAttentionBaseParams.seqSize *
                (int64_t)this->MultiHeadQ;  // BNSD
        } else {
            multiSeqOffset =
                (int64_t)sIdx *
                this->tilingData->promptAttentionBaseParams.seqSize *
                (int64_t)this->MultiHeadQ;  // BSH
        }
        return multiSeqOffset;
    }

    __aicore__ inline int64_t CalMultiSeqLseOffset(int sIdx,
                                                   PFAComputeParam* params) {
        int64_t lseOffset = 0;
        // B offset
        lseOffset = (int64_t)sIdx *
                        this->tilingData->promptAttentionBaseParams.seqSize *
                        (int64_t)this->tilingData->promptAttentionBaseParams
                            .headNumSize +
                    (int64_t)params->batchNOffset *
                        tilingData->promptAttentionBaseParams.seqSize +
                    (int64_t)params->sOuterOffset;
        return lseOffset;
    }

    __aicore__ inline void CalPrefixCoreOffset(PFAComputeParam* params) {
        this->tensorBPrefixCoreOffset = 0;
        this->valuePrefixCoreOffset = 0;
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BSH_VOID LoopSOuterOffsetInit(
        int64_t seqListOffsetSize, int sIdx) {
        CalPseShiftOffset(sIdx);

        uint64_t attenMaskBatchOffset = 0;
        if (attenMaskBatch != 1) {
            attenMaskBatchOffset =
                (uint64_t)sIdx *
                (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize *
                (uint64_t)tilingData->promptAttentionBaseParams.maskQsSize;
        }
        attenMaskCoreOffset =
            attenMaskBatchOffset +
            (uint64_t)(this->tailParams->sOuterOffset) *
                (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize;

        tensorACoreOffset =
            (int64_t)seqListOffsetSize +
            (int64_t)this->tailParams->sOuterOffset * (int64_t)MultiHeadQ +
            (int64_t)this->tailParams->batchNOffset *
                (int64_t)tilingData->promptAttentionBaseParams.headSize;
        int64_t seqInnerOffsetSize;
        // This is the offset required from the GM of KV to the starting address
        // of each batch. Each batch needs to be offset by the length of the
        // entire previous batch.
        seqInnerOffsetSize =
            tilingData->promptAttentionBaseParams.seqSize ==
                    tilingData->promptAttentionBaseParams.seqInnerSize
                ? (seqListOffsetSize) / headNumRatio
                : (int64_t)sIdx *
                      (int64_t)
                          tilingData->promptAttentionBaseParams.seqInnerSize *
                      (int64_t)MultiHeadKV;
        tensorBCoreOffset = (int64_t)seqInnerOffsetSize +
                            this->tailParams->batchNOffset / headNumRatio *
                                tilingData->promptAttentionBaseParams.headSize;

        valueCoreOffset = tensorBCoreOffset;
        this->tailParams->SoftMaxOffset =
            this->CalMultiSeqLseOffset(sIdx, this->tailParams);

        // MSD
        this->antiqParamBatchOffsetPerToken =
            (int64_t)sIdx *
            (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize;
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BNSD_VOID LoopSOuterOffsetInit(
        int64_t seqListOffsetSize, int sIdx) {
        uint64_t head_stride_q =
            tilingData->promptAttentionBaseParams.headSize *
            tilingData->promptAttentionBaseParams.seqSize;
        uint32_t head_stride_kv;

        head_stride_kv = tilingData->promptAttentionBaseParams.headSize *
                         tilingData->promptAttentionBaseParams.seqInnerSize;
        uint32_t seq_stride = tilingData->promptAttentionBaseParams.headSize;

        CalPseShiftOffset(sIdx);

        uint64_t attenMaskBatchOffset = 0;
        if (attenMaskBatch != 1) {
            attenMaskBatchOffset =
                (uint64_t)sIdx *
                (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize *
                (uint64_t)tilingData->promptAttentionBaseParams.maskQsSize;
        }
        attenMaskCoreOffset =
            attenMaskBatchOffset +
            ((uint64_t)this->tailParams->sOuterOffset) *
                (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize;

        tensorACoreOffset =
            (int64_t)seqListOffsetSize +
            (int64_t)this->tailParams->batchNOffset * (int64_t)head_stride_q +
            (int64_t)this->tailParams->sOuterOffset * (int64_t)seq_stride;
        int64_t seqInnerOffsetSize;
        seqInnerOffsetSize =
            tilingData->promptAttentionBaseParams.seqSize ==
                    tilingData->promptAttentionBaseParams.seqInnerSize
                ? (seqListOffsetSize) / (int64_t)headNumRatio
                : (int64_t)sIdx * (int64_t)head_stride_kv *
                      (int64_t)
                          tilingData->promptAttentionBaseParams.headNumSize /
                      (int64_t)headNumRatio;
        tensorBCoreOffset = (int64_t)seqInnerOffsetSize +
                            (int64_t)this->tailParams->batchNOffset /
                                (int64_t)headNumRatio * (int64_t)head_stride_kv;

        valueCoreOffset = tensorBCoreOffset;

        this->tailParams->attentionOutOffset =
            (int64_t)seqListOffsetSize +
            (int64_t)this->tailParams->batchNOffset * (int64_t)head_stride_q +
            (int64_t)this->tailParams->sOuterOffset * (int64_t)seq_stride;
        this->tailParams->SoftMaxOffset =
            this->CalMultiSeqLseOffset(sIdx, this->tailParams);

        // MSD
        this->antiqParamBatchOffsetPerToken =
            (int64_t)sIdx *
            (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize;
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BSH_INT64 GetBmm1TensorBOffset(
        PFAComputeParam* params, int32_t sInnerLoopIdx,
        int32_t firstInnerMargin) {
        int64_t prefixIdxOffset = 0;
        return this->tensorBCoreOffset +
               (((int64_t)sInnerLoopIdx - prefixIdxOffset) *
                    (int64_t)params->singleProcessSInnerSize +
                (int64_t)firstInnerMargin) *
                   (int64_t)this->MultiHeadKV;
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BNSD_INT64 GetBmm1TensorBOffset(
        PFAComputeParam* params, int32_t sInnerLoopIdx,
        int32_t firstInnerMargin) {
        int64_t prefixIdxOffset = 0;
        return this->tensorBCoreOffset +
               (((int64_t)sInnerLoopIdx - prefixIdxOffset) *
                    (int64_t)params->singleProcessSInnerSize +
                (int64_t)firstInnerMargin) *
                   (int64_t)this->tilingData->promptAttentionBaseParams
                       .headSize;
    }

    TEMPLATE_MASKTYPE
    __aicore__ inline TYPENAME_MASKTYPE_HALF_VOID ElewiseCompute(
        LocalTensor<computeType>& mmResUb, uint32_t sOuterSize,
        uint32_t sInnerSize, uint32_t maskCopyInCol, bool useMask,
        event_t& copyIn, uint32_t type) {
        uint32_t computeSize = sOuterSize * sInnerSize;
        if (useMask) {
            this->attenMaskUb = this->tempBmm2Queue.template DeQue<U>();
            Muls(this->attenMaskUb, this->attenMaskUb,
                 static_cast<computeType>(-10000.0), computeSize);
            pipe_barrier(PIPE_V);
            Add(mmResUb, mmResUb, this->attenMaskUb, computeSize);
            pipe_barrier(PIPE_V);
            tempBmm2Queue.FreeTensor(this->attenMaskUb);
        }
    }

    TEMPLATE_MASKTYPE
    __aicore__ inline TYPENAME_MASKTYPE_BOOL_VOID ElewiseCompute(
        LocalTensor<computeType>& mmResUb, uint32_t sOuterSize,
        uint32_t sInnerSize, uint32_t maskCopyInCol, bool useMask,
        event_t& copyIn, uint32_t type) {
        if (useMask) {
            this->attenMaskUb = this->tempBmm2Queue.template DeQue<
                U>();  // Dequeue the attentio mask tensor from the queue.
            this->attenMaskUb.SetSize(sOuterSize * maskCopyInCol);
            LocalTensor<uint8_t> selectSpace =
                selectSpaceUb.Get<uint8_t>(this->selectSpaceUbSize);
            computeType scalar;
            if constexpr (PFAT::calcMode == Mode::HighPrecision ||
                          IsSameType<T, bfloat16_t>::
                              value) {  // Set the size of the attention mask
                                        // tensor.
                uint32_t tmp = 0xFF7FFFFF;  // minimum value of fp32
                scalar = *((float*)&tmp);
            } else {
                uint32_t tmp = 0xFBFF;  // minimum value of fp16
                scalar = *((half*)&tmp);
            }
            SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
            selectWithBytesMaskShapeInfo.firstAxis = sOuterSize;
            selectWithBytesMaskShapeInfo.srcLastAxis = sInnerSize;
            selectWithBytesMaskShapeInfo.maskLastAxis = maskCopyInCol;
            if (type == 0) {
                SelectWithBytesMask(mmResUb, mmResUb, scalar, this->attenMaskUb,
                                    selectSpace, selectWithBytesMaskShapeInfo);
            } else if (type == 1) {  //  swape param 2 and param 3 of
                                     //  SelectWithBytesMask to compute
                                     //  attenMaskPre for band mode
                SelectWithBytesMask(mmResUb, scalar, mmResUb, this->attenMaskUb,
                                    selectSpace, selectWithBytesMaskShapeInfo);
            }
            pipe_barrier(PIPE_V);
            tempBmm2Queue.FreeTensor(this->attenMaskUb);
        }
    }

    TEMPLATE_MASKTYPE
    __aicore__ inline TYPENAME_MASKTYPE_INT8_VOID ElewiseCompute(
        LocalTensor<T>& mmResUb, uint32_t sOuterSize, uint32_t sInnerSize,
        uint32_t maskCopyInCol, bool useMask, event_t& copyIn, uint32_t type) {
        if (useMask) {
            this->attenMaskUb = this->tempBmm2Queue.template DeQue<U>();
            this->attenMaskUb.SetSize(sOuterSize * maskCopyInCol);
            LocalTensor<uint8_t> selectSpace =
                selectSpaceUb.Get<uint8_t>(this->selectSpaceUbSize);
            computeType scalar;
            if constexpr (PFAT::calcMode == Mode::HighPrecision ||
                          IsSameType<T, bfloat16_t>::value) {
                uint32_t tmp = 0xFF7FFFFF;  // minimum value of fp32
                scalar = *((float*)&tmp);
            } else {
                uint32_t tmp = 0xFBFF;  // minimum value of fp16
                scalar = *((half*)&tmp);
            }
            SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
            selectWithBytesMaskShapeInfo.firstAxis = sOuterSize;
            selectWithBytesMaskShapeInfo.srcLastAxis = sInnerSize;
            selectWithBytesMaskShapeInfo.maskLastAxis = maskCopyInCol;
            if (type == 0) {
                SelectWithBytesMask(mmResUb, mmResUb, scalar, this->attenMaskUb,
                                    selectSpace, selectWithBytesMaskShapeInfo);
            } else if (type == 1) {
                SelectWithBytesMask(
                    mmResUb, scalar, mmResUb, this->attenMaskUb, selectSpace,
                    selectWithBytesMaskShapeInfo);  // swape param 2 and param 3
                                                    // of SelectWithBytesMask to
                                                    // compute attenMaskPre for
                                                    // band mode
            }
            pipe_barrier(PIPE_V);
            tempBmm2Queue.FreeTensor(this->attenMaskUb);
        }
    }

    __aicore__ inline void ComputePseShiftOffset(int sInnerOffsetDataSize) {
        if (!(this->tailParams->usePseShift)) {
            return;
        }

        this->tailParams->pseShiftOffset =
            pseShiftCoreOffset + (uint64_t)sInnerOffsetDataSize;
    }

    __aicore__ inline void ComputeAttenMaskOffset(
        int64_t sInnerOffsetDataSize) {
        int64_t delta;
        if (attentionMaskType == 2 || attentionMaskType == 3 ||
            attentionMaskType ==
                4) {  // 2:leftUp mode of sparseMode, 3:rightdown mode of
                      // sparseMode, 4:band mode of sparseMode
            if (attentionMaskType == 2) {
                delta = this->tailParams->sOuterOffset - sInnerOffsetDataSize +
                        tilingData->promptAttentionBaseParams.nextTokens;
            } else {
                delta = this->tailParams->sOuterOffset - sInnerOffsetDataSize +
                        this->tailParams->nextTokensPerBatch;
            }

            if (delta < 0) {
                this->tailParams->attenMaskOffset =
                    ((int32_t)singleProcessSOuterSizeWhole + delta) > 0
                        ? (-delta)
                        : singleProcessSOuterSizeWhole;
            } else {
                this->tailParams->attenMaskOffset =
                    (((int32_t)this->tailParams->singleProcessSInnerSize -
                      delta) > 0
                         ? delta
                         : this->tailParams->singleProcessSInnerSize) *
                    attentionMaskStride;
            }
        } else {
            this->tailParams->attenMaskOffset =
                attenMaskCoreOffset + (uint64_t)sInnerOffsetDataSize;
        }
    }

    __aicore__ inline void ComputeAttenMaskOffsetPre(
        int64_t sInnerOffsetDataSize) {
        if (attentionMaskType == 0 || attentionMaskType == 1) {
            return;
        }
        int64_t delta = this->tailParams->sOuterOffset - sInnerOffsetDataSize -
                        this->tailParams->preTokensPerBatch - 1;
        if (delta < 0) {
            this->tailParams->attenMaskOffsetPre =
                ((int32_t)singleProcessSOuterSizeWhole + delta) > 0
                    ? (-delta)
                    : singleProcessSOuterSizeWhole;
        } else {
            this->tailParams->attenMaskOffsetPre =
                (((int32_t)this->tailParams->singleProcessSInnerSize - delta) >
                         0
                     ? delta
                     : this->tailParams->singleProcessSInnerSize) *
                attentionMaskStride;
        }
    }

    __aicore__ inline void initOffset();

    __aicore__ inline void InitTensorSize(
        const PromptAttentionSingleCoreTensorSize* tensorSizeTiling);

    __aicore__ inline void GetSingleCoreParam(int sIdx);

    __aicore__ inline void GetSparseParam(int64_t* preTokens,
                                          int64_t* nextTokens, int sIdx,
                                          PFAComputeParam*& params);

    __aicore__ inline void InitOutputSingleCore();
    __aicore__ inline void InitLseOutputSingleCore();
};

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::PostQuant2PerChannelBF16(
    LocalTensor<computeType>& bmm2ResUb, LocalTensor<int8_t>& outputQuantRes) {
    if constexpr (IsSameType<computeType, float>::value) {
        LocalTensor<bfloat16_t> quantScale2Ub =
            quantScale2Size16Ub.Get<bfloat16_t>(perChannelQuantUBSize);
        LocalTensor<bfloat16_t> quantOffset2Ub;
        DataCopy(quantScale2Ub,
                 quantScale2BF16Gm[(uint64_t)this->preHeadParams->batchNOffset *
                                   perChannelQuantUBSize],
                 perChannelQuantUBSize);
        if (isQuantOffset2Exist) {
            quantOffset2Ub =
                quantOffset2Size16Ub.Get<bfloat16_t>(perChannelQuantUBSize);
            DataCopy(
                quantOffset2Ub,
                quantOffset2BF16Gm[(uint64_t)this->preHeadParams->batchNOffset *
                                   perChannelQuantUBSize],
                perChannelQuantUBSize);
        }
        auto quantParamCast = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(quantParamCast);
        WaitFlag<HardEvent::MTE2_V>(quantParamCast);
        LocalTensor<float> quantScale2UbFloatBF16 =
            quantScale2FloatUb.Get<float>(perChannelQuantUBSize);
        LocalTensor<float> quantOffset2UbFloatBF16;
        Cast(quantScale2UbFloatBF16, quantScale2Ub, RoundMode::CAST_NONE,
             quantScale2Ub.GetSize());
        if (isQuantOffset2Exist) {
            quantOffset2UbFloatBF16 =
                quantOffset2FloatUb.Get<float>(perChannelQuantUBSize);
            Cast(quantOffset2UbFloatBF16, quantOffset2Ub, RoundMode::CAST_NONE,
                 quantOffset2Ub.GetSize());
            pipe_barrier(PIPE_V);
            AscendQuant(outputQuantRes, bmm2ResUb, quantScale2UbFloatBF16,
                        quantOffset2UbFloatBF16);
        } else {
            pipe_barrier(PIPE_V);
            AscendQuant(outputQuantRes, bmm2ResUb, quantScale2UbFloatBF16,
                        static_cast<float>(0));
        }
    }
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::PostQuant2PerChannelFP16(
    LocalTensor<computeType>& bmm2ResUb, LocalTensor<int8_t>& outputQuantRes) {
    if constexpr (IsSameType<computeType, float>::value) {
        LocalTensor<half> quantScale2Ub =
            quantScale2Size16Ub.Get<half>(perChannelQuantUBSize);
        LocalTensor<half> quantOffset2Ub;
        DataCopy(quantScale2Ub,
                 quantScale2FP16Gm[(uint64_t)this->preHeadParams->batchNOffset *
                                   perChannelQuantUBSize],
                 perChannelQuantUBSize);
        if (isQuantOffset2Exist) {
            quantOffset2Ub =
                quantOffset2Size16Ub.Get<half>(perChannelQuantUBSize);
            DataCopy(
                quantOffset2Ub,
                quantOffset2FP16Gm[(uint64_t)this->preHeadParams->batchNOffset *
                                   perChannelQuantUBSize],
                perChannelQuantUBSize);
        }

        auto quantParamCast = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(quantParamCast);
        WaitFlag<HardEvent::MTE2_V>(quantParamCast);
        LocalTensor<float> quantScale2UbFloatFP16 =
            quantScale2FloatUb.Get<float>(perChannelQuantUBSize);
        LocalTensor<float> quantOffset2UbFloatFP16;

        Cast(quantScale2UbFloatFP16, quantScale2Ub, RoundMode::CAST_NONE,
             quantScale2Ub.GetSize());
        if (isQuantOffset2Exist) {
            quantOffset2UbFloatFP16 =
                quantOffset2FloatUb.Get<float>(perChannelQuantUBSize);
            Cast(quantOffset2UbFloatFP16, quantOffset2Ub, RoundMode::CAST_NONE,
                 quantOffset2Ub.GetSize());
            pipe_barrier(PIPE_V);
            AscendQuant(outputQuantRes, bmm2ResUb, quantScale2UbFloatFP16,
                        quantOffset2UbFloatFP16);
        } else {
            pipe_barrier(PIPE_V);
            AscendQuant(outputQuantRes, bmm2ResUb, quantScale2UbFloatFP16,
                        static_cast<float>(0));
        }
    }
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::PostQuant2PerChannelFP32(
    LocalTensor<computeType>& bmm2ResUb, LocalTensor<int8_t>& outputQuantRes) {
    LocalTensor<float> quantScale2UbTmp =
        quantScale2FloatUb.Get<float>(perChannelQuantUBSize);
    LocalTensor<computeType> quantScale2Ub =
        quantScale2UbTmp.template ReinterpretCast<computeType>();
    LocalTensor<float> quantOffset2UbTmp;
    LocalTensor<computeType> quantOffset2Ub;
    if (isQuantOffset2Exist) {
        quantOffset2UbTmp =
            quantOffset2FloatUb.Get<float>(perChannelQuantUBSize);
        quantOffset2Ub =
            quantOffset2UbTmp.template ReinterpretCast<computeType>();
    }

    DataCopy(quantScale2UbTmp,
             quantScale2FP32Gm[(uint64_t)this->preHeadParams->batchNOffset *
                               perChannelQuantUBSize],
             perChannelQuantUBSize);
    if (isQuantOffset2Exist) {
        DataCopy(
            quantOffset2UbTmp,
            quantOffset2FP32Gm[(uint64_t)this->preHeadParams->batchNOffset *
                               perChannelQuantUBSize],
            perChannelQuantUBSize);
    }
    auto quantParamCast = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(quantParamCast);
    WaitFlag<HardEvent::MTE2_V>(quantParamCast);

    if constexpr (IsSameType<computeType, half>::value) {
        Cast(quantScale2Ub, quantScale2UbTmp, RoundMode::CAST_ROUND,
             perChannelQuantUBSize);
        if (isQuantOffset2Exist) {
            Cast(quantOffset2Ub, quantOffset2UbTmp, RoundMode::CAST_ROUND,
                 perChannelQuantUBSize);
        }
    }

    pipe_barrier(PIPE_V);
    if (isQuantOffset2Exist) {
        AscendQuant(outputQuantRes, bmm2ResUb, quantScale2Ub, quantOffset2Ub,
                    perChannelQuantUBSize, perChannelQuantUBSize,
                    bmm2ResUb.GetSize());
    } else {
        AscendQuant(outputQuantRes, bmm2ResUb, quantScale2Ub,
                    static_cast<computeType>(0), perChannelQuantUBSize,
                    bmm2ResUb.GetSize());
    }
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::CalPseShiftOffset(int sIdx) {
    if (!(this->tailParams->usePseShift)) {
        return;
    }
    uint64_t pseShiftBatchOffset = 0;
    uint64_t pseShiftN =
        (uint64_t)tilingData->promptAttentionBaseParams.headNumSize;
    uint64_t pseShiftS1 =
        (uint64_t)tilingData->promptAttentionBaseParams.pseShiftS1Size;
    uint64_t pseShiftS2 =
        (uint64_t)tilingData->promptAttentionBaseParams.pseShiftS2Size;

    if (pseShiftBatch != 1) {
        pseShiftBatchOffset =
            (uint64_t)sIdx * pseShiftN * pseShiftS1 * pseShiftS2;
    }

    pseShiftCoreOffset =
        pseShiftBatchOffset +
        (uint64_t)this->tailParams->batchNOffset * pseShiftS1 * pseShiftS2 +
        ((uint64_t)this->tailParams->sOuterOffset) * pseShiftS2;
}

// quant: add quant functions
template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::QuantCompute(
    LocalTensor<int8_t> quantResUb, LocalTensor<computeType> mmResUb,
    float scale, float offset, uint32_t computeSize) {
    pipe_barrier(PIPE_V);
    AscendQuant(quantResUb, mmResUb, scale, offset, computeSize);
}

template <typename PFAT>
__aicore__ inline void JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
    __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
    __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
    __gm__ uint8_t* blocktable, __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
    const JittorInferFlashAttentionTilingData* __restrict tiling,
    __gm__ uint8_t* gmTiling, TPipe* tPipe) {
    tmp_block_idx = GetBlockIdx();

    // init global buffer
    tilingData = tiling;
    key_ptr = key;
    value_ptr = value;

    // For small B*N perform skip core optimization
    if constexpr (PFAT::MM_TYPE != MatMulType::MM_IBSHARE_NORM) {
        if (tilingData->promptAttentionSingleCoreParams.actualCoreNums <=
            (GetBlockNum() * GetTaskRation() / 2 + 1)) {
            if (tmp_block_idx & 0x1) {
                tmp_block_idx =
                    (tmp_block_idx + GetBlockNum() * GetTaskRation()) / 2;
            } else {
                tmp_block_idx = tmp_block_idx / 2;
            }
        }
    }

    queryGm.SetGlobalBuffer((__gm__ T*)query);
    attentionOutGm.SetGlobalBuffer((__gm__ O*)attentionOut);
    attentionOutInitGm.SetGlobalBuffer((__gm__ half*)attentionOut);
    softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);
    workspaceGm.SetGlobalBuffer((__gm__ mmOutputType*)workspace);

    pipe = tPipe;
    typeByteNum = tilingData->promptAttentionBaseParams.typeByteNum;
    outputTypeByteNum = tilingData->promptAttentionBaseParams.outputTypeByteNum;
    softmaxTypeByteNum =
        tilingData->promptAttentionBaseParams.softmaxTypeByteNum;
    headNumRatio = tilingData->promptAttentionBaseParams.headNumRatio;
    maskDataType = tilingData->promptAttentionBaseParams.attenMaskElemType;
    maskTypeByteNum = tilingData->promptAttentionBaseParams.maskTypeByteNum;
    attenMaskBatch = tilingData->promptAttentionSingleCoreParams.attenMaskBatch;
    pseShiftTypeByteNum =
        tilingData->promptAttentionBaseParams.pseShiftTypeByteNum;
    pseShiftBatch = tilingData->promptAttentionSingleCoreParams.pseShiftBatch;

    keyGm.SetGlobalBuffer((__gm__ KV_T*)key);
    valueGm.SetGlobalBuffer((__gm__ KV_T*)value);

    initOffset();

    isActualLenDimsNull = true;
    isActualLenDimsKVNull = true;
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsNull) {
        actualSeqLengthsGm.SetGlobalBuffer(
            (__gm__ int64_t*)actualSeqLengths,
            tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsNull = false;
    }
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsKVNull) {
        actualSeqLengthsKVGm.SetGlobalBuffer(
            (__gm__ int64_t*)actualSeqLengthsKV,
            tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsKVNull = false;
    }

    uint32_t preAccumSOuter = 0;
    uint32_t h = tilingData->promptAttentionBaseParams.headNumSize *
                 tilingData->promptAttentionBaseParams.headSize;
    uint32_t s = tilingData->promptAttentionBaseParams.seqSize;
    if constexpr ((PFAT::calcMode != Mode::HighPrecision) &&
                  (IsSameType<T, half>::value ||
                   IsSameType<T, int8_t>::value)) {
        this->negativeScalar = NEGATIVE_MIN_VAULE_FP16;
    }
    uint64_t maskSize =
        (tilingData->promptAttentionTensorSizeRect.attenMaskUbSize) * sizeof(U);
    maskBmm2ShareSize =
        (tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize) *
        sizeof(computeType);
    if (maskBmm2ShareSize < maskSize) {
        maskBmm2ShareSize = maskSize;
    }

    if ((pseShift != NULL) &&
        (tilingData->promptAttentionBaseParams.usePseShift == 1)) {
        uint32_t pseShiftSize =
            (tilingData->promptAttentionTensorSizeRect.pseShiftUbSize) *
            sizeof(pseShiftType);
        if (maskBmm2ShareSize < pseShiftSize) {
            maskBmm2ShareSize = pseShiftSize;
        }
    }

    PFA_InitQueue(softmaxOutQueue, 1,
                  2 * tilingData->promptAttentionTensorSizeRect.softmaxMaxSize *
                      sizeof(float));
    PFA_InitBuffer(tempBmm2Ub,
                   tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize *
                       sizeof(mmOutputType));

    PFA_InitBuffer(softmaxExpUb_,
                   tilingData->promptAttentionTensorSizeRect.softmaxExpSize *
                       sizeof(computeType));
    PFA_InitQueue(tempBmm2Queue, 1, maskBmm2ShareSize);
    PFA_InitQueue(Bmm1Queue, 2,
                  tilingData->promptAttentionTensorSizeRect.mmResUbSize *
                      sizeof(mmOutputType));

    if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
        PFA_InitQueue(msdInQueue, 1,
                      tilingData->promptAttentionTensorSizeRect.msdInQueueSize);
        PFA_InitBuffer(
            msdAMaxTmpBuff,
            tilingData->promptAttentionTensorSizeRect.msdAMaxTmpBuffSize +
                PFA_BUFFER_SIZE_BYTE_256B);
        PFA_InitBuffer(
            msdAMaxResBuff[0],
            tilingData->promptAttentionTensorSizeRect.msdAMaxResBuffSize +
                PFA_BUFFER_SIZE_BYTE_256B);
        PFA_InitBuffer(
            msdAMaxResBuff[1],
            tilingData->promptAttentionTensorSizeRect.msdAMaxResBuffSize +
                PFA_BUFFER_SIZE_BYTE_256B);
        PFA_InitBuffer(
            msdQRowSumBuff[0],
            tilingData->promptAttentionTensorSizeRect.msdQRowSumBuffSize +
                PFA_BUFFER_SIZE_BYTE_256B);
        PFA_InitBuffer(
            msdQRowSumBuff[1],
            tilingData->promptAttentionTensorSizeRect.msdQRowSumBuffSize +
                PFA_BUFFER_SIZE_BYTE_256B);
        PFA_InitBuffer(msdSoftmaxResAmaxBuff[0],
                       tilingData->promptAttentionTensorSizeRect
                               .msdSoftmaxResAmaxBuffSize +
                           PFA_BUFFER_SIZE_BYTE_256B);
        PFA_InitBuffer(msdSoftmaxResAmaxBuff[1],
                       tilingData->promptAttentionTensorSizeRect
                               .msdSoftmaxResAmaxBuffSize +
                           PFA_BUFFER_SIZE_BYTE_256B);
        PFA_InitBuffer(msdSoftmaxRowSumScaleBuff[0],
                       tilingData->promptAttentionTensorSizeRect
                               .msdSoftmaxRowSumScaleBuffSize +
                           PFA_BUFFER_SIZE_BYTE_256B);
        PFA_InitBuffer(msdSoftmaxRowSumScaleBuff[1],
                       tilingData->promptAttentionTensorSizeRect
                               .msdSoftmaxRowSumScaleBuffSize +
                           PFA_BUFFER_SIZE_BYTE_256B);
        PFA_InitBuffer(
            msdScaleBuff,
            tilingData->promptAttentionTensorSizeRect.msdScaleBuffSize);
        PFA_InitBuffer(
            msdOffsetBuff,
            tilingData->promptAttentionTensorSizeRect.msdOffsetBuffSize);
        PFA_InitBuffer(
            msdTmpMm1Buff,
            tilingData->promptAttentionTensorSizeRect.msdTmpMm1BuffSize);
        PFA_InitBuffer(
            msdTmpMm2Buff,
            tilingData->promptAttentionTensorSizeRect.msdTmpMm2BuffSize);
        PFA_InitQueue(
            msdOutQueue, 1,
            tilingData->promptAttentionTensorSizeRect
                .msdOutQueueSize);  // 8k， msd模式下，新增ub,大小为：max_qs * d
                                    // * sizeof(bf16)
    }

    if (tilingData->promptAttentionBaseParams.isSoftMaxLseEnable) {
        PFA_InitBuffer(lseReduceGMask, 256);
        LocalTensor<uint32_t> tmpLseGMask =
            lseReduceGMask.template Get<uint32_t>(64);
        tmpLseGMask.SetValue(
            0, 2155905152U);  // binary num is 10000000 10000000 10000000
                              // 10000000, used for GatherMask
        tmpLseGMask.SetValue(1, 2155905152U);
    }

    if (tilingData->promptAttentionTensorSizeRect.selectSpaceUbSize != 0) {
        PFA_InitBuffer(
            selectSpaceUb,
            tilingData->promptAttentionTensorSizeRect.selectSpaceUbSize);
    }
    if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
        PFA_InitBuffer(PABmm1UB, 64);  // dcci refresh 64B
        PFA_InitBuffer(PABmm2UB, 64);  // dcci refresh 64B
    }
    // Use queue prefetching parameters. Enqueue a new calculation parameter
    // each time when calculating the outer tail. The head parameter of the
    // queue is used for calculation. After the calculation, the queue head is
    // dequeued.
    tailId = 0;
    headId = 0;
    queSize = 0;
    tailParams = &pfaParamsQueue[tailId];
    headParams = &pfaParamsQueue[headId];
    preHeadParams = &pfaParamsQueue[headId];
    isGlobalFirstCompute = true;
    mm1SingleCoreNPrev = 0;
    mm2MStridePrev = 0;
    mm2KaStridePrev = 0;

    tailParams->gmPingpong = 0;
    tailParams->useMask = false;
    tailParams->usePseShift = false;

    attentionMaskType = tilingData->promptAttentionBaseParams.sparseMode;
    if ((attenMask != NULL) &&
        (tilingData->promptAttentionBaseParams.useMask == 1)) {
        tailParams->useMask = true;
        attenMaskGm.SetGlobalBuffer((__gm__ U*)attenMask);
        attentionMaskStride = tilingData->promptAttentionBaseParams.maskKVsSize;
    }

    if ((pseShift != NULL) &&
        (tilingData->promptAttentionBaseParams.usePseShift == 1)) {
        tailParams->usePseShift = true;
        pseShiftGm.SetGlobalBuffer((__gm__ pseShiftType*)pseShift);
        pseShiftStride = tilingData->promptAttentionBaseParams.pseShiftS2Size;

        if constexpr (AscendC::IsSameType<pseShiftCastType, float>::value) {
            PFA_InitBuffer(
                pseShiftCastUb,
                (tilingData->promptAttentionTensorSizeRect.pseShiftUbSize) *
                    sizeof(float));
        }
    }

    softmaxExpUb = softmaxExpUb_.Get<computeType>(
        tilingData->promptAttentionTensorSizeRect.softmaxExpSize);

    if (tilingData->promptAttentionInitOutputParams.needInit == 1) {
        InitOutputSingleCore();
        InitLseOutputSingleCore();
    }
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::InitLseOutputSingleCore() {
    auto& initParams = tilingData->promptAttentionInitOutputParams;
    if (this->tilingData->promptAttentionBaseParams.isSoftMaxLseEnable ==
        true) {
        int64_t coreNum = GetBlockNum() * GetTaskRation();
        int64_t singleCoreLseSize =
            initParams.totalSoftMaxLseOutputSize / coreNum;
        if (tmp_block_idx == coreNum - 1) {
            singleCoreLseSize += initParams.totalSoftMaxLseOutputSize % coreNum;
        }
        InitOutput<float>(
            softmaxLseGm[tmp_block_idx *
                         (initParams.totalSoftMaxLseOutputSize / coreNum)],
            singleCoreLseSize, 3e+99);
        SyncAll();
    }
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::InitOutputSingleCore() {
    auto& initParams = tilingData->promptAttentionInitOutputParams;
    int64_t tailSize = (int64_t)initParams.totalOutputSize -
                       tmp_block_idx * (int64_t)initParams.singleCoreSize;
    if (tailSize > 0) {
        int64_t singleInitOutputSize = tailSize < initParams.singleCoreSize
                                           ? tailSize
                                           : initParams.singleCoreSize;
        if constexpr (IsSameType<O, int8_t>::value) {
            // InitOutput instr do not support int8/uint8, we Use half replace
            // int8/uint8 because D is 32bytes aligned, it will has no remainder
            // even if we Converted into half
            InitOutput<half>(
                attentionOutInitGm[tmp_block_idx *
                                   (int64_t)initParams.singleCoreSize / 2],
                singleInitOutputSize / 2, 0);
        } else {
            InitOutput<O>(attentionOutGm[tmp_block_idx *
                                         (int64_t)initParams.singleCoreSize],
                          singleInitOutputSize, 0);
        }
    }
    SyncAll();
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::initOffset() {
    offsetSS = tilingData->promptAttentionBaseParams.seqSize *
               tilingData->promptAttentionBaseParams.seqSize;
    offsetSH = tilingData->promptAttentionBaseParams.seqSize *
               tilingData->promptAttentionBaseParams.headSize;
    offsetSTypeNum =
        tilingData->promptAttentionBaseParams.seqSize * typeByteNum;
    offsetNSTypeNum =
        tilingData->promptAttentionBaseParams.headNumSize * offsetSTypeNum;
    offsetNSS = tilingData->promptAttentionBaseParams.headNumSize * offsetSS;
    offsetNSH = tilingData->promptAttentionBaseParams.headNumSize * offsetSH;
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::InitTensorSize(
    const PromptAttentionSingleCoreTensorSize* tensorSizeTiling) {
    mmResUbSize = tensorSizeTiling->mmResUbSize;
    attenMaskUbSize = tensorSizeTiling->attenMaskUbSize;
    pseShiftUbSize = tensorSizeTiling->pseShiftUbSize;
    maskSize = tensorSizeTiling->maskSize;
    softmaxMaxSize = tensorSizeTiling->softmaxMaxSize;
    softmaxSumSize = tensorSizeTiling->softmaxSumSize;
    softmaxExpSize = tensorSizeTiling->softmaxExpSize;
    spmTmpSize = tensorSizeTiling->spmTmpSize;
    scmTmpSize = tensorSizeTiling->scmTmpSize;
    bmm2ResUbSize = tensorSizeTiling->bmm2ResUbSize;
    tmpMMResBmm2PreUbSize = tensorSizeTiling->tmpMMResBmm2PreUbSize;
    tmpSoftmaxBmm2UbSize = tensorSizeTiling->tmpSoftmaxBmm2UbSize;
    selectSpaceUbSize = tensorSizeTiling->selectSpaceUbSize;
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::SoftmaxBasicComputeFirstNoTail(
    LocalTensor<computeType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
    LocalTensor<float>& softmaxSumUb, uint32_t souterSize) {
    LocalTensor<computeType> null;
    SoftMaxShapeInfo softmaxShapeInfo;
    if (this->headParams->isInnerTail) {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(
                this->headParams->singleProcessSInnerBmmTail)};
    } else {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),  // translate into type uint32_t
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(
                this->headParams->singleProcessSInnerSizeNow)};
    }
    SoftmaxFlashV2<computeType, false, true, true>(
        mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb, null, softmaxSumUb,
        softmaxMaxUb, softmaxFlashTilingData, softmaxShapeInfo);

    if (this->isSoftmaxResNeedUpdate) {
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<computeType, float>(
            mmResUb, softmaxMaxUb, this->negativeScalar, 0.0, softmaxShapeInfo);
        this->isSoftmaxLseNeedUpdate = true;
    }
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::SoftmaxBasicComputeNoTail(
    LocalTensor<computeType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
    LocalTensor<float>& softmaxSumUb, LocalTensor<computeType>& softmaxExpUb,
    uint32_t souterSize) {
    SoftMaxShapeInfo softmaxShapeInfo;
    if (this->headParams->isInnerTail) {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(
                this->headParams->singleProcessSInnerBmmTail)};
    } else {  // if inner has no tail
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(
                this->headParams->singleProcessSInnerSizeNow)};
    }

    SoftmaxFlashV2<computeType, true, true, true>(
        mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb, softmaxExpUb,
        softmaxSumUb, softmaxMaxUb, softmaxFlashTilingData, softmaxShapeInfo);
    if (this->isSoftmaxResNeedUpdate) {
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<computeType, float>(
            mmResUb, softmaxMaxUb, this->negativeScalar, 0.0, softmaxShapeInfo);
        this->isSoftmaxLseNeedUpdate = true;
    }
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::SoftmaxComputeFirstTail(
    LocalTensor<computeType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
    LocalTensor<float>& softmaxSumUb, uint32_t souterSize) {
    LocalTensor<computeType> null;
    SoftMaxShapeInfo softmaxShapeInfo;
    if (this->headParams->isInnerTail) {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(
                this->headParams->singleProcessSInnerBmmTail)};
    } else {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(
                this->headParams->singleProcessSInnerSizeNow)};
    }

    SoftmaxFlashV2<computeType, false, true, false>(
        mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb, null, softmaxSumUb,
        softmaxMaxUb, softmaxFlashTilingData, softmaxShapeInfo);
    if (this->isSoftmaxResNeedUpdate) {
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<computeType, float>(
            mmResUb, softmaxMaxUb, this->negativeScalar, 0.0, softmaxShapeInfo);
        this->isSoftmaxLseNeedUpdate = true;
    }
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::SoftmaxComputeTail(
    LocalTensor<computeType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
    LocalTensor<float>& softmaxSumUb, LocalTensor<computeType>& softmaxExpUb,
    uint32_t souterSize) {
    SoftMaxShapeInfo softmaxShapeInfo;
    if (this->headParams->isInnerTail) {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(
                this->headParams->singleProcessSInnerBmmTail)};
    } else {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(
                this->headParams->singleProcessSInnerSizeNow)};
    }

    SoftmaxFlashV2<computeType, true, true, false>(
        mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb, softmaxExpUb,
        softmaxSumUb, softmaxMaxUb, softmaxFlashTilingData, softmaxShapeInfo);
    if (this->isSoftmaxResNeedUpdate) {
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<computeType, float>(
            mmResUb, softmaxMaxUb, this->negativeScalar, 0.0, softmaxShapeInfo);
        this->isSoftmaxLseNeedUpdate = true;
    }
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::Bmm2UpdateDivNoTail(
    LocalTensor<computeType>& bmm2ResPreUb, LocalTensor<float>& softmaxSumUb) {
    PFAComputeParam* params = this->preHeadParams;
    int32_t headLoop = (tilingData->promptAttentionBaseParams.headSize +
                        softmaxTypeByteNum - 1) /
                       softmaxTypeByteNum;
    constexpr int32_t REPEAT_DATA_NUM = 256 / sizeof(computeType);

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = headLoop;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = headLoop;

    int32_t loop =
        tilingData->promptAttentionBaseParams.headSize / REPEAT_DATA_NUM;
    int32_t remain =
        tilingData->promptAttentionBaseParams.headSize % REPEAT_DATA_NUM;
    if constexpr (IsSameType<computeType, half>::value) {
        constexpr int32_t FP32_BLOCK_NUM = 8;
        constexpr int32_t FP32_MASK_NUM = 64;
        CopyRepeatParams copyRepeatParams{2, 1, 16, 8};
        int32_t calcSize = params->singleProcessSOuterSize * FP32_BLOCK_NUM;
        LocalTensor<float> tmpBuffer =
            tempBmm2Queue.template AllocTensor<float>();
        LocalTensor<half> tmpHalfBuffer =
            tmpBuffer[calcSize * 2].template ReinterpretCast<half>();

        int32_t repeat = (calcSize + FP32_MASK_NUM - 1) / FP32_MASK_NUM;
        Copy(tmpBuffer, softmaxSumUb, FP32_MASK_NUM, repeat, copyRepeatParams);
        Copy(tmpBuffer[FP32_BLOCK_NUM], softmaxSumUb, FP32_MASK_NUM, repeat,
             copyRepeatParams);
        pipe_barrier(PIPE_V);
        Cast(tmpHalfBuffer, tmpBuffer, RoundMode::CAST_ROUND, calcSize * 2);
        pipe_barrier(PIPE_V);

        for (int i = 0; i < loop; i++) {
            Div(bmm2ResPreUb[i * REPEAT_DATA_NUM],
                bmm2ResPreUb[i * REPEAT_DATA_NUM], tmpHalfBuffer,
                REPEAT_DATA_NUM, params->singleProcessSOuterSize, repeatParams);
        }
        if (remain) {
            Div(bmm2ResPreUb[loop * REPEAT_DATA_NUM],
                bmm2ResPreUb[loop * REPEAT_DATA_NUM], tmpHalfBuffer, remain,
                params->singleProcessSOuterSize, repeatParams);
        }
        tempBmm2Queue.FreeTensor(tmpBuffer);
    } else {
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResPreUb[i * REPEAT_DATA_NUM],
                bmm2ResPreUb[i * REPEAT_DATA_NUM], softmaxSumUb,
                REPEAT_DATA_NUM, params->singleProcessSOuterSize, repeatParams);
        }
        if (remain) {
            Div(bmm2ResPreUb[loop * REPEAT_DATA_NUM],
                bmm2ResPreUb[loop * REPEAT_DATA_NUM], softmaxSumUb, remain,
                params->singleProcessSOuterSize, repeatParams);
        }
    }
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::UpdateVmul(
    LocalTensor<computeType>& softmaxExpUb) {
    LocalTensor<computeType> bmm2ResPreUb =
        tempBmm2Ub.Get<computeType>(bmm2ResUbSize);

    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride =
        (tilingData->promptAttentionBaseParams.headSize + softmaxTypeByteNum -
         1) /
        softmaxTypeByteNum;
    repeatParams.dstRepStride =
        (tilingData->promptAttentionBaseParams.headSize + softmaxTypeByteNum -
         1) /
        softmaxTypeByteNum;

    // only support singleProcessSOuterSize <=255, headsize 32B align
    int32_t numOneRep = 256 / sizeof(computeType);
    int32_t loop = tilingData->promptAttentionBaseParams.headSize / numOneRep;
    int32_t remain = tilingData->promptAttentionBaseParams.headSize % numOneRep;

    for (int i = 0; i < loop; i++) {
        Mul(bmm2ResPreUb[i * numOneRep], softmaxExpUb,
            bmm2ResPreUb[i * numOneRep], numOneRep,
            this->headParams->singleProcessSOuterSize, repeatParams);
    }
    if (remain) {
        Mul(bmm2ResPreUb[loop * numOneRep], softmaxExpUb,
            bmm2ResPreUb[loop * numOneRep], remain,
            this->headParams->singleProcessSOuterSize, repeatParams);
    }
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::Bmm2UpdateAdd(
    LocalTensor<computeType>& bmm2ResUb) {
    LocalTensor<computeType> bmm2ResPreUb =
        tempBmm2Ub.Get<computeType>(bmm2ResUbSize);
    Add(bmm2ResPreUb, bmm2ResUb, bmm2ResPreUb, bmm2ResUbSize);
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::GetSingleCoreParam(int sIdx) {
    int64_t actualSeqLengthPerBatch = 0;
    int64_t actualSeqLengthKVPerBatch = 0;
    int64_t actualSeqQMin = 1;
    int64_t actualSeqKVmin = 1;
    if (isActualLenDimsNull) {
        actualSeqLengthPerBatch = tilingData->promptAttentionBaseParams.seqSize;
    } else {
        actualSeqLengthPerBatch =
            (tilingData->promptAttentionBaseParams.actualSeqLengthsSize ==
             actualSeqQMin)
                ? actualSeqLengthsGm.GetValue(0)
                : actualSeqLengthsGm.GetValue(sIdx);
    }
    if (isActualLenDimsKVNull) {
        actualSeqLengthKVPerBatch =
            tilingData->promptAttentionBaseParams.seqInnerSize;
    } else {
        actualSeqLengthKVPerBatch =
            (tilingData->promptAttentionBaseParams.actualSeqLengthsKVSize ==
             actualSeqKVmin)
                ? actualSeqLengthsKVGm.GetValue(0)
                : actualSeqLengthsKVGm.GetValue(sIdx);
    }

    this->tailParams->singleProcessSInnerSize =
        tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
    singleProcessSOuterSizeWhole =
        tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
    MultiHeadQ = tilingData->promptAttentionBaseParams.headSize *
                 tilingData->promptAttentionBaseParams.headNumSize;
    MultiHeadKV = MultiHeadQ / headNumRatio;

    if (attentionMaskType != 4) {
        this->tailParams->preTokensPerBatch =
            (int64_t)tilingData->promptAttentionBaseParams.preTokens;
        this->tailParams->nextTokensPerBatch =
            (int64_t)tilingData->promptAttentionBaseParams.nextTokens;
        actualSeqLengthPerBatch =
            ((int64_t)actualSeqLengthPerBatch >
             (int64_t)actualSeqLengthKVPerBatch + actualKVPrefixLen +
                 (int64_t)tilingData->promptAttentionBaseParams.preTokens) &&
                    (attentionMaskType != 4)
                ? (int64_t)actualSeqLengthKVPerBatch + actualKVPrefixLen +
                      (int64_t)tilingData->promptAttentionBaseParams.preTokens
                : actualSeqLengthPerBatch;
    } else {
        this->tailParams->preTokensPerBatch =
            (int64_t)tilingData->promptAttentionBaseParams.preTokens -
            actualSeqLengthKVPerBatch - actualKVPrefixLen +
            actualSeqLengthPerBatch;
        this->tailParams->nextTokensPerBatch =
            (int64_t)tilingData->promptAttentionBaseParams.nextTokens +
            actualSeqLengthKVPerBatch + actualKVPrefixLen -
            actualSeqLengthPerBatch;
        actualSeqLengthPerBatch =
            ((int64_t)actualSeqLengthPerBatch >
             (int64_t)actualSeqLengthKVPerBatch + actualKVPrefixLen +
                 (int64_t)this->tailParams->preTokensPerBatch)
                ? (int64_t)actualSeqLengthKVPerBatch + actualKVPrefixLen +
                      (int64_t)this->tailParams->preTokensPerBatch
                : actualSeqLengthPerBatch;
    }

    singleProcessSOuterSizeTail =
        (actualSeqLengthPerBatch % singleProcessSOuterSizeWhole != 0)
            ? actualSeqLengthPerBatch % singleProcessSOuterSizeWhole
            : singleProcessSOuterSizeWhole;
    this->tailParams->unalignSInner =
        (actualSeqLengthKVPerBatch %
             this->tailParams->singleProcessSInnerSize !=
         0)
            ? actualSeqLengthKVPerBatch %
                  this->tailParams->singleProcessSInnerSize
            : this->tailParams->singleProcessSInnerSize;
    uint32_t maxInnerLoopPromptTimes =
        (actualSeqLengthKVPerBatch + this->tailParams->singleProcessSInnerSize -
         1) /
        this->tailParams->singleProcessSInnerSize;
    this->tailParams->singleProcessSInnerSizeTail =
        (this->tailParams->unalignSInner + typeByteNum - 1) / typeByteNum *
        typeByteNum;
    this->tailParams->maskInnerTailAlign =
        (this->tailParams->unalignSInner + maskTypeByteNum - 1) /
        maskTypeByteNum * maskTypeByteNum;
    this->tailParams->padSize =
        this->tailParams->maskInnerTailAlign - this->tailParams->unalignSInner;

    if (pseShiftTypeByteNum != 0) {
        this->tailParams->pseShiftInnerTailAlign =
            (this->tailParams->unalignSInner + pseShiftTypeByteNum - 1) /
            pseShiftTypeByteNum * pseShiftTypeByteNum;
        this->tailParams->pseShiftPadSize =
            this->tailParams->pseShiftInnerTailAlign -
            this->tailParams->unalignSInner;
    }

    maxInnerLoopTimes = maxInnerLoopPromptTimes + maxInnerLoopPrefixTimes;
    InitTensorSize(&tilingData->promptAttentionTensorSizeRect);
    transposeTilingData = tilingData->transposeTilingDataRect;
    softmaxTilingData = tilingData->softmaxTilingDataRect;
    softmaxFlashTilingData = tilingData->softmaxFlashTilingDataRect;

    this->tailParams->actualSeqLengthPerBatch = actualSeqLengthPerBatch;
    this->tailParams->actualSeqLengthKVPerBatch = actualSeqLengthKVPerBatch;
}

template <typename PFAT>
__aicore__ inline void
JittorInferFlashAttentionS1s2Bns1X910Base<PFAT>::GetSparseParam(
    int64_t* preTokens, int64_t* nextTokens, int sIdx,
    PFAComputeParam*& params) {
    if (attentionMaskType == 3) {
        *preTokens = 214748647;
        *nextTokens = params->actualSeqLengthKVPerBatch + actualKVPrefixLen -
                      params->actualSeqLengthPerBatch;
    }
    if (attentionMaskType == 4) {
        int64_t actualSeqLengthPerBatchUser = 0;
        int64_t actualSeqLengthKVPerBatchUser = 0;
        int64_t actualSeqQMin = 1;
        int64_t actualSeqKVmin = 1;
        if (isActualLenDimsNull) {
            actualSeqLengthPerBatchUser =
                tilingData->promptAttentionBaseParams.seqSize;
        } else {
            actualSeqLengthPerBatchUser =
                (tilingData->promptAttentionBaseParams.actualSeqLengthsSize ==
                 actualSeqQMin)
                    ? actualSeqLengthsGm.GetValue(0)
                    : actualSeqLengthsGm.GetValue(sIdx);
        }
        if (isActualLenDimsKVNull) {
            actualSeqLengthKVPerBatchUser =
                tilingData->promptAttentionBaseParams.seqInnerSize;
        } else {
            actualSeqLengthKVPerBatchUser =
                (tilingData->promptAttentionBaseParams.actualSeqLengthsKVSize ==
                 actualSeqKVmin)
                    ? actualSeqLengthsKVGm.GetValue(0)
                    : actualSeqLengthsKVGm.GetValue(sIdx);
        }
        *preTokens = (int64_t)tilingData->promptAttentionBaseParams.preTokens -
                     actualSeqLengthKVPerBatchUser - actualKVPrefixLen +
                     actualSeqLengthPerBatchUser;
        *nextTokens =
            (int64_t)tilingData->promptAttentionBaseParams.nextTokens +
            actualSeqLengthKVPerBatchUser + actualKVPrefixLen -
            actualSeqLengthPerBatchUser;
    }
    params->preTokensPerBatch = *preTokens;
    params->nextTokensPerBatch = *nextTokens;
}

#endif  // JITTOR_INFER_FLASH_ATTENTION_S1S2_BNS1_X910_BASE_H
