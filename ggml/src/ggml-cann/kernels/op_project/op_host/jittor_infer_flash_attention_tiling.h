/**
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License. 
*/

/*!
* \file jittor_infer_flash_attention_tiling.h
* \brief
*/
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_JITTORINFERFLASHATTENTION_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_JITTORINFERFLASHATTENTION_H_
#include <cstdint>
#include <vector>
#include <queue>
#include "exe_graph/runtime/tiling_context.h"
#include "../utils/inc/tiling/data_copy_transpose_tiling_def.h"
#include "../utils/inc/tiling/data_copy_transpose_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../utils/inc/error/ops_error.h"
#include "register/op_def_registry.h"
#ifdef ASCENDC_OP_TEST
#define PFA_EXTERN_C extern "C"
#else
#define PFA_EXTERN_C
#endif

#include "jittor_infer_flash_attention_tiling_const.h"
#include "jittor_infer_flash_attention_tiling_context.h"
#include "jittor_infer_flash_attention_tiling_struct.h"

namespace optiling { 

constexpr uint32_t BYTE_BLOCK = 32; // The block size of datacopy, which moves data at the block granularity.
constexpr uint32_t SOFTMAX_BUFFER_NUM = 3;

constexpr uint32_t NUM_0 = 0;
constexpr uint32_t NUM_1 = 1;
constexpr uint32_t NUM_2 = 2;
constexpr uint32_t NUM_3 = 3;
constexpr uint32_t NUM_4 = 4;
constexpr uint32_t INDEX_2 = 2;
constexpr uint32_t INDEX_3 = 3;
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t ATTENTION_OUT_INDEX = 0;
constexpr uint32_t PSE_SHIFT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 6;
constexpr uint32_t DEQ_SCALE1_INDEX = 7;
constexpr uint32_t QUANT_SCALE1_INDEX = 8;
constexpr uint32_t DEQ_SCALE2_INDEX = 9;
constexpr uint32_t QUANT_SCALE2_INDEX = 10;
constexpr uint32_t QUANT_OFFSET2_INDEX = 11;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 12;
constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 13;

constexpr uint32_t INPUT_QKV_SHAPE_MIN_DIMS = 2;
constexpr uint32_t INPUT_QKV_SHAPE_MAX_DIMS = 4;

constexpr uint32_t ATTR_N_INDEX = 0;
constexpr uint32_t ATTR_SCALE_INDEX = 1;
constexpr uint32_t ATTR_PRE_TOKEN_INDEX = 2;
constexpr uint32_t ATTR_NEXT_TOKEN_INDEX = 3;
constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 5;

constexpr uint64_t EMPTY_KV_TILING_KEY = 20;
constexpr uint32_t LOOP_BEGIN_NUM = 0;
constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr uint32_t SPARSE_MODE_BAND = 4;
constexpr uint32_t SPARSE_MODE_INT_MAX = 214748647;
constexpr uint32_t ATTR_SPARSE_MODE = 6;
constexpr uint32_t ATTR_INNER_PRECISE = 7;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;
constexpr uint32_t PSE_SHIFT_DIM = 4;
constexpr uint32_t ATTENTION_MASK_DIM2 = 2;
constexpr uint32_t ATTENTION_MASK_DIM3 = 3;
constexpr uint32_t ATTENTION_MASK_DIM4 = 4;
constexpr int32_t BLOCK_SIZE_BASE = 128;  // The current requirement is a multiple of 128, and to prevent cross block handling, the mm base is also set to 128.
constexpr int32_t BLOCK_SIZE_MAX = 512;

constexpr uint32_t CVDIFF_S2_THRESHOLDS = 1;
constexpr uint32_t CVDIFF_SMALL_QS_THRESHOLDS = 16;
constexpr uint32_t CVDIFF_MM1RES_UB_SIZE = 16384; // 128 * 128
constexpr uint32_t CVDIFF_SOUTER_FACTOR_DEFAULT = 128;
constexpr uint32_t CVDIFF_SMALL_KV_THRESHOLDS = 1024;
constexpr uint32_t CVDIFF_SINNER_FACTOR_SMALL_KVS = 512;   // kv_s <= 512 scene sinner slice size
constexpr uint32_t CVDIFF_SINNER_FACTOR_DEFAULT = 1024;    // CV diff general scene sinner slice size
constexpr uint32_t CVDIFF_SINNER_FACTOR_SMALL_QS = 2048;   // q_s <= 16 scene sinner slice size
constexpr uint32_t CVDIFF_MSD_BUFFER_SIZE_512B = 512; // 0.5k
constexpr uint32_t CVDIFF_MSD_BUFFER_SIZE_1024B = 1024; // 0.5k

constexpr uint32_t SPLIT_DOUBLE_UB = 2;
constexpr uint32_t DSPLIT_THRESHOLDS_512 = 1024;
constexpr uint64_t DSPLIT_S2_D_TILING_KEY = 400;
constexpr uint64_t DSPLIT_S2_TILING_KEY = 300;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint64_t BENCHMARK_TILING_KEY = 1000000000000000000;
constexpr uint32_t THIRTY_ONE = 31;
constexpr uint32_t FROM_FUSED_FLAG = 71;
constexpr uint32_t MATMUL_NORM_MIN_SEQ = 128;
constexpr uint32_t MATMUL_NORM_MIN_HEADSIZE = 128;

constexpr uint32_t BLIMIT = 65536;
constexpr uint32_t NLIMIT = 256;  // n <= 256
constexpr uint32_t SLIMIT = 20971520;  // s、kvs <= 20M
constexpr uint32_t DLIMIT = 1024; // D <= 512
constexpr uint32_t TLIMIT = 65536; // T <= 64K

constexpr uint32_t MSD_UB_BASE_WIDTH = 16;
constexpr uint32_t MSD_UB_HEGHT = 256;
constexpr uint32_t MSD_UB_INQUEUE = 8;
constexpr uint32_t MSD_UB_TMP_NM = 16;
constexpr uint32_t NO_MSD_UB_BMM2 = 64;
constexpr uint32_t NO_MSD_HIGH_PERFORMANCE = 2;
constexpr uint32_t NO_MSD_HIGH_PRECISION = 3;
constexpr uint32_t ONE_BLK_SIZE_PFA = 32;
constexpr uint32_t MAX_SOFTMAX_COMPUTE_LINES = 4;
constexpr uint32_t COMPUTELINE_FOR_BIG_D = 1;
constexpr uint32_t MAX_COMPUTELINES = 16;
constexpr uint32_t UB_SIZE_FOR_1_K = 1024;
constexpr uint32_t MSD_BIG_D = 256;
constexpr uint32_t CV_RATIO = 2;

constexpr int64_t SAMEAB_D_LIMIT_192 = 192L;
constexpr int64_t HIGH_PERF_BUFFER_NUM = 6L;
constexpr int64_t HIGH_PERF_API_BUFFER_MULTIPLE = 2L;
constexpr int64_t FRACTAL_NUM = 16;
constexpr int64_t AIV_AIC_NUM_RATIO = 2L;
constexpr int64_t S1_VEC2_BASE_SIZE_MAX = 512L;
constexpr int64_t BMM_BASICBLOCK_M_128 = 128L;
constexpr int64_t BMM_BASICBLOCK_N_256 = 256L;
constexpr int64_t BMM_BASICBLOCK_N_128 = 128L;
constexpr int64_t BMM1_DEPTH_A1_2 = 2L;
constexpr int64_t BMM1_DEPTH_A1_3 = 3L;
constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
constexpr int64_t MAX_AIC_NUM = 24L;
constexpr int64_t MAX_AIV_NUM = 48L;
constexpr int64_t INVALID_ROW_SPARSE_RATIO = 6L;
constexpr int64_t HIGH_PERF_BLOCK_SIZE = 128L;

enum AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_MODE,
    PREFIX_MODE,
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
    BAND_LEFT_UP_CAUSAL_MODE
};

enum class SparseTypeEnum {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    BAND_COMPRESS = 6,
    RIGHT_DOWN_CAUSAL = 7,
    RIGHT_DOWN_CAUSAL_BAND = 8,
    BAND_LEFT_UP_CAUSAL = 9
};

BEGIN_TILING_DATA_DEF(PromptAttentionBaseParams)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, headNumSize);
  TILING_DATA_FIELD_DEF(uint32_t, seqSize);
  TILING_DATA_FIELD_DEF(uint32_t, headSize);
  TILING_DATA_FIELD_DEF(float, scaleValue);
  TILING_DATA_FIELD_DEF(int32_t, preTokens);
  TILING_DATA_FIELD_DEF(int32_t, nextTokens);
  TILING_DATA_FIELD_DEF(int32_t, blockSize);
  TILING_DATA_FIELD_DEF(int32_t, blockTableDim2);
  TILING_DATA_FIELD_DEF(int32_t, PABlockNumSum);
  TILING_DATA_FIELD_DEF(uint32_t, dimNumOfseq);
  TILING_DATA_FIELD_DEF(uint32_t, typeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, seqInnerSize);
  TILING_DATA_FIELD_DEF(uint32_t, usePseShift);
  TILING_DATA_FIELD_DEF(uint32_t, useMask);
  TILING_DATA_FIELD_DEF(uint32_t, headNumRatio);
  TILING_DATA_FIELD_DEF(uint32_t, attenMaskElemType);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, pseMaskMaxSize);
  TILING_DATA_FIELD_DEF(uint32_t, maskTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, outputTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
  TILING_DATA_FIELD_DEF(uint32_t, alignedHeadSize);
  TILING_DATA_FIELD_DEF(uint32_t, splitS2);
  TILING_DATA_FIELD_DEF(uint32_t, splitD);
  TILING_DATA_FIELD_DEF(uint32_t, layoutType);
  TILING_DATA_FIELD_DEF(uint32_t, PAlayoutType);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftS1Size);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftS2Size);
  TILING_DATA_FIELD_DEF(uint32_t, maskKVsSize);
  TILING_DATA_FIELD_DEF(uint32_t, maskQsSize);
  TILING_DATA_FIELD_DEF(uint32_t, isLayoutSH);
  TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsNull);
  TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsKVNull);
  TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsSize);
  TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize);
  TILING_DATA_FIELD_DEF(uint32_t, deqScaleFlag);
  TILING_DATA_FIELD_DEF(uint32_t, deqScale2Flag);
  TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxOuterSize);
  TILING_DATA_FIELD_DEF(uint32_t, isSoftMaxLseEnable);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionBaseParamsOp, PromptAttentionBaseParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSeqParams)
  // Temporary reuse
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, CoreHeadNumTail);       // coreNStart
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, actualS1);              // coreNEnd
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, actualCoreNums);        // coreSidStart
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, singleCoreHeadNumSize); // coreSidEnd
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreSeqPosStart);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreSeqPosEnd);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSeqParamsOp, PromptAttentionSeqParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSingleCoreParams)
  TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
  TILING_DATA_FIELD_DEF(uint32_t, singleProcessSOuterSize);
  TILING_DATA_FIELD_DEF(uint32_t, multiSmaxsInnerLoopTimes);
  TILING_DATA_FIELD_DEF(uint32_t, actualCoreNums);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftBatch);
  TILING_DATA_FIELD_DEF(uint32_t, attenMaskBatch);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSingleCoreParamsOp, PromptAttentionSingleCoreParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSingleCoreTensorSize)
  TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, attenMaskUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, maskSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxMaxSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxSumSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxExpSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxValueSize);
  TILING_DATA_FIELD_DEF(uint32_t, spmTmpSize);
  TILING_DATA_FIELD_DEF(uint32_t, scmTmpSize);
  TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, tmpMMResBmm2PreUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, tmpSoftmaxBmm2UbSize);
  TILING_DATA_FIELD_DEF(uint32_t, selectSpaceUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, tmpSoftMaxV2Size);
  TILING_DATA_FIELD_DEF(uint32_t, mm1TmpUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, mm2TmpUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbMsdSize);
  TILING_DATA_FIELD_DEF(uint32_t, tempBmm2QueueMsdSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdInQueueSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdQRowSumBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdAMaxTmpBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdAMaxResBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdSoftmaxResAmaxBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdSoftmaxRowSumScaleBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdScaleBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdOffsetBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdTmpMm1BuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdTmpMm2BuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdOutQueueSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdComputeLines);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSingleCoreTensorSizeOp, PromptAttentionSingleCoreTensorSize)

BEGIN_TILING_DATA_DEF(PromptAttentionInitOutputParams)
  TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize);
  TILING_DATA_FIELD_DEF(int64_t, totalOutputSize);
  TILING_DATA_FIELD_DEF(int64_t, totalSoftMaxLseOutputSize);
  TILING_DATA_FIELD_DEF(uint32_t, needInit);
  TILING_DATA_FIELD_DEF(uint32_t, isOneN);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionInitOutputParamsOp, PromptAttentionInitOutputParams)

BEGIN_TILING_DATA_DEF(JittorInferFlashAttentionTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingDataRect);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingDataRect);

  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionBaseParams, promptAttentionBaseParams);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSeqParams, promptAttentionSeqParams);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSingleCoreParams, promptAttentionSingleCoreParams);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSingleCoreTensorSize, promptAttentionTensorSizeRect);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionInitOutputParams, promptAttentionInitOutputParams);

  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingDataRect);
  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingDataRect);
  TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingDataRect);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(JittorInferFlashAttention, JittorInferFlashAttentionTilingData)

BEGIN_TILING_DATA_DEF(PFAInputParams)
    TILING_DATA_FIELD_DEF(int64_t, bSize);
    TILING_DATA_FIELD_DEF(int64_t, n2Size);
    TILING_DATA_FIELD_DEF(int64_t, gSize);
    TILING_DATA_FIELD_DEF(float, keepProb);
    TILING_DATA_FIELD_DEF(float, scaleValue);
    TILING_DATA_FIELD_DEF(int64_t, preTokens);
    TILING_DATA_FIELD_DEF(int64_t, nextTokens);
    // in pse encoding scenes, s1 and s2 might not equal with s1, s2 in Q, K
    TILING_DATA_FIELD_DEF(int64_t, pseS1Size);
    TILING_DATA_FIELD_DEF(int64_t, pseS2Size);
    TILING_DATA_FIELD_DEF(uint32_t, pseBSize);
    TILING_DATA_FIELD_DEF(uint32_t, bandIndex);

    // 1: BSH/BSND, 2: SBH, 3: BNSD
    TILING_DATA_FIELD_DEF(uint8_t, layoutType);
    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    TILING_DATA_FIELD_DEF(uint8_t, pseShapeType);
    // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskShapeType);
    // 0: fp16, 1: bool(uint8)
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskDataType);
    // ALL: 0, NONE: 1, ANY: 2, CAUSAL: 3, BAND: 4 };
    TILING_DATA_FIELD_DEF(uint8_t, attenMaskCompressMode);
    // 0: high precise, 1: high performance, 2: invalid line high precise
    TILING_DATA_FIELD_DEF(uint8_t, implMode);
    TILING_DATA_FIELD_DEF(uint8_t, sparseType);
    TILING_DATA_FIELD_DEF(uint8_t, pseEncodeType);
    TILING_DATA_FIELD_DEF(uint8_t, isSoftMaxLseEnable);
    TILING_DATA_FIELD_DEF(uint16_t, remain);
    TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
    TILING_DATA_FIELD_DEF(uint32_t, pseType);
    TILING_DATA_FIELD_DEF(uint32_t, rsv1);
    TILING_DATA_FIELD_DEF(int64_t, qStartIdx);
    TILING_DATA_FIELD_DEF(int64_t, kvStartIdx);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFAInputParamsOp, PFAInputParams)

BEGIN_TILING_DATA_DEF(PFAMultiCoreParams)
    TILING_DATA_FIELD_DEF(int32_t, coreNum);
    TILING_DATA_FIELD_DEF(int32_t, reserve);
    // BN2GS1.o
    TILING_DATA_FIELD_DEF(int64_t, totalSize);
    // BN2GS1.o / core_num
    TILING_DATA_FIELD_DEF(int64_t, splitFactorSize);
    TILING_DATA_FIELD_DEF(int64_t, splitFactorTailSize);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 48, sparseStartIdx);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFAMultiCoreParamsOp, PFAMultiCoreParams)

BEGIN_TILING_DATA_DEF(PFACoreParams)
    TILING_DATA_FIELD_DEF(int32_t, s1BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, s1BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, s1OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, s1Vec2OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, s2BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, s2BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, s2OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, dBaseSize);
    TILING_DATA_FIELD_DEF(int32_t, dBaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, dOuterSize);
    TILING_DATA_FIELD_DEF(int32_t, bBaseSize);
    TILING_DATA_FIELD_DEF(int32_t, bBaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, bOuterSize);
    TILING_DATA_FIELD_DEF(int32_t, n2BaseSize);
    TILING_DATA_FIELD_DEF(int32_t, n2BaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, n2OuterSize);
    TILING_DATA_FIELD_DEF(int32_t, gBaseSize);
    TILING_DATA_FIELD_DEF(int32_t, gBaseTailSize);
    TILING_DATA_FIELD_DEF(int64_t, gOuterSize);
    TILING_DATA_FIELD_DEF(int32_t, rsvd);
    TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS1);
    TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS2);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFACoreParamsOp, PFACoreParams)

BEGIN_TILING_DATA_DEF(PFATensorSizeParams)
    TILING_DATA_FIELD_DEF(int32_t, bmm1ResUbSize);
    TILING_DATA_FIELD_DEF(int32_t, attenMaskUbSize);
    TILING_DATA_FIELD_DEF(int32_t, pseUbSize);
    TILING_DATA_FIELD_DEF(int32_t, dropMaskUbSize);
    TILING_DATA_FIELD_DEF(int32_t, castUbSize);
    TILING_DATA_FIELD_DEF(int32_t, softmaxMaxUbSize);
    TILING_DATA_FIELD_DEF(int32_t, softmaxSumUbSize);
    TILING_DATA_FIELD_DEF(int32_t, softmaxExpUbSize);
    TILING_DATA_FIELD_DEF(int32_t, apiTmpBufferBytes);
    TILING_DATA_FIELD_DEF(int32_t, bmm2ResUbSize);
    TILING_DATA_FIELD_DEF(int32_t, inputQueBytes);
    TILING_DATA_FIELD_DEF(int32_t, outputQueBytes);
    // API buffer use remain space of ub
    TILING_DATA_FIELD_DEF(int32_t, tmpBufBytes);
    TILING_DATA_FIELD_DEF(int32_t, softmaxMaxOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, softmaxSumOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, maxSumApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, customSoftmaxApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, pseTbufOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, dropoutApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, maxSumApiSize);
    TILING_DATA_FIELD_DEF(int32_t, customSoftmaxApiSize);
    TILING_DATA_FIELD_DEF(int32_t, dropoutApiSize);
    TILING_DATA_FIELD_DEF(int32_t, attenMaskApiSize);
    TILING_DATA_FIELD_DEF(int32_t, attenMaskApiOffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2Size);
    TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2OffsetBytes);
    // workspace
    TILING_DATA_FIELD_DEF(int32_t, wkspSection1OffsetBytes);
    TILING_DATA_FIELD_DEF(int32_t, wkspSection2OffsetBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PFATensorSizeParamsOp, PFATensorSizeParams)

BEGIN_TILING_DATA_DEF(MLAGeneralTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(PFAInputParams, PFAinputParams);
    TILING_DATA_FIELD_DEF_STRUCT(PFAMultiCoreParams, PFAmultiCoreParams);
    TILING_DATA_FIELD_DEF_STRUCT(PFACoreParams, PFAcoreParams);
    TILING_DATA_FIELD_DEF_STRUCT(PFATensorSizeParams, PFAtensorSizeParams);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
    TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingDataTailCore);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(JittorInferFlashAttention_4000000000000000000, MLAGeneralTilingData)
REGISTER_TILING_DATA_CLASS(JittorInferFlashAttention_4000000000000000001, MLAGeneralTilingData)

class BufferNum {
public:
    // sum and max always use fp32, shape is (S1, 1), inner axis align 32B.
    size_t bufferS1S2Num; // unit: input dtype
    size_t bufferS1DNum;
    size_t bufferExpNum; // unit: input dtype, shape: [S1, 1], inner axis align 32B.
};

class JittorInferFlashAttentionTiling {
public:
    JittorInferFlashAttentionTiling(fe::PlatFormInfos* platFormInfo): ascendcPlatform(platFormInfo) {}
    ge::graphStatus RunBigKernelTilingWithParams(ContextParamsForPFATiling& contextKeyParams,
                                                uint64_t& tilingKey, uint32_t& blockDimToBeSet,
                                                JittorInferFlashAttentionTilingData& tilingData);
    ge::graphStatus JittorInferFlashAttentionSetTilingData(gert::TilingContext* context,
                                                    JittorInferFlashAttentionTilingData& tilingData);
    bool CheckNonEmptyShapeExceptions(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* shape,
                                      const std::string &sName);
    bool fromPFA_ = true;
protected:
    ge::graphStatus ConvertContextToPFAParams(gert::TilingContext* context, ContextParamsForPFATiling& contextKeyParams);
    ge::graphStatus TilingGetTilingKeyAttentionAscendC(uint64_t& tilingKey, ContextParamsForPFATiling& contextKeyParams,
                                                      bool useNewTiling, JittorInferFlashAttentionTilingData& tilingData);
    void JittorInferFlashAttentionSplitNS(ContextParamsForPFATiling& contextKeyParams, JittorInferFlashAttentionTilingData& tilingData, uint32_t curCoreNum, std::vector<int64_t>& actualSeqLengths);
    void JittorInferFlashAttentionSplitNSNew(ContextParamsForPFATiling& contextKeyParams, JittorInferFlashAttentionTilingData& tilingData, uint32_t curCoreNum, std::vector<int64_t>& actualSeqLengths,
                                                    std::vector<int64_t>& actualSeqLengthsKV, bool useBalanceTiling);
    void GetPreNextTokensLeftUp(JittorInferFlashAttentionTilingData& tilingData, uint32_t actualSeqLength, uint32_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
    void SetSplitCoreMode(JittorInferFlashAttentionTilingData& tilingData, uint32_t sOuterFactor);
    void JittorInferFlashAttentionSplitSeqOneN(JittorInferFlashAttentionTilingData& tilingData, uint32_t curCoreNum, bool isVectorCore);
    bool EnableMTE2BmmPipe(JittorInferFlashAttentionTilingData& tilingData, matmul_tiling::MatmulApiTiling& bmm,
                          TCubeTiling& bmmTilingData, uint32_t sOuterFactor, uint32_t sInnerFactor);
    void EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData);
    bool JittorInferFlashAttentionCheckBmm1(JittorInferFlashAttentionTilingData& tilingData, TCubeTiling& bmm1TilingData,
                                      int64_t l1SizeRemain, int64_t l0CSize,
                                      uint32_t sOuterFactor, uint32_t sInnerFactor,
                                      bool allGM = false, bool autoBaseMNK = false);
    bool JittorInferFlashAttentionCheckBmm2(JittorInferFlashAttentionTilingData& tilingData, TCubeTiling& bmm1TilingData,
                                      int64_t l1SizeRemain, int64_t l0CSize,
                                      uint32_t sOuterFactor, uint32_t sInnerFactor,
                                      uint32_t dSplitFactor, bool allGM = false, bool autoBaseMNK = false);
    void JittorInferFlashAttentionSetTensorSize(JittorInferFlashAttentionTilingData& tilingData,
                        PromptAttentionSingleCoreTensorSize& tensorSize, uint32_t sOuterFactor, uint32_t sInnerFactor);
    bool JittorInferFlashAttentionCheckArgsLegal(JittorInferFlashAttentionTilingData& tilingData, int64_t ubSize, int64_t l1Size,
                                            int64_t l0CSize, uint32_t typeByteSize, uint32_t& sOuterFactor,
                                            uint32_t sInnerFactor, bool& updateDiv, uint32_t maskTypeSize, uint32_t dSplitFactor);
    ge::graphStatus AdjustBasicBlock(JittorInferFlashAttentionTilingData& tilingData, uint32_t& sOuterFactor);
    ge::graphStatus JittorInferFlashAttentionApiTiling(JittorInferFlashAttentionTilingData& tilingData, uint32_t typeSize,
                                                  uint32_t sOuterFactor, uint32_t softmaxSInnerFactor, uint32_t softmaxSOuterFactor);
    ge::graphStatus GetRectangleFactor(uint32_t seqSplit, std::queue<uint32_t>& sQueue, int32_t threshold = 16);
    ge::graphStatus SetInputLayout(const char* layout);
    bool GetApiTmpSize(const uint32_t sOuterFactor, const uint32_t sInnerFactor,
                        const uint32_t typeByteSize);
    uint32_t CalculateL1SizeUsed(JittorInferFlashAttentionTilingData& tilingData, const uint32_t typeByteSize);
    bool CheckInputDimAndHeadNum(ContextParamsForPFATiling& contextKeyParams, uint32_t nQAttr, uint32_t nKVAttr);
    bool SetTilingHeadNumRatio(ContextParamsForPFATiling& contextKeyParams, const int32_t* numQueryHeads,
                              const int32_t* numKeyValueHeads, JittorInferFlashAttentionTilingData& tilingData);
    void JittorInferFlashAttentionInitOutputSplit(uint64_t totalSize, JittorInferFlashAttentionTilingData &tilingData,
                                            uint32_t curCoreNum);
    void JittorInferFlashAttentionInitSoftmaxLseOutputSplit(uint64_t totalSize, JittorInferFlashAttentionTilingData &tilingData);
    void Align(uint32_t &num);
    ge::graphStatus GetBasicShape(uint32_t &b, uint32_t &s, uint32_t &h, uint32_t &seqInnerSize,
                                const gert::StorageShape *queryShape, const gert::StorageShape *keyShape, const uint32_t n);
    ge::graphStatus GetBasicShape910B(uint32_t &b, uint32_t &s, uint32_t &h, uint32_t &seqInnerSize,
                                      const gert::StorageShape *queryShape, const gert::StorageShape *keyShape, const uint32_t n);

    size_t GetPFAWorkSpaceSize(JittorInferFlashAttentionTilingData& tilingData);
    void GetMatMulType(matmul_tiling::DataType &mmInputType, matmul_tiling::DataType &mmOutputType);
    ge::graphStatus CheckKeyValueParamsConsistency(const ContextParamsForPFATiling& contextKeyParams);
    bool CheckActualSeqLength(ContextParamsForPFATiling& contextKeyParams, uint32_t b, uint32_t sQ, uint32_t sKV,
                              const gert::Tensor* actualSeqLenQ, const gert::Tensor* actualSeqLenKV, InputLayout inLayout, JittorInferFlashAttentionTilingData& tilingData);
    bool CheckPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape *pseShiftShape,
                                  uint32_t b, uint32_t n, uint32_t s1, uint32_t s2);
    bool CheckPATypeAndShape(ContextParamsForPFATiling& contextKeyParams, const gert::Tensor* actualSeqLenKV,
                                  int32_t b, int32_t n, int32_t h, int32_t headNumRatio);
    bool CheckAttenMaskShape(ContextParamsForPFATiling& contextKeyParams, const int32_t* sparseMode, const gert::StorageShape* attenMaskShape,
                            uint32_t sQ, uint32_t sK, uint32_t batchSize);
    ge::graphStatus JittorInferFlashAttentionCVDiffSetTensorSize(JittorInferFlashAttentionTilingData& tilingData,
        PromptAttentionSingleCoreTensorSize& tensorSize, uint32_t sOuterFactor,
        uint32_t sInnerFactor, uint32_t softmaxSOuterFactor);
    bool JittorInferFlashAttentionComputeCVDiffParams(JittorInferFlashAttentionTilingData& tilingData,
        int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t typeByteSize,
        uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t maskTypeSize, uint32_t &softmaxSOuterFactor);
    bool FindOptimalTilingBasicBLock(JittorInferFlashAttentionTilingData& tilingData,
        uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor,
        int64_t ubSize, uint32_t typeByteSize, uint32_t maskTypeSize);
    bool FindOptimalTilingSouter(JittorInferFlashAttentionTilingData& tilingData,
        uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor,
        int64_t ubSize, uint32_t typeByteSize, uint32_t maskTypeSize);
    void InferTilingMod(const ContextParamsForPFATiling& contextKeyParams, const std::vector<int64_t>& actualSeqLengths, const std::vector<int64_t>& actualSeqLengthsKV,
        uint32_t actualSeqArrayLen, uint32_t hDivN, uint32_t seqInnerSize, int32_t sparseModeVal);
    ge::graphStatus AdjustCVTiling(uint32_t hDivN, uint32_t n, int64_t middle_actualSeqLengths,
        int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t maskElemSize,
        uint32_t& sOuterFactor, uint32_t& sInnerFactor, JittorInferFlashAttentionTilingData& tilingData);
    ge::graphStatus AdjustCVTilingCVDiff(int64_t ubSize, int64_t l1Size, int64_t l0CSize,
        uint32_t maskElemSize, uint32_t& sOuterFactor, uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor,
        JittorInferFlashAttentionTilingData& tilingData);
    bool CheckSparseModeRightDown(ContextParamsForPFATiling& contextKeyParams, const std::vector<int64_t>& actualSeqLengths,
                                  const std::vector<int64_t>& actualSeqLengthsKV, size_t lenDims);
    ge::graphStatus GetAndCheckEmptyQueryShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape *queryShape) const;
    void UpdateTilingKeyFlag(ContextParamsForPFATiling& contextKeyParams, uint64_t& tilingKey);
    int64_t JittorInferFlashAttentionSetMsdUbSize(JittorInferFlashAttentionTilingData& tilingData, PromptAttentionSingleCoreTensorSize& tensorSize, int32_t sInnerFactorTmp) const;
  
    ge::graphStatus CheckIOType(ContextParamsForPFATiling& contextKeyParams, JittorInferFlashAttentionTilingData& tilingData, int32_t& outputDataTypeSize);
    ge::graphStatus CheckMaskType(ContextParamsForPFATiling& contextKeyParams, JittorInferFlashAttentionTilingData& tilingData, uint32_t& maskElemSize);
    void SetMaskSize(const gert::StorageShape* attenMaskShape, JittorInferFlashAttentionTilingData& tilingData);
    ge::graphStatus CheckShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* queryShape, const gert::StorageShape* keyShape, 
                              const gert::StorageShape* valueShape, const gert::StorageShape* outShape, const gert::StorageShape* pseShiftShape,
                              const gert::StorageShape* attenMaskShape);
    uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum);

    protected:
        ContextParamsForPFATiling* contextKeyParamsPtr = nullptr;
        int64_t ubSizeRemain = 1;
        bool isSOuterNoTail = true;
        bool isSInnerNoTail = true;
        bool isDNoTail = true;
        bool enableQuantBF16 = false;
        bool enableMatmulNorm = false;
        bool enablePA = false;
        InputLayout inputLayout = InputLayout::BSH;
        ge::DataType inputType{ge::DT_FLOAT16};
        ge::DataType outputType{ge::DT_FLOAT16};
        ge::DataType intputKeyType{ge::DT_FLOAT16};
        ge::DataType intputValueType{ge::DT_FLOAT16};
        ge::DataType pseShiftElemType{ge::DT_FLOAT16};
        uint32_t dataTypeSize = FLOAT32SIZE;
        uint32_t coreNum;
        uint32_t aivNum;
        uint32_t aicNum;
        uint32_t typeByteNum;
        uint32_t outputTypeByteNum;
        uint32_t softmaxTypeByteNum;
        uint32_t pseShiftTypeByteNum = 0;
        uint32_t pseShiftElemSize = 0;
        uint32_t pseMaskMaxSize = 0;
        uint32_t pseShiftBatch = 0;
        uint32_t pseShiftS1 = 0;
        uint32_t pseShiftS2 = 0;
        uint32_t usePseShift = 0;
        uint32_t tmpS2 = 0;  // In the PA scenario, there is no S2 axis. Use the change amount to normalize the S2 length in both PA and non PA scenarios
        int32_t blockTableDim2 = 1;
        int32_t PABlockNumSum = 1;
        uint32_t maskTypeByteNum;
        uint32_t maxQuerySeq = 0;
        int64_t apiTmpSize = 1;
        uint32_t softmaxDataTypeNZ_ = FLOAT32SIZE;
        uint32_t softmaxDataTypeSize = FLOAT32SIZE; // BF16 calculates through FP32
        platform_ascendc::SocVersion curShortSocName;
        uint32_t dataTypeSize_ = 4;
        uint32_t layoutType = 0;
        uint32_t PAlayoutType = 0;
        platform_ascendc::PlatformAscendC ascendcPlatform;
        TilingMod tilingMod = TilingMod::CVSAME;
        SplitCoreMode splitCoreMode = SplitCoreMode::SPLIT_NBS_VECTOR;
        uint32_t splitD = 0;
        uint32_t splitS2 = 1; // It can only be 0 when the D axis is split
        uint64_t innerPrecise = HIGH_PERFORMANCE;
        size_t defaultSysWorkspaceSize;
        matmul_tiling::PlatformInfo ascendPlatformInfo;

        MLAGeneralTilingData mlaTilingData;
    };
    // end of class JittorInferFlashAttention
  PFA_EXTERN_C ge::graphStatus TilingJittorInferFlashAttention(gert::TilingContext* context);
}

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_JITTORINFERFLASHATTENTION_H_
