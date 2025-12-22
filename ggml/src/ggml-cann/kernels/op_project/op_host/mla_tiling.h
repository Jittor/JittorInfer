/*!
 * \file mla_tiling.h
 * \brief
 */

#ifndef MLA_TILING_H_
#define MLA_TILING_H_
#include <cstdint>
#include <queue>
#include <vector>

#include "../utils/inc/error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "mla_tiling_context.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MLAHeadParams)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocks);
TILING_DATA_FIELD_DEF(float, tor);
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
TILING_DATA_FIELD_DEF(uint32_t, headSize);
TILING_DATA_FIELD_DEF(uint32_t, paraSize);
TILING_DATA_FIELD_DEF(uint32_t, maskTypeND);
TILING_DATA_FIELD_DEF(uint32_t, taskNum);
TILING_DATA_FIELD_DEF(uint32_t, kvSplitNum);
TILING_DATA_FIELD_DEF(uint32_t, splitTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MLAHeadParamsOp, MLAHeadParams)

BEGIN_TILING_DATA_DEF(NdMLADecodingMtpTilingTP1)
TILING_DATA_FIELD_DEF(uint32_t, tailBatch);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, batchIdx);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, qSeqLen);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, kvBlockIdx);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, prevKVSeqlen);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, batchSize);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, splitNum);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, prevSplitNumSum);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NdMLADecodingMtpTilingTP1Op,
                           NdMLADecodingMtpTilingTP1)

BEGIN_TILING_DATA_DEF(NdMLATiling)
TILING_DATA_FIELD_DEF(uint32_t, curQNBlockTile);
TILING_DATA_FIELD_DEF(uint32_t, maxKVseqlen);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, qSeqLen);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, kvSeqlen);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, qSeqHigh);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, qSeqLow);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, oSeqHigh);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, oSeqLow);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, maskHigh);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, maskLow);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NdMLATilingOp, NdMLATiling)

BEGIN_TILING_DATA_DEF(MLATilingData)
TILING_DATA_FIELD_DEF_STRUCT(MLAHeadParams, MLAHead);
TILING_DATA_FIELD_DEF_STRUCT(NdMLADecodingMtpTilingTP1,
                             ndMLADecodingMtpTilingTP1);
TILING_DATA_FIELD_DEF_STRUCT(NdMLATiling, ndMLATiling);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MLA, MLATilingData)

// Fix signature to match definition in mla_tiling.cpp (use references)
ge::graphStatus GetMLATilingParam(ContextParamsForMLA& contextParamsForMLA,
                                  uint32_t& blockDimToBeSet,
                                  MLATilingData& tilingData);

size_t GetMLAWorkSpaceSize(ContextParamsForMLA& contextParams,
                           uint32_t& blockDim);

}  // namespace optiling

#endif  // MLA_TILING_H_