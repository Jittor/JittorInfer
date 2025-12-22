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
#include "mla_prefill_tiling_context.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MLAPrefillHeadParams)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
TILING_DATA_FIELD_DEF(uint32_t, kvRealHeads);
TILING_DATA_FIELD_DEF(float, tor);
TILING_DATA_FIELD_DEF(uint32_t, headStride);
TILING_DATA_FIELD_DEF(uint32_t, maskStride);
// NUM8 = 0
// NUM9
TILING_DATA_FIELD_DEF(uint32_t, totalQBlkNum);
TILING_DATA_FIELD_DEF(uint32_t, headSize);
TILING_DATA_FIELD_DEF(uint32_t, paraSize);
TILING_DATA_FIELD_DEF(uint32_t, prefillTilingKey);
// NUM13 = 0
// NUM14
TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqLen);
TILING_DATA_FIELD_DEF(uint32_t, maskType);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSizeV);
TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqLenBNSD);
TILING_DATA_FIELD_DEF(uint32_t, windowSize);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MLAPrefillHeadParamsOp, MLAPrefillHeadParams)

BEGIN_TILING_DATA_DEF(MLAPrefillTiling)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, qSeqLen);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, kvSeqlen);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, mUbd);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, indexNN);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, qSeqHigh);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, qSeqLow);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, kSeqHigh);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, kSeqLow);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, vSeqHigh);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, vSeqLow);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, oSeqHigh);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, oSeqLow);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, totalQBlkNum);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MLAPrefillTilingOp, MLAPrefillTiling)

BEGIN_TILING_DATA_DEF(MLAPrefillTilingData)
TILING_DATA_FIELD_DEF_STRUCT(MLAPrefillHeadParams, MLAPrefillHead);
TILING_DATA_FIELD_DEF_STRUCT(MLAPrefillTiling, prefillTiling);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MLAPrefill, MLAPrefillTilingData)

ge::graphStatus GetMLAPrefillTilingParam(
    ContextParamsForMLA& contextParamsForMLA, uint32_t& blockDimToBeSet,
    MLAPrefillTilingData& tilingData);

size_t GetMLAPrefillWorkSpaceSize(ContextParamsForMLA& contextParams,
                                  uint32_t& blockDim);

}  // namespace optiling

#endif  // MLA_TILING_H_