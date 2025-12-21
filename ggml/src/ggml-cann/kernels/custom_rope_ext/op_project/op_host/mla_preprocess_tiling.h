
#include <cstdint>
#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(PpMatmulTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numBatch);
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, m0);
TILING_DATA_FIELD_DEF(uint32_t, k0);
TILING_DATA_FIELD_DEF(uint32_t, n0);
TILING_DATA_FIELD_DEF(uint32_t, mLoop);
TILING_DATA_FIELD_DEF(uint32_t, kLoop);
TILING_DATA_FIELD_DEF(uint32_t, nLoop);
TILING_DATA_FIELD_DEF(uint32_t, coreLoop);
TILING_DATA_FIELD_DEF(uint32_t, swizzleCount);
TILING_DATA_FIELD_DEF(uint32_t, swizzleDirect);
TILING_DATA_FIELD_DEF(uint32_t, enShuffleK);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(uint32_t, enLoadAllAmat);
TILING_DATA_FIELD_DEF(uint32_t, b0matPingPongBufferLen);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PpMatmulTilingDataOp, PpMatmulTilingData)

BEGIN_TILING_DATA_DEF(MLAPreprocessTilingData)
//   TILING_DATA_FIELD_DEF(uint32_t, size);
    TILING_DATA_FIELD_DEF(uint32_t, numCore);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, perTaskNum);
    TILING_DATA_FIELD_DEF(uint32_t, resTaskNum);

    TILING_DATA_FIELD_DEF_STRUCT(PpMatmulTilingData, mm1);
    TILING_DATA_FIELD_DEF_STRUCT(PpMatmulTilingData, mm2);
    TILING_DATA_FIELD_DEF_STRUCT(PpMatmulTilingData, mm3);

    // rms1
    TILING_DATA_FIELD_DEF(uint32_t, rmsNumCore1);
    TILING_DATA_FIELD_DEF(uint32_t, rmsNumCol1);
    TILING_DATA_FIELD_DEF(uint32_t, rmsNumRow1);
    TILING_DATA_FIELD_DEF(uint32_t, rmsQuantMin1);

    // rms2
    TILING_DATA_FIELD_DEF(uint32_t, rmsNumCore2);
    TILING_DATA_FIELD_DEF(uint32_t, rmsNumCol2);
    TILING_DATA_FIELD_DEF(uint32_t, rmsNumRow2);
    TILING_DATA_FIELD_DEF(uint32_t, rmsQuantMin2);

    TILING_DATA_FIELD_DEF(uint32_t, hiddenSizeQ);
    TILING_DATA_FIELD_DEF(uint32_t, headNumQ);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, concatSize);
    TILING_DATA_FIELD_DEF(uint32_t, rotaryCoeff);
    TILING_DATA_FIELD_DEF(uint32_t, ntokens);
    TILING_DATA_FIELD_DEF(uint32_t, realCore);
    TILING_DATA_FIELD_DEF(uint32_t, nlCoreRun);
    TILING_DATA_FIELD_DEF(uint32_t, lCoreRun);
    TILING_DATA_FIELD_DEF(uint32_t, maxNPerLoopForUb);
    TILING_DATA_FIELD_DEF(uint32_t, preCoreLoopTime);
    TILING_DATA_FIELD_DEF(uint32_t, preCoreLoopNLast);
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreLoopTime);
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreLoopNLast);

    // EinSumQuant
    TILING_DATA_FIELD_DEF(uint32_t, esqFrontCore);
    TILING_DATA_FIELD_DEF(uint32_t, esqTailCore);
    TILING_DATA_FIELD_DEF(uint32_t, esqFrontCoreBatch);
    TILING_DATA_FIELD_DEF(uint32_t, esqTailCoreBatch);
    TILING_DATA_FIELD_DEF(uint32_t, esqHeadNum);
    TILING_DATA_FIELD_DEF(uint32_t, esqColNum);
    TILING_DATA_FIELD_DEF(uint32_t, esqUbHeadLoop);
    TILING_DATA_FIELD_DEF(uint32_t, esqHeadPerLoop);
    TILING_DATA_FIELD_DEF(uint32_t, esqHeadTail);
    TILING_DATA_FIELD_DEF(uint32_t, esqColLoop);
    TILING_DATA_FIELD_DEF(uint32_t, esqColTail);

    TILING_DATA_FIELD_DEF(uint32_t, maxWorksapceSize);
    TILING_DATA_FIELD_DEF(uint32_t, pertokenWorksapce);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MLAPreprocess, MLAPreprocessTilingData)
}
