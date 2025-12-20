#include <array>
#include <chrono>
#include <vector>
#include <queue>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>
#include <securec.h>
#include <cstdint>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <unistd.h>
#include <stdio.h>
#include "mla_prefill_tiling.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#define UNUSED_VALUE(x) (void)(x)

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {

constexpr std::array<int32_t, 8> PP_MM = { 16, 32, 48, 64, 80, 96, 112, 128 };
constexpr std::array<int32_t, 6> QN_TILE_LIST = { 128, 64, 32, 16, 8, 1 };

using IndexArr = std::array<int32_t, 4>;

inline uint32_t GetHigh32Bit(uint64_t v) { return static_cast<uint32_t>(v >> 32); }
inline uint32_t GetLoww32Bit(uint64_t v) { return static_cast<uint32_t>(v); }

inline int32_t ConvertValueToIndexMM(int32_t val, int32_t idxBound) // 16, 7
{
    return (val > PP_MM[idxBound]) ? idxBound : (val / 16 - 1);
}
const int32_t PP_NN_NUM = 16;
constexpr std::array<int32_t, PP_NN_NUM> PP_NN = {16,  32,  48,  64,  80,  96,  112, 128,
                                                    144, 160, 176, 192, 208, 224, 240, 256};
inline int32_t ConvertValueToIndexNN(int32_t val, int32_t idxBound)
{
    return (val > PP_NN[idxBound]) ? idxBound : (val / 16 - 1);
}

int32_t GetKvFactor(ContextParamsForMLA& contextParams, int32_t kvSeqlen)
{
    auto kFirstDimVal = contextParams.batch;
    int32_t kvFactor = (static_cast<int64_t>(contextParams.batch) * static_cast<int64_t>(contextParams.maxKvSeqLen)
        == kFirstDimVal) ? contextParams.maxKvSeqLen : kvSeqlen;
    return kvFactor;
}

void PrefillTilingHead(ContextParamsForMLA& contextParams, const float tor, AddrOffsets &addrOffsets,
    MLAPrefillTilingData& tilingData, int32_t kvRealHeads)
{
    tilingData.MLAPrefillHead.set_batch(static_cast<uint32_t>(contextParams.batch));
    tilingData.MLAPrefillHead.set_maxSeqLen(static_cast<uint32_t>(contextParams.maxSeqLen));
    tilingData.MLAPrefillHead.set_numHeads(static_cast<uint32_t>(contextParams.numHeads));
    tilingData.MLAPrefillHead.set_embeddingSize(static_cast<uint32_t>(contextParams.embeddingSize));
    tilingData.MLAPrefillHead.set_kvRealHeads(static_cast<uint32_t>(kvRealHeads));
    tilingData.MLAPrefillHead.set_tor(tor);
    tilingData.MLAPrefillHead.set_headStride(static_cast<uint32_t>(contextParams.headStride));
    tilingData.MLAPrefillHead.set_maskStride(static_cast<uint32_t>(contextParams.maskStride));
    tilingData.MLAPrefillHead.set_totalQBlkNum(static_cast<uint32_t>(addrOffsets.totalQBlkNum));
    tilingData.MLAPrefillHead.set_headSize(static_cast<uint32_t>(TILING_HEAD_SIZE_PREFILL));
    tilingData.MLAPrefillHead.set_paraSize(static_cast<uint32_t>(TILING_PARA_SIZE_PREFILL));
    tilingData.MLAPrefillHead.set_maxKvSeqLen(contextParams.maxKvSeqLen);
    tilingData.MLAPrefillHead.set_maskType(contextParams.maskType);
    tilingData.MLAPrefillHead.set_embeddingSizeV(static_cast<uint32_t>(contextParams.embeddingSizeV));
    tilingData.MLAPrefillHead.set_maxKvSeqLenBNSD(contextParams.maxKvSeqLen); // for bnsd, not used
    tilingData.MLAPrefillHead.set_windowSize(contextParams.windowSize);
}

ge::graphStatus PrefillTilingParam(ContextParamsForMLA& contextParams, float tor, AddrOffsets &addrOffsets,
    int32_t kvRealHeads,MLAPrefillTilingData& tilingData)
{
    std::vector<uint32_t> qSeqLenList(50,0);
    std::vector<uint32_t> kvSeqlenList(50,0);
    std::vector<uint32_t> mUbdList(50,0);
    std::vector<uint32_t> indexNNList(50,0);
    std::vector<uint32_t> qSeqHighList(50,0);
    std::vector<uint32_t> qSeqLowList(50,0);
    std::vector<uint32_t> kSeqHighList(50,0);
    std::vector<uint32_t> kSeqLowList(50,0);
    std::vector<uint32_t> vSeqHighList(50,0);
    std::vector<uint32_t> vSeqLowList(50,0);
    std::vector<uint32_t> oSeqHighList(50,0);
    std::vector<uint32_t> oSeqLowList(50,0);
    std::vector<uint32_t> totalQBlkNumList(50,0);

    for (int32_t seqIdx = 0; seqIdx < contextParams.batch; seqIdx++) {
        int32_t qSeqlen = *(contextParams.qSeqLen + seqIdx);
        int32_t qSeqlenAligned = (qSeqlen + 16 - 1) / 16 * 16;
        int32_t kvSeqlen = *(contextParams.kvSeqLen + seqIdx);
        int32_t embeddingSizeAligned = (contextParams.embeddingSize + 16 - 1) / 16 * 16;
        int32_t nIbd = 0;

        int32_t kvSeqlenAligned = (kvSeqlen + 16 - 1) / 16 * 16;
        int32_t nUbd = std::min(LONG_SEQ_LEN, kvSeqlenAligned);
        nIbd = ConvertValueToIndexNN(nUbd, PP_NN_NUM - 1);
        indexNNList[seqIdx] = static_cast<uint32_t>(PP_NN[nIbd]);
    
        int32_t mUbd = std::min(LONG_SEQ_LEN, qSeqlenAligned);
        int32_t mIbd = ConvertValueToIndexMM(mUbd, 8 - 1);
        mUbd = PP_MM[mIbd];
        addrOffsets.totalQBlkNum += (qSeqlen != 0 && kvSeqlen != 0) ? ((qSeqlen + mUbd - 1) / mUbd) : 0;

        mUbdList[seqIdx] = static_cast<uint32_t>(mUbd);;
        qSeqLenList[seqIdx] = static_cast<uint32_t>(qSeqlen);
        kvSeqlenList[seqIdx] = static_cast<uint32_t>(kvSeqlen);
        qSeqHighList[seqIdx] = GetHigh32Bit(addrOffsets.addrQSeqOffset);
        qSeqLowList[seqIdx] = GetLoww32Bit(addrOffsets.addrQSeqOffset);
        kSeqHighList[seqIdx] = GetHigh32Bit(addrOffsets.addrKSeqOffset);
        kSeqLowList[seqIdx] = GetLoww32Bit(addrOffsets.addrKSeqOffset);
        vSeqHighList[seqIdx] = GetHigh32Bit(addrOffsets.addrVSeqOffset);
        vSeqLowList[seqIdx] = GetLoww32Bit(addrOffsets.addrVSeqOffset);
        oSeqHighList[seqIdx] = GetHigh32Bit(addrOffsets.addrOSeqOffset);
        oSeqLowList[seqIdx] = GetLoww32Bit(addrOffsets.addrOSeqOffset);
        totalQBlkNumList[seqIdx] = static_cast<uint32_t>(addrOffsets.totalQBlkNum);

        auto kvFactor = GetKvFactor(contextParams, kvSeqlen);
        addrOffsets.addrQSeqOffset += static_cast<uint64_t>(qSeqlen) * contextParams.numHeads * contextParams.embeddingSizeV;
        addrOffsets.addrKSeqOffset += static_cast<uint64_t>(kvFactor) * kvRealHeads * contextParams.embeddingSizeV;
        addrOffsets.addrVSeqOffset += static_cast<uint64_t>(kvFactor) * kvRealHeads * contextParams.embeddingSizeV;
        addrOffsets.addrOSeqOffset += static_cast<uint64_t>(qSeqlen) * contextParams.numHeads * contextParams.embeddingSizeV;
    }
    tilingData.prefillTiling.set_qSeqLen(qSeqLenList.data());
    tilingData.prefillTiling.set_kvSeqlen(kvSeqlenList.data());
    tilingData.prefillTiling.set_mUbd(mUbdList.data());
    tilingData.prefillTiling.set_indexNN(indexNNList.data());
    tilingData.prefillTiling.set_qSeqHigh(qSeqHighList.data());
    tilingData.prefillTiling.set_qSeqLow(qSeqLowList.data());
    tilingData.prefillTiling.set_kSeqHigh(kSeqHighList.data());
    tilingData.prefillTiling.set_kSeqLow(kSeqLowList.data());
    tilingData.prefillTiling.set_vSeqHigh(vSeqHighList.data());
    tilingData.prefillTiling.set_vSeqLow(vSeqLowList.data());
    tilingData.prefillTiling.set_oSeqHigh(oSeqHighList.data());
    tilingData.prefillTiling.set_oSeqLow(oSeqLowList.data());
    tilingData.prefillTiling.set_totalQBlkNum(totalQBlkNumList.data());
    PrefillTilingHead(contextParams, tor, addrOffsets, tilingData, kvRealHeads);
    addrOffsets.block = static_cast<int64_t>(contextParams.numHeads) * addrOffsets.totalQBlkNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetMLAPrefillTilingParam(ContextParamsForMLA& contextParams, uint32_t& blockDimToBeSet, MLAPrefillTilingData& tilingData)
{
    AddrOffsets addrOffsets;
    float tor = contextParams.tor;
    int32_t kvRealHeads = contextParams.kvHeads > 0 ? contextParams.kvHeads : contextParams.numHeads;
    auto ret1 = PrefillTilingParam(contextParams, tor, addrOffsets, kvRealHeads, tilingData);
    if(ret1 == ge::GRAPH_FAILED) return ge::GRAPH_FAILED;

    blockDimToBeSet = contextParams.aicNum;
    if(addrOffsets.block < 0 || static_cast<uint32_t>(addrOffsets.block) > UINT32_MAX) return ge::GRAPH_FAILED;
    uint32_t logicalblock = static_cast<uint32_t>(addrOffsets.block);
    if (blockDimToBeSet > logicalblock) {
        blockDimToBeSet = logicalblock;
    }
    tilingData.MLAPrefillHead.set_blockDim(static_cast<uint32_t>(blockDimToBeSet));
    return ge::GRAPH_SUCCESS;
}

size_t GetMLAPrefillWorkSpaceSize(ContextParamsForMLA& contextParams, uint32_t& blockDim)
{
    size_t sysWorkspaceSize, workspaceSize;
    const uint64_t defaultSysWorkspaceSize910B = 16U * 1024U * 1024U;
    sysWorkspaceSize = defaultSysWorkspaceSize910B;

    uint64_t dataLenFloat = sizeof(float);
    uint64_t sSize = static_cast<uint64_t>(blockDim) * static_cast<uint64_t>(32768) * 16 * dataLenFloat;
    workspaceSize = sysWorkspaceSize + sSize + sSize + sSize + sSize * 2;
    // kernelInfo.GetScratchSizes() = {sSize, sSize, sSize, sSize * 2}; // oTmp/S/P
    return workspaceSize;
}

}