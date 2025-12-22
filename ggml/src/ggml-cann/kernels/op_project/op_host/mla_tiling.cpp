#include "mla_tiling.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include <securec.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

#define UNUSED_VALUE(x) (void)(x)

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {

constexpr std::array<int32_t, 8> PP_MM = {16, 32, 48, 64, 80, 96, 112, 128};
constexpr std::array<int32_t, 6> QN_TILE_LIST = {128, 64, 32, 16, 8, 1};

using IndexArr = std::array<int32_t, 4>;

inline uint32_t GetHigh32Bit(uint64_t v) {
    return static_cast<uint32_t>(v >> 32);
}
inline uint32_t GetLow32Bit(uint64_t v) { return static_cast<uint32_t>(v); }

inline int32_t ConvertValueToIndexMM(int32_t val, int32_t idxBound)  // 16, 7
{
    return (val > PP_MM[idxBound]) ? idxBound : (val / 16 - 1);
}
const int32_t PP_NN_NUM = 16;
constexpr std::array<int32_t, PP_NN_NUM> PP_NN = {
    16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256};
inline int32_t ConvertValueToIndexNN(int32_t val, int32_t idxBound) {
    return (val > PP_NN[idxBound]) ? idxBound : (val / 16 - 1);
}

void GetDecodingNormalTaskTiling(ContextParamsForMLA& contextParams,
                                 int32_t& curTask, int32_t& prevTaskNum,
                                 std::vector<uint32_t>& normalBatchIdx,
                                 std::vector<uint32_t>& normalBeginQ,
                                 std::vector<uint32_t>& normalCurKvSeq,
                                 std::vector<uint32_t>& normalCurQLen) {
    int32_t maxQPerJob = contextParams.quantFlag ? 1 : 2;
    int32_t qRowIdx = 0;
    for (int32_t taskIdx = 0; taskIdx < contextParams.batch; taskIdx++) {
        int32_t batchIdx = contextParams.batchList[taskIdx].batchIdx;
        int32_t qSeqLen = contextParams.qSeqLen == nullptr
                              ? 1
                              : *(contextParams.qSeqLen + batchIdx);
        int32_t kvSeqlen = *(contextParams.kvSeqLen + batchIdx);
        int32_t curKvSeq = kvSeqlen - qSeqLen;
        if (prevTaskNum >= contextParams.normalTaskNum) {
            curTask = taskIdx;
            break;
        }
        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq += maxQPerJob) {
            int32_t curQLen =
                ((qSeqLen - qSeq) > maxQPerJob) ? maxQPerJob : (qSeqLen - qSeq);
            curKvSeq += curQLen;
            normalBatchIdx[prevTaskNum] = static_cast<uint32_t>(batchIdx);
            normalBeginQ[prevTaskNum] = static_cast<uint32_t>(batchIdx) *
                                            static_cast<uint32_t>(qSeqLen) +
                                        static_cast<uint32_t>(qSeq);
            normalCurKvSeq[prevTaskNum] = static_cast<uint32_t>(curKvSeq);
            normalCurQLen[prevTaskNum] = static_cast<uint32_t>(curQLen);
            prevTaskNum++;
            qRowIdx += curQLen;
        }
    }
}

ge::graphStatus GetNdMLADecodingMtpTilingTP1(ContextParamsForMLA& contextParams,
                                             MLATilingData& tilingData) {
    int32_t curTask = 0;
    int32_t prevTaskNum = 0;
    std::vector<uint32_t> batchIdxList(50, 0);
    std::vector<uint32_t> qSeqLenList(50, 0);
    std::vector<uint32_t> kvBlockIdxList(50, 0);
    std::vector<uint32_t> prevKVSeqlenList(50, 0);
    std::vector<uint32_t> batchSizeList(50, 0);
    std::vector<uint32_t> splitNumList(50, 0);
    std::vector<uint32_t> prevSplitNumSumList(50, 0);
    GetDecodingNormalTaskTiling(contextParams, curTask, prevTaskNum,
                                batchIdxList, qSeqLenList, kvBlockIdxList,
                                prevKVSeqlenList);
    tilingData.ndMLADecodingMtpTilingTP1.set_tailBatch(
        static_cast<uint32_t>(contextParams.batch - curTask));

    std::unordered_map<int, int> batchTilingMap;
    for (int32_t taskIdx = curTask; taskIdx < contextParams.batch; taskIdx++) {
        int32_t batchIdx = contextParams.batchList[taskIdx].batchIdx;
        int32_t qSeqLen = contextParams.qSeqLen == nullptr
                              ? 1
                              : *(contextParams.qSeqLen + batchIdx);
        int32_t qLoop = contextParams.quantFlag ? qSeqLen : 1;
        int32_t splitNum = contextParams.splitKVNum;
        for (int32_t qSeq = 0; qSeq < qLoop; qSeq++) {
            int32_t nowKVSeqlen =
                contextParams.batchList[taskIdx].kvSeqlen - qLoop + qSeq + 1;
            int32_t kvBlocks = (nowKVSeqlen + contextParams.blockSize - 1) /
                               contextParams.blockSize;
            if (kvBlocks < splitNum) {
                splitNum = kvBlocks;
            }
            int32_t prevKVSeqlen = 0;
            int32_t splitBlocks = (kvBlocks + splitNum - 1) / splitNum;
            if (qSeqLen == 2 && splitBlocks == 1 &&
                nowKVSeqlen % contextParams.blockSize == 1) {
                splitNum--;
                splitBlocks = (kvBlocks + splitNum - 1) / splitNum;
            }
            int32_t resBlocks = (splitNum - kvBlocks % splitNum) % splitNum;
            batchTilingMap[batchIdx * qSeqLen + qSeq] = prevTaskNum;
            for (int kvBlockIdx = 0; kvBlockIdx < splitNum; kvBlockIdx++) {
                int32_t nowBlocks =
                    kvBlockIdx < resBlocks ? splitBlocks - 1 : splitBlocks;
                batchIdxList[prevTaskNum] = static_cast<uint32_t>(batchIdx);
                qSeqLenList[prevTaskNum] = static_cast<uint32_t>(qSeqLen);
                kvBlockIdxList[prevTaskNum] =
                    kvBlockIdx == splitNum - 1
                        ? static_cast<uint32_t>(nowKVSeqlen - prevKVSeqlen)
                        : static_cast<uint32_t>(nowBlocks) *
                              static_cast<uint32_t>(contextParams.blockSize);
                prevKVSeqlenList[prevTaskNum] =
                    static_cast<uint32_t>(prevKVSeqlen);
                batchSizeList[prevTaskNum] =
                    contextParams.quantFlag
                        ? static_cast<uint32_t>(batchIdx) *
                                  static_cast<uint32_t>(qSeqLen) +
                              static_cast<uint32_t>(qSeq)
                        : static_cast<uint32_t>(batchIdx) *
                              static_cast<uint32_t>(qSeqLen);
                splitNumList[prevTaskNum] = static_cast<uint32_t>(splitNum);
                prevSplitNumSumList[prevTaskNum] =
                    static_cast<uint32_t>(contextParams.prevSplitNumSum);
                prevKVSeqlen += kvBlockIdxList[prevTaskNum];
                prevTaskNum++;
            }
            contextParams.prevSplitNumSum += splitNum;
        }
    }

    // GetDecodingTailBatchTiling
    for (int32_t taskIdx = curTask; taskIdx < contextParams.batch; taskIdx++) {
        int32_t batchIdx = contextParams.batchList[taskIdx].batchIdx;
        int32_t qSeqLen = contextParams.qSeqLen == nullptr
                              ? 1
                              : *(contextParams.qSeqLen + batchIdx);
        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq++) {
            uint32_t batchTilingOffset =
                contextParams.quantFlag
                    ? static_cast<uint32_t>(
                          batchTilingMap[batchIdx * qSeqLen + qSeq])
                    : static_cast<uint32_t>(batchTilingMap[batchIdx * qSeqLen]);
            uint32_t qOffset =
                contextParams.quantFlag ? 0 : static_cast<uint32_t>(qSeq);
            if (((taskIdx - curTask) * qSeqLen + qSeq) * 2 == 0) {
                batchIdxList[prevTaskNum] = batchTilingOffset;
                qSeqLenList[prevTaskNum] = qOffset;
            } else if (((taskIdx - curTask) * qSeqLen + qSeq) * 2 == 2) {
                kvBlockIdxList[prevTaskNum] = batchTilingOffset;
                prevKVSeqlenList[prevTaskNum] = qOffset;
            } else if (((taskIdx - curTask) * qSeqLen + qSeq) * 2 == 4) {
                batchSizeList[prevTaskNum] = batchTilingOffset;
                splitNumList[prevTaskNum] = qOffset;
            }
        }
    }
    tilingData.ndMLADecodingMtpTilingTP1.set_batchIdx(batchIdxList.data());
    tilingData.ndMLADecodingMtpTilingTP1.set_qSeqLen(qSeqLenList.data());
    tilingData.ndMLADecodingMtpTilingTP1.set_kvBlockIdx(kvBlockIdxList.data());
    tilingData.ndMLADecodingMtpTilingTP1.set_prevKVSeqlen(
        prevKVSeqlenList.data());
    tilingData.ndMLADecodingMtpTilingTP1.set_batchSize(batchSizeList.data());
    tilingData.ndMLADecodingMtpTilingTP1.set_splitNum(splitNumList.data());
    tilingData.ndMLADecodingMtpTilingTP1.set_prevSplitNumSum(
        prevSplitNumSumList.data());
    return ge::GRAPH_SUCCESS;
}

void GetNdMLAMtpTilingTP1(ContextParamsForMLA& contextParams,
                          uint32_t& blockDimToBeSet,
                          MLATilingData& tilingData) {
    bool isFP16 = static_cast<int32_t>(contextParams.type) < 2;
    int32_t maxQPerJob = isFP16 ? 2 : 1;
    int32_t prevTaskNum = 0;
    int32_t totalTaskNum = contextParams.totalTaskNum;
    std::vector<uint32_t> batchIdxList(50, 0);
    std::vector<uint32_t> beginQList(50, 0);
    std::vector<uint32_t> curKvSeqList(50, 0);
    std::vector<uint32_t> curQLenList(50, 0);
    for (int32_t seqIdx = 0; seqIdx < contextParams.batch; seqIdx++) {
        int32_t qSeqLen = contextParams.qSeqLen == nullptr
                              ? 1
                              : *(contextParams.qSeqLen + seqIdx);
        int32_t batchIdx = contextParams.batchList[seqIdx].batchIdx;
        int32_t kvSeqlen = contextParams.batchList[seqIdx].kvSeqlen;
        int32_t beginQ = contextParams.batchList[seqIdx].startQIdx;
        int32_t curKvSeq = kvSeqlen - qSeqLen;
        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq += maxQPerJob) {
            if (totalTaskNum <= static_cast<int32_t>(blockDimToBeSet) &&
                (prevTaskNum % static_cast<int32_t>(blockDimToBeSet) == 0)) {
                maxQPerJob = 1;
            }
            int32_t curQLen =
                ((qSeqLen - qSeq) > maxQPerJob) ? maxQPerJob : (qSeqLen - qSeq);
            curKvSeq += curQLen;
            batchIdxList[prevTaskNum] = static_cast<uint32_t>(batchIdx);
            beginQList[prevTaskNum] =
                static_cast<uint32_t>(beginQ) + static_cast<uint32_t>(qSeq);
            curKvSeqList[prevTaskNum] = static_cast<uint32_t>(curKvSeq);
            curQLenList[prevTaskNum] = static_cast<uint32_t>(curQLen);
            prevTaskNum++;
            totalTaskNum -= curQLen;
        }
    }
    tilingData.ndMLADecodingMtpTilingTP1.set_batchIdx(batchIdxList.data());
    tilingData.ndMLADecodingMtpTilingTP1.set_qSeqLen(beginQList.data());
    tilingData.ndMLADecodingMtpTilingTP1.set_kvBlockIdx(curKvSeqList.data());
    tilingData.ndMLADecodingMtpTilingTP1.set_prevKVSeqlen(curQLenList.data());
}

int32_t GetQNBlockTile(ContextParamsForMLA& contextParams, int32_t qSeqLen) {
    int32_t tileListIdx = static_cast<int32_t>(std::ceil(std::log2(qSeqLen)));
    tileListIdx = (tileListIdx > 5) ? 5 : tileListIdx;
    int32_t qNBlockTile = QN_TILE_LIST[tileListIdx];
    int32_t group = contextParams.numHeads / contextParams.kvHeads;
    qNBlockTile = (qNBlockTile > group) ? group : qNBlockTile;

    return qNBlockTile;
}

int32_t GetMaxQseqlen(ContextParamsForMLA& contextParams) {
    if (contextParams.qSeqLen == nullptr) return 1;
    auto maxQSeqlenIter = std::max_element(
        contextParams.qSeqLen, contextParams.qSeqLen + contextParams.batch);
    auto maxQseqlen =
        maxQSeqlenIter != contextParams.qSeqLen + contextParams.batch
            ? *maxQSeqlenIter
            : 1;
    return maxQseqlen;
}

int32_t GetMaxKVseqlen(ContextParamsForMLA& contextParams) {
    auto maxKVSeqlenIter =
        std::max_element(contextParams.kvSeqLen,
                         contextParams.kvSeqLen + contextParams.numTokens);
    auto maxKVseqlen =
        maxKVSeqlenIter != contextParams.kvSeqLen + contextParams.numTokens
            ? *maxKVSeqlenIter
            : 1;
    return maxKVseqlen;
}

ge::graphStatus GetNdMLATiling(ContextParamsForMLA& contextParams,
                               MLATilingData& tilingData) {
    AddrOffsets addrOffsets{};
    auto qSeqLen = contextParams.qSeqLen;
    int32_t maxQseqlen = GetMaxQseqlen(contextParams);
    if (maxQseqlen <= 0) return ge::GRAPH_FAILED;

    int32_t maxKVseqlen = GetMaxKVseqlen(contextParams);
    if (maxKVseqlen <= 0) return ge::GRAPH_FAILED;

    int32_t curQNBlockTile = GetQNBlockTile(contextParams, maxQseqlen);

    uint32_t emptySeq = (contextParams.qSeqLen == nullptr) ? 1 : 0;

    std::vector<uint32_t> qSeqLenList(50, 0);
    std::vector<uint32_t> kvSeqlenList(50, 0);
    std::vector<uint32_t> qSeqHighList(50, 0);
    std::vector<uint32_t> qSeqLowList(50, 0);
    std::vector<uint32_t> oSeqHighList(50, 0);
    std::vector<uint32_t> oSeqLowList(50, 0);
    std::vector<uint32_t> maskHighList(50, 0);
    std::vector<uint32_t> maskLowList(50, 0);

    for (int32_t seqIdx = 0; seqIdx < contextParams.batch; seqIdx++) {
        int32_t qSeqLen = 1;
        qSeqLen = (emptySeq == 1) ? 1 : *(contextParams.qSeqLen + seqIdx);
        qSeqLen = (*(contextParams.kvSeqLen + seqIdx) == 0) ? 0 : qSeqLen;
        int32_t kvSeqlen = *(contextParams.kvSeqLen + seqIdx);

        qSeqLenList[seqIdx] = static_cast<uint32_t>(qSeqLen);
        kvSeqlenList[seqIdx] = static_cast<uint32_t>(kvSeqlen);
        qSeqHighList[seqIdx] = GetHigh32Bit(addrOffsets.addrQSeqOffset);
        qSeqLowList[seqIdx] = GetLow32Bit(addrOffsets.addrQSeqOffset);
        oSeqHighList[seqIdx] = GetHigh32Bit(addrOffsets.addrOSeqOffset);
        oSeqLowList[seqIdx] = GetLow32Bit(addrOffsets.addrOSeqOffset);
        maskHighList[seqIdx] = GetHigh32Bit(addrOffsets.addrMaskOffset);
        maskLowList[seqIdx] = GetLow32Bit(addrOffsets.addrMaskOffset);
        uint64_t addressQffset =
            static_cast<uint64_t>(contextParams.numHeads * qSeqLen);
        uint64_t addressOffset = static_cast<uint64_t>(
            contextParams.numHeads * contextParams.embeddingSize * qSeqLen);
        uint64_t addressMaskOffset =
            static_cast<uint64_t>(qSeqLen * maxKVseqlen);
        addrOffsets.addrQSeqOffset += addressQffset;
        addrOffsets.addrOSeqOffset += addressOffset;
        addrOffsets.addrMaskOffset += addressMaskOffset;
    }
    tilingData.ndMLATiling.set_curQNBlockTile(
        static_cast<uint32_t>(curQNBlockTile));
    tilingData.ndMLATiling.set_maxKVseqlen(static_cast<uint32_t>(maxKVseqlen));
    tilingData.ndMLATiling.set_qSeqLen(qSeqLenList.data());
    tilingData.ndMLATiling.set_kvSeqlen(kvSeqlenList.data());
    tilingData.ndMLATiling.set_qSeqHigh(qSeqHighList.data());
    tilingData.ndMLATiling.set_qSeqLow(qSeqLowList.data());
    tilingData.ndMLATiling.set_oSeqHigh(oSeqHighList.data());
    tilingData.ndMLATiling.set_oSeqLow(oSeqLowList.data());
    tilingData.ndMLATiling.set_maskHigh(maskHighList.data());
    tilingData.ndMLATiling.set_maskLow(maskLowList.data());
    return ge::GRAPH_SUCCESS;
}

void GetTilingHead(ContextParamsForMLA& contextParams,
                   uint32_t& blockDimToBeSet, MLATilingData& tilingData) {
    tilingData.MLAHead.set_batch(static_cast<uint32_t>(contextParams.batch));
    tilingData.MLAHead.set_headSize(static_cast<uint32_t>(18));
    tilingData.MLAHead.set_paraSize(contextParams.mtpTp1Flag
                                        ? static_cast<uint32_t>(7)
                                        : static_cast<uint32_t>(8));
    tilingData.MLAHead.set_numHeads(
        static_cast<uint32_t>(contextParams.numHeads));
    tilingData.MLAHead.set_headDim(
        static_cast<uint32_t>(contextParams.embeddingSize));
    tilingData.MLAHead.set_numBlocks(
        static_cast<uint32_t>(contextParams.numBlocks));
    tilingData.MLAHead.set_blockSize(
        static_cast<uint32_t>(contextParams.blockSize));
    tilingData.MLAHead.set_maxBlocks(
        static_cast<uint32_t>(contextParams.maxNumBlocksPerQuery));
    tilingData.MLAHead.set_tor(contextParams.tor);
    tilingData.MLAHead.set_kvHeads((contextParams.kvHeads == 0)
                                       ? contextParams.numHeads
                                       : contextParams.kvHeads);
    tilingData.MLAHead.set_maskTypeND(
        static_cast<uint32_t>(contextParams.maskType));

    if (contextParams.flashDecoding) {
        tilingData.MLAHead.set_taskNum(
            static_cast<uint32_t>(contextParams.normalTaskNum));
        tilingData.MLAHead.set_kvSplitNum(
            static_cast<uint32_t>(contextParams.splitKVNum));
        tilingData.MLAHead.set_splitTaskNum(
            static_cast<uint32_t>(contextParams.prevSplitNumSum));
    } else {
        tilingData.MLAHead.set_taskNum(
            static_cast<uint32_t>(contextParams.totalTaskNum));
    }

    tilingData.MLAHead.set_blockDim(static_cast<uint32_t>(blockDimToBeSet));
}

ge::graphStatus GetMLATilingParam(ContextParamsForMLA& contextParams,
                                  uint32_t& blockDimToBeSet,
                                  MLATilingData& tilingData) {
    blockDimToBeSet = contextParams.aicNum;
    if (contextParams.mtpTp1Flag) {
        if (contextParams.flashDecoding) {
            GetNdMLADecodingMtpTilingTP1(contextParams, tilingData);
        } else {
            GetNdMLAMtpTilingTP1(contextParams, blockDimToBeSet, tilingData);
        }
    } else {
        GetNdMLATiling(contextParams, tilingData);
        blockDimToBeSet = contextParams.batch == 32 ? 20 : blockDimToBeSet;
    }
    GetTilingHead(contextParams, blockDimToBeSet, tilingData);
    return ge::GRAPH_SUCCESS;
}

size_t GetMLAWorkSpaceSize(ContextParamsForMLA& contextParams,
                           uint32_t& blockDim) {
    size_t sysWorkspaceSize, workspaceSize;
    const uint64_t defaultSysWorkspaceSize910B = 16U * 1024U * 1024U;
    sysWorkspaceSize = defaultSysWorkspaceSize910B;

    uint32_t dataLenHalf = sizeof(uint16_t);
    uint32_t dataLenFloat = sizeof(float);
    uint32_t dataLenInt = sizeof(int32_t);
    uint64_t basicWorkSpaceHalf =
        blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenHalf;
    uint64_t basicWorkSpaceFloat =
        blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenFloat;
    uint64_t basicWorkSpaceInt8 =
        blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenInt;
    uint64_t oCoreWorkSpaceSize =
        contextParams.flashDecoding && contextParams.mtpTp1Flag
            ? WORKSPACE_BLOCK_SIZE_DB * contextParams.flashDecodingTaskNum *
                  contextParams.splitKVNum * dataLenFloat * 2
            : 0;
    uint64_t lWorkSpaceSize =
        contextParams.flashDecoding && contextParams.mtpTp1Flag
            ? contextParams.numHeads * contextParams.flashDecodingTaskNum *
                  contextParams.splitKVNum * dataLenFloat * 2 * 8
            : 0;
    if (contextParams.quantFlag) {
        uint64_t sWorkSpaceSize = contextParams.mtpTp1Flag
                                      ? basicWorkSpaceFloat * 2
                                      : basicWorkSpaceFloat;
        uint64_t pWorkSpaceSize = basicWorkSpaceInt8;
        uint64_t oTempWorkSpaceSize = basicWorkSpaceInt8 * 2;
        workspaceSize = sysWorkspaceSize + sWorkSpaceSize * 2 + pWorkSpaceSize +
                        oTempWorkSpaceSize + basicWorkSpaceFloat +
                        oCoreWorkSpaceSize + lWorkSpaceSize;
    } else {
        uint64_t sWorkSpaceSize = contextParams.mtpTp1Flag
                                      ? basicWorkSpaceFloat * 4
                                      : basicWorkSpaceFloat * 2;
        uint64_t pWorkSpaceSize = contextParams.mtpTp1Flag
                                      ? basicWorkSpaceHalf * 4
                                      : basicWorkSpaceHalf * 2;
        uint64_t oTempWorkSpaceSize = contextParams.mtpTp1Flag
                                          ? basicWorkSpaceFloat * 4
                                          : basicWorkSpaceFloat * 2;
        uint64_t goWorkSpaceSize = contextParams.mtpTp1Flag
                                       ? basicWorkSpaceFloat * 2
                                       : basicWorkSpaceFloat;
        workspaceSize = sysWorkspaceSize + sWorkSpaceSize + 512 +
                        pWorkSpaceSize + oTempWorkSpaceSize + goWorkSpaceSize +
                        oCoreWorkSpaceSize + lWorkSpaceSize;
    }
    return workspaceSize;
}

}  // namespace optiling