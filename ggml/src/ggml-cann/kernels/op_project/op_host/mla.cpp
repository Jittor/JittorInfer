#include "cstdio"
#include <numeric>
#include <algorithm>
#include <functional>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include "register/op_def_registry.h"
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "../utils/inc/log/ops_log.h"
#include "mla_tiling.h"

namespace optiling {

int32_t CalcSplitNum(ContextParamsForMLA& contextKeyParams, int32_t blockDim, int32_t minKVSeqlen, int32_t blockSize)
{
    if (blockDim - contextKeyParams.flashDecodingTaskNum <= 4 || contextKeyParams.quantFlag) {
        return 1;
    }
    if (blockSize == 0 || blockDim == 0) {
        return 1;
    }
    int32_t minKVBlocks = (minKVSeqlen + blockSize - 1) / blockSize;
    for (int32_t splitNum = 2; splitNum <= 6; splitNum++) {
        if ((contextKeyParams.flashDecodingTaskNum * splitNum) % blockDim != 0) {
            continue;
        }
        int32_t repeatTimesPerBlock = contextKeyParams.flashDecodingTaskNum * splitNum / blockDim;
        if (minKVSeqlen / splitNum >= blockSize &&
            repeatTimesPerBlock + 6 <= minKVBlocks - (minKVBlocks + splitNum - 1) / splitNum * repeatTimesPerBlock) {
            return splitNum;
        } else {
            return 1;
        }
    }
    return 1;
}

ge::graphStatus BatchSeqSort(ContextParamsForMLA& contextKeyParams, uint32_t endIndex, uint32_t sortDim)
{
    if (sortDim == 0 || endIndex <= 0) {
        return ge::GRAPH_SUCCESS;
    }
    std::sort(contextKeyParams.batchList.begin(), contextKeyParams.batchList.end());
    std::reverse(contextKeyParams.batchList.begin(), contextKeyParams.batchList.begin() + endIndex);
    uint32_t batchSortInfo = (contextKeyParams.batch + sortDim - 1) / sortDim;
    for (uint32_t sortIdx = 0; sortIdx < batchSortInfo; sortIdx++) {
        uint32_t startIndex = sortIdx * sortDim;
        uint32_t sortEnd = std::min(startIndex + sortDim, endIndex);
        if (sortIdx % 2 == 0) {
            std::sort(contextKeyParams.batchList.begin() + startIndex, contextKeyParams.batchList.begin() + sortEnd);
        } else {
            std::sort(contextKeyParams.batchList.begin() + startIndex,
                        contextKeyParams.batchList.begin() + sortEnd, std::greater<BatchNode>());
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetFlashDecodingInfo(ContextParamsForMLA& contextKeyParams, uint32_t blockDim)
{
    if (blockDim == 0) {
        return ge::GRAPH_FAILED;
    }
    contextKeyParams.tailBatch = contextKeyParams.batch % blockDim;
    contextKeyParams.tailTaskNum = contextKeyParams.totalTaskNum % blockDim;
    contextKeyParams.flashDecodingTaskNum = contextKeyParams.quantFlag ? contextKeyParams.tailTaskNum : contextKeyParams.tailBatch;
    auto minKVSeqlen = std::min_element(contextKeyParams.kvSeqLen, contextKeyParams.kvSeqLen + contextKeyParams.kvSeqLength);
    auto minQSeqlen = contextKeyParams.qSeqLen != nullptr ? *std::min_element(contextKeyParams.qSeqLen, contextKeyParams.qSeqLen + contextKeyParams.batch) : 1;
    auto maxQSeqlen = contextKeyParams.qSeqLen != nullptr ? *std::max_element(contextKeyParams.qSeqLen, contextKeyParams.qSeqLen + contextKeyParams.batch) : 1;
    contextKeyParams.flashDecoding = !contextKeyParams.quantFlag && contextKeyParams.flashDecodingTaskNum != 0 &&
                            *minKVSeqlen >= 2048 &&
                            ((minQSeqlen == 2 && maxQSeqlen == 2) ||
                            (minQSeqlen == 1 && maxQSeqlen == 1));
    if (!contextKeyParams.flashDecoding) {
        if(BatchSeqSort(contextKeyParams, contextKeyParams.batch, blockDim) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    if(BatchSeqSort(contextKeyParams, contextKeyParams.batch - contextKeyParams.flashDecodingTaskNum, blockDim) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    contextKeyParams.splitKVNum = blockDim / contextKeyParams.flashDecodingTaskNum > 1 ?  blockDim / contextKeyParams.flashDecodingTaskNum :
                        CalcSplitNum(contextKeyParams, blockDim, *minKVSeqlen, contextKeyParams.blockSize);
    contextKeyParams.flashDecoding = contextKeyParams.splitKVNum == 1 ? false : true;
    int32_t taskNum = contextKeyParams.quantFlag ? contextKeyParams.totalTaskNum : contextKeyParams.batch;
    contextKeyParams.normalTaskNum = taskNum / blockDim * blockDim;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToMLAParams(gert::TilingContext* context, ContextParamsForMLA& contextKeyParams)
{
    auto kcacheShape = context->GetInputShape(2)->GetOriginShape();
    auto KDims = kcacheShape.GetDimNum();
    auto tableShape = context->GetInputShape(4)->GetOriginShape();
    contextKeyParams.kNz = (kcacheShape.GetDim(KDims - 1) == 16 || kcacheShape.GetDim(KDims - 1) == 32) ? 1 : 0;
    if (contextKeyParams.kNz) {
        contextKeyParams.embeddingSize = static_cast<int32_t>(kcacheShape.GetDim(3)) *
                            static_cast<int32_t>(kcacheShape.GetDim(1));
        contextKeyParams.blockSize = static_cast<int32_t>(kcacheShape.GetDim(2));
    } else {
        contextKeyParams.embeddingSize = static_cast<int32_t>(kcacheShape.GetDim(3));
        contextKeyParams.blockSize = static_cast<int32_t>(kcacheShape.GetDim(1));
    }

    contextKeyParams.kvSeqLen =  const_cast<int64_t*>(context->GetInputTensor(5)->GetData<int64_t>());
    contextKeyParams.kvSeqLength = context->GetInputTensor(5)->GetShapeSize();
    contextKeyParams.qSeqLen =  context->GetInputTensor(7) != nullptr ? const_cast<int64_t*>(context->GetInputTensor(7)->GetData<int64_t>()) : nullptr;
    contextKeyParams.qSeqLength = context->GetInputTensor(7) != nullptr ? context->GetInputTensor(7)->GetShapeSize() : 0;

    contextKeyParams.numTokens = static_cast<int32_t>(context->GetInputShape(5)->GetOriginShape().GetDim(0));
    contextKeyParams.numBlocks = static_cast<int32_t>(kcacheShape.GetDim(0));
    contextKeyParams.maxNumBlocksPerQuery = static_cast<int32_t>(tableShape.GetDim(1));
    contextKeyParams.batch = contextKeyParams.numTokens;
    auto attrs = context->GetAttrs();
    contextKeyParams.tor =*attrs->GetAttrPointer<float>(1);
    contextKeyParams.numHeads = *attrs->GetAttrPointer<int32_t>(0);
    contextKeyParams.kvHeads = *attrs->GetAttrPointer<int32_t>(2);
    contextKeyParams.maskType = *attrs->GetAttrPointer<int32_t>(3);

    contextKeyParams.quantFlag = (static_cast<int32_t>(contextKeyParams.type) < 2) ? 0 : 1;
    auto maxQSeqlen = contextKeyParams.qSeqLen != nullptr ? *std::max_element(contextKeyParams.qSeqLen, contextKeyParams.qSeqLen + contextKeyParams.qSeqLength) : 1;
    contextKeyParams.mtpTp1Flag = ((contextKeyParams.numHeads == 128) ||
                           (context->GetInputTensor(0)->GetDataType() == ge::DT_INT8 && maxQSeqlen > 1));
    if (contextKeyParams.mtpTp1Flag || static_cast<int32_t>(contextKeyParams.type) >= 2) {
        contextKeyParams.maskType = 0;
    }
    int32_t beginQ = 0;
    for (int32_t batchIdx = 0; batchIdx < contextKeyParams.batch; batchIdx++) {
        contextKeyParams.batchList.push_back(BatchNode(batchIdx, *(contextKeyParams.kvSeqLen + batchIdx), beginQ));
        beginQ = contextKeyParams.qSeqLen == nullptr ? beginQ + 1 : beginQ + *(contextKeyParams.qSeqLen + batchIdx);
    }
    contextKeyParams.totalTaskNum = contextKeyParams.qSeqLen != nullptr ?
                          std::accumulate(contextKeyParams.qSeqLen, contextKeyParams.qSeqLen + contextKeyParams.batch, static_cast<int32_t>(0)) :
                          contextKeyParams.batch;

    contextKeyParams.workspaceSize = context->GetWorkspaceSizes(1);

    auto platformInfoPtr = context->GetPlatformInfo();
    if(platformInfoPtr == nullptr) return ge::GRAPH_FAILED;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    contextKeyParams.aivNum = ascendcPlatform.GetCoreNumAiv();
    contextKeyParams.aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, contextKeyParams.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, contextKeyParams.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, contextKeyParams.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, contextKeyParams.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, contextKeyParams.l0BSize);

    contextKeyParams.defaultSysWorkspaceSize = 0;

    uint32_t blockDim = contextKeyParams.aicNum;
    if (contextKeyParams.mtpTp1Flag) {
        if(GetFlashDecodingInfo(contextKeyParams, blockDim) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetMLATilingKeyTypeBase(gert::TilingContext* context, ContextParamsForMLA &contextParamsForMLA)
{
    if (context->GetInputTensor(0)->GetDataType() == ge::DT_BF16) {
        contextParamsForMLA.type = TilingKeyType::TILING_BF16_DATA;
    } else if (context->GetInputTensor(0)->GetDataType() == ge::DT_FLOAT16) {
        contextParamsForMLA.type = TilingKeyType::TILING_HALF_DATA;
    } else if (context->GetInputTensor(1)->GetDataType() == ge::DT_FLOAT16) {
        contextParamsForMLA.type = TilingKeyType::TILING_INT8_HALF_DATA;
    } else {
        contextParamsForMLA.type = TilingKeyType::TILING_INT8_BF16_DATA;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenMLATilingKey(ContextParamsForMLA &contextParams, gert::TilingContext* context)
{
    uint32_t dataType = static_cast<int32_t>(contextParams.type);
    uint32_t tilingKey = dataType + (contextParams.kNz << 4) + (contextParams.mtpTp1Flag << 2) + (contextParams.flashDecoding << 6);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MLATilingFunc(gert::TilingContext* context) {
    if (context == nullptr) {
        OPS_LOG_E("MLA", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    if (context->GetRawTilingData() == nullptr) {
        OPS_LOG_E("MLA", "tiling context GetRawTilingData is nullptr!");
        return ge::GRAPH_FAILED;
    }
    ContextParamsForMLA contextParamsForMLA;
    GetMLATilingKeyTypeBase(context, contextParamsForMLA);
    ConvertContextToMLAParams(context, contextParamsForMLA);

    MLATilingData tilingData;
    if(memset_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
                0, context->GetRawTilingData()->GetCapacity()) != EOK)
                return ge::GRAPH_FAILED;

    uint32_t blockDimToBeSet;
    if(GetMLATilingParam(contextParamsForMLA, blockDimToBeSet, tilingData) == ge::GRAPH_FAILED) return ge::GRAPH_FAILED;
    size_t* workspaces = contextParamsForMLA.workspaceSize;
    workspaces[0] = GetMLAWorkSpaceSize(contextParamsForMLA, blockDimToBeSet);

    if(GenMLATilingKey(contextParamsForMLA, context) == ge::GRAPH_FAILED) return ge::GRAPH_FAILED;
    context->SetBlockDim(blockDimToBeSet);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {

static ge::graphStatus MLAInferShape(gert::InferShapeContext* context) {
    const gert::Shape* query_nope_shape = context->GetInputShape(0);
    gert::Shape* attentionOutShape = context->GetOutputShape(0);
    *attentionOutShape = *query_nope_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MLAInferDataType(gert::InferDataTypeContext* context) {
  // default set q's dtype as ifa's output type
  const auto outputType = context->GetInputDataType(1);
  context->SetOutputDataType(0, outputType);
  return GRAPH_SUCCESS;
}

}

namespace ops {

class MLA : public OpDef {
public:
    explicit MLA(const char* name) : OpDef(name)
    {
        this->Input("qNope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("qRope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("ctKV")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("kRope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("blockTables")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("contextLens")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("qseqlen")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("qkDescale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("pvDescale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Output("attenOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Attr("headNum").Int();
        this->Attr("qkScale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("kvHeadNum").Int();
        this->Attr("maskType").AttrType(OPTIONAL).Int(0);
        this->Attr("calcType").AttrType(OPTIONAL).Int(0);
        this->Attr("cacheMode").AttrType(OPTIONAL).Int(0);


        this->SetInferShape(ge::MLAInferShape).SetInferDataType(ge::MLAInferDataType);

        this->AICore().SetTiling(optiling::MLATilingFunc);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")   // set value of aclnn support
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false"); //set jit compile flag
        this->AICore().AddConfig("ascend910b", aicore_config);

    }
};

OP_ADD(MLA);
}
