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
#include "mla_prefill_tiling.h"

namespace optiling {

enum MaskType {
    MASK_TYPE_NONE = 0,
    MASK_TYPE_NORM = 1,
    MASK_TYPE_ALIBI = 2,
    MASK_TYPE_LOOK_AHEAD = 3,
    MASK_TYPE_MASK_FREE = 4,
    MASK_TYPE_CAUSAL_MASK = 5,
    MASK_TYPE_SWA_NORM = 6
};

inline ge::graphStatus GetPrefillMaskInfo(gert::TilingContext* context, ContextParamsForMLA& contextKeyParams)
{
    auto attrs = context->GetAttrs();
    auto maskType = *attrs->GetAttrPointer<int64_t>(3);
    if (maskType == MASK_TYPE_NONE || maskType == MASK_TYPE_CAUSAL_MASK) {
        return ge::GRAPH_SUCCESS; 
    }
    auto maskShape = context->GetInputShape(7)->GetOriginShape();
    auto maskDim = maskShape.GetDimNum();
    int32_t maxSeq = maskShape.GetDim(maskDim - 1);
    contextKeyParams.maxSeqLen = maxSeq;
    if(maskType != MASK_TYPE_MASK_FREE) return ge::GRAPH_FAILED;
    if(maskDim != 2) return ge::GRAPH_FAILED;
    if(maskShape.GetDim(1) != 512) return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToMLAPrefillParams(gert::TilingContext* context, ContextParamsForMLA& contextKeyParams)
{
    int64_t queryDim = context->GetInputShape(0)->GetOriginShape().GetDimNum();
    int32_t embed = 0; // headdim
    auto attrs = context->GetAttrs();
    if (queryDim == 3) {
        embed = context->GetInputShape(0)->GetOriginShape().GetDim(2) +
                    context->GetInputShape(1)->GetOriginShape().GetDim(2);
    } else {
        embed = (context->GetInputShape(0)->GetOriginShape().GetDim(1) +
                    context->GetInputShape(1)->GetOriginShape().GetDim(1)) / *attrs->GetAttrPointer<int64_t>(0);
    }
    contextKeyParams.kvSeqLen = const_cast<int32_t*>(context->GetInputTensor(6)->GetData<int32_t>());
    contextKeyParams.qSeqLen = const_cast<int32_t*>(context->GetInputTensor(5)->GetData<int32_t>());
    contextKeyParams.numTokens = static_cast<int32_t>(context->GetInputShape(6)->GetOriginShape().GetDim(0));
    contextKeyParams.batch = contextKeyParams.numTokens;
    auto maxKvSeq = *std::max_element(contextKeyParams.kvSeqLen, contextKeyParams.kvSeqLen + contextKeyParams.batch);
    contextKeyParams.maxKvSeqLen = maxKvSeq;
    if(GetPrefillMaskInfo(context, contextKeyParams) == ge::GRAPH_FAILED) return ge::GRAPH_FAILED;
    contextKeyParams.tor = *attrs->GetAttrPointer<float>(1);
    contextKeyParams.numHeads = *attrs->GetAttrPointer<int64_t>(0);
    contextKeyParams.embeddingSize = embed;
    contextKeyParams.embeddingSizeV = embed - 64;
    contextKeyParams.kvHeads = *attrs->GetAttrPointer<int64_t>(2) == 0 ? *attrs->GetAttrPointer<int64_t>(0) : *attrs->GetAttrPointer<int64_t>(2);
    contextKeyParams.maskType = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(3));

    auto valueDim = context->GetInputShape(4)->GetOriginShape().GetDimNum();
    if(!((context->GetInputShape(4)->GetOriginShape().GetDim(valueDim - 1) == contextKeyParams.embeddingSizeV) ||
                  (context->GetInputShape(4)->GetOriginShape().GetDim(valueDim - 1) == contextKeyParams.kvHeads * contextKeyParams.embeddingSizeV))) 
        return ge::GRAPH_FAILED;
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
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetMLAPrefillTilingKeyTypeBase(gert::TilingContext* context, ContextParamsForMLA &contextParamsForMLA)
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

uint32_t GetPrefillTilingKey(ContextParamsForMLA& contextParams)
{
    uint32_t prefillTilingKey = 1;
    if (contextParams.maskType == 0) {
        prefillTilingKey = 1;
    } else if (contextParams.maskType == 6) {
        prefillTilingKey = 3;
    } else if (contextParams.maskType > 0) {
        prefillTilingKey = 2;
    } 
    return prefillTilingKey;
}

ge::graphStatus GenMLAPrefillTilingKey(ContextParamsForMLA &contextParams, gert::TilingContext* context)
{
    // currently only support fp16/bf16 maskfree prefill or bf16 prefill kernel
    uint32_t tilingKey = GetPrefillTilingKey(contextParams);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFunc(gert::TilingContext* context) {
    if (context == nullptr) {
        OPS_LOG_E("MLAPrefill", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    if (context->GetRawTilingData() == nullptr) {
        OPS_LOG_E("MLAPrefill", "tiling context GetRawTilingData is nullptr!");
        return ge::GRAPH_FAILED;
    }
    ContextParamsForMLA contextParamsForMLAPrefill;
    GetMLAPrefillTilingKeyTypeBase(context, contextParamsForMLAPrefill);
    ConvertContextToMLAPrefillParams(context, contextParamsForMLAPrefill);

    MLAPrefillTilingData tilingData;
    if(memset_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
                0, context->GetRawTilingData()->GetCapacity()) != EOK)
                return ge::GRAPH_FAILED;

    uint32_t blockDimToBeSet;
    if(GetMLAPrefillTilingParam(contextParamsForMLAPrefill, blockDimToBeSet, tilingData) == ge::GRAPH_FAILED) return ge::GRAPH_FAILED;
    size_t* workspaces = contextParamsForMLAPrefill.workspaceSize;
    workspaces[0] = GetMLAPrefillWorkSpaceSize(contextParamsForMLAPrefill, blockDimToBeSet);

    if(GenMLAPrefillTilingKey(contextParamsForMLAPrefill, context) == ge::GRAPH_FAILED) return ge::GRAPH_FAILED;
    context->SetBlockDim(blockDimToBeSet);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* query_shape = context->GetInputShape(0);
    gert::Shape* attentionOutShape = context->GetOutputShape(0);
    *attentionOutShape = *query_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  // default set q's dtype as ifa's output type
  const auto outputType = context->GetInputDataType(0);
  context->SetOutputDataType(0, outputType);
  return GRAPH_SUCCESS;
}

}

namespace ops {

class MLAPrefill : public OpDef {
public:
    explicit MLAPrefill(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("qRope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("kRope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("qSeqLen")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("kvSeqLen")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("mask")
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


        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);

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

OP_ADD(MLAPrefill);
}
