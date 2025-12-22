
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "../utils/inc/log/ops_log.h"
#include "cstdio"
#include "jittor_infer_flash_attention_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus ConvertContextToPFAParams(
    gert::TilingContext* context, ContextParamsForPFATiling& contextKeyParams) {
    contextKeyParams.opName = context->GetNodeName();
    bool inputOutputIsNullPtr =
        (context->GetInputDesc(QUERY_INDEX) == nullptr) ||
        (context->GetInputDesc(KEY_INDEX) == nullptr) ||
        (context->GetInputDesc(VALUE_INDEX) == nullptr) ||
        (context->GetOutputDesc(ATTENTION_OUT_INDEX) == nullptr) ||
        (context->GetInputShape(QUERY_INDEX) == nullptr) ||
        (context->GetInputShape(KEY_INDEX) == nullptr) ||
        (context->GetInputShape(VALUE_INDEX) == nullptr) ||
        (context->GetOutputShape(ATTENTION_OUT_INDEX) == nullptr);
    OPS_ERR_IF(inputOutputIsNullPtr,
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                                           "q, k, v or attenOut is nullptr!"),
               return ge::GRAPH_FAILED);

    contextKeyParams.pseShift =
        context->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMask =
        context->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    contextKeyParams.actualSeqenceLengthQ =
        context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    contextKeyParams.actualSeqenceLengthKV =
        context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    contextKeyParams.inputDataType =
        context->GetInputDesc(QUERY_INDEX)->GetDataType();
    contextKeyParams.kDataType =
        context->GetInputDesc(KEY_INDEX)->GetDataType();
    contextKeyParams.vDataType =
        context->GetInputDesc(VALUE_INDEX)->GetDataType();
    contextKeyParams.blockTable = nullptr;
    contextKeyParams.pseShiftDataType =
        (contextKeyParams.pseShift != nullptr)
            ? context->GetOptionalInputDesc(PSE_SHIFT_INDEX)->GetDataType()
            : contextKeyParams.inputDataType;
    contextKeyParams.maskDataType =
        (contextKeyParams.attentionMask != nullptr)
            ? context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType()
            : contextKeyParams.inputDataType;
    contextKeyParams.outputDataType = context->GetOutputDesc(0)->GetDataType();
    contextKeyParams.queryInputShape = context->GetInputShape(QUERY_INDEX);
    contextKeyParams.keyInputShape = context->GetInputShape(KEY_INDEX);
    contextKeyParams.valueInputShape = context->GetInputShape(VALUE_INDEX);
    contextKeyParams.pseShiftShape =
        context->GetOptionalInputShape(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMaskShape =
        context->GetOptionalInputShape(ATTEN_MASK_INDEX);
    contextKeyParams.deqScale1Shape =
        context->GetOptionalInputShape(DEQ_SCALE1_INDEX);
    contextKeyParams.scale1Shape =
        context->GetOptionalInputShape(QUANT_SCALE1_INDEX);
    contextKeyParams.deqScale2Shape =
        context->GetOptionalInputShape(DEQ_SCALE2_INDEX);
    contextKeyParams.scale2Shape =
        context->GetOptionalInputShape(QUANT_SCALE2_INDEX);
    contextKeyParams.offset2Shape =
        context->GetOptionalInputShape(QUANT_OFFSET2_INDEX);
    contextKeyParams.outputShape = context->GetOutputShape(0);
    auto attrs = context->GetAttrs();
    contextKeyParams.innerPrecisePtr =
        attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
    contextKeyParams.sparseMode =
        attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE);
    contextKeyParams.preToken =
        attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
    contextKeyParams.nextToken =
        attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
    contextKeyParams.scaleValue =
        attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    contextKeyParams.layout =
        attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    contextKeyParams.numKeyValueHeads =
        attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    contextKeyParams.workspaceSize = context->GetWorkspaceSizes(1);

    contextKeyParams.deqScaleType =
        (context->GetOptionalInputDesc(DEQ_SCALE1_INDEX) != nullptr)
            ? context->GetOptionalInputDesc(DEQ_SCALE1_INDEX)->GetDataType()
            : contextKeyParams.inputDataType;
    contextKeyParams.deqScale2Type =
        (context->GetOptionalInputDesc(DEQ_SCALE2_INDEX) != nullptr)
            ? context->GetOptionalInputDesc(DEQ_SCALE2_INDEX)->GetDataType()
            : contextKeyParams.inputDataType;

    contextKeyParams.quantScale2Type =
        (context->GetOptionalInputDesc(QUANT_SCALE2_INDEX) != nullptr)
            ? context->GetOptionalInputDesc(QUANT_SCALE2_INDEX)->GetDataType()
            : ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type =
        (context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX) != nullptr)
            ? context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX)->GetDataType()
            : ge::DT_FLOAT;

    auto platformInfoPtr = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfoPtr == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "platformInfoPtr is null!"),
               return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    contextKeyParams.aivNum = ascendcPlatform.GetCoreNumAiv();
    contextKeyParams.aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                   contextKeyParams.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,
                                   contextKeyParams.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,
                                   contextKeyParams.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,
                                   contextKeyParams.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,
                                   contextKeyParams.l0BSize);

    contextKeyParams.defaultSysWorkspaceSize = 0;

    return ge::GRAPH_SUCCESS;
}

PFA_EXTERN_C ge::graphStatus PFATilingFunc(gert::TilingContext* context) {
    if (context == nullptr) {
        OPS_LOG_E("JittorInferFlashAttention", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    if (context->GetRawTilingData() == nullptr) {
        OPS_LOG_E("JittorInferFlashAttention",
                  "tiling context GetRawTilingData is nullptr!");
        return ge::GRAPH_FAILED;
    }
    JittorInferFlashAttentionTiling flashTiling(nullptr);
    JittorInferFlashAttentionTilingData tilingData;
    OPS_ERR_IF(memset_s(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity(), 0,
                        context->GetRawTilingData()->GetCapacity()) != EOK,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "fail to memset tiling data"),
               return ge::GRAPH_FAILED);
    ContextParamsForPFATiling contextParamsForPFATiling = {
        .pseShift = nullptr,
        .attentionMask = nullptr,
        .actualSeqenceLengthQ = nullptr,
        .actualSeqenceLengthKV = nullptr,
        .blockTable = nullptr,
        .inputDataType = ge::DataType::DT_FLOAT16,
        .kDataType = ge::DataType::DT_FLOAT16,
        .vDataType = ge::DataType::DT_FLOAT16,
        .pseShiftDataType = ge::DataType::DT_FLOAT16,
        .maskDataType = ge::DataType::DT_FLOAT16,
        .blockTableType = ge::DataType::DT_FLOAT16,  // Initialize pfa context
        .outputDataType = ge::DataType::DT_FLOAT16,
        .opName = nullptr,
        .queryInputShape = nullptr,
        .keyInputShape = nullptr,
        .valueInputShape = nullptr,
        .pseShiftShape = nullptr,
        .attentionMaskShape = nullptr,
        .deqScale1Shape = nullptr,  // Initialize pfa context
        .scale1Shape = nullptr,
        .deqScale2Shape = nullptr,
        .scale2Shape = nullptr,
        .offset2Shape = nullptr,
        .blockTableShape = nullptr,
        .outputShape = nullptr,  // Initialize pfa context
        .lseoutputShape = nullptr,
        .innerPrecisePtr = nullptr,
        .headsNumber = nullptr,
        .sparseMode = nullptr,
        .preToken = nullptr,
        .nextToken = nullptr,
        .scaleValue = nullptr,
        .blockSize = nullptr,  // Initialize pfa context
        .layout = nullptr,
        .numKeyValueHeads = nullptr,
        .workspaceSize = nullptr,
        .deqScaleType = ge::DataType::DT_FLOAT16,
        .deqScale2Type = ge::DataType::DT_FLOAT16,
        .quantScale2Type = ge::DataType::DT_FLOAT16,
        .quantOffset2Type = ge::DataType::DT_FLOAT16,  // Initialize pfa context
        .softmaxLseFlag = nullptr,                     // Initialize pfa context
        .isSoftMaxLseEnable = false,
        .aivNum = 0,
        .aicNum = 0,
        .ubSize = 0,
        .l1Size = 0,
        .l0CSize = 0,
        .l0ASize = 0,
        .l0BSize = 0,
        .defaultSysWorkspaceSize = 0,
    };
    auto ret = ConvertContextToPFAParams(context, contextParamsForPFATiling);
    uint64_t tilingKey = 7;  // 7: default tiling key
    uint32_t blockDimToBeSet;
    ret = flashTiling.RunBigKernelTilingWithParams(
        contextParamsForPFATiling, tilingKey, blockDimToBeSet, tilingData);
    tilingKey += BENCHMARK_TILING_KEY;
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDimToBeSet);
    flashTiling.JittorInferFlashAttentionSetTilingData(context, tilingData);
    return ret;
}
}  // namespace optiling

namespace ge {
const uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
static ge::graphStatus PFAInferShape(gert::InferShapeContext* context) {
    // query shape : (B, S, H)
    const gert::Shape* query_shape = context->GetInputShape(0);
    const gert::Shape* value_shape = context->GetInputShape(2);
    OPS_LOG_E_IF_NULL(context, query_shape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, value_shape, return ge::GRAPH_FAILED)

    // attentionOut: (B, S, H)
    gert::Shape* attentionOutShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)

    // Get attr
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED)
    std::string inputLayout =
        std::string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));

    // Set output shape
    *attentionOutShape = *query_shape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus PFAInferDataType(gert::InferDataTypeContext* context) {
    // default set q's dtype as ifa's output type
    ge::DataType outputType = context->GetInputDataType(0);
    if (context->GetOptionalInputDataType(9) !=
        ge::DT_UNDEFINED) {  // 9 is quant_scale2's index
        outputType = ge::DT_INT8;
    } else if (outputType == ge::DT_INT8) {
        outputType = ge::DT_FLOAT16;
    }
    // attention_out, outidx:0
    context->SetOutputDataType(0, outputType);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {

class JittorInferFlashAttention : public OpDef {
   public:
    explicit JittorInferFlashAttention(const char* name) : OpDef(name) {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_BF16,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("pseShift")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("attenMask")
            .ParamType(OPTIONAL)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_BOOL,  ge::DT_BOOL,    ge::DT_BOOL,
                 ge::DT_BOOL,    ge::DT_INT8,  ge::DT_INT8,    ge::DT_INT8,
                 ge::DT_UINT8,   ge::DT_UINT8, ge::DT_UINT8,   ge::DT_BOOL,
                 ge::DT_INT8,    ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_BOOL,
                 ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_BOOL,    ge::DT_INT8,
                 ge::DT_INT8,    ge::DT_INT8,  ge::DT_UINT8,   ge::DT_UINT8,
                 ge::DT_UINT8,   ge::DT_BOOL,  ge::DT_INT8,    ge::DT_UINT8,
                 ge::DT_BOOL,    ge::DT_BOOL,  ge::DT_INT8,    ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND});
        this->Input("actualSeqLengths")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("actualSeqLengthsKV")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("deqScale1")
            .ParamType(OPTIONAL)
            .DataType(
                {ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                 ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                 ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                 ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_FLOAT,
                 ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                 ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                 ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                 ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("quantScale1")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("deqScale2")
            .ParamType(OPTIONAL)
            .DataType(
                {ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                 ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                 ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                 ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_FLOAT,
                 ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                 ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                 ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
                 ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("quantScale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("quantOffset2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Output("attentionOut")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16,
                 ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_INT8,
                 ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16,
                 ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,
                 ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_BF16,
                 ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16,
                 ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                 ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8})
            .FormatList({ge::FORMAT_ND});
        this->Attr("numHeads").AttrType(REQUIRED).Int(1);
        this->Attr("scaleValue").AttrType(OPTIONAL).Float(1.0);
        this->Attr("preTokens").AttrType(OPTIONAL).Int(214748647);
        this->Attr("nextTokens").AttrType(OPTIONAL).Int(0);
        this->Attr("inputLayout").AttrType(OPTIONAL).String("BSH");
        this->Attr("numKeyValueHeads").AttrType(OPTIONAL).Int(0);
        this->Attr("sparseMode").AttrType(OPTIONAL).Int(0);
        this->Attr("innerPrecise").AttrType(OPTIONAL).Int(1);

        this->SetInferShape(ge::PFAInferShape)
            .SetInferDataType(ge::PFAInferDataType);

        this->AICore().SetTiling(optiling::PFATilingFunc);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value",
                           "support_aclnn")  // set value of aclnn support
            .ExtendCfgInfo(
                "jitCompile.flag",
                "static_false,dynamic_false");  // set jit compile flag
        this->AICore().AddConfig("ascend910b", aicore_config);
    }
};

OP_ADD(JittorInferFlashAttention);
}  // namespace ops
