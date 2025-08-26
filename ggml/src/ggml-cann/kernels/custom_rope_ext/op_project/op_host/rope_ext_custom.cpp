
#include "register/op_def_registry.h"
#include "rope_ext_custom_tiling.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    RopeExtCustomTilingData tiling;
    uint32_t shape0 = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t shape1 = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t shape2 = context->GetInputShape(0)->GetOriginShape().GetDim(2);
    context->SetBlockDim(shape0 * shape1 * shape2);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();

    const int32_t* ne0 = attrs->GetAttrPointer<int32_t>(0);
    const int32_t* ne1 = attrs->GetAttrPointer<int32_t>(1);
    const int32_t* s1 = attrs->GetAttrPointer<int32_t>(2);
    const int32_t* s2 = attrs->GetAttrPointer<int32_t>(3);
    const int32_t* n_dims = attrs->GetAttrPointer<int32_t>(4);
    const float* freq_scale = attrs->GetAttrPointer<float>(5);
    const float* theta_scale = attrs->GetAttrPointer<float>(6);
    const float* ext_factor = attrs->GetAttrPointer<float>(7);
    const float* attn_factor = attrs->GetAttrPointer<float>(8);
    const float* corr_dims_v_0 = attrs->GetAttrPointer<float>(9);
    const float* corr_dims_v_1 = attrs->GetAttrPointer<float>(10);
    const float* logf_1_freq_scale = attrs->GetAttrPointer<float>(11);
    const int32_t* pos_len = attrs->GetAttrPointer<int32_t>(12);

    tiling.set_ne0(*ne0);
    tiling.set_ne1(*ne1);
    tiling.set_s1(*s1);
    tiling.set_s2(*s2);
    tiling.set_n_dims(*n_dims);
    tiling.set_freq_scale(*freq_scale);
    tiling.set_theta_scale(*theta_scale);
    tiling.set_ext_factor(*ext_factor);
    tiling.set_attn_factor(*attn_factor);
    tiling.set_corr_dims_v_0(*corr_dims_v_0);
    tiling.set_corr_dims_v_1(*corr_dims_v_1);
    tiling.set_logf_1_freq_scale(*logf_1_freq_scale);
    tiling.set_pos_len(*pos_len);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

// outputshape = xshape
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
// outputshape = xshape
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class RopeExtCustom : public OpDef {
   public:
    explicit RopeExtCustom(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("pos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("dst")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("ne0").Int();
        this->Attr("ne1").Int();
        this->Attr("s1").Int();
        this->Attr("s2").Int();
        this->Attr("n_dims").Int();
        this->Attr("freq_scale").Float();
        this->Attr("theta_scale").Float();
        this->Attr("ext_factor").Float();
        this->Attr("attn_factor").Float();
        this->Attr("corr_dims_v_0").Float();
        this->Attr("corr_dims_v_1").Float();
        this->Attr("logf_1_freq_scale").Float();
        this->Attr("pos_len").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(RopeExtCustom);
}  // namespace ops
