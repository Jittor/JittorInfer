
#include "register/op_def_registry.h"
#include "rope_ext_custom_v2_tiling.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    RopeExtCustomV2TilingData tiling;
    uint32_t shape0 = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t shape1 = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t shape2 = context->GetInputShape(0)->GetOriginShape().GetDim(2);
    context->SetBlockDim(shape0 * shape1 * shape2);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();

    const int32_t* ne0 = attrs->GetAttrPointer<int32_t>(0);
    const int32_t* ne1 = attrs->GetAttrPointer<int32_t>(1);
    const int32_t* pos_len = attrs->GetAttrPointer<int32_t>(2);

    tiling.set_ne0(*ne0);
    tiling.set_ne1(*ne1);
    tiling.set_pos_len(*pos_len);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class RopeExtCustomV2 : public OpDef {
   public:
    explicit RopeExtCustomV2(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("cos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("sin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("dst")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("ne0").Int();
        this->Attr("ne1").Int();
        this->Attr("pos_len").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(RopeExtCustomV2);
}  // namespace ops
