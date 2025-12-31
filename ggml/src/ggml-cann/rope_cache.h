#include "aclnn_ops.h"
#include "ggml-impl.h"
#include "es_graph_builder.h"

class RopeCache {
   public:
    RopeCache(ggml_backend_cann_context& ctx, ggml_tensor* dst);
    void* sin_final_buffer;
    void* cos_final_buffer;
    int64_t final_size;
    std::array<int64_t, 4> final_shape;
    ge::Operator GetCosOp(ge::Graph& graph, const std::string& name) const;
    ge::Operator GetSinOp(ge::Graph& graph, const std::string& name) const;
    // ES API versions - directly create Const using EsGraphBuilder
    ge::es::EsTensorHolder GetCosEsTensor(ge::es::EsGraphBuilder& graph_builder) const;
    ge::es::EsTensorHolder GetSinEsTensor(ge::es::EsGraphBuilder& graph_builder) const;
    ~RopeCache() {
        ACL_CHECK(aclrtFreeHost(sin_final_buffer));
        ACL_CHECK(aclrtFreeHost(cos_final_buffer));
    }
};
