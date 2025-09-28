#ifndef LLAMA_GRAPH_QWEN2GE_H
#define LLAMA_GRAPH_QWEN2GE_H

#include "llama-graph-qwen2.h"

class llm_qwen2_context_ge : public llm_qwen2_context {
  protected:
    ggml_tensor * build_attn_indices();
    ggml_tensor * build_length_q();
    ggml_tensor * build_length_kv();
  public:
    llm_qwen2_context_ge(llama_context & lctx, std::vector<uint8_t> & buf_compute_meta, const llama_ubatch & ubatch,
                         const llm_build_cb & cb, bool worst_case, int print_layer = -1) :
        llm_qwen2_context(lctx, buf_compute_meta, ubatch, cb, worst_case, print_layer) {}
    struct ggml_cgraph * build_qwen2_ge();
};

struct ggml_cgraph * llm_build_qwen2_ge(llama_context & lctx, std::vector<uint8_t> & buf_compute_meta,
                                        const llama_ubatch & ubatch, llm_build_cb & cb, bool worst_case,
                                        int print_layer = -1);

void llm_update_qwen2_ge(llama_context & lctx);

#endif  // LLAMA_GRAPH_QWEN2GE_H
