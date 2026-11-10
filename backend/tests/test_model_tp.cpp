#include "../models/deepseek-v2/model-deepseek-v2.hpp"
#include "../models/llama/model-llama.hpp"
#include <forch/ops/input.hpp>
#include <iostream>
#include <map>
#include <set>

using namespace finer;
using namespace forch;

void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

struct Weights : nn::Loader {
  std::map<std::string, Tensor> tensors;
  Tensor                        load(const std::string &name) override {
    return tensors.at(name);
  }
  void add(const std::string &name, Shape shape, DType dtype) {
    size_t count = nn::WeightShard::element_size(dtype);
    for (auto dim : shape)
      count *= dim;
    std::vector<uint8_t> bytes(count);
    for (size_t i = 0; i < count; ++i)
      bytes[i] = (i * 37 + i / 251) % 251;
    tensors.emplace(name,
      Tensor({},
        std::make_unique<nn::WeightShard::ShardConstant>(
          std::move(bytes), shape, dtype)));
  }
  void mlp(const std::string &prefix, int64_t hidden, int64_t intermediate,
    DType dtype) {
    add(prefix + ".gate_proj.weight", {intermediate, hidden}, dtype);
    add(prefix + ".up_proj.weight", {intermediate, hidden}, dtype);
    add(prefix + ".down_proj.weight", {hidden, intermediate}, dtype);
  }
};

void slice_matches(
  const Tensor &local, const Tensor &full, int axis, int start) {
  const auto *a     = dynamic_cast<const op::Const *>(local.get_op())->buffer();
  const auto *b     = dynamic_cast<const op::Const *>(full.get_op())->buffer();
  const auto  width = nn::WeightShard::element_size(full.dtype());
  const auto  rows = local.shape()[0], cols = local.shape()[1];
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col) {
      const auto source = (row + (axis == 0 ? start : 0)) * full.shape()[1]
        + col + (axis == 1 ? start : 0);
      require(
        std::memcmp(a + (row * cols + col) * width, b + source * width, width)
          == 0,
        "checkpoint shard differs");
    }
}

int reductions(const Tensor &tensor, std::set<const void *> &seen) {
  if (!seen.insert(tensor.get_op()).second)
    return 0;
  int count = dynamic_cast<const op::AllReduce *>(tensor.get_op()) ? 1 : 0;
  for (const auto &input : tensor.get_inputs())
    count += reductions(input, seen);
  return count;
}
int reductions(const Tensor &tensor) {
  std::set<const void *> seen;
  return reductions(tensor, seen);
}

void llama_test() {
  using namespace models::llama;
  const LlamaConfig    config({{"model_type", "llama"}, {"vocab_size", 32},
       {"hidden_size", 16}, {"intermediate_size", 32}, {"num_attention_heads", 8},
       {"num_key_value_heads", 2}, {"head_dim", 2}, {"num_hidden_layers", 1},
       {"max_position_embeddings", 128}, {"rms_norm_eps", 1e-6},
       {"attention_bias", true}, {"mlp_bias", true}});
  engine::EngineConfig ec;
  ec.logits_topk = 8;
  for (int size : {1, 2, 4, 8})
    for (int rank = 0; rank < size; ++rank) {
      ec.tp = ProcessGroup(rank, size);
      LlamaForCausalLM model(config, ec);
      auto            *attention = (*model.model->layers)[0]->self_attn;
      require(attention->num_attention_heads == 8 / size
          && attention->num_key_value_heads == std::max(1, 2 / size),
        "Llama local heads mismatch");
      Weights weights;
      for (auto name : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        int out = std::string(name) == "k_proj" || std::string(name) == "v_proj"
          ? 4
          : 16;
        weights.add(
          std::string("a.") + name + ".weight", {out, 16}, DType::FP16);
        weights.add(std::string("a.") + name + ".bias", {out}, DType::FP16);
      }
      attention->load(&weights, "a");
      auto projections =
        attention->qkv_proj.forward(op::create_input({3, 16}, DType::FP16));
      for (int i = 0; i < 3; ++i) {
        const std::string name  = std::array{"q_proj", "k_proj", "v_proj"}[i];
        const int         start = i == 0
                  ? rank * 16 / size
                  : (size <= 2 ? rank * 4 / size : rank / (size / 2) * 2);
        slice_matches(projections[i].get_inputs()[0].get_inputs()[1],
          weights.tensors.at("a." + name + ".weight"), 0, start);
      }
      slice_matches(attention->o_proj->weight(),
        weights.tensors.at("a.o_proj.weight"), 1, rank * 16 / size);
      auto output = attention->o_proj->forward(
        op::create_input({3, 16 / size}, DType::FP16));
      require(
        output.shape() == Shape{3, 16} && reductions(output) == (size > 1),
        "Llama output not reduced");
      if (size > 1)
        require(
          dynamic_cast<const op::AllReduce *>(output.get_inputs()[0].get_op()),
          "replicated bias must follow reduction");
      require(model.lm_head->local_out_features() == 32 / size,
        "Llama vocabulary must be sharded");
    }
}

void deepseek_test() {
  using namespace models::deepseek_v2;
  nlohmann::json       raw = {{"model_type", "deepseek_v2"}, {"vocab_size", 32},
          {"hidden_size", 16}, {"intermediate_size", 32}, {"num_attention_heads", 8},
          {"num_key_value_heads", 8}, {"num_hidden_layers", 2},
          {"max_position_embeddings", 128}, {"kv_lora_rank", 4},
          {"qk_nope_head_dim", 2}, {"qk_rope_head_dim", 2}, {"v_head_dim", 2},
          {"moe_intermediate_size", 16}, {"n_routed_experts", 4},
          {"n_shared_experts", 1}, {"num_experts_per_tok", 2},
          {"first_k_dense_replace", 1}};
  engine::EngineConfig ec;
  ec.logits_topk = 8;
  for (bool lora : {false, true}) {
    raw["q_lora_rank"] = lora ? nlohmann::json(4) : nlohmann::json(nullptr);
    DeepseekV2Config config(raw);
    for (int size : {1, 2, 4, 8})
      for (int rank = 0; rank < size; ++rank) {
        ec.tp = ProcessGroup(rank, size);
        DeepseekV2ForCausalLM model(config, ec);
        auto                 *attention = (*model.model->layers)[0]->self_attn;
        require(attention->num_heads == 8 / size, "MLA heads not local");
        require(attention->kv_a_proj_with_mqa->out_features == 6,
          "MLA latent must remain replicated");
        Weights weights;
        if (lora) {
          weights.add("a.q_a_proj.weight", {4, 16}, DType::BF16);
          weights.add("a.q_a_layernorm.weight", {4}, DType::BF16);
          weights.add("a.q_b_proj.weight", {32, 4}, DType::BF16);
        } else
          weights.add("a.q_proj.weight", {32, 16}, DType::BF16);
        weights.add("a.kv_a_proj_with_mqa.weight", {6, 16}, DType::BF16);
        weights.add("a.kv_a_layernorm.weight", {4}, DType::BF16);
        weights.add("a.kv_b_proj.weight", {32, 4}, DType::BF16);
        weights.add("a.o_proj.weight", {16, 16}, DType::BF16);
        attention->load(&weights, "a");
        slice_matches(attention->kv_b_proj->weight(),
          weights.tensors.at("a.kv_b_proj.weight"), 0, rank * 32 / size);
        slice_matches(attention->o_proj->weight(),
          weights.tensors.at("a.o_proj.weight"), 1, rank * 16 / size);
        // Absorption is model math: inspect W_uk/W_uv slices and transposes
        // without a cache or backend, for both query projections and all ranks.
        const auto local_heads = attention->num_heads;
        auto       absorbed    = attention->absorb_query(
          op::create_input({3, local_heads, 2}, DType::BF16));
        auto expanded = attention->unabsorb_output(
          op::create_input({1, 3, local_heads, 4}, DType::BF16));
        require(absorbed.shape() == Shape{1, 3, local_heads, 4}
            && expanded.shape() == Shape{3, local_heads * 2},
          "model-owned MLA projection dimensions differ");
        int rejected = 0;
        try {
          attention->absorb_query(
            op::create_input({3, local_heads, 4}, DType::BF16));
        } catch (const forch::exception &) {
          ++rejected;
        }
        try {
          attention->unabsorb_output(
            op::create_input({3, local_heads, 4}, DType::BF16));
        } catch (const forch::exception &) {
          ++rejected;
        }
        require(
          rejected == 2, "model-owned MLA projections accepted wrong geometry");
        for (const auto &[output, begin] :
          {std::pair{absorbed, 0}, std::pair{expanded, 2}}) {
          const auto &matmul = output.get_inputs()[0].get_inputs()[0];
          const auto *op =
            dynamic_cast<const forch::op::BatchedMatMul *>(matmul.get_op());
          require(op && !op->transpose_a && op->transpose_b == (begin == 0),
            "MLA absorption matmul orientation differs");
          const auto &slice = matmul.get_inputs()[1].get_inputs()[0];
          const auto *cut =
            dynamic_cast<const forch::op::Slice *>(slice.get_op());
          require(cut && cut->slice.size() == 3 && cut->slice[1].begin == begin
              && cut->slice[1].end == begin + 2,
            "MLA projection selected wrong checkpoint rows");
          require(slice.get_inputs()[0].get_inputs()[0].get_op()
              == attention->kv_b_proj->weight().get_op(),
            "MLA projection must use this model rank's weight shard");
        }
        auto *moe = (*model.model->layers)[1]->moe;
        require(moe && moe->experts->size() == 4, "TP must retain all experts");
        weights.add("m.gate.weight", {4, 16}, DType::BF16);
        weights.mlp("m.shared_experts", 16, 16, DType::BF16);
        for (int e = 0; e < 4; ++e)
          weights.mlp("m.experts." + std::to_string(e), 16, 16, DType::BF16);
        moe->load(&weights, "m");
        for (int e = 0; e < 4; ++e) {
          auto      *expert = (*moe->experts)[e].get();
          const auto prefix = "m.experts." + std::to_string(e);
          slice_matches(expert->gate_proj->weight(),
            weights.tensors.at(prefix + ".gate_proj.weight"), 0,
            rank * 16 / size);
          slice_matches(expert->down_proj->weight(),
            weights.tensors.at(prefix + ".down_proj.weight"), 1,
            rank * 16 / size);
        }
        auto input   = op::create_input({3, 16}, DType::BF16);
        auto scores  = op::create_input({3, 2}, DType::FP32);
        auto indices = op::create_input({3, 2}, DType::INT32);
        for (const auto &out : {moe->forward_dense_with(input, scores, indices),
               moe->forward_moe_with(input, scores, indices)}) {
          require(out.shape() == Shape{3, 16}, "MoE output geometry changed");
          require(reductions(out) == (size > 1 ? 2 : 0),
            "routed and shared outputs must each be reduced once");
        }
      }
  }
}
void lm_head_test() {
  for (int size : {1, 2, 4, 8})
    for (int rank = 0; rank < size; ++rank)
      for (int k : {1, 5, 16}) {
        models::VocabParallelLMHead head(
          4, 16, k, ProcessGroup(rank, size), DType::FP32);
        Weights weights;
        weights.add("model.embed_tokens.weight", {16, 4}, DType::FP32);
        head.load(&weights, "model.embed_tokens");
        slice_matches(head.weight(),
          weights.tensors.at("model.embed_tokens.weight"), 0, rank * 16 / size);
        auto result = head.forward(op::create_input({3, 4}, DType::FP32));
        require(result.values.shape() == Shape{3, k}
            && result.indices.shape() == Shape{3, k}
            && result.indices.dtype() == DType::INT64,
          "LM head topk output differs");
      }
}
int main() {
  try {
    lm_head_test();
    llama_test();
    deepseek_test();
    std::cout << "PASS Llama/DeepSeek TP shards and collectives\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
