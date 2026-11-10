#pragma once

#include "../config.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace finer::models::qwen3 {

struct Qwen3Config : public finer::models::ModelConfig {
  bool        attention_bias;
  int64_t     head_dim;
  std::string hidden_act;
  int64_t     hidden_size;
  int64_t     intermediate_size;
  int64_t     max_position_embeddings;
  int64_t     num_attention_heads;
  int64_t     num_hidden_layers;
  int64_t     num_key_value_heads;
  float       rms_norm_eps;
  float       rope_theta;
  bool        tie_word_embeddings;

  bool                 is_moe      = false;
  int64_t              num_experts = 0, num_experts_per_tok = 0;
  int64_t              moe_intermediate_size = 0;
  int64_t              decoder_sparse_step   = 1;
  bool                 norm_topk_prob        = false;
  std::vector<int64_t> mlp_only_layers;

  Qwen3Config(const nlohmann::json &j) : ModelConfig(j) {
    if (model_type != "qwen3" && model_type != "qwen3_moe") {
      throw std::runtime_error(
        "Invalid model type: expected 'qwen3' or 'qwen3_moe', got '"
        + model_type + "'");
    }

    is_moe = model_type == "qwen3_moe";

    attention_bias          = j["attention_bias"].get<bool>();
    head_dim                = j["head_dim"].get<int64_t>();
    hidden_act              = j["hidden_act"].get<std::string>();
    hidden_size             = j["hidden_size"].get<int64_t>();
    intermediate_size       = j["intermediate_size"].get<int64_t>();
    max_position_embeddings = j["max_position_embeddings"].get<int64_t>();
    num_attention_heads     = j["num_attention_heads"].get<int64_t>();
    num_hidden_layers       = j["num_hidden_layers"].get<int64_t>();
    num_key_value_heads     = j["num_key_value_heads"].get<int64_t>();
    rms_norm_eps            = j["rms_norm_eps"].get<float>();
    rope_theta              = j["rope_theta"].get<float>();
    tie_word_embeddings     = j["tie_word_embeddings"].get<bool>();

    if (is_moe) {
      num_experts           = j.at("num_experts").get<int64_t>();
      num_experts_per_tok   = j.at("num_experts_per_tok").get<int64_t>();
      moe_intermediate_size = j.at("moe_intermediate_size").get<int64_t>();
      decoder_sparse_step   = j.value("decoder_sparse_step", int64_t{1});
      norm_topk_prob        = j.value("norm_topk_prob", false);
      mlp_only_layers = j.value("mlp_only_layers", std::vector<int64_t>{});
      if (num_experts <= 0 || num_experts_per_tok <= 0
        || num_experts_per_tok > num_experts || moe_intermediate_size <= 0
        || decoder_sparse_step <= 0 || hidden_act != "silu")
        throw std::invalid_argument("Invalid Qwen3 MoE configuration");
      for (auto layer : mlp_only_layers)
        if (layer < 0 || layer >= num_hidden_layers)
          throw std::invalid_argument(
            "Qwen3 MoE: mlp_only_layers out of range");
      if (j.value("use_sliding_window", false)
        || (j.contains("rope_scaling") && !j.at("rope_scaling").is_null()))
        throw std::invalid_argument(
          "Qwen3 MoE: sliding window/scaled RoPE unsupported");
    }
  }

  bool layer_is_moe(int64_t index) const {
    return is_moe && (index + 1) % decoder_sparse_step == 0
      && std::find(mlp_only_layers.begin(), mlp_only_layers.end(), index)
      == mlp_only_layers.end();
  }

  static Qwen3Config from_huggingface(const std::filesystem::path &model_path) {
    return {nlohmann::json::parse(std::ifstream(model_path / "config.json"))};
  }
};

} // namespace finer::models::qwen3