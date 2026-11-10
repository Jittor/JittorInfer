#pragma once

#include "../config.hpp"

#include <forch/forch.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

namespace finer::models::llama {

struct LlamaRopeScaling {
  std::string rope_type;
  float       factor;
  float       low_freq_factor;
  float       high_freq_factor;
  int64_t     original_max_position_embeddings;
};

struct LlamaConfig : public finer::models::ModelConfig {
  bool                            attention_bias;
  int64_t                         head_dim;
  std::string                     hidden_act;
  int64_t                         hidden_size;
  int64_t                         intermediate_size;
  int64_t                         max_position_embeddings;
  bool                            mlp_bias;
  int64_t                         num_attention_heads;
  int64_t                         num_hidden_layers;
  int64_t                         num_key_value_heads;
  float                           rms_norm_eps;
  float                           rope_theta;
  std::optional<LlamaRopeScaling> rope_scaling;
  bool                            tie_word_embeddings;

  LlamaConfig(const nlohmann::json &j) : ModelConfig(j) {
    if (model_type != "llama") {
      throw std::runtime_error(
        "Invalid model type: expected 'llama', got '" + model_type + "'");
    }

    attention_bias          = j.value("attention_bias", false);
    hidden_act              = j.value("hidden_act", std::string("silu"));
    hidden_size             = j["hidden_size"].get<int64_t>();
    intermediate_size       = j["intermediate_size"].get<int64_t>();
    max_position_embeddings = j["max_position_embeddings"].get<int64_t>();
    mlp_bias                = j.value("mlp_bias", false);
    num_attention_heads     = j["num_attention_heads"].get<int64_t>();
    num_hidden_layers       = j["num_hidden_layers"].get<int64_t>();
    num_key_value_heads = j.value("num_key_value_heads", num_attention_heads);
    rms_norm_eps        = j["rms_norm_eps"].get<float>();
    rope_theta          = j.value("rope_theta", 10000.0F);
    tie_word_embeddings = j.value("tie_word_embeddings", false);
    head_dim = j.value("head_dim", hidden_size / num_attention_heads);

    const nlohmann::json *rope_config = nullptr;
    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
      rope_config = &j["rope_scaling"];
    } else if (j.contains("rope_parameters")
      && !j["rope_parameters"].is_null()) {
      rope_config = &j["rope_parameters"];
    }

    if (rope_config != nullptr) {
      const auto &rope = *rope_config;
      rope_scaling     = LlamaRopeScaling{
            .rope_type =
          rope.value("rope_type", rope.value("type", std::string("default"))),
            .factor           = rope.value("factor", 1.0F),
            .low_freq_factor  = rope.value("low_freq_factor", 1.0F),
            .high_freq_factor = rope.value("high_freq_factor", 1.0F),
            .original_max_position_embeddings = rope.value(
          "original_max_position_embeddings", max_position_embeddings),
      };
    }
  }

  static LlamaConfig from_huggingface(const std::filesystem::path &model_path) {
    return {nlohmann::json::parse(std::ifstream(model_path / "config.json"))};
  }
};

} // namespace finer::models::llama
