#pragma once

#include "../config.hpp"
#include <fstream>
#include <optional>
#include <string>

namespace finer::models::deepseek_v2 {

/// `rope_scaling` section of DeepSeek-V2 config.json. Only "default" /
/// "linear" / "yarn" are supported for now; "dynamic" (NTK) changes the base
/// at runtime and cannot be a static graph — see DeepseekV2RotaryEmbedding.
struct RopeScaling {
  std::string type;                             // "yarn" | "linear" | "dynamic"
  float       factor;                           // scaling factor
  float       mscale;                           // yarn: mscale
  float       mscale_all_dim;                   // yarn: mscale_all_dim
  int64_t     original_max_position_embeddings; // yarn
  float       beta_fast;                        // yarn
  float       beta_slow;                        // yarn

  RopeScaling() = default;
  RopeScaling(const nlohmann::json &j) {
    type           = j.value("type", std::string("default"));
    factor         = j.value("factor", 1.0F);
    mscale         = j.value("mscale", 1.0F);
    mscale_all_dim = j.value("mscale_all_dim", 0.0F);
    original_max_position_embeddings =
      j.value("original_max_position_embeddings", int64_t(0));
    beta_fast = j.value("beta_fast", 32.0F);
    beta_slow = j.value("beta_slow", 1.0F);
  }
};

struct DeepseekV2Config : public finer::models::ModelConfig {
  bool        attention_bias;
  std::string hidden_act;
  int64_t     hidden_size;
  int64_t intermediate_size; // dense MLP size (first_k_dense_replace prefix)
  int64_t kv_lora_rank;
  int64_t max_position_embeddings;
  int64_t moe_intermediate_size;
  int64_t n_routed_experts; // 0 when config.json has null (dense-only)
  int64_t n_shared_experts; // 0 when config.json has null
  int64_t num_attention_heads;
  int64_t num_experts_per_tok;
  int64_t num_hidden_layers;
  int64_t num_key_value_heads;
  int64_t qk_nope_head_dim;
  int64_t qk_rope_head_dim;
  int64_t v_head_dim;
  bool    norm_topk_prob;
  int64_t first_k_dense_replace;
  int64_t moe_layer_freq;
  float   routed_scaling_factor;
  int64_t n_group;
  int64_t topk_group;
  std::string topk_method;
  std::string scoring_func;
  float       rms_norm_eps;
  float       rope_theta;
  bool        tie_word_embeddings;

  // null in DeepSeek-V2-Lite (direct q_proj), set in DeepSeek-V2 (q_a/q_b)
  std::optional<int64_t>     q_lora_rank;
  std::optional<RopeScaling> rope_scaling;

  DeepseekV2Config(const nlohmann::json &j) : ModelConfig(j) {
    if (model_type != "deepseek_v2") {
      throw std::runtime_error(
        "Invalid model type: expected 'deepseek_v2', got '" + model_type + "'");
    }

    attention_bias          = j.value("attention_bias", false);
    hidden_act              = j.value("hidden_act", std::string("silu"));
    hidden_size             = j["hidden_size"].get<int64_t>();
    intermediate_size       = j["intermediate_size"].get<int64_t>();
    kv_lora_rank            = j["kv_lora_rank"].get<int64_t>();
    max_position_embeddings = j["max_position_embeddings"].get<int64_t>();
    moe_intermediate_size   = j["moe_intermediate_size"].get<int64_t>();
    n_routed_experts =
      j.contains("n_routed_experts") && !j["n_routed_experts"].is_null()
      ? j["n_routed_experts"].get<int64_t>()
      : 0;
    n_shared_experts =
      j.contains("n_shared_experts") && !j["n_shared_experts"].is_null()
      ? j["n_shared_experts"].get<int64_t>()
      : 0;
    num_attention_heads   = j["num_attention_heads"].get<int64_t>();
    num_experts_per_tok   = j.value("num_experts_per_tok", int64_t(0));
    num_hidden_layers     = j["num_hidden_layers"].get<int64_t>();
    num_key_value_heads   = j["num_key_value_heads"].get<int64_t>();
    qk_nope_head_dim      = j["qk_nope_head_dim"].get<int64_t>();
    qk_rope_head_dim      = j["qk_rope_head_dim"].get<int64_t>();
    v_head_dim            = j["v_head_dim"].get<int64_t>();
    norm_topk_prob        = j.value("norm_topk_prob", false);
    first_k_dense_replace = j.value("first_k_dense_replace", int64_t(0));
    moe_layer_freq        = j.value("moe_layer_freq", int64_t(1));
    routed_scaling_factor = j.value("routed_scaling_factor", 1.0F);
    n_group               = j.value("n_group", int64_t(1));
    topk_group            = j.value("topk_group", int64_t(1));
    topk_method           = j.value("topk_method", std::string("greedy"));
    scoring_func          = j.value("scoring_func", std::string("softmax"));
    rms_norm_eps          = j.value("rms_norm_eps", 1e-6F);
    rope_theta            = j.value("rope_theta", 10000.0F);
    tie_word_embeddings   = j.value("tie_word_embeddings", false);

    if (j.contains("q_lora_rank") && !j["q_lora_rank"].is_null()) {
      q_lora_rank = j["q_lora_rank"].get<int64_t>();
    }
    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
      rope_scaling = RopeScaling(j["rope_scaling"]);
      if (rope_scaling->type == "dynamic") {
        // TODO: NTK dynamic rope recomputes the base from the current seq_len
        // at runtime; a static graph cannot express it.
      }
    }

    if (hidden_act != "silu") {
      throw std::runtime_error(
        "deepseek_v2: only hidden_act='silu' is supported, got '" + hidden_act
        + "'");
    }
    // Not ported on purpose: aux_loss / seq_aux / AddAuxiliaryLoss and the
    // MoEGate training branch are training-only; ep_size (expert parallelism)
    // is a multi-device concern, single device here. Training-only paths
    // (dropout, gradient checkpointing) are likewise out of scope.
  }

  /// Combined query head dim (qk_nope + qk_rope), used for softmax scaling
  /// and the K cache layout.
  int64_t q_head_dim() const {
    return qk_nope_head_dim + qk_rope_head_dim;
  }

  static DeepseekV2Config from_huggingface(
    const std::filesystem::path &model_path) {
    return {nlohmann::json::parse(std::ifstream(model_path / "config.json"))};
  }
};

} // namespace finer::models::deepseek_v2
