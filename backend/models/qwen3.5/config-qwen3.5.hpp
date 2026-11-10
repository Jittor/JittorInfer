#pragma once

#include "../config.hpp"

#include <forch/forch.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace finer::models::qwen35 {

/**
 * Qwen3.5 文本塔配置。
 *
 * 兼容两种 config.json 形态：
 *   - 外层多模态壳 model_type="qwen3_5"，文本字段在 text_config 子对象；
 *   - 纯文本 model_type="qwen3_5_text"，字段平铺。
 * MoE 对应 qwen3_5_moe / qwen3_5_moe_text，共用注意力配置。
 * 归一化后统一从 text 读取；tie_word_embeddings 在外层时向下兼容。
 * （ModelConfig 只需要 model_type/vocab_size，bos/eos 在这里没有消费者。）
 */
struct Qwen35Config : public ModelConfig {
  // full-attention 层
  int64_t head_dim;
  int64_t num_attention_heads;
  int64_t num_key_value_heads;
  bool    attn_output_gate;
  bool    attention_bias;
  // Gated DeltaNet 层
  int64_t linear_num_key_heads;
  int64_t linear_num_value_heads;
  int64_t linear_key_head_dim;
  int64_t linear_value_head_dim;
  int64_t linear_conv_kernel_dim;
  // 公共
  int64_t           hidden_size;
  int64_t           intermediate_size;
  int64_t           num_hidden_layers;
  float             rms_norm_eps;
  float             rope_theta;            // rope_parameters.rope_theta
  float             partial_rotary_factor; // 0.25 → rotary_dim = head_dim / 4
  bool              tie_word_embeddings;   // 多模态壳在外层
  std::vector<bool> layer_is_full_attention;

  bool    is_moe      = false;
  int64_t num_experts = 0, num_experts_per_tok = 0;
  int64_t moe_intermediate_size = 0, shared_expert_intermediate_size = 0;
  std::vector<int64_t> mlp_only_layers;

  // GDN“标量”参数（linear_attn.A_log / gated-norm norm.weight）在检查点里
  // 的存储 dtype。HF 对这类参数声明 fp32、加载时静默 cast；我们改为在部署
  // 配置（bin 的 model_options dict，schema 由本类定义）里显式声明，注册
  // dtype == 文件 dtype → 加载保持 mmap 别名，零拷贝零转换。实测 0.8B 与
  // Qwen3.5-27B 存 F32（默认值即适配），Qwen3.8-27B 全系存 BF16（配
  // "bf16"）。声明与文件不符时 Parameter 严格校验 fail-fast。
  forch::DType gdn_param_dtype = forch::DType::FP32;

  // 派生维度（缓存便捷值）
  int64_t key_dim;    // linear_num_key_heads * linear_key_head_dim
  int64_t value_dim;  // linear_num_value_heads * linear_value_head_dim
  int64_t conv_dim;   // 2 * key_dim + value_dim
  int64_t rotary_dim; // head_dim * partial_rotary_factor

  explicit Qwen35Config(const nlohmann::json &raw,
    const nlohmann::json &options = nlohmann::json::object())
      : Qwen35Config(raw, normalize(raw), options) {}

  static Qwen35Config from_huggingface(
    const std::filesystem::path &model_path) {
    return Qwen35Config(
      nlohmann::json::parse(std::ifstream(model_path / "config.json")));
  }

private:
  Qwen35Config(const nlohmann::json &raw, const nlohmann::json &text,
    const nlohmann::json &options)
      : ModelConfig(text) {
    is_moe = model_type == "qwen3_5_moe_text";
    if (model_type != "qwen3_5_text" && !is_moe) {
      throw std::runtime_error("Invalid model type: expected "
                               "qwen3_5_text or qwen3_5_moe_text, got '"
        + model_type + "'");
    }

    head_dim               = text["head_dim"].get<int64_t>();
    num_attention_heads    = text["num_attention_heads"].get<int64_t>();
    num_key_value_heads    = text["num_key_value_heads"].get<int64_t>();
    attn_output_gate       = text.value("attn_output_gate", false);
    attention_bias         = text.value("attention_bias", false);
    linear_num_key_heads   = text["linear_num_key_heads"].get<int64_t>();
    linear_num_value_heads = text["linear_num_value_heads"].get<int64_t>();
    linear_key_head_dim    = text["linear_key_head_dim"].get<int64_t>();
    linear_value_head_dim  = text["linear_value_head_dim"].get<int64_t>();
    linear_conv_kernel_dim = text["linear_conv_kernel_dim"].get<int64_t>();
    hidden_size            = text["hidden_size"].get<int64_t>();
    if (is_moe) {
      num_experts           = text.at("num_experts").get<int64_t>();
      num_experts_per_tok   = text.at("num_experts_per_tok").get<int64_t>();
      moe_intermediate_size = text.at("moe_intermediate_size").get<int64_t>();
      shared_expert_intermediate_size =
        text.at("shared_expert_intermediate_size").get<int64_t>();
      mlp_only_layers = text.value("mlp_only_layers", std::vector<int64_t>{});
      if (num_experts <= 0 || num_experts_per_tok <= 0
        || num_experts_per_tok > num_experts || moe_intermediate_size <= 0
        || shared_expert_intermediate_size <= 0)
        throw std::invalid_argument("Qwen3.5: invalid MoE dimensions");
    }
    intermediate_size = text.value("intermediate_size", moe_intermediate_size);
    if (intermediate_size <= 0
      || (!mlp_only_layers.empty() && !text.contains("intermediate_size")))
      throw std::invalid_argument(
        "Qwen3.5: dense layers require intermediate_size");
    num_hidden_layers = text["num_hidden_layers"].get<int64_t>();
    for (auto idx : mlp_only_layers)
      if (idx < 0 || idx >= num_hidden_layers)
        throw std::invalid_argument(
          "Qwen3.5: mlp_only_layers index out of range");
    rms_norm_eps        = text["rms_norm_eps"].get<float>();
    tie_word_embeddings = raw.value(
      "tie_word_embeddings", text.value("tie_word_embeddings", false));

    const auto &rope = text.at("rope_parameters");
    if (rope.value("rope_type", std::string("default")) != "default") {
      throw std::runtime_error(
        "qwen3_5: only 'default' rope_type is supported (text path)");
    }
    rope_theta            = rope["rope_theta"].get<float>();
    partial_rotary_factor = rope.value("partial_rotary_factor", 1.0F);

    // ── 部署侧选项（bin 的 model_options dict，schema 由本类定义）───────
    // 未知值直接抛错而不是降级默认：typo 成 bf6 之类若静默用 fp32，
    // 到 Parameter 加载才报 mismatch，排查反而绕远。
    if (options.contains("gdn_param_dtype")) {
      auto s = options["gdn_param_dtype"].get<std::string>();
      if (s == "fp32" || s == "f32" || s == "float32") {
        gdn_param_dtype = forch::DType::FP32;
      } else if (s == "bf16" || s == "bfloat16") {
        gdn_param_dtype = forch::DType::BF16;
      } else {
        throw std::runtime_error(
          "qwen3_5: unknown model_options.gdn_param_dtype '" + s
          + "' (expected fp32|bf16)");
      }
    }

    key_dim    = linear_num_key_heads * linear_key_head_dim;
    value_dim  = linear_num_value_heads * linear_value_head_dim;
    conv_dim   = key_dim * 2 + value_dim;
    rotary_dim = static_cast<int64_t>(
      static_cast<double>(head_dim) * partial_rotary_factor);

    const auto &layer_types = text.at("layer_types");
    layer_is_full_attention.reserve(layer_types.size());
    for (const auto &t : layer_types) {
      auto type = t.get<std::string>();
      if (type != "linear_attention" && type != "full_attention") {
        throw std::runtime_error("qwen3_5: unknown layer type '" + type + "'");
      }
      layer_is_full_attention.push_back(type == "full_attention");
    }
    if (static_cast<int64_t>(layer_is_full_attention.size())
      != num_hidden_layers) {
      throw std::runtime_error(
        "qwen3_5: layer_types size != num_hidden_layers");
    }

    auto act = text.value("hidden_act", std::string("silu"));
    if (act != "silu") {
      throw std::runtime_error("qwen3_5: unsupported hidden_act '" + act + "'");
    }
  }

  // 外层壳 → text 平铺
  static nlohmann::json normalize(const nlohmann::json &raw) {
    nlohmann::json text =
      raw.contains("text_config") ? raw.at("text_config") : raw;
    return text;
  }
};

} // namespace finer::models::qwen35
