#pragma once

/// @file config.hpp
/// @brief HF config.json 里**真正均匀**的字段。防御式解析：家族差异
/// （llama 的 eos 是数组、部分 config 缺 use_cache/bos、VL 嵌在
/// text_config）不允许在公共层炸——那些字段由各模型 config 自己解析。

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace finer::models {

struct ModelConfig {
  std::string model_type;
  /// 缺失时为 0：不静默通过——后续按它建 Embedding/ lm_head，权重装载
  /// 会以形状不匹配大声报错。
  int64_t vocab_size = 0;

  explicit ModelConfig(const nlohmann::json &j)
      : model_type(j.at("model_type").get<std::string>()),
        vocab_size(j.value("vocab_size", int64_t{0})) {}
};

} // namespace finer::models
