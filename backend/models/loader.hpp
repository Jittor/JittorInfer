#pragma once

/// @file loader.hpp
/// @brief RunnerLoader — 唯一的模型注册点：model_type → builder。
/// 接新模型 = 新目录（models/<name>/ 三件套）+ kRunners 里一行。

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "../engine/config.hpp"
#include "../engine/model-runner.hpp"
#include "deepseek-v2/runner-deepseekv2.hpp"
#include "llama/runner-llama.hpp"
#include "qwen3.5/runner-qwen35.hpp"
#include "qwen3/runner-qwen3.hpp"

namespace finer::models {

/// 从模型目录的 config.json 解析 model_type（生产路径上唯一需要的公共
/// 解析；其余字段由各模型的 config 自己 .value()/.get()）。
inline std::string resolve_model_type(const nlohmann::json &j) {
  return j.at("model_type").get<std::string>();
}

namespace detail {

template <class R>
std::unique_ptr<engine::ModelRunner> build_runner(
  const engine::EngineConfig &cfg, const std::filesystem::path &model_dir,
  const nlohmann::json &model_options) {
  return std::make_unique<R>(cfg, model_dir, model_options);
}

} // namespace detail

struct RunnerEntry {
  const char *model_type;
  std::unique_ptr<engine::ModelRunner> (*build)(const engine::EngineConfig &,
    const std::filesystem::path &, const nlohmann::json &);
};

/// 静态表，不用自注册：header-only + 一文件一可执行，自注册会把所有模型
/// 编进每个二进制。
inline constexpr auto kRunners = std::to_array<RunnerEntry>({
  {"qwen3", &detail::build_runner<qwen3::Qwen3Runner>},
  {"qwen3_moe", &detail::build_runner<qwen3::Qwen3Runner>},
  {"deepseek_v2", &detail::build_runner<deepseek_v2::DeepseekV2Runner>},
  {"llama", &detail::build_runner<llama::LlamaRunner>},
  {"qwen3_5", &detail::build_runner<qwen35::Qwen35Runner>},
  {"qwen3_5_moe", &detail::build_runner<qwen35::Qwen35Runner>},
  {"qwen3_5_moe_text", &detail::build_runner<qwen35::Qwen35Runner>},
  {"qwen3_5_text", &detail::build_runner<qwen35::Qwen35Runner>},
});

inline std::unique_ptr<engine::ModelRunner> load_runner(
  const std::filesystem::path &model_dir, const engine::EngineConfig &cfg,
  const nlohmann::json &model_options = {}) {
  const auto config_json =
    nlohmann::json::parse(std::ifstream(model_dir / "config.json"));
  const auto type = resolve_model_type(config_json);
  for (const auto &entry : kRunners)
    if (type == entry.model_type)
      return entry.build(cfg, model_dir, model_options);
  throw std::runtime_error("Unsupported model_type: " + type);
}

} // namespace finer::models
