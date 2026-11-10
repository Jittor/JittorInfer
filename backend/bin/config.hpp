#pragma once

#include "../engine/config.hpp"
#include "../server/config.hpp"

#include <filesystem>

namespace finer::bin {

struct Config {
  // required: model path
  std::filesystem::path model;

  // optional: engine-specific configuration
  engine::EngineConfig engine;

  // 模型构造选项（不透明 dict）。bin/engine 层不解释其内容，由各模型的
  // Config 自行定义 schema 并解析 —— vLLM hf-overrides 的模式：部署侧显式
  // 覆写，避免把单个模型的特例泄漏进通用配置层。
  // 现有消费者：qwen3.5 的 gdn_param_dtype（fp32|bf16，GDN 标量参数存储
  // dtype）。
  nlohmann::json model_options = nlohmann::json::object();

  // server-side config (transport, scheduler limits, generation defaults).
  // Built from the same JSON; mirrors the engine limits above.
  server::ServerConfig server;

  Config(const nlohmann::json &j) {
    model = j["model"].get<std::string>();
    if (j.contains("model_options")) {
      model_options = j["model_options"];
    }
    if (j.contains("engine")) {
      engine = engine::EngineConfig(j["engine"]);
    }
    if (j.contains("server")) {
      server = server::ServerConfig(j, engine);
    }
  }
};

} // namespace finer::bin