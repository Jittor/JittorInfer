#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

#include "../engine/config.hpp"

namespace finer::server {

struct ServerConfig {
  // --- Transport / handshake ---
  std::string handshake_address = "tcp://127.0.0.1:29550";
  uint32_t    engine_index      = 0;

  // --- ReadyResponse metadata (reported to the vLLM-compat frontend) ---
  int64_t     max_model_len = 4096;
  std::string vllm_version  = "finer-engine-0.1";

  // --- Scheduler ---
  // Mirrors engine::EngineConfig so the slot layout matches the compiled
  // graph — no second parse of the "engine" section, see the ctor.
  int64_t max_num_request = 4;

  // --- Generation defaults ---
  int64_t  max_new_tokens = 512;
  float    temperature    = 1.0f;
  float    top_p          = 0.9f;
  int64_t  top_k          = 50;
  uint64_t seed           = 42;

  ServerConfig() = default;
  ServerConfig(
    const nlohmann::json &j, const engine::EngineConfig &engine_cfg) {
    auto &s           = j["server"];
    handshake_address = s.value("handshake_address", handshake_address);
    vllm_version      = s.value("vllm_version", vllm_version);

    // Scheduler limits come from the engine config actually used to compile
    // the graph — single source of truth for n_context/max_bs/chunk_size.
    max_model_len   = engine_cfg.n_context;
    max_num_request = engine_cfg.max_num_request;

    auto &g        = j["generation"];
    max_new_tokens = g.value("max_new_tokens", max_new_tokens);
    temperature    = g.value("temperature", 1.0f);
    top_p          = g.value("top_p", 0.9f);
    top_k          = g.value("top_k", 50);
    seed           = g.value("seed", seed);
  }
};

} // namespace finer::server
