// NPU regression: recycled and never-used slots must start from the same
// state as a fresh request, even after padding ran while they were idle.
#include "../models/qwen3.5/runner-qwen35.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("usage: test_gdn_slot_lifecycle config.json");
    const auto j = nlohmann::json::parse(std::ifstream(argv[1]));
    const finer::engine::EngineConfig cfg(j.at("engine"));
    if (cfg.max_num_request < 3 || cfg.n_context < 16)
      throw std::runtime_error("test requires B>=3 and C>=16");
    const auto options = j.value("model_options", nlohmann::json::object());
    const auto model   = j.at("model").get<std::string>();
    const finer::models::qwen35::Qwen35Config model_cfg(
      nlohmann::json::parse(std::ifstream(model + "/config.json")), options);
    const auto layers = cfg.n_model_layers > 0
      ? std::min<int64_t>(cfg.n_model_layers, model_cfg.num_hidden_layers)
      : model_cfg.num_hidden_layers;
    if (std::all_of(model_cfg.layer_is_full_attention.begin(),
          model_cfg.layer_is_full_attention.begin() + layers,
          [](bool full) { return full; }))
      throw std::runtime_error("test requires an executed GDN layer");
    finer::models::qwen35::Qwen35Runner runner(cfg, model, options);
    if (!cfg.optimization.fused_gdn_attention
      && (runner.graph_infos().size() != 1
        || runner.graph_infos().front().max_tokens_per_slot != 1))
      throw std::runtime_error(
        "FP32 eager GDR must expose only a one-token graph");
    const std::vector<int64_t> prompt{101, 102, 103};
    std::vector<int64_t>       positions(cfg.max_num_request, 0);
    auto admit = [&](int64_t slot) { runner.admit_request(slot, prompt, {}); };
    auto step  = [&](bool slot1, bool slot2) {
      std::vector<std::vector<int64_t>>    tokens(cfg.max_num_request);
      std::vector<finer::engine::SlotStep> slots(cfg.max_num_request);
      tokens[0] = {300 + positions[0]};
      for (int64_t sid = 1; sid <= 2; ++sid)
        if (sid == 1 ? slot1 : slot2)
          tokens[sid] = {prompt.at(positions[sid])};
      for (int64_t sid = 0; sid < cfg.max_num_request; ++sid)
        slots[sid] = {positions[sid], tokens[sid]};
      auto out = runner.step(0, slots);
      for (const auto &row : out.vals)
        for (auto v : row)
          if (!std::isfinite(v))
            throw std::runtime_error("nonfinite output");
      for (int64_t sid = 0; sid < cfg.max_num_request; ++sid)
        positions[sid] += tokens[sid].size();
      return out;
    };
    admit(0);
    admit(1);
    std::vector<finer::engine::StepOutput> reference;
    for (size_t i = 0; i < prompt.size(); ++i)
      reference.push_back(step(true, false));
    runner.reset_slot(1); // release; slot 2 has never admitted a request
    positions[1] = 0;
    for (int i = 0; i < 5; ++i)
      step(false, false); // another request keeps padding execution alive
    admit(1);
    admit(2);
    // Eager attention reduces over a flattened cache; slot placement can
    // change FP32 summation rounding. BSND should remain bitwise identical.
    const float tolerance = cfg.optimization.fuse_attention ? 0.F : 1e-5F;
    for (size_t i = 0; i < prompt.size(); ++i) {
      const auto out = step(true, true);
      for (int64_t sid : {1, 2})
        if (out.indices[sid] != reference[i].indices[1]
          || !std::equal(out.vals[sid].begin(), out.vals[sid].end(),
            reference[i].vals[1].begin(), reference[i].vals[1].end(),
            [=](float a, float b) { return std::abs(a - b) <= tolerance; }))
          throw std::runtime_error("request state leaked into slot "
            + std::to_string(sid) + " at token " + std::to_string(i));
    }
    std::cout << "PASS GDN lifecycle: recycled and first-use slots after idle "
                 "padding, 6 output rows match fresh admission; tolerance="
              << tolerance << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
