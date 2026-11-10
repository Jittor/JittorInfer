#pragma once
#include "../server/output-sink.hpp"
#include <map>
#include <stdexcept>

namespace finer::tests {

struct RecordingSink : server::OutputSink {
  std::map<std::string, std::vector<int64_t>> tokens;
  std::map<std::string, server::FinishReason> finished;
  void send_outputs(const server::OutputBatch &batch) override {
    for (const auto &out : batch.outputs) {
      if (finished.contains(out.request_id))
        throw std::runtime_error("output after terminal response");
      auto &row = tokens[out.request_id];
      row.insert(row.end(), out.new_token_ids.begin(), out.new_token_ids.end());
      if (out.finish_reason)
        finished.emplace(out.request_id, *out.finish_reason);
    }
  }
  void send_utility(uint64_t, const nlohmann::json &) override {}
  void sync_tokens(std::vector<int64_t> &) override {}
  bool all_have_capacity(bool local) override {
    return local;
  }
};

} // namespace finer::tests
