#pragma once
#include "protocol.hpp"

namespace finer::server {

// Scheduler output and rank coordination, independent of socket ownership.
// In-process tests use the same scheduler with a recording sink.
class OutputSink {
public:
  virtual ~OutputSink()                               = default;
  virtual void send_outputs(const OutputBatch &batch) = 0;
  virtual void send_utility(uint64_t call_id, const nlohmann::json &result) = 0;
  virtual void sync_tokens(std::vector<int64_t> &tokens)                    = 0;
  virtual bool all_have_capacity(bool local)                                = 0;
};

} // namespace finer::server
