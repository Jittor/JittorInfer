#pragma once

#include "sampler.hpp"

#include <algorithm>
#include <stdexcept>

namespace finer::sampler {

struct TopK : Sampler {
  struct Param {
    int64_t top_k;
  };

  explicit TopK(const Param &param) : top_k_(param.top_k) {
    if (top_k_ < 0) {
      throw std::invalid_argument("top_k must be >= 0");
    }
  }

  Logits sample(const Logits &logits) override {
    if (top_k_ == 0) {
      return logits;
    }

    const auto candidate_size = static_cast<int64_t>(logits.data.size());
    if (top_k_ >= candidate_size) {
      return logits;
    }

    Logits filtered = logits;
    std::sort(filtered.data.begin(), filtered.data.end(),
      [](const Logits::Unit &lhs, const Logits::Unit &rhs) {
        return lhs.logit > rhs.logit;
      });
    filtered.data.resize(static_cast<size_t>(top_k_));
    return filtered;
  }

private:
  int64_t top_k_;
};

} // namespace finer::sampler
