#pragma once

#include "sampler.hpp"

#include <memory>

namespace finer::sampler {

struct Chain : protected Sampler,
               private std::vector<std::unique_ptr<Sampler>> {
  using std::vector<std::unique_ptr<Sampler>>::vector;
  using std::vector<std::unique_ptr<Sampler>>::push_back;
  using std::vector<std::unique_ptr<Sampler>>::emplace_back;

  Logits sample(const Logits &logits) override {
    Logits current_logits = logits;
    for (const auto &sampler : *this) {
      current_logits = sampler->sample(current_logits);
    }
    return current_logits;
  }
};

} // namespace finer::sampler