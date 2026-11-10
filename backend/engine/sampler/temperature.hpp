#pragma once

#include "sampler.hpp"

#include <cmath>
#include <stdexcept>

namespace finer::sampler {

struct Temperature : Sampler {
  struct Param {
    float temperature;
  };

  explicit Temperature(const Param &param) : temperature_(param.temperature) {
    if (!std::isfinite(temperature_) || temperature_ <= 0.0F) {
      throw std::invalid_argument("temperature must be finite and > 0");
    }
  }

  Logits sample(const Logits &logits) override {
    Logits scaled = logits;
    for (auto &unit : scaled.data) {
      unit.logit /= temperature_;
    }
    return scaled;
  }

private:
  float temperature_;
};

} // namespace finer::sampler
