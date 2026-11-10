#pragma once

#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace finer::sampler {

struct TopP : Sampler {
  struct Param {
    float top_p;
  };

  explicit TopP(const Param &param) : top_p_(param.top_p) {
    if (!std::isfinite(top_p_) || top_p_ <= 0.0F || top_p_ > 1.0F) {
      throw std::invalid_argument("top_p must be finite and in (0, 1]");
    }
  }

  Logits sample(const Logits &logits) override {
    if (logits.data.empty() || top_p_ >= 1.0F) {
      return logits;
    }

    Logits filtered = logits;
    std::sort(filtered.data.begin(), filtered.data.end(),
      [](const Logits::Unit &lhs, const Logits::Unit &rhs) {
        return lhs.logit > rhs.logit;
      });

    const float         max_logit = filtered.data.front().logit;
    std::vector<double> exp_values;
    exp_values.reserve(filtered.data.size());

    double exp_sum = 0.0;
    for (const auto &unit : filtered.data) {
      const double shifted = static_cast<double>(unit.logit - max_logit);
      const double value   = std::exp(shifted);
      exp_values.push_back(value);
      exp_sum += value;
    }

    double cumulative = 0.0;
    size_t keep_count = 0;
    for (double exp_value : exp_values) {
      cumulative += exp_value / exp_sum;
      ++keep_count;
      if (cumulative >= static_cast<double>(top_p_)) {
        break;
      }
    }

    filtered.data.resize(keep_count);
    return filtered;
  }

private:
  float top_p_;
};

} // namespace finer::sampler
