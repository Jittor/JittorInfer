#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace finer::sampler {

struct Logits {
  struct Unit {
    float   logit;
    int64_t token_id;
  };

  std::vector<Unit> data;

  Logits(std::span<float>                   logits,
    std::optional<std::span<const int64_t>> token_ids = std::nullopt) {
    auto logits_size = static_cast<int64_t>(logits.size());
    if (token_ids.has_value() && token_ids->size() != logits_size) {
      throw std::invalid_argument("token_ids size must match logits size");
    }
    data.reserve(logits_size);
    for (int64_t i = 0; i < logits_size; ++i) {
      data.push_back({logits[i], token_ids.has_value() ? (*token_ids)[i] : i});
    }
  }
};

/**
 * Randomly sample a token from the filtered logit candidates.
 *
 * Computes softmax (with numerical stability) over the surviving logits,
 * then draws from the resulting multinomial distribution.
 *
 * This is the final step after Temperature → TopK → TopP filtering.
 *
 * 随机源必须由调用方传入：采样要**按请求可复现**（seed 由客户端给，scheduler
 * 每槽持一个生成器）， 函数级 static 生成器做不到这一点。模板参数 = 任何
 * UniformRandomBitGenerator （生产用 std::mt19937_64）。
 */
template <class Gen>
inline int64_t random_sample(const Logits &logits, Gen &gen) {
  if (logits.data.empty()) {
    throw std::runtime_error("random_sample: no candidates");
  }

  // Numerically stable softmax
  float max_logit = logits.data[0].logit;
  for (const auto &unit : logits.data) {
    if (unit.logit > max_logit)
      max_logit = unit.logit;
  }

  std::vector<double> probs;
  probs.reserve(logits.data.size());
  double sum = 0.0;
  for (const auto &unit : logits.data) {
    double p = std::exp(static_cast<double>(unit.logit - max_logit));
    probs.push_back(p);
    sum += p;
  }

  // Multinomial sample
  std::uniform_real_distribution<double> dist(0.0, sum);
  double                                 r = dist(gen);

  double cumulative = 0.0;
  for (size_t i = 0; i < probs.size(); ++i) {
    cumulative += probs[i];
    if (r <= cumulative) {
      return logits.data[i].token_id;
    }
  }

  // Fallback (floating-point edge case)
  return logits.data.back().token_id;
}

struct Sampler {
  virtual ~Sampler() = default;

  virtual Logits sample(const Logits &logits) = 0;

  // optional callback to update internal state after a token is sampled
  // used for samplers like repetition / presence / frequency penalty
  virtual void apply(int64_t token_id) {}
};

} // namespace finer::sampler