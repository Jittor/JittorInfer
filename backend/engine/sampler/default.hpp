#pragma once

#include "chain.hpp"
#include "temperature.hpp"
#include "top-k.hpp"
#include "top-p.hpp"

#include <cstdint>
#include <memory>

namespace finer::sampler {

/**
 * Default sampler with chain:
 * Temperature scaling -> Top-K filtering -> Top-P (nucleus) filtering
 */
struct Default : protected Chain {
  struct Param {
    int64_t top_k;
    float   top_p;
    float   temperature;
  };

  using Chain::apply;
  using Chain::sample;

  explicit Default(const Param &param) {
    emplace_back(
      std::make_unique<Temperature>(Temperature::Param{param.temperature}));
    emplace_back(std::make_unique<TopK>(TopK::Param{param.top_k}));
    emplace_back(std::make_unique<TopP>(TopP::Param{param.top_p}));
  }
};

} // namespace finer::sampler