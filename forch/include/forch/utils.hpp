#pragma once

#include "forch/forch.hpp"

#include <memory>

namespace forch::utils {

struct IntervalTimer {
  IntervalTimer(std::string entity);
  ~IntervalTimer();

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace forch::utils
