#pragma once

#include "forch/forch.hpp"
#include <memory>

namespace forch::huggingface {

struct Safetensors {
  explicit Safetensors(const std::string &path);
  Tensor load(const std::string &name);

private:
  struct Impl;
  std::shared_ptr<Impl> impl;
};

} // namespace forch::huggingface