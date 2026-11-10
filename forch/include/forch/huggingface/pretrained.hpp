#pragma once

#include "forch/forch.hpp"
#include "forch/nn/module.hpp"

#include <memory>

namespace forch::huggingface {

struct Pretrained : public nn::Loader {
  explicit Pretrained(const std::string &path);
  Tensor load(const std::string &name) override;

private:
  struct Impl;
  std::shared_ptr<Impl> impl;
};

} // namespace forch::huggingface