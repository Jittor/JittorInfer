
#include "forch/forch.hpp"
#include "forch/exception.hpp"
#include "forch/ops/runtime.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace forch {

std::shared_ptr<spdlog::logger> logger() {
  static auto logger = []() {
    spdlog::cfg::load_env_levels("FORCH_LOG_LEVEL");
    auto logger = spdlog::stdout_color_mt("forch");
    logger->info("forch logger initialized with level: {}",
      spdlog::level::to_string_view(logger->level()));
    return logger;
  }();
  return logger;
}

namespace {
uint64_t _g_gen_id() {
  static std::atomic<uint64_t> g_id = 0;
  return g_id.fetch_add(1);
}
} // namespace

struct Tensor::Impl {
  uint64_t    id;
  std::string name;

  TensorList                input;
  std::unique_ptr<op::Base> op;
  Shape                     shape;
  DType                     dtype;

  Impl(TensorList input, std::unique_ptr<op::Base> op)
      : id(_g_gen_id()), name(""), input(std::move(input)), op(std::move(op)) {
    // shape must be computed to check for errors in op construction early
    shape = this->op->shape(this->input);
    dtype = this->op->dtype(this->input);
  }
};

uint64_t Tensor::get_id() const {
  return _impl->id;
}

const std::string &Tensor::get_name() const {
  return _impl->name;
}

Tensor::Tensor(TensorList input, std::unique_ptr<op::Base> op)
    : _impl(std::make_shared<Tensor::Impl>(std::move(input), std::move(op))) {}

const TensorList &Tensor::get_inputs() const {
  return _impl->input;
}

const op::Base *Tensor::get_op() const {
  return _impl->op.get();
}

const void *Tensor::get_data() const {
  auto *op = dynamic_cast<op::Runtime *>(_impl->op.get());
  if (op == nullptr) {
    throw exception(
      "Tensor::get_data can only be called on Tensors with Runtime op");
  }
  return op->get_data();
}

const Shape &Tensor::shape() const {
  return _impl->shape;
}

int64_t Tensor::dim() const {
  return static_cast<int64_t>(_impl->shape.size());
}

DType Tensor::dtype() const {
  return _impl->dtype;
}

Tensor &Tensor::set_name(const std::string &name) {
  _impl->name = name;
  return *this;
}

} // namespace forch