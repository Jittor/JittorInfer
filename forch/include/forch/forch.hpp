#pragma once

#include <cstdint>
#include <memory>
#include <spdlog/logger.h>
#include <string>
#include <type_traits>
#include <vector>

namespace forch {

std::shared_ptr<spdlog::logger> logger();

using Shape  = std::vector<int64_t>;
using Buffer = uint8_t *;

enum class DType { FP64, FP32, FP16, BF16, INT32, INT64, INT8, BOOL };
template <class T>
constexpr DType to_dtype() {
  if constexpr (std::is_same_v<T, double>) {
    return DType::FP64;
  } else if constexpr (std::is_same_v<T, float>) {
    return DType::FP32;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return DType::INT32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return DType::INT64;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return DType::INT8;
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type for to_dtype");
  }
}

struct Tensor;
using TensorList = std::vector<Tensor>;

namespace op {
struct Base;
}

struct Tensor {
  uint64_t           get_id() const;
  const std::string &get_name() const;

  Tensor(TensorList input, std::unique_ptr<op::Base> op);

  const TensorList &get_inputs() const;
  const op::Base   *get_op() const;
  const void       *get_data() const;
  const Shape      &shape() const;
  int64_t           dim() const;
  DType             dtype() const;

  Tensor &set_name(const std::string &name);

  struct Impl;
  std::shared_ptr<Impl> _impl;
};

struct TensorNameSetter {
  std::string name;
  Tensor      operator()(Tensor &&tensor) const {
    tensor.set_name(name);
    return tensor;
  }
};
inline TensorNameSetter set_name(const std::string &name) {
  return TensorNameSetter{name};
}

} // namespace forch