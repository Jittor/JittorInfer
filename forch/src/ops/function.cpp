#include "forch/ops/function.hpp"
#include "forch/exception.hpp"
#include "forch/ops/cast.hpp"
#include "forch/ops/const.hpp"
#include <algorithm>

namespace forch::op {

Shape broadcast_shape(const TensorList &inputs) {
  if (inputs.empty()) {
    return {};
  }

  if (inputs.size() == 1) {
    return inputs[0].shape();
  }

  auto res = std::max_element(
    inputs.begin(), inputs.end(), [](const Tensor &a, const Tensor &b) {
      return a.shape().size() < b.shape().size();
    })->shape();

  for (size_t i = 1; i <= res.size(); i++) {
    for (const auto &input : inputs) {
      auto x =
        input.shape().size() > i ? input.shape()[input.shape().size() - i] : 1;
      if (x != 1 && res[res.size() - i] != 1 && x != res[res.size() - i]) {
        // 保留形状信息，便于定位是哪个算子/哪两个输入不兼容
        throw exception("Incompatible shapes for broadcasting: dim "
          + std::to_string(i) + " got " + std::to_string(x) + " vs "
          + std::to_string(res[res.size() - i]));
      } else if (x != 1) {
        res[res.size() - i] = x;
      }
    }
  }

  return res;
}

DType broadcast_dtype(const TensorList &inputs) {
  if (inputs.empty()) {
    return DType::FP32; // default dtype
  }

  DType res = inputs[0].dtype();
  for (const auto &input : inputs) {
    if (input.dtype() != res) {
      throw exception("Incompatible dtypes. Forch does not support implicit "
                      "dtype promotion.");
    }
  }
  return res;
}

Tensor NegBuilder::operator()(const Tensor &input) const {
  // Neg can be implemented as Mul with -1, so we can reuse the broadcasting
  // logic

  const auto dtype = input.dtype();
  return mul(
    input, op::load_from_vector(std::vector<float>{-1.0F}) | op::cast(dtype));
}

} // namespace forch::op