#pragma once

#include "forch/forch.hpp"
#include "forch/ops.hpp"

namespace forch::op {

struct Const : public Base {
  virtual Buffer buffer() const = 0;

  void accept(Visitor *visitor, const Tensor &self) const override {
    visitor->visit(this, self);
  }

  Shape shape(const TensorList & /*unused*/ = {}) const override = 0;
};

struct ConstFromBuffer : public Const {
  ConstFromBuffer(Buffer buffer, Shape shape, DType dtype)
      : _buffer(buffer), _shape(std::move(shape)), _dtype(dtype) {}

  Shape shape(const TensorList & /*unused*/) const override {
    return _shape;
  }

  DType dtype(const TensorList & /*unused*/) const override {
    return _dtype;
  }

  Buffer buffer() const override {
    return _buffer;
  }

private:
  Buffer _buffer;
  Shape  _shape;
  DType  _dtype;
};

template <class T, DType = to_dtype<T>()>
struct ConstFromVector : public Const {
  ConstFromVector(std::vector<T> &&data, Shape shape)
      : data(std::move(data)), _shape(std::move(shape)) {}
  ConstFromVector(ConstFromVector &)             = delete;
  ConstFromVector &operator=(ConstFromVector &)  = delete;
  ConstFromVector(ConstFromVector &&)            = default;
  ConstFromVector &operator=(ConstFromVector &&) = default;

  Shape shape(const TensorList & /*unused*/) const override {
    return _shape;
  }

  DType dtype(const TensorList & /*unused*/) const override {
    return to_dtype<T>();
  }

  Buffer buffer() const override {
    return reinterpret_cast<Buffer>(data.data());
  }

private:
  mutable std::vector<T> data;

  Shape _shape;
};

/**
 * 从 buffer 新建一个 Const Tensor，buffer 需要存活到 Graph 运行结束。
 */
inline Tensor load_from_buffer(Buffer buffer, Shape shape, DType dtype) {
  return {
    {}, std::make_unique<ConstFromBuffer>(buffer, std::move(shape), dtype)};
}

/**
 * create Const Tensor from vector,
 * and take ownership of the vector.
 */
template <class T>
inline Tensor load_from_vector(std::vector<T> &&data, Shape shape) {
  return {{},
    std::make_unique<ConstFromVector<T>>(std::move(data), std::move(shape))};
}

template <class T>
inline Tensor load_from_vector(std::vector<T> &&data) {
  const auto size = data.size();
  return {{},
    std::make_unique<ConstFromVector<T>>(
      std::move(data), Shape{static_cast<int64_t>(size)})};
}

} // namespace forch::op