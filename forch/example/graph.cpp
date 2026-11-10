#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/mat_mul.hpp"

#include <iostream>

int main() {
  using namespace forch;

  const int n = 3;
  const int m = 2;
  const int k = 1;

  float buffer_a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  float buffer_c[] = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};

  auto A = op::load_from_buffer(
    reinterpret_cast<Buffer>(buffer_a), {n, m}, DType::FP32)
             .set_name("input.A");
  auto B = op::create_input({m, k}, DType::FP32).set_name("input.B");
  auto C = op::load_from_buffer(
    reinterpret_cast<Buffer>(buffer_c), {n, k}, DType::FP32)
             .set_name("input.C");

  auto X = op::mat_mul()(A, B).set_name("Result.A*B");
  auto Y = op::add(X, C).set_name("Result.A*B+C");

  std::cout << "Final output tensor Y: " << Y.get_name() << std::endl;

  return 0;
}