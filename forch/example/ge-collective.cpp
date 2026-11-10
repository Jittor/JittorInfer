// Run from the Forch root after activating CANN and allocating two NPUs:
//   python script/forch-ge-run.py --devices 0,1 --tee -- \
//     ./build/example/forch-example-ge-collective
// Replace 0,1 with the allocated physical card IDs. For the parent project's
// build, the executable is ../build/forch/example/forch-example-ge-collective.
// Each rank contributes rank_id + 1; with two ranks, both receive 1 + 2 = 3.

#include "forch/backend/forch-ge.hpp"
#include "forch/distributed.hpp"
#include "forch/ops/collective.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <iostream>

int main() {
  using namespace forch;

  const auto group = process_group_from_env();
  if (group.world_size() < 2) {
    std::cerr << "This example requires at least two ranks.\n";
    return 1;
  }

  GEBackend backend(AscendDevice::from_env());
  auto      x     = op::create_input({1}, DType::FP32).set_name("x");
  auto      y     = op::all_reduce(x, group).set_name("sum");
  auto      graph = backend.build(Graph({y}).set_name("example-ge-collective"));
  graph.compile();
  graph.load();

  const float value  = static_cast<float>(group.rank_id() + 1);
  auto        data   = op::load_from_vector<float>({value}, {1}).set_name("x");
  auto        result = graph.run({data}).get("sum");
  if (result.shape() != Shape{1} || result.dtype() != DType::FP32) {
    std::cerr << "Unexpected all_reduce output shape or dtype.\n";
    return 1;
  }

  const float sum      = *static_cast<const float *>(result.get_data());
  const float world    = static_cast<float>(group.world_size());
  const float expected = world * (world + 1) / 2;
  std::cout << "rank " << group.rank_id() << ": input=" << value
            << ", all_reduce=" << sum << ", expected=" << expected << '\n';
  return sum == expected ? 0 : 1;
}
