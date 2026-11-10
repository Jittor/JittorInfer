#include "forch/pass/debug.hpp"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace forch::pass {

namespace {

bool has_name(
  const std::vector<std::string> &tensor_names, const std::string &name) {
  return std::find(tensor_names.begin(), tensor_names.end(), name)
    != tensor_names.end();
}

void collect_matched_tensors(const Tensor &tensor,
  const std::vector<std::string>          &tensor_names,
  std::unordered_set<uint64_t> &visited, TensorList &matched) {
  if (!visited.insert(tensor.get_id()).second) {
    return;
  }

  if (has_name(tensor_names, tensor.get_name())) {
    matched.push_back(tensor);
  }

  for (const auto &input : tensor.get_inputs()) {
    collect_matched_tensors(input, tensor_names, visited, matched);
  }
}

} // namespace

Graph AppendDebugTensor::operator()(const Graph &graph) const {
  if (tensor_names.empty()) {
    return graph;
  }

  auto outputs = graph.outputs();

  std::unordered_set<uint64_t> visited;
  std::unordered_set<uint64_t> existing_output_ids;
  existing_output_ids.reserve(outputs.size());
  for (const auto &output : outputs) {
    existing_output_ids.insert(output.get_id());
  }

  TensorList matched;
  for (const auto &output : outputs) {
    collect_matched_tensors(output, tensor_names, visited, matched);
  }

  for (const auto &tensor : matched) {
    if (existing_output_ids.insert(tensor.get_id()).second) {
      outputs.push_back(tensor);
    }
  }

  Graph transformed(std::move(outputs));
  transformed.set_name(graph.name());
  return transformed;
}

} // namespace forch::pass