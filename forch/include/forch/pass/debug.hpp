#pragma once

#include "forch/graph.hpp"

#include <utility>
#include <vector>

namespace forch::pass {

/**
 * 将指定名称的张量追加到图输出中，便于调试。
 *
 * 该 pass 从输入图的 outputs 出发遍历可达张量。
 * 若某个张量名称位于 `tensor_names` 中，则将该张量追加到输出列表。
 * 追加过程按张量 id 去重，避免重复输出同一节点。
 *
 * 返回的新图会保留原图名称。
 */
struct AppendDebugTensor {
  /// 需要追加到图输出中的张量名称列表。
  std::vector<std::string> tensor_names;

  /// 使用用于匹配的张量名称列表构造该 pass。
  AppendDebugTensor(std::vector<std::string> tensor_names)
      : tensor_names(std::move(tensor_names)) {}

  /**
   * 应用到 `graph`，返回追加了调试输出的新图。
   */
  Graph operator()(const Graph &graph) const;
};

} // namespace forch::pass