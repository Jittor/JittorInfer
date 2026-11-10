// Turning a Forch DAG into a GE graph: the Visitor that emits Es nodes, the
// GEGraph handle, and the build options that go with AddGraph. The backend,
// its session and the execution path live in backend.cpp.

// Forch Header
#include "forch/backend/forch-ge.hpp"
#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/ops.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/multi_output.hpp"
#include "forch/ops/runtime.hpp"

#include "../graph.hpp"

// Forch-ge Header
#include "forch-ge-common.hpp"
#include "op-registry.hpp"

// CANN C-only Header
#include "esb_funcs.h"
#include "ge_session_shim.h"
#include "graph/c_types.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace forch {

namespace {

const char *precision_mode_str(GEBackend::GEBuildOptions::PrecisionMode pm) {
  using PM = GEBackend::GEBuildOptions::PrecisionMode;
  switch (pm) {
  case PM::ForceFP16:
    return "fp16";
  case PM::AutoToFP16:
    return "mixed_float16";
  case PM::AutoToBF16:
    return "mixed_bfloat16";
  case PM::Origin:
    return "origin";
  }
  return "origin";
}

const char *const_lifecycle_str(GEBackend::GEBuildOptions::ConstLifecycle lc) {
  using LC = GEBackend::GEBuildOptions::ConstLifecycle;
  switch (lc) {
  case LC::Session:
    return "session";
  case LC::Graph:
    return "graph";
  }
  return "session";
}

std::string format_host_input_indexes(const std::vector<int64_t> &indexes) {
  std::string result;
  for (size_t i = 0; i < indexes.size(); ++i) {
    if (i > 0) {
      result += ';';
    }
    result += std::to_string(indexes[i]);
  }
  return result;
}

} // namespace

GEBackend::GEGraph::GEGraph(std::shared_ptr<Impl> impl)
    : Backend::Graph(std::move(impl)) {}

GEBackend::GEGraph::Impl *GEBackend::GEGraph::impl() {
  return static_cast<Impl *>(_impl.get());
}

const GEBackend::GEGraph::Impl *GEBackend::GEGraph::impl() const {
  return static_cast<const Impl *>(_impl.get());
}

GEBackend::GEGraph::Impl::~Impl() {
  try {
    if (ge_state == GEBackendState::LOADED) {
      ge_backend.impl()->unload_graph(this);
    }
  } catch (const std::exception &e) {
    logger()->error("Failed to unload GE graph");
  }
}

namespace detail {

struct GEGraphBuilder : public op::Visitor {
  EsCGraphBuilder          *builder;
  GEBackend::GEGraph::Impl *graph;
  std::vector<int64_t>      auto_H2D_copy_input_indexes;

  /// 已物化的多输出节点 → 全部输出 holders。按 node 指针去重,
  /// 保证同一逻辑节点(如同一 TopK)每图只调一次底层建图函数。
  std::map<const op::MultiOutputBase *, std::vector<EsCTensorHolder *>>
    built_multi_;

  explicit GEGraphBuilder(GEBackend::GEGraph::Impl *graph)
      : builder(EsCreateGraphBuilder(graph->graph_name.c_str())), graph(graph) {
    if (builder == nullptr) {
      throw forch::exception(
        std::string("forch-ge: EsCreateGraphBuilder failed: ")
        + forch_ge_last_error());
    }
  }

  GEGraphBuilder(const GEGraphBuilder &)            = delete;
  GEGraphBuilder &operator=(const GEGraphBuilder &) = delete;
  GEGraphBuilder(GEGraphBuilder &&)                 = delete;
  GEGraphBuilder &operator=(GEGraphBuilder &&)      = delete;

  ~GEGraphBuilder() override {
    if (builder != nullptr) {
      EsDestroyGraphBuilder(builder);
    }
  }

  void visit(const op::Const *op, const Tensor &self) override {
    const auto      &shape = op->shape();
    const CDtypeDesc desc  = CDtypeDesc(op->dtype());

    int64_t shape_size = 1;
    for (auto d : shape) {
      shape_size *= d;
    }
    const size_t data_size = static_cast<size_t>(shape_size) * desc.size;

    // 内容哈希（FNV-1a）：既用于内容寻址命名，也用于常量审计。
    // **必须把 dtype 与 shape 一并计入**：只哈希字节时，"字节相同但形状/类型
    // 不同"的两个常量会得到同名，而 GE 的 session 级 var 按名复用**首个创建者**
    // （不校验形状）→ 后建的图会读到形状不对的变量 → 乱码。
    // 例：同一段 int32 值在不同图里做 [N] / [1,N] / [N,1] 等不同形状的常量。
    const auto *cbytes = static_cast<const uint8_t *>(op->buffer());
    uint64_t    h      = 1469598103934665603ULL;
    auto        mix64  = [&h](uint64_t v) {
      for (int i = 0; i < 8; ++i) {
        h ^= (v >> (i * 8)) & 0xFFULL;
        h *= 1099511628211ULL;
      }
    };
    mix64(static_cast<uint64_t>(op->dtype()));
    mix64(static_cast<uint64_t>(shape.size()));
    for (auto d : shape) {
      mix64(static_cast<uint64_t>(d));
    }
    for (size_t k = 0; k < data_size; ++k) {
      h ^= cbytes[k];
      h *= 1099511628211ULL;
    }
    // 常量按**内容哈希**命名（内容寻址）：
    //   - 同内容跨图共享（session 级 var 复用 ✓，权重不翻倍）
    //   - 异内容名字不同 → 不串扰（修复：ES 自动名 Const_<N> 在多图会话中
    //     同名异值，后建图读到先建图的常量 → 乱码）
    std::vector<uint8_t> cname(32);
    std::snprintf(reinterpret_cast<char *>(cname.data()), cname.size(),
      "cst_%016llx", static_cast<unsigned long long>(h));
    EsCTensorHolder *value  = forch_ge_create_const_named(builder, op->buffer(),
       data_size, shape.data(), shape.size(), desc.type,
       reinterpret_cast<const char *>(cname.data()));
    auto            *holder = value;
    if (holder == nullptr) {
      throw forch::exception(
        std::string("forch-ge: forch_ge_create_const_named failed for Const: ")
        + forch_ge_last_error());
    }
    register_es_tensor(self, holder);
  }

  void visit(const op::Input *op, const Tensor &self) override {
    auto input_ge_id = static_cast<int64_t>(graph->input_id_ge_to_forch.size());
    graph->input_id_ge_to_forch.push_back(self);

    if (op->forch_ge_auto_H2D_copy_enabled()) {
      auto_H2D_copy_input_indexes.push_back(input_ge_id);
    }

    const auto      &shape = op->shape();
    const CDtypeDesc desc  = CDtypeDesc(op->dtype());

    EsCTensorHolder *holder = EsCreateGraphInputWithDetails(builder,
      input_ge_id, self.get_name().c_str(), /*type=*/nullptr, desc.type,
      C_FORMAT_ND, shape.data(), static_cast<int64_t>(shape.size()));
    if (holder == nullptr) {
      throw forch::exception(
        std::string("forch-ge: EsCreateGraphInputWithDetails failed: ")
        + forch_ge_last_error());
    }
    register_es_tensor(self, holder);
  }

  /// 物化 self 的全部输入并收集它们的 holder。GEOpContext 只是个视图,所以
  /// 返回值必须活到 emit 调用结束。
  std::vector<EsCTensorHolder *> input_holders(const Tensor &self) {
    std::vector<EsCTensorHolder *> holders;
    holders.reserve(self.get_inputs().size());
    for (const auto &input : self.get_inputs()) {
      holders.push_back(build_es_tensor(input));
    }
    return holders;
  }

  void visit(const op::Function *op, const Tensor &self) override {
    const auto holders = input_holders(self);
    register_es_tensor(
      self, GEOpRegistry::emit({builder, holders, self.get_inputs()}, op));
  }

  void visit(const op::MultiOutputPort *op, const Tensor &self) override {
    // 端口张量的输入 == 多输出节点的输入。
    const auto holders = input_holders(self);

    // 节点只物化一次(按 node 指针去重)。
    const auto *node = op->node.get();
    auto        it   = built_multi_.find(node);
    if (it == built_multi_.end()) {
      it = built_multi_
             .emplace(node,
               GEOpRegistry::emit({builder, holders, self.get_inputs()}, node))
             .first;
    }

    // 本端口按自身 tensor id 注册进 pool,供后续复用 / 图输出。路数与
    // n_outputs() 一致由注册表保证,这里只需防住端口自身的越界。
    if (op->index >= it->second.size()) {
      throw forch::exception("forch-ge: MultiOutputPort index out of range");
    }
    register_es_tensor(self, it->second[op->index]);
  }

  void visit(const op::Runtime * /*unused*/, const Tensor &self) override {
    // 带上张量名：裸 Runtime 混入建图 DAG 时能直接定位是哪个节点
    throw forch::exception("Forch Runtime Tensor ('" + self.get_name()
      + "') cannot be used in GE graph, use Input instead");
  }

  void register_es_tensor(
    const Tensor &tensor, EsCTensorHolder *es_tensor) const {
    graph->pool.emplace(tensor.get_id(), es_tensor);
  }

  // Build or retrieve the GE tensor for the given Forch tensor.
  EsCTensorHolder *build_es_tensor(const Tensor &tensor) {
    auto it = graph->pool.find(tensor.get_id());
    if (it != graph->pool.end()) {
      return it->second;
    }
    tensor.get_op()->accept(this, tensor);
    return graph->pool.at(tensor.get_id());
  }
};

} // namespace detail

GEBackend::GEGraph::Impl::Impl(
  GEBackend ge_backend, TensorList outputs, const std::string &name)
    : outputs(std::move(outputs)),
      graph_name(name.empty() ? std::string("ge-graph") : name),
      ge_backend(std::move(ge_backend)) {}

void GEBackend::GEGraph::Impl::compile() {
  ge_backend.impl()->compile_graph(this);
}

void GEBackend::GEGraph::Impl::load() {
  ge_backend.impl()->load_graph(this);
}

TensorList GEBackend::GEGraph::Impl::run(const TensorList &inputs_unorderd) {
  return ge_backend.impl()->run_graph(this, inputs_unorderd);
}

GEBackend::GEGraph GEBackend::Impl::build(
  std::shared_ptr<Backend::Impl> backend_impl, const forch::Graph &graph,
  const GEBuildOptions &ge_options) {
  auto ge_backend = GEBackend(
    std::static_pointer_cast<GEBackend::Impl>(std::move(backend_impl)));

  std::string graph_name = graph.name();
  if (graph_name.empty()) {
    graph_name = "ge-graph";
  }

  auto ge_graph = GEBackend::GEGraph(std::make_shared<GEBackend::GEGraph::Impl>(
    ge_backend, graph.outputs(), graph_name));
  auto *ge_graph_impl = ge_graph.impl();

  detail::GEGraphBuilder builder(ge_graph_impl);
  int64_t                output_index = 0;
  for (const auto &tensor : ge_graph_impl->outputs) {
    auto    *holder = builder.build_es_tensor(tensor);
    uint32_t rc     = EsSetGraphOutput(holder, output_index++);
    if (rc != 0U) {
      throw forch::exception(
        std::string("forch-ge: EsSetGraphOutput failed: rc=")
        + std::to_string(rc) + " ge_msg=" + forch_ge_get_ge_error_msg()
        + " shim=" + forch_ge_last_error());
    }
  }
  std::string host_input_indexes_str = ge_options.enable_host_input_indexes
    ? format_host_input_indexes(builder.auto_H2D_copy_input_indexes)
    : "";
  const char *const options[][2]     = {
    {"ge.exec.precision_mode_v2",
          precision_mode_str(ge_options.precision_mode)},
    {"ge.exec.hostInputIndexes", host_input_indexes_str.c_str()},
    {"ge.constLifecycle", const_lifecycle_str(ge_options.const_lifecycle)}};
  const uint32_t graph_id = graph_cnt++;
  detail::shim_check(
    forch_ge_session_add_graph(ge_session, graph_id, builder.builder, options,
      sizeof(options) / sizeof(options[0])),
    "forch_ge_session_add_graph");
  ge_graph_impl->ge_backend_graph_id = graph_id;
  ge_graph_impl->ge_state = GEBackend::GEGraph::Impl::GEBackendState::ADDED;

  return ge_graph;
}

} // namespace forch
