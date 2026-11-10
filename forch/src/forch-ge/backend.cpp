// The GEBackend facade, its session and runtime tensors, and the execution
// path. Graph construction lives in graph.cpp.

#include "forch/backend/forch-ge.hpp"
#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/ops.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/runtime.hpp"
#include "forch/utils.hpp"

#include "forch-ge-common.hpp"

#include "ge_session_shim.h"
#include "magic_enum.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace forch {

namespace {
// One GE backend per process. Copies share Backend::_impl.
bool backend_created = false;

struct RtTensorDeleter {
  void operator()(forch_ge_rt_tensor_t *t) const noexcept {
    forch_ge_rt_tensor_destroy(t);
  }
};
using RtTensorPtr = std::unique_ptr<forch_ge_rt_tensor_t, RtTensorDeleter>;

// Both views borrow storage: the caller must keep its buffer owner alive.
RtTensorPtr make_rt_tensor_from_const(const op::Const *op) {
  const auto              &shape = op->shape();
  const detail::CDtypeDesc desc(op->dtype());
  auto                    *handle =
    forch_ge_rt_tensor_from_ptr(op->buffer(), FORCH_GE_PLACEMENT_HOST,
      shape.data(), static_cast<int64_t>(shape.size()), desc.type, C_FORMAT_ND);
  if (handle == nullptr) {
    throw forch::exception(
      std::string("forch-ge: forch_ge_rt_tensor_from_ptr failed: ")
      + forch_ge_last_error());
  }
  return RtTensorPtr(handle);
}

RtTensorPtr make_rt_tensor_from_device_ptr(
  const void *dev_ptr, const Shape &shape, detail::CDtypeDesc desc) {
  auto *handle =
    forch_ge_rt_tensor_from_ptr(dev_ptr, FORCH_GE_PLACEMENT_DEVICE_HBM,
      shape.data(), static_cast<int64_t>(shape.size()), desc.type, C_FORMAT_ND);
  if (handle == nullptr) {
    throw forch::exception(
      std::string("forch-ge: forch_ge_rt_tensor_from_ptr failed: ")
      + forch_ge_last_error());
  }
  return RtTensorPtr(handle);
}

forch_ge_rt_tensor_info_t query(const forch_ge_rt_tensor_t *t) {
  forch_ge_rt_tensor_info_t info;
  detail::shim_check(
    forch_ge_rt_tensor_info(t, &info), "forch_ge_rt_tensor_info");
  return info;
}

void dump_dims(
  std::ostream &os, const int64_t *dims, int64_t rank, const char *label) {
  os << label << "=" << rank << "d(";
  for (int64_t i = 0; i < rank; ++i) {
    os << dims[i];
    if (i != rank - 1) {
      os << ",";
    }
  }
  os << ")";
}

} // namespace

struct op::GERuntime : op::Runtime {
  // Reverse destruction order: destroy the borrowed GE view before storage.
  std::shared_ptr<Device::Buffer> device_mem;
  RtTensorPtr                     rt;

  explicit GERuntime(RtTensorPtr t, std::shared_ptr<Device::Buffer> mem = {})
      : device_mem(std::move(mem)), rt(std::move(t)) {}

  void        dump(std::ostream &os) const override;
  const void *get_data() const override;
  Shape       shape(const TensorList       &/*unused*/) const override;
  DType       dtype(const TensorList       &/*unused*/) const override;
};

void op::GERuntime::dump(std::ostream &os) const {
  const auto info = query(rt.get());
  os << "op::GERuntime: ";
  dump_dims(os, info.origin_dims, info.origin_rank, "origin_shape");
  os << ", ";
  dump_dims(os, info.storage_dims, info.storage_rank, "storage_shape");
  os << ", origin_format=" << magic_enum::enum_name(info.origin_format);
  os << ", storage_format=" << magic_enum::enum_name(info.storage_format);
}

const void *op::GERuntime::get_data() const {
  return query(rt.get()).data;
}

Shape op::GERuntime::shape(const TensorList & /*unused*/) const {
  const auto info = query(rt.get());
  return Shape(info.storage_dims, info.storage_dims + info.storage_rank);
}

DType op::GERuntime::dtype(const TensorList & /*unused*/) const {
  return detail::c_to_forch_dtype(query(rt.get()).dtype);
}

GEBackend::Impl::Impl(AscendDevice device, const Options &options)
    : device(std::move(device)) {
  if (std::exchange(backend_created, true)) {
    throw forch::exception(
      "GEBackend: already initialized; construct one backend and share it");
  }
  bool initialized = false;
  try {
    utils::IntervalTimer timer("Initializing GE Backend");
    init_ge(options);
    initialized = true;
    ge_session  = forch_ge_session_create();
    if (ge_session == nullptr) {
      throw forch::exception(
        std::string("forch-ge: forch_ge_session_create failed: ")
        + forch_ge_last_error());
    }
  } catch (...) {
    if (initialized) {
      (void)forch_ge_finalize();
    } else {
      backend_created = false;
    }
    throw;
  }
}

GEBackend::Impl::~Impl() {
  utils::IntervalTimer timer("Finalizing GE Backend");
  forch_ge_session_destroy(ge_session);
  try {
    detail::shim_check(forch_ge_finalize(), "forch_ge_finalize");
  } catch (const std::exception &e) {
    logger()->error("forch_ge_finalize failed: {}", e.what());
  }
}

Backend::Graph GEBackend::Impl::build(
  std::shared_ptr<Backend::Impl> backend_impl, const forch::Graph &graph) {
  return build(std::move(backend_impl), graph, {});
}

Tensor GEBackend::Impl::move_to_device(const Tensor &tensor) const {
  const auto *op = dynamic_cast<const forch::op::Const *>(tensor.get_op());
  if (op == nullptr) {
    throw forch::exception(
      "move_to_device: only op::Const tensors are supported");
  }

  const auto              &shape    = op->shape();
  const detail::CDtypeDesc desc     = detail::CDtypeDesc(op->dtype());
  const void              *host_ptr = op->buffer();

  // Validate before allocating, including the signed shape product GE uses.
  int64_t shape_size = 1;
  for (auto d : shape) {
    if (d < 0) {
      throw forch::exception(
        "move_to_device: Const tensor has negative dimension");
    }
    if (d != 0 && shape_size > std::numeric_limits<int64_t>::max() / d) {
      throw forch::exception(
        "move_to_device: Const tensor element count overflows");
    }
    shape_size *= d;
  }
  if (static_cast<uint64_t>(shape_size)
    > std::numeric_limits<size_t>::max() / desc.size) {
    throw forch::exception("move_to_device: Const tensor byte count overflows");
  }
  const size_t data_size = static_cast<size_t>(shape_size) * desc.size;
  auto         device_mem =
    std::make_shared<Device::Buffer>(device.allocate(data_size));
  device.host_to_device(device_mem->data(), host_ptr, data_size);
  auto rt = make_rt_tensor_from_device_ptr(device_mem->data(), shape, desc);

  Tensor out = {{},
    std::make_unique<forch::op::GERuntime>(
      std::move(rt), std::move(device_mem))};
  out.set_name(tensor.get_name());
  return out;
}

GEBackend::GEBackend(AscendDevice device)
    : GEBackend(std::move(device), Options::from_env()) {}

GEBackend::GEBackend(AscendDevice device, const Options &options)
    : Backend(std::make_shared<Impl>(std::move(device), options)) {}

GEBackend::GEBackend(std::shared_ptr<Impl> impl) : Backend(std::move(impl)) {}

GEBackend::Impl *GEBackend::impl() {
  return static_cast<Impl *>(_impl.get());
}
const GEBackend::Impl *GEBackend::impl() const {
  return static_cast<const Impl *>(_impl.get());
}

GEBackend::GEGraph GEBackend::build(
  const forch::Graph &graph, const GEBuildOptions &ge_options) {
  return impl()->build(_impl, graph, ge_options);
}

Tensor GEBackend::move_to_device(const Tensor &tensor) const {
  return impl()->move_to_device(tensor);
}

void GEBackend::Impl::compile_graph(GEBackend::GEGraph::Impl *graph) {
  if (graph->ge_state != GEBackend::GEGraph::Impl::GEBackendState::ADDED) {
    throw forch::exception(
      "Graph must be added to a backend before compilation");
  }
  detail::shim_check(
    forch_ge_session_compile(ge_session, graph->ge_backend_graph_id),
    "forch_ge_session_compile");
  graph->ge_state = GEBackend::GEGraph::Impl::GEBackendState::COMPILED;
}

void GEBackend::Impl::load_graph(GEBackend::GEGraph::Impl *graph) {
  if (graph->ge_state != GEBackend::GEGraph::Impl::GEBackendState::COMPILED) {
    throw forch::exception("Graph must be compiled before loading");
  }
  detail::shim_check(
    forch_ge_session_load(ge_session, graph->ge_backend_graph_id),
    "forch_ge_session_load");
  graph->ge_state = GEBackend::GEGraph::Impl::GEBackendState::LOADED;
}

void GEBackend::Impl::unload_graph(GEBackend::GEGraph::Impl *graph) {
  if (graph->ge_state != GEBackend::GEGraph::Impl::GEBackendState::LOADED) {
    throw forch::exception("Graph must be loaded before unloading");
  }
  detail::shim_check(
    forch_ge_session_remove(ge_session, graph->ge_backend_graph_id),
    "forch_ge_session_remove");
  graph->ge_state = GEBackend::GEGraph::Impl::GEBackendState::COMPILED;
}

namespace {

/**
 * 从 Forch Tensor 构建用于 GE 图输入的 runtime tensor 句柄。
 * 仅支持 op::Const 和 op::GERuntime 两种来源。
 * 对 op::Const，构造的 RtTensorPtr 由 builder 临时持有。
 */
struct GERunInputBuilder : public op::Visitor {
  std::vector<forch_ge_rt_tensor_t *> result;
  std::vector<RtTensorPtr>            temp_tensors;

  std::vector<forch_ge_rt_tensor_t *> build(const TensorList &inputs) {
    for (const auto &input : inputs) {
      input.get_op()->accept(this, input);
    }
    return std::exchange(result, {});
  }

  void visit(const op::Const *op, const Tensor & /*unused*/) override {
    temp_tensors.emplace_back(make_rt_tensor_from_const(op));
    result.push_back(temp_tensors.back().get());
  }

  void visit(const op::Runtime *op, const Tensor & /*unused*/) override {
    const auto *op_gert = dynamic_cast<const op::GERuntime *>(op);
    if (op_gert == nullptr) {
      throw forch::exception(
        "GE Graph only supports op::GERuntime as input data source");
    }
    result.push_back(op_gert->rt.get());
  }

  void visit(const op::Input * /*unused*/, const Tensor & /*unused*/) override {
    throw forch::exception(
      "forch Input Placeholder Tensor is not valid as runtime input data");
  }
  void visit(
    const op::Function * /*unused*/, const Tensor & /*unused*/) override {
    throw forch::exception(
      "forch Function Tensor is not valid as runtime input data");
  }
};

} // namespace

TensorList GEBackend::Impl::run_graph(
  GEBackend::GEGraph::Impl *graph, const TensorList &inputs_unorderd) {
  if (graph->ge_state != GEBackend::GEGraph::Impl::GEBackendState::LOADED) {
    throw forch::exception("Graph must be loaded before running");
  }

  TensorList inputs_ordered;
  for (const auto &param : graph->input_id_ge_to_forch) {
    auto it = std::ranges::find_if(inputs_unorderd,
      [param](const Tensor &t) { return t.get_name() == param.get_name(); });
    if (it == inputs_unorderd.end()) {
      throw forch::exception(
        std::string("GEGraph::run missing input tensor: ") + param.get_name());
    }
    if (dynamic_cast<const op::Input *>(param.get_op())
          ->forch_ge_auto_H2D_copy_enabled()
      && dynamic_cast<const op::GERuntime *>(it->get_op()) != nullptr) {
      throw forch::exception(
        std::string(
          "GEGraph::run auto H2D copy enabled, expected host input, but got: ")
        + param.get_name());
    }
    inputs_ordered.push_back(*it);
  }

  GERunInputBuilder ge_inputs_builder;
  auto              ge_inputs = ge_inputs_builder.build(inputs_ordered);

  forch_ge_rt_tensor_t **out_arr = nullptr;
  size_t                 n_out   = 0;
  detail::shim_check(
    forch_ge_session_run(ge_session, graph->ge_backend_graph_id,
      ge_inputs.data(), ge_inputs.size(), &out_arr, &n_out),
    "forch_ge_session_run");

  TensorList outputs;
  outputs.reserve(n_out);
  for (size_t i = 0; i < n_out; ++i) {
    RtTensorPtr rt(out_arr[i]); // 接管所有权
    outputs.push_back(
      Tensor(TensorList{}, std::make_unique<op::GERuntime>(std::move(rt)))
        .set_name(graph->outputs[i].get_name()));
  }
  // out_arr 数组本身仍然要释放（其元素已被 RtTensorPtr 接管，传入 nullptr-len
  // 为 0 的语义即可由 free 自然处理）：
  std::free(static_cast<void *>(out_arr));

  return outputs;
}

} // namespace forch
