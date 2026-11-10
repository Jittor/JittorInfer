#pragma once

/**
 * Private implementation header shared across forch-ge. Common utilities are
 * implemented in common.cpp, Graph methods in graph.cpp, and Backend methods
 * and runtime tensors in backend.cpp. Process setup lives in init.cpp;
 * dispatch and HCCL attributes have their own headers. Not installed.
 */

#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"

#include "../graph.hpp"

#include "esb_funcs.h"
#include "ge_session_shim.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace forch::detail {

void shim_check(int rc, const char *what);

struct CDtypeDesc {
  C_DataType type;
  size_t     size;

  CDtypeDesc() = default;
  CDtypeDesc(C_DataType type, size_t size);
  CDtypeDesc(DType dtype); // NOLINT(google-explicit-constructor)
};

DType c_to_forch_dtype(C_DataType dtype);

} // namespace forch::detail

namespace forch {

namespace detail {
struct GEGraphBuilder;

} // namespace detail

struct GEBackend::GEGraph::Impl : Backend::Graph::Impl {
  friend struct detail::GEGraphBuilder;
  friend struct GEBackend::Impl;

  Impl(GEBackend ge_backend, TensorList outputs,
    const std::string &name = "ge-graph");
  Impl(Impl &&)                 = default;
  Impl &operator=(Impl &&)      = delete;
  Impl(const Impl &)            = delete;
  Impl &operator=(const Impl &) = delete;
  ~Impl() override;

  void       compile() override;
  void       load() override;
  TensorList run(const TensorList &inputs) override;

private:
  std::map<uint64_t, EsCTensorHolder *> pool;
  TensorList                            input_id_ge_to_forch;
  TensorList                            outputs;
  std::string                           graph_name;

  enum struct GEBackendState : uint8_t {
    NONE,
    ADDED,
    COMPILED,
    LOADED,
  };
  GEBackendState ge_state = GEBackendState::NONE;
  GEBackend      ge_backend;
  uint32_t       ge_backend_graph_id = 0;
};

struct GEBackend::Impl : Backend::Impl {
public:
  friend struct GEBackend;
  friend struct GEBackend::GEGraph::Impl;

  Impl(AscendDevice device, const Options &options);
  ~Impl() override;

  Backend::Graph build(std::shared_ptr<Backend::Impl> backend_impl,
    const forch::Graph                               &graph) override;

  GEBackend::GEGraph build(std::shared_ptr<Backend::Impl> backend_impl,
    const forch::Graph &graph, const GEBuildOptions &ge_options);

  Tensor move_to_device(const Tensor &tensor) const override;

private:
  AscendDevice        device;
  forch_ge_session_t *ge_session = nullptr;
  uint32_t            graph_cnt  = 0;

  /// GEInitializeV2 options. Device activation is handled by the caller. Runs
  /// once per process, for the first GEBackend built. Defined in init.cpp.
  static void init_ge(const Options &options);

  void       compile_graph(GEBackend::GEGraph::Impl *graph);
  void       load_graph(GEBackend::GEGraph::Impl *graph);
  void       unload_graph(GEBackend::GEGraph::Impl *graph);
  TensorList run_graph(
    GEBackend::GEGraph::Impl *graph, const TensorList &inputs_unorderd);
};

} // namespace forch
