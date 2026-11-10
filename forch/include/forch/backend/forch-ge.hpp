#pragma once

#include "forch/device/ascend.hpp"
#include "forch/distributed.hpp"
#include "forch/graph.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace forch {

struct GEBackend : public Backend {
  using Backend::build;

  struct GEGraph : public Backend::Graph {
    struct Impl;

    explicit GEGraph(std::shared_ptr<Impl> impl);
    Impl       *impl();
    const Impl *impl() const;

  protected:
    friend struct GEBackend;
  };

  struct GEBuildOptions {
    enum struct PrecisionMode {
      Origin,
      AutoToFP16,
      AutoToBF16,
      // force all ops that support FP16 to use FP16, use with caution
      ForceFP16,
    } precision_mode = PrecisionMode::Origin;

    enum struct ConstLifecycle {
      // 常量节点按 session 级别存储，session 内多图间同内容常量可复用设备内存
      // （跨图权重共享）
      Session,
      // 常量节点按 graph 级别存储（在线推理默认行为）
      Graph,
    } const_lifecycle = ConstLifecycle::Session;

    // Hybrid GE submodels may renumber inputs without rebasing this global
    // option. Runtime tensors still describe host/device placement when off.
    bool enable_host_input_indexes = true;
  };

  /**
   * How this process configures GE and, when it runs collectives, its
   * peers. Pass it to the constructor; the GE globals behind it exist per
   * process, so it takes effect once, when the first GEBackend is built.
   */
  struct Options {
    /// This process's place in the communicator. world_size == 1 skips HCOM
    /// setup entirely.
    ProcessGroup group;

    /**
     * Required once world_size > 1: HCOM only learns the topology from a rank
     * table passed to GE's global init. Its `device_id` entries are physical
     * card ids. Inline rank tables are not an option -- CANN 9.0.0 has no
     * such init key, and passing one is rejected at graph compile time.
     */
    std::string rank_table_file;

    std::chrono::seconds hccl_timeout{120};

    /// Escape hatch, forwarded verbatim to GEInitializeV2. Use it for keys
    /// this struct does not model; these win over the derived ones.
    std::vector<std::pair<std::string, std::string>> extra_ge_options;

    /**
     * The launcher's half of the contract (see script/forch-ge-run.py):
     * FORCH_RANK_ID, FORCH_WORLD_SIZE, FORCH_GROUP_ID,
     * FORCH_GE_RANK_TABLE_FILE and FORCH_GE_HCCL_TIMEOUT. Unset variables keep
     * the defaults above, so a single-process run needs no environment.
     */
    static Options from_env();
  };

  struct Impl;

  /// Configures this process from the environment, i.e. `Options::from_env()`.
  explicit GEBackend(AscendDevice device);

  /// Configures this process explicitly. Construct exactly one GEBackend: a
  /// second construction throws, even with identical options or after teardown.
  /// Copying a backend handle shares its existing implementation and is safe.
  /// Initialization and teardown must be called from a single thread.
  GEBackend(AscendDevice device, const Options &options);

  explicit GEBackend(std::shared_ptr<Impl> impl);

  GEBackend::GEGraph build(
    const forch::Graph &graph, const GEBuildOptions &ge_options);

  Tensor move_to_device(const Tensor &tensor) const override;

  Impl       *impl();
  const Impl *impl() const;
};

namespace op {

struct GERuntime;

}

} // namespace forch
