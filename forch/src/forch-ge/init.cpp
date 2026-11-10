// Process-wide bring-up of the GE backend: turning
// GEBackend::Options into the GEInitializeV2 option map. This is the only
// place that knows the `ge.exec.*` spelling of any of it.

#include "forch/backend/forch-ge.hpp"
#include "forch/distributed.hpp"
#include "forch/exception.hpp"

#include "../utils.hpp"
#include "forch-ge-common.hpp"

#include "ge_session_shim.h"

#include <array>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace forch {

namespace {

const char *env_or_null(const char *name) {
  const char *value = std::getenv(name);
  return (value != nullptr && value[0] != '\0') ? value : nullptr;
}

} // namespace

GEBackend::Options GEBackend::Options::from_env() {
  Options options;
  options.group = process_group_from_env();
  if (const char *table = env_or_null("FORCH_GE_RANK_TABLE_FILE")) {
    options.rank_table_file = table;
  }
  options.hccl_timeout = std::chrono::seconds(
    detail::getenv_int("FORCH_GE_HCCL_TIMEOUT", options.hccl_timeout.count()));
  return options;
}

void GEBackend::Impl::init_ge(const Options &options) {
  if (options.hccl_timeout.count() <= 0) {
    throw forch::exception("GEBackend: hccl_timeout must be positive");
  }
  if (options.group.world_size() > 1 && options.rank_table_file.empty()) {
    throw forch::exception("GEBackend: world_size is "
      + std::to_string(options.group.world_size())
      + " but no rank_table_file was given; HCOM learns the topology only "
        "from a rank table passed to GE's global init (set "
        "FORCH_GE_RANK_TABLE_FILE or Options::rank_table_file)");
  }

  // A map, so extra_ge_options can overwrite a derived key rather than
  // sitting next to it and losing to whichever the shim inserts first.
  std::map<std::string, std::string> ge_options;
  if (options.group.world_size() > 1) {
    ge_options["ge.exec.rankId"] = std::to_string(options.group.rank_id());
    ge_options["ge.exec.rankTableFile"] = options.rank_table_file;
    // Single-server deployment; podName is required alongside it but its value
    // is not interpreted for a one-node rank table.
    ge_options["ge.exec.deployMode"] = "0";
    ge_options["ge.exec.podName"]    = "localhost";
    ge_options["ge.exec.hcclExecuteTimeOut"] =
      std::to_string(options.hccl_timeout.count());
  }
  for (const auto &[key, value] : options.extra_ge_options) {
    ge_options[key] = value;
  }

  std::vector<std::array<const char *, 2>> raw;
  raw.reserve(ge_options.size());
  for (const auto &[key, value] : ge_options) {
    raw.push_back({key.c_str(), value.c_str()});
  }
  // The shim consumes C array rows; std::array preserves that layout.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  using RawRow = const char *const(*)[2];
  detail::shim_check(
    forch_ge_init(reinterpret_cast<RawRow>(raw.data()), raw.size()),
    "forch_ge_init");
}

} // namespace forch
