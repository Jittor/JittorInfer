#pragma once

#include "forch/exception.hpp"

#include <cstdint>
#include <string>

namespace forch {

/**
 * Participants in a collective. Construction establishes rank membership;
 * read-only accessors preserve it. group_id identifies the communicator, with
 * 0 reserved for the default world group. This value does not create a group.
 * world_size is known at graph construction for collective shape inference.
 */
class ProcessGroup {
public:
  explicit ProcessGroup(
    int64_t rank_id = 0, int64_t world_size = 1, int64_t group_id = 0)
      : rank_id_(rank_id), world_size_(world_size), group_id_(group_id) {
    if (world_size < 1) {
      throw exception("ProcessGroup: world_size must be >= 1, got "
        + std::to_string(world_size));
    }
    if (rank_id < 0 || rank_id >= world_size) {
      throw exception("ProcessGroup: rank_id " + std::to_string(rank_id)
        + " is outside [0, " + std::to_string(world_size) + ")");
    }
    if (group_id < 0) {
      throw exception("ProcessGroup: group_id must be >= 0");
    }
  }

  int64_t rank_id() const {
    return rank_id_;
  }
  int64_t world_size() const {
    return world_size_;
  }
  int64_t group_id() const {
    return group_id_;
  }

  bool operator==(const ProcessGroup &) const = default;

private:
  int64_t rank_id_;
  int64_t world_size_;
  int64_t group_id_;
};

/**
 * Reads FORCH_RANK_ID, FORCH_WORLD_SIZE and FORCH_GROUP_ID, the generic half of
 * the contract that `script/forch-ge-run.py` exports. Returns the
 * single-process group when neither is set, so a program that runs no
 * collectives needs no environment at all.
 */
ProcessGroup process_group_from_env();

} // namespace forch
