#include "forch/distributed.hpp"

#include "utils.hpp"

namespace forch {

ProcessGroup process_group_from_env() {
  return ProcessGroup(detail::getenv_int<int64_t>("FORCH_RANK_ID", 0),
    detail::getenv_int<int64_t>("FORCH_WORLD_SIZE", 1),
    detail::getenv_int<int64_t>("FORCH_GROUP_ID", 0));
}

} // namespace forch
