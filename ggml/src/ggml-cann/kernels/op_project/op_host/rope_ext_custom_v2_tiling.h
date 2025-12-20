
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RopeExtCustomV2TilingData)
//   TILING_DATA_FIELD_DEF(uint32_t, size);
TILING_DATA_FIELD_DEF(int32_t, ne0);
TILING_DATA_FIELD_DEF(int32_t, ne1);
TILING_DATA_FIELD_DEF(int32_t, pos_len);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RopeExtCustomV2, RopeExtCustomV2TilingData)
}  // namespace optiling
