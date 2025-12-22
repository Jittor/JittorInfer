
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RopeExtCustomTilingData)
// TILING_DATA_FIELD_DEF(uint32_t, size);
TILING_DATA_FIELD_DEF(int32_t, ne0);
TILING_DATA_FIELD_DEF(int32_t, ne1);
TILING_DATA_FIELD_DEF(int32_t, s1);
TILING_DATA_FIELD_DEF(int32_t, s2);
TILING_DATA_FIELD_DEF(int32_t, n_dims);
TILING_DATA_FIELD_DEF(float, freq_scale);
TILING_DATA_FIELD_DEF(float, theta_scale);
TILING_DATA_FIELD_DEF(float, ext_factor);
TILING_DATA_FIELD_DEF(float, attn_factor);
TILING_DATA_FIELD_DEF(float, corr_dims_v_0);
TILING_DATA_FIELD_DEF(float, corr_dims_v_1);
TILING_DATA_FIELD_DEF(float, logf_1_freq_scale);
TILING_DATA_FIELD_DEF(int32_t, pos_len);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RopeExtCustom, RopeExtCustomTilingData)
}  // namespace optiling
