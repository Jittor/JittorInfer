/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file jittor_infer_flash_attention_tiling_struct.h
 * \brief
 */
#ifndef JITTOR_INFER_FLASH_ATTENTION_TILING_STRUCT_H
#define JITTOR_INFER_FLASH_ATTENTION_TILING_STRUCT_H
#include <cstdint>
namespace optiling {

enum class InputLayout {
    SH,
    BSH,
    BNSD,
    NSD,
    BSND,
    NONE,
};

enum class TilingMod {
    CVSAME = 0,
    CVDIFF,
};

enum class SplitCoreMode {
    SPLIT_NBS_VECTOR = 0,
    SPLIT_NBS_CUBE,
    SPLIT_ONEN_VECTOR,
    SPLIT_ONEN_CUBE,
    BALANCE_VECTOR,
    BALANCE_CUBE,
};
}  // namespace optiling

#endif  // JITTOR_INFER_FLASH_ATTENTION_TILING_STRUCT_H