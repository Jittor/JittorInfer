/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file jittor_infer_flash_attention_tiling_const.h
 * \brief
 */
#ifndef JITTOR_INFER_FLASH_ATTENTION_TILING_CONST_H
#define JITTOR_INFER_FLASH_ATTENTION_TILING_CONST_H
#include <cstdint>

namespace optiling {
constexpr uint32_t INT8SIZE = 1;
constexpr uint32_t UINT8SIZE = 1;
constexpr uint32_t FLOAT16SIZE = 2;
constexpr uint32_t BFLOAT16SIZE = 2;
constexpr uint32_t FLOAT32SIZE = 4;
constexpr uint32_t BOOLSIZE = 1;

constexpr uint32_t FIRST_DIM = 0;
constexpr uint32_t SECOND_DIM = 1;
constexpr uint32_t THIRD_DIM = 2;
constexpr size_t DIM_NUM_3 = 3;
constexpr uint32_t N_SIZE_8 = 8;
constexpr uint32_t N_SIZE_16 = 16;
constexpr uint32_t N_SIZE_32 = 32;
constexpr uint32_t N_SIZE_64 = 64;
constexpr uint32_t N_SIZE_128 = 128;
constexpr uint32_t D_SIZE_192 = 192;
constexpr uint32_t D_SIZE_128 = 128;
constexpr int HIGH_PRECISION = 0;
constexpr int HIGH_PERFORMANCE = 1;
constexpr uint32_t MSD_HIGH_PERFORMANCE_EXPEND_NUM = 2;
constexpr uint32_t MSD_HIGH_PRECISION_EXPEND_NUM = 3;

constexpr uint32_t MAX_BATCH = 256U;
constexpr int64_t MAX_VAR_LEN_SEQ_LEN = 4096L;
constexpr int64_t BALANCE_LOAD_LIST_SIZE = 8L;
constexpr int64_t COF[BALANCE_LOAD_LIST_SIZE] = {256, 384, 512, 640, 768, 896, 960, 1024};
} // namespace optiling

#endif // JITTOR_INFER_FLASH_ATTENTION_TILING_CONST_H