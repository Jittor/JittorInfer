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
 * \file jittor_infer_flash_attention_tiling_context.h
 * \brief
 */
#ifndef JITTOR_INFER_FLASH_ATTENTION_TILING_CONTEXT_H
#define JITTOR_INFER_FLASH_ATTENTION_TILING_CONTEXT_H
#include <cstdint>
#include <queue>
#include <vector>

#include "../utils/inc/error/ops_error.h"
#include "../utils/inc/tiling/data_copy_transpose_tiling.h"
#include "../utils/inc/tiling/data_copy_transpose_tiling_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

/*
contextParams is a new structured defined for the use of
FusedInferAttentionScore op. It is meant to catch and organize all the necessary
variables passed by FIAS tilling function. It will be used as the input to the
new 'runBigKernelWithParams' function in PFA tilling. The old PFA tillingContext
will also be transformed to this structure in the future.
*/
struct ContextParamsForPFATiling {
    const gert::Tensor *pseShift;
    const gert::Tensor *attentionMask;
    const gert::Tensor *actualSeqenceLengthQ;
    const gert::Tensor *actualSeqenceLengthKV;
    const gert::Tensor *blockTable;

    ge::DataType inputDataType;
    ge::DataType kDataType;
    ge::DataType vDataType;
    ge::DataType pseShiftDataType;
    ge::DataType maskDataType;
    ge::DataType blockTableType;
    ge::DataType outputDataType;
    const char *opName;
    const gert::StorageShape *queryInputShape;
    const gert::StorageShape *keyInputShape;
    const gert::StorageShape *valueInputShape;
    const gert::StorageShape *pseShiftShape;
    const gert::StorageShape *attentionMaskShape;
    const gert::StorageShape *deqScale1Shape;
    const gert::StorageShape *scale1Shape;
    const gert::StorageShape *deqScale2Shape;
    const gert::StorageShape *scale2Shape;
    const gert::StorageShape *offset2Shape;
    const gert::StorageShape *blockTableShape;
    const gert::StorageShape *outputShape;
    const gert::StorageShape *lseoutputShape;

    const int64_t *innerPrecisePtr;
    const int32_t *headsNumber;
    const int32_t *sparseMode;
    const int64_t *preToken;
    const int64_t *nextToken;
    const float *scaleValue;
    const int32_t *blockSize;
    const char *layout;
    const int32_t *numKeyValueHeads;
    size_t *workspaceSize;
    ge::DataType deqScaleType;
    ge::DataType deqScale2Type;
    ge::DataType quantScale2Type;
    ge::DataType quantOffset2Type;
    const bool *softmaxLseFlag;
    bool isSoftMaxLseEnable;

    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    size_t defaultSysWorkspaceSize;
};

}  // namespace optiling

#endif  // JITTOR_INFER_FLASH_ATTENTION_TILING_CONTEXT_H