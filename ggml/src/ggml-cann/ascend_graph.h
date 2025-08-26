/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DAVINCI_GRAPH_UTILS_H
#define DAVINCI_GRAPH_UTILS_H
#include <map>
#include <string>
#include <vector>

#include "all_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

//  #include "ggml-common.h"
//  #include "ggml-cann.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
//  #include "ggml-cann/aclnn_ops.h"
#include "ggml-cann/ascend_graph_ops.h"
// #include "ggml-cann/common.h"
#include <acl/acl_rt.h>

#include "common.h"

using namespace ge;
using std::string;
using std::vector;

ge::Graph build_ascend_graph(ggml_cgraph* cgraph,
                             ggml_backend_cann_context& cann_ctx,
                             std::vector<gert::Tensor>& input_init,
                             std::vector<gert::Tensor>& output_init);

Status reuse_ascend_graph(uint32_t graph_id, ge::Session* session,
                          ggml_cgraph* cgraph, const aclrtStream& stream,
                          std::vector<gert::Tensor>& input_init,
                          std::vector<gert::Tensor>& output_init);
#endif  // DAVINCI_GRAPH_UTILS_H
