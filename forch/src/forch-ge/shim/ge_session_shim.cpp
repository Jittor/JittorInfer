/**
 * forch-ge: GE Session shim implementation.
 *
 * 这是 forch-ge 内部唯一保留 _GLIBCXX_USE_CXX11_ABI=0 + google=ascend_private
 * 的 TU。 所有需要 ge::GeSession / gert::Tensor / std::map<AscendString,...>
 * 的代码都集中在此。
 */
#include "ge_session_shim.h"

#include "ge/compliant_node_builder.h"
#include "ge/es_c_graph_builder.h"
#include "ge/esb_funcs.h"
#include "ge/ge_api_v2.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/tensor_data.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error; // NOLINT

void set_error(const char *msg) {
  g_last_error.assign((msg != nullptr) ? msg : "");
}

void set_ge_error(ge::Status status, const char *prefix) {
  std::string msg = (prefix != nullptr) ? prefix : "GE call failed";
  msg += ": status=";
  msg += std::to_string(status);
  msg += "\n== GE Error Message ==\n";
  msg += ge::GEGetErrorMsgV3().GetString();
  g_last_error = std::move(msg);
}

enum class InitState { Uninitialized, Active, Finalized };
InitState g_init_state = InitState::Uninitialized;

} // namespace

extern "C" {

struct forch_ge_session {
  ge::GeSession *session;
};

struct forch_ge_rt_tensor {
  gert::Tensor tensor;
};

const char *forch_ge_last_error(void) {
  return g_last_error.c_str();
}

const char *forch_ge_get_ge_error_msg(void) {
  static thread_local std::string last_msg;
  last_msg.assign(ge::GEGetErrorMsgV3().GetString());
  return last_msg.c_str();
}

int forch_ge_init(const char *const (*options)[2], size_t n_options) {
  if (g_init_state != InitState::Uninitialized) {
    set_error("forch_ge_init: GE was already initialized in this process");
    return -1;
  }
  if (n_options > 0 && options == nullptr) {
    set_error("forch_ge_init: null options with non-zero count");
    return -1;
  }
  std::map<ge::AscendString, ge::AscendString> ge_options;
  ge_options.emplace("ge.graphRunMode", "0");
  for (size_t i = 0; i < n_options; ++i) {
    const char *key = options[i][0];
    const char *val = options[i][1];
    if (key == nullptr || val == nullptr) {
      set_error("forch_ge_init: null key or value in options");
      return -1;
    }
    ge_options.insert_or_assign(ge::AscendString(key), ge::AscendString(val));
  }

  auto status = ge::GEInitializeV2(ge_options);
  if (status != ge::SUCCESS) {
    set_ge_error(status, "GEInitializeV2 failed");
    return -1;
  }
  g_init_state = InitState::Active;
  return 0;
}

int forch_ge_finalize(void) {
  if (g_init_state == InitState::Active) {
    auto status  = ge::GEFinalizeV2();
    g_init_state = InitState::Finalized;
    if (status != ge::SUCCESS) {
      set_ge_error(status, "GEFinalizeV2 failed");
      return -1;
    }
  }
  return 0;
}

forch_ge_session_t *forch_ge_session_create(void) {
  auto *handle = new (std::nothrow) forch_ge_session_t();
  if (handle == nullptr) {
    set_error("forch_ge_session_create: out of memory");
    return nullptr;
  }
  std::map<ge::AscendString, ge::AscendString> options;
  handle->session = new (std::nothrow) ge::GeSession(options);
  if (handle->session == nullptr) {
    delete handle;
    set_error("forch_ge_session_create: failed to construct GeSession");
    return nullptr;
  }
  return handle;
}

void forch_ge_session_destroy(forch_ge_session_t *s) {
  if (s == nullptr) {
    return;
  }
  delete s->session;
  delete s;
}

int forch_ge_session_add_graph(forch_ge_session_t *s, uint32_t graph_id,
  EsCGraphBuilder *builder, const char *const (*options)[2], size_t n_options) {
  if (s == nullptr || s->session == nullptr || builder == nullptr) {
    set_error("forch_ge_session_add_graph: null argument");
    return -1;
  }
  if (n_options > 0 && options == nullptr) {
    set_error("forch_ge_session_add_graph: null options with non-zero count");
    return -1;
  }

  // EsBuildGraphAndReset 返回的 EsCGraph* 与 ge::Graph* 是同一对象的两种视图，
  // 参见 es_graph_builder.h 中 GeGraphsToEsCGraphs 的实现。我们只需在 AddGraph
  // 后销毁这个 ge::Graph（GeSession::AddGraph 内部会做注册所需的拷贝）。
  EsCGraph *esc_graph = EsBuildGraphAndReset(builder);
  if (esc_graph == nullptr) {
    set_error("forch_ge_session_add_graph: EsBuildGraphAndReset returned null");
    return -1;
  }
  std::unique_ptr<ge::Graph> ge_graph(reinterpret_cast<ge::Graph *>(esc_graph));

  std::map<ge::AscendString, ge::AscendString> ge_options;
  for (size_t i = 0; i < n_options; ++i) {
    const char *key = options[i][0];
    const char *val = options[i][1];
    if (key == nullptr || val == nullptr) {
      set_error("forch_ge_session_add_graph: null key or value in options");
      return -1;
    }
    ge_options.emplace(key, val);
  }

  auto status = s->session->AddGraph(graph_id, *ge_graph, ge_options);
  if (status != ge::SUCCESS) {
    set_ge_error(status, "GeSession::AddGraph failed");
    return -1;
  }
  return 0;
}

int forch_ge_session_compile(forch_ge_session_t *s, uint32_t graph_id) {
  if (s == nullptr || s->session == nullptr) {
    set_error("forch_ge_session_compile: null session");
    return -1;
  }
  auto status = s->session->CompileGraph(graph_id);
  if (status != ge::SUCCESS) {
    set_ge_error(status, "GeSession::CompileGraph failed");
    return -1;
  }
  return 0;
}

int forch_ge_session_load(forch_ge_session_t *s, uint32_t graph_id) {
  if (s == nullptr || s->session == nullptr) {
    set_error("forch_ge_session_load: null session");
    return -1;
  }
  const std::map<ge::AscendString, ge::AscendString> options;
  auto status = s->session->LoadGraph(graph_id, options, nullptr);
  if (status != ge::SUCCESS) {
    set_ge_error(status, "GeSession::LoadGraph failed");
    return -1;
  }
  return 0;
}

int forch_ge_session_remove(forch_ge_session_t *s, uint32_t graph_id) {
  if (s == nullptr || s->session == nullptr) {
    set_error("forch_ge_session_remove: null session");
    return -1;
  }
  auto status = s->session->RemoveGraph(graph_id);
  if (status != ge::SUCCESS) {
    set_ge_error(status, "GeSession::RemoveGraph failed");
    return -1;
  }
  return 0;
}

namespace {

// 借用一份 gert::Tensor 用于送入 RunGraph：复用 addr，禁用 manager
// 避免重复释放。
gert::Tensor make_view(const gert::Tensor &src) {
  gert::StorageShape ss;
  ss.MutableOriginShape()  = src.GetOriginShape();
  ss.MutableStorageShape() = src.GetStorageShape();

  gert::Tensor     view(ss, src.GetFormat(), src.GetDataType());
  const auto      &td = src.GetTensorData();
  gert::TensorData td_view(
    td.GetAddr(), nullptr, td.GetSize(), td.GetPlacement());
  view.SetData(std::move(td_view));
  return view;
}

} // namespace

int forch_ge_session_run(forch_ge_session_t *s, uint32_t graph_id,
  forch_ge_rt_tensor_t *const *inputs, size_t n_in,
  forch_ge_rt_tensor_t ***outputs_out, size_t *n_out) {
  if (s == nullptr || s->session == nullptr || outputs_out == nullptr
    || n_out == nullptr) {
    set_error("forch_ge_session_run: null argument");
    return -1;
  }
  *outputs_out = nullptr;
  *n_out       = 0;

  std::vector<gert::Tensor> ge_inputs;
  ge_inputs.reserve(n_in);
  for (size_t i = 0; i < n_in; ++i) {
    if (inputs[i] == nullptr) {
      set_error("forch_ge_session_run: null input tensor");
      return -1;
    }
    ge_inputs.emplace_back(make_view(inputs[i]->tensor));
  }

  std::vector<gert::Tensor> ge_outputs;
  auto status = s->session->RunGraph(graph_id, ge_inputs, ge_outputs);
  if (status != ge::SUCCESS) {
    set_ge_error(status, "GeSession::RunGraph failed");
    return -1;
  }

  auto **arr = static_cast<forch_ge_rt_tensor_t **>(
    std::malloc(sizeof(forch_ge_rt_tensor_t *) * ge_outputs.size()));
  if (arr == nullptr && !ge_outputs.empty()) {
    set_error("forch_ge_session_run: out of memory");
    return -1;
  }
  for (size_t i = 0; i < ge_outputs.size(); ++i) {
    arr[i] = new (std::nothrow) forch_ge_rt_tensor_t();
    if (arr[i] == nullptr) {
      for (size_t j = 0; j < i; ++j) {
        delete arr[j];
      }
      std::free(arr);
      set_error("forch_ge_session_run: out of memory");
      return -1;
    }
    arr[i]->tensor = std::move(ge_outputs[i]);
  }

  *outputs_out = arr;
  *n_out       = ge_outputs.size();
  return 0;
}

forch_ge_rt_tensor_t *forch_ge_rt_tensor_from_ptr(const void *data_ptr,
  forch_ge_placement_t placement, const int64_t *dims, int64_t dim_num,
  C_DataType dtype, C_Format format) {
  auto *handle = new (std::nothrow) forch_ge_rt_tensor_t();
  if (handle == nullptr) {
    set_error("forch_ge_rt_tensor_from_ptr: out of memory");
    return nullptr;
  }

  gert::Shape shape;
  shape.SetDimNum(dim_num);
  for (int64_t i = 0; i < dim_num; ++i) {
    shape.SetDim(i, dims[i]);
  }
  gert::StorageShape storage_shape;
  storage_shape.MutableOriginShape()  = shape;
  storage_shape.MutableStorageShape() = shape;
  const gert::StorageFormat storage_format(
    static_cast<ge::Format>(format), static_cast<ge::Format>(format), {});

  gert::Tensor t(
    storage_shape, storage_format, static_cast<ge::DataType>(dtype));

  const size_t elem_size =
    static_cast<size_t>(ge::GetSizeInBytes(1, t.GetDataType()));
  const size_t data_size = static_cast<size_t>(t.GetShapeSize()) * elem_size;

  if (data_ptr == nullptr && data_size > 0) {
    delete handle;
    set_error("forch_ge_rt_tensor_from_ptr: null data_ptr with non-zero size");
    return nullptr;
  }

  gert::TensorData td(const_cast<void *>(data_ptr), nullptr, data_size,
    static_cast<gert::TensorPlacement>(placement));
  t.SetData(std::move(td));
  handle->tensor = std::move(t);
  return handle;
}

void forch_ge_rt_tensor_destroy(forch_ge_rt_tensor_t *t) {
  delete t;
}

int forch_ge_rt_tensor_info(
  const forch_ge_rt_tensor_t *t, forch_ge_rt_tensor_info_t *out) {
  if (t == nullptr || out == nullptr) {
    set_error("forch_ge_rt_tensor_info: null argument");
    return -1;
  }
  const auto &origin  = t->tensor.GetOriginShape();
  const auto &storage = t->tensor.GetStorageShape();

  out->data           = const_cast<void *>(t->tensor.GetAddr());
  out->origin_rank    = static_cast<int64_t>(origin.GetDimNum());
  out->storage_rank   = static_cast<int64_t>(storage.GetDimNum());
  out->origin_dims    = &origin[0];
  out->storage_dims   = &storage[0];
  out->dtype          = static_cast<C_DataType>(t->tensor.GetDataType());
  out->origin_format  = static_cast<C_Format>(t->tensor.GetOriginFormat());
  out->storage_format = static_cast<C_Format>(t->tensor.GetStorageFormat());
  return 0;
}

// 带自定义名字的 Const 节点：镜像 es_graph_builder.h 的 EsConst 模板，
// 但名字由调用方给定（内容寻址），不再走 GenerateNodeName 自动命名。
EsCTensorHolder *forch_ge_create_const_named(EsCGraphBuilder *owner,
  const void *value, size_t data_size, const int64_t *dims, size_t dim_num,
  C_DataType data_type, const char *name) {
  if (owner == nullptr || value == nullptr || name == nullptr) {
    set_error("forch_ge_create_const_named: null owner/value/name");
    return nullptr;
  }
  auto &builder  = *owner;
  auto  ge_graph = builder.GetGraph();

  ge::Shape      shape{std::vector<int64_t>(dims, dims + dim_num)};
  ge::TensorDesc td{shape, ge::FORMAT_ND, static_cast<ge::DataType>(data_type)};
  td.SetOriginShape(shape);
  auto tensor = std::unique_ptr<ge::Tensor>(new (std::nothrow) ge::Tensor(td));
  if (tensor == nullptr) {
    set_error("forch_ge_create_const_named: tensor create failed");
    return nullptr;
  }
  tensor->SetData(reinterpret_cast<const uint8_t *>(value), data_size);
  auto av = ge::AttrValue();
  if (av.SetAttrValue(*tensor) != ge::GRAPH_SUCCESS) {
    set_error("forch_ge_create_const_named: SetAttrValue failed");
    return nullptr;
  }

  auto node =
    ge::es::CompliantNodeBuilder(ge_graph)
      .OpType("Const")
      .Name(name)
      .IrDefOutputs(
        {{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
      .IrDefAttrs({{"value", ge::es::CompliantNodeBuilder::kEsAttrOptional,
        "VT_TENSOR", av}})
      .InstanceOutputShape("y", tensor->GetTensorDesc().GetShape().GetDims())
      .InstanceOutputDataType("y", static_cast<ge::DataType>(data_type))
      .InstanceOutputFormat("y", ge::FORMAT_ND)
      .Build();
  return builder.GetTensorHolderFromNode(node, 0);
}

EsCTensor *forch_ge_create_tensor(const void *value, size_t data_size,
  const int64_t *dims, size_t dim_num, C_DataType data_type) {
  if (dim_num > 0 && dims == nullptr) {
    set_error("forch_ge_create_tensor: null dims with non-zero dim_num");
    return nullptr;
  }
  if (value == nullptr && data_size > 0) {
    set_error("forch_ge_create_tensor: null value with non-zero data_size");
    return nullptr;
  }
  std::vector<int64_t> dims_vec;
  if (dim_num > 0) {
    dims_vec.assign(dims, dims + dim_num);
  }
  ge::Shape      shape{dims_vec};
  ge::TensorDesc td{shape, ge::FORMAT_ND, static_cast<ge::DataType>(data_type)};
  td.SetOriginShape(shape);

  auto tensor = std::unique_ptr<ge::Tensor>(new (std::nothrow) ge::Tensor(td));
  if (tensor == nullptr) {
    std::string msg = "forch_ge_create_tensor: connot create tensor of type("
      + std::to_string(data_type) + ")";
    set_error(msg.c_str());
    return nullptr;
  }
  tensor->SetData(reinterpret_cast<const uint8_t *>(value), data_size);
  return reinterpret_cast<EsCTensor *>(tensor.release());
}

} // extern "C"
