/**
 * forch-ge: GE Session shim.
 *
 * 这是 forch-ge 内部唯一保留 CANN C++
 * ABI（_GLIBCXX_USE_CXX11_ABI=0、google=ascend_private） 的翻译单元。CANN 9.0.0
 * 的 GE Session（GeSession / GEInitializeV2 / gert::Tensor）只暴露 C++ 类型，无
 * C 替代物，故此 shim 把它们包装为不透明 C 句柄，让 forch-ge 其余 TU 可以
 * 用默认 ABI 编译，并只依赖 CANN 的 C
 * 头（esb_funcs.h、es_*_c.h、graph/c_types.h）。
 */
#ifndef FORCH_GE_SESSION_SHIM_H
#define FORCH_GE_SESSION_SHIM_H

#include "graph/c_types.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct EsCGraphBuilder;

typedef struct forch_ge_session   forch_ge_session_t;
typedef struct forch_ge_rt_tensor forch_ge_rt_tensor_t;

/* ---- 全局生命周期 ----
 * 集合通信（HCOM 插件）的 ranktable / rankId 等选项只有在这一层才传得进去：
 * 实测经 AddGraph 传的 ge.exec.* 到不了 HcomPlugin::InitializeHcom，插件报
 * "InitializePlugin: Init without ranktable and masterInfo"。
 *
 * options 语义同 forch_ge_session_add_graph；无选项时传 nullptr, 0。
 * 必须在任何 forch_ge_session_create 之前调用。
 *
 * 每进程只允许成功初始化一次，后续调用（包括 finalize 后）一律返回 -1。
 * 初始化和析构仅支持单线程调用，内部不加锁。
 */
int forch_ge_init(const char *const (*options)[2], size_t n_options);
int forch_ge_finalize(void);

/* ---- session ---- */
forch_ge_session_t *forch_ge_session_create(void);
void                forch_ge_session_destroy(forch_ge_session_t *s);

/* ---- 图生命周期 ----
 * 调用方通过 EsCreateGraphBuilder 取得 EsCGraphBuilder*，往里加节点与输出后
 * 直接交给 add_graph：shim 内部调用 EsBuildGraphAndReset 把 builder 的当前
 * 状态构成的 ge::Graph 注册到 session，然后销毁该 ge::Graph。builder 本身
 * 不被消耗，仍由调用方负责通过 EsDestroyGraphBuilder 析构。
 *
 * options 是一个由 {key, value} 字符串对组成的数组，长度为 n_options，直接
 * 透传给 GeSession::AddGraph 的选项 map。所有指针只在调用期间需要保持有效。
 */
int forch_ge_session_add_graph(forch_ge_session_t *s, uint32_t graph_id,
  struct EsCGraphBuilder *builder, const char *const (*options)[2],
  size_t                  n_options);
int forch_ge_session_compile(forch_ge_session_t *s, uint32_t graph_id);
int forch_ge_session_load(forch_ge_session_t *s, uint32_t graph_id);
int forch_ge_session_remove(forch_ge_session_t *s, uint32_t graph_id);

/* 输出数组由 shim 通过 malloc 分配；其每个元素由调用方各自接管所有权
 * （forch_ge_rt_tensor_destroy），外层数组用 free() 释放。 */
int forch_ge_session_run(forch_ge_session_t *s, uint32_t graph_id,
  forch_ge_rt_tensor_t *const *inputs, size_t n_in,
  forch_ge_rt_tensor_t ***outputs_out, size_t *n_out);

/* ---- runtime tensor 构造 ---- */
typedef enum {
  FORCH_GE_PLACEMENT_DEVICE_HBM = 0,
  FORCH_GE_PLACEMENT_HOST       = 1,
  FORCH_GE_PLACEMENT_FOLLOWING  = 2,
  FORCH_GE_PLACEMENT_DEVICE_P2P = 3,
} forch_ge_placement_t;

/* 统一构造接口：data_size 由 shim 内部根据 shape/dtype 计算。
 * host:   零拷贝，调用方负责 data_ptr 在 run 完成前一直存活。
 * device: 调用方负责 data_ptr 生命周期。 */
forch_ge_rt_tensor_t *forch_ge_rt_tensor_from_ptr(const void *data_ptr,
  forch_ge_placement_t placement, const int64_t *dims, int64_t dim_num,
  C_DataType dtype, C_Format format);

void forch_ge_rt_tensor_destroy(forch_ge_rt_tensor_t *t);

/* ---- runtime tensor info ----
 * 一次 FFI 调用拿到 op::GERuntime 需要的全部 metadata。
 * origin_dims / storage_dims 指针指向 t 内部，仅在 t 存活期间有效。
 */
typedef struct {
  void          *data;
  int64_t        origin_rank;
  int64_t        storage_rank;
  const int64_t *origin_dims;
  const int64_t *storage_dims;
  C_DataType     dtype;
  C_Format       origin_format;
  C_Format       storage_format;
} forch_ge_rt_tensor_info_t;

int forch_ge_rt_tensor_info(
  const forch_ge_rt_tensor_t *t, forch_ge_rt_tensor_info_t *out);

/* ---- 错误信息：thread-local，最近一次失败的消息 ---- */
const char *forch_ge_last_error(void);

/* 直接查询 GE 全局错误消息（通过 GEGetErrorMsgV3）。返回的指针仅在调用线程
 * 下次调用本函数前有效。当上层 C 接口（如 EsSetGraphOutput）返回非 0 时，
 * 真正的诊断信息在这里。 */
const char *forch_ge_get_ge_error_msg(void);

/// `data_size` is the total size of value in bytes
struct EsCTensor *forch_ge_create_tensor(const void *value, size_t data_size,
  const int64_t *dims, size_t dim_num, C_DataType data_type);

// 带自定义名字的 Const 节点创建：常量按内容哈希命名（内容寻址），使
// session 级 var 复用（constLifecycle=Session）只合并同内容常量，
// 规避不同内容同名常量的串扰（GE 按 Const_<N> 自动命名时的缺陷）。
struct EsCTensorHolder *forch_ge_create_const_named(EsCGraphBuilder *owner,
  const void *value, size_t data_size, const int64_t *dims, size_t dim_num,
  C_DataType data_type, const char *name);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FORCH_GE_SESSION_SHIM_H */
