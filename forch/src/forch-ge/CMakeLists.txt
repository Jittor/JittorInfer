add_subdirectory(tests)

# =============================================================================
# 1. CANN：路径、库句柄、es_all 代码生成
# =============================================================================

set(ASCEND_INSTALL_PATH $ENV{ASCEND_HOME_PATH})
set(ASCEND_LIB_DIR ${ASCEND_INSTALL_PATH}/lib64)
message(STATUS "ASCEND_LIB_DIR = ${ASCEND_LIB_DIR}")

# CANN 库改用 find_library 拿绝对路径，避免 link_directories。
find_library(GE_GRAPH        NAMES graph        REQUIRED HINTS ${ASCEND_LIB_DIR})
find_library(GE_GRAPH_BASE   NAMES graph_base   REQUIRED HINTS ${ASCEND_LIB_DIR})
find_library(GE_C_SEC        NAMES c_sec        REQUIRED HINTS ${ASCEND_LIB_DIR})
find_library(GE_RUNNER_V2    NAMES ge_runner_v2 REQUIRED HINTS ${ASCEND_LIB_DIR})

# 生成 es_all：带 EsXxx 的 _c.h 头文件以及 libes_all.so
list(APPEND CMAKE_MODULE_PATH "${ASCEND_INSTALL_PATH}/include/ge/cmake")
find_package(GenerateEsPackage REQUIRED)
add_library(opgraph_all INTERFACE)
set_target_properties(opgraph_all PROPERTIES
  INTERFACE_LIBRARY_OUTPUT_DIRECTORY  "${ASCEND_INSTALL_PATH}/opp/built-in/op_proto"
)
add_es_library(
  ES_LINKABLE_AND_ALL_TARGET es_all
  OPP_PROTO_TARGET  opgraph_all
  OUTPUT_PATH       ${CMAKE_BINARY_DIR}/output
)

# =============================================================================
# 2. shim：CANN 旧 ABI 边界，转 C 接口
#
# shim 是 forch-ge 的内部实现细节：
#   - 头文件 ge_session_shim.h 不安装；
#   - target 不进入 forchTargets export set；
#   - forch-ge 以 PRIVATE 方式链接它。
# =============================================================================

add_subdirectory(shim)

# =============================================================================
# 3. forch-ge：默认 ABI，shim 的唯一上游消费者
#
# forch-ge 显式声明自己需要的 CANN include/lib（不依赖 shim 传播）。
# 对外（消费者通过 find_package(forch) 拿到的 forch::forch-ge）：
#   - install tree 上通过 forch::forch-ge-cann 自动链接 CANN 库，不暴露 CANN 头文件；
#   - CANN 路径在消费者侧由 forchConfig.cmake 按 $ENV{ASCEND_HOME_PATH} 重新解析。
# =============================================================================

add_library(forch-ge SHARED
  backend.cpp
  common.cpp
  graph.cpp
  init.cpp
)
target_include_directories(forch-ge PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/shim    # for ge_session_shim.h
  ${ASCEND_INSTALL_PATH}/include      # for graph/c_types.h, esb_funcs.h, es_*_c.h
)
add_subdirectory(ops)

# forch-ge 直接链接 CANN 库（不再通过 shim 间接获取）。
# 顺序很重要：es_all 先于 graph/graph_base 以解决弱符号问题。
target_link_libraries(forch-ge PRIVATE
  forch-ge-shim
  es_all
  ${GE_GRAPH}
  ${GE_GRAPH_BASE}
  ${GE_C_SEC}
  ${GE_RUNNER_V2}
)

# The public GE constructor takes AscendDevice; propagate its library.
target_link_libraries(forch-ge PUBLIC forch forch-ascend)

install(TARGETS forch-ge
  EXPORT forchTargets
  LIBRARY DESTINATION lib
)
install(FILES
  ${CMAKE_BINARY_DIR}/output/lib64/libes_all.so
  DESTINATION lib
)

# 为安装树的消费者补齐 CANN 依赖：消费者 find_package(forch) 后，
# forchConfig.cmake 会创建 IMPORTED target forch::forch-ge-cann，
# 然后通过 forchTargets 中 forch::forch-ge 的 INSTALL_INTERFACE_LINK_LIBRARIES
# 引到。
target_link_libraries(forch-ge INTERFACE
  $<INSTALL_INTERFACE:forch::forch-ge-cann>
)
target_include_directories(forch-ge INTERFACE
  $<INSTALL_INTERFACE:include>
)
