add_library(forch-ge-shim OBJECT ge_session_shim.cpp)
set_target_properties(forch-ge-shim PROPERTIES POSITION_INDEPENDENT_CODE ON)

# shim 是 forch-ge 的内部实现细节：
#   - 头文件 ge_session_shim.h 不安装；
#   - target 不进入 forchTargets export set；
#   - CANN include/lib 全部 PRIVATE——forch-ge 自己（以及消费者）不会通过 shim
#     间接拿到 CANN 头文件。
target_include_directories(forch-ge-shim PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ASCEND_INSTALL_PATH}/include
)
target_link_libraries(forch-ge-shim PRIVATE
  es_all
  ${GE_GRAPH}
  ${GE_GRAPH_BASE}
  ${GE_C_SEC}
  ${GE_RUNNER_V2}
)
target_compile_definitions(forch-ge-shim PRIVATE
  _GLIBCXX_USE_CXX11_ABI=0
  google=ascend_private
)

# 注意：本 target 不进入 forchTargets export set，对应的 ge_session_shim.h
# 也不安装。shim 是 forch-ge 的内部实现细节。
