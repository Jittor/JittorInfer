set(ASCEND_INSTALL_PATH $ENV{ASCEND_HOME_PATH})
find_library(FORCH_ACL_RT NAMES ascendcl aclrt REQUIRED
  HINTS "${ASCEND_INSTALL_PATH}/lib64")
add_library(forch-ascend SHARED device.cpp)
target_include_directories(forch-ascend PRIVATE "${ASCEND_INSTALL_PATH}/include")
target_link_libraries(forch-ascend PUBLIC forch PRIVATE ${FORCH_ACL_RT})
target_link_libraries(forch-ascend INTERFACE
  $<INSTALL_INTERFACE:forch::forch-ascend-cann>)
install(TARGETS forch-ascend EXPORT forchTargets LIBRARY DESTINATION lib)
