cmake_minimum_required(VERSION 3.18)

# Project name
project(Forch)
set(FORCH_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile Flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -rdynamic -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -rdynamic -O3")

# Create compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(forch SHARED)
target_include_directories(
  forch PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(
  forch SYSTEM PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
)

option(FORCH_ENABLE_GE "Enable GE backend" ON)
option(FORCH_ENABLE_ASCEND "Enable Ascend device support" ${FORCH_ENABLE_GE})
if(FORCH_ENABLE_GE AND NOT FORCH_ENABLE_ASCEND)
  message(FATAL_ERROR "FORCH_ENABLE_GE requires FORCH_ENABLE_ASCEND")
endif()

add_subdirectory(src)
add_subdirectory(example)

install(TARGETS forch
  EXPORT forchTargets
  LIBRARY DESTINATION lib
)
install(DIRECTORY include/ 
  DESTINATION include/
)
install(DIRECTORY third_party/
  DESTINATION include/
)
install(EXPORT forchTargets
  FILE forchTargets.cmake
  NAMESPACE forch::
  DESTINATION lib/cmake/forch
)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/forchConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/forchConfig.cmake
  INSTALL_DESTINATION lib/cmake/forch
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/forchConfig.cmake
  DESTINATION lib/cmake/forch
)