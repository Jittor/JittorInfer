cmake_minimum_required(VERSION 3.16)
project(forch_infer_backend CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(forch REQUIRED COMPONENTS GE)
find_package(ZeroMQ REQUIRED)
find_package(Threads REQUIRED)
set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)

add_library(forch_infer INTERFACE)
target_include_directories(
  forch_infer INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
  forch_infer INTERFACE
  forch::forch-ge
)


# Collect all binary target
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/bin/*.cpp")


# Create executable for each source file in the bin directory
foreach(SRC_FILE ${SRC_FILES})
  get_filename_component(TARGET_NAME ${SRC_FILE} NAME_WE)
  add_executable(${TARGET_NAME} ${SRC_FILE})
  target_link_libraries(${TARGET_NAME} PRIVATE forch_infer)
endforeach()

# Server transports use ZMQ.
target_link_libraries(engine_server PRIVATE libzmq Threads::Threads MPI::MPI_CXX)

# Regression checks and model comparison harnesses are opt-in.
option(BUILD_TESTING "Build backend regression checks and NPU harnesses" OFF)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
