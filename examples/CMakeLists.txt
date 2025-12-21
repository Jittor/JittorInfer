# dependencies

find_package(Threads REQUIRED)

# flags

llama_add_compile_flags()

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/common)
include_directories(${CMAKE_SOURCE_DIR}/third_party)
if (EMSCRIPTEN)
else()
    add_subdirectory(server)
    add_subdirectory(mpi_server)
endif()
