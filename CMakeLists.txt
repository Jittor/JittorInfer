# 版本检查
cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("llama.cpp" C CXX)

# 先尝试查找MPI包
find_package(MPI QUIET)
if(MPI_FOUND)
    message(STATUS "MPI found: ${MPI_C_INCLUDE_PATH}")
    message(STATUS "MPI libraries: ${MPI_C_LIBRARIES}")
else()
    message(STATUS "MPI not found (will be optional)")
endif()

# 头文件检查
include(CheckIncludeFileCXX)

# 编译选项：开启未使用的警告
set(CMAKE_WARN_UNUSED_CLI YES)

# 编译选项：启动编译输出
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 编译选项：设置默认构建类型为 Release；条件：不是 Xcode (苹果)、不是 MSVC (Microsoft Visual C++)、不是 CMAKE_BUILD_TYPE
if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 判断是否为独立项目
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)
else()
    set(LLAMA_STANDALONE OFF)
endif()

# 判断依赖：EMSCRIPTEN，MINGW
option(BUILD_SHARED_LIBS "build shared libraries" ON)

#
# option list
#

# debug
option(LLAMA_ALL_WARNINGS           "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY "llama: enable all compiler warnings in 3rd party libs" OFF)

# build
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF)

# MPI
option(LLAMA_MPI "llama: enable MPI support" ON)

# 如果启用了MPI但未找到MPI，给出错误提示
if(LLAMA_MPI AND NOT MPI_FOUND)
    message(FATAL_ERROR 
        "LLAMA_MPI is enabled but MPI was not found.\n"
        "Please either:\n"
        "  1. Install MPI (e.g., sudo apt-get install libopenmpi-dev on Ubuntu)\n"
        "  2. Set LLAMA_MPI=OFF to disable MPI support (cmake -DLLAMA_MPI=OFF)\n"
        "  3. Specify MPI path manually (cmake -DMPI_C_COMPILER=/path/to/mpicc)"
    )
endif()

# JITTOR OPS
option(LLAMA_USE_JITTOR_OPS "llama: enable jittor ops" OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD    "llama: enable thread sanitizer"    OFF)
option(LLAMA_SANITIZE_ADDRESS   "llama: enable address sanitizer"   OFF)
option(LLAMA_SANITIZE_UNDEFINED "llama: enable undefined sanitizer" OFF)

# utils
option(LLAMA_BUILD_COMMON "llama: build common utils library" ON)

# extra artifacts
option(LLAMA_BUILD_TESTS    "llama: build tests"          ON)
option(LLAMA_BUILD_EXAMPLES "llama: build examples"       ON)
option(LLAMA_BUILD_SERVER   "llama: build server example" ${LLAMA_STANDALONE})

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

# override ggml options
set(GGML_ALL_WARNINGS   ${LLAMA_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${LLAMA_FATAL_WARNINGS})

# change the default for these ggml options
if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()

# transition helpers
function (llama_option_depr TYPE OLD NEW)
    if (${OLD})
        message(${TYPE} "${OLD} is deprecated and will be removed in the future.\nUse ${NEW} instead\n")
        set(${NEW} ON PARENT_SCOPE)
    endif()
endfunction()

llama_option_depr(WARNING     LLAMA_KOMPUTE             GGML_KOMPUTE)
llama_option_depr(WARNING     LLAMA_METAL               GGML_METAL)
llama_option_depr(WARNING     LLAMA_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY)
llama_option_depr(WARNING     LLAMA_NATIVE              GGML_NATIVE)
llama_option_depr(WARNING     LLAMA_RPC                 GGML_RPC)
llama_option_depr(WARNING     LLAMA_SYCL                GGML_SYCL)
llama_option_depr(WARNING     LLAMA_SYCL_F16            GGML_SYCL_F16)
llama_option_depr(WARNING     LLAMA_CANN                GGML_CANN)

# sanitizers are ignored.

#
# 3rd-party
#

if (NOT TARGET ggml)
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

#
# build the library
#

add_subdirectory(src)

#
# utils, programs, examples and tests
#

# if (LLAMA_BUILD_COMMON)
#     add_subdirectory(common)
# endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

#
# install
#

include(GNUInstallDirs)

set(LLAMA_BUILD_NUMBER        ${BUILD_NUMBER})
set(LLAMA_BUILD_COMMIT        ${BUILD_COMMIT})
set(LLAMA_INSTALL_VERSION 0.0.${BUILD_NUMBER})

set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(LLAMA_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(LLAMA_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

set(LLAMA_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama.h)

set_target_properties(llama
    PROPERTIES
        PUBLIC_HEADER "${LLAMA_PUBLIC_HEADERS}")

install(TARGETS llama LIBRARY PUBLIC_HEADER)

# configure project version
set(PROJECT_VERSION ${LLAMA_INSTALL_VERSION})
