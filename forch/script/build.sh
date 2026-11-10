#!/usr/bin/env bash
set -eou pipefail

cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug

# or, build with release
# cmake -S . -B build -D CMAKE_BUILD_TYPE=Release

cmake --build build -j

# 目前 ge 中 es_all 的增量编译有一定的问题，若发现在编译 es_all_build/generated_code/* 时没有并行编译，
# 可以删除 build/src/forch-ge 目录清理环境，从而触发正常的并行编译。