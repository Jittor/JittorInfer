#!/usr/bin/env bash
set -eou pipefail

# examples 中大部分是 ai 生成，不 tidy

clang-tidy-18 -p build/ src/**.cpp