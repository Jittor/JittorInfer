#!/usr/bin/env bash
set -eou pipefail

cd "$(dirname "$0")"/..

BUILD_DIR=${BUILD_DIR:-build}
FORCH_BUILD_DIR=${BUILD_DIR}/forch
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Debug}
NPROC=${NPROC:-$(nproc)}

# Build forch
cmake -S forch -B ${FORCH_BUILD_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
cmake --build ${FORCH_BUILD_DIR} -j${NPROC}
cmake --install ${FORCH_BUILD_DIR} --prefix ${FORCH_BUILD_DIR}/install
