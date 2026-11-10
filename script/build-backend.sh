#!/usr/bin/env bash
set -eou pipefail

cd "$(dirname "$0")"/..

BUILD_DIR=${BUILD_DIR:-build}
FORCH_BUILD_DIR=${BUILD_DIR}/forch
BACKEND_BUILD_DIR=${BUILD_DIR}/backend
THIRD_PARTY_BUILD_DIR=${BUILD_DIR}/third_party
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Debug}
NPROC=${NPROC:-$(nproc)}

# Build backend
cmake -S backend -B ${BACKEND_BUILD_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
  -DCMAKE_PREFIX_PATH="$(realpath ${FORCH_BUILD_DIR}/install);$(realpath ${THIRD_PARTY_BUILD_DIR}/install)"
cmake --build ${BACKEND_BUILD_DIR} -j${NPROC}
