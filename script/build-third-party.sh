#!/usr/bin/env bash
set -eou pipefail

cd "$(dirname "$0")"/..

BUILD_DIR=${BUILD_DIR:-build}
THIRD_PARTY_BUILD_DIR=${BUILD_DIR}/third_party
mkdir -p "${THIRD_PARTY_BUILD_DIR}/install"
INSTALL_PREFIX=$(realpath ${THIRD_PARTY_BUILD_DIR}/install)
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Debug}
NPROC=${NPROC:-$(nproc)}

echo "=== Building third-party libraries → ${INSTALL_PREFIX} ==="

for lib_dir in third_party/*/; do
  name=$(basename "$lib_dir")
  [ -f "$lib_dir/CMakeLists.txt" ] || continue

  echo "--- Building ${name} ---"
  cmake -S "$lib_dir" -B "${THIRD_PARTY_BUILD_DIR}/build-${name}" \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DBUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
  cmake --build "${THIRD_PARTY_BUILD_DIR}/build-${name}" -j${NPROC}
  cmake --install "${THIRD_PARTY_BUILD_DIR}/build-${name}"
done

echo "=== Third-party libraries installed to ${INSTALL_PREFIX} ==="
