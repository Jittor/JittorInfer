#!/bin/bash
# 装上测速工具：git clone 到 <repo>/build/inference-benchmarker，然后 cargo build --release。
#
#   bash script/benchmark/install-inference-benchmarker.sh
#
# bench-performance.sh 用这里装出的二进制。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

if [[ $# -ne 0 ]]; then
  echo "!! 固定装到 $REPO_ROOT/build/inference-benchmarker，不接受安装目录" >&2
  exit 2
fi

DIR="$REPO_ROOT/build/inference-benchmarker"
REPO="${IB_REPO:-git@github.com:CoderOJ/inference-benchmarker.git}"
BIN="$DIR/target/release/inference-benchmarker"

if [ -x "$BIN" ]; then
  echo "=== 已经装好：$BIN"
  exit 0
fi

export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || {
  echo "!! 需要 cargo（rustup）；已经有 rustup 的话 export PATH=\$HOME/.cargo/bin:\$PATH" >&2
  exit 1
}

if [ -d "$DIR/.git" ]; then
  echo "=== $DIR 已是 git 仓库，直接编译"
else
  echo "=== git clone $REPO $DIR"
  git clone "$REPO" "$DIR"
fi

cd "$DIR"
cargo build --release --bin inference-benchmarker
echo "=== 完成：$BIN"
