#!/usr/bin/env bash
set -eou pipefail

for name in $(find example include src -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \))
do
	echo Formatting $name
	clang-format -i $name
done
