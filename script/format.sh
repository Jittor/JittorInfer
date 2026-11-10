#!/usr/bin/env bash
set -eou pipefail

for name in $(find backend -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \))
do
	echo Formatting $name
	clang-format-18 -i $name
done
