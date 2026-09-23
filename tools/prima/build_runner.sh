#!/usr/bin/env bash
# Build the small PRIMA C runner without installing PRIMA into Python.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 PRIMA_SOURCE OUTPUT_BINARY" >&2
  exit 2
fi

prima_source="$(realpath "$1")"
output_binary="$(realpath -m "$2")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

if [[ ! -f "$prima_source/CMakeLists.txt" ]]; then
  echo "PRIMA_SOURCE must point to a libprima/prima checkout" >&2
  exit 2
fi

fc="${FC:-gfortran}"
cc="${CC:-cc}"
read -r -a fc_flags <<< "${PRIMA_FORTRAN_FLAGS:-}"

cmake -S "$prima_source" -B "$build_dir" \
  -DPRIMA_ENABLE_C=ON -DPRIMA_ENABLE_PYTHON=OFF \
  -DPRIMA_ENABLE_EXAMPLES=OFF -DPRIMA_ENABLE_TESTING=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_Fortran_COMPILER="$fc" \
  -DCMAKE_Fortran_FLAGS="${PRIMA_FORTRAN_FLAGS:-}"
cmake --build "$build_dir" --target primac -j "${PRIMA_BUILD_JOBS:-4}"

"$cc" -O2 -I"$prima_source/c/include" \
  -c "$repo_root/tools/prima/runner.c" -o "$build_dir/runner.o"
mkdir -p "$(dirname "$output_binary")"
"$fc" "${fc_flags[@]}" "$build_dir/runner.o" \
  "$build_dir/c/libprimac.a" "$build_dir/fortran/libprimaf.a" \
  -lm -o "$output_binary"
echo "$output_binary"
