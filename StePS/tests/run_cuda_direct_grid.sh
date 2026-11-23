#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
nvcc="${CUDA_PATH:-/usr/local/cuda}/bin/nvcc"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
mpi_inc=$(mpicxx --showme:compile)
mpi_libs=$(mpicxx --showme:link)

for topology in free t3 cylinder cylinder-real; do
    case "$topology" in
        free) flags="" ;;
        t3) flags="-DPERIODIC" ;;
        cylinder) flags="-DPERIODIC_Z" ;;
        cylinder-real) flags="-DPERIODIC_Z -DPERIODIC_Z_NOLOOKUP" ;;
    esac
    for precision in double single; do
        precision_flag=""
        if [ "$precision" = single ]; then precision_flag="-DUSE_SINGLE_PRECISION"; fi
        echo "Testing: $topology $precision"
        # CUDA_ARCH_FLAGS may select a native architecture for the installed driver.
        # shellcheck disable=SC2086
        "$nvcc" -O3 -std=c++17 ${CUDA_ARCH_FLAGS:-} \
            -Xcompiler=-fopenmp,-ffunction-sections,-fdata-sections -Xlinker --gc-sections \
            $mpi_inc -DUSE_CUDA -DEWALD_INTERPOLATION_ORDER=2 $flags $precision_flag \
            "$root/tests/cuda_direct_grid.cu" $mpi_libs -o "$tmp/test"
        "$tmp/test"
    done
done
