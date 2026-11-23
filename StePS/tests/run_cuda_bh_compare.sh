#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
nvcc="${CUDA_PATH:-/usr/local/cuda}/bin/nvcc"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
mpi_inc=$(mpicxx --showme:compile)
mpi_libs=$(mpicxx --showme:link)
common="-std=c++17 -fopenmp -ffunction-sections -fdata-sections -I$root/src"

run_case() {
    flags="-DUSE_BH=0.4 -DUSE_CUDA $*"
    echo "Testing: $flags"
    # Compile CPU and GPU solvers together under different entry-point names.
    # shellcheck disable=SC2086
    mpicxx $common $flags -Dforces=cpu_forces -Dforces_periodic=cpu_forces_periodic \
        -Dforces_periodic_z=cpu_forces_periodic_z \
        -Drecalculate_softening=cpu_recalculate_softening \
        -c "$root/src/forces.cc" -o "$tmp/cpu.o"
    # shellcheck disable=SC2086
    mpicxx $common $flags -UUSE_CUDA -Dget_cubic_weights=cpu_get_cubic_weights \
        -c "$root/src/ewald_space.cc" -o "$tmp/ewald.o"
    # shellcheck disable=SC2086
    mpicxx $common $flags -c "$root/src/utils.cc" -o "$tmp/utils.o"
    # shellcheck disable=SC2086
    mpicxx $common $flags -c "$root/tests/cuda_bh_compare.cc" -o "$tmp/test.o"
    # shellcheck disable=SC2086
    "$nvcc" -std=c++17 -Xcompiler '-fopenmp,-ffunction-sections,-fdata-sections' \
        $mpi_inc $flags -c "$root/src/forces_cuda.cu" -o "$tmp/cuda.o"
    # shellcheck disable=SC2086
    "$nvcc" -Xcompiler -fopenmp -Xlinker --gc-sections \
        "$tmp/test.o" "$tmp/cpu.o" "$tmp/ewald.o" "$tmp/utils.o" "$tmp/cuda.o" \
        $mpi_libs -o "$tmp/test"
    # The CPU cylindrical solver shares a temporary between OpenMP threads.
    OMP_NUM_THREADS=1 "$tmp/test" > "$tmp/output" || {
        cat "$tmp/output"
        return 1
    }
    case "$flags" in
        *-DRANDOMIZE_BH=*)
            awk '
                /Octree force calculation started with random/ {
                    message = $0
                    if (message ~ /the$/ || message ~ /\)$/) {
                        if (getline continuation <= 0) exit 1
                        message = message "\n" continuation
                    }
                    ++count
                    if (count <= 24) {
                        if (count % 2) previous = message
                        else if (message != previous) {
                            print "CPU/GPU randomization messages differ:" > "/dev/stderr"
                            print previous "\n" message > "/dev/stderr"
                            exit 1
                        }
                    }
                }
                END { if (count != 25) exit 1 }
            ' "$tmp/output" ;;
        *)
            if grep -q 'Octree force calculation started with random' "$tmp/output"; then
                echo "Unexpected randomization message" >&2
                return 1
            fi ;;
    esac
    grep 'maxerr=' "$tmp/output" | tail -1
}

for topology in free t3 cylinder cylinder-real; do
    case "$topology" in
        free) topo="" ;;
        t3) topo="-DPERIODIC" ;;
        cylinder) topo="-DPERIODIC_Z" ;;
        cylinder-real) topo="-DPERIODIC_Z -DPERIODIC_Z_NOLOOKUP" ;;
    esac
    # shellcheck disable=SC2086
    run_case -DEWALD_INTERPOLATION_ORDER=2 $topo
    # shellcheck disable=SC2086
    run_case -DEWALD_INTERPOLATION_ORDER=2 -DRANDOMIZE_BH=123 $topo
    # shellcheck disable=SC2086
    run_case -DEWALD_INTERPOLATION_ORDER=2 -DUSE_SINGLE_PRECISION $topo
done
for topology in -DPERIODIC -DPERIODIC_Z; do
    for order in 0 4; do
        run_case "-DEWALD_INTERPOLATION_ORDER=$order" -DRANDOMIZE_BH=123 "$topology"
        if [ "$topology" = -DPERIODIC ]; then
            run_case "-DEWALD_INTERPOLATION_ORDER=$order" -DUSE_SINGLE_PRECISION "$topology"
        fi
    done
done
