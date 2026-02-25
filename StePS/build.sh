#!/usr/bin/env bash
set -e

if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: conda environment not activated. Run: conda activate steps" >&2
    exit 1
fi

make clean

# Convenience conda compiler/library paths. Any extra arguments are forwarded
# to make (see README for the available build options).
make -j8 \
  CXX=x86_64-conda-linux-gnu-c++ \
  MPI_INC="-I$CONDA_PREFIX/include" \
  MPI_LIBS="-L$CONDA_PREFIX/lib -lmpi" \
  HDF5_INC="-I$CONDA_PREFIX/include" \
  HDF5_LIBS="-L$CONDA_PREFIX/lib -lhdf5" \
  "$@"
