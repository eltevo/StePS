# StePS

**STEreographically Projected cosmological Simulations**

StePS is a C++ cosmological N-body code for dark-matter-only simulations in $\mathbb{R}^3$, $S^1 \times \mathbb{R}^2$, and $\mathbb{T}^3$ topological manifolds. It supports direct summation on CPUs and NVIDIA GPUs, and a CPU Barnes-Hut octree solver for larger particle counts.

## Authors and Contributors

### Primary author

**Gábor Rácz**<br>
Department of Physics, University of Helsinki<br>
Jet Propulsion Laboratory, California Institute of Technology<br>
Department of Physics of Complex Systems, Eötvös Loránd University<br>
Department of Physics & Astronomy, Johns Hopkins University

### Contributors

**Viola Varga**<br>
Department of Physics, University of Helsinki

**Balázs Pál**<br>
Department of Physics of Complex Systems, Eötvös Loránd University<br>
Institute for Particle and Nuclear Physics, HUN-REN Wigner Research Centre for Physics

## Main features
- Optimized to run dark matter only N-body simulations in ΛCDM, wCDM or CPL (w0waCDM) cosmologies.
- Running simulations with other expansion rates are possible by using external tabulated expansion histories.
- Able to run standard periodic ($\mathbb{T}^3$), cylindrical ($S^1 \times \mathbb{R}^2$), and non-periodic spherical ($\mathbb{R}^3$) cosmological simulations.
- Direct [CPU & GPU], Octree (a.k.a. Barnes-Hut)[CPU only], and randomized Octree [CPU only] force calculation.
- Can be used to make periodic, quasi-periodic, cylindrical or spherical glass.
- Available for GNU/Linux and Darwin (macOS).
- Written in C++ with MPI, OpenMP and CUDA parallelization.
- Able to use multiple GPUs simultaneously in a large computing cluster.
- Supported Initial Condition formats are HDF5, Gadget2, and ASCII.
- Supported output formats are ASCII and HDF5.

The StePS initial condition generator is maintained separately at [`eltevo/stepsic`](https://github.com/eltevo/stepsic).

## Repository Layout

All commands in this document assume you are in the StePS source directory:

```bash
cd StePS/StePS
```

Important files and directories:

| Path | Purpose |
| --- | --- |
| `src/` | C++ and CUDA source files |
| `examples/` | Example parameter files and bundled initial conditions |
| `Template-LinuxGCC-Makefile` | GNU/Linux template for GCC or conda GCC |
| `Template-LinuxICC-Makefile` | GNU/Linux template for Intel compilers |
| `Template-Darwin-Makefile` | macOS template for LLVM/Clang |
| `environment.yml` | Conda environment for the recommended Linux build |
| `build.sh` | Convenience conda build script; defaults to a CPU build and forwards make variables such as `USING_CUDA=YES` |
| `CHANGELOG.md` | Version history |

## Requirements

Required:

- C++17 compiler: GCC, Intel `icpc`/`icpx`, or Clang
- MPI implementation: OpenMPI is recommended
- OpenMP support
- `make`

Optional:

- HDF5 for HDF5 input/output. The templates enable it by default with `-DHAVE_HDF5`.
- NVIDIA CUDA Toolkit for GPU acceleration. CUDA builds are supported on GNU/Linux only.

CUDA and Barnes-Hut are currently separate build modes: a CUDA executable cannot also use `-DUSE_BH`.

## Installation

### Get the Source

```bash
git clone https://github.com/eltevo/StePS.git
cd StePS/StePS
```

### Recommended Linux Build with Conda

The conda environment provides a consistent GCC toolchain, OpenMPI, HDF5, and build utilities. It does not install CUDA; for GPU builds, install the CUDA Toolkit system-wide first.

```bash
conda env create -f environment.yml
conda activate steps
cp Template-LinuxGCC-Makefile Makefile
./build.sh
```

With no extra arguments, `build.sh` runs a clean build with the conda compiler, MPI, and HDF5 paths and produces the CPU executable:

```text
build/StePS
```

Any additional arguments are forwarded to `make`. For a GPU build, pass `USING_CUDA=YES`. The Makefile defaults `CUDA_PATH` to `/usr/local/cuda`; override it with `CUDA_PATH=...` if the CUDA Toolkit is installed elsewhere:

```bash
./build.sh USING_CUDA=YES
./build.sh USING_CUDA=YES CUDA_PATH=/opt/cuda
```

This produces the CUDA executable:

```text
build/StePS_CUDA
```

Before running `./build.sh`, edit `Makefile` if you need a different topology, precision, force solver, or compile-time cosmology option.

### Manual Linux Build

Install a compiler, MPI, OpenMP, and optionally HDF5/CUDA through your system or cluster environment. Then copy the closest template and edit paths in `Makefile`:

```bash
cp Template-LinuxGCC-Makefile Makefile
```

Set these variables as needed:

```make
CXX       = g++
MPI_INC   = -I/path/to/mpi/include
MPI_LIBS  = -L/path/to/mpi/lib -lmpi
HDF5_INC  = -I/path/to/hdf5/include
HDF5_LIBS = -L/path/to/hdf5/lib -lhdf5
CUDA_PATH = /path/to/cuda
```

Build:

```bash
make -j
```

For CUDA, enable the build mode (`CUDA_PATH` above must point at the CUDA Toolkit):

```make
USING_CUDA = YES
```

Then rebuild:

```bash
make clean
make -j
```

### Intel Compiler Build

For Intel compiler environments, start from the Intel template:

```bash
cp Template-LinuxICC-Makefile Makefile
```

Set these variables to match your installed Intel, MPI, HDF5, and CUDA modules:

```make
CXX       = icpx          # or icpc
MPI_INC   = -I/path/to/intel/mpi/include
MPI_LIBS  = -L/path/to/intel/mpi/lib -lmpi
HDF5_INC  = -I/path/to/hdf5/include
HDF5_LIBS = -L/path/to/hdf5/lib -lhdf5
CUDA_PATH = /path/to/cuda
```

Build:

```bash
make -j
```

### macOS Build

CUDA is not supported by the macOS template. Install LLVM/Clang, OpenMPI, libomp, and HDF5, then start from:

```bash
cp Template-Darwin-Makefile Makefile
```

Set these variables to your Homebrew or local installation paths:

```make
CXX       = /path/to/llvm/bin/clang
MPI_INC   = -I/path/to/mpi/include
MPI_LIBS  = -L/path/to/mpi/lib -lmpi
OMP_INC   = -I/path/to/libomp/include
OMP_LIBS  = -L/path/to/libomp/lib
HDF5_INC  = -I/path/to/hdf5/include
HDF5_LIBS = -L/path/to/hdf5/lib -lhdf5
```

Build:

```bash
make -j
```

## Compile-Time Configuration

StePS uses compile-time options to produce an executable optimized for one simulation family. Edit the top of `Makefile` before building.

### Core Build Options

| Option | Description |
| --- | --- |
| `USING_CUDA = YES` | Builds `build/StePS_CUDA` with CUDA force kernels. Linux only. |
| `USING_CUDA = NO` | Builds the CPU executable `build/StePS`. |
| `OPT += -DUSE_SINGLE_PRECISION` | Uses 32-bit precision in force calculation. Faster and lower-memory, especially on GPUs. |
| `OPT += -DHAVE_HDF5` | Enables HDF5 input and output. Required for `OUTPUT_FORMAT 2`. |
| `OPT += -DGLASS_MAKING` | Enables glass-making mode with reversed gravity. |
| `OPT += -DSAVE_ACCELERATIONS` | Saves calculated accelerations to HDF5 snapshots. Requires HDF5 output. |

### Force Solvers

| Option | Description |
| --- | --- |
| No force-solver flag | Direct summation. |
| `OPT += -DUSE_BH=0.25` | CPU Barnes-Hut octree solver with opening angle `theta = 0.25`. |
| `OPT += -DRANDOMIZE_BH=123456` | Randomizes Barnes-Hut domains using the given seed. Recommended with Barnes-Hut. |

Barnes-Hut is CPU-only. Do not combine `USING_CUDA = YES` with `-DUSE_BH`.

### Boundary Conditions and Topology

| Option | Description |
| --- | --- |
| No boundary flag | Non-periodic `R^3` topology. |
| `OPT += -DPERIODIC` | Fully periodic `T^3` topology. |
| `OPT += -DPERIODIC_Z` | Cylindrical `S^1 x R^2` topology with periodic `z`. |
| `OPT += -DPERIODIC_Z_RSPACELOOKUP` | Builds the cylindrical lookup table by direct real-space summation. Requires `-DPERIODIC_Z`. |
| `OPT += -DPERIODIC_Z_NOLOOKUP` | Uses direct real-space summation of periodic images for every cylindrical interaction. Requires `-DPERIODIC_Z`. |
| `EWALD_INTERPOLATION_ORDER = 0` | NGP interpolation for Ewald lookup tables. |
| `EWALD_INTERPOLATION_ORDER = 2` | CIC interpolation. Current template default. |
| `EWALD_INTERPOLATION_ORDER = 4` | TSC interpolation. More accurate, usually slower. |

Use only one of `-DPERIODIC` and `-DPERIODIC_Z`.

The compile-time boundary flag and the parameter-file `IS_PERIODIC` setting must match. If they contradict each other, StePS exits with an error.

### Cosmology Options

Set the background cosmology at compile time:

| Option | Description |
| --- | --- |
| `OPT += -DCOSMOPARAM=0` | Standard ΛCDM. |
| `OPT += -DCOSMOPARAM=1` | wCDM with constant dark-energy equation of state `w0`. |
| `OPT += -DCOSMOPARAM=2` | w0waCDM, also known as CPL. |
| `OPT += -DCOSMOPARAM=-1` | Read tabulated expansion history from an ASCII file. |

## Running Simulations

Run StePS from the source directory so relative paths in the example parameter files resolve correctly.

The output directory specified by `OUT_DIR` must already exist. StePS checks this at startup and exits if the directory is missing.

### CPU Executable

Use `OMP_NUM_THREADS` to set OpenMP threads per MPI task:

```bash
export OMP_NUM_THREADS=8
mpirun -np 1 ./build/StePS ./examples/LCDM_SP_1860_com_VOI100.param
```

The CPU executable also accepts an optional third argument that sets the OpenMP thread count:

```bash
mpirun -np 1 ./build/StePS ./examples/LCDM_SP_1860_com_VOI100.param 8
```

On some MPI installations, process binding can slow down hybrid MPI/OpenMP runs. Disable binding when needed:

```bash
mpirun -np 4 --bind-to none ./build/StePS ./examples/LCDM_SP_1860_com_VOI100.param
```

### CUDA Executable

For CUDA builds, the optional third argument is the number of GPUs per MPI task. If omitted, StePS uses one GPU per MPI task.

```bash
export OMP_NUM_THREADS=1
mpirun -np 1 ./build/StePS_CUDA ./examples/LCDM_SP_1860_com_VOI100.param 1
```

For multiple GPUs per task, set both the OpenMP thread count and the GPU count to the number of GPUs assigned to that task:

```bash
export OMP_NUM_THREADS=2
mpirun -np 1 ./build/StePS_CUDA ./examples/LCDM_SP_1860_com_VOI100.param 2
```

On clusters, launch the MPI process count, OpenMP thread count, and GPU count to match the resources allocated by the scheduler.

## Example Workflows

### Included Spherical LambdaCDM Example

The repository includes a compactified `R^3` ΛCDM example with an HDF5 initial condition:

```text
examples/LCDM_SP_1860_com_VOI100.param
examples/ic/IC_LCDM_SP_1860Mpc_Nr224_Nhp32_ds105_z63_VOI100.hdf5
```

Create the output directory:

```bash
mkdir -p ./examples/LCDM_SP_1860_com_VOI100
```

Run on one GPU:

```bash
export OMP_NUM_THREADS=1
mpirun -np 1 ./build/StePS_CUDA ./examples/LCDM_SP_1860_com_VOI100.param 1
```

Run on CPU:

```bash
export OMP_NUM_THREADS=8
mpirun -np 1 ./build/StePS ./examples/LCDM_SP_1860_com_VOI100.param
```

This example contains about 1.8 million particles, so GPU acceleration is recommended when available.

### Non-Comoving LambdaCDM Example

The non-comoving example uses the same bundled initial condition family:

```text
examples/LCDM_SP_1860_noncom_VOI100.param
```

For non-comoving cosmological runs, `a_max` is interpreted as the final physical time in Gyr. StePS prints a warning if `a_max` is left at `1.0`, because that is often intended as a final scale factor in comoving runs.

### Cylindrical Examples

The cylindrical `S^1 x R^2` parameter files are:

```text
examples/LCDM_cylindrical_Lz100_D1000.param
examples/LCDM_cylindrical_Lz500_D1000.param
```

Use them with an executable built with:

```make
OPT += -DPERIODIC_Z
```

These parameter files reference cylindrical initial conditions and, for Barnes-Hut correction, glass files that are not bundled with this source tree. Generate or provide matching files before running them. The StePS initial condition generator is available at [`eltevo/stepsic`](https://github.com/eltevo/stepsic).

## Parameter Files

Parameter files are plain text files with one keyword and one value per line. Lines in the example files are grouped for readability, but StePS scans by keyword.

### Cosmological Parameters

| Parameter | Meaning |
| --- | --- |
| `Omega_b` | Baryon density parameter. |
| `Omega_lambda` | Dark-energy density parameter. |
| `Omega_m` | Matter density parameter. |
| `Omega_r` | Radiation density parameter. |
| `HubbleConstant` | Hubble constant in `km/s/Mpc`. |
| `a_start` | Initial scale factor for cosmological runs. |
| `a_max` | Final scale factor for comoving runs, or final physical time in Gyr for non-comoving runs. |

### Simulation Parameters

| Parameter | Meaning |
| --- | --- |
| `COSMOLOGY` | `1` for cosmological simulations, `0` for traditional N-body simulations. |
| `IS_PERIODIC` | Boundary behavior. `0` = open boundaries, `1` = nearest images, `2` or larger = Ewald/high-precision periodic forces where supported. |
| `COMOVING_INTEGRATION` | `1` for comoving integration, `0` for physical-coordinate integration. |
| `L_BOX` | Periodic box length or characteristic source-box size. |
| `R_SIM` | Simulation radius for `R^3` or cylindrical `S^1 x R^2` runs. |
| `IC_FILE` | initial condition file path. Relative paths are resolved from the run directory. |
| `IC_FORMAT` | `0` = ASCII, `1` = Gadget2 binary, `2` = Gadget-HDF5/HDF5. |
| `OUT_DIR` | Output directory. It must exist before startup. |
| `OUT_LST` | Optional output time/redshift list. If unavailable, StePS uses `FIRST_T_OUT` and `H_OUT`. |
| `OUTPUT_TIME_VARIABLE` | `0` = physical time in Gyr, `1` = redshift. |
| `OUTPUT_FORMAT` | `0` = ASCII, `2` = HDF5. HDF5 requires `-DHAVE_HDF5`. |
| `REDSHIFT_CONE` | `0` = normal snapshots, `1` = redshift cone output. |
| `MIN_REDSHIFT` | Minimum redshift used in redshift-cone simulations. |
| `ACC_PARAM` | Timestep accuracy parameter. Smaller values are more accurate and slower. |
| `STEP_MIN` | Minimum timestep in Gyr. |
| `STEP_MAX` | Maximum timestep in Gyr. |
| `PARTICLE_RADII` | Softening length of the minimum-mass particle. Comoving units when `COMOVING_INTEGRATION = 1`, physical units otherwise. |
| `FIRST_T_OUT` | First output time or redshift, depending on `OUTPUT_TIME_VARIABLE`. |
| `H_OUT` | Output spacing in time or redshift, depending on `OUTPUT_TIME_VARIABLE`. |
| `SNAPSHOT_START_NUMBER` | Initial snapshot number for restart or continuation workflows. |
| `H_INDEPENDENT_UNITS` | `0` = Mpc/Msol-style units, `1` = Mpc/h and Msol/h-style units. |
| `TIME_LIMIT_IN_MIN` | Wall-clock time limit in minutes. `0` or omitted disables the limit. |

### Cylindrical Force Table Parameters

These parameters are used by `S^1 x R^2` builds:

| Parameter | Meaning |
| --- | --- |
| `RADIAL_FORCE_ACCURACY` | Integration resolution for radial force lookup-table calculation. |
| `RADIAL_FORCE_TABLE_SIZE` | Number of radial lookup-table samples. |

### Barnes-Hut Parameters

These parameters are used only when the executable is built with `-DUSE_BH`:

| Parameter | Meaning |
| --- | --- |
| `RADIAL_BH_FORCE_CORRECTION` | `0` disables radial correction, `1` enables it for non-periodic and cylindrical runs. |
| `GLASS_FILE_FOR_BH_FORCE_CORRECTION` | Glass file used to estimate the correction. Use `None` to use the IC file. |
| `RADIAL_BH_FORCE_TABLE_SIZE` | Lookup-table size for radial BH force correction. |
| `RADIAL_BH_FORCE_TABLE_ITERATION` | Iteration count for randomized BH correction table calculation. |

### Alternative Cosmology Parameters

These are active only when the matching `COSMOPARAM` value is compiled in:

| Compile option | Parameter | Meaning |
| --- | --- | --- |
| `-DCOSMOPARAM=1` | `w0` | Constant dark-energy equation of state. |
| `-DCOSMOPARAM=2` | `w0` | Dark-energy equation of state at `z = 0`. |
| `-DCOSMOPARAM=2` | `wa` | CPL evolution parameter. |
| `-DCOSMOPARAM=-1` | `EXPANSION_FILE` | ASCII file with columns `t [Gyr]`, `a(t)`, `H(t) [km/s/Mpc]`. |
| `-DCOSMOPARAM=-1` | `INTERPOLATION_ORDER` | Interpolation order for tabulated histories. Supported values: `1`, `2`, `3`. |

## Output Files

### Snapshot Output

If `OUTPUT_FORMAT = 2` and HDF5 support is enabled, StePS writes HDF5 snapshots. The HDF5 header includes simulation metadata such as cosmology, geometry, version, compiler, build date, git branch, and git commit.

If `OUTPUT_FORMAT = 0`, particle snapshots are written as ASCII files:

```text
z*.dat, t*.dat:
x[Mpc]  y[Mpc]  z[Mpc]  v_x[km/s]  v_y[km/s]  v_z[km/s]  M[1e11 M_sol]
```

Redshift-cone ASCII output:

```text
redshift_cone.dat:
x[Mpc]  y[Mpc]  z[Mpc]  v_x[km/s]  v_y[km/s]  v_z[km/s]  M[1e11 M_sol]  R[Mpc]  z
```

### Log Files

Standard simulation log:

```text
Logfile.dat:
1. Time [Gyr]
2. Max_Error [internal units]
3. Step_Size [Gyr]
4. Scale factor
5. Redshift
6. Hubble_Parameter [km/s/Mpc]
7. Deceleration_Parameter
8. Omega_m
```

Glass-making log:

```text
Glass_logfile.dat:
1. Cosmic time [Gyr]
2. Scale factor (a/a_start)
3. Redshift
4. Hubble_Parameter [km/s/Mpc]
5. Deceleration_Parameter
6. Mean(F) [internal units]
7. Max(F) [internal units]
8. Mean(A) [internal units]
9. Max(A) [internal units]
10. Mean(disp) [Mpc]
11. Max(disp) [Mpc]
12. Mean(velocity) [km/s]
13. Max(velocity) [km/s]
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Missing parameter file!` | Pass a parameter file as the first executable argument. |
| `output directory does not exist` | Create the directory named by `OUT_DIR` before running. |
| HDF5 output falls back or fails | Build with `OPT += -DHAVE_HDF5` and verify `HDF5_INC`/`HDF5_LIBS`. |
| CUDA build cannot find `nvcc` | Set `CUDA_PATH` to the CUDA Toolkit installation. |
| CUDA run exits with Barnes-Hut error | Rebuild without `-DUSE_BH`, or use the CPU executable. |
| Boundary-condition startup error | Rebuild with the compile-time topology matching `IS_PERIODIC`. |
| Hybrid MPI/OpenMP run is unexpectedly slow | Try `mpirun --bind-to none ...` and verify thread/process placement. |

## Citation

If you publish academic work using StePS or StePS-generated data, please cite the relevant papers listed in the repository-level [`README.md`](../README.md#steps---stereographically-projected-cosmological-simulations).

If you use the separate initial condition generator, please also cite the `stepsic` paper referenced by the root README.

## Acknowledgement

The development of this code has been supported by the Department of Physics of Complex Systems, ELTE. GR thanks the Department of Physics & Astronomy, Johns Hopkins University, for supporting this work. GR acknowledges sponsorship of a NASA Postdoctoral Program Fellowship and support from JPL, which is run under contract by California Institute of Technology for NASA. The developer acknowledges support from NSF award 1616974. GR acknowledges the support of the Research Council of Finland grant 354905 and the European Research Council via ERC Consolidator grant KETJU, no. 818930.
