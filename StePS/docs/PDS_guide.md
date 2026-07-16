# Poincaré Dodecahedral Space (PDS) Topology in StePS

## Overview

StePS v2.1 adds experimental support for **Poincaré Dodecahedral Space** (PDS) as a fourth
topology option alongside R³, T³, and S¹×R².  PDS is the quotient manifold S³/I* — a compact,
positively curved 3-manifold constructed by identifying opposite faces of a spherical dodecahedron
with a π/5 (36°) twist.  The isometry group I* is the **binary icosahedral group**, which has
120 elements.

PDS has historically been proposed as a candidate topology for the observable Universe (Luminet
et al. 2003, *Nature* 425, 593), where a total density Ω_tot ≈ 1.018 implies positive curvature
and a characteristic cell size of about 3.1 Gpc.  Running N-body simulations in PDS topology
allows direct tests of this hypothesis by computing the clustering statistics in the correct
background geometry.

> **Status:** EXPERIMENTAL.  The code compiles and runs, but the time integrator is still
> approximate (see [Known Limitations](#known-limitations)).  Do not use PDS results for
> publication without first running the suggested validation tests.

---

## Physics

### Geometry of S³/I*

Particles live on the 3-sphere S³ ⊂ ℝ⁴.  Their positions are stored as **unit quaternions**
q = (q₀, q₁, q₂, q₃) in the separate array `PDS_Q[4*N]`.

The binary icosahedral group I* has 120 elements — generated at runtime by BFS closure from two
known generators in `src/pds_group.h`.  Each particle in the **fundamental domain** of S³/I* has
119 image copies at other locations on S³, making I* the group of periodic images analogous to
lattice translations in T³.

The **fundamental domain** is the Voronoi cell of the identity element e₀ = (1,0,0,0):
```
q ∈ domain  ⟺  q₀ = max_{g ∈ I*} (q · g)
```
Boundary wrapping is equivalent to finding the nearest domain centre and left-multiplying by its
conjugate.

### Force Law

The gravitational force on S³ uses the **background-compensated kernel** (per unit
source mass, curvature radius R):

```
|F| = G M [1 − V(χ)/V_S³] / (R² sin²χ),     V(χ)/V_S³ = (2χ − sin 2χ) / (2π)
```

i.e. a point mass plus a uniform *negative* background of equal total mass.  On a
compact space a bare point mass is inconsistent (its flux has nowhere to end), and
in comoving cosmological simulations the homogeneous mean density is already part
of the Friedmann expansion — peculiar forces must be sourced by fluctuations only.
This is the exact analogue of dropping the k = 0 mode in T³ Ewald summation.

The compensated kernel reduces to the Newtonian G M/r² as χ → 0 (r = Rχ) and goes
smoothly to zero at the antipode.  The direction is the unit geodesic tangent at
the field particle pointing toward the source.  The geodesic distance is
`χ = arccos(p · q)` where p and q are unit quaternions.

> **Conformal Jacobian (the force→drift mapping).** The kernel above is the
> *physical geodesic* acceleration, but the integrator drifts particles in the
> flat stereographic chart (`x_stereo += v·dt`, see Known Limitations). The
> stereographic map is conformal, `ds² = Ω²·dx_stereo²`, with
> `Ω = 2R²/(R² + r²)`, so the correct **coordinate** acceleration is `a_phys/Ω`.
> Each PDS force is therefore divided by `Ω(r_i)` for its target particle `i`,
> using the identity **`Ω = 1 + q₀`** (q₀ = quaternion scalar part of `i`).
> Omitting this made gravity ~Ω ≈ 2× too strong throughout the domain and caused
> a runaway over-growth (`D ∝ a^1.6` instead of `a`); fixed in v2.2.3.0. Ω varies
> only ~3.5% across the domain (2.00 at the centre → 1.93 at r_in = 960 Mpc), so
> the per-particle factor captures the effect to a few percent.

> **Why not the bare 1/(R² sin²χ) kernel?**  I* is closed under negation, the bare
> kernel satisfies G(π−χ) = G(χ), and antipodal images pull in exactly opposite
> directions — so the bare force from the full 120-image system of any source
> **cancels identically**.  (StePS ≤ v2.1.1.0 used the bare kernel with a 1D
> "Ewald correction" table; that table was exactly −G(χ_nearest), and Ewald-mode
> runs had essentially zero gravity.  Fixed in v2.2.0.0.)

### Exact Image Summation

S³ is compact: every source has exactly 119 non-trivial I* images, so — unlike the
infinite lattice sums of T³ — the image sum is **finite and exact**.  For
IS_PERIODIC ≥ 2 StePS sums the compensated force over **all 120 images of every
source**, including the particle's own self-images.  No Ewald table is involved.

Two structural facts worth knowing:

- **Homogeneity / zero self-force:** the self-image distances
  χ(q, g·q) = arccos(Re g) do not depend on q, and each image shell is symmetric
  around the particle, so the self-image force vanishes identically everywhere
  (validated by Test 4 of the test suite).
- **Anisotropy:** the 119-image correction is *not* isotropic — its radial part
  varies with direction by up to 0.45·F_nearest and it has a transverse component
  up to 0.19·F_nearest near the domain boundary (see
  `data/pds_anisotropy/REPORT.md`).  This is why no 1D table can replace the
  exact sum; the directional residual scales as (r/R)⁵ near the centre, matching
  Roukema & Różański (2009).

For IS_PERIODIC = 1 (nearest-image mode, fast tests / glass making) only the
single closest I* image of each source contributes, with the same compensated
kernel.

---

## Building StePS with PDS Support

Enable PDS by uncommenting one line in your Makefile:

```makefile
OPT += -DPOINCARE_DODECAHEDRAL  # Poincare Dodecahedral Space (S^3/I*). EXPERIMENTAL.
```

No other Makefile changes are needed.  PDS is mutually exclusive with `-DPERIODIC` and
`-DPERIODIC_Z` — compile a separate binary for PDS runs.

### Recommended Makefile settings

```makefile
OPT += -DPOINCARE_DODECAHEDRAL
OPT += -DHAVE_HDF5
OPT += -DCOSMOPARAM=0          # standard ΛCDM
EWALD_INTERPOLATION_ORDER=2    # not used in PDS, safe to leave at default
```

---

## Parameter File

PDS simulations use the same `.param` format as other StePS modes.  The key parameters are:

| Parameter | Meaning | Typical value |
|---|---|---|
| `IS_PERIODIC` | 1 = nearest-image only; ≥ 2 = exact 120-image summation | 2 |
| `COSMOLOGY` | 1 for cosmological integration | 1 |
| `Omega_m` | Matter density | 0.3111 |
| `Omega_lambda` | Cosmological constant | 0.6889 |
| `Omega_r` | Radiation density | 0.0 |
| `HubbleConstant` | H₀ in km s⁻¹ Mpc⁻¹ | 67.66 |
| `PDS_R_CURV` | Curvature radius of S³ in Mpc | 3100.0 |
| `L_BOX` | Not used in PDS mode; set equal to 2×PDS_R_CURV | 6200.0 |
| `R_SIM` | Simulation radius (= inscribed radius of dodecahedron ≈ 0.31 R_curv) | 960.0 |
| `PARTICLE_RADII` | Plummer softening length in Mpc | (from IC generator) |
| `ACC_PARAM` | Relative force accuracy parameter | 0.005 |
| `STEP_MIN` | Minimum time step | 0.00002 |
| `STEP_MAX` | Maximum time step | 0.03125 |

### Setting PDS_R_CURV

The curvature radius R is set from the cosmological constraint that the volume of S³/I* equals
the observed universe volume.  For the Luminet et al. (2003) best-fit:

```
Ω_total ≈ 1.018  →  Ω_k = −0.018  →  R ≈ 3100 Mpc
```

More generally: R = c / (H₀ √(Ω_k)) (for negative Ω_k = 1 − Ω_total).

The **inscribed radius** of the dodecahedral fundamental domain (the inradius) is:
```
r_in ≈ R · arccos(1/√5) ≈ 0.3094 R
```
Set `R_SIM` to this value so that all particles inside the fundamental domain are covered.

---

## Quick Start Example

### 1 · Compile

```bash
# cp Template-LinuxGCC-Makefile Makefile
# Edit Makefile: uncomment -DPOINCARE_DODECAHEDRAL, set library paths
# make
# or create PDS-LinuxGCC-Makefile and
make -f PDS-LinuxGCC-Makefile
```

### 2 · Prepare a parameter file - done

Save the following as `examples/PDS_test.param`:

```
# Poincare Dodecahedral Space test simulation
# LCDM cosmology, Luminet et al. (2003) best-fit curvature

Cosmological parameters:
------------------------
Omega_b         0.0
Omega_lambda    0.6889
Omega_m         0.3111
Omega_r         0.0
HubbleConstant  67.66
a_start         0.03125
a_max           1.0

Simulation parameters:
-----------------------
COSMOLOGY               1
IS_PERIODIC             2
COMOVING_INTEGRATION    1
PDS_R_CURV              3100.0
L_BOX                   6200.0
R_SIM                   960.0
IC_FILE                 ./examples/ic/pds_test_n1000_cubical_Lx1919_Ly1919_Lz1919_Ng10_Nm64_z31_LPT1_cic/ic.hdf5
IC_FORMAT               2
OUT_DIR                 ./examples/PDS_test_output/
OUT_LST                 ./examples/outredshifts.txt
OUTPUT_TIME_VARIABLE    1
OUTPUT_FORMAT           2
REDSHIFT_CONE           0
MIN_REDSHIFT            0.0003012504
ACC_PARAM               0.005
STEP_MIN                0.00002
STEP_MAX                0.03125
PARTICLE_RADII          10.0
FIRST_T_OUT             1.0
H_OUT                   1.0
```

### 3 · Generate initial conditions

[stepsic](https://github.com/eltevo/stepsic) has a **native PDS geometry**
(`GEOMETRY = "pds"`, since 2026-06-10): particles are placed on a regular
stereographic Cartesian grid clipped to the dodecahedral fundamental domain,
particle masses carry the conformal volume factor Ω(r)³ = (2R²/(R²+r²))³ so the
comoving density on S³ is exactly uniform, flat-space LPT displacements are
applied (Phase 7A approximation, valid for R_domain/R_curv ≈ 0.3), particles
displaced out of the domain are wrapped back with the exact velocity Jacobian,
and the IC file contains a `PartType1/Quaternions` (N, 4) dataset that StePS
uses directly (skipping the projection + wrap at load time).

Run from the stepsic repository root:

```bash
conda activate stepsic
cd STEPSIC_ROOT
python stepsic.py STEPS_ROOT/examples/PDS_test_ic.toml
```

The IC file will be written to:
```
/v/csabai/GitHub/steps_dodeca/data/ic/pds_test_pds_Rcurv3100_L1200_Ng16_Nm64_z31_LPT1_cic/ic.hdf5
```
This path is already set in `PDS_test.param` as `IC_FILE`.

Key settings in [examples/PDS_test_ic.toml](../examples/PDS_test_ic.toml):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `GEOMETRY` | `"pds"` | Native PDS: domain-clipped grid + quaternions |
| `PDS_R_CURV` | `3100.0` | Curvature radius of S³ [Mpc] |
| `LBOX` | `[1200, 1200, 1200]` | LPT mesh box; must enclose the domain (≥ 2·R·tan 10.7° ≈ 1172 Mpc) |
| `PERIODIC` | `[0, 0, 0]` | StePS handles the S³/I* identifications |
| `TYPE` | `"grid"` | Regular lattice clipped to the domain |
| `NGRID` | `16` | 16³ grid → 1552 particles inside the domain (~38%) |
| `REDSHIFT` | `31` | Matches `a_start = 0.03125` |
| `LPTORDER` | `1` | Zel'dovich (use 2 for production) |
| `COSMOLOGY` | `"Planck2018EE+BAO"` | Matches `PDS_test.param` ("best" values) |
| `HINDEPENDENT` | `false` | Distances in Mpc, not Mpc/h |

For legacy flat-approximation ICs (`GEOMETRY = "cubical"`), StePS still converts
Cartesian coordinates to quaternions at load time whenever
`PartType1/Quaternions` is absent — but the cubical box tiles into 120 copies
under wrapping and destroys the LPT correlations, so the native pds geometry
should always be preferred.

### 4 · Run

The example writes its output under the workspace data folder
(`/v/csabai/GitHub/steps_dodeca/data/pds_run_7a/`); large outputs do not belong
in the git repositories.

```bash
mkdir -p /v/csabai/GitHub/steps_dodeca/data/pds_run_7a
mpirun -np 4 ./build/StePS examples/PDS_test.param
```

Or on a GPU node (much faster; the trailing argument is the number of GPUs —
see [Running on GPUs](#running-on-gpus-cuda)):

```bash
mpirun -np 1 ./build/StePS_CUDA examples/PDS_test.param 4
```

At startup, StePS will print:

```
PDS (S^3/I*) exact 120-image force summation is on (background-compensated kernel).
PDS curvature radius		3100.000000 Mpc
...
	Reading /PartType1/Quaternions
...
The particle masses are consistent with the cosmological parameters set in the parameter file:
rho_part/rho_cosm - 1 = -6.9e-08
PDS: Binary icosahedral group I* generated (120 elements).
```

There is no Ewald table any more — nothing is cached between runs, and old
`PDS_Ewald_table.hdf5` files are ignored.

### 5 · Read output snapshots

Output snapshots are standard StePS HDF5 files.  PDS-specific header fields:

```python
import h5py

with h5py.File("snapshot_0001.hdf5", "r") as f:
    topology = f["/Header"].attrs["TopologicalManifold"]  # "S^3/I*"
    R_curv   = f["/Header"].attrs["R_curvature_Mpc"]
    Omega_k  = f["/Header"].attrs["Omega_k"]
    order    = f["/Header"].attrs["PDS_I_star_order"]     # 120
    quats    = f["PartType1/Quaternions"][:]               # shape (N, 4)
    print(f"Topology: {topology},  R = {R_curv:.1f} Mpc,  Ω_k = {Omega_k:.4f}")
```

---

## Barnes-Hut tree force (experimental)

Direct 120-image summation is O(N²) and becomes prohibitive at research-grade
resolution (N ≳ 10⁵). Since v2.2.2.0 a **Barnes-Hut octree** force is available
for PDS on both **CPU and GPU**, scaling as O(N log N).

```bash
conda activate stepsic

# CPU build:
make -f PDS-Linux_BH-Makefile          # builds build/StePS_BH (theta = 0.3)
# IMPORTANT: --bind-to none, or all OpenMP threads pile onto one core (~18x slower)
OMP_NUM_THREADS=16 mpirun --bind-to none -x OMP_NUM_THREADS -np 1 \
    ./build/StePS_BH <paramfile>

# GPU build (much faster; <n_gpu> is the trailing argument as for StePS_CUDA):
make -f PDS-Linux_CUDA_BH-Makefile     # builds build/StePS_CUDA_BH (theta = 0.3)
# Use --bind-to none here too: the octree build is parallel on the host CPU, so
# binding to one core slows it down (the GPU kernels are unaffected).
OMP_NUM_THREADS=32 mpirun --bind-to none -x OMP_NUM_THREADS -np 1 \
    ./build/StePS_CUDA_BH <paramfile> 4
```

> **Build dir note:** all PDS makefiles share `build/` with the same object
> names but different macros (`USE_BH`, `USE_CUDA`). Run `rm -f build/*.o` (or
> `make -f <makefile> clean`) when switching between StePS / StePS_BH /
> StePS_CUDA / StePS_CUDA_BH.

On the GPU the octree is built on the host with a **Morton (Z-order) sort**
(parallel: `__gnu_parallel::sort` + OpenMP) into a flattened DFS-preorder array
with a per-node "escape"/skip pointer, then walked **iteratively** on the device
— no recursion, no per-thread stack. Each GPU thread handles one field particle
and does the 120 per-image stackless walks. Device buffers are persistent
(allocated once, reused across steps) and the tree is staged through pinned host
memory, so there is no per-step `cudaMalloc`/`cudaFree`. The field-particle range
is split across GPUs (one OpenMP thread per device), exactly like the exact
kernel.

The per-step log line reports the split, e.g.
`[host tree build 0.06s (…%), GPU section …]`, so you can see how much time is
host-side tree construction vs GPU force evaluation. At N ≈ 7.8×10⁵ the parallel
Morton build is ~0.06 s (down from ~0.30 s for the original recursive build).

The opening angle θ is set by the macro value (`-DUSE_BH=0.3` in the Makefile).
θ ≈ 0.3–0.35 gives ~0.5–1 % force accuracy; smaller θ is more accurate and
slower. The implementation (`forces_pds_bh` in `src/forces.cc`) evaluates the
opening test **separately for each of the 120 I\* images** in the geodesic
metric — a node far in the identity image can be the physical neighbour across a
shared dodecahedral face in another image, so a single opening test is wrong.
See `examples/pds_tests/pds_bh_prototype.cc` for the standalone θ-vs-accuracy
validator and `pds_bh_prototype_README.md` for the design notes.

**Validation (test7b, N = 12 240, θ = 0.3):** the end-to-end z = 31 → 0 run
matches the exact 120-image run with density-field cross-correlation 0.93 (z = 0)
and power-spectrum agreement to ~5 % across all scales, in 78 s on 16 CPU cores
(vs ~450 s for the exact run on 4× H200). Individual particle trajectories
diverge over time (chaotic N-body), as expected; the *statistics* match.

**GPU port validation (test7b):** `StePS_CUDA_BH` reproduces the CPU `StePS_BH`
run to printed precision at every snapshot through z = 0 (RMS Δx = 0), 4-GPU and
1-GPU runs are bit-identical, and GPU-BH vs exact gives the same density
cross-correlation (0.935 at z = 0). Force eval ≈ 0.025 s on one GPU (vs 0.25 s on
16 CPU cores); the whole test7b run finishes in < 60 s on a single GPU. At
N = 7.8×10⁵ the GPU tree force is ≈ 0.35 s/eval on 4 GPUs (~9.4×10⁵ nodes) versus
an extrapolated ~46 min/eval for exact direct summation on the same GPUs (~10⁴×).

**Momentum-conservation audit:** Barnes-Hut's monopole approximation breaks
Newton's third law in general, so the net force S = Σ_i M_i a_i was checked
against the exact force. The exact PDS force is itself not perfectly
pairwise-antisymmetric (the compact-S³ force projected to 3D gives an inherent
|S|/Σ|M a| ≈ 0.17 % early, rising to ≈ 0.66 % at z = 0). The tree force stays
within **0.91–1.04× of that exact baseline at every epoch** and converges to
exactly 1.0× as θ → 0 — i.e. it adds no momentum drift of its own. The
integrated run confirms this: the bulk momentum |P|/(M v_rms) of the θ = 0.3 run
tracks the exact run (~0.2–0.5 %) and *decreases* from the IC value rather than
accumulating. Reproduce with
`./pds_bh_prototype <snapshot> <particle_radii> 0 momentum`.

> **Status:** EXPERIMENTAL (CPU and GPU), monopole-only. Accuracy (~5 % in P(k)
> at θ = 0.3) and momentum conservation are validated, and the GPU port
> reproduces the CPU run to printed precision; use the exact direct summation
> (`IS_PERIODIC ≥ 2`, no `USE_BH`) for the reference runs these are validated
> against.

---

## Running on GPUs (CUDA)

Since v2.2.1.0 the PDS CUDA path is validated on real hardware (4× NVIDIA
H200): multi-GPU snapshots are bit-identical to single-GPU, and agree with the
CPU build to RMS ≈ 10⁻¹⁰ Mpc with identical adaptive-timestep output redshifts.

> **Important (fixed in v2.2.4.0).** That validation was done below ~4.3×10⁶ particles
> (on 4 GPUs).  Above that, **all** the CUDA force kernels had a coverage bug: the
> grid-stride loop was bounded by the thread count `nthreads = 32·SMs·BLOCKSIZE` instead
> of the per-GPU particle count `N_GPU`, so the last `N_GPU − nthreads` particles of each
> GPU never received a force and stayed frozen at their IC positions (visible as smooth,
> unevolved bands in z=0 renders).  Fixed in v2.2.4.0 (the kernels now use `N_GPU` as the
> loop bound).  Runs at or below the threshold (e.g. the test7b validation, the
> 7.8×10⁵-particle PDS runs, the v2.2.3.0 Gadget4 comparison) were fully covered and are
> unaffected; larger multi-GPU runs made before v2.2.4.0 should be re-run.

### Build

`PDS-Linux_CUDA-Makefile` builds against the **stepsic conda environment**
(CUDA toolkit, OpenMPI and HDF5 all come from `$CONDA_PREFIX`), so activate it
first:

```bash
conda activate stepsic
make -f PDS-Linux_CUDA-Makefile        # produces build/StePS_CUDA
```

Set `CUDA_ARCH` for your GPU generation (default `sm_90` = Hopper/H100/H200;
use `sm_80` for A100, `sm_86` for consumer Ampere, …).

### Run

`StePS_CUDA` takes an extra trailing argument: the **number of GPUs per MPI
task**.  On a single node always use *one* MPI task and give it all the GPUs:

```bash
mpirun -np 1 ./build/StePS_CUDA <paramfile> 4     # one node, 4 GPUs
```

Inside the task one OpenMP thread drives each GPU (thread i → device i) and
the particle range is block-split between them.  Two pitfalls:

- **Do not start several MPI ranks on one node** — each rank assigns its GPUs
  starting from device 0, so the ranks pile onto the same GPUs.  Multiple MPI
  tasks are for multi-node runs (one task per node, each with the GPUs-per-node
  argument).
- If the trailing argument is omitted, the default is **one GPU per MPI task**.

### Troubleshooting

If `mpirun` itself segfaults instantly with no output (even
`mpirun -np 1 hostname`), the conda-forge hwloc library may be crashing while
serializing the node's hardware topology to XML at PMIx startup (observed with
libhwloc 2.12–2.13 on H200 nodes; the crash is triggered by a PCI/GPU I/O
object).  Workaround — generate an I/O-free topology once and load it via
environment variables:

```bash
lstopo-no-graphics --no-io --of xml $CONDA_PREFIX/etc/hwloc-topology-noio.xml
export HWLOC_XMLFILE=$CONDA_PREFIX/etc/hwloc-topology-noio.xml
export HWLOC_THISSYSTEM=1
```

(Put the two exports in `$CONDA_PREFIX/etc/conda/activate.d/` to make them
automatic; regenerate the XML after hardware or kernel changes.)

---

## Visualizing Snapshots

`../tools/Visualization` contains a Millennium-simulation-style renderer that
works directly on PDS snapshots (added in v2.2.1.0):

- **`millennium_render.py`** — adaptively smoothed logarithmic projected
  density maps (dark background, magma colormap, cosmic-age/redshift info box,
  scale bar).  Pure numpy/scipy/matplotlib, no py-sphviewer needed.  Use as a
  module (`render_snapshot("snapshot_0010.hdf5")`) or CLI
  (`./millennium_render.py snapshot.hdf5 out.png`).
- **`PDS_Millennium_View.ipynb`** — companion notebook: snapshot overview
  table, z = 0 render, XY/XZ/YZ projections, redshift-evolution mosaic, and an
  animated GIF of the run.  Defaults to `data/pds_tests/test7b`; point
  `SNAP_DIR` at any other output directory.

The older scatter-plot explorers (`data/pds_tests/PDS_explorer*.ipynb`) remain
useful for per-particle diagnostics (velocities, S³ angular distributions,
NaN screening).

---

## IS_PERIODIC Levels

| IS_PERIODIC | Mode | Use case |
|---|---|---|
| 1 | Nearest image only (compensated kernel) | Fast tests, glass making |
| ≥ 2 | Exact 120-image summation (compensated kernel) | Science runs |

Since v2.2.0.0 all values ≥ 2 are equivalent: the image sum over the compact
S³ is exact, so there are no Ewald resolution levels (and no cached table).
The exact mode costs ≈ 2.3× a nearest-image force evaluation.

---

## Validation Tests

The automated suite covers most of these checks (8 tests, ~2 min, builds the
needed binary variants itself):

```bash
conda activate stepsic
cd StePS/StePS
python examples/pds_tests/run_tests.py
```

Tests: free flight / Hubble drag, boundary wrapping, two-particle Newtonian
limit, homogeneity (zero self-image force, IS_PERIODIC = 2), multi-particle
stability, Python/C++ exact-force cross-validation (< 10⁻⁶), end-to-end
stepsic-PDS-IC run with growing density contrast, and an R³ regression of the
shared sources.  The force-law study `examples/pds_tests/pds_anisotropy_study.py`
regenerates `data/pds_anisotropy/REPORT.md`.

The manual sanity checks below remain useful for debugging by hand.

### Test 1 — I* group closure

```bash
# A small standalone test compiled from pds_group.h
cat > pds_test.cc << 'EOF'
#include <stdio.h>
#include "src/pds_group.h"
int main() {
    pds_init();
    // Check: e0 at domain centre
    double e0[4] = {1,0,0,0};
    printf("pds_in_domain(e0) = %d  (expected 1)\n", pds_in_domain(e0));
    // Check: all 120 elements are unit quaternions
    int ok = 1;
    for(int g=0; g<120; g++) {
        double n2 = 0;
        for(int k=0;k<4;k++) n2 += PDS_I_STAR[g][k]*PDS_I_STAR[g][k];
        if(fabs(n2-1.0) > 1e-12) { printf("Element %d not unit!\n",g); ok=0; }
    }
    printf("All 120 elements are unit quaternions: %s\n", ok?"PASS":"FAIL");
    // Check: force cancellation at centre (wrap back to origin from a small displacement)
    double q_eps[4] = {0.9999, 0.01, 0.005, 0.002};
    double q_out[4]; pds_normalise(q_eps);
    pds_wrap(q_eps, q_out);
    printf("Wrap of near-e0 point: q0=%.6f  (expected ~1)\n", q_out[0]);
    return 0;
}
EOF
g++ -std=c++11 -O2 -lm pds_test.cc -o pds_test && ./pds_test
```

Expected output:
```
PDS: Binary icosahedral group I* generated (120 elements).
pds_in_domain(e0) = 1  (expected 1)
All 120 elements are unit quaternions: PASS
Wrap of near-e0 point: q0=0.999983  (expected ~1)
```

### Test 2 — Force law flat-space limit

With a two-particle simulation (N=2), place the particles at chi ≪ 1 and verify the force
matches G M / r²:

```python
import numpy as np

R = 3100.0   # Mpc
chi = 0.001  # ≈ r = 3.1 Mpc separation
r   = R * chi

# Gauss's law on S³
F_S3 = 1.0 / (R**2 * np.sin(chi)**2)

# Newtonian
F_flat = 1.0 / r**2

print(f"S³ force:     {F_S3:.8e}")
print(f"Newtonian:    {F_flat:.8e}")
print(f"Relative err: {abs(F_S3/F_flat - 1):.2e}  (expected < chi^2/6 ≈ {chi**2/6:.2e})")
```

### Test 3 — Symmetry cancellation at domain centre

A particle at e₀ = (1,0,0,0) surrounded by a uniform shell of 119 image copies should
experience zero net force.  This is automatic by the I* symmetry, but can be verified by
running a single-particle simulation with IS_PERIODIC ≥ 2 and checking that the printed
force magnitude is ≈ 0.

### Test 4 — Energy conservation

Run a small test simulation (N ≈ 100, IS_PERIODIC = 2) for 10 dynamical times and check
that the total energy (kinetic + potential) is conserved to better than 1% over the run.

### Test 5 — Regression: T³ and R³ unchanged

After the code changes, verify existing T³ and R³ simulations reproduce byte-identical
snapshots compared to a reference run.

---

## Validation against Gadget4 (gold standard)

The end-to-end growth/clustering test compares a PDS run to a **flat Gadget4 run built
from the same IC realization**.  Gadget4 tracks linear growth to ~1% and is the reference;
a **StePS-T³ run is not a safe reference** — `testCubic128` (`IS_PERIODIC=2` with a
low-res 63³ Ewald table) was found to over-grow ~3× vs Gadget at z=10, tracking `D∝a^~1.5`.
The PDS mode is immune (exact 120-image sum, no Ewald table), but this is why the
reference is Gadget4.

Tooling lives in `examples/pds_tests/`:
- `GADGET4_REFERENCE.md` — reproducible Gadget4 build (conda SYSTYPE, FFTW3/zlib, the
  `CPP=mpicxx` gotcha) and run setup.
- `rescale_ic_to_gadget.py` — converts a stepsic h-independent (Mpc) IC to the Mpc/h
  convention Gadget4 enforces (`Hubble=100`): coords/BoxSize ×h, masses ×10h, velocities
  unchanged.
- `validate_growth.py` — compares **median geodesic displacement** `R·χ` (the primary,
  coordinate-invariant growth gate), **large-scale P(k)** in the matched central physical
  region, and counts-in-cells σ² (resolution-confounded — see below).

**Result (v2.2.3.0 conformal fix, test128 vs gadget128_flat):** the PDS displacement
growth matches Gadget to **1–2%** at every output (`PDS/Gadget = 1.00–1.02`, z=15→0).  The
σ² appears ~3–4× high, but this is a **sampling artifact, not over-growth**: it is already
present at the IC (z=30, where there is no real structure — the power is below shot noise),
it is constant through evolution, and it traces to the coarser PDS particle load (see the
matched-resolution note under "Flat-spectrum initial conditions").

---

## Known Limitations

### Approximate time integrator

The current KDK leapfrog in `step.cc` operates on the 3D `x[3*i]` and `v[3*i]` arrays using
flat Euclidean arithmetic.  The PDS position update uses the quaternion array `PDS_Q` only
for the force calculation and boundary wrapping; the drift phase (`x += v * dt`) is Euclidean.

This is a first-order approximation valid when:
- The simulation volume ≪ R_curv (curvature corrections ∝ r²/R²)
- Time steps are small (so drift trajectories remain close to geodesics)

> **Note (v2.2.3.0).** The *leading* conformal effect of the flat drift was **not**
> the ~10% (r/R)² curvature correction below — it was a ~100% (factor Ω ≈ 2) error.
> The physical geodesic force was being applied directly as the stereographic
> coordinate acceleration without the conformal Jacobian, making peculiar gravity
> ~2× too strong (runaway over-growth). That dominant term is now corrected by
> dividing each force by Ω = 1 + q₀ (see [Force Law](#force-law)). What remains
> below is the genuinely second-order residual:

After the Ω correction, the leftover error is the **spatial variation** of Ω across
the domain (~3.5%, since Ω = 2.00 at the centre → 1.93 at r_in = 960 Mpc) plus the
velocity-dependent Christoffel terms of the conformal metric, both ∝ (r/R)².
For Ω_k ≈ −0.018 and simulation radius r_in ≈ 960 Mpc this is
(r_in/R)² ≈ (960/3100)² ≈ 10%, still not negligible for precision science.

**Future work:** Replace the drift phase with the geodesic exponential map on S³:
```
q(t + dt) = cos(|v|*dt/R) * q(t) + sin(|v|*dt/R) * v_hat
```
where v_hat is the unit tangent velocity vector (4D, perpendicular to q).

### Force direction projection

The 4D geodesic tangent force is projected to 3D by discarding the e₀ component and using
(t₁, t₂, t₃).  The error is ∝ sin(|q₁:₄|) and is small for particles near the domain centre
but can reach ~30% for particles at the domain boundary.

### Initial-condition spectrum: flat (7A) and discrete S³/I* (7B)

By default the native stepsic PDS IC clips the particle load to the fundamental
domain, weights masses with the conformal factor, and generates the LPT
displacement from the flat-space P(k) on a box-periodic FFT mesh (the **Phase 7A**
approximation).  The large-scale *growth* is correct (the PDS matches Gadget4 to
1–2%, see "Validation against Gadget4"); the flat spectrum only mis-states the
largest-mode statistics that carry the topology signature.

**Phase 7B — discrete S³/I* eigenmode spectrum (implemented).** stepsic can now add
the true discrete eigenmodes of S³/I* on top of the flat small-scale LPT.  The
Laplacian eigenvalues are k_n = √(n(n+2))/R_curv, and only the I*-invariant modes
survive — the first non-trivial mode is at **n=12** (then 20, 24, 30, 32, 36…), the
famous suppression of the largest fluctuations.  Enable it with `PDS_DISCRETE_NMAX`
in the IC config (0 = off):

```toml
GEOMETRY = "pds"
PDS_R_CURV = 3100.0
PDS_DISCRETE_NMAX = 24      # add I*-invariant modes up to n=24 (12, 20, 24)
```

The flat field is high-passed above k(n_max) and the discrete invariant modes supply
all larger scales (replacing the flat — and PDS-*forbidden* — power there).  Modes are
synthesised with the cosmological P(k_n) (variance validated to ~6% of the analytic
m_n P/(V k_n²)).  Implementation: `stepsic/s3harmonics.py` (eigenmodes, multiplicities,
GRF), `stepsic/s3lpt.py` (S³ Zel'dovich displacement, hybrid splice), tests in
`tests/test_s3harmonics.py`.  Cost grows with n_max (the per-point projection is
O((n+1)²·120)); n_max ≲ 24–30 is the practical range and captures the topology-defining
modes.

> **Choosing `NGRID` — physical vs stereographic resolution.** The particle grid is
> regular in *stereographic* coordinates with spacing `LBOX/NGRID`, but the conformal
> factor Ω ≈ 2 makes the *physical* spacing ≈ `Ω·LBOX/NGRID` — about **2× coarser** than
> the same NGRID in a flat box.  To match a flat reference of physical spacing `Δ`, use
> `NGRID ≈ Ω·LBOX/Δ` (e.g. NGRID=256 for a 1200 Mpc box matches a flat 9.4 Mpc load).
> Under-resolving relative to a flat comparison shows up as an apparent counts-in-cells
> σ² excess that is pure sampling noise (present already at the IC), *not* over-growth —
> matching the resolution closes it (σ² ratio vs Gadget 4.0×→1.3× going 128→256).

### Particle load: grid vs glass

Orthogonal to the *spectrum* (7A/7B) is the choice of the **unperturbed particle load**.
The default is a regular Cartesian grid in stereographic coordinates clipped to the
fundamental domain (`TYPE = "grid"`). That grid is convenient but its rectangular lattice
does **not** respect the S³/I* topology, and it imprints sharp **Bragg peaks** at the
lattice frequency: in the raw IC the peak high-k power is ~3×10⁴ × shot, and ~2.5×10³ × shot
still at z=15. The imprint washes out under non-linear growth by z≈2, but it dominates the
*early-time* small-scale field.

A **glass** load removes this. The recipe (validated as `test256glass` vs the grid run
`test256disc`, identical except the load):

1. **Poisson load** — `stepsic` with `TYPE = "random"` for `GEOMETRY = "pds"` draws `NPART`
   points uniformly on S³ and folds them into the fundamental domain via the I* group
   (`pds.wrap`); equal masses, no preferred directions. Set `LPTORDER = 0` to write the
   unperturbed load.
2. **Reverse-gravity relaxation** — build a glass-making binary
   (`PDS-Linux_CUDA_BH-GlassMaking-Makefile`; adds `-DGLASS_MAKING` → `G = -1`, separate
   `build_glass_bh/`) and relax the Poisson load. **Use an Einstein-de Sitter background**
   (`.param` `Omega_m = 1, Omega_lambda = 0`; the `EdS` cosmology in stepsic) — dark energy
   would freeze the relaxation before it reaches a glass (per G. Racz). **Use the Barnes-Hut
   build** (theta≈0.3): direct summation is O(N²) and impractical at N~6×10⁶ (stalls for
   hours on the first force eval), whereas BH relaxes the same load in ~18 min on 4× H200.
   Start from **Poisson, not a grid** — a grid is already a force equilibrium under repulsive
   gravity and keeps its lattice anisotropy. The S³/I* topology has no domain boundary
   (the wrap makes it periodic-like), so the glass fills the cell cleanly.
3. **Build the IC** — `stepsic` with `TYPE = "glass"`, `INPUT_GLASS = <relaxed glass snapshot>`,
   then the usual Zel'dovich + `PDS_DISCRETE_NMAX` spectrum on top.

```toml
# 1. unperturbed Poisson load (LPTORDER=0); run reverse-gravity glass making on its output
TYPE = "random"
GEOMETRY = "pds"
NPART = 6275736        # match the grid run's in-domain count
LPTORDER = 0
COSMOLOGY = "EdS"      # masses self-consistent with the EdS glass-making run

# 3. final IC from the relaxed glass (same spectrum as the grid run)
TYPE = "glass"
INPUT_GLASS = "/path/to/glass_run/snapshot_<last>.hdf5"
LPTORDER = 1
PDS_DISCRETE_NMAX = 20
```

Quality of the result: the glass is **sub-Poisson** (counts var/mean ≈ 0.28 vs 1.0 for a
Poisson load) with essentially **no Bragg peaks** (peak high-k P/shot ≈ 5 vs the grid's
~3×10⁴). The two production runs agree at z=0 (large-scale P(k) within ~3%) — the glass
mainly cleans the early-time field.

> **Measuring the PDS P(k) — keep the FFT cube inside the domain.** A Cartesian P(k) cube
> that pokes outside the fundamental domain includes hard vacuum in its corners (e.g. a
> half=400 Mpc cube has corners at 400·√3≈693 Mpc, ~7% of its volume empty). FFT-ing that
> bounded shape convolves the true clustering with the *shape's own* power spectrum (the
> survey-window/mask effect), producing a huge, non-growing low-k "pedestal" that is **100%
> geometry** — a uniform, unclustered point set in the same shape reproduces it to a few
> percent, and it is identical for the grid and glass loads. Earlier notebook versions
> subtracted the first snapshot to remove it (which also introduced a spurious dip near
> k≈0.03–0.05 /Mpc at the envelope's k-space falloff). The clean fix is simply to **use a
> cube that fits inside the domain** (`half ≤ ~350` Mpc for R_curv=3100 Mpc; the notebook
> uses `HALF=300`, corner 520 Mpc, verified 100% inside): the window artifact and the dip
> both vanish and the raw PDS P(k) is directly comparable to Gadget with no subtraction (low-k
> ratio ~0.85–0.9×). Note this restricts the accessible k range — the discrete n=12,20 modes
> (k~0.004–0.007 /Mpc) fall *below* the fitting cube's fundamental, so a Cartesian sub-cube
> cannot probe the topology modes; use the intrinsic full-domain spectrum for those.

Implementation: `stepsic/geometry.py` (`create_pds_random_particles`), the `EdS` entry in
`stepsic/config/cosmology.toml`, and the `pds` branch of `CosmoData.rescale_snapshot_size`.

### Force methods on the GPU

Both kernels (and the non-PDS CUDA kernels) had the large-N multi-GPU coverage bug
fixed in **v2.2.4.0** — see "Running on GPUs (CUDA)" above; their loop bound is now the
per-GPU particle count `N_GPU`.

Two PDS force kernels run on the GPU:
- `ForceKernel_pds` — direct O(N²) exact compensated 120-image sum. Validated
  since v2.2.1.0: multi-GPU bit-identical to single-GPU, agrees with the CPU
  build to round-off. A 12 240-particle test7b run to z = 0 takes ≈ 7.5 min on
  4× H200.
- `ForceKernel_pds_bh` — O(N log N) Barnes-Hut tree (v2.2.2.0, `-DUSE_BH`). See
  [Barnes-Hut tree force](#barnes-hut-tree-force-experimental).

Use exact direct CUDA for reference/validation runs; use the GPU Barnes-Hut
build for large production runs. There is still no GPU Barnes-Hut tree for the
non-PDS topologies (R³, S¹×R², T³).

---

## Halo catalogs, anisotropy stacking, and the small-domain experiment

Post-processing analyses of the flagship runs (2026-07). Representative, executable
versions live in `tools/Visualization/` (see its README); full pipelines and raw outputs
in `/scratch/csabai/halo_catalogs{,50}/` and `/scratch/csabai/stack3d/`.

### Halo catalogs (StePS_HF) in the matched frame

To compare Gadget and PDS halo-by-halo, catalogs are built in the **matched frame**:
PDS stereographic coordinates (Mpc, origin-centred) and Gadget `x/h − L/2` (Mpc/h → Mpc,
then the −L/2 shift, verified by FFT phase correlation of the density fields; do NOT use
`np.mod(x+L/2,L)−L/2`, which is a pure relabeling and silently misaligns the frames).
Recipe:

1. cut a cube that fits inside the dodecahedral domain (±300 Mpc for R_curv=3100;
   ±11.5 Mpc for R_curv=129.17) plus a buffer ≥ the finder search radius;
2. **de-conformalize the PDS masses**: matched mass = m_native/Ω(r)³ at the particle's
   *current* position (grid load → uniform, equals Gadget's m_p to 0.06%; glass load →
   quantize to ≤32 levels, StePS_HF scans `np.unique(Masses)` and cannot handle a
   continuous mass distribution);
3. run StePS_HF in PERIODIC mode on the cut box (the fake periodicity only touches the
   discarded buffer). Gotchas: `SEARCH_RADIUS` must exceed the largest R200b (else
   IndexError at line ~192); `ParticleIDs` must be 0..N−1 (used as array indices);
4. post-filter halo centres to the inner cube; add Ω(center), M_phys = M·Ω³, R_phys = R·Ω
   columns for the PDS.

Validation: Gadget/PDS-grid mass functions overlap and **98% of the top-500 grid-IC PDS
halos match a Gadget halo within 5 Mpc** (median offset 2.4 Mpc, mass ratio 1.10).
Open issue: the 1200 Mpc PDS *glass* catalog shows ~1.7× abundance and ~2× matched-pair
masses (mirrors its ~30% high-k P(k) excess); treat its halo masses with caution.

### Anisotropy stacking (octahedral-group method of Racz+2021)

Stacking particle cutouts around massive halos in fixed simulation axes, folding over the
48 operations of the octahedral group, and binning directions in the fundamental triangle
(face [001] / edge [011] / corner [111]) against a random-rotation control (see
`tools/Visualization/Halo_stacking_anisotropy.ipynb`):

- **Grid-IC lattice memory**: blatant at z=15 (cross-hatch; lattice-phase modulation
  >2.2×), erased near halos by z≈2 — but the octahedral fold reveals a **+7–9%
  box-axis (face) density excess re-emerging at z=0** in both grid-IC runs (T³ and PDS
  alike ⇒ a grid-IC artifact, not topology): collapse axes stay grid-aligned.
- **Glass loads**: no lattice at any epoch; only a ~3% residual cubic imprint
  (suspect: the cubic FFT mesh + CIC interpolation shared by all ICs).
- **PDS wraparound rule**: any cutout analysis needs **I\* image augmentation**
  (all 120 images q → g⊗q) whenever |centre| + R√3 exceeds the face inradius
  0.1584·R_curv — particles are stored only in the fundamental domain, and un-wrapped
  cutouts contain artificial geometry-locked vacuum.

### Small-domain (50 Mpc) experiment

Two glass-IC runs at L=50 Mpc: Gadget4 T³ vs PDS with R_curv = 3100/24 = 129.17 Mpc —
a deliberately **topology-dominated** box (k₁₂ = 0.100, k₂₀ = 0.162 /Mpc vs
k_box = 0.126). Ingredients: the flat periodic glass from a 64³ tile relaxed with the new
`T3-Linux_CUDA-GlassMaking-Makefile` (PERIODIC + GLASS_MAKING; Ewald interpolation order 2
only; single GPU) tiled 4³; the PDS glass re-charted from the flagship relaxed glass by
exact scale invariance (stereo coordinates × 1/24 — an S³/I* glass is a point set on the
sphere, so it rescales with R_curv for free). Findings
(`tools/Visualization/Gadget_vs_PDS_50Mpc_comparison.ipynb`):

- the runs share the IC fine structure (corr ≈ 0.9 at z=30 in the matched frame) and
  **physically decorrelate by z=0** — the different box-scale (topology) modes drive
  divergent non-linear growth; halo cross-matching is impossible *by design*;
- PDS50 develops ~1.6× more small-scale power by z=0 (topology-fed collapse) and, after
  the I* wraparound correction, a strong topology-locked anisotropy: face/2-fold-axis cone
  density 0.72× the isotropic control — halo environments align with the icosahedral
  eigenmode pattern. The flat T³ 50 Mpc run shows only few-% direction effects at
  r/L ≲ 0.1 (the torus force anisotropy of Racz+2021 lives at larger r/L);
- the glass **tiling** leaves only a ~1.5× median excess at the tile reciprocal vectors at
  z=15, gone by z=0 (three orders of magnitude weaker than grid Bragg peaks).

## File Reference

| File | Changes |
|---|---|
| `src/pds_group.h` | I* group (BFS generation), quaternion algebra, geodesic distance, **bare and background-compensated force kernels**, fundamental domain test, boundary wrapping, velocity Jacobian, image enumeration |
| `src/global_variables.h` | `PDS_Q`, `PDS_R_CURV` under `#elif defined(POINCARE_DODECAHEDRAL)`; `pds_wrap_ic()` prototype |
| `src/main.cc` | Global variable definitions; PDS forward declarations; **IC wrap + broadcast before the initial force calculation**; PDS-volume density check; topology warning |
| `src/step.cc` | `pds_wrap_ic()` (IC wrapping); PDS boundary wrapping after drift with velocity transform; force dispatch to `forces_pds()`; MPI broadcast of `PDS_Q` |
| `src/forces.cc` | `forces_pds()`: O(N²) CPU force — **exact compensated 120-image summation** (IS_PERIODIC ≥ 2, incl. self-images) or nearest-image mode (IS_PERIODIC = 1). `forces_pds_bh()`: O(N log N) Barnes-Hut tree force with per-image geodesic opening test (`-DUSE_BH`, experimental). Both **divide the stored force by the conformal Jacobian Ω = 1 + q₀** (v2.2.3.0) |
| `PDS-Linux_BH-Makefile` | CPU build with the PDS Barnes-Hut force (`build/StePS_BH`); stepsic conda toolchain |
| `examples/pds_tests/pds_bh_prototype.cc` | Standalone Barnes-Hut θ-vs-accuracy validator (+ `pds_bh_prototype_plot.py`, `pds_bh_prototype_README.md`) |
| `src/forces_cuda.cu` | `ForceKernel_pds` CUDA kernel with I* in `__constant__` memory (uploaded per device inside the per-GPU OpenMP section — constant memory is not shared between GPUs), same two force modes. `ForceKernel_pds_bh` / `forces_pds_bh_cuda`: GPU Barnes-Hut tree force (host-built flattened octree with escape pointers, stackless per-image device traversal; `-DUSE_BH`). Both kernels apply the same **Ω = 1 + q₀ conformal division** (v2.2.3.0) |
| `PDS-Linux_CUDA_BH-Makefile` | GPU build with the PDS Barnes-Hut force (`build/StePS_CUDA_BH`) |
| `src/inputoutput.cc` | Reads `/PartType1/Quaternions` when present; PDS fields in HDF5 snapshot headers |
| All Makefiles | `#OPT += -DPOINCARE_DODECAHEDRAL` (commented out by default); `PDS-LinuxGCC-Makefile` (CPU) and `PDS-Linux_CUDA-Makefile` (GPU, stepsic conda-env toolchain, `CUDA_ARCH` knob) have it enabled |
| `../tools/Visualization/millennium_render.py` | Millennium-style adaptive density renderer (module + CLI), works on PDS snapshots; companion notebook `PDS_Millennium_View.ipynb` |
| `../stepsic/stepsic/pds.py` | NumPy port of the PDS primitives (reference implementation, IC generation, force cross-validation) |

---

## Background Reading

- Luminet, J.-P. et al. (2003) "Dodecahedral space topology as an explanation for weak wide-angle temperature correlations in the cosmic microwave background", *Nature* 425, 593–595.
- Weeks, J. R. (2001) "The Poincaré Dodecahedral Space and the Mystery of the Missing Fluctuations", *Notices of the AMS* 51, 610–619.
- Gomero, G. I. et al. (2016) "Simulating Cosmic Microwave Background maps in multiconnected spaces", *Phys. Rev. D* 94, 043501.
- Lehoucq, R. et al. (2002) "Eigenmodes of three-dimensional spherical spaces and their application to cosmology", *Class. Quantum Grav.* 19, 4683.
