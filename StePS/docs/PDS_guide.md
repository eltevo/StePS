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

## Running on GPUs (CUDA)

Since v2.2.1.0 the PDS CUDA path is validated on real hardware (4× NVIDIA
H200): multi-GPU snapshots are bit-identical to single-GPU, and agree with the
CPU build to RMS ≈ 10⁻¹⁰ Mpc with identical adaptive-timestep output redshifts.

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

## Known Limitations

### Approximate time integrator

The current KDK leapfrog in `step.cc` operates on the 3D `x[3*i]` and `v[3*i]` arrays using
flat Euclidean arithmetic.  The PDS position update uses the quaternion array `PDS_Q` only
for the force calculation and boundary wrapping; the drift phase (`x += v * dt`) is Euclidean.

This is a first-order approximation valid when:
- The simulation volume ≪ R_curv (curvature corrections ∝ r²/R²)
- Time steps are small (so drift trajectories remain close to geodesics)

For Ω_k ≈ −0.018 and simulation radius r_in ≈ 960 Mpc, the curvature correction is:
(r_in/R)² ≈ (960/3100)² ≈ 10%, which is not negligible for precision science.

**Future work:** Replace the drift phase with the geodesic exponential map on S³:
```
q(t + dt) = cos(|v|*dt/R) * q(t) + sin(|v|*dt/R) * v_hat
```
where v_hat is the unit tangent velocity vector (4D, perpendicular to q).

### Force direction projection

The 4D geodesic tangent force is projected to 3D by discarding the e₀ component and using
(t₁, t₂, t₃).  The error is ∝ sin(|q₁:₄|) and is small for particles near the domain centre
but can reach ~30% for particles at the domain boundary.

### Flat-spectrum initial conditions (Phase 7A approximation)

The native stepsic PDS IC (see Quick Start) clips the particle load to the
fundamental domain and weights masses with the conformal factor, but the LPT
displacement field is still generated from the flat-space P(k) on a
box-periodic FFT mesh.  The mode statistics of S³/I* (discrete spectrum
k_n = √(n(n+2))/R_curv restricted to I*-invariant modes) are Phase 7B work.

### CUDA kernel: no BH tree

The `ForceKernel_pds` CUDA kernel uses direct O(N²) summation only (the exact
compensated 120-image sum, same as the CPU path).  There is no PDS Barnes-Hut
tree implementation.  Since v2.2.1.0 the CUDA build is compile-tested and
physics-validated on GPU hardware (see [Running on GPUs](#running-on-gpus-cuda)):
multi-GPU results are bit-identical to single-GPU and agree with the CPU build
to floating-point round-off.  Direct summation on modern GPUs is fast — a
12 240-particle test7b run to z = 0 takes ≈ 7.5 minutes on 4× H200 — so for
large N prefer the CUDA build over the CPU one.

---

## File Reference

| File | Changes |
|---|---|
| `src/pds_group.h` | I* group (BFS generation), quaternion algebra, geodesic distance, **bare and background-compensated force kernels**, fundamental domain test, boundary wrapping, velocity Jacobian, image enumeration |
| `src/global_variables.h` | `PDS_Q`, `PDS_R_CURV` under `#elif defined(POINCARE_DODECAHEDRAL)`; `pds_wrap_ic()` prototype |
| `src/main.cc` | Global variable definitions; PDS forward declarations; **IC wrap + broadcast before the initial force calculation**; PDS-volume density check; topology warning |
| `src/step.cc` | `pds_wrap_ic()` (IC wrapping); PDS boundary wrapping after drift with velocity transform; force dispatch to `forces_pds()`; MPI broadcast of `PDS_Q` |
| `src/forces.cc` | `forces_pds()`: O(N²) CPU force — **exact compensated 120-image summation** (IS_PERIODIC ≥ 2, incl. self-images) or nearest-image mode (IS_PERIODIC = 1) |
| `src/forces_cuda.cu` | `ForceKernel_pds` CUDA kernel with I* in `__constant__` memory (uploaded per device inside the per-GPU OpenMP section — constant memory is not shared between GPUs), same two force modes |
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
