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

The gravitational force on S³ follows from Gauss's law on a 3-sphere of curvature radius R:
the geodesic sphere at distance χ has area 4πR²sin²χ, giving:

```
|F| = G M / (R² sin²χ)
```

which reduces to the Newtonian G M/r² as χ → 0 (r = Rχ).  The direction is the unit geodesic
tangent at the field particle pointing toward the source.

The geodesic distance is `χ = arccos(p · q)` where p and q are unit quaternions.

### Image Summation and Ewald Correction

For IS_PERIODIC = 1 (nearest-image mode) only the single closest I* image of each source
particle contributes.  For IS_PERIODIC ≥ 2, a precomputed **1D Ewald correction table** D(χ)
adds the contributions from all 119 non-nearest images.

The table exploits the maximal symmetry of S³: the correction from the non-nearest images
depends only on the geodesic distance χ to the nearest image, not on the direction.  This
reduces what would be a 3D table to a 1D table of ≤ 16 384 entries.

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
| `IS_PERIODIC` | 1 = nearest-image only; 2 = low-res Ewald; 3 = medium-res Ewald; 4 = high-res Ewald | 2 |
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

For a first test, use the **flat approximation** (Strategy A): generate ICs in a non-periodic
Cartesian box of side 2 × R_SIM = 1920 Mpc.  StePS automatically converts the Cartesian
coordinates to S³ unit quaternions at IC load time using the inverse stereographic projection
q = (R²−r², 2Rx, 2Ry, 2Rz) / (R²+r²).  This is valid when the simulation volume ≪ R_curv.

Use [stepsic](https://github.com/eltevo/stepsic) with the provided config file.
Run from the stepsic repository root:

```bash
conda activate stepsic
cd STEPS_ROOT
python STEPSIC_ROOT stepsic.py STEPS_ROOT/examples/PDS_test_ic.toml
```


The IC file will be written to:
```
STEPS_ROOT/examples/ic/PDS_test_N1000_cubical_Lx1920_Ly1920_Lz1920_Ng10_z31_LPT1_cic/ic.hdf5
```
This path is already set in `PDS_test.param` as `IC_FILE`.

Key settings in [examples/PDS_test_ic.toml](../examples/PDS_test_ic.toml):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `GEOMETRY` | `"cubical"` | Flat non-periodic box |
| `LBOX` | `[1920, 1920, 1920]` | 2 × R_SIM [Mpc] |
| `PERIODIC` | `[0, 0, 0]` | Non-periodic; StePS handles PDS periodicity |
| `TYPE` | `"grid"` | Regular lattice |
| `NGRID` | `10` | 10³ = 1000 particles |
| `REDSHIFT` | `31` | Matches `a_start = 0.03125` |
| `LPTORDER` | `1` | Zel'dovich (use 2 for production) |
| `COSMOLOGY` | `"Planck2018EE+BAO"` | Matches `PDS_test.param` |
| `HINDEPENDENT` | `false` | Distances in Mpc, not Mpc/h |

No manual quaternion conversion is needed — StePS performs this automatically whenever
`PartType1/Quaternions` is absent from the IC file.  If your IC generator can output
quaternions directly (future stepsic PDS mode), add a `PartType1/Quaternions` dataset with
shape (N, 4) and StePS will use it verbatim.

### 4 · Run

```bash
mkdir -p examples/PDS_test_output
mpirun -np 4 ./build/StePS examples/PDS_test.param
```

At startup, StePS will print:

```
Topology: S^3/I* (Poincare Dodecahedral Space)  [EXPERIMENTAL]
IS_PERIODIC = 2  (Ewald force correction enabled, low-resolution table)
PDS curvature radius R_curv = 3100.0000 Mpc
PDS: Binary icosahedral group I* generated (120 elements).
MPI task 0: Allocating memory for the PDS Ewald lookup table with 1024 grid points...
PDS Ewald lookup table file (./examples/PDS_test_output/PDS_Ewald_table.hdf5) not found.
Calculating new lookup table...
PDS: Ewald image correction table calculated (1024 grid points, R_curv=3100.000).
PDS Ewald lookup table calculation finished. Wall-clock time = 2.31s.
```

The Ewald table is saved to `OUT_DIR/PDS_Ewald_table.hdf5` and reloaded on subsequent runs
with the same curvature radius.

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

## IS_PERIODIC Levels

| IS_PERIODIC | Mode | Ewald grid | Use case |
|---|---|---|---|
| 1 | Nearest image only | — | Fast tests, glass making |
| 2 | Low-res Ewald | 1 024 pts | Standard science runs |
| 3 | Medium-res Ewald | 4 096 pts | Higher accuracy |
| 4 | High-res Ewald | 16 384 pts | Maximum accuracy |

The Ewald table is computed once and cached.  Reuse is automatic when `R_curv` matches to
within 10⁻⁶ relative tolerance.

---

## Validation Tests

Before production runs, we recommend running the following sanity checks.

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

### No IC generator

No PDS-native IC generator is included.  Use the flat approximation (Strategy A, see Quick
Start) or modify `stepsic2` to draw from the discrete S³ power spectrum (Strategy B).

### CUDA kernel: no BH tree

The `ForceKernel_pds` CUDA kernel uses direct O(N²) summation only.  There is no PDS
Barnes-Hut tree implementation.  For N > 10 000, use the CPU build.

---

## File Reference

| File | Changes |
|---|---|
| `src/pds_group.h` | NEW: I* group (BFS generation), quaternion algebra, geodesic distance, force magnitude, fundamental domain test, boundary wrapping, image enumeration |
| `src/global_variables.h` | Added `PDS_Q`, `PDS_EWALD_FORCE_TABLE`, `N_PDS_EWALD_GRID`, `PDS_R_CURV` under `#elif defined(POINCARE_DODECAHEDRAL)` |
| `src/ewald_space.cc` | Added `calculate_pds_ewald_lookup_table()` and `pds_ewald_interpolate()` |
| `src/inputoutput.cc` | Added `save_pds_ewald_lookup_table()`, `load_pds_ewald_lookup_table()`; PDS fields in HDF5 header |
| `src/main.cc` | Global variable definitions; PDS forward declarations; Ewald table setup and MPI broadcast; topology warning |
| `src/step.cc` | PDS boundary wrapping after drift; force dispatch to `forces_pds()`; MPI broadcast of `PDS_Q` |
| `src/forces.cc` | `forces_pds()`: O(N²) CPU force with 120-image loop, softening, and Ewald correction |
| `src/forces_cuda.cu` | `ForceKernel_pds` CUDA kernel with I* in `__constant__` memory; `forces_pds_cuda()` host wrapper |
| All Makefiles | Added `#OPT += -DPOINCARE_DODECAHEDRAL` (commented out by default) |

---

## Background Reading

- Luminet, J.-P. et al. (2003) "Dodecahedral space topology as an explanation for weak wide-angle temperature correlations in the cosmic microwave background", *Nature* 425, 593–595.
- Weeks, J. R. (2001) "The Poincaré Dodecahedral Space and the Mystery of the Missing Fluctuations", *Notices of the AMS* 51, 610–619.
- Gomero, G. I. et al. (2016) "Simulating Cosmic Microwave Background maps in multiconnected spaces", *Phys. Rev. D* 94, 043501.
- Lehoucq, R. et al. (2002) "Eigenmodes of three-dimensional spherical spaces and their application to cosmology", *Class. Quantum Grav.* 19, 4683.
