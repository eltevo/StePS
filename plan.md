# StePS PDS Upgrade Plan - By Claude Code

## Context

StePS (STEreographically Projected cosmological Simulations) is a mature N-body code (~11,000 lines C++/CUDA) that currently supports three topologies: R³ (open), T³ (fully periodic), and S¹×R² (cylindrical). The goal is to add Poincaré Dodecahedral Space (PDS) as a fourth topology, motivated by its status as a leading candidate for the true topology of the Universe (Luminet et al. 2003). PDS is a compact, positively curved 3-manifold constructed as the quotient S³/I*, where I* is the binary icosahedral group (120 elements). Adding PDS requires new physics (curved-space gravity), new boundary conditions (12 face identifications with π/5 rotations), a new Ewald-like correction scheme, and eventually a new IC generator.

This plan covers three deliverables: (1) a code review with concrete improvements, (2) a PDS upgrade strategy with phases and file-level detail, and (3) guidance on the IC problem.

---

## Part I — Code Review: Bugs, Optimizations, and Usability

### Confirmed Bug

**`step.cc:258-261`** — Dead/broken MPI max-time tracking:
```cpp
// Bug: comparison of variable to itself, assignment to self — both no-ops
if(i==1 || force_calc_time > force_calc_time)
    force_calc_time = force_calc_time;
```
The intent was to track the maximum force-calculation time across MPI threads. The code was probably meant to be:
```cpp
if(i == 1 || mpi_time_array[i] > max_force_calc_time)
    max_force_calc_time = mpi_time_array[i];
```
The `redistribute_workload()` call still receives the correct `mpi_time_array`, so redistribution itself works. The broken tracking only affects a local diagnostic variable, but it should be fixed for clarity.

### Performance Optimizations

1. **`forces.cc:52-87` — `force_softening()` recomputes `SOFT_CONST[]` on every call.**
   The five polynomial coefficients depend only on `beta`, which is fixed (or changes only when `recalculate_softening()` is called). They should be precomputed as static cached values, refreshed only when `beta` changes. This function is inside the O(N²) inner loop — the savings are real.

2. **`ewald_space.cc` — Ewald table generation is single-threaded.**
   Both `calculate_ewald_force_table()` (T³) and the S¹×R² equivalent are expensive one-time computations. They are naturally parallelizable with OpenMP. For a large grid this can take minutes; OpenMP parallelization of the outer loop would reduce startup time.

3. **`forces.cc` — Octree rebuilt from scratch every timestep.**
   The Barnes-Hut octree is fully reconstructed at each force evaluation. For slowly-evolving systems, a partially cached tree with lazy refinement would reduce overhead. This is a medium-effort improvement but significant for large N BH runs.

4. **`forces.cc` — Three separate force functions (`forces`, `forces_periodic`, `forces_periodic_z`) with large structural overlap.**
   Softening kernel application, OMP parallel loop setup, and MPI range management are duplicated. Refactoring the common parts into shared helpers (e.g., `force_pair(i, j, bc_wrapper)` with a topology-specific boundary callback) would reduce the ~1,400 lines by ~30% and make PDS integration cleaner.

### Usability / Developer Experience Issues

1. **Topology is compile-time only.** Users must recompile (and track which binary does what) for each topology. The runtime variable `IS_PERIODIC` is stored in snapshots and parameter files but the actual code path is fully determined by `-DPERIODIC` / `-DPERIODIC_Z` at build time. There is no runtime check that the parameter file's topology intent matches the compiled binary — a silent mismatch is possible. **Recommendation:** Add a startup assertion that prints clearly which topology was compiled in, and warn loudly if the parameter file's `IS_PERIODIC` value is inconsistent.

2. **`EWALD_INTERPOLATION_ORDER` is a Makefile variable, not a runtime parameter.** Users who want to change interpolation accuracy must recompile. Moving it to the parameter file would improve usability.

3. **`StePS_IC/` is deprecated but present without clear in-repo warnings.** The README refers users to `stepsic2` but `StePS_IC.py` has no deprecation notice in its own header. This misleads new users. Add a prominent `DEPRECATED.md` or notice in `StePS_IC/README.txt`.

4. **Parameter file has undocumented or partially documented parameters.** Parameters like `RADIAL_FORCE_ACCURACY`, `RADIAL_BH_FORCE_TABLE_SIZE`, etc., appear only in Makefile comments or deep in `read_paramfile.cc`. A single authoritative parameter reference (e.g., annotated `.param` template) would help.

5. **`global_variables.h` topology tables are `#ifdef`-gated.** The current pattern:
   ```cpp
   #ifdef PERIODIC
       extern REAL *T3_EWALD_FORCE_TABLE;
   #elif defined(PERIODIC_Z) && !defined(PERIODIC_Z_NOLOOKUP)
       extern REAL *S1R2_EWALD_FORCE_TABLE;
   #endif
   ```
   means the header changes meaning depending on compile flags. For PDS, a third `#elif defined(POINCARE_DODECAHEDRAL)` block is the natural extension, consistent with the existing pattern.

---

## Part II — PDS Upgrade Strategy

### Mathematical Foundations

**S³ and I*:**
- Represent particle positions as unit quaternions q = (q₀, q₁, q₂, q₃) ∈ S³ ⊂ ℝ⁴
- The binary icosahedral group I* has 120 elements, expressible as unit quaternions
- The fundamental domain of S³/I* is a spherical dodecahedron (inradius ≈ R·arccos(1/√5))
- The 12 face identifications pair opposite faces via a rotation by π/5 (36°)
- The 120 I* generators can be listed explicitly as constants (they are fixed, derived from the golden ratio τ = (1+√5)/2)

**Curved-space force law on S³:**
- Geodesic separation: χ(p, q) = 2 arccos(|p · q|) (p, q unit quaternions)
- Curved-space Green's function: G(χ) = (π − χ) / (4π² R sin χ)
- Force vector: **F** = G(χ) · (gradient term on S³)
- This differs significantly from 1/r² at separations χ ~ π (antipodal), and the force is finite even at χ = π

**Fundamental domain membership test:**
```
p is in the fundamental domain iff:
   |p · e₀| ≥ |p · g(e₀)| for all g ∈ I*
```
where e₀ = (1, 0, 0, 0) is the "north pole" and g(e₀) means quaternion multiplication. Since I* has 120 elements, this is 120 dot-product comparisons — cheap.

**Boundary wrapping:**
Given particle q that has drifted outside the fundamental domain:
1. Find g* = argmax_{g ∈ I*} |q · g(e₀)|
2. Set new position q' = conj(g*) ⊗ q (apply inverse rotation)

**Curved Friedmann background:**
The code already stores `Omega_k` in `global_variables.h`. The `CALCULATE_Hubble_param()` in `friedmann_solver.cc` needs to include the `Omega_k · a⁻²` term if `Omega_k ≠ 0`. Check whether this term is already included (if not, add it as a one-line fix). For PDS: Ω_total ≈ 1.018, so Ω_k ≈ −0.018.

---

### Phase 0 — Preparatory Code Quality Fixes (1 week)

These are independent improvements that make PDS integration cleaner. Do these first.

**Files:** `step.cc`, `forces.cc`, `global_variables.h`, `friedmann_solver.cc`, `StePS_IC/README.txt`

1. **Fix the MPI tracking bug** in `step.cc:258-261` (trivial — one line).
2. **Precompute `force_softening` coefficients**: Add a cached struct updated only in `recalculate_softening()`.
3. **Add startup topology assertion**: In `main.cc`, print which topology was compiled in and warn if `IS_PERIODIC` value is inconsistent with the compiled topology.
4. **Verify `CALCULATE_Hubble_param`** handles `Omega_k ≠ 0`. If `Omega_k·a⁻²` is missing from `H(a)`, add it.
5. **Deprecation notice**: Add `DEPRECATED` header to `StePS_IC/src/StePS_IC.py` and `StePS_IC/README.txt`.
6. **Parallelize Ewald table generation** with OpenMP (outer loop in `calculate_ewald_force_table()`).

---

### Phase 1 — PDS Compile Flag and I* Group (1 week)

**Files:** All 4 Makefile templates, `global_variables.h`, new file `pds_group.h`

1. **Add compile flag** to all Makefile templates:
   ```makefile
   #OPT += -DPOINCARE_DODECAHEDRAL  # S^3/I* Poincare Dodecahedral Space
   ```
   This follows the `-DPERIODIC` / `-DPERIODIC_Z` pattern exactly.

2. **Create `StePS/src/pds_group.h`** — a header-only file containing:
   - The 120 I* group elements as a `const double PDS_I_STAR[120][4]` array of unit quaternions
   - Quaternion multiplication helper: `quat_mult(p, q)`
   - Quaternion conjugate: `quat_conj(q)`
   - Geodesic distance: `pds_chi(p, q)` = 2 arccos(|p·q|)
   - Curved Green's function: `pds_green(chi, R_curv)`
   - Fundamental domain test: `pds_in_domain(q)` (returns bool)
   - Wrapping function: `pds_wrap(q, q_out)` (finds canonical representative)

   The 120 I* elements are fixed constants derived from the golden ratio and can be hard-coded (they are well-known; see Weeks 2001 or Luminet 2003 appendices).

3. **Extend `global_variables.h`** with:
   ```cpp
   #elif defined(POINCARE_DODECAHEDRAL)
       extern REAL *PDS_EWALD_FORCE_TABLE;
       extern int N_PDS_EWALD_GRID;     // 1D table indexed by geodesic distance
       extern REAL PDS_R_CURV;          // Curvature radius in internal units
   #endif
   ```

---

### Phase 2 — Boundary Conditions (1 week)

**File:** `step.cc`

Add a `#elif defined(POINCARE_DODECAHEDRAL)` block in both `calculate_init_h()` and `step()` where boundary wrapping occurs:

```cpp
#elif defined(POINCARE_DODECAHEDRAL)
for(i = 0; i < N; i++) {
    double q[4] = {x[4*i], x[4*i+1], x[4*i+2], x[4*i+3]};  // unit quaternion position
    double q_out[4];
    pds_wrap(q, q_out);  // maps to fundamental domain
    x[4*i]=q_out[0]; x[4*i+1]=q_out[1]; x[4*i+2]=q_out[2]; x[4*i+3]=q_out[3];
}
```

**Note on coordinate storage:** PDS particles live on S³ and require 4D coordinates (unit quaternions). The current `x[]` array stores 3D Cartesian coordinates as `x[3*i+k]`. For PDS mode, you have two options:
- **Option A (preferred):** Store 4D quaternion positions in a separate `q[4*N]` array for PDS mode, leaving `x[3*N]` as the projected/ambient-space position for force calculations. Add `#ifdef POINCARE_DODECAHEDRAL extern REAL *q; #endif` to `global_variables.h`.
- **Option B:** Repurpose `x[3*i]..x[3*i+2]` as q₁,q₂,q₃ and add a separate `q0[N]` array for q₀. Less clean but avoids changing array size.

Option A is cleaner and recommended.

Also update force dispatch in `step()`:
```cpp
#if defined(PERIODIC)
    forces_periodic(x, F, ID_MPI_min, ID_MPI_max);
#elif defined(PERIODIC_Z)
    forces_periodic_z(x, F, ID_MPI_min, ID_MPI_max);
#elif defined(POINCARE_DODECAHEDRAL)
    forces_pds(q, F, ID_MPI_min, ID_MPI_max);
#else
    forces(x, F, ID_MPI_min, ID_MPI_max);
#endif
```

---

### Phase 3 — Ewald/Image Table for PDS (2–3 weeks)

**File:** `ewald_space.cc`, `inputoutput.cc`, `main.cc`

#### 3a. PDS Force Table Structure

Unlike T³ (3D table) and S¹×R² (2D table), the PDS table needs only a **1D table indexed by geodesic distance χ ∈ [0, π]**. The reason: S³ is maximally symmetric, and I* acts isotropically enough that the Ewald correction depends only on χ. This simplifies the table enormously.

The table stores the **correction** to the nearest-image curved-space force from all non-nearest I* images. Let:
- D(χ) = Σ_{g ∈ I*, g ≠ nearest} G(χ_g)   (sum over 119 other images)

This requires iterating over all 120 I* images of a source particle, computing geodesic distances and forces, and summing. The table can be tabulated as a function of the nearest-image separation χ.

**Table generation function** (`calculate_pds_ewald_lookup_table` in `ewald_space.cc`):
```c
void calculate_pds_ewald_lookup_table(int Ngrid, REAL R_curv, REAL *PDS_EWALD_FORCE_TABLE) {
    // For each chi in [0, pi] (Ngrid points):
    //   For a "source" at north pole e0 and "field point" at chi:
    //     Compute position of field point as unit quaternion p
    //     For each g in I* (120 elements):
    //       Apply g to source: source_image = g * e0
    //       Compute chi_g = pds_chi(p, source_image)
    //       Accumulate force from G(chi_g)
    //     Subtract nearest-image force
    //     Store correction in table
}
```
The symmetry reduction: due to I* symmetry and S³ isotropy, the table only needs Ngrid ~ 1000 points in χ. This is far cheaper than the T³ (Ngrid³ grid).

#### 3b. I/O for the Table

In `inputoutput.cc`, add:
```cpp
#ifdef POINCARE_DODECAHEDRAL
void save_pds_ewald_lookup_table(const char *filename, int Ngrid, REAL R_curv, REAL *PDS_EWALD_FORCE_TABLE);
void load_pds_ewald_lookup_table(const char *filename, int *Ngrid, REAL *R_curv, REAL **PDS_EWALD_FORCE_TABLE);
#endif
```
Follow the exact pattern of `save_t3_ewald_lookup_table()` / `load_t3_ewald_lookup_table()`.

#### 3c. Main Loop Integration

In `main.cc`, add alongside the existing T³/S¹×R² table setup:
```cpp
#elif defined(POINCARE_DODECAHEDRAL)
    // Try to load cached table; recompute if not found or R_curv changed
    if(load_pds_ewald_lookup_table(pds_table_file, &N_PDS_EWALD_GRID, &PDS_R_CURV, &PDS_EWALD_FORCE_TABLE) != 0) {
        printf("Calculating PDS Ewald lookup table...\n");
        calculate_pds_ewald_lookup_table(N_PDS_EWALD_GRID, PDS_R_CURV, PDS_EWALD_FORCE_TABLE);
        save_pds_ewald_lookup_table(pds_table_file, N_PDS_EWALD_GRID, PDS_R_CURV, PDS_EWALD_FORCE_TABLE);
    }
#endif
```

---

### Phase 4 — Force Calculation (2 weeks)

**Files:** `forces.cc`, `forces_cuda.cu`

#### 4a. CPU Force Function

Add `forces_pds(REAL* q, REAL* F, int ID_min, int ID_max)` to `forces.cc`:

```cpp
void forces_pds(REAL* q, REAL* F, int ID_min, int ID_max) {
    // For each particle i in [ID_min, ID_max]:
    //   F[3*i] = F[3*i+1] = F[3*i+2] = 0
    //   For each particle j != i:
    //     Compute nearest-image separation chi_min over 120 I* images of j
    //     Compute curved-space force from nearest image using G(chi_min)
    //     If IS_PERIODIC >= 2: add Ewald correction from PDS_EWALD_FORCE_TABLE
    //     Accumulate to F[3*i]
}
```

The force direction on S³: for particles at positions p and q (unit quaternions), the force on p from q points along the geodesic from p toward q. The unit tangent vector at p in the direction of q is:
```
t = (q - (p·q)p) / |q - (p·q)p|   (normalized component of q perpendicular to p)
```
So: F_on_p = M_j * G(χ) * t

For IS_PERIODIC==1 (nearest image only), no table lookup needed — just sum over 120 I* images.
For IS_PERIODIC>=2, add the Ewald correction from the 1D table.

#### 4b. GPU Force Function

In `forces_cuda.cu`, add a CUDA kernel `forces_pds_kernel`:
- Each thread block handles a subset of i particles
- Inner loop: for each j, compute forces from all 120 I* images of particle j
- The 120 I* quaternion constants can be stored in CUDA constant memory (`__constant__ double PDS_I_STAR_CUDA[120][4]`)
- This is a contained modification: the existing CUDA kernel structure (block decomposition, shared memory tiling) remains unchanged; only the pairwise force formula changes

---

### Phase 5 — I/O and HDF5 Headers (3 days)

**File:** `inputoutput.cc`

Update `write_snapshot_hdf5()` to write PDS metadata in the snapshot header:
```
/Header/Geometry = "POINCARE_DODECAHEDRAL"
/Header/R_curvature_Mpc = PDS_R_CURV (in Mpc)
/Header/Omega_k = Omega_k
/Header/PDS_I_star_order = 120
```
Follow the pattern of how `"CYLINDRICAL"` / `"PERIODIC"` are written.

---

### Phase 6 — Initial Conditions (4–8 weeks, research-level)

This is the hardest and most open-ended part. Three strategies, in increasing accuracy:

#### Strategy A: Flat approximation (fast, ~0 new IC code)
Use existing stepsic2 (or StePS_IC) to generate T³ ICs, then interpret the fundamental domain as the PDS cell. Valid when simulation volume ≪ R_curv (curvature corrections ~(L/R)² ≈ a few percent for realistic setups). This gives a fast first simulation to validate the dynamical code.

#### Strategy B: Approximate curved ICs via tilted power spectrum (weeks)
The PDS power spectrum suppresses modes below the curvature scale. For small Ω_k, this can be approximated by modifying the input power spectrum P(k) to be zero for k < 2π/R_curv and using the discrete S³ eigenmode spacing. This requires:
- Computing allowed discrete k values: k_n = √(n(n+2)) / R_curv (n = 1, 2, 3, ...)
- Sampling the power spectrum only at these k values
- Can be implemented as a post-processing step on stepsic2 output

Extend `StePS_IC/src/powerspec.py` with a `pds_discrete_power_spectrum()` function.

#### Strategy C: Full S³ eigenmode ICs (months, publish-quality)
Expand the density field in I*-invariant eigenmodes of the Laplacian on S³ (Gegenbauer polynomials). Only modes consistent with I* symmetry contribute (a discrete subset of modes indexed by n ≥ 1). This requires a new standalone IC tool (call it `pds_ic`) that:
1. Reads a ΛCDM power spectrum
2. Computes I*-invariant modes (there are ~N_modes of them for a given max n)
3. Draws Gaussian amplitudes from P(k_n)
4. Synthesizes a density field on the dodecahedral fundamental domain
5. Solves for particle displacements (Zeldovich approximation on curved space)
6. Outputs in StePS HDF5 format

This is a standalone research project. Reference: Lehoucq et al. (2002), Riazuelo et al. (2004) for the eigenmode approach.

**Recommendation:** Start with Strategy A to validate the dynamical code, then implement Strategy B for physically motivated ICs, and defer Strategy C to a dedicated paper.

---

## Part III — Revised Assessment of the Rough Plan

The original `PDS_upgrade_ideas.md` plan is correct in structure but underestimates some challenges:

| Item in rough plan | Assessment |
|---|---|
| New Makefile flag | ✓ Straightforward |
| Boundary condition in step.cc | ✓ Doable but requires 4D quaternion position storage (not 3D Cartesian) — needs coordinate design decision |
| Ewald lookup table in ewald_space.cc | ✓ Simpler than T³ (1D table, not 3D), because of I* symmetry |
| Table I/O in inputoutput.cc | ✓ Mechanical extension of existing pattern |
| Main.cc integration | ✓ Mechanical |
| Force calculation in forces.cc/forces_cuda.cu | ✓ Core work: new formula, 120-image loop |
| "Gravity physics is wrong" — curved Green's function | ⚠️ This is physics work, not just coding; needs validation against known S³ solutions |
| IC generation | ⚠️ Most open-ended; recommend phased approach above |
| Curved Friedmann background | ✓ May already be partially implemented (Omega_k exists); needs verification |

---

## Verification Strategy

1. **Unit tests for `pds_group.h`**: Verify all 120 I* elements are unit quaternions, closed under multiplication, and that the fundamental domain test + wrap function round-trips correctly.
2. **Force law sanity**: At small χ, G(χ) should approach the flat-space 1/r² (since sin χ ≈ χ for small χ). Test the force kernel at χ ≪ 1.
3. **Energy conservation**: Run a test simulation with a few particles and verify the Hamiltonian is conserved (to integration-error level).
4. **Image summation**: For a single particle at the center of the fundamental domain (at e₀), all 120 images should contribute equal forces that cancel by symmetry. Test that Σ F = 0.
5. **T³ regression**: After code changes, verify T³ and S¹×R² runs reproduce previous results exactly (snapshot comparison).
6. **PDS glass**: Generate a glass (repulsive gravity run) in PDS topology and verify the particle distribution is isotropic within the fundamental domain.

---

## Critical Files Summary

| File | Changes |
|---|---|
| `StePS/src/global_variables.h` | Add PDS table variables under new `#elif POINCARE_DODECAHEDRAL` block |
| `StePS/src/step.cc` | Fix MPI bug; add PDS boundary wrapping; add PDS force dispatch |
| `StePS/src/forces.cc` | Add `forces_pds()`; precompute softening coefficients |
| `StePS/src/forces_cuda.cu` | Add `forces_pds_kernel()` CUDA kernel with 120-image loop |
| `StePS/src/ewald_space.cc` | Add `calculate_pds_ewald_lookup_table()`; parallelize existing tables |
| `StePS/src/inputoutput.cc` | Add PDS table save/load; add PDS HDF5 header fields |
| `StePS/src/main.cc` | Add PDS table setup; add startup topology consistency check |
| `StePS/src/friedmann_solver.cc` | Verify/add Ω_k term in `CALCULATE_Hubble_param()` |
| All 4 Makefiles | Add `-DPOINCARE_DODECAHEDRAL` option (commented out by default) |
| New: `StePS/src/pds_group.h` | I* elements, quaternion ops, geodesic distance, Green's function, wrap/domain test |
| `StePS_IC/src/powerspec.py` | Add `pds_discrete_power_spectrum()` for Strategy B ICs |

---

## Estimated Effort

| Phase | Effort |
|---|---|
| Phase 0: Code quality | 1 week |
| Phase 1: Flag + I* group | 1 week |
| Phase 2: Boundary conditions | 1 week |
| Phase 3: Ewald table | 2–3 weeks |
| Phase 4: Force calculation | 2 weeks |
| Phase 5: I/O | 3 days |
| Phase 6A: Flat ICs (validation) | ~0 (use existing stepsic2) |
| Phase 6B: Approximate curved ICs | 2–4 weeks |
| Phase 6C: Full eigenmode ICs | 2–3 months |
| Testing & validation | 2–3 weeks |
| **Total (Phases 0–5 + 6B)** | **~3 months** |

---

## Implementation Status (as of 2026-06-03)

### Completed

| Item | File(s) | Notes |
|------|---------|-------|
| Phase 0–5 all merged | `src/pds_group.h`, `src/step.cc`, `src/forces.cc`, `src/ewald_space.cc`, `src/inputoutput.cc`, `src/main.cc`, `src/global_variables.h` | See CHANGELOG v2.1.0.0 |
| **Velocity transformation at face crossings** | `src/pds_group.h`, `src/step.cc` | New function `pds_stereo_vel_transform()`. Critical correctness fix: velocities were previously left in the wrong frame after pds_wrap. See CHANGELOG v2.1.1.0. |
| **MPI broadcast after IC wrapping** | `src/main.cc` | Wrapped x/PDS_Q broadcast to all ranks before first timestep. |
| **PDS validation test suite** | `examples/pds_tests/run_tests.py` | 3 tests, 7 s total. Parallel execution via ThreadPoolExecutor. Corrected inradius = 18°. |
| **PDS_explorer notebook** | `examples/PDS_explorer.ipynb` | Multi-snapshot interactive visualisation. |
| **PDS test report notebook** | `examples/pds_tests/pds_test_report.ipynb` | Hubble-drag, boundary-wrapping, and convergence analyses with plots. |

### Identified Issues Not Yet Fixed

| Issue | Root cause | Required action |
|-------|-----------|-----------------|
| **No LSS in PDS_test.param** | Cubical IC places 912/1000 particles outside the fundamental domain; post-wrap density–velocity coherence is destroyed even with correct velocity transform | New PDS IC (see Phase 7 below) |
| **6 particle pairs < 10 Mpc after wrapping** | Cubical grid tiles into 120 copies; some pairs from adjacent copies collapse to < softening after wrapping | Fixed by proper IC; can also increase softening temporarily |
| **PDS_explorer reference circle too large** | Draws circle at geodesic `SimulationRadius` (960 Mpc) in stereo coordinates; domain extends only to ≈ 575 Mpc stereo | Fix in PDS_explorer: draw at stereo outradius = R_curv × tan(χ_out/2) ≈ 575 Mpc |
| **Initial force computed with un-wrapped positions** | `forces_pds()` is called before `calculate_init_h()` in main.cc | Move initial force calc to after the IC-wrap broadcast, or call pds_wrap in IC loading |

---

## Phase 7 — Proper PDS Initial Conditions

This is now the **top priority** for making PDS simulations physically meaningful.

### Background

The old `StePS/StePS_IC/` IC generator is deprecated. The replacement is **`stepsic`** (`../stepsic/`, `git clone https://github.com/eltevo/stepsic`), a fully restructured IC generator with support for cubical, spherical, and cylindrical geometries. The `stepsic` codebase will be the basis for PDS IC generation.

**Key geometry facts** (all must be respected by the IC):

| Quantity | Value |
|----------|-------|
| I* nearest neighbour distance | 36° |
| Fundamental domain **inradius** (face midpoints) | χ_in = **18°**, physical = R_curv × π/10 ≈ 974 Mpc |
| Fundamental domain **outradius** (vertices) | χ_out ≈ **20–21°**, stereo ≈ 575 Mpc |
| Face midpoints in stereographic coords | r_stereo = R_curv × tan(9°) ≈ **491 Mpc** |
| Face-to-face diameter (stereo box to inscribe) | ≈ **982 Mpc** |

### Why the current cubical IC fails

The 1919 Mpc cubical IC box corners reach χ ≈ 56° >> χ_in = 18°: 912/1000 particles are outside the fundamental domain. After `pds_wrap` they are redistributed from 120 uncorrelated copies of the space, destroying all LPT correlations. The velocity transformation (`pds_stereo_vel_transform`) correctly rotates each individual particle's velocity to match its new position, but cannot reconstruct the global density–velocity coherence that is necessary for structure formation.

### Phase 7A — Dodecahedral domain IC (minimal viable, 1–2 weeks)

**Goal:** generate an IC where ALL particles are inside the fundamental domain at t=0, with physically motivated LPT displacements and velocities.

**Steps in `stepsic`:**

1. **Add a `PDS` geometry class** alongside the existing `Spherical` and `Cylindrical` classes in `stepsic/geometry.py`. The class defines the dodecahedral domain by the inradius constraint χ < χ_in = 18°, expressed in stereographic Cartesian coordinates as:
   ```python
   r_stereo = R_curv * np.tan(np.radians(9.0))   # ≈ 491 Mpc — face midpoints
   ```
   Use a rejection-sampling Poisson glass (or hierarchical grid) to place N particles uniformly inside the dodecahedral domain. The simplest approach: generate candidates inside the inscribed sphere of radius 491 Mpc (guaranteed inside the domain), then optionally allow particles in the dodecahedral corners (up to ≈ 575 Mpc stereo) by checking `pds_in_domain(q)` for each candidate.

2. **Flat-space LPT displacements** (valid for R_sim ≪ R_curv). For the current setup R_sim ≈ 974 Mpc, R_curv = 3100 Mpc, R_sim/R_curv ≈ 0.31 — curvature corrections are of order (R_sim/R_curv)² ≈ 10%, acceptable for a first simulation. Use the standard Zel'dovich (1LPT) or 2LPT displacement field from the flat-space power spectrum P(k) evaluated at k = n/R_curv (discrete modes). The power spectrum should have zero power at k < 2π/R_curv (no super-horizon modes).

3. **Check post-displacement domain membership.** After applying LPT displacements, verify all particles remain inside the domain (or clip them). The displacement magnitude should be much less than the domain size at z_start ≈ 30.

4. **IC file format.** Write in StePS HDF5 format (existing `stepsic/io.py` `write_gadget2_hdf5()`). Include a `PartType1/Quaternions` dataset with unit quaternions to avoid the stereographic-projection cost at IC read time (StePS will use these directly and skip the `pds_wrap` call entirely at startup).

5. **Softening and mass.** Target mean inter-particle spacing > 2× softening after placement. For 1000 particles in a domain of radius ≈ 974 Mpc, mean spacing ≈ 280 Mpc; softening = 10 Mpc is fine. For 10,000 particles, mean spacing ≈ 130 Mpc.

**Immediate validation:**
- Run with the new IC and verify `errmax` at startup is < 10³ (not 10⁸)
- Verify all particles stay in the domain at z=0 (chi < 21°)
- Verify RMS density contrast grows monotonically from z_start to z=0
- Run the existing `examples/pds_tests/run_tests.py` — should still pass

### Phase 7B — Curved-space power spectrum (2–4 weeks)

Replace the flat-space P(k) evaluation with the S³ discrete spectrum. The allowed modes on S³ are labelled by integer n ≥ 1 with wavenumber k_n = √(n(n+2))/R_curv. The I*-invariant subset (modes that respect the dodecahedral symmetry) is a small fraction of all S³ modes — Lehoucq et al. (2002) enumerate them. The displacement amplitude is drawn from P(k_n) as usual.

**Implementation pointer in `stepsic`:** extend `stepsic/pk.py` with:
```python
def pds_allowed_modes(R_curv, n_max):
    """Enumerate I*-invariant S³ modes up to multipole n_max.
    Returns array of effective wavenumbers k_n = sqrt(n*(n+2))/R_curv."""
    ...

def pds_power_spectrum_samples(pk_func, R_curv, n_max):
    """Sample P(k) at the allowed PDS mode wavenumbers."""
    k_modes = pds_allowed_modes(R_curv, n_max)
    return k_modes, pk_func(k_modes)
```

### Phase 7C — I*-invariant eigenmode IC (months, publication quality)

Full expansion of δ(r) in I*-invariant eigenfunctions of the S³ Laplacian. See Lehoucq et al. (2002, A&A 386, 55) and Riazuelo et al. (2004, Phys. Rev. D 69, 103518). This is a standalone research project and should be treated as a separate task.

### Changes required in `StePS/src/` for a proper PDS IC

1. **Move initial force calculation after IC-wrap** in `main.cc`: the `forces_pds()` call at line ≈ 1898 currently uses un-wrapped `PDS_Q`. It should be moved to after the `calculate_init_h()` + broadcast block so forces are always computed on wrapped positions.

2. **Remove forced re-wrap if IC contains `PartType1/Quaternions`**: `inputoutput.cc` already checks for this dataset and uses it directly. When the IC provides pre-computed quaternions, `calculate_init_h()` should skip `pds_wrap` for in-domain particles (it already does — `pds_quat_same(q_in, q_out)` returns true → velocity transform is skipped). No code change needed here.

3. **Add `SimulationRadius` header in stereo units** (optional but helpful): `inputoutput.cc` should write both the geodesic radius (`SimulationRadius_physical_Mpc`) and the stereographic radius (`SimulationRadius_stereo_Mpc = R_curv * tan(chi_in/2)`) to the HDF5 header. `PDS_explorer.ipynb` can then draw the reference circle at the correct stereo radius.

---

## Summary: Immediate Next Actions

1. **Add PDS geometry to `stepsic`** (`../stepsic/stepsic/geometry.py`) — implement `PDSGeometry` class with dodecahedral domain sampling.
2. **Generate a proper PDS IC** with N = 1000–10000 particles, all inside χ < 18°, using 1LPT flat-space displacements as a first approximation.
3. **Fix `PDS_explorer.ipynb`** reference circle to use stereo outradius (≈ 575 Mpc) instead of geodesic `SimulationRadius`.
4. **Move initial force calculation** in `main.cc` to after `calculate_init_h()` + broadcast.
5. **Run a new PDS simulation** with the new IC and verify structure formation (growing density variance, converging particle pairs).
