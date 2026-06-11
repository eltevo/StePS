# Change Log
All notable changes to the StePS simulation code is documented in this file.

## [v2.2.1.0] - 2026-06-11

### Fixed
- **PDS: multi-GPU CUDA runs computed zero forces on every GPU except device 0
  (critical physics bug).** The I* group table lives in `__constant__ double
  PDS_I_STAR_DEV[120][4]`, and CUDA constant memory is *per device* — but the
  `cudaMemcpyToSymbol` upload in `forces_pds_cuda()` ran once in the serial
  section, so only the current device (GPU 0) ever received the table. The
  remaining GPUs ran `ForceKernel_pds` with an all-zero table: every image
  quaternion g⊗q evaluated to zero, `pds_force_dir_dev()` hit its
  `len2 < 1e-24` guard, and those GPUs returned exactly zero force. With the
  block particle decomposition this froze the upper (n−1)/n of the particle
  array on the initial grid (visually: a contiguous spatial region of the
  domain never evolving). The upload now happens inside the per-GPU OpenMP
  section right after `cudaSetDevice()`, with `pds_init()` called *before* the
  copy (the old order also uploaded the still-empty table on the first call).
  Per-GPU upload errors are now reported and returned.
  Verified on 4× NVIDIA H200: 4-GPU vs 1-GPU snapshots bit-identical through
  z = 0.5; 4-GPU vs CPU run (identical ICs) agree to RMS ≈ 10⁻¹⁰ Mpc through
  z = 0.7 with identical adaptive-timestep output redshifts.

### Changed
- **`PDS-Linux_CUDA-Makefile` reworked for the stepsic conda-env toolchain**
  (CUDA 12.9 + OpenMPI 5 + HDF5 from `$CONDA_PREFIX`; activate the env before
  building): `CUDA_PATH ?= $(CONDA_PREFIX)`, new `CUDA_ARCH ?= sm_90` knob
  passed as `-arch=` (H200/Hopper default), MPI/HDF5 include+lib paths taken
  from the env. Fixed a latent typo — link flags were defined as `CUDALFLAGS`
  but the rules referenced `CUDALDFLAGS`, so they were silently dropped — and
  the new `CUDALDFLAGS` no longer links `-lmpi_cxx` (removed C++ bindings,
  absent from OpenMPI ≥ 5) and adds an rpath to `$(CONDA_PREFIX)/lib` so the
  binary runs without `LD_LIBRARY_PATH`. The PDS CUDA build is now
  compile-tested and physics-validated on GPU hardware (it previously was not).

### Added
- **Millennium-style snapshot renderer** in `../tools/Visualization`:
  - `millennium_render.py` — SPH-like adaptively smoothed, logarithmic
    projected density maps in the classic Millennium/Gadget look (dark
    background, magma colormap, cosmic-age/redshift info box, Mpc +
    light-year scale bar). Pure numpy/scipy/matplotlib — no py-sphviewer
    dependency: per-particle smoothing lengths from the k-th nearest
    neighbour, deposited via a multi-scale histogram + Gaussian-filter stack
    in O(N log N). Works for any StePS HDF5 snapshot (R³, S¹×R², T³, and PDS);
    slice or full-projection modes, XY/XZ/YZ planes, animated-GIF helper,
    usable as a module or CLI.
  - `PDS_Millennium_View.ipynb` — companion notebook (defaults to the
    `data/pds_tests/test7b` run): snapshot overview table, full-size z = 0
    render, three projection planes, redshift-evolution mosaic, per-snapshot
    PNG frames + `evolution.gif`, and a parameter-tuning guide.

## [v2.2.0.0] - 2026-06-10

### Fixed
- **PDS: `IS_PERIODIC >= 2` produced ~zero gravitational forces (critical physics bug).**
  The 1D Ewald correction table was built with the bare S³ kernel 1/(R²sin²χ).
  The binary icosahedral group I* is closed under negation, the bare kernel
  satisfies G(π−χ) = G(χ), and antipodal images pull in exactly opposite
  directions — so the force from the full 120-image system of any source
  cancels identically, and the tabulated 119-image "correction" was exactly
  −G(χ_nearest). The total PDS force in Ewald mode was therefore zero up to
  table-interpolation error (median residual ≈ 9×10⁻⁵ of the nearest-image
  force). The previous validation tests all ran with `IS_PERIODIC = 1` and
  could not catch this; it also contributed to the missing structure formation
  in the `PDS_test.param` run. Quantified in
  `data/pds_anisotropy/REPORT.md` (study script:
  `examples/pds_tests/pds_anisotropy_study.py`).
- **PDS: initial forces were computed on un-wrapped IC positions.**
  `forces_pds()` ran before `calculate_init_h()` wrapped the IC into the
  fundamental domain and before the wrapped x/PDS_Q broadcast. The wrap is now
  factored into `pds_wrap_ic()` (`step.cc`), called on rank 0 *before* the
  initial force calculation, followed immediately by the x/PDS_Q broadcast.
  Verified: `h_start` is bit-identical for 1 vs 2 MPI ranks.
- **PDS: `PDS_R_CURV` was re-broadcast with `MPI_FLOAT` regardless of precision**
  (main.cc), corrupting the value on non-root ranks in double-precision builds
  with `IS_PERIODIC >= 2`. The redundant broadcast was removed
  (`read_paramfile.cc` already broadcasts it with the correct datatype).
- **PDS: mass/density consistency check used the wrong volume.** The check
  divided the total mass by the R³ sphere volume 4πR_sim³/3; it now uses the
  fundamental-domain volume V = π²R³_curv/60. A >1% mismatch is a warning (not
  a fatal error) under the experimental PDS topology, so deliberately
  non-cosmological validation test loads remain possible.
- Removed the dead self-comparison in the per-rank force-time bookkeeping in
  `main.cc` (`if(i==1 || force_calc_time > force_calc_time)`), a leftover of
  the v2.1.0.0 fix.

### Changed
- **PDS forces: exact 120-image summation with a background-compensated kernel
  replaces the nearest-image + 1D Ewald table scheme.**
  - New kernel `pds_green_compensated()` in `pds_group.h`:
    [1 − V(χ)/V_S³]/(R²sin²χ) with V(χ)/V_S³ = (2χ − sin 2χ)/(2π) — a point
    mass plus uniform negative background (the mean density is already in the
    Friedmann expansion; the exact analogue of dropping the k = 0 mode in T³
    Ewald summation). Finite at the antipode, → 1/r² as χ → 0, and it breaks
    the antipodal degeneracy that nullified the bare-kernel image sum.
  - For `IS_PERIODIC >= 2`, `forces_pds()` (and the CUDA `ForceKernel_pds`)
    now sums the compensated force over ALL 120 I* images of every source —
    exact on the compact S³, no lookup table needed. The j == i self-images
    are included (their sum is identically zero by the symmetry of the image
    constellation — χ(q, gq) = arccos(Re g) is independent of q — which is the
    PDS form of homogeneity, verified by the new Test 4).
  - `IS_PERIODIC == 1` remains a nearest-image-only fast mode, now with the
    compensated kernel. The former Ewald grid resolutions for
    IS_PERIODIC = 2/3/4 are obsolete; all values >= 2 select the exact sum.
  - Removed: `calculate_pds_ewald_lookup_table()`, `pds_ewald_interpolate()`,
    `save/load_pds_ewald_lookup_table()`, the `PDS_EWALD_FORCE_TABLE` /
    `N_PDS_EWALD_GRID` globals and the table setup/broadcast in `main.cc`.
    Old `PDS_Ewald_table.hdf5` files are no longer read.
  - Measured cost: ~2.3× per force evaluation (N=1000, CPU; 0.53 s → 1.24 s
    per step) — the price of exact, anisotropy-free image forces.
  - Cross-validated against an independent NumPy implementation
    (`stepsic/stepsic/pds.py`): max relative error 5×10⁻⁷ in single-precision
    snapshots, 4×10⁻¹² in double precision (new Test 6).

### Added
- **Native PDS initial-condition generation in stepsic** (`GEOMETRY = 'pds'`,
  see ../stepsic): stereographic Cartesian grid clipped to the dodecahedral
  fundamental domain, conformal-volume (Ω³) mass weighting for an exactly
  uniform comoving density on S³, flat-space 1LPT/2LPT displacements
  (Phase 7A approximation), out-of-domain particles wrapped with the exact
  velocity Jacobian, and a `/PartType1/Quaternions` (N,4) dataset that StePS
  reads directly. Example config: `examples/PDS_test_ic.toml` (NGRID=16 →
  1552 particles); `examples/PDS_test.param` updated to the new IC and to the
  Planck2018EE+BAO "best" parameters used by stepsic. With the new IC the
  StePS density check passes at the 10⁻⁷ level and `errmax` at startup is
  physical (no more 1.28×10⁸ from collapsing wrapped pairs).
- **`stepsic/stepsic/pds.py`** — NumPy port of `pds_group.h` (I* generation,
  wrapping, kernels, stereographic maps, velocity Jacobian) with a unit-test
  suite `stepsic/tests/test_pds.py` (17 tests).
- **Force-law study** `examples/pds_tests/pds_anisotropy_study.py` → results
  in `data/pds_anisotropy/REPORT.md`: documents the bare-kernel cancellation,
  the compensated-correction anisotropy (radial spread up to 0.45 and
  transverse component up to 0.19 of F_nearest at the domain boundary — too
  large for ANY 1D table, justifying exact summation), and reproduces the
  Roukema & Różański (2009) χ⁵ scaling of the anisotropic residual
  (fitted exponent 5.00).
- **Test suite extended from 3 to 8 tests** (`examples/pds_tests/run_tests.py`,
  outputs now under `data/pds_tests/`): Test 4 homogeneity (zero self-image
  force anywhere, drift < 10⁻¹⁶ Mpc), Test 5 multi-particle stability,
  Test 6 Python/C++ force cross-validation (< 10⁻⁶), Test 7 end-to-end
  stepsic-PDS-IC → StePS run with growing density contrast, Test 8 R³
  regression with a non-PDS build of the shared sources. The harness builds
  and caches the `StePS_saveacc` and `StePS_r3` binary variants.

## [v2.1.1.0] - TBA 2026-06-03

### Added
- **PDS physics validation test suite** (`examples/pds_tests/run_tests.py`). Three quick-running tests (1–2 particles, total wall time ≈ 7 s) that exercise the fundamental PDS physics paths:
  - Test 1 — Single particle free flight: verifies Hubble drag (`v_snap ∝ a⁻³/²`, < 2% residual), no NaN/Inf, quaternion unit-norm preserved to machine precision.
  - Test 2 — Fast particle boundary wrapping: particle starts just inside a dodecahedral face and immediately crosses it; checks all positions remain within the fundamental domain (max χ ≤ outradius ≈ 21°) and at least one crossing is detected.
  - Test 3 — Two-particle S³ gravitational convergence: verifies that the mutual S³ force produces measurable y-axis convergence consistent with the Newtonian limit (ratio meas/pred = 0.23, within the [0.1–10] sanity band; the gap from 1 is expected given Hubble damping and the first-snapshot offset).
  - Tests run with `concurrent.futures.ThreadPoolExecutor` for 3× wall-time speedup vs sequential.
  - Each test cleans its output directory before running, preventing stale-snapshot contamination.
  - Companion analysis notebook `examples/pds_tests/pds_test_report.ipynb` with embedded plots and HTML summary table.

- **`PDS_explorer.ipynb`** — interactive multi-snapshot explorer notebook for PDS simulation output (`examples/PDS_explorer.ipynb`): 3-D scatter with chi-coloured points, 2-D projections, S³ angular distribution, and multi-panel redshift evolution view.

### Fixed
- **`step.cc` — PDS velocity transformation at face crossings (critical correctness fix).**  
  When `pds_wrap` applies a non-trivial group element ḡ ∈ I* (i.e. whenever a particle crosses a dodecahedral face), the stereographic velocity `v = dx/dt` must also be transformed through the corresponding isometry. Without this, the velocity remains in the frame of the *old* position and becomes physically inconsistent with the *new* position in a different face of the domain.  
  The fix adds `pds_stereo_vel_transform()` to `src/pds_group.h` and calls it in both wrapping sites in `step.cc` (`calculate_init_h` and the drift-step block). The transformation is the exact Jacobian of `x_out = f(ḡ · f⁻¹(x_in))`:
  1. **Lift** `v_in` to S³ tangent via `u = [∂q/∂x]|_{x_in} · v_in`
  2. **Transport** `w = ḡ · u`  where  `ḡ = q_out · conj(q_in)`
  3. **Project** back to R³ via `v_out = [∂x/∂q]|_{q_out} · w`  
  Verified analytically: reduces to the identity when ḡ = 1 (no face crossing).

- **`main.cc` — broadcast wrapped positions after `calculate_init_h()`.**  
  `calculate_init_h()` wraps IC positions and transforms velocities on rank 0 but did not broadcast the result. Other MPI ranks therefore carried stale un-wrapped positions through the first timestep. A `MPI_Bcast` of `x[]` and `PDS_Q[]` is now issued immediately after the call, guarded by `#ifdef POINCARE_DODECAHEDRAL`.

- **`run_tests.py` — corrected PDS fundamental-domain inradius.**  
  The constant `CHI_IN` was set to `arccos(1/√5) ≈ 63.4°`, which is unrelated to the actual I* inradius. The nearest group element in I* is at χ = 36°, so the face midpoints (inradius) are at **χ_in = 18°** and the vertices (outradius) at **χ ≈ 20–21°**. The physical inradius = R_curv × 18° × π/180 ≈ 974 Mpc for R_curv = 3100 Mpc. Test 2 now uses the correct value with a +3° tolerance to accommodate vertices.

### Notes (PDS diagnostic findings)
- **Apparent "half-filled sphere" in snapshot explorers** is expected, not a bug. `R_SIM = 960 Mpc` in the parameter file is the *physical geodesic arc-length* radius (≈ the inradius). In *stereographic Cartesian* coordinates the fundamental domain extends only to ≈ 491 Mpc (face midpoints) and ≈ 575 Mpc (vertices). A reference circle drawn at the geodesic `SimulationRadius` value in stereo coords will therefore be ≈ 1.7× larger than the actual domain boundary.
- **No visible LSS in `PDS_test.param` run** — the cubical IC (`Lx = Ly = Lz = 1919 Mpc`) extends well beyond the dodecahedral fundamental domain (outradius ≈ 575 Mpc stereo). After `pds_wrap`, 912/1000 particles are remapped from 120 different copies of the space. Even with the corrected velocity transformation, the resulting density–velocity correlation is scrambled (particles from 120 statistically uncorrelated copies fill the domain uniformly). Additionally, 6 particle pairs land within the softening length (< 10 Mpc) after wrapping, producing initial accelerations with `errmax ≈ 1.28×10⁸` and particle velocities reaching ≈ 10⁵ km/s. A proper PDS IC is required; see plan.md for the roadmap.

## [v2.1.0.0] - TBA 2026-05-18 

### Added
- **EXPERIMENTAL: Poincaré Dodecahedral Space (S³/I*) topology.** Enable with `-DPOINCARE_DODECAHEDRAL`. Includes:
  - Runtime generation of the 120-element binary icosahedral group I* via BFS closure (`src/pds_group.h`)
  - Correct curved-space gravitational force law on S³: F = GM/(R²sin²χ) (Gauss's law on S³)
  - Boundary wrapping via Voronoi cell of the fundamental domain
  - 1D Ewald-style correction table D(χ) for contributions from all 119 non-nearest images (IS_PERIODIC ≥ 2)
  - Ewald table I/O in HDF5 format with automatic reuse when curvature radius matches
  - Multi-GPU CUDA kernel `ForceKernel_pds` with I* elements in `__constant__` memory
  - PDS metadata in HDF5 snapshot headers: `TopologicalManifold = "S^3/I*"`, `R_curvature_Mpc`, `Omega_k`, `PDS_I_star_order`
  - Four Ewald precision levels via IS_PERIODIC = 1/2/3/4 (nearest-image / 1024 / 4096 / 16384 grid points)
  - Documentation and quick-start guide in `docs/PDS_guide.md`

### Changed
- `pds_green()` force formula corrected to 1/(R²sin²χ) (Gauss's law on S³, giving correct 1/r² flat-space limit)
- Topology consistency warning at startup: mismatches between parameter file IS_PERIODIC and compiled topology are reported
- Gravitational softening coefficients in `force_softening()` are now cached per thread to avoid recomputation in the O(N²) inner loop

### Fixed
- MPI max-time tracking bug in `step.cc`: was comparing `force_calc_time` to itself (a no-op); now correctly tracks maximum over all MPI threads

## [v2.0.1.0] - 2026-06-04

### Added
- Accelerations can be saved to HDF5 snapshots.
- Added BH glass making warning message.
- MPI particle workload is redistributed, if significant workload imbalance is detected.

### Changed
- Updated Ewald split parameters in $S^1 \times \mathbb{R}^2$ Ewald-summation.
- MPI workload balance is printed out directly in MPI mode.
- Actual state saved as a snapshot if the simulation wall-clock time limit reached.
- Mass density is checked both $T^3$, $S^1 \times \mathbb{R}^2$, and $\mathbb{R}^3$ simulations, in the case of cosmological runs.
- Increased accuracy for mass density check in cosmological simulations

### Fixed
- Fixed gadget2 binary loading
- Fixed output list loading bug on newer libc libraries.

## [v2.0.0.0] - 2026-02-20

### Added
- Barnes-Hut (Octree) force calculation option (CPU only) [J. Barnes, P. Hut Nature 324 (6096) (1986) 446–449.]
- Random domain center shift option for periodic (T^3) Barnes-Hut simulations
- Random domain center shift and rotation option for spherical (R^3) Barnes-Hut simulations
- Random domain center shift and rotation in cylindrical (S^1xR^2) Barnes-Hut simulations
- Implemented S^1xR^2 topological manifold (cylindrically symmetric boundary conditions)
- Implemented Ewald summation in S^1xR^2 topology [Tornberg, A.-K. 2015, Advances in Computational Mathematics, 42, 227–248]
- Added radial force correction option in R^3 and S^1xR^2 Octree methods for radial simulation stability 
- Ewald lookup table I/O (for both T^3 and S^1xR^2 topological manifolds)
- Glass making logfile is produced during glass making

### Changed
- Updated makefile templates
- w0 and wa parameter values of wCDM and w0waCDM cosmologies are saved into the hdf5 snapshots
- Simulation radius is saved into the hdf5 snapshot header
- Simulation geometry (T^3, R^3, or S^1xR^2) is saved into the hdf5 snapshot header
- Cosmological parameters are saved to the logfiles
- Executable info (version, git commit ID, git branch, compiler, build date) are saved into the hdf5 snapshots and logfiles.
- Optimized gravitational softening calculation on GPUs
- Optimized force calculation CUDA kernels
- Individual GPU force calculation time is printed out
- Number of OpenMP and MPI threads are printed out during startup.

### Fixed
- Fixed constant-resolution periodic initial condition reading from HDF5 format
- Fixed H0 independent unit bugs
- Fixed high-accuracy Ewald summation option in fully periodic (T^3) simulations
- Simulation box size is saved properly in 32bit mode.
- Fixed tstart and ASCII format overwrite bugs.

## [v1.0.2.2] - 2024-10-25

### Added
- Added error messages for non-comoving cosmological simulations
- Added better descriptions to the README file

### Changed
- Updated linux-gcc makefile template

### Fixed
- Fixed malloc bug in the read_OUT_LST function
- Fixed memory allocation typos in the HDF5 reader function

## [v1.0.2.0] - 2024-01-23


### Added
- Added error message for non-comoving non-standard simulations
- Added non-comoving example simulation

### Changed
- Omega_dm parameter is changed to Omega_m in the paramfile.
- Updated example simulations (better filenames, Planck 2018 parameters, updated readme)
- Simulation wall-clock time is written out in hours too at the end of the simulation.
- Next output time/redshift is written to stdout in every timestep.

### Fixed
- Fixed redshift output bug

## [v1.0.1.0] - 2022-07-11


### Added
- Added option for wCDM cosmology parametrization
- Added option for w0waCDM cosmology parametrization
- Added option for using tabulated expansion history

### Changed

### Fixed
- Fixed deceleration parameter calculation


## [v1.0.0.0] - 2022-02-28

First github release.

### Main features
- Dark matter only LambdaCDM cosmological N-body simulations
- Parallelized with MPI, OpenMP and CUDA
- Direct force calculation
- HDF5, Gadget2 and ASCII input formats
- ASCII an HDF5 output formats
- Options for standard periodic and non-periodic spherical cosmological simulations
- Periodic, quasi-periodic or spherical glass generation
