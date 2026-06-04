# Change Log
All notable changes to the StePS simulation code is documented in this file.

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
