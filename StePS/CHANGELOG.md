# Change Log
All notable changes to the StePS simulation code is documented in this file.

## [Unreleased] - code review follow-up (2026-07)

Findings from an independent code review (`Code_reviews.md`), verified and addressed.

### Changed
- **PDS force now uses the full stereographic Jacobian.** The kernels return the physical
  geodesic acceleration as a 4D tangent `t` at `q`; the exact pushforward into the
  conformally-flat chart the drift integrates in is

      dx_i = ( R*t_{i+1} - x_i*t_0 ) / Omega,   Omega = 1 + q0 = 2R^2/(R^2+r^2)

  The `t_0` term was previously discarded (only `t_{i+1}/Omega` was kept), while the same
  full Jacobian was already implemented correctly for velocities in
  `pds_stereo_vel_transform()`. Fixed at **all four sites**: `forces.cc` (direct + BH) and
  `forces_cuda.cu` (direct + BH). Implemented as `x_i/R = q_{i+1}/Omega` so the correction
  depends only on `PDS_Q`, never on `x[]` being in sync.
  - **Effect on results:** forces change by ~1% median, **4.4% max** at the domain
    boundary (direction by <=1.3 deg); the error scales as `(r/R)^2`. Runs made before
    this change are not bit-comparable with runs made after.
  - Verified: C++ matches an independently derived NumPy formula to `1.15e-13`;
    **CPU and CUDA agree to `5.6e-12`** (0 of 2000 particles above `1e-5`).
  - This supersedes part of the planned Phase-1 work: the e0-drop is now gone, though the
    flat drift itself remains (see docs/PDS_guide.md "Known Limitations").

### Fixed
- **Test 6 (Python/C++ force cross-validation) was silently broken.** Its NumPy reference
  was never updated when the `1/Omega` conformal factor was introduced, so the test had
  been failing by **~50%** (measured 48.9-50.0%) and no independent check validated the
  PDS force scaling. Reference corrected and extended to the full pushforward; it now
  passes at `4.76e-07` against a `1e-6` gate.
  - Known coverage gap: Test 6's 50 particles are far enough apart that the per-pair
    softening floor is never exercised, and its reference assumes a uniform softening
    whereas StePS uses mass-dependent `SOFT_LENGTH[i] = cbrt(M[i]*const_beta)`.
- **`stepsic` could not run on current SciPy/NumPy.** `scipy.special.sph_harm` was removed
  in SciPy 1.17 and `np.trapz` in NumPy 2.4, breaking the discrete-spectrum (Phase 7B) IC
  path advertised in `pyproject.toml` (`scipy>=1.8`, no ceiling): 4 test failures. Added a
  version-portable shim (`sph_harm_y` when available, legacy argument order otherwise —
  verified bit-identical) rather than pinning. Suite goes 27 -> **31/31** on both
  SciPy 1.15/NumPy 2.2 and SciPy 1.17/NumPy 2.4.
- **`s3lpt.tangent_basis` returned a rank-2 frame at coordinate-axis quaternions**,
  including `q = (1,0,0,0)` — the stereographic origin, i.e. a particle at the box centre.
  A vanishing Gram-Schmidt candidate consumed an output row, so one physical gradient
  direction was silently dropped from the Zel'dovich displacement. Rebuilt to discard the
  axis most nearly parallel to `q` (guaranteeing every retained candidate has norm
  `>= 1/sqrt(2)`) plus a re-orthogonalization pass; this also removes catastrophic
  cancellation *near* the axes, which degraded orthogonality to ~1e-5. Now machine
  precision everywhere, with `sum_a e_a e_a^T = I - qq^T` to `4.8e-16`. Bit-identical to
  the old frame at generic points (checked over 50k quaternions).

- **PDS softening is now the standard StePS cubic spline** instead of a distance floor
  (P2-2). The old `chi_eff = max(chi, chi_soft)` held the force at its value at `chi_soft`
  down to `chi -> 0+` and then dropped it discontinuously to zero at coincidence. The
  spline is applied to the areal radius `A = R sin(chi)` and spliced so that
  `A*S(A) == 1/A^2` for `A >= beta`, hence the far field is unchanged to `4.4e-16` while
  the force now goes smoothly to zero. Host (`pds_green_softened`, `pds_group.h`) and
  device (`pds_green_soft_dev`, `forces_cuda.cu`) versions agree to `7e-15`; CPU vs GPU
  end-to-end `4.06e-11`.
  - **Effect on results:** close-pair forces change substantially — the softened force
    peaks inside `beta` (at `A ~ 0.4 beta`) rather than being clamped at `1/beta^2`.
    Only pairs closer than `beta` are affected.
- **IC quaternions are now the integrated state** (P2-3). When `/PartType1/Quaternions` is
  present, `pds_wrap_ic()` wraps those values and re-derives `x[]` from the result, instead
  of rebuilding `q` from the (float32) Cartesian array and discarding what the IC supplied.
  New global `PDS_Q_FROM_IC` records provenance.

### Added
- **Intrinsic S^3 geodesic integrator, opt-in via `-DPDS_INTRINSIC`** (P1-1; roadmap
  Phase 1). Default **off** — the validated stopgap remains the production path until this
  has been exercised on full-size runs.
  - State becomes `(PDS_Q, PDS_U)` with `U` the physical peculiar velocity as a 4D tangent
    (`U _|_ q`, `dq/dt = U/R`). `x[]`/`v[]` become derived buffers refreshed by
    `pds_sync_stereo_for_output()`. `PDS_U` lives on rank 0 only and is never broadcast.
  - Drift is `pds_exp_map()` — the exact geodesic flow with parallel transport — replacing
    `x += v*h`. Wrapping becomes a pure isometry on `(q, U)` instead of a 3x3 stereographic
    Jacobian on `v`.
  - New header-only helpers in `pds_group.h`: `pds_exp_map`, `pds_rotate_tangent`,
    `pds_project_tangent`, and `pds_tangent_from_stereo_vel` /
    `pds_stereo_vel_from_tangent` (factored out of `pds_stereo_vel_transform`, which they
    reproduce bit-identically).
  - **No force buffer was widened.** The roadmap called for a 4-wide `PDS_FDIM` through
    every MPI and CUDA buffer; that turned out to be unnecessary, because the summed force
    is a tangent and so `t0 = -(t1 q1 + t2 q2 + t3 q3)/q0` is recoverable exactly from what
    is already transmitted. `q0 >= 0.95` throughout the fundamental domain, and the
    reconstruction is accurate to `1.8e-11`.
  - Under the flag the force routines return the *raw* tangent: the stereographic
    pushforward is deliberately skipped, since the drift no longer happens in the chart.
    This supersedes the P1-2a correction *for that build only*.
  - Validation: `examples/pds_tests/test_intrinsic_kinematics.cc`, 9/9 checks at machine
    precision (exp map preserves `|q|`, `U.q`, `|U|`; matches the analytic great circle;
    `exp(h1) o exp(h2) == exp(h1+h2)`; closed-loop transport is the identity; isometries
    commute with the exp map, so wrapping cannot bend a trajectory). Free-particle
    great-circle deviation improves from `3.5e-04` to `2.3e-09` rad on a ~100 Mpc
    trajectory (5 orders of magnitude), while the two integrators agree on the final
    position to 0.1% (375.84 vs 376.24 Mpc).
  - **Fixed during review: the closing half-kick was being lost.** KDK applies kick(h/2),
    drift(h), kick(h/2); the *second* kick (`step.cc`, the `errmax` loop) acted on `v[]`,
    which under `PDS_INTRINSIC` is a derived buffer that
    `pds_sync_stereo_for_output()` overwrites from `U` — so half of every step's
    acceleration was silently discarded. Two-particle convergence came out exactly 2x low
    (`dd = 0.015` vs `0.027 Mpc`). The closing kick now acts on `U`; `calculate_init_h()`
    likewise now pushes the raw tangent into the chart before comparing it with `v[]`,
    which it previously mixed. Both fixes are `#ifdef`-guarded, so the default path is
    byte-identical.
  - **The suite now passes 9/9 against the intrinsic build too**
    (`PDS_TEST_EXTRA_OPT=-DPDS_INTRINSIC python3 run_tests.py`), including Test 6's force
    cross-validation against the raw-tangent convention (`5.99e-11`).
  - **Validated against Gadget4 at production size (2026-08) - and it buys nothing here.**
    A full 1200 Mpc / 256^3 grid-IC run to z=0 with `-DPDS_INTRINSIC` (5.22 h, +5.2%
    runtime) was compared with the stopgap v2 run and the untouched `gadget256_flat`.
    Because the (r/R)^2 error is ~0 at the domain centre and largest at the boundary, a
    global P(k) averages it away; the discriminating statistic is the **radial slope of
    counts-in-cells sigma^2(PDS)/sigma^2(Gadget)** (15 Mpc cells, shells kept inside the
    491 Mpc face inradius, errors from 8 independent octants):

    | run | slope per 1000 Mpc |
    |---|---|
    | v1 (pre-review) | -0.177 +- 0.062 |
    | v2 (t0 force fix, stopgap) | -0.063 +- 0.062 |
    | intrinsic | -0.057 +- 0.061 |

    The intrinsic integrator changes the residual radial bias by `-0.007 +- 0.087`, i.e.
    **0.1 sigma - nothing**. Particle-level: intrinsic vs stopgap differ by a median
    **0.0054 Mpc = 0.1% of the interparticle spacing**, against **0.387 Mpc (8.2%)** for
    the t0 force fix - a factor 70.
  - **Conclusion: the roadmap's premise for Phase 1 was wrong.** The residual `(r/R)^2`
    error was attributed to the flat drift, but it actually lived in the **force->chart
    mapping** - the `Omega` factor (v2.2.3.0) and the `t0` term (P1-2a). Those are already
    in the default build and did the real work (v1->v2 flattened the slope by -0.113, ~17x
    anything the drift contributes). This is physically sensible: the drift error per step
    is `O((v*h/R)^3)` and StePS's adaptive timestep keeps `v*h/R` tiny, so curvature bites
    in how the force is mapped into the chart, not in how the step is taken along it.
  - **Keep the flag, keep it off.** The implementation is correct and passes 9/9; it is
    simply not the bottleneck at `R_curv = 3100 Mpc`. It is the right foundation if the
    code is ever pushed to large `r/R` (smaller `R_curv`, or a survey-scale box where
    `r/R -> 0.3+`), where the drift error grows as `(r/R)^3`.
  - Tier B (re-running all three production sims intrinsically, ~15 h / ~26 GPU-h) was
    **not** done: on this evidence it would reproduce the existing results to ~0.1% of a
    particle spacing.
  - Open, and larger than anything above: PDS **over-clusters Gadget4 by 1.4-2.9%** at
    z=0, radius-independent and therefore not an integrator effect. Traced below.
- `PDS_R_CURV` vs background-cosmology consistency check (`read_paramfile.cc`). A PDS is
  the quotient of a **closed** S^3, so a self-consistent background needs `Omega_k < 0`
  with `R_curv = (c/H0)/sqrt(|Omega_k|)`. The shipped `PDS_test.param` has
  `Omega_m + Omega_lambda + Omega_r = 1` exactly (`Omega_k = 0`) alongside
  `PDS_R_CURV = 3100 Mpc`, which would require `Omega_k = -2.04`. Now warns loudly (and
  reports the implied value) but continues — running the topology on a flat background is
  a legitimate numerical experiment, it just must be deliberate.
- **Test 9 — "IC quaternions govern the run"** (`examples/pds_tests/run_tests.py`). The old
  check only grepped the log for `Reading /PartType1/Quaternions`, which a run that then
  discarded them still prints. Test 9 writes a deliberately inconsistent IC (Coordinates
  and Quaternions describing configurations 260 Mpc apart) and asserts which one the
  integrated state follows: post-fix it tracks the quaternions to `5.3e-15 Mpc`; against
  the pre-fix code it tracks the decoy Coordinates to `4.8e-06 Mpc` and fails.
- **Test 6 now exercises the softening kernel.** Its 50 particles were spread widely enough
  that no pair was ever closer than `beta`, so the entire softened branch was dead code as
  far as the test was concerned — which is how a discontinuous kernel survived there. It
  now adds 14 close companions straddling the spline's two regions and both joins, uses the
  real mass-dependent per-pair `beta`, and reads positions/masses back from the written IC
  (they are stored float32; using the in-memory float64 arrays made close pairs disagree at
  `1e-4` for reasons that had nothing to do with the force kernel). Max rel err went from
  `4.76e-07` to **`5.98e-11`** while covering 94 sub-beta image pairs.
- `tests/test_s3lpt.py` (5 tests) and `tests/test_stereo_pushforward.py` (3 tests). All 8
  fail against the pre-fix code. The pushforward tests pin the Jacobian without running a
  simulation — the check that would have caught the dropped `t_0` term, and which also
  guards the documented error magnitude from drifting again.

### Removed / infrastructure
- Untracked four accidentally-committed artifacts: a compiled `pds_bh_prototype` (ELF),
  a `.pyc`, a Jupyter checkpoint (1.8 MB), and a broken absolute `examples/ic` symlink
  (`/v/scratch/astro/...`). `.gitignore` extended accordingly; `examples/ic` is now a
  machine-local symlink users create themselves.
- De-hardcoded the validation workflow: `OUT_BASE` defaulted to a nonexistent
  `/v/csabai/...` path (so `run_tests.py` could not run at all outside one machine) — now
  repo-relative and overridable via `PDS_TEST_OUT_BASE`. Compiler is auto-detected
  (`CXX` / conda / `g++`) instead of hardcoded to `x86_64-conda-linux-gnu-c++`, and
  Test 7 rewrites the example's `IC_DIR` key by regex rather than matching one literal
  path that would silently no-op if the example changed.
- Corrected the error magnitude documented for the force projection in
  `docs/PDS_guide.md`: it claimed ~30% (from an incorrect `sin(|q_1:4|)` scaling), and the
  review estimated ~7%; the measured worst case is **4.9%**.

### Note on build directories
`build/` is shared by every makefile variant, and objects are **not** rebuilt when the
`-D` options change. Building e.g. `PDS-LinuxGCC-Makefile` and then
`PDS-Linux_CUDA-Makefile` links a `main.o` compiled without `-DUSE_CUDA`, leaving
`n_GPU = 0` and crashing with an integer divide-by-zero in `forces_pds_cuda`, or failing
to link with `undefined reference to THETA`. Always `make -f <makefile> clean` when
switching variants.

## [Unreleased] - PDS/Gadget softening-convention mismatch (2026-08)

Chasing the residual PDS-vs-Gadget offset left over after the code review.

### Where it is NOT

- **Not the ICs.** Counts-in-cells sigma^2 measured in the domain-fitting cube (half-width
  280 Mpc, so the cube *corners* at 485 Mpc stay inside the 491 Mpc face inradius) gives a
  PDS/Gadget ratio of **1.0009 at z=30**, with the particle counts matching to one particle
  (1728000 vs 1727999). The offset develops dynamically from z~5 onward, reaching ~1.03 at
  z=0.
  - Measurement trap worth recording: a first attempt used a half-width 450 Mpc cube, whose
    corners reach 779 Mpc - far outside the domain - and the vacuum corners inflated the
    PDS variance to a spurious **7.7x**. Any cube used for PDS statistics must satisfy
    `half*sqrt(3) < 0.1584*R_curv`.
- **Not the integrator** (see the intrinsic-integrator validation above: 0.1 sigma).

### What it is

The two codes parameterise gravitational softening differently and the numbers are **not
interchangeable**:

| | parameter | force Newtonian beyond |
|---|---|---|
| StePS | `PARTICLE_RADII`; pair `beta = SOFT_LENGTH[i]+SOFT_LENGTH[j]` is the **full spline support** | `beta` = 2*`PARTICLE_RADII` |
| Gadget-2/3/4 | `SofteningComoving` = the **Plummer-equivalent** `eps`; spline support is `2.8*eps` | `2.8*eps` |

Matching prescription: **`PARTICLE_RADII = 1.4 * eps_gadget`**.

For the 1200 Mpc / 256^3 flagship pair, Gadget4 used `eps = 0.101555` Mpc/h = 0.15 Mpc
(support **0.42 Mpc**) while StePS used `PARTICLE_RADII = 0.1` (support **0.20 Mpc**) -
StePS was resolving pairs **2.1x closer**. The matched value is `PARTICLE_RADII = 0.21`.

Measured z=0 sigma^2 ratio to Gadget4 (mean +- s.e.m. over 8 octants, shot noise
subtracted, cube half-width 280 Mpc):

| cell | v1 (floor softening) | v2 (spline softening, current) |
|---|---|---|
| 40 Mpc | 0.9944 +- 0.0047 | 1.0141 +- 0.0044 |
| 25 Mpc | 0.9997 +- 0.0052 | 1.0187 +- 0.0061 |
| 15 Mpc | 1.0073 +- 0.0078 | 1.0291 +- 0.0084 |

### Tested directly - and the softening is NOT the cause

A matched-softening run (`PARTICLE_RADII = 0.21`, i.e. support 0.42 Mpc = Gadget's, same
IC, default build, 1.97 h / 1966 steps - notably faster than the 5.22 h / 4996 steps at
`PR = 0.1`, since the larger softening permits larger timesteps) gives results
**indistinguishable** from `PR = 0.1`:

| cell | `PR=0.1` (support 0.20 Mpc) | `PR=0.21` (support 0.42 Mpc) |
|---|---|---|
| 40 Mpc | 1.0141 +- 0.0044 | 1.0133 +- 0.0044 |
| 25 Mpc | 1.0187 +- 0.0061 | 1.0189 +- 0.0060 |
| 15 Mpc | 1.0291 +- 0.0084 | 1.0289 +- 0.0087 |

**Doubling the softening changed nothing**, so the convention mismatch - though real, and
worth documenting - does not explain the offset. In hindsight the reasoning that pointed at
it was wrong: the measurable cells (15-40 Mpc) are 35-200x larger than the softening scale,
so softening cannot affect them, and the apparent "grows toward smaller cells" trend is not
a softening signature at these scales.

### Remaining candidates (unresolved)

With the ICs, the integrator, and the softening all excluded, the residual **~1.4-2.9%
over-clustering of PDS relative to flat Gadget4 at z=0** is most likely one of:

1. **A genuine S^3 effect.** The background-compensated kernel on S^3 departs from flat
   `1/r^2` at `O((r/R)^2)`, which over the 500 Mpc analysis region at `R_curv = 3100 Mpc`
   is ~2.6% - the right order of magnitude. If so this is physics, not a defect.
2. **Force-scheme difference.** StePS uses pure Barnes-Hut (`theta = 0.3`); Gadget4 uses
   TreePM with `PMGRID = 256` (mesh cell 4.7 Mpc). Different accuracy at intermediate
   scales.

**Flat control: run, and it could not settle the question - because StePS's own T^3 path
is far less accurate than its PDS path.** A fresh StePS T^3 run (128^3, 1200 Mpc, CPU
Barnes-Hut + Ewald, softening matched to Gadget at `PARTICLE_RADII = 0.42`, 7 h) was
started from the *same* IC as `gadget128_flat`. Measured with CIC on a deliberately
non-commensurate mesh, shot noise subtracted:

| epoch | P(k) ratio, k < 0.05 | median ID-matched abs(dx) |
|---|---|---|
| z ~ 30 | **0.995** | 0.014 Mpc |
| z = 5 | 0.673 | 4.47 Mpc |
| z = 1 | 0.774 | 7.98 Mpc |
| z = 0 | 0.741 | 9.00 Mpc |

The two runs start identical (0.995 at z~30, 0.1% of the interparticle spacing), which
validates the pipeline, and then **StePS-T^3 loses ~26-33% of its large-scale power by
z=5**. An independent, older run at the same resolution (`testCubic128`) reproduces this
to within ~1%, so it is reproducible and not a one-off. It is also consistent with the
roadmap's own note that `testCubic128` is "too slow and not accurate enough" to be a
reference.

Reproduce with the makefile already in the repo (no new makefile is needed - and note
CUDA+Barnes-Hut is rejected at runtime for non-PDS topologies, so the T^3 control must run
on CPU):

```bash
make -f Cubic-Linux_BH-Makefile BUILD_DIR=./build_t3 -j8      # CPU Barnes-Hut + Ewald
# IC: gadget128_flat/ic_mpch.hdf5 converted to StePS units (Mpc, 1e11 Msol)
# param: L_BOX 1200, IS_PERIODIC 2, PARTICLE_RADII 0.42, OUT_DIR must end in '/'
```

**Scope: this does NOT touch the PDS results.** The PDS force path (compensated S^3
kernel, exact 120-image sum) shares no code with the T^3 Ewald path, and PDS agrees with
Gadget4 to ~2% where T^3 is off by ~30%. Cause of the T^3 discrepancy not established;
candidates are `EWALD_INTERPOLATION_ORDER=2` (order 4 hits a duplicate-symbol link error,
see below), the BH opening angle, and `ACC_PARAM`. **Worth its own investigation, but it
is a pre-existing issue in a different code path and is not a regression from this work.**

### Net status of the ~2% offset

Excluded: **ICs** (identical at z=30), **integrator** (0.1 sigma), **softening** (direct
test, no effect). Not excluded, and not distinguishable with the tools to hand: a genuine
S^3 effect vs a StePS-vs-Gadget force-scheme difference. One piece of evidence leans
toward the latter - the excess is radius-independent (radial slope -0.063 +- 0.062,
consistent with zero), whereas a curvature effect should scale as (r/R)^2 and vanish at
the domain centre. **Recorded as an open question**; it does not affect any conclusion in
this changelog, all of which concern PDS-vs-PDS comparisons or survive a 2% shift.

Note the awkward corollary either way: the P2-2 spline fix, correct in itself, moved StePS
from *accidentally* agreeing with Gadget (v1: 0.994-1.007) to visibly over-clustering it
(v2). Two compensating errors, not one good agreement.

**No source change is implied** - the softening length is a run *setting*, and the
convention is now documented in `docs/PDS_guide.md`.

## [Unreleased] - re-run & verification after the code review (2026-08)

All three affected PDS simulations were re-run from their **original ICs** with the
corrected code (P1-2a force projection + P2-2 spline softening; `PDS_INTRINSIC` off), so
the comparison isolates the code changes. Gadget4 runs were not affected and were not
re-run. Restart-from-checkpoint fidelity was verified independently (restarting the
completed run from its own z=0.2 snapshot reproduced the continuous integration to
**0.001% of the interparticle spacing**).

### Result: every scientific conclusion survives

| quantity | v1 (pre-review) | v2 (corrected) |
|---|---|---|
| P(k), all three runs | - | **+2-3%**, shape unchanged |
| PDS/Gadget, 1200 grid, 0.02<k<0.2 | 0.995 +- 0.017 | **1.017 +- 0.018** |
| PDS/Gadget, 1200 glass, k>0.2 | 1.391 | 1.422 |
| halos, PDS grid (N / top-500 match) | 13614 / 0.98 | **13719 / 0.98** |
| halos, PDS glass (N / mass ratio) | 25460 / 2.18 | **25242 / 2.194** |
| PDS50 face cone (paired controls) | 0.768 | **0.810** |

- **"PDS matches Gadget4 to 1-2%" holds** - the ratio moves from 0.5% low to 1.7% high.
- **The PDS50 topology-locked face deficit holds** (~0.77-0.81 vs a face *excess* of 1.11
  in the flat T^3 control).
- **The 1200 Mpc glass anomaly is NOT a force bug.** Its ~1.8x halo abundance, ~2.2x
  matched masses and high-k excess are all unchanged by the corrections, which supports the
  "residual glass-load noise" hypothesis rather than a force-projection artifact.
- Attribution: the entire P(k) shift comes from the **t_0 force fix**, not the softening.
  Production runs use `ParticleRadi = 0.1 Mpc` so `beta ~ 0.2 Mpc` - only 4% of the
  interparticle spacing, affecting `k >~ 30 /Mpc`, far beyond the measured range. This is
  independently confirmed by the radial gradient in particle displacements
  (0.047 Mpc at the centre -> 0.469 Mpc at 500 Mpc, the `(r/R)^2` signature).

### Corrected: the I* wraparound "correction" was control noise

An earlier entry reported that augmenting PDS50 cutouts with all 120 I* images
strengthened the anisotropy from 0.801 to 0.849. **That was wrong**, on two counts:

1. **Wrong radius.** The stated trigger - cutouts reaching past the *face inradius*
   (0.1584 R_curv = 20.5 Mpc) - is not the right test. The fundamental domain extends to
   its **circumradius** (~25.3 Mpc, ~1.26x the inradius) and particles fill it out to
   there. The 50 Mpc cutouts reach 24.7 Mpc, i.e. still inside. Measured directly, I*
   images contribute **1 865 of 49.3 M cutout points (0.004%)** even with a deliberately
   generous 40 Mpc image mask, and the cone ratios are unchanged to three decimals for
   both v1 and v2.
2. **The 0.048 "improvement" was realization noise.** Absolute cone ratios carry a
   **+-0.05** systematic from control-rotation realization: the control is ~240 heavily
   overlapping cutouts of essentially one structure, converging to isotropy only as
   1/sqrt(N_rot). The same v1 data yielded 0.801, 0.741 and 0.768 in three runs that
   differed only in RNG ordering.

**Methodological rule going forward:** quote run-to-run *differences* only from **paired**
controls (identical rotation sequences for both members) and prefer NCTRL >= 6. Absolute
ratios should be quoted with the +-0.05 systematic. The face deficit itself is robust and
was never in doubt - only the claim about the wraparound correction's effect was wrong.

## [Unreleased] - tools, validation & analysis (2026-07)

### Added
- **Halo catalogs for the flagship runs** (pipeline in `/scratch/csabai/halo_catalogs{,50}/`,
  representative analysis in `tools/Visualization/Halo_catalogs_analysis.ipynb`). StePS_HF
  run on matched-frame cube cutouts of Gadget256/PDS-grid/PDS-glass (1200 Mpc) and the
  50 Mpc glass pair. Key conventions established:
  - **Matched frame**: PDS stereographic coords (Mpc, origin-centred); Gadget `x/h - L/2`
    (FFT-phase-correlation verified for both box sizes). WARNING: `np.mod(x+L/2,L)-L/2`
    looks like this shift but is a **no-op relabeling** that silently misaligns the frames.
  - **PDS mass de-conformalization**: matched-frame particle mass = m_native/Omega(r)^3 at
    the current position (grid load then equals Gadget's m_p to 0.06%; glass masses
    quantized to 32 levels because StePS_HF scans `np.unique(Masses)`).
  - **StePS_HF usage notes**: SO finder in PERIODIC mode on the cut box; SEARCH_RADIUS must
    exceed the largest R200b (else IndexError); ParticleIDs must be 0..N-1 (used as array
    indices); analysis cube must fit inside the dodecahedral domain.
  - Validation: Gadget/PDS-grid mass functions overlap; **98% of the top-500 PDS grid-IC
    halos match a Gadget halo within 5 Mpc** (median offset 2.4 Mpc, mass ratio 1.10).
    Open issue: the 1200 Mpc PDS *glass* run shows ~1.7x halo abundance / ~2x matched
    masses (consistent with its ~30% high-k P(k) excess) - not reproduced at 50 Mpc;
    suspect residual glass-load noise.
- **Small-domain (50 Mpc) experiment** with glass ICs: Gadget4 T^3 (`gadget50_glass`) vs
  PDS with R_curv = 3100/24 = 129.17 Mpc (`test50glass`), where the discrete S^3/I* modes
  (k_12 = 0.100, k_20 = 0.162 /Mpc) dominate the box - a deliberately topology-dominated
  test. New `T3-Linux_CUDA-GlassMaking-Makefile` (PERIODIC + GLASS_MAKING + Ewald; order-2
  interpolation only - order 4 hits a duplicate-symbol link error; run single-GPU) to relax
  a 64^3 periodic glass tile (tiled 4^3 -> 256^3; tiling leaves only a ~1.5x median excess
  at the tile harmonics at z=15, gone by z=0). The PDS glass was re-charted from the
  flagship relaxed glass by exact scale invariance (stereo coords x 1/24). Findings: the
  two runs share the IC fine structure (corr 0.9 at z=30) and physically decorrelate by
  z=0; PDS50 develops ~1.6x more small-scale power (topology-driven collapse). Notebook:
  `tools/Visualization/Gadget_vs_PDS_50Mpc_comparison.ipynb`.
- **Halo-environment stacking / anisotropy analysis** (Racz, Szapudi, Csabai & Dobos 2021
  octahedral-group method; notebook `tools/Visualization/Halo_stacking_anisotropy.ipynb`,
  full scripts in `/scratch/csabai/stack3d/`):
  - grid-IC lattice memory: real-space cross-hatch + >2.2x lattice-phase modulation at
    z=15, erased near halos by z~2; but the 48-op octahedral fold reveals a **+7-9%
    box-axis (face) density excess re-emerging at z=0** in both grid-IC runs - collapse
    axes stay grid-aligned. Identical in T^3 and PDS => grid-IC artifact, not topology.
  - glass loads: no lattice at any epoch; residual ~3% cubic imprint (suspect: cubic FFT
    mesh + CIC interpolation of IC generation).
  - PDS50 (topology-dominated box): genuine S^3/I* anisotropy - the face/2-fold-axis cone
    density is **0.77-0.81x** the isotropic control against a face *excess* of 1.11 in the
    flat T^3 run: halo environments align with the icosahedral eigenmode pattern.
    T^3 50 Mpc shows only few-% effects at r/L <~ 0.1.
    (Superseded numbers: this entry previously read "0.72x after the I* wraparound
    correction". See the 2026-08 correction below - the wraparound augmentation is a no-op
    for these cutouts, and absolute cone ratios carry a +-0.05 control-realization
    systematic.)

### Fixed
- `tools/Utils/inputoutput.py`: made the legacy `pygadgetreader`/`glio` imports optional
  and always uninstall the `past.translation` import hook afterwards - previously the hook
  intercepted ALL later imports (h5py, astropy) and crashed on compiled modules, so
  StePS_HF could not run in environments without the legacy readers.

## [v2.2.4.0] - 2026-06-21

### Added
- **Glass-making initial conditions for the PDS (S^3/I*) topology.** A grid-free,
  isotropic particle load for the dodecahedral domain, as an alternative to the
  rectangular stereographic grid (`test256disc`) whose lattice does not respect the
  S^3/I* topology.  Workflow:
  1. **Poisson load** (`stepsic`, new `TYPE = 'random'` for `GEOMETRY = 'pds'`): draws
     `NPART` points uniformly on S^3 and folds them into the fundamental domain via the
     I* group (`pds.wrap`) — equal-mass, no preferred directions.
  2. **Reverse-gravity relaxation** with a new glass-making build
     (`PDS-Linux_CUDA_BH-GlassMaking-Makefile`, and a direct-summation variant
     `PDS-Linux_CUDA-GlassMaking-Makefile`; both add `-DGLASS_MAKING` → `G = -1` and a
     separate `build_glass*/` dir).  Run with an **Einstein-de Sitter** background
     (`.param` `Omega_m = 1, Omega_lambda = 0`; new `EdS` cosmology in `stepsic`) so
     dark energy does not freeze the relaxation — per G. Racz's recommendation.  Use the
     **Barnes-Hut** binary (theta=0.30): direct summation is O(N^2) and impractical at
     N~6.3x10^6 (it stalls for hours per step), while BH relaxes the same load in ~18 min.
  3. **Build the IC** from the relaxed glass (`stepsic` `TYPE = 'glass'`), with the same
     Zel'dovich + discrete S^3/I* spectrum (n=12,20) as the grid run.
- **Validation (grid load vs glass load, identical otherwise; `test256disc` vs
  `test256glass`).** The glass is sub-Poisson (counts var/mean = 0.28 vs 1.0) with **no
  lattice Bragg peaks**: peak high-k P/shot is ~5x for the glass vs ~3x10^4 in the raw grid
  IC and ~2.5x10^3 at z=15.  The grid imprint washes out under non-linear growth by z~2, so
  the two runs agree at z=0 (low-k P(k) within 3%); the glass mainly cleans the *early-time*
  small-scale field.  (Separately: the large static **low-k "pedestal"** once seen in the
  PDS P(k) is a **survey-window/mask artifact** — an FFT analysis cube whose corners stick out
  past the dodecahedral domain into vacuum — not a grid artifact and not physics; measuring in
  a cube that fits inside the domain (`half ≤ ~350` Mpc) removes it entirely, so no
  first-snapshot subtraction is needed.)  See `docs/PDS_guide.md` and
  `tools/Visualization/Gadget_vs_PDS_comparison.ipynb`.

### Fixed
- **CRITICAL: multi-GPU force kernels left the last particles of each GPU frozen for
  large N.** All five CUDA force kernels (`ForceKernel_pds_bh`, `ForceKernel_pds`,
  `ForceKernel`, and the two `ForceKernel_periodic_z` launches in `forces_cuda.cu`) use a
  grid-stride loop `for(ii=tid; ii<n; ii+=stride)` with `stride = blockDim.x*gridDim.x`,
  but the launches passed **`n = nthreads = 32*mprocessors*BLOCKSIZE`** (the thread count)
  as the loop bound instead of the **per-GPU particle count `N_GPU`**.  Because
  `nthreads == stride`, the loop executes exactly once per thread and only the first
  `nthreads` particles of each GPU's range receive a force; the remaining
  `N_GPU - nthreads` particles get F = 0 (stale device buffer) and never evolve.
  - **Trigger:** only when `N_GPU > nthreads`, i.e. (4× H200, BLOCKSIZE=256 →
    nthreads ≈ 1.08x10^6) above ~**4.3x10^6 particles on 4 GPUs**.  Smaller runs were
    fully covered and are unaffected (e.g. the 7.8x10^5-particle PDS validation and the
    v2.2.3.0 Gadget4 comparison stand).
  - **Symptom:** a strided/banded pattern where ~the last 1/n_GPU of *each* GPU's
    (ID-ordered, hence spatial) range stays at its IC position — visible as smooth,
    unevolved bands in z=0 renders of a 6.3x10^6-particle run.  Diagnosed from the
    per-ID-block median displacement showing a clean period-`n_GPU` pattern, identical in
    independent runs (so a code, not IC, bug).
  - **Fix:** pass `N_GPU` (the GPU's particle count) as the kernel loop bound so the
    grid-stride covers every particle; also changed the in-loop `if(i>ID_max) return;`
    to `break;` (a `return` would be wrong mid-grid-stride).  Verified: re-running the
    6.3x10^6-particle PDS IC now gives a *uniform* per-ID-block displacement (min/max = 1.0
    across 32 blocks) with no frozen bands.

## [v2.2.3.0] - 2026-06-18

### Fixed
- **PDS (S^3/I*) structure over-growth — missing stereographic conformal factor
  in the force→drift mapping.** PDS runs grew structure far too fast (long-standing
  complaint; the dodecahedral run collapsed into bright clusters while the matched
  flat T^3 run showed only a mild cosmic web). Root cause: `forces_pds()` /
  `forces_pds_bh()` return the **physical geodesic** acceleration on S^3
  (`pds_green_compensated ≈ G M / r^2_geodesic`, correct magnitude and direction),
  but `step.cc` drifts the particle in the **stereographic coordinate** chart
  (`x_stereo += v*dt`) treating it as flat. The stereographic map is conformal,
  `ds^2 = Omega^2 dx_stereo^2` with `Omega = 2R^2/(R^2 + r^2)`, so the correct
  coordinate acceleration is `a_phys/Omega`, not `a_phys`. With `Omega ≈ 2` across
  the fundamental domain the peculiar gravity was **~2x too strong everywhere**,
  turning the linear growth `D ∝ a` into `D ∝ a^~1.6` — a runaway over-growth.
  - **Fix:** divide the PDS force on each target particle `i` by
    `Omega(r_i) = 2R^2/(R^2 + r_i^2)`. Uses the identity **`Omega = 1 + q0`**
    (`q0` = quaternion scalar part of particle `i`, since
    `q0 = (R^2 - r^2)/(R^2 + r^2)`), so the correction is a single multiply by
    `1.0/(1.0 + qi[0])`. Applied in all four PDS force paths before the force is
    stored: `forces_pds()` (CPU direct) and `forces_pds_bh()` (CPU BH) in
    `forces.cc`; `ForceKernel_pds` (CUDA direct) and the CUDA BH kernel in
    `forces_cuda.cu`. The 4D-tangent force *direction* was already correct and is
    unchanged.
  - **Confirmation:** a standalone reimplementation gives `F_code/F_correct =
    Omega(r_i)` exactly (2.00 at the domain centre, 1.93 at the edge), direction
    dot-product 1.000.
  - **Validation** (test128, N = 7.8x10^5, R_curv = 3100 Mpc, z = 31→0, 4x H200,
    ~15 min): measured against the matched flat T^3 StePS run (same IC realization,
    `testCubic128`) with the coordinate-invariant geodesic displacement `R*chi`.
    The PDS growth relative to linear theory went from a runaway ramp
    (`sim/lin` 1.00→1.16→1.46→1.72→1.93 over z = 15→2) to **flat ≈1.0**
    (1.00→1.00→0.99→0.98→0.98) — i.e. it now tracks linear growth. Counts-in-cells
    over-clustering at z = 0 dropped **~16x** (sigma^2 ratio vs flat 49.5 → 3.2; the
    small residual is a cross-geometry resolution/shot-noise baseline visible even
    at z = 10 where the physics is linear, not a force error).
  - **Note:** this captures the dominant (constant-Omega) effect exactly; the
    leftover spatial variation of Omega (~3.5% across the domain) and the
    velocity-dependent Christoffel terms remain part of the first-order integrator
    approximation (see PDS guide "Known Limitations"). Earlier component tests
    (force = M/r^2, Hubble drag = a^-2, T^3 mode = Gadget4) were all correct; the
    defect was confined to the PDS coordinate mapping.

## [v2.2.2.0] - 2026-06-15

### Added
- **EXPERIMENTAL: Barnes-Hut (octree) tree force for the PDS (S^3/I*) topology**
  on both **CPU and GPU**, enabled with `-DUSE_BH=<theta>` in a PDS build.
  Direct 120-image summation is O(N^2) and becomes prohibitive at research-grade
  N (a 7.8x10^5-particle run would take weeks); the tree force is O(N log N).
  - CPU: `forces_pds_bh()` in `forces.cc` (`PDS-Linux_BH-Makefile` → `StePS_BH`).
  - GPU: `forces_pds_bh_cuda()` / `ForceKernel_pds_bh` in `forces_cuda.cu`
    (`PDS-Linux_CUDA_BH-Makefile` → `StePS_CUDA_BH`). The octree is built and
    flattened on the host (DFS preorder + per-node "escape"/skip pointer) and
    walked iteratively on the GPU — no recursion, no per-thread stack. One GPU
    thread per field particle does the 120 per-image stackless walks; the I*
    table is copied to each device's `__constant__` memory; the field-particle
    range is split across GPUs. (`main.cc` no longer rejects USE_BH+USE_CUDA for
    the PDS topology; other topologies still have no GPU tree.)
  - **GPU port validation (test7b):** GPU-BH reproduces CPU-BH to printed
    precision at every snapshot through z=0 (RMS Δx = 0), 4-GPU = 1-GPU
    bit-identical, and GPU-BH vs exact density cross-correlation 0.935 (z=0).
    Force eval 0.025 s (1 GPU) vs 0.25 s (16 CPU cores); the full test7b run
    takes < 60 s on one GPU. At N = 7.8x10^5 the GPU tree force is ~0.35 s/eval
    (4 GPUs, ~9.4x10^5 nodes) vs an extrapolated ~46 min/eval for exact direct
    on the same GPUs (~10^4x), turning a multi-week run into minutes.
  - **Key design point:** the octree opening test is evaluated *per I* image*,
    in the S^3 geodesic metric. A node that is far in the identity image can be
    the *physical neighbour across a shared dodecahedral face* in another image,
    so a single identity-image opening test lumps near images into monopoles and
    gives forces 10-45x too large (and divergent as theta shrinks). Descending
    the tree once per image (genuinely far images terminate shallow) fixes this;
    node angular size uses the conformal factor of the stereographic map,
    `ang = 2R*nodesize/(R^2 + r_C^2)`, preserved by I* isometries.
  - Reuses the shared `OctreeNode`/`insert_particle` scaffolding (extended with a
    mass-weighted `soft` field). `forces_pds()` dispatches to the tree force when
    `USE_BH` is defined; both call sites (initial force, integrator) pick it up.
  - New CPU build file `PDS-Linux_BH-Makefile` (stepsic conda toolchain) →
    `build/StePS_BH`.
  - **Validation** (test7b, N=12 240): standalone force check converges to exact
    (rel. err 3x10^-5 as theta->0); at theta=0.3 the end-to-end z=31->0 run
    matches the exact 120-image run with density cross-correlation 0.93 (z=0) /
    0.95 (z=1) and P(k) agreement to ~5% across all scales, no NaNs. Runtime
    78 s (16 CPU cores) vs ~450 s for the exact run on 4x H200 GPUs.
    Tooling: `examples/pds_tests/pds_bh_prototype.cc` (standalone θ-vs-accuracy
    validator) and `pds_bh_prototype_plot.py`.
  - **Momentum-conservation audit (passed):** Barnes-Hut monopole forces break
    Newton's third law in general, but the exact PDS force is itself not
    perfectly pairwise-antisymmetric (projected-to-3D compact-S^3 force; net
    |S|/sum|M a| ~ 0.17% early, ~0.66% at z=0). The tree net force stays within
    0.91-1.04x that exact baseline at every epoch and converges to 1.0x as
    theta->0 — no momentum drift of its own. The integrated run's bulk momentum
    tracks the exact run and decreases from the IC value rather than
    accumulating. Audit mode: `pds_bh_prototype <snap> <radii> 0 momentum`.
  - **GPU performance:** the host octree build was found to dominate the GPU
    force step (60-78% at N=7.8x10^5, GPUs idle meanwhile — the cause of the
    fluctuating GPU utilisation / low power). Now built with a parallel Morton
    (Z-order) sort into the flattened array (`__gnu_parallel::sort` + OpenMP),
    cutting the build from ~0.30 s to ~0.06 s at that N (~5x); the tree is
    structurally identical (same node count, density cross-correlation 0.935 vs
    exact unchanged, trajectories match the recursive build to ~1e-10).
    Device buffers are now persistent (no per-step cudaMalloc/cudaFree) and the
    tree is staged through pinned host memory. NOTE: the GPU build's tree
    construction is parallel, so it too needs `mpirun --bind-to none` to avoid
    OpenMP threads binding to one core.
  - **Not yet done:** quadrupole terms (would tighten the per-particle error
    tail); overlapping/​hiding the remaining host build behind GPU work (or a
    full GPU-side tree build) to push utilisation higher; Morton-ordering the
    field particles to cut warp divergence in the kernel.

### Fixed
- `ewald_space.cc` did not include `<string.h>`, so any `-DPERIODIC` (T^3) build
  failed to compile (`'memset' was not declared`). Added the include.

### Notes
- **Both Barnes-Hut builds must launch with `mpirun --bind-to none`** (or run the
  binary directly without mpirun). With the default binding, `mpirun -np 1`
  confines all OpenMP threads to a single core. This slows the CPU tree force
  ~18x (4.8 s vs 0.25 s per force evaluation on this node), and from v2.2.2.0 it
  also throttles the **GPU** build's parallel host-side octree construction.
  Recommended launch:
  ```
  # CPU:  OMP_NUM_THREADS=16 mpirun --bind-to none -x OMP_NUM_THREADS -np 1 ./build/StePS_BH <param>
  # GPU:  OMP_NUM_THREADS=32 mpirun --bind-to none -x OMP_NUM_THREADS -np 1 ./build/StePS_CUDA_BH <param> <n_gpu>
  ```
  The exact-direct builds (`StePS`, `StePS_CUDA`) do not need it.

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
