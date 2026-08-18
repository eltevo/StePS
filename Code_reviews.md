# Code reviews

## 1st review (2026-07)

An independent review raised 8 findings. All 8 were verified against the code and by
running the suites: **no false positives**. Two P1s were re-discoveries of limitations
already documented in `StePS/docs/PDS_guide.md`; two severity figures were overstated (one
of them also wrong in our own docs); two P2s turned out to be *worse* than reported. The
highest-value finding was P1-2b — the force regression test had been silently dead, which
is why P1-2a survived unnoticed.

| id | verdict | status |
|---|---|---|
| P1-1 | confirmed; already documented | in progress (intrinsic S³ integrator) |
| P1-2a | confirmed; severity 7% → **4.9%** | fixed, all 4 sites |
| P1-2b | confirmed; ~50% exactly | fixed |
| P1-3 | confirmed | fixed (warning) |
| P1-4 | confirmed exactly (3/6, 16/17) | fixed |
| P2-1 | confirmed; **worse** than reported | fixed |
| P2-2 | confirmed | fixed |
| P2-3 | confirmed; severity reduced | fixed |
| P2-4 | confirmed; **blocking**, not just non-portable | fixed |

---

## Findings

### [P1-1] Flat Cartesian drift

> The PDS integrator still performs a flat Cartesian drift. [step.cc (line 194)](StePS/src/step.cc:194) updates x += v h; it does not follow an S³ great-circle/geodesic or include the stereographic metric's connection terms. Face wrapping keeps particles in the domain but does not correct the trajectory between faces.

**Confirmed.** Already documented as a Known Limitation with "Future work: replace the
drift phase with the geodesic exponential map on S³", i.e. Phase 1 of `plan.md`. Not a new
discovery — but the review is right that after the P1-2a fix this is the dominant remaining
error, ~10% at r/R ≈ 0.3 from the residual Ω variation plus velocity-dependent Christoffel
terms.

**Action:** being implemented as the intrinsic S³ integrator (see CHANGELOG).

---

### [P1-2a] Force not transformed through the full stereographic Jacobian

> The S³ force is not transformed through the full stereographic Jacobian. [forces.cc (line 1615)](StePS/src/forces.cc:1615) discards the tangent's t0 component and merely divides t[1:4] by 1+q0. The branch already uses the correct full Jacobian for velocity transformations in pds_stereo_vel_transform(). The omission produces up to roughly 7% radial error near the domain's outer radius.

**Mechanism confirmed exactly.** The exact pushforward of a tangent `t` at `q` is

```
dx_i = ( R·t_{i+1} − x_i·t₀ ) / Ω,      Ω = 1 + q₀ = 2R²/(R²+r²)
```

and the code kept only the first term. The review's sharpest observation is that the
correct full Jacobian **already existed in our own tree** — `pds_stereo_vel_transform()`,
`src/pds_group.h` Step 3 — making the fix a transcription of working code rather than a new
derivation. That materially lowered the cost.

**Severity adjusted.** Measured over the actual domain (`r ≤ 0.1584·R`):

| quantity | median | max at boundary |
|---|---|---|
| force magnitude error | 0.8 % | **4.9 %** |
| force direction error | 0.16° | 1.4° |

So ~5%, not ~7%. Our own `PDS_guide.md` was worse: it claimed **~30%**, from an incorrect
`∝ sin(|q₁:₄|)` scaling. The true scaling is `(r/R)²`. Both figures corrected.

**Action:** fixed at all four sites (`forces.cc` direct + BH, `forces_cuda.cu` direct + BH),
written as `x_i/R = q_{i+1}/Ω` so it depends only on `PDS_Q`, never on `x[]` being in sync.
C++ matches an independently derived NumPy formula to `1.15e-13`; CPU vs CUDA agree to
`5.6e-12`. Existing runs shift by ~1% median / 4.4% max.

---

### [P1-2b] Force regression test never updated — *highest-value finding*

> The repository's independent force test confirms an unresolved inconsistency: C++ versus NumPy differs by approximately 50%. Commit 8f11a8c added the 1/Ω factor, but the reference test was never updated. Therefore, no independent test currently validates the new scaling.

**Confirmed and quantified.** Test 6's NumPy reference omitted the `1/Ω` factor that
`forces.cc` applies. For the test's own configuration Ω ∈ [1.955, 2.000], so the metric
`|F_cpp − F_py|/|F_py| = q₀/(1+q₀)` sat at **48.9–50.0%** — matching the reviewer's estimate
precisely. The only independent validation of the PDS force had been dead since the Ω
commit, which is exactly why P1-2a went unnoticed.

**Action:** reference corrected and extended to the full pushforward; now passes at
`4.76e-07` against a `1e-6` gate.

**Gap found while fixing it:** Test 6's 50 particles are spread widely enough that the
per-pair softening floor is *never* exercised, and its reference assumed a uniform softening
while StePS uses mass-dependent `SOFT_LENGTH[i] = cbrt(M[i]·const_beta)`. Closed alongside
P2-2.

---

### [P1-3] Background cosmology can be inconsistent with the geometry

> The background cosmology can be inconsistent with the simulated geometry. [read_paramfile.cc (line 755)](StePS/src/read_paramfile.cc:755) checks only that PDS_R_CURV > 0. The included example sets a finite S³ radius while Ωm + ΩΛ + Ωr = 1, giving Ωk = 0. A finite positively curved S³ background requires Ωk < 0 and a consistent curvature-radius relation. At minimum, this should be a loud validation warning.

**Confirmed.** `PDS_test.param` has `Ωm + ΩΛ + Ωr = 1` exactly. With `c/H0 = 4428 Mpc`, a
curvature radius of 3100 Mpc requires **Ωk = −2.04** (`Ωm + ΩΛ ≈ 3.04`) — nowhere near the
ΛCDM values in the same file. The topology is decorative with respect to the expansion
history.

We agree with "warning, not error": running the topology on a flat background is a
legitimate numerical experiment, and is what every production run to date has done. It just
has to be a deliberate choice.

**Action:** consistency check added, reporting implied Ωk and implied R_curv. Verified on
three cases — flat + finite R (warns), self-consistent closed (silent; round-trips
3100 → 3099.39 Mpc), closed but mismatched (warns with % offset).

---

### [P1-4] Discrete-spectrum IC path cannot run on current dependencies

> The advertised discrete-spectrum IC path cannot run with current allowed dependencies. [s3harmonics.py (line 93)](stepsic/s3harmonics.py:93) imports scipy.special.sph_harm, which is absent in the installed SciPy 1.18, while pyproject.toml allows all SciPy versions ≥1.8. Three of six harmonic tests fail. The geometry tests also use removed np.trapz, yielding 16/17 passes.

**Confirmed exactly, including both counts.** `sph_harm` was removed in SciPy 1.17 and
`np.trapz` in NumPy 2.4. Reproduced: `test_s3harmonics.py` 3 failed / 3 passed,
`test_pds.py` 1 failed / 16 passed.

**Action:** rather than pinning a ceiling, added a version-portable shim — `sph_harm_y`
where available, legacy argument order otherwise (verified bit-identical, max diff `0.0`) —
and `np.trapz` → `np.trapezoid` with fallback. Suites pass on SciPy 1.15/NumPy 2.2 *and*
SciPy 1.17/NumPy 2.4.

---

### [P2-1] S³ LPT tangent basis degenerates at coordinate-axis points — *worse than reported*

> The S³ LPT tangent basis degenerates at coordinate-axis points. [s3lpt.py (line 44)](stepsic/s3lpt.py:44) increments the output-basis index even when a candidate vector has zero norm. At the north pole, for example, it returns a zero basis vector and misses one physical gradient direction.

**Confirmed** — rank 2 instead of 3 at `q = e₀, e₁, e₂` (not `e₃`, whose zero candidate
falls outside the first three tried). Note `q = e₀` is the **stereographic origin**, i.e. a
particle at the box centre, so this is reachable by an ordinary grid load, not a
measure-zero curiosity. The dead `ok` mask in the source is the tell.

Because `displacement_field()` accumulates `Ψ = Σₐ (∇ₐf) êₐ`, a rank-2 frame makes
`Σₐ êₐêₐᵀ` a rank-2 projector and silently drops a gradient direction from the Zel'dovich
displacement.

**Worse than reported:** the problem is not confined to the exact axes. A first patch that
only fixed the degenerate slot still lost orthogonality to **1.0e-5** *near* the axes,
because the near-parallel candidate is a tiny difference of large numbers.

**Action:** rebuilt to discard the axis most nearly parallel to `q` (`argmax|q_k|`),
guaranteeing every retained candidate has norm ≥ 1/√2, plus a re-orthogonalization pass.
Machine precision everywhere; `Σₐ êₐêₐᵀ = I − qqᵀ` to `4.8e-16`. Bit-identical to the old
frame at generic points over 50k quaternions, so existing ICs are unaffected.

---

### [P2-2] PDS softening is a discontinuous distance floor

> PDS softening is a discontinuous distance floor rather than the existing smooth StePS kernel. [forces.cc (line 1587)](StePS/src/forces.cc:1587) gives a constant-magnitude force below χsoft, but exactly zero at coincidence. That discontinuity can increase integration error and two-body heating.

**Confirmed.** `chi_eff = max(chi, chi_soft)` holds |F| at `green(χ_soft)` all the way down
to χ → 0⁺, then the `chi < 1e-12` guard drops it to 0 — so the force does *not* go to zero
as particles approach, unlike the smooth spline kernel StePS uses on every non-PDS path
(`_soft_rebuild_cache`, the standard two-region Monaghan-style polynomial).

**Action:** replaced with that same spline kernel, applied to geodesic separation.

---

### [P2-3] Native quaternion IC data are read and then overwritten

> Native quaternion IC data are read and then overwritten. [step.cc (line 55)](StePS/src/step.cc:55) reconstructs q_in from Cartesian coordinates and replaces PDS_Q, even when /PartType1/Quaternions was present. The test only checks that the dataset was logged as read, not that its values governed the run.

**Confirmed** — `inputoutput.cc:1319` reads the dataset into `PDS_Q`, then `main.cc:1839`
calls `pds_wrap_ic()`, which rebuilds `q` from `x[]` and overwrites it.

**Severity reduced.** The stereographic map is a genuine bijection on S³ minus the antipode,
so the round-trip is mathematically lossless; the practical cost is float precision, not
wrong physics. But the reviewer's real point — the test asserts only that the dataset was
*logged*, not that its values were *used* — is correct, and fixing it removes a silent
precision loss.

**Action:** IC quaternions are honoured when present, with the Cartesian reconstruction as
fallback; test strengthened to assert the values actually govern the run.

---

### [P2-4] Validation/install workflow is machine-specific — *blocking, not just non-portable*

> The validation/install workflow is machine-specific. [run_tests.py (line 67)](StePS/examples/pds_tests/run_tests.py:67) hard-codes /v/csabai/... and an x86_64-conda-linux-gnu-c++ compiler. The repository also tracks a broken absolute examples/ic symlink, a compiled executable, a .pyc, and a notebook checkpoint.

**Confirmed, and worse:** `/v/csabai` does not exist on this machine at all, so
`run_tests.py` could not run *here* — not merely "on another machine". All four tracked
artifacts confirmed, including a 1.8 MB Jupyter checkpoint and an `examples/ic` symlink
pointing at a nonexistent `/v/scratch/astro/pds_tests/`.

**Action:** `OUT_BASE` is repo-relative and overridable via `PDS_TEST_OUT_BASE`; compiler
auto-detected (`CXX` → conda → `g++`); Test 7 rewrites the example's `IC_DIR` by regex
instead of matching one literal path that would silently no-op. Artifacts untracked,
`.gitignore` extended, `examples/ic` documented as a machine-local symlink.

---

## Issues found while addressing the review (not in the original report)

1. **Shared `build/` directory across makefile variants.** Objects are not rebuilt when the
   `-D` options change, so building `PDS-LinuxGCC-Makefile` then `PDS-Linux_CUDA-Makefile`
   links a `main.o` compiled without `-DUSE_CUDA` — leaving `n_GPU = 0` and crashing with an
   integer divide-by-zero inside `forces_pds_cuda`, or failing to link with
   `undefined reference to THETA`. Not a product bug, but a confusing trap that cost real
   debugging time. Always `make -f <makefile> clean` when switching variants.
2. **Test 6 never exercised the softening floor** (see P1-2b).

## Test coverage added

| file | tests | catches |
|---|---|---|
| `stepsic/tests/test_s3lpt.py` | 5 | rank-deficient / ill-conditioned tangent frames |
| `stepsic/tests/test_stereo_pushforward.py` | 3 | dropped `t₀`, dropped `1/Ω`, drifting error magnitude |

All 8 fail against the pre-fix code. The pushforward tests pin the Jacobian without running
a simulation — millisecond-scale, and exactly the check that would have caught P1-2a.
