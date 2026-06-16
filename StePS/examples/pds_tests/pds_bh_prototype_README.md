# PDS Barnes-Hut prototype (step 1 feasibility)

Standalone CPU validator for a Barnes-Hut tree force on the Poincaré
Dodecahedral Space (S³/I\*) topology, measured against the exact 120-image
summation used by `forces_pds()` / `ForceKernel_pds`. It does **not** touch the
StePS build.

## Files
- `pds_bh_prototype.cc` — reads a StePS HDF5 snapshot, builds an octree, and
  compares exact vs BH forces on a random sample of field particles over a
  sweep of opening angles θ.
- `pds_bh_prototype_plot.py` — runs the binary on one or more snapshots and
  plots accuracy and speedup vs θ (`pds_bh_tradeoff.png`).

## Build & run (inside the stepsic conda env)
```bash
conda activate stepsic
g++ -O3 -std=c++17 -fopenmp pds_bh_prototype.cc \
    -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lhdf5 \
    -Wl,-rpath,$CONDA_PREFIX/lib -o pds_bh_prototype
OMP_NUM_THREADS=16 ./pds_bh_prototype <snapshot.hdf5> [particle_radii] [n_sample]
python3 pds_bh_prototype_plot.py <snap_z0.hdf5> <snap_z3.hdf5> --radii 10
```

## Key design result

A naive tree walk with a single (identity-image) opening test **fails
catastrophically** for PDS: the BH force is 10–45× too large and *diverges* from
exact as θ shrinks. Reason — at the dodecahedral domain boundary a particle's
nearest physical neighbour across a shared face is one of the 119 I\* *images*,
sitting at small geodesic distance χ. Lumping such a near image into a monopole
is wildly wrong.

**Fix:** evaluate the opening test **separately for each of the 120 images**, in
the S³ geodesic metric. We descend the tree once per image using
χ(qᵢ, g·q_C) for that image. Genuinely far images (most of the 119) terminate
at shallow levels, so the cost is far below 120 deep walks — only the identity
and the ~12 face-adjacent images descend deep. Node angular size on S³ uses the
conformal factor of the stereographic map,
`ang = 2R·nodesize/(R²+r_C²)`, preserved by I\* isometries.

With the per-image test the BH force **converges to exact** (rel. error
≈ 3×10⁻⁵ at θ=0.05, i.e. round-off + COM-conversion level), confirming
correctness.

## Measured tradeoff (test7b, N=12 240, z=0)

| θ | mean \|ΔF/F\| | 99th pct | speedup @ N=12 240 |
|------|------|------|------|
| 0.7  | 3.6% | 22%  | ~1100× |
| 0.5  | 1.6% | 8.7% | ~540× |
| 0.35 | 0.52% | 2.9% | ~230× |
| 0.25 | 0.21% | 1.1% | ~110× |
| 0.15 | 0.056% | 0.35% | ~36× |

- **θ ≈ 0.3–0.35** gives ~0.5–1% mean force accuracy (GADGET-class), the
  recommended operating point.
- Accuracy improves as structure forms (the near-neighbour clustering force,
  which BH captures best, comes to dominate); the early-time, smoother field is
  the harder case.
- The speedup column is at this small N; since exact ∝ N and BH ∝ log N, the
  advantage grows ~ N/log N. Extrapolated to test128 (N≈7.8×10⁵, 64× larger)
  the per-step force speedup at θ=0.35 is O(10⁴), turning a ~weeks-long direct
  run into hours.

## Limitations / next steps (toward production)
- COM is taken as the stereographic centroid mapped to a quaternion; for large
  cells the intrinsic S³ (Karcher) mean would be marginally more accurate
  (negligible at the few-% level targeted here).
- Monopole only. A quadrupole term would tighten the 99th-percentile tail.
- Softening on accepted internal nodes uses a mass-weighted node value; close
  interactions (where softening matters) are always resolved to leaves, so this
  is exact where it counts.
- **Step 2 (done, v2.2.2.0):** integrated as `forces_pds_bh()` in `forces.cc`
  behind `-DUSE_BH`, built by `PDS-Linux_BH-Makefile` (`build/StePS_BH`).
  End-to-end test7b z=31→0 at θ=0.3 matches the exact run: density-field
  cross-correlation 0.93 (z=0) / 0.95 (z=1), P(k) within ~5%, no NaNs; 78 s on
  16 CPU cores vs ~450 s for the exact run on 4× H200. Operational note: run
  with `mpirun --bind-to none` (or no mpirun) or all OpenMP threads bind to one
  core (~18× slower).
- **Momentum audit (passed):** `./pds_bh_prototype <snap> <radii> 0 momentum`
  reports the net force |S|/Σ|M a| over all particles for the exact force and
  for BH at several θ. The exact PDS force has an inherent ~0.17% (early) to
  ~0.66% (z=0) net-force imbalance (compact-S³ projected force is not perfectly
  pairwise-antisymmetric); BH stays within 0.91–1.04× of that at every epoch and
  → 1.0× as θ→0, so it adds no momentum drift. The integrated run's bulk
  momentum tracks the exact run and shrinks from the IC value.
- **Step 3 (done, v2.2.2.0):** CUDA port — `forces_pds_bh_cuda` /
  `ForceKernel_pds_bh` in `forces_cuda.cu`, built by `PDS-Linux_CUDA_BH-Makefile`
  (`build/StePS_CUDA_BH`). Host-built flattened octree (DFS preorder + escape
  pointers), stackless per-image traversal on the GPU. Reproduces the CPU run to
  printed precision (RMS Δx = 0 through z=0), 4-GPU = 1-GPU bit-identical,
  density cross-correlation 0.935 vs exact. Force eval 0.025 s (1 GPU) vs 0.25 s
  (16 CPU cores); at N=7.8×10⁵ ~0.35 s/eval on 4 GPUs vs ~46 min for exact
  direct (~10⁴×).
