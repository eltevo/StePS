# Visualization & analysis notebooks

Executable notebooks for the StePS PDS validation campaign (2026). Figures are embedded;
re-execution needs the simulation outputs under `/scratch/csabai/` (snapshots + halo
catalogs) and, where noted, the `stepsic` package on `sys.path`. Conventions (matched
frame, PDS mass de-conformalization, wraparound rule) are documented in
`../../StePS/docs/PDS_guide.md` and the CHANGELOG.

| notebook | contents | data dependency |
|---|---|---|
| `Gadget_vs_PDS_comparison.ipynb` | 1200 Mpc **2×2** — {Gadget4 T³, StePS PDS} × {grid load, glass load}: matched density slices (mass-weighted, de-conformalized), P(k) evolution in the domain-fitting cube (no window pedestal), P(k) ratios & growth, grid-vs-glass Bragg panel, and §6 separating the **load effect from the topology effect** | `gadget256_flat`, `gadget256_glass`, `test256disc_v2`, `test256glass_v2` |
| `Gadget_vs_PDS_50Mpc_comparison.ipynb` | 50 Mpc topology-dominated glass pair: matched slices (shared IC at z=30 → decorrelated by z=0), P(k), tiling signature | `gadget50_glass`, `test50glass_v2` |
| `Halo_catalogs_analysis.ipynb` | StePS_HF matched-frame catalogs: pipeline summary, mass functions (both box sizes), halo-by-halo cross-match (98% top-500 grid-IC ↔ Gadget) | `halo_catalogs{,50}/` |
| `Halo_stacking_anisotropy.ipynb` | anisotropy stacking (Rácz+2021 octahedral method) over **9 runs**: grid-IC lattice memory & epoch stacks, lattice phase, 3D + O_h fold, direction cones, PDS50 I* wraparound note, and §6 decomposing the face excess at fixed load and fixed topology. With four glass runs the excess is **the grid lattice alone** (~+0.06); the residual previously attributed to the cubic FFT/CIC mesh is not supported | snapshots + catalogs (incl. the realization-B pair and the a=16 glass) |
| `Gadget_vs_PDS_1024_comparison.ipynb` | as above plus **two 1024³ PDS glass runs** (402M particles each) — `1024³ᴬ` carries the 2×2's own realization A to high resolution (so §1–§2 show the *same* structures at 4× resolution), while `1024³ᴮ` with its phase-matched 256³ partner does the resolution analysis in §7 at fixed LPT order. Large runs are read in chunks and cut to the analysis cube on the fly | the 2×2 above, `test1024glassA_run`, `test1024glass_run`, `test256glass_pm_run` |
| `PDS_glass_variance_study.ipynb` | why the low-k features are what they are: 10 runs across **three realizations, two codes, two topologies and two resolutions**; the Rácz+2022 complementary pair; the Ω³ chart gradient; and the glass-relaxation cure for the glass anomaly. §5 adds the **matched 1024³ pair** — identical in everything but the large-scale realization — which drops the k ≈ 0.021 peak from 1.998 to 1.273 | the 256³ glass family, `gadget256_glass{,B,C}`, `glass256_long_run`, `test1024glass{,A}_run` |
| `PDS_Millenium_View.ipynb` | Millennium-style renders of PDS runs | run snapshots |

**Estimator note.** `power_spectrum()` weights particles by **m/Ω³** for PDS runs
(`clustering_pk(..., deconf=True)`, the default). Weighting by raw counts leaves the glass
load's Ω³ chart gradient in the spectrum and shows up as ~2000 Mpc³ of spurious power in the
fundamental bin — see `PDS_glass_variance_study.ipynb` §3.

Older/auxiliary scripts: `millennium_render.py`. Heavy pipelines that generated the
catalogs and full stacking figure sets live outside the repo in
`/scratch/csabai/halo_catalogs{,50}/` and `/scratch/csabai/stack3d/` (each with a README).
