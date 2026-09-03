# Visualization & analysis notebooks

Executable notebooks for the StePS PDS validation campaign (2026). Figures are embedded;
re-execution needs the simulation outputs under `/scratch/csabai/` (snapshots + halo
catalogs) and, where noted, the `stepsic` package on `sys.path`. Conventions (matched
frame, PDS mass de-conformalization, wraparound rule) are documented in
`../../StePS/docs/PDS_guide.md` and the CHANGELOG.

| notebook | contents | data dependency |
|---|---|---|
| `Gadget_vs_PDS_comparison.ipynb` | 1200 Mpc **2×2** — {Gadget4 T³, StePS PDS} × {grid load, glass load}: matched density slices (mass-weighted, de-conformalized), P(k) evolution in the domain-fitting cube (no window pedestal), P(k) ratios & growth, grid-vs-glass Bragg panel, and §6 separating the **load effect from the topology effect** | `gadget256_flat`, `gadget256_glass`, `test256disc`, `test256glass` |
| `Gadget_vs_PDS_50Mpc_comparison.ipynb` | 50 Mpc topology-dominated glass pair: matched slices (shared IC at z=30 → decorrelated by z=0), P(k), tiling signature | `gadget50_glass`, `test50glass` |
| `Halo_catalogs_analysis.ipynb` | StePS_HF matched-frame catalogs: pipeline summary, mass functions (both box sizes), halo-by-halo cross-match (98% top-500 grid-IC ↔ Gadget) | `halo_catalogs{,50}/` |
| `Halo_stacking_anisotropy.ipynb` | anisotropy stacking (Rácz+2021 octahedral method): grid-IC lattice memory & epoch stacks, lattice phase, 3D + O_h fold, direction cones, PDS50 I* wraparound note, and §6 decomposing the face excess into **lattice vs cubic-mesh** components at fixed topology | snapshots + catalogs (incl. `gadget256_glass`); `stepsic` importable |
| `PDS_Millenium_View.ipynb` | Millennium-style renders of PDS runs | run snapshots |

Older/auxiliary scripts: `millennium_render.py`. Heavy pipelines that generated the
catalogs and full stacking figure sets live outside the repo in
`/scratch/csabai/halo_catalogs{,50}/` and `/scratch/csabai/stack3d/` (each with a README).
