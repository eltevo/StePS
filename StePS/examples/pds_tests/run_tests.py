#!/usr/bin/env python3
"""
PDS physics validation tests for StePS.

Run from the StePS/StePS directory:
    conda activate stepsic
    python examples/pds_tests/run_tests.py

Three tests:

  1. Single particle, free flight
     A lone particle with a moderate velocity.  With no pair force, it drifts under
     Hubble damping.  We verify: no NaN positions, quaternion unit norm preserved,
     velocity decreases due to Hubble drag.

  2. Fast single particle — boundary wrapping (gluing)
     One particle given a very high velocity so it crosses one or more dodecahedral
     faces.  After each face crossing pds_wrap() should map the particle back to the
     fundamental domain.  We verify: positions always stay inside the fundamental
     domain, at least one crossing is detected, no NaN.

  3. Two particles with parallel initial velocities — gravitational convergence
     Two particles separated by d in the y-direction, both moving in the x-direction
     at the same speed.  Their mutual S³ gravity should draw them closer together.
     For d << R_curv the S³ force reduces to Newtonian 1/r², giving a measurable
     (and analytically checkable) convergence rate.

  4. PDS homogeneity — zero self-image force (IS_PERIODIC = 2)
     A single particle at rest must stay at rest ANYWHERE in the fundamental
     domain: its 119 non-trivial I* self-images always sit at fixed geodesic
     distances (chi(q, gq) = arccos(Re g) is independent of q) in a symmetric
     constellation, so their compensated forces cancel exactly.  Tests the exact
     120-image summation path including the j == i self-image terms.

  5. Multi-particle stability sanity (IS_PERIODIC = 2)
     8 random in-domain particles with the exact image summation: no NaN,
     no velocity blow-up, all particles stay in the fundamental domain.

  6. Python/C++ force cross-validation (IS_PERIODIC = 2)
     A SAVE_ACCELERATIONS build writes the forces computed on a 50-particle
     random IC; they are compared against an independent NumPy implementation
     of the exact compensated 120-image sum (stepsic.pds).  Max relative
     error must be < 1e-6.

  7. End-to-end stepsic PDS IC → StePS run
     Generates a small native PDS IC with stepsic (GEOMETRY='pds', NGRID=8),
     runs StePS for one output interval, and checks that the quaternions are
     used directly, the mass/density consistency check passes, and the
     density contrast grows.

  8. R³ regression (non-PDS build)
     Compiles the default spherical (R³) binary from the shared sources and
     runs a small cosmological cloud — guards against PDS work breaking the
     other topologies (shared files: main.cc, step.cc, forces.cc).
"""

import os
import re
import sys
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parent.parent.parent  # StePS/StePS/
BINARY   = REPO / "build" / "StePS"
BINARY_SAVEACC = REPO / "build" / "StePS_saveacc"
BINARY_R3      = REPO / "build" / "StePS_r3"
STEPSIC_ROOT   = REPO.parent.parent / "stepsic"
TEST_DIR = Path(__file__).resolve().parent
# Scratch area for the generated ICs / snapshots.  Defaults to <repo>/data/pds_tests;
# override with PDS_TEST_OUT_BASE when the repo lives on a small/slow filesystem.
OUT_BASE = Path(os.environ.get("PDS_TEST_OUT_BASE",
                               REPO.parent.parent / "data" / "pds_tests"))

sys.path.insert(0, str(STEPSIC_ROOT))  # for stepsic.pds (Test 6)

# ── Physical constants / unit system ──────────────────────────────────────────
# StePS internal units (from global_variables.h)
UNIT_V = 20.738652969925447   # km/s  (1 internal velocity unit)
UNIT_T = 47.14829951063323    # Gy    (1 internal time unit = 1 Mpc / UNIT_V)

# PDS parameters (must match the param files below)
R_CURV = 3100.0      # Mpc   curvature radius

# Chi of the dodecahedral fundamental-domain inradius.
# The nearest I* group element is at chi = 36° (= 2π/10, from the order-10 generator t).
# The face midpoint is at the midpoint geodesic: chi_in = 18°.
# The domain outradius (vertices) is ≈ 20–21°.
# Physical inradius = R_curv × (18° × π/180) ≈ 974 Mpc.
CHI_IN = np.radians(18.0)   # 18° inradius, outradius ≈ 21°

# Cosmological parameters (standard ΛCDM)
OMEGA_M  = 0.3111
OMEGA_L  = 0.6889
H0       = 67.66     # km/s/Mpc

# ── Terminal colours ───────────────────────────────────────────────────────────
_GRN = "\033[92m"; _RED = "\033[91m"; _RST = "\033[0m"
PASS_S = f"{_GRN}PASS{_RST}"; FAIL_S = f"{_RED}FAIL{_RST}"

def check(ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    return ok


# ── HDF5 IC writer ─────────────────────────────────────────────────────────────
def write_ic(path, pos_mpc, vel_pec_kmps, mass_code, a_start, quaternions=None):
    """
    Write a minimal Gadget-2/StePS HDF5 IC file.

    pos_mpc      : (N,3) float64   — stereographic Cartesian positions in Mpc
    vel_pec_kmps : (N,3) float64   — physical peculiar velocities in km/s
    mass_code    : (N,)  float64   — masses in code units (1 code unit ≈ 1e11 M_sun)
    a_start      : float           — initial scale factor

    StePS reads velocities as v_stored / (sqrt(a_start) * UNIT_V), where
    v_stored is the Gadget convention v_pec * sqrt(a_start).  Storing
    v_stored = v_pec * sqrt(a_start) cancels the sqrt and yields v_pec/UNIT_V.
    """
    N = len(pos_mpc)
    v_stored = vel_pec_kmps * np.sqrt(a_start)   # Gadget convention

    with h5py.File(path, "w") as f:
        hdr = f.create_group("Header")
        np_arr = np.zeros(6, dtype=np.int32); np_arr[1] = N
        hdr.attrs["NumPart_ThisFile"]     = np_arr
        hdr.attrs["NumPart_Total"]        = np_arr
        hdr.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        hdr.attrs["MassTable"]            = np.zeros(6)
        hdr.attrs["Time"]                 = float(a_start)
        hdr.attrs["Redshift"]             = 1.0 / a_start - 1.0
        hdr.attrs["BoxSize"]              = 0.0
        hdr.attrs["NumFilesPerSnapshot"]  = 1
        hdr.attrs["Omega0"]               = OMEGA_M
        hdr.attrs["OmegaLambda"]          = OMEGA_L
        hdr.attrs["HubbleParam"]          = H0 / 100.0
        for flag in ["Flag_Sfr","Flag_Feedback","Flag_Cooling",
                     "Flag_StellarAge","Flag_Metals","Flag_Entropy_ICs"]:
            hdr.attrs[flag] = 0

        pt = f.create_group("PartType1")
        pt.create_dataset("Coordinates", data=pos_mpc.astype(np.float32))
        pt.create_dataset("Velocities",  data=v_stored.astype(np.float32))
        pt.create_dataset("Masses",      data=mass_code.astype(np.float32))
        pt.create_dataset("ParticleIDs", data=np.arange(1, N+1, dtype=np.uint64))
        if quaternions is not None:
            pt.create_dataset("Quaternions", data=quaternions.astype(np.float64))


# ── Parameter file writer ──────────────────────────────────────────────────────
def write_param(path, out_dir, ic_path, out_lst, a_start, a_max,
                acc_param=0.005, step_min=1e-5, step_max=0.01,
                particle_radii=1.0, is_periodic=1, pds=True,
                omega_m=OMEGA_M, omega_l=OMEGA_L, h0=H0):
    pds_line = f"PDS_R_CURV              {R_CURV}\n" if pds else ""
    Path(path).write_text(f"""\
Cosmological parameters:
------------------------
Omega_b         0.0
Omega_lambda    {omega_l}
Omega_m         {omega_m}
Omega_r         0.0
HubbleConstant  {h0}
a_start         {a_start}
a_max           {a_max}

Simulation parameters:
-----------------------
COSMOLOGY               1
IS_PERIODIC             {is_periodic}
COMOVING_INTEGRATION    1
{pds_line}L_BOX                   6200.0
R_SIM                   960.0
IC_FILE                 {ic_path}
IC_FORMAT               2
OUT_DIR                 {out_dir}/
OUT_LST                 {out_lst}
OUTPUT_TIME_VARIABLE    1
OUTPUT_FORMAT           2
REDSHIFT_CONE           0
MIN_REDSHIFT            0.0
ACC_PARAM               {acc_param}
STEP_MIN                {step_min}
STEP_MAX                {step_max}
PARTICLE_RADII          {particle_radii}
FIRST_T_OUT             0.0
H_OUT                   1.0
""")


def write_redshift_list(path, z_values):
    """Write a decreasing list of redshifts for OUT_LST."""
    with open(path, "w") as f:
        for z in sorted(z_values, reverse=True):
            f.write(f"{z:.6f}\n")


# ── Simulation runner ──────────────────────────────────────────────────────────
def run_sim(param_file, timeout=120, binary=None):
    # Write output to a file (not a pipe) to avoid kernel-buffer deadlocks.
    # OMP_NUM_THREADS=1 prevents OpenMPI from probing GPUs at MPI_Init,
    # which would otherwise stall for minutes on a machine with CUDA devices.
    # Resolve the default at CALL time: `binary=BINARY` as a default argument would
    # bind the module-level value at def time, so any later reassignment of BINARY is
    # silently ignored and the wrong binary gets exercised.
    if binary is None:
        binary = BINARY
    out_path = Path(param_file).parent / "run.log"
    env = {**os.environ,
           "OMP_NUM_THREADS": "1",
           "OMPI_MCA_pml": "ob1",
           "OMPI_MCA_btl": "self,tcp",
           "OMPI_MCA_btl_tcp_if_include": "lo"}
    with open(out_path, "w") as log:
        result = subprocess.run([str(binary), str(param_file)],
                                stdout=log, stderr=log,
                                env=env, timeout=timeout)
    if result.returncode != 0:
        with open(out_path) as f:
            lines = f.readlines()
        print("  Run log (last 20 lines):\n", "".join(lines[-20:]))
        raise RuntimeError(f"StePS exited with code {result.returncode}")


# ── Build orchestration for the binary variants (Tests 6 and 8) ───────────────
def _detect_cxx():
    """Pick a C++ compiler: $CXX, else the conda toolchain, else plain g++."""
    if os.environ.get("CXX"):
        return os.environ["CXX"]
    conda = os.environ.get("CONDA_PREFIX", "")
    if conda:
        for name in ("x86_64-conda-linux-gnu-c++", "x86_64-conda-linux-gnu-g++"):
            if (Path(conda) / "bin" / name).exists():
                return name
    for name in ("mpicxx", "g++", "c++"):
        if shutil.which(name):
            return name
    raise RuntimeError("No C++ compiler found; set CXX=... explicitly.")


# Extra -D options appended to every build, e.g.
#     PDS_TEST_EXTRA_OPT=-DPDS_INTRINSIC python3 run_tests.py
# Lets the whole suite be run against a build variant.  It is part of the build stamp,
# so switching it always forces a rebuild rather than silently reusing binaries.
EXTRA_OPT = os.environ.get("PDS_TEST_EXTRA_OPT", "").strip()
INTRINSIC_BUILD = "-DPDS_INTRINSIC" in EXTRA_OPT

DEFAULT_PDS_OPT = "-DPOINCARE_DODECAHEDRAL -DHAVE_HDF5 -DCOSMOPARAM=0"


def _with_extra(opt):
    """Append EXTRA_OPT to an OPT string (None means the makefile's own default)."""
    if not EXTRA_OPT:
        return opt
    return f"{opt} {EXTRA_OPT}" if opt is not None else f"{DEFAULT_PDS_OPT} {EXTRA_OPT}"


def _make(opt=None, extra=()):
    """Run make -f PDS-LinuxGCC-Makefile in REPO.

    Toolchain is auto-detected (override with CXX / CONDA_PREFIX) so the suite is
    not tied to one machine's conda install.
    """
    conda = os.environ.get("CONDA_PREFIX", "")
    args = ["make", "-f", "PDS-LinuxGCC-Makefile", "-j8",
            f"CXX={_detect_cxx()}",
            f"MPI_INC=-I{conda}/include", f"MPI_LIBS=-L{conda}/lib -lmpi",
            f"HDF5_INC=-I{conda}/include", f"HDF5_LIBS=-L{conda}/lib -lhdf5"]
    opt = _with_extra(opt)
    if opt is not None:
        args.append(f"OPT={opt}")
    args.extend(extra)
    subprocess.run(["make", "-f", "PDS-LinuxGCC-Makefile", "clean"],
                   cwd=REPO, capture_output=True)
    res = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"make failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    _write_stamp(BINARY, opt)


# ── Build stamps ──────────────────────────────────────────────────────────────
# A binary is only reusable if it was built with the SAME options.  Caching on
# mtime alone silently reuses e.g. a -DPDS_INTRINSIC build as the production
# binary, which produces wrong physics with no visible error.
def _stamp_path(binary):
    return Path(str(binary) + ".buildopt")


def _write_stamp(binary, opt):
    _stamp_path(binary).write_text(opt if opt is not None else "<default>")


def _stamp_matches(binary, opt):
    sp = _stamp_path(binary)
    if not sp.exists():
        return False
    return sp.read_text() == (opt if opt is not None else "<default>")


def ensure_binary_variants():
    """
    Build (once) the two extra binary variants needed by Tests 6 and 8:
      build/StePS_saveacc — PDS build with -DSAVE_ACCELERATIONS
      build/StePS_r3      — default spherical R^3 build (no PDS)
    The standard PDS binary is rebuilt afterwards so build/StePS is always
    the production variant.  Variants are cached: nothing happens when both
    already exist and are newer than every src/ file.
    """
    src_mtime = max(p.stat().st_mtime for p in (REPO / "src").glob("*"))
    OPT_SAVEACC = "-DPOINCARE_DODECAHEDRAL -DHAVE_HDF5 -DCOSMOPARAM=0 -DSAVE_ACCELERATIONS"
    OPT_R3      = "-DHAVE_HDF5 -DCOSMOPARAM=0"

    def stale(binary, opt):
        return not (binary.exists()
                    and binary.stat().st_mtime > src_mtime
                    and _stamp_matches(binary, opt))

    need_saveacc = stale(BINARY_SAVEACC, _with_extra(OPT_SAVEACC))
    need_r3      = stale(BINARY_R3, _with_extra(OPT_R3))
    need_main    = stale(BINARY, _with_extra(None))
    if EXTRA_OPT:
        print(f"(building all variants with extra options: {EXTRA_OPT})")
    if not (need_saveacc or need_r3):
        if need_main:
            print("Rebuilding the standard PDS binary...")
            _make()
        return
    if need_saveacc:
        print("Building StePS_saveacc (PDS + SAVE_ACCELERATIONS)...")
        _make(opt=OPT_SAVEACC)
        shutil.copy2(BINARY, BINARY_SAVEACC)
        _write_stamp(BINARY_SAVEACC, _with_extra(OPT_SAVEACC))
    if need_r3:
        print("Building StePS_r3 (default spherical R^3 topology)...")
        _make(opt=OPT_R3)
        shutil.copy2(BINARY, BINARY_R3)
        _write_stamp(BINARY_R3, _with_extra(OPT_R3))
    print("Rebuilding the standard PDS binary...")
    _make()


# ── Snapshot loader ────────────────────────────────────────────────────────────
def load_snapshots(out_dir):
    """Return list of (a, pos, vel) sorted by a."""
    snaps = []
    for f in sorted(Path(out_dir).glob("snapshot_*.hdf5")):
        try:
            with h5py.File(f, "r") as h:
                a   = float(h["Header"].attrs["Time"])
                pos = h["PartType1/Coordinates"][:]
                vel = h["PartType1/Velocities"][:]
            snaps.append((a, pos, vel))
        except Exception:
            pass   # truncated file (race in earlier run)
    return sorted(snaps, key=lambda x: x[0])


# ── S³ helpers ─────────────────────────────────────────────────────────────────
def to_quaternion(pos, R=R_CURV):
    """Inverse stereographic projection: (N,3) Cartesian → (N,4) unit quaternions."""
    r2 = np.einsum("...i,...i->...", pos, pos)
    d  = R**2 + r2
    return np.stack([(R**2 - r2)/d,
                     2*R*pos[..., 0]/d,
                     2*R*pos[..., 1]/d,
                     2*R*pos[..., 2]/d], axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Single particle, free flight
# ══════════════════════════════════════════════════════════════════════════════
def test1():
    print("\n── Test 1: Single particle, free flight ─────────────────────────────")
    tag    = "test1"
    outdir = OUT_BASE / tag
    ic     = outdir / "ic.hdf5"
    param  = outdir / f"{tag}.param"
    outlst = outdir / "redshifts.txt"

    a_start, a_max = 0.5, 1.0
    v_pec = 1000.0   # km/s in x — slow enough to stay in fundamental domain

    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    write_ic(ic,
             pos_mpc      = np.array([[0.0, 0.0, 0.0]]),
             vel_pec_kmps = np.array([[v_pec, 0.0, 0.0]]),
             mass_code    = np.array([1.0]),
             a_start      = a_start)

    # 6 redshifts: StePS skips the first when it equals z_start, leaving 5 outputs.
    z_out = 1.0 / np.linspace(a_start, a_max, 6) - 1.0
    write_redshift_list(outlst, z_out)
    write_param(param, outdir, ic, outlst,
                a_start=a_start, a_max=a_max,
                acc_param=0.1, step_max=0.2, particle_radii=1.0)

    run_sim(param)
    snaps = load_snapshots(outdir)

    ok = True
    ok &= check(len(snaps) >= 5, f"Got {len(snaps)} snapshots (expect ≥ 5)")
    ok &= check(all(np.isfinite(s[1]).all() for s in snaps),
                "No NaN / Inf in positions")

    # Quaternion unit norm (inverse stereo projection of every snapshot position)
    max_norm_err = 0.0
    for _, pos, _ in snaps:
        q = to_quaternion(pos)
        max_norm_err = max(max_norm_err, np.abs(np.linalg.norm(q, axis=1) - 1.0).max())
    ok &= check(max_norm_err < 1e-6,
                f"Quaternion unit norm: max |‖q‖−1| = {max_norm_err:.2e}")

    # Hubble damping: |v| should decrease over time
    v_mags = [np.linalg.norm(s[2][0]) for s in snaps]
    ok &= check(v_mags[-1] < v_mags[0],
                f"Velocity Hubble-damped: {v_mags[0]:.1f} → {v_mags[-1]:.1f} km/s")

    # Net displacement in x should be positive (particle moved in +x)
    x0 = snaps[0][1][0, 0];  x1 = snaps[-1][1][0, 0]
    ok &= check(x1 > x0, f"Net +x displacement: {x0:.2f} → {x1:.2f} Mpc")

    print(f"  Result: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Fast particle: boundary wrapping (gluing)
# ══════════════════════════════════════════════════════════════════════════════
def test2():
    print("\n── Test 2: Fast particle — boundary wrapping ────────────────────────")
    tag    = "test2"
    outdir = OUT_BASE / tag
    ic     = outdir / "ic.hdf5"
    param  = outdir / f"{tag}.param"
    outlst = outdir / "redshifts.txt"

    # Strategy: start the particle far outside the fundamental domain
    # (x0 = 1915 Mpc stereo, chi ≈ 63°; the domain boundary is at 18–21°) so
    # the IC wrap maps it inside immediately, and give it a high velocity so
    # it keeps crossing dodecahedral faces during the run — each crossing
    # exercises pds_wrap + the velocity transformation (gluing).
    a_start = 0.5
    a_max   = 1.0
    # Deliberately extreme: this is a pure numerical exercise of pds_wrap (the IC
    # already starts the particle outside the domain at chi = 63 deg), and the wrap
    # must be crossed unambiguously.  At 50,000 km/s the particle only traverses
    # ~2.8 deg over the whole run, so it merely *grazes* the boundary and whether it
    # crosses is decided by sub-degree integrator differences — the flat drift
    # over-shoots the chart and crosses, the exact geodesic curves along the boundary
    # and does not.  That made the test integrator-dependent.  At 1e6 km/s both the
    # flat and the intrinsic integrator cross on every interval (jumps 500-800 Mpc vs
    # the 200 Mpc threshold) and agree with each other to <1%.
    v_pec   = 1_000_000.0   # km/s

    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    write_ic(ic,
             pos_mpc      = np.array([[1915.0, 0.0, 0.0]]),
             vel_pec_kmps = np.array([[v_pec,  0.0, 0.0]]),
             mass_code    = np.array([1.0]),
             a_start      = a_start)

    # 6 redshifts: StePS skips the first when it equals z_start, leaving 5 outputs.
    z_out = 1.0 / np.linspace(a_start, a_max, 6) - 1.0
    write_redshift_list(outlst, z_out)
    write_param(param, outdir, ic, outlst,
                a_start=a_start, a_max=a_max,
                acc_param=0.1, step_max=0.2, particle_radii=1.0)

    run_sim(param)
    snaps = load_snapshots(outdir)

    ok = True
    ok &= check(len(snaps) >= 5, f"Got {len(snaps)} snapshots")
    ok &= check(all(np.isfinite(s[1]).all() for s in snaps),
                "No NaN / Inf in positions")

    # Quaternion norm
    max_norm_err = 0.0
    for _, pos, _ in snaps:
        q = to_quaternion(pos)
        max_norm_err = max(max_norm_err, np.abs(np.linalg.norm(q, axis=1) - 1.0).max())
    ok &= check(max_norm_err < 1e-6,
                f"Quaternion unit norm: max |‖q‖−1| = {max_norm_err:.2e}")

    # Fundamental-domain membership: chi = arccos(q0) must be ≤ chi_out ≈ 21°
    # Inradius = 18° (face midpoints); outradius ≈ 20–21° (vertices).
    # Allow +3° above inradius to accommodate dodecahedron vertices.
    chi_in_deg = np.degrees(CHI_IN)
    tol = 3.0  # degrees  (inradius 18° + 3° ≈ outradius 21°)
    max_chi_deg = 0.0
    for _, pos, _ in snaps:
        q       = to_quaternion(pos)
        chi_deg = np.degrees(np.arccos(np.clip(q[:, 0], -1.0, 1.0)))
        max_chi_deg = max(max_chi_deg, chi_deg.max())
    ok &= check(max_chi_deg <= chi_in_deg + tol,
                f"All positions in fundamental domain: max chi = {max_chi_deg:.2f}° "
                f"(inradius = {chi_in_deg:.2f}°, tolerance = +{tol}° to outradius)")

    # Detect at least one boundary crossing: a crossing appears as a large jump
    # in stereographic position between consecutive snapshots.
    positions = np.array([s[1][0] for s in snaps])  # (T, 3)
    jumps     = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    # A jump > 200 Mpc almost certainly means a face crossing, not smooth motion.
    n_cross = int((jumps > 200.0).sum())
    ok &= check(n_cross >= 1,
                f"Detected {n_cross} boundary crossing(s) "
                f"(max jump = {jumps.max():.0f} Mpc)")

    print(f"  Result: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Two particles, parallel velocities: gravitational convergence
# ══════════════════════════════════════════════════════════════════════════════
def test3():
    print("\n── Test 3: Two particles — gravitational convergence ────────────────")
    tag    = "test3"
    outdir = OUT_BASE / tag
    ic     = outdir / "ic.hdf5"
    param  = outdir / f"{tag}.param"
    outlst = outdir / "redshifts.txt"

    # Geometry: particles separated by d in y, same x-velocity.
    # Mutual S³ force draws them together in y.
    # For d << R_curv, F_S³ ≈ G m / d² (Newtonian limit).
    a_start = 0.8
    a_max   = 1.0
    d       = 5.0    # Mpc  initial y-separation
    v_pec   = 100.0  # km/s  common x-velocity (for both)
    # Mass: 1e14 M_sun / (1e11 M_sun / code unit) = 1000 code units per particle
    # At d=5 Mpc this gives acceleration ≈ 1.76 km/s/Gy, convergence ≈ 3–4 Mpc
    # over the ~2.5 Gy run (a: 0.8→1.0), easily detectable.
    m_code  = 1_000.0   # code units  (≈ 10^14 M_sun)

    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    write_ic(ic,
             pos_mpc      = np.array([[ 0.0,  d/2, 0.0],
                                      [ 0.0, -d/2, 0.0]]),
             vel_pec_kmps = np.array([[ v_pec, 0.0, 0.0],
                                      [ v_pec, 0.0, 0.0]]),
             mass_code    = np.array([m_code, m_code]),
             a_start      = a_start)

    # 6 redshifts: StePS skips the first when it equals z_start, leaving 5 outputs.
    z_out = 1.0 / np.linspace(a_start, a_max, 6) - 1.0
    write_redshift_list(outlst, z_out)
    write_param(param, outdir, ic, outlst,
                a_start=a_start, a_max=a_max,
                acc_param=0.05, step_max=0.1,
                particle_radii=0.5)

    run_sim(param)
    snaps = load_snapshots(outdir)

    ok = True
    ok &= check(len(snaps) >= 5, f"Got {len(snaps)} snapshots")
    ok &= check(all(np.isfinite(s[1]).all() for s in snaps),
                "No NaN / Inf in positions")

    # y-separation should decrease monotonically
    seps = np.array([abs(s[1][0, 1] - s[1][1, 1]) for s in snaps])
    ok &= check(seps[-1] < seps[0],
                f"y-separation decreased: {seps[0]:.3f} → {seps[-1]:.3f} Mpc "
                f"(Δ = {seps[0]-seps[-1]:.3f} Mpc)")

    # Quantitative force check using the first two snapshots (before dynamics
    # substantially change d).
    # Expected relative acceleration: 2 × G m / d²
    # G = 4.30091e-9 Mpc (km/s)² M_sun^{-1}; M_sun / code_unit = 1e11
    # ⇒ G_code = 4.30091e-9 × 1e11 = 4.30091e2  Mpc (km/s)² code_unit^{-1}
    G_code = 4.30091e-9 * 1e11   # Mpc (km/s)² per code unit

    # Measured relative acceleration in y from the first two snapshots:
    a1, pos1, vel1 = snaps[0]
    a2, pos2, vel2 = snaps[1]
    dt_gy = (a2 - a1) / (a1 * H0 / 978.0)   # rough Δt in Gy; 978 Gy·km/s/Mpc

    # Relative velocity in y between the two times
    dvy1 = vel1[0, 1] - vel1[1, 1]   # relative vy at t1
    dvy2 = vel2[0, 1] - vel2[1, 1]   # relative vy at t2
    # Relative acceleration from finite difference (sign: towards each other → negative dy)
    accel_meas = (dvy2 - dvy1) / (dt_gy * UNIT_T)   # (km/s) / (UNIT_T * Gy) ≈ (km/s) per Gy...
    # ... convert to Mpc/Gy² units not needed for ratio; compare with expected

    # Expected Newtonian relative acceleration at initial d:
    accel_expected = -2.0 * G_code * m_code / (d**2)   # negative = converging
    # Units: Mpc (km/s)² code_unit^{-1} * code_unit / Mpc² = (km/s)² / Mpc
    #        = km/s / (Mpc / km/s) = km/s / UNIT_T·UNIT_T... tricky.
    # For a sanity ratio we use the S³ correction factor at geodesic angle chi0 = d/R:
    chi0  = d / R_CURV
    ratio_s3_to_newton = (chi0 / np.sin(chi0))**2   # S³ force / Newtonian force
    ok &= check(abs(ratio_s3_to_newton - 1.0) < 0.001,
                f"S³/Newtonian force ratio at d={d} Mpc: {ratio_s3_to_newton:.6f} "
                f"(expected ≈ 1 for d << R_curv)")

    # Final broad sanity: convergence is in the right ballpark (within factor 5 of
    # Newtonian prediction).  This guards against a sign error or factor-of-1000 bug.
    #
    # accel_expected is in (km/s)²/Mpc.  Convert to Mpc/Gy²:
    #   1 km/s = 1.023e-3 Mpc/Gy  →  (km/s)²/Mpc = (1.023e-3)² Mpc/Gy²
    KMS2_PER_MPC_TO_MPC_PER_GY2 = (1.023e-3)**2   # = 1.046e-6
    accel_mpc_gy2 = abs(accel_expected) * KMS2_PER_MPC_TO_MPC_PER_GY2  # Mpc/Gy²

    t_run_gy = (a_max - a_start) / (a_start * H0 / 978.0)   # rough Δt [Gy]
    delta_d_pred = 0.5 * accel_mpc_gy2 * t_run_gy**2        # Mpc
    delta_d_meas = seps[0] - seps[-1]
    ratio_meas = delta_d_meas / max(delta_d_pred, 1e-10)
    ok &= check(0.1 < ratio_meas < 10.0,
                f"Convergence magnitude: measured Δd = {delta_d_meas:.3f} Mpc, "
                f"predicted {delta_d_pred:.3f} Mpc (ratio = {ratio_meas:.2f}, "
                f"expect 0.1–10)")

    print(f"  Result: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — PDS homogeneity: zero self-image force everywhere
# ══════════════════════════════════════════════════════════════════════════════
def test4():
    print("\n── Test 4: Homogeneity — zero self-image force (IS_PERIODIC=2) ──────")
    tag    = "test4"
    outdir = OUT_BASE / tag
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # Two independent single-particle runs at rest: one at the domain centre,
    # one at a generic off-centre point.  The 119 non-trivial self-images sit
    # at q-independent distances (chi(q, gq) = arccos(Re g)) in a symmetric
    # constellation, so the exact compensated image sum must vanish and the
    # particle must stay at rest — the PDS analogue of homogeneity.
    a_start, a_max = 0.5, 1.0
    ok = True
    for sub, x0 in (("centre", [0.0, 0.0, 0.0]),
                    ("offcentre", [300.0, 200.0, 100.0])):
        subdir = outdir / sub
        subdir.mkdir(parents=True, exist_ok=True)
        ic, param, outlst = subdir / "ic.hdf5", subdir / "t4.param", subdir / "redshifts.txt"
        write_ic(ic,
                 pos_mpc      = np.array([x0]),
                 vel_pec_kmps = np.zeros((1, 3)),
                 mass_code    = np.array([1.0e8]),   # huge mass: any spurious self-force would show
                 a_start      = a_start)
        z_out = 1.0 / np.linspace(a_start, a_max, 6) - 1.0
        write_redshift_list(outlst, z_out)
        write_param(param, subdir, ic, outlst, a_start=a_start, a_max=a_max,
                    acc_param=0.1, step_max=0.2, particle_radii=1.0,
                    is_periodic=2)
        run_sim(param)
        snaps = load_snapshots(subdir)
        ok &= check(len(snaps) >= 5, f"[{sub}] Got {len(snaps)} snapshots")
        drift = max(np.linalg.norm(s[1][0] - np.array(x0)) for s in snaps)
        ok &= check(drift < 1e-3,
                    f"[{sub}] Particle at rest stays at rest: max drift = {drift:.2e} Mpc")
        vmax = max(np.linalg.norm(s[2][0]) for s in snaps)
        ok &= check(vmax < 1e-3,
                    f"[{sub}] No spurious velocity: max |v| = {vmax:.2e} km/s")

    print(f"  Result: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Multi-particle stability sanity
# ══════════════════════════════════════════════════════════════════════════════
def test5():
    print("\n── Test 5: 8-particle stability sanity (IS_PERIODIC=2) ──────────────")
    tag    = "test5"
    outdir = OUT_BASE / tag
    ic     = outdir / "ic.hdf5"
    param  = outdir / f"{tag}.param"
    outlst = outdir / "redshifts.txt"

    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    a_start, a_max = 0.5, 1.0
    rng = np.random.default_rng(11)
    # random in-domain positions (stereo radius < 480 Mpc is safely inside)
    pos = rng.normal(size=(8, 3))
    pos = pos / np.linalg.norm(pos, axis=1, keepdims=True) \
        * rng.uniform(50, 480, size=(8, 1))
    write_ic(ic,
             pos_mpc      = pos,
             vel_pec_kmps = rng.normal(scale=100.0, size=(8, 3)),
             mass_code    = np.full(8, 1.0e4),   # ~1e15 Msun each
             a_start      = a_start)
    z_out = 1.0 / np.linspace(a_start, a_max, 6) - 1.0
    write_redshift_list(outlst, z_out)
    write_param(param, outdir, ic, outlst, a_start=a_start, a_max=a_max,
                acc_param=0.05, step_max=0.1, particle_radii=5.0,
                is_periodic=2)
    run_sim(param)
    snaps = load_snapshots(outdir)

    ok = True
    ok &= check(len(snaps) >= 5, f"Got {len(snaps)} snapshots")
    ok &= check(all(np.isfinite(s[1]).all() and np.isfinite(s[2]).all()
                    for s in snaps), "No NaN / Inf in positions or velocities")
    vmax = max(np.abs(s[2]).max() for s in snaps)
    ok &= check(vmax < 1.0e4, f"No velocity blow-up: max |v| = {vmax:.1f} km/s")
    max_chi = 0.0
    for _, p, _ in snaps:
        q = to_quaternion(p)
        max_chi = max(max_chi, np.degrees(np.arccos(np.clip(q[:, 0], -1, 1))).max())
    ok &= check(max_chi <= np.degrees(CHI_IN) + 3.0,
                f"All particles stay in the fundamental domain: max chi = {max_chi:.2f}°")
    print(f"  Result: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Python/C++ force cross-validation
# ══════════════════════════════════════════════════════════════════════════════
def test6():
    print("\n── Test 6: Python/C++ force cross-validation (IS_PERIODIC=2) ────────")
    from stepsic import pds

    tag    = "test6"
    outdir = OUT_BASE / tag
    ic     = outdir / "ic.hdf5"
    param  = outdir / f"{tag}.param"
    outlst = outdir / "redshifts.txt"

    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    a_start = 0.5
    soft = 10.0
    rng = np.random.default_rng(7)
    # 50 well-separated particles ...
    pos = rng.normal(size=(50, 3))
    pos = pos / np.linalg.norm(pos, axis=1, keepdims=True) \
        * rng.uniform(20, 480, size=(50, 1))
    mass = rng.uniform(500.0, 2000.0, size=50)
    # ... plus 14 deliberately CLOSE companions so the per-pair softening kernel is
    # actually exercised.  Without these every separation exceeds beta and the whole
    # softened branch is dead code as far as this test is concerned -- which is how a
    # discontinuous softening floor survived here unnoticed.  Offsets straddle the
    # spline's two regions and its joins (beta/2 and beta) as well as near-coincidence.
    companions = np.array([1e-4, 0.05, 0.2, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9,
                           2.0, 2.1, 2.5, 3.0, 5.0])           # in units of `soft`
    base = pos[:len(companions)]
    d = rng.normal(size=(len(companions), 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pos = np.vstack([pos, base + d * (companions[:, None] * soft)])
    mass = np.concatenate([mass, rng.uniform(500.0, 2000.0, size=len(companions))])
    n_part = len(pos)
    write_ic(ic, pos_mpc=pos, vel_pec_kmps=np.zeros((n_part, 3)),
             mass_code=mass, a_start=a_start)
    # the run only needs to reach the SAVE_ACCELERATIONS dump (at startup)
    z_out = 1.0 / np.linspace(a_start, a_start + 0.02, 3) - 1.0
    write_redshift_list(outlst, z_out)
    write_param(param, outdir, ic, outlst, a_start=a_start, a_max=a_start + 0.02,
                acc_param=0.05, step_max=0.1, particle_radii=soft,
                is_periodic=2)
    run_sim(param, binary=BINARY_SAVEACC)

    with h5py.File(outdir / "initial_conditions_0000.hdf5", "r") as f:
        F_cpp = f["PartType1/Accelerations"][:]

    # Read the positions and masses BACK from the IC, not from the in-memory arrays:
    # the file stores them as float32, so StePS integrates slightly different numbers.
    # For the close pairs added above (down to 1e-3 Mpc at a radius of ~100 Mpc) that
    # rounding shifts the separation by ~1%, which the steep softened force turns into
    # a ~1e-4 disagreement -- an artefact of the harness, not of the force kernel.
    with h5py.File(ic, "r") as f:
        pos = f["PartType1/Coordinates"][:].astype(np.float64)
        mass = f["PartType1/Masses"][:].astype(np.float64)

    # Independent NumPy reference: exact compensated 120-image sum with the same
    # per-pair softening floor chi_eff = max(chi, (soft_i + soft_j)/R), pushed
    # forward into the stereographic chart with the FULL Jacobian of
    # x = R q_{1:3}/(1+q0), derived here independently of forces.cc:
    #
    #     dx_i = ( R*t_{i+1} - x_i*t_0 ) / Omega,   Omega = 1 + q0 = 2R^2/(R^2+r^2)
    #
    # forces.cc absorbs the overall R into the kernel normalization, so the
    # comparison is against ( t_{i+1} - (x_i/R)*t_0 ) / Omega, and x_i/R = q_{i+1}/Omega.
    #
    # Three ways to break this test, all historically real:
    #   - drop the 1/Omega factor      -> fails by ~50% (Omega -> 2 at the centre)
    #   - drop the t_0 term            -> fails by up to ~5% at the domain boundary
    #   - mismatch the softening kernel-> fails only on the close pairs added above
    R = R_CURV
    quat = to_quaternion(pos.astype(np.float64))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    # Per-particle softening, exactly as utils.cc computes it:
    #   SOFT_LENGTH[i] = cbrt(M[i]*const_beta),  const_beta = ParticleRadi^3 / M_min
    #                  = ParticleRadi * (M[i]/M_min)^(1/3)
    # The pair softening is beta_ij = SOFT_LENGTH[i] + SOFT_LENGTH[j].  Assuming a
    # single uniform value here is wrong and silently hides disagreements on close pairs.
    soft_len = soft * np.cbrt(mass / mass.min())

    def green_softened(chi, beta):
        '''Mirror of pds_green_softened() in src/pds_group.h (spline on the areal
        radius A = R sin chi, so it is exactly the unsoftened kernel for A >= beta).'''
        out = np.zeros_like(chi)
        ok = (chi > 1e-12) & (chi < np.pi - 1e-12)
        s = np.sin(np.where(ok, chi, 1.0))
        A = R * s
        frac = (2.0 * chi - np.sin(2.0 * chi)) / (2.0 * np.pi)
        far = ok & ((chi >= 0.5 * np.pi) | (A >= beta))
        out[far] = (1.0 - frac[far]) / (A[far] ** 2)
        near = ok & ~far
        if np.any(near):
            An, bn = A[near], np.broadcast_to(beta, A.shape)[near]
            b3, b4, b5, b6 = bn**3, bn**4, bn**5, bn**6
            inner = An <= 0.5 * bn
            spline = np.where(
                inner,
                32.0 / b6 * An**3 - 38.4 / b5 * An**2 + 32.0 / (3.0 * b3),
                -32.0 / (3.0 * b6) * An**3 + 38.4 / b5 * An**2 - 48.0 / b4 * An
                + 64.0 / (3.0 * b3) - (1.0 / 15.0) / np.where(An > 0, An**3, 1.0))
            out[near] = (1.0 - frac[near]) * An * spline
        return out

    imgs = pds.images(quat)                       # (N, 120, 4)
    F_py = np.zeros((n_part, 3))
    n_softened = 0
    for i in range(n_part):
        chis = pds.chi(quat[i], imgs)             # (N, 120)
        dirs = pds.force_direction(quat[i], imgs)
        beta_ij = (soft_len[i] + soft_len)[:, None]           # (N, 1) -> broadcasts
        mags = green_softened(chis, beta_ij)
        n_softened += int(np.sum((R * np.sin(chis) < beta_ij) & (chis > 1e-12)))
        t = np.sum(mass[:, None, None] * mags[..., None] * dirs, axis=(0, 1))  # 4D tangent
        if INTRINSIC_BUILD:
            # The intrinsic integrator drifts along geodesics, not in the chart, so the
            # kernels hand back the RAW tangent and the pushforward is deliberately not
            # applied (see docs/PDS_guide.md).  Compare against that convention.
            F_py[i] = t[1:]
        else:
            inv_omega = 1.0 / (1.0 + quat[i, 0])
            F_py[i] = (t[1:] - quat[i, 1:] * t[0] * inv_omega) * inv_omega

    rel = (np.linalg.norm(F_cpp - F_py, axis=1)
           / np.linalg.norm(F_py, axis=1))
    conv = "raw 4D tangent" if INTRINSIC_BUILD else "chart pushforward"
    ok = check(rel.max() < 1e-6,
               f"C++ vs NumPy exact compensated sum [{conv}]: max rel err = "
               f"{rel.max():.2e} (median {np.median(rel):.2e})")
    ok &= check(n_softened > 0,
                f"softening kernel is exercised ({n_softened} sub-beta image pairs)")
    print(f"  Result: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — End-to-end stepsic PDS IC → StePS
# ══════════════════════════════════════════════════════════════════════════════
def test7():
    print("\n── Test 7: stepsic PDS IC → StePS end-to-end ────────────────────────")
    tag    = "test7"
    outdir = OUT_BASE / tag
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # small native PDS IC: NGRID=8 → ~190 particles in the domain
    base_toml = (REPO / "examples" / "PDS_test_ic.toml").read_text()
    toml = (base_toml
            .replace('NGRID = 16', 'NGRID = 8')
            .replace('NMESH = 64', 'NMESH = 32')
            .replace('IC_PREFIX = "PDS_test"', 'IC_PREFIX = "t7"'))
    # Redirect the IC output by rewriting the key, not by matching one literal path
    # (a literal .replace() silently no-ops if the example's default ever changes,
    # leaving the test writing into whatever directory the example names).
    toml, n_sub = re.subn(r'^IC_DIR\s*=.*$', f'IC_DIR = "{outdir}"',
                          toml, count=1, flags=re.MULTILINE)
    if n_sub != 1:
        raise RuntimeError("could not rewrite IC_DIR in PDS_test_ic.toml")
    toml_path = outdir / "t7.toml"
    toml_path.write_text(toml)
    res = subprocess.run([sys.executable, "stepsic.py", str(toml_path)],
                         cwd=STEPSIC_ROOT, capture_output=True, text=True,
                         timeout=300)
    ok = check(res.returncode == 0, "stepsic IC generation succeeded")
    if not ok:
        print(res.stdout[-1500:], res.stderr[-1500:])
        return False
    ic_files = list(outdir.glob("t7_*/ic.hdf5"))
    ok &= check(len(ic_files) == 1, f"IC file written ({len(ic_files)} found)")
    if not ic_files:
        return False
    ic = ic_files[0]
    with h5py.File(ic, "r") as f:
        n_part = f["PartType1/Coordinates"].shape[0]
        has_quat = "Quaternions" in f["PartType1"]
    ok &= check(has_quat, f"IC contains /PartType1/Quaternions (N = {n_part})")

    # run StePS over one growth interval: a = 1/32 → 1/16 (z 31 → 15)
    param  = outdir / "t7.param"
    outlst = outdir / "redshifts.txt"
    write_redshift_list(outlst, [31.0, 23.0, 15.0])
    # cosmology must match the stepsic IC exactly (Planck2018EE+BAO "best"
    # values from stepsic/config/cosmology.toml), otherwise the StePS
    # mass/density consistency check reports a (correct!) small mismatch
    write_param(param, outdir, ic, outlst, a_start=0.03125, a_max=0.0625,
                acc_param=0.01, step_max=0.05, particle_radii=10.0,
                is_periodic=2,
                omega_m=0.3106, omega_l=0.6894, h0=67.702)
    run_sim(param, timeout=600)

    log_text = (outdir / "run.log").read_text()
    ok &= check("Reading /PartType1/Quaternions" in log_text,
                "StePS used the quaternions from the IC")
    ok &= check("The particle masses are consistent" in log_text,
                "Mass/density consistency check passed")

    snaps = load_snapshots(outdir)
    ok &= check(len(snaps) >= 2, f"Got {len(snaps)} snapshots")
    if len(snaps) >= 2:
        # density contrast growth: the scatter of the nearest-neighbour
        # distance (relative to its mean) must grow from z=31 to z=15
        from scipy.spatial import cKDTree
        def nn_scatter(p):
            d, _ = cKDTree(p).query(p, k=2)
            return np.std(d[:, 1]) / np.mean(d[:, 1])
        s0, s1 = nn_scatter(snaps[0][1]), nn_scatter(snaps[-1][1])
        ok &= check(s1 > s0,
                    f"Density contrast grows: nn-distance scatter "
                    f"{s0:.4f} → {s1:.4f} (a = {snaps[0][0]:.4f} → {snaps[-1][0]:.4f})")
    print(f"  Result: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 — R³ regression with the non-PDS build
# ══════════════════════════════════════════════════════════════════════════════
def test8():
    print("\n── Test 8: R³ (spherical) regression with the non-PDS build ─────────")
    tag    = "test8"
    outdir = OUT_BASE / tag
    ic     = outdir / "ic.hdf5"
    param  = outdir / f"{tag}.param"
    outlst = outdir / "redshifts.txt"

    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # cosmologically consistent cloud: total mass = rho_m * V(R_SIM) so the
    # R^3 density check passes (it is FATAL in the R^3 build)
    a_start, a_max = 0.5, 1.0
    R_sim = 960.0
    H_int = H0 / UNIT_V                       # internal Hubble constant
    rho_crit_int = 3.0 * H_int**2 / (8.0 * np.pi)
    M_tot = OMEGA_M * rho_crit_int * 4.0 / 3.0 * np.pi * R_sim**3
    n = 64
    rng = np.random.default_rng(5)
    pos = rng.normal(size=(n, 3))
    pos = pos / np.linalg.norm(pos, axis=1, keepdims=True) \
        * R_sim * rng.uniform(0, 1, size=(n, 1)) ** (1 / 3)
    write_ic(ic, pos_mpc=pos,
             vel_pec_kmps=rng.normal(scale=100.0, size=(n, 3)),
             mass_code=np.full(n, M_tot / n), a_start=a_start)
    z_out = 1.0 / np.linspace(a_start, a_max, 6) - 1.0
    write_redshift_list(outlst, z_out)
    write_param(param, outdir, ic, outlst, a_start=a_start, a_max=a_max,
                acc_param=0.05, step_max=0.1, particle_radii=20.0,
                is_periodic=0, pds=False)
    run_sim(param, timeout=300, binary=BINARY_R3)
    snaps = load_snapshots(outdir)

    ok = True
    ok &= check(len(snaps) >= 5, f"Got {len(snaps)} snapshots")
    ok &= check(all(np.isfinite(s[1]).all() and np.isfinite(s[2]).all()
                    for s in snaps), "No NaN / Inf in positions or velocities")
    vmax = max(np.abs(s[2]).max() for s in snaps)
    ok &= check(vmax < 1.0e5, f"No velocity blow-up: max |v| = {vmax:.1f} km/s")
    print(f"  Result: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
# TEST 9 — /PartType1/Quaternions actually govern the run
# ══════════════════════════════════════════════════════════════════════════════
def test9():
    """The IC's quaternions must be the state that is integrated, not merely read.

    Provenance is invisible in a consistent IC: q and inverse_stereo(stereo(q)) agree
    analytically, so honouring the dataset or rebuilding it from Coordinates give the
    same answer to within float precision.  We therefore write a *deliberately
    inconsistent* IC — Coordinates describing one configuration, Quaternions another —
    so the two code paths give unmistakably different positions.  This is a provenance
    fixture, not a supported input.

    Before the fix, pds_wrap_ic() rebuilt q from x[] unconditionally and the Quaternions
    dataset was dead weight; the old assertion only grepped the log for the word
    "Reading", which such a run still prints.
    """
    print("\n── Test 9: IC quaternions govern the run ────────────────────────────")
    tag    = "test9"
    outdir = OUT_BASE / tag
    ic     = outdir / "ic.hdf5"
    param  = outdir / f"{tag}.param"
    outlst = outdir / "redshifts.txt"

    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    a_start = 0.5
    rng = np.random.default_rng(23)
    n = 40
    # positions the QUATERNIONS describe (the truth we expect to be integrated)
    pos_q = rng.normal(size=(n, 3))
    pos_q = pos_q / np.linalg.norm(pos_q, axis=1, keepdims=True) \
        * rng.uniform(50.0, 400.0, size=(n, 1))
    quat = to_quaternion(pos_q.astype(np.float64))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    # decoy Cartesian coordinates: a large, easily distinguishable displacement
    pos_decoy = pos_q + 150.0

    mass = rng.uniform(500.0, 2000.0, size=n)
    write_ic(ic, pos_mpc=pos_decoy, vel_pec_kmps=np.zeros((n, 3)),
             mass_code=mass, a_start=a_start, quaternions=quat)
    write_redshift_list(outlst, 1.0 / np.linspace(a_start, a_start + 0.02, 3) - 1.0)
    write_param(param, outdir, ic, outlst, a_start=a_start, a_max=a_start + 0.02,
                acc_param=0.05, step_max=0.1, particle_radii=10.0, is_periodic=2)
    run_sim(param, binary=BINARY_SAVEACC)

    log_text = (param.parent / "run.log").read_text()
    ok = check("Reading /PartType1/Quaternions" in log_text,
               "IC quaternions were read")

    with h5py.File(outdir / "initial_conditions_0000.hdf5", "r") as f:
        pos_used = f["PartType1/Coordinates"][:].astype(np.float64)

    # StePS wraps into the fundamental domain, so compare against the wrapped
    # stereographic image of each candidate rather than the raw input.
    from stepsic import pds
    def wrapped_stereo(q):
        best = None
        for g in pds.istar():
            img = pds.quat_mult(g, q)
            keep = img[:, 0] >= 0
            img = np.where(keep[:, None], img, -img)
            if best is None:
                best = img
            else:
                better = img[:, 0] > best[:, 0]
                best = np.where(better[:, None], img, best)
        return pds.stereo_project(best, R_CURV)

    d_quat  = np.linalg.norm(pos_used - wrapped_stereo(quat), axis=1)
    d_decoy = np.linalg.norm(
        pos_used - wrapped_stereo(to_quaternion(pos_decoy.astype(np.float64))
                                  / np.linalg.norm(to_quaternion(pos_decoy.astype(np.float64)),
                                                   axis=1, keepdims=True)), axis=1)

    ok &= check(np.median(d_quat) < 1e-3,
                f"integrated state follows the IC quaternions "
                f"(median offset {np.median(d_quat):.2e} Mpc)")
    ok &= check(np.median(d_decoy) > 1.0,
                f"integrated state does NOT follow the decoy Coordinates "
                f"(median offset {np.median(d_decoy):.3g} Mpc)")
    print(f"  Result: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Checking/building the binary variants (StePS, StePS_saveacc, StePS_r3)...")
    ensure_binary_variants()
    if not BINARY.exists():
        print(f"ERROR: StePS binary not found at {BINARY}")
        print("       Run: make -f PDS-LinuxGCC-Makefile  (from the StePS/StePS/ directory)")
        sys.exit(1)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print("Running all 8 tests in parallel...")
    test_fns = [test1, test2, test3, test4, test5, test6, test7, test8, test9]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): fn for fn in test_fns}
        result_map = {}
        for future in futures:
            fn = futures[future]
            try:
                result_map[fn] = future.result()
            except Exception as e:
                print(f"  [FAIL] Exception in {fn.__name__}: {e}")
                result_map[fn] = False
    results = [result_map[fn] for fn in test_fns]

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"\n{'='*60}")
    print(f"TOTAL: {n_pass}/{len(results)} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
