#!/usr/bin/env python3
"""
PDS validation harness: compare a StePS Poincare-dodecahedral (PDS, S^3/I*) run to a
flat Gadget4 gold-standard run built from the SAME initial-condition realization, and to
linear theory.  Phase-0 regression gate for the PDS development roadmap.

Why Gadget4 (not StePS-T^3) as the reference: a StePS-T^3 run (testCubic128) was found to
over-grow ~3x vs Gadget at z=10 from the identical IC (low-res Ewald table), so it is NOT
a trustworthy flat reference.  Gadget4 tracks linear growth to ~1% and is the gold
standard here.

Measures
--------
1. Geodesic displacement R*chi from the IC for the PDS run vs comoving displacement for
   the flat Gadget run (median).  Coordinate-invariant; the PRIMARY growth gate.  Robust
   while <50% of PDS particles wrap (z >~ 2); below that use --image-min (slower).
2. Large-scale P(k) in the matched central physical region (shot-noise subtracted) -- the
   precise clustering comparison; valid at all z.
3. Counts-in-cells sigma^2 (secondary; resolution-confounded -- see the note printed).

Usage
-----
    python validate_growth.py [--pds DIR] [--gad DIR] [--pds-ic FILE] [--gad-ic FILE]
                              [--rcurv 3100] [--zref 15] [--image-min]
"""
import argparse, glob, numpy as np, h5py
from scipy.integrate import quad

# ---- cosmology (test128 / gadget128_flat) ------------------------------------------
OM, OL = 0.3106, 0.6894
HUBBLE_H = 0.67702
A_START = 1.0 / 32.0          # z = 31

def Efunc(a): return np.sqrt(OM / a**3 + OL)
def Dgrow(a):
    I, _ = quad(lambda ap: 1.0 / (ap * Efunc(ap))**3, 1e-8, a)
    return 2.5 * OM * Efunc(a) * I

# ---- loaders -----------------------------------------------------------------------
def _load(fname):
    with h5py.File(fname, 'r') as h:
        x = h['PartType1/Coordinates'][:].astype(np.float64)
        z = float(h['Header'].attrs['Redshift'])
        ids = h['PartType1/ParticleIDs'][:]
        L = float(h['Header'].attrs['BoxSize'])
    o = np.argsort(ids)
    return x[o], z, L

def load_gadget(fname):
    """Gadget Mpc/h -> physical Mpc."""
    x, z, L = _load(fname); return x / HUBBLE_H, z, L / HUBBLE_H

def pds_to_quat(x, R):
    r2 = (x**2).sum(1); den = R**2 + r2
    q = np.empty((len(x), 4)); q[:, 0] = (R**2 - r2) / den
    q[:, 1:] = 2 * R * x / den[:, None]; return q

def pds_to_physical(x, R):
    """PDS stereographic coords -> geodesic normal (physical comoving) coords."""
    r = np.sqrt((x**2).sum(1)); r = np.where(r < 1e-9, 1e-9, r)
    chi = 2 * np.arctan2(r, R)
    return (R * chi / r)[:, None] * x

# ---- geodesic displacement ---------------------------------------------------------
def pds_geodesic_disp(qic, q, R, image_min=False, chunk=200000):
    """median physical geodesic displacement R*chi.  image_min=True minimizes over the
    120 I* images (handles >50% wrapping at low z) in chunks to bound memory."""
    if not image_min:
        d = np.clip(np.einsum('ij,ij->i', qic, q), -1, 1)
        return np.median(R * np.arccos(d))
    from stepsic import pds                      # reuse I* group + quaternion algebra
    g = pds.istar()                              # (120,4)
    out = np.empty(len(q))
    for s in range(0, len(q), chunk):
        qs = q[s:s+chunk]; imgs = pds.quat_mult(g, qs[:, None, :])   # (n,120,4)
        dots = np.clip(np.einsum('nij,nj->ni', imgs, qic[s:s+chunk]), -1, 1)
        out[s:s+chunk] = R * np.arccos(dots).min(1)
    return np.median(out)

def flat_disp(xic, x, L):
    dx = x - xic; dx -= L * np.round(dx / L)
    return np.median(np.sqrt((dx**2).sum(1)))

# ---- clustering --------------------------------------------------------------------
def sigma2(x, half, ng):
    """counts-in-cells variance, shot-noise subtracted, in a centered physical cube."""
    x = x - np.median(x, 0)
    m = (np.abs(x) < half).all(1); p = x[m] + half; Lc = 2 * half
    idx = np.clip((p / Lc * ng).astype(int), 0, ng - 1)
    g = np.zeros((ng,) * 3); np.add.at(g, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    nb = g.mean(); return g.var() / nb**2 - 1.0 / nb

def pk_central(x, half, ng):
    """large-scale P(k) in a centered physical cube; shot-noise subtracted.
    Returns (k_array, P_array) in (1/Mpc, Mpc^3)."""
    x = x - np.median(x, 0); m = (np.abs(x) < half).all(1); xx = x[m] + half
    L = 2 * half; N = len(xx)
    g = np.zeros((ng, ng, ng)); idx = np.clip((xx / L * ng).astype(int), 0, ng - 1)
    np.add.at(g, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    d = g / g.mean() - 1.0; f = np.fft.fftn(d)
    P = np.abs(f)**2 * L**3 / ng**6
    kf = 2 * np.pi / L
    kk = np.sqrt(sum(np.meshgrid(*[np.fft.fftfreq(ng, 1/ng)**2]*3, indexing='ij'))) * kf
    shot = L**3 / N
    kbins = np.arange(1, ng // 2)
    ks, Ps = [], []
    for q in kbins:
        sel = (kk >= (q - 0.5) * kf) & (kk < (q + 0.5) * kf)
        if sel.sum() > 0:
            ks.append(q * kf); Ps.append(P[sel].mean() - shot)
    return np.array(ks), np.array(Ps)

# ---- main comparison ---------------------------------------------------------------
def run(pds_dir, gad_dir, pds_ic, gad_ic, R, zref=15.0, image_min=False,
        half=300.0, ng_s2=8, ng_pk=16):
    xic, _, _ = _load(pds_ic); qic = pds_to_quat(xic, R)
    pds_s = []
    for f in sorted(glob.glob(pds_dir + '/snapshot_0*.hdf5')):
        x, z, _ = _load(f)
        pds_s.append((z, pds_geodesic_disp(qic, pds_to_quat(x, R), R, image_min),
                      sigma2(pds_to_physical(x, R), half, ng_s2), pds_to_physical(x, R)))
    gxic, _, _ = load_gadget(gad_ic); gad_s = []
    for f in sorted(glob.glob(gad_dir + '/snapshot_0*.hdf5')):
        x, z, L = load_gadget(f)
        gad_s.append((z, flat_disp(gxic, x, L), sigma2(x, half, ng_s2), x))

    def near(series, z): return min(series, key=lambda t: abs(t[0] - z))
    pr, gr = near(pds_s, zref), near(gad_s, zref)
    linr = Dgrow(1/(1+zref)) - Dgrow(A_START)

    print(f"\n  PDS ({pds_dir.split('/')[-1]}) vs Gadget4 gold standard")
    print(f"  geodesic-displacement growth (normalized to z={zref:g}) + central P(k) ratio")
    print(f"  {'z':>5} | {'GAD d/lin':>9} {'PDS d/lin':>9} {'PDS/GAD':>8} | "
          f"{'P_PDS/P_GAD (low-k)':>18} | {'sig2 PDS/GAD':>12}")
    verdict = []
    for z in [15, 10, 5, 3, 2, 1, 0]:
        pz, gz = near(pds_s, z), near(gad_s, z)
        if abs(pz[0]-z) > 0.5: continue
        lin = Dgrow(1/(1+z)) - Dgrow(A_START)
        gdl = (gz[1]/gr[1])/(lin/linr); pdl = (pz[1]/pr[1])/(lin/linr)
        dratio = (pz[1]/pr[1])/(gz[1]/gr[1])
        kp, Pp = pk_central(pz[3], half, ng_pk); kg, Pg = pk_central(gz[3], half, ng_pk)
        nlow = max(1, len(Pp)//3)
        pkr = np.nanmean(Pp[:nlow]) / np.nanmean(Pg[:nlow]) if np.nanmean(Pg[:nlow]) > 0 else np.nan
        s2r = pz[2]/gz[2] if gz[2] > 0 else np.nan
        print(f"  {z:5d} | {gdl:9.2f} {pdl:9.2f} {dratio:8.2f} | {pkr:18.2f} | {s2r:12.2f}")
        if z <= 5: verdict.append(dratio)
    worst = max(verdict)
    print(f"\n  GATE (displacement growth, z<=5): max PDS/Gadget = {worst:.2f}  "
          f"-> {'PASS (<=1.3)' if worst <= 1.3 else 'FAIL (>1.3)'}")
    print("  NOTE: sigma^2 is resolution-confounded (PDS coarser particle load); the")
    print("        large-scale P(k) ratio and the displacement growth are the reliable measures.")
    return pds_s, gad_s

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pds', default='/scratch/csabai/test128_fix')
    ap.add_argument('--gad', default='/scratch/csabai/gadget128_flat')
    ap.add_argument('--pds-ic', default='/scratch/csabai/test128/snapshot_0000.hdf5')
    ap.add_argument('--gad-ic', default='/scratch/csabai/gadget128_flat/ic_mpch.hdf5')
    ap.add_argument('--rcurv', type=float, default=3100.0)
    ap.add_argument('--zref', type=float, default=15.0)
    ap.add_argument('--image-min', action='store_true',
                    help='minimize geodesic displacement over I* images (needed below z~2)')
    a = ap.parse_args()
    run(a.pds, a.gad, a.pds_ic, a.gad_ic, a.rcurv, a.zref, a.image_min)
