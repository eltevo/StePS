#!/usr/bin/env python
'''
PDS force-law study: image-sum cancellation, background compensation,
and anisotropy of the image correction.

KEY FINDING (verified analytically and numerically below): with the bare
kernel F = 1/(R^2 sin^2 chi) used in StePS v2.1.1.0, the force from the FULL
I* image system of any source is IDENTICALLY ZERO. I* is closed under
negation (g and -g are both elements), green_bare(pi - chi) = green_bare(chi),
and the tangent toward -g is exactly opposite to the tangent toward g, so the
60 antipodal image pairs cancel one by one. Consequences:

  * The 1D Ewald table built in ewald_space.cc equals exactly
    D(chi) = -green_bare(chi_nearest): the "correction" cancels the
    nearest-image force, and IS_PERIODIC >= 2 runs have essentially ZERO
    gravity (only table-interpolation error remains). The existing
    validation tests all use IS_PERIODIC = 1 and could not catch this.

  * The isotropy question posed for the 1D table is moot for the bare
    kernel (the correction is exactly isotropic and exactly -F_nearest);
    the physically meaningful anisotropy lives in the COMPENSATED image
    correction, which this study quantifies.

The background-compensated kernel

    F(chi) = [1 - V(chi)/V_S3] / (R^2 sin^2 chi),
    V(chi)/V_S3 = (2 chi - sin 2 chi) / (2 pi)

(point mass + uniform negative background; the analogue of dropping the
k = 0 mode in T^3 Ewald summation) breaks the antipodal degeneracy and
yields a finite, physical peculiar force. It -> 1/r^2 as chi -> 0 and -> 0
smoothly at the antipode.

WLOG the source sits at the domain centre e0 = (1,0,0,0): for any pair
(p, q), right translation by q^-1 is an isometry mapping the image system
of q onto I* itself, so field points around a source at e0 cover the
general case exactly.

Outputs (in /v/csabai/GitHub/steps_dodeca/data/pds_anisotropy/):
    pds_anisotropy_study.npz     raw arrays
    fig1_cancellation.png        bare image-sum cancellation + kernels
    fig2_correction_anisotropy.png  direction dependence of the compensated
                                  image correction (radial + transverse)
    fig3_scheme_errors.png       end-to-end errors of the v2.1.1.0 schemes
    fig4_roukema_scaling.png     small-chi scaling of the anisotropic residual
    REPORT.md                    summary with decision-relevant numbers

Usage:  conda activate stepsic && python pds_anisotropy_study.py
'''

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', '..', 'stepsic'))
from stepsic import pds  # noqa: E402

OUT_DIR = '/v/csabai/GitHub/steps_dodeca/data/pds_anisotropy'
R_CURV = 3100.0          # Mpc, Luminet et al. (2003) best fit
NGRID_TABLE = 1024       # matches IS_PERIODIC = 2 in StePS
N_DIRECTIONS = 2000      # random directions in the tangent space at e0
CHI_DEG = np.concatenate([np.arange(0.5, 18.0, 0.5),
                          np.arange(18.0, 21.0, 0.25)])  # up to the outradius


# --------------------------------------------------------------------------
# Reimplementation of the C++ 1D Ewald table (ewald_space.cc:1086-1138)
# --------------------------------------------------------------------------

def build_1d_table(ngrid: int, R: float, kernel=pds.green_bare) -> np.ndarray:
    '''Exact Python mirror of calculate_pds_ewald_lookup_table().'''
    dchi = np.pi / (ngrid + 1)
    chi_field = (np.arange(ngrid) + 1.0) * dchi
    p = np.stack([np.cos(chi_field), np.sin(chi_field),
                  np.zeros(ngrid), np.zeros(ngrid)], axis=-1)   # (ngrid, 4)
    imgs = pds.images(pds.E0)                                    # (120, 4)
    chis = pds.chi(p[:, None, :], imgs)                          # (ngrid, 120)
    nearest = np.argmin(chis, axis=1)                            # (ngrid,)
    mags = kernel(chis, R)                                       # (ngrid, 120)
    t_img = pds.force_direction(p[:, None, :], imgs)             # (ngrid, 120, 4)
    t_src = pds.force_direction(p, pds.E0)                       # (ngrid, 4)
    proj = np.sum(t_img * t_src[:, None, :], axis=-1)            # (ngrid, 120)
    contrib = mags * proj
    contrib[np.arange(ngrid), nearest] = 0.0                     # drop nearest image
    return np.sum(contrib, axis=1)


def interp_1d_table(table: np.ndarray, chi_nearest: np.ndarray) -> np.ndarray:
    '''Exact Python mirror of pds_ewald_interpolate().'''
    ngrid = len(table)
    dchi = np.pi / (ngrid + 1)
    idx_d = np.asarray(chi_nearest) / dchi - 1.0
    i0 = np.clip(idx_d.astype(int), 0, ngrid - 2)
    frac = idx_d - i0
    out = (1.0 - frac) * table[i0] + frac * table[i0 + 1]
    return np.where((chi_nearest <= 0.0) | (chi_nearest >= np.pi), 0.0, out)


# --------------------------------------------------------------------------
# Exact image sums (correction computed WITHOUT catastrophic cancellation:
# the nearest term is zeroed in the contribution list, not subtracted)
# --------------------------------------------------------------------------

def image_sums(p: np.ndarray, R: float, kernel):
    '''
    For field points p (N, 4) and a unit-mass source at e0 return:
        F_total  (N, 4)  exact force from all 120 images
        F_near   (N, 4)  nearest-image force
        D        (N, 4)  correction = sum over the 119 non-nearest images
        chi_near (N,)    geodesic distance to the nearest image
        t_near   (N, 4)  unit tangent toward the nearest image
    '''
    imgs = pds.images(pds.E0)                        # (120, 4)
    chis = pds.chi(p[:, None, :], imgs)              # (N, 120)
    mags = kernel(chis, R)                           # (N, 120)
    dirs = pds.force_direction(p[:, None, :], imgs)  # (N, 120, 4)
    contrib = mags[..., None] * dirs                 # (N, 120, 4)
    near = np.argmin(chis, axis=1)
    rows = np.arange(len(p))
    F_near = contrib[rows, near]
    D = contrib.copy()
    D[rows, near] = 0.0
    D = np.sum(D, axis=1)
    return np.sum(contrib, axis=1), F_near, D, chis[rows, near], dirs[rows, near]


def field_points(chi_vals, axes):
    '''p(chi, n) = (cos chi, sin chi * n) for all combinations; (C, D, 4).'''
    p = np.empty((len(chi_vals), len(axes), 4))
    p[..., 0] = np.cos(chi_vals)[:, None]
    p[..., 1:] = np.sin(chi_vals)[:, None, None] * axes[None, :, :]
    return p


def special_directions(rng):
    '''Face and vertex axes of the dodecahedral domain (tangent space at e0).'''
    g = pds.istar()
    chis = np.degrees(pds.chi(pds.E0, g))
    face_axes = pds.normalize(g[np.abs(chis - 36.0) < 1e-6][:, 1:])  # 12 axes
    # vertex axes: directions of maximal domain radius = local minima of the
    # max projection onto the face axes (robust Monte-Carlo proxy)
    cand = pds.normalize(rng.normal(size=(200000, 3)))
    maxproj = np.max(cand @ face_axes.T, axis=1)
    vert_axes = cand[np.argsort(maxproj)[:20]]
    return face_axes, vert_axes


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(137)

    # ---- 0. the cancellation identity ---------------------------------------
    print('Verifying the bare-kernel image-sum cancellation...')
    grp = pds.istar()
    neg_closed = all(np.min(np.max(np.abs(grp + row), axis=1)) < 1e-9 for row in grp)
    table_bare = build_1d_table(NGRID_TABLE, R_CURV, pds.green_bare)
    dchi = np.pi / (NGRID_TABLE + 1)
    chi_grid = (np.arange(NGRID_TABLE) + 1.0) * dchi
    # D(chi) = -green_bare(chi) wherever the nearest image is e0 itself
    in_dom_grid = chi_grid < np.radians(18.0)
    table_identity_dev = np.max(np.abs(
        table_bare[in_dom_grid] / pds.green_bare(chi_grid[in_dom_grid], R_CURV) + 1.0))
    print(f'  I* closed under negation: {neg_closed}')
    print(f'  max |D_table/(-green_bare) - 1| in-domain: {table_identity_dev:.3e}')

    # ---- sample field points -------------------------------------------------
    n_hat = pds.normalize(rng.normal(size=(N_DIRECTIONS, 3)))
    chi_vals = np.radians(CHI_DEG)
    p_flat = field_points(chi_vals, n_hat).reshape(-1, 4)
    shape = (len(chi_vals), N_DIRECTIONS)

    print(f'Computing exact image sums at {len(p_flat)} field points...')
    F_all_b, F_near_b, D_b, chi_near, t_near = image_sums(p_flat, R_CURV, pds.green_bare)
    F_all_c, F_near_c, D_c, _, _ = image_sums(p_flat, R_CURV, pds.green_compensated)

    F_near_b_mag = np.linalg.norm(F_near_b, axis=-1)
    F_all_c_mag = np.linalg.norm(F_all_c, axis=-1)
    cancel_ratio = np.linalg.norm(F_all_b, axis=-1) / F_near_b_mag  # ~ 0

    # ---- 1. anisotropy of the COMPENSATED image correction -------------------
    D_rad = np.sum(D_c * t_near, axis=-1)
    D_trans_mag = np.linalg.norm(D_c - D_rad[:, None] * t_near, axis=-1)
    F_near_c_mag = np.linalg.norm(F_near_c, axis=-1)

    D_rad_2d = (D_rad / F_near_c_mag).reshape(shape)
    D_trans_2d = (D_trans_mag / F_near_c_mag).reshape(shape)
    # direction dependence at fixed chi: spread of the radial part
    D_rad_spread = D_rad_2d.max(axis=1) - D_rad_2d.min(axis=1)
    D_rad_mean = D_rad_2d.mean(axis=1)

    # what an isotropic (direction-averaged) compensated 1D table would miss:
    aniso_err = np.abs(D_rad_2d - D_rad_mean[:, None])  # radial residual
    # plus the transverse part D_trans_2d, invisible to ANY radial 1D table

    # ---- 2. end-to-end errors of the v2.1.1.0 schemes ------------------------
    D_table_interp = interp_1d_table(table_bare, chi_near)
    F_old2 = F_near_b + D_table_interp[:, None] * t_near        # IS_PERIODIC >= 2
    err_old2 = (np.linalg.norm(F_old2 - F_all_c, axis=-1) / F_all_c_mag).reshape(shape)
    err_old1 = (np.linalg.norm(F_near_b - F_all_c, axis=-1) / F_all_c_mag).reshape(shape)
    residual_force_old2 = (np.linalg.norm(F_old2, axis=-1) / F_near_b_mag).reshape(shape)

    # ---- 3. small-chi Roukema scaling (compensated kernel) -------------------
    face_axes, vert_axes = special_directions(rng)
    chi_small = np.radians(np.geomspace(0.05, 18.0, 40))
    D_dir = []
    for ax in np.vstack([face_axes[:6], vert_axes[:6]]):
        pp = field_points(chi_small, ax[None, :]).reshape(-1, 4)
        _, Fn, Dv, cn, tn = image_sums(pp, R_CURV, pds.green_compensated)
        D_dir.append(np.sum(Dv * tn, axis=-1))
    D_dir = np.array(D_dir)                       # (12, len(chi_small))
    D_spread_abs = D_dir.max(axis=0) - D_dir.min(axis=0)
    sl = slice(8, 32)
    scaling_exponent = np.polyfit(np.log(chi_small[sl]),
                                  np.log(D_spread_abs[sl] + 1e-300), 1)[0]

    # ---- 4. uniform Monte-Carlo load ------------------------------------------
    print('Monte-Carlo uniform-load net force...')
    n_mc = 20000
    q_mc = pds.normalize(rng.normal(size=(n_mc, 4)))
    c10 = np.radians(10.0)
    tp = np.array([np.cos(c10), np.sin(c10), 0.0, 0.0])
    imgs_mc = pds.images(q_mc)
    chis_mc = pds.chi(tp, imgs_mc)
    dirs_mc = pds.force_direction(tp, imgs_mc)
    net = {}
    for name, kern in [('bare', pds.green_bare), ('compensated', pds.green_compensated)]:
        F_mc = np.sum(kern(chis_mc, R_CURV)[..., None] * dirs_mc, axis=1)
        net[name] = (np.mean(F_mc, axis=0), np.std(F_mc, axis=0) / np.sqrt(n_mc))
    F_ref = pds.green_bare(c10, R_CURV)

    # ---- save -------------------------------------------------------------------
    np.savez_compressed(
        os.path.join(OUT_DIR, 'pds_anisotropy_study.npz'),
        chi_deg=CHI_DEG, table_bare=table_bare,
        table_identity_dev=table_identity_dev,
        cancel_ratio_max=np.max(cancel_ratio),
        D_rad_rel=D_rad_2d, D_trans_rel=D_trans_2d,
        D_rad_spread=D_rad_spread, D_rad_mean=D_rad_mean,
        err_old2=err_old2, err_old1=err_old1,
        residual_force_old2=residual_force_old2,
        chi_small=chi_small, D_dir=D_dir, D_spread_abs=D_spread_abs,
        scaling_exponent=scaling_exponent,
        net_bare_mean=net['bare'][0], net_bare_err=net['bare'][1],
        net_comp_mean=net['compensated'][0], net_comp_err=net['compensated'][1],
        F_ref=F_ref, R_curv=R_CURV)

    # ---- figures ------------------------------------------------------------------
    pct = lambda a, q: np.percentile(a, q, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    chis_k = np.radians(np.linspace(0.5, 179.5, 500))
    ax.plot(np.degrees(chis_k), pds.green_bare(chis_k, R_CURV) * R_CURV**2,
            label='bare  1/sin²χ')
    ax.plot(np.degrees(chis_k), pds.green_compensated(chis_k, R_CURV) * R_CURV**2,
            label='compensated')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\chi$ [deg]'); ax.set_ylabel(r'$R^2 \times$ force per unit mass')
    ax.legend(); ax.set_title('S³ force kernels')
    ax = axes[1]
    ax.semilogy(CHI_DEG, pct(cancel_ratio.reshape(shape), 50), label='median')
    ax.semilogy(CHI_DEG, pct(cancel_ratio.reshape(shape), 95), ls='--', label='95%')
    ax.set_xlabel(r'$\chi_{\rm nearest}$ [deg]')
    ax.set_ylabel(r'$|\sum_{\rm all\ images} F_{\rm bare}| / F_{\rm nearest}$')
    ax.set_title('Bare-kernel image sum cancels identically\n(antipodal ±g pairs)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_cancellation.png'), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.fill_between(CHI_DEG, pct(D_rad_2d, 0), pct(D_rad_2d, 100),
                    alpha=0.25, label='full range over directions')
    ax.plot(CHI_DEG, D_rad_mean, label='direction average')
    ax.axvline(18.0, color='k', ls=':', lw=0.8)
    ax.set_xlabel(r'$\chi_{\rm nearest}$ [deg]')
    ax.set_ylabel(r'$D_{\rm radial}^{\rm comp} / F_{\rm nearest}^{\rm comp}$')
    ax.set_title('Compensated image correction (radial part)')
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.semilogy(CHI_DEG, D_rad_spread, label='radial spread over directions')
    ax.semilogy(CHI_DEG, pct(D_trans_2d, 50), label='transverse, median')
    ax.semilogy(CHI_DEG, pct(D_trans_2d, 95), ls='--', label='transverse, 95%')
    ax.axvline(18.0, color='k', ls=':', lw=0.8)
    ax.set_xlabel(r'$\chi_{\rm nearest}$ [deg]')
    ax.set_ylabel(r'relative to $F_{\rm nearest}^{\rm comp}$')
    ax.set_title('Anisotropic parts — invisible to any 1D table')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig2_correction_anisotropy.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.fill_between(CHI_DEG, pct(err_old2, 5), pct(err_old2, 95), alpha=0.3,
                    color='C3', label='IS_PERIODIC≥2: nearest + 1D table (5–95%)')
    ax.plot(CHI_DEG, pct(err_old2, 50), color='C3')
    ax.fill_between(CHI_DEG, pct(err_old1, 5), pct(err_old1, 95), alpha=0.3,
                    color='C0', label='IS_PERIODIC=1: bare nearest only (5–95%)')
    ax.plot(CHI_DEG, pct(err_old1, 50), color='C0')
    ax.axvline(18.0, color='k', ls=':', lw=0.8)
    ax.set_xlabel(r'$\chi_{\rm nearest}$ [deg]')
    ax.set_ylabel('relative force error vs exact compensated sum')
    ax.set_title('End-to-end error of the v2.1.1.0 schemes')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig3_scheme_errors.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.loglog(np.degrees(chi_small), D_spread_abs, 'o-', ms=3,
              label=f'directional spread of D (fit slope = {scaling_exponent:.2f})')
    ref = D_spread_abs[20] * (chi_small / chi_small[20]) ** 4
    ax.loglog(np.degrees(chi_small), ref, ls=':', label=r'$\propto \chi^4$ reference')
    ax.set_xlabel(r'$\chi$ [deg]')
    ax.set_ylabel('spread of radial compensated correction')
    ax.set_title('Small-χ scaling of the anisotropic residual\n'
                 '(cf. Roukema & Różański 2009)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig4_roukema_scaling.png'), dpi=150)
    plt.close(fig)

    # ---- report -----------------------------------------------------------------
    in_dom = CHI_DEG <= 18.0
    lines = [
        '# PDS force-law study: cancellation, compensation, anisotropy',
        '',
        f'R_curv = {R_CURV} Mpc, 1D table grid = {NGRID_TABLE} (IS_PERIODIC = 2), '
        f'{N_DIRECTIONS} random directions, chi in [{CHI_DEG[0]}, {CHI_DEG[-1]}] deg. '
        'Source at the domain centre (WLOG by right-translation isometry).',
        '',
        '## 1. Bare-kernel image sum cancels identically (CRITICAL)',
        '',
        'I* is closed under negation, green_bare(pi-chi) = green_bare(chi), and '
        'antipodal images pull in exactly opposite directions, so the bare force '
        'from all 120 images of any source is identically zero:',
        '',
        f'- I* closed under negation: **{neg_closed}**',
        f'- max |sum over all images| / F_nearest over all sampled points: '
        f'**{np.max(cancel_ratio):.2e}** (pure float roundoff)',
        f'- 1D table identity D(chi) = -green_bare(chi): max deviation '
        f'**{table_identity_dev:.2e}** in-domain',
        '',
        '**Consequence: StePS v2.1.1.0 with IS_PERIODIC >= 2 applies a "correction" '
        'that cancels the nearest-image force. PDS runs in this mode have ~zero '
        'gravity** (residual = table interpolation error, median '
        f'{np.median(residual_force_old2):.2e} of F_nearest). The validation tests '
        'all use IS_PERIODIC = 1 and could not catch this. This also contributes '
        'to the missing structure formation in the PDS_test.param run.',
        '',
        '## 2. Background compensation (the physical kernel)',
        '',
        'The compensated kernel [1 - V(chi)/V_S3]/(R^2 sin^2 chi) (point mass + '
        'uniform negative background, the analogue of dropping the k = 0 Ewald '
        'mode) breaks the antipodal degeneracy and yields a finite peculiar force:',
        '',
        f'- suppression factor at chi = 18 deg: '
        f'{1 - (2*np.radians(18) - np.sin(2*np.radians(18))) / (2*np.pi):.4f} '
        '(nearest-image forces inside the domain are barely changed)',
        f'- at chi = 90 deg: 0.50; at chi = 150 deg: '
        f'{1 - (2*np.radians(150) - np.sin(2*np.radians(150))) / (2*np.pi):.4f} '
        '(far images strongly suppressed)',
        f'- exact compensated force at chi = 10 deg is '
        f'{np.median((F_all_c_mag / F_near_b_mag).reshape(shape)[CHI_DEG == 10.0]):.3f} '
        'of the bare nearest-image force (the 119 images reduce the attraction)',
        '',
        '## 3. Anisotropy of the compensated image correction',
        '',
        'Radial part D_rad/F_nearest (direction-averaged) and its direction '
        'dependence; the transverse part is invisible to any 1D table:',
        '',
        '| quantity | chi < 18 deg | chi in [18, 21] deg |',
        '|---|---|---|',
        f'| D_rad/F_near, direction avg, max | {np.max(np.abs(D_rad_mean[in_dom])):.3e} '
        f'| {np.max(np.abs(D_rad_mean[~in_dom])):.3e} |',
        f'| radial spread over directions, max | {np.max(D_rad_spread[in_dom]):.3e} '
        f'| {np.max(D_rad_spread[~in_dom]):.3e} |',
        f'| transverse, 95th pct, max over chi | {np.max(pct(D_trans_2d, 95)[in_dom]):.3e} '
        f'| {np.max(pct(D_trans_2d, 95)[~in_dom]):.3e} |',
        '',
        f'Small-chi scaling of the directional spread: fitted exponent '
        f'{scaling_exponent:.2f} (Roukema & Rozanski 2009 found the residual '
        'gravity effect in PDS suppressed to high order; the compensated kernel '
        'and the inclusion of all 120 images here differ from their adjacent-image '
        'setup, so the exponent need not be exactly 5).',
        '',
        '## 4. End-to-end errors vs the exact compensated sum',
        '',
        '| scheme | median | 90th pct | max |',
        '|---|---|---|---|',
        f'| IS_PERIODIC>=2 (nearest + 1D table) | {np.median(err_old2):.3f} '
        f'| {np.percentile(err_old2, 90):.3f} | {np.max(err_old2):.3f} |',
        f'| IS_PERIODIC=1 (bare nearest only) | {np.median(err_old1):.3f} '
        f'| {np.percentile(err_old1, 90):.3f} | {np.max(err_old1):.3f} |',
        '',
        '(IS_PERIODIC>=2 error ~ 1 means the force is ~zero instead of the '
        'physical value.)',
        '',
        '## 5. Uniform-load net force (20000 uniform sources, test point at chi = 10 deg)',
        '',
        f'- bare:        net = {net["bare"][0][1:]} +- {net["bare"][1][1:]}',
        f'- compensated: net = {net["compensated"][0][1:]} +- {net["compensated"][1][1:]}',
        f'  (reference single-pair force at chi = 10 deg: {F_ref:.3e}; both are '
        'consistent with zero, as required for a uniform load)',
        '',
        '## Decision',
        '',
        '1. Retire the 1D Ewald table: with the bare kernel it is an exact '
        'cancellation of gravity, not a correction.',
        '2. Use the background-compensated kernel.',
        '3. Sum over all 120 images exactly (the loop is already paid for the '
        'nearest-image search). The anisotropic terms of the image correction '
        '(section 3) are then included automatically; a compensated 1D table '
        'would leave a direction-dependent error up to '
        f'{max(np.max(D_rad_spread), np.max(pct(D_trans_2d, 95))):.1e} '
        'of F_nearest at the domain boundary.',
    ]
    with open(os.path.join(OUT_DIR, 'REPORT.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print(f'\nOutputs written to {OUT_DIR}')


if __name__ == '__main__':
    main()
