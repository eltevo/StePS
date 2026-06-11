#!/usr/bin/env python3

#*******************************************************************************#
#  millennium_render.py - "Millennium simulation / Gadget" style renderer for   #
#     STEreographically Projected cosmological Simulations                      #
#     (works for any StePS HDF5 snapshot, including S^3/I* PDS topology runs)   #
#    Copyright (C) 2026 Gabor Racz, Istvan Csabai                               #
#                                                                               #
#    This program is free software; you can redistribute it and/or modify       #
#    it under the terms of the GNU General Public License as published by       #
#    the Free Software Foundation; either version 2 of the License, or          #
#    (at your option) any later version.                                        #
#                                                                               #
#    This program is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#    GNU General Public License for more details.                               #
#*******************************************************************************#

"""
Render StePS snapshots in the classic "Millennium simulation" style:
SPH-like adaptively smoothed, logarithmic projected density maps on a dark
background with an age/redshift info box and a physical scale bar.

No external SPH library is needed; the adaptive smoothing is done with a
multi-scale histogram + Gaussian filter stack (numpy/scipy only).

Typical use (see also PDS_Millennium_View.ipynb in this folder):

    from millennium_render import load_snapshot, render_snapshot
    fig, ax = render_snapshot("snapshot_0010.hdf5")
    fig.savefig("snapshot_0010.png")
"""

__author__ = "Gabor Racz, Istvan Csabai"
__version__ = "0.1.0"
__year__ = "2026"

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

# 1/H0 in Gyr for H0 = 1 km/s/Mpc
_HUBBLE_TIME_GYR = 977.792
# 1 Mpc in million light-years
_MPC_IN_MLY = 3.2616


def load_snapshot(fname, filter_nan=True):
    """Load a StePS HDF5 snapshot into a dict (pos, mass, vel, hdr)."""
    with h5py.File(fname, "r") as f:
        pos = f["PartType1/Coordinates"][:]
        mass = f["PartType1/Masses"][:]
        vel = f["PartType1/Velocities"][:]
        hdr = {k: v for k, v in f["Header"].attrs.items()}
    hdr = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in hdr.items()}
    if filter_nan:
        ok = np.isfinite(pos).all(axis=1)
        if not ok.all():
            print(f"WARNING: {len(pos) - ok.sum()}/{len(pos)} particles have "
                  f"non-finite positions in {fname}; they were dropped.")
        pos, mass, vel = pos[ok], mass[ok], vel[ok]
    return dict(pos=pos, mass=mass, vel=vel, hdr=hdr)


def age_of_universe_gyr(z, h0, omega_m, omega_lambda):
    """Age of a flat LCDM universe at redshift z, in Gyr (analytic formula)."""
    a = 1.0 / (1.0 + z)
    h0_inv_gyr = _HUBBLE_TIME_GYR / h0
    return (2.0 / 3.0) * h0_inv_gyr / np.sqrt(omega_lambda) * \
        np.arcsinh(np.sqrt(omega_lambda / omega_m) * a ** 1.5)


def adaptive_density_map(pos2d, mass, extent, npix=1024, n_ngb=24,
                         hmin_pix=0.8, hmax_frac=0.08):
    """
    SPH-like adaptively smoothed surface density map.

    Each particle gets a smoothing length h = half the distance to its
    n_ngb-th nearest neighbour (in the 2D projection).  Particles are grouped
    into logarithmic h bins; each group is deposited on a mass-weighted 2D
    histogram and blurred with a Gaussian of the matching width, then the
    maps are summed.  This reproduces the Gadget/Millennium look without an
    SPH library and runs in O(N log N).

    pos2d  : (N,2) projected coordinates [Mpc]
    mass   : (N,) particle masses
    extent : (xmin, xmax, ymin, ymax) of the map [Mpc]
    npix   : output map resolution (npix x npix)
    returns: (npix, npix) surface density [mass / Mpc^2], y-axis first index
    """
    xmin, xmax, ymin, ymax = extent
    pix = (xmax - xmin) / npix

    tree = cKDTree(pos2d)
    dist, _ = tree.query(pos2d, k=n_ngb)
    h = 0.5 * dist[:, -1]
    h = np.clip(h, hmin_pix * pix, hmax_frac * (xmax - xmin))

    # logarithmic smoothing-length bins, ~2 bins per octave
    nbin = max(1, int(np.ceil(np.log2(h.max() / h.min()) * 2)))
    hbin_edges = np.geomspace(h.min() * 0.999, h.max() * 1.001, nbin + 1)
    ibin = np.digitize(h, hbin_edges) - 1

    img = np.zeros((npix, npix))
    bins = [np.linspace(xmin, xmax, npix + 1), np.linspace(ymin, ymax, npix + 1)]
    for b in range(nbin):
        sel = ibin == b
        if not sel.any():
            continue
        hist, _, _ = np.histogram2d(pos2d[sel, 0], pos2d[sel, 1],
                                    bins=bins, weights=mass[sel])
        sigma = np.sqrt(hbin_edges[b] * hbin_edges[b + 1]) / pix
        img += gaussian_filter(hist, sigma=sigma)

    return img.T / pix ** 2   # transpose: histogram2d returns x as first axis


def _pick_scalebar_length(half_size):
    """Round scale-bar length ~1/4 of the half map size."""
    target = half_size / 2.0
    candidates = np.array([10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000])
    return float(candidates[np.argmin(np.abs(candidates - target))])


def render_map(img, extent, hdr=None, cmap="magma", dyn_range=4.5,
               vmax_percentile=99.9, figsize=10.0, info_box=True,
               scale_bar=True, title=None, ax=None):
    """
    Display an adaptively smoothed density map in Millennium style.

    img       : surface density map from adaptive_density_map()
    extent    : (xmin, xmax, ymin, ymax) [Mpc]
    hdr       : snapshot header dict (for the age/redshift info box)
    dyn_range : decades of surface density shown below the maximum
    returns   : (fig, ax)
    """
    floor = img[img > 0].min() if (img > 0).any() else 1.0
    logimg = np.log10(img + floor)
    vmax = np.percentile(logimg, vmax_percentile)
    vmin = vmax - dyn_range

    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize, figsize))
    else:
        fig = ax.figure
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(logimg, origin="lower", extent=extent, cmap=cmap,
              norm=Normalize(vmin=vmin, vmax=vmax), interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    if info_box and hdr is not None:
        z = float(hdr["Redshift"])
        h0 = float(hdr["HubbleParam"]) * 100.0
        age = age_of_universe_gyr(z, h0, float(hdr["Omega0"]),
                                  float(hdr["OmegaLambda"]))
        ax.text(0.015, 0.985,
                f"Age = {age:.3f} billion years\nRedshift = {z:.4f}",
                transform=ax.transAxes, ha="left", va="top",
                color="black", fontsize=9 * figsize / 10,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                          boxstyle="square,pad=0.35"))

    if scale_bar:
        xmin, xmax, ymin, ymax = extent
        bar = _pick_scalebar_length((xmax - xmin) / 2.0)
        x1 = xmax - 0.06 * (xmax - xmin)
        x0 = x1 - bar
        y0 = ymin + 0.085 * (ymax - ymin)
        ax.plot([x0, x1], [y0, y0], color="white", lw=1.8 * figsize / 10)
        ax.text((x0 + x1) / 2, y0 + 0.015 * (ymax - ymin), f"{bar:.0f}Mpc",
                ha="center", va="bottom", color="white",
                fontsize=9 * figsize / 10)
        ax.text((x0 + x1) / 2, y0 - 0.015 * (ymax - ymin),
                f"({bar * _MPC_IN_MLY:.2f} million light-years)",
                ha="center", va="top", color="white",
                fontsize=5.5 * figsize / 10)

    if title is not None:
        ax.set_title(title, color="white", fontsize=11 * figsize / 10)
    return fig, ax


def render_snapshot(fname, plane="XY", slice_thickness=None, slice_center=0.0,
                    r_plot=None, npix=1024, n_ngb=24, cmap="magma",
                    dyn_range=4.5, figsize=10.0, ax=None, **kwargs):
    """
    One-call Millennium-style rendering of a StePS HDF5 snapshot.

    plane           : "XY", "XZ" or "YZ" projection plane
    slice_thickness : if set, only particles within +-thickness/2 of
                      slice_center along the third axis are used;
                      None projects the full volume (best for small N)
    r_plot          : half size of the map [Mpc]; default = SimulationRadius
    returns         : (fig, ax)
    """
    snap = load_snapshot(fname)
    pos, mass, hdr = snap["pos"], snap["mass"], snap["hdr"]

    axes = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[plane.upper()]
    if slice_thickness is not None:
        sel = np.abs(pos[:, axes[2]] - slice_center) <= slice_thickness / 2.0
        pos, mass = pos[sel], mass[sel]

    if r_plot is None:
        r_plot = float(hdr.get("SimulationRadius", np.abs(pos).max()))
    extent = (-r_plot, r_plot, -r_plot, r_plot)

    inside = (np.abs(pos[:, axes[0]]) < r_plot * 1.2) & \
             (np.abs(pos[:, axes[1]]) < r_plot * 1.2)
    pos2d = pos[np.ix_(inside, [axes[0], axes[1]])]
    img = adaptive_density_map(pos2d, mass[inside], extent,
                               npix=npix, n_ngb=n_ngb)
    return render_map(img, extent, hdr=hdr, cmap=cmap, dyn_range=dyn_range,
                      figsize=figsize, ax=ax, **kwargs)


def make_evolution_gif(snapshot_files, out_gif, fps=3, **render_kwargs):
    """Render every snapshot and assemble an animated GIF (needs Pillow)."""
    from PIL import Image
    import io
    frames = []
    for f in snapshot_files:
        fig, _ = render_snapshot(f, **render_kwargs)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor="black",
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"Wrote {out_gif} ({len(frames)} frames)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: ./millennium_render.py <snapshot.hdf5> [out.png]")
        sys.exit(2)
    fig, _ = render_snapshot(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "millennium_render.png"
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight",
                pad_inches=0.02)
    print(f"Wrote {out}")
