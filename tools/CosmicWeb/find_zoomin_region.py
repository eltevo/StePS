#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import sys
import os
from scipy.signal import fftconvolve
# adding ../Utils/ to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from inputoutput import load_density_hdf5


_VERSION="v0.0.1.0"
_AUTHOR="Gabor Racz"
_YEAR="2026"
_DESCRIPTION="StePS Zoom-in Target Identifier: Find optimal zoom-in regions in complex geometries (T3, S1R2, R3) based on precomputed density fields and snapshot data."


def load_snapshot_data(snapshot_path):
    """Reads coordinates, IDs, and Masses from a single or split HDF5 snapshot."""
    all_pos, all_ids, all_mass = [], [], []
    header_attrs = {}

    if snapshot_path.endswith('.0.hdf5'):
        base_path = snapshot_path[:-7]
        file_idx = 0
        while True:
            cur_path = f"{base_path}.{file_idx}.hdf5"
            if not os.path.exists(cur_path):
                break
            with h5py.File(cur_path, 'r') as f:
                if file_idx == 0:
                    header_attrs = dict(f['Header'].attrs)
                
                pos = f['PartType1/Coordinates'][:]
                all_pos.append(pos)
                all_ids.append(f['PartType1/ParticleIDs'][:])
                mass = f['PartType1/Masses'][:]
                all_mass.append(mass)
            file_idx += 1
    else:
        with h5py.File(snapshot_path, 'r') as f:
            header_attrs = dict(f['Header'].attrs)
            pos = f['PartType1/Coordinates'][:]
            all_pos.append(pos)
            all_ids.append(f['PartType1/ParticleIDs'][:])
            mass = f['PartType1/Masses'][:]
            all_mass.append(mass)

    pos = np.concatenate(all_pos, axis=0)
    ids = np.concatenate(all_ids, axis=0)
    masses = np.concatenate(all_mass, axis=0)
    return pos, ids, masses, header_attrs

def create_spherical_kernel(radius, dx, dy, dz):
    """Creates a 3D spherical top-hat kernel normalized to sum=1."""
    px, py, pz = int(np.ceil(radius/dx)), int(np.ceil(radius/dy)), int(np.ceil(radius/dz))
    rx = np.arange(-px, px+1) * dx
    ry = np.arange(-py, py+1) * dy
    rz = np.arange(-pz, pz+1) * dz
    
    mx, my, mz = np.meshgrid(rx, ry, rz, indexing='ij')
    r2 = mx**2 + my**2 + mz**2
    
    mask = (r2 <= radius**2).astype(np.float32)
    vol_cells = np.sum(mask)
    if vol_cells == 0:
        raise ValueError("Grid resolution is too low to resolve the target radius.")
    
    return mask / vol_cells, mask

def apply_topology_convolution(field, kernel, geom):
    """Pads the domain based on geometry to ensure boundary conditions are respected."""
    kx, ky, kz = kernel.shape
    px, py, pz = kx//2, ky//2, kz//2
    
    if geom == 'T3':
        padded = np.pad(field, ((px, px), (py, py), (pz, pz)), mode='wrap')
    elif geom == 'S1R2':
        padded = np.pad(field, ((px, px), (py, py), (0, 0)), mode='edge')
        padded = np.pad(padded, ((0, 0), (0, 0), (pz, pz)), mode='wrap')
    elif geom == 'R3':
        padded = np.pad(field, ((px, px), (py, py), (pz, pz)), mode='edge')
    else:
        raise ValueError(f"Unknown geometry: {geom}")
        
    return fftconvolve(padded, kernel, mode='valid')

def calc_periodic_distance(p1, p2, geom, L, Lz):
    """Calculates spatial distance respecting periodic boundaries."""
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    dz = abs(p1[2] - p2[2])
    
    if geom == 'T3':
        dx = min(dx, L - dx)
        dy = min(dy, L - dy)
        dz = min(dz, L - dz)
    elif geom == 'S1R2':
        dz = min(dz, Lz - dz)
        
    return np.sqrt(dx**2 + dy**2 + dz**2)

def periodic_1d_com(pos_1d, masses, box_size):
    """Mass-weighted phase-mapping center of mass for a 1D periodic array."""
    theta = pos_1d * (2.0 * np.pi / box_size)
    xi = np.cos(theta)
    zeta = np.sin(theta)
    
    xi_mean = np.average(xi, weights=masses)
    zeta_mean = np.average(zeta, weights=masses)
    
    theta_mean = np.arctan2(zeta_mean, xi_mean)
    return (theta_mean * box_size / (2.0 * np.pi)) % box_size

def calculate_topology_com(positions, masses, geom, L, Lz):
    """Calculates mass-weighted 3D center of mass combining standard and periodic means appropriately."""
    com = np.zeros(3)
    if geom == 'T3':
        com[0] = periodic_1d_com(positions[:, 0], masses, L)
        com[1] = periodic_1d_com(positions[:, 1], masses, L)
        com[2] = periodic_1d_com(positions[:, 2], masses, L)
    elif geom == 'S1R2':
        com[0] = np.average(positions[:, 0], weights=masses)
        com[1] = np.average(positions[:, 1], weights=masses)
        com[2] = periodic_1d_com(positions[:, 2], masses, Lz)
    elif geom == 'R3':
        com = np.average(positions, axis=0, weights=masses)
    return com

def identify_regions(density_file, snap_file, ic_file, target_r, target_delta=0.0, ncand=4, rmax_frac=0.75):
    # Loading the Density Field
    data, meta = load_density_hdf5(density_file)
    print(data.keys())
    delta = data['overdensity']
    Nx, Ny, Nz = meta['grid_res_Nx'], meta['grid_res_Ny'], meta['grid_res_Nz']
    L = meta['L']
    Lz = meta.get('Lz', L)
    geom = meta['geometry']

    # Reconstructing Grid Coordinates strictly based on Geometry
    dx = L / Nx
    dy = L / Ny
    dz = Lz / Nz if geom == 'S1R2' else L / Nz

    if geom == 'R3':
        x_1d = np.linspace(-L/2 + dx/2, L/2 - dx/2, Nx)
        y_1d = np.linspace(-L/2 + dy/2, L/2 - dy/2, Ny)
        z_1d = np.linspace(-L/2 + dz/2, L/2 - dz/2, Nz)
    elif geom == 'S1R2':
        x_1d = np.linspace(-L/2 + dx/2, L/2 - dx/2, Nx)
        y_1d = np.linspace(-L/2 + dy/2, L/2 - dy/2, Ny)
        z_1d = np.linspace(0 + dz/2, Lz - dz/2, Nz)
    elif geom == 'T3':
        x_1d = np.linspace(0 + dx/2, L - dx/2, Nx)
        y_1d = np.linspace(0 + dy/2, L - dy/2, Ny)
        z_1d = np.linspace(0 + dz/2, L - dz/2, Nz)
        
    X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

    print(f"--- Scanning {geom} volume with spherical mask (R = {target_r:.2f} Mpc) ---")
    if geom in ['R3', 'S1R2']:
        print(f"--- Limiting search to {rmax_frac*100:.1f}% of the simulation boundary ---")
    
    # Convolution
    kernel_1x, _ = create_spherical_kernel(target_r, dx, dy, dz)
    kernel_1_5x, _ = create_spherical_kernel(target_r * 1.5, dx, dy, dz)

    delta_mean = apply_topology_convolution(delta, kernel_1x, geom)
    delta_mean_15 = apply_topology_convolution(delta, kernel_1_5x, geom)
    delta_sq_mean = apply_topology_convolution(delta**2, kernel_1x, geom)
    variance = np.clip(delta_sq_mean - delta_mean**2, 0, None)

    # Filtering edges and exclusion zones
    valid_mask = np.ones_like(delta, dtype=bool)
    pad_r = target_r * 1.5 # Safe padding against grid edges
    
    if geom == 'R3':
        dist_from_center = np.sqrt(X**2 + Y**2 + Z**2)
        valid_mask &= (X >= x_1d[0] + pad_r) & (X <= x_1d[-1] - pad_r)
        valid_mask &= (Y >= y_1d[0] + pad_r) & (Y <= y_1d[-1] - pad_r)
        valid_mask &= (Z >= z_1d[0] + pad_r) & (Z <= z_1d[-1] - pad_r)
        valid_mask &= (dist_from_center <= (L / 2.0) * rmax_frac)
        
    elif geom == 'S1R2':
        dist_from_center = np.sqrt(X**2 + Y**2) # Distance to Z-axis
        valid_mask &= (X >= x_1d[0] + pad_r) & (X <= x_1d[-1] - pad_r)
        valid_mask &= (Y >= y_1d[0] + pad_r) & (Y <= y_1d[-1] - pad_r)
        valid_mask &= (dist_from_center <= (L / 2.0) * rmax_frac)
        
    elif geom == 'T3':
        dist_from_center = np.sqrt((X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2) # Reference only

    # Finding top candidates
    diff = np.abs(delta_mean - target_delta)
    diff[~valid_mask] = np.inf
    
    flat_indices = np.argsort(diff.flatten())
    candidates = []
    
    print("--- Searching for optimal isolated candidates ---")
    for idx in flat_indices:
        if len(candidates) >= ncand:
            break
        if diff.flatten()[idx] == np.inf:
            break
            
        coord_idx = np.unravel_index(idx, diff.shape)
        pos_cand = np.array([X[coord_idx], Y[coord_idx], Z[coord_idx]])
        
        separated = True
        for c in candidates:
            dist_to_c = calc_periodic_distance(pos_cand, c['pos'], geom, L, Lz)
            if dist_to_c < (2.0 * target_r):
                separated = False
                break
                
        if separated:
            candidates.append({
                'pos': pos_cand,
                'delta': delta_mean[coord_idx],
                'var': variance[coord_idx],
                'delta_15': delta_mean_15[coord_idx],
                'dist': dist_from_center[coord_idx]
            })

    if not candidates:
        print("No valid candidates found! Try reducing target_r, relaxing boundaries, or increasing rmax_frac.")
        return

    # User selection
    print(f"\n--- TOP {len(candidates)} CANDIDATES ---")
    for i, c in enumerate(candidates):
        print(f"[{i+1}] Center: ({c['pos'][0]:.2f}, {c['pos'][1]:.2f}, {c['pos'][2]:.2f}) Mpc")
        print(f"    Overdensity (1.0 R): {c['delta']:.4e} | Target diff: {abs(c['delta'] - target_delta):.4e}")
        print(f"    Overdensity (1.5 R): {c['delta_15']:>8.5f}")
        print(f"    Variance    (1.0 R): {c['var']:>8.5f}")
        print(f"    Dist to Center/Axis: {c['dist']:.2f} Mpc\n")

    choice = -1
    while choice not in range(1, len(candidates) + 1):
        try:
            choice = int(input(f"Select a candidate (1-{len(candidates)}): "))
        except ValueError:
            pass
    selected = candidates[choice-1]

    # Particle ID matching
    print(f"\n--- Loading Snapshot to find IDs at z=0 ---")
    pos_snap, ids_snap, _, _ = load_snapshot_data(snap_file) # Masses not needed here
    
    dx_snap = np.abs(pos_snap[:, 0] - selected['pos'][0])
    dy_snap = np.abs(pos_snap[:, 1] - selected['pos'][1])
    dz_snap = np.abs(pos_snap[:, 2] - selected['pos'][2])
    
    if geom == 'T3':
        dx_snap = np.minimum(dx_snap, L - dx_snap)
        dy_snap = np.minimum(dy_snap, L - dy_snap)
        dz_snap = np.minimum(dz_snap, L - dz_snap)
    elif geom == 'S1R2':
        dz_snap = np.minimum(dz_snap, Lz - dz_snap)
        
    dist_snap = np.sqrt(dx_snap**2 + dy_snap**2 + dz_snap**2)
    
    target_ids = ids_snap[dist_snap <= target_r]
    print(f"Found {len(target_ids)} particles within the selected {target_r} Mpc sphere.")

    print(f"--- Loading IC to calculate Center of Mass ---")
    pos_ic, ids_ic, masses_ic, ic_header = load_snapshot_data(ic_file)
    
    print("Matching Particle IDs...")
    sort_idx = np.argsort(ids_ic)
    sorted_ids_ic = ids_ic[sort_idx]
    match_indices_sorted = np.searchsorted(sorted_ids_ic, target_ids)
    valid_matches = sorted_ids_ic[match_indices_sorted] == target_ids
    match_indices = sort_idx[match_indices_sorted[valid_matches]]
    
    target_ic_pos = pos_ic[match_indices]
    target_ic_masses = masses_ic[match_indices]
    
    # Calculate Final IC Coordinates
    com = calculate_topology_com(target_ic_pos, target_ic_masses, geom, L, Lz)

    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Geometry: {geom}")
    print(f"Number of tracked particles: {len(target_ic_pos)}")
    print(f"Target Initial Condition CoM: x={com[0]:.4f}, y={com[1]:.4f}, z={com[2]:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    print(f"StePS Zoom-in region finder {_VERSION} by {_AUTHOR}, {_YEAR}")
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("-d", "--density", type=str, required=True, help="Path to precomputed HDF5 density field")
    parser.add_argument("-s", "--snap", type=str, required=True, help="Path to z=0 snapshot (or .0.hdf5 link)")
    parser.add_argument("-i", "--ic", type=str, required=True, help="Path to Initial Conditions snapshot")
    parser.add_argument("-r", "--radius", type=float, required=True, help="Search radius [Mpc]")
    parser.add_argument("--delta", type=float, default=0.0, help="Target overdensity (default=0.0)")
    parser.add_argument("--ncand", type=int, default=4, help="Number of candidate regions to output (default=4, max=50)")
    parser.add_argument("--rmax_frac", type=float, default=0.75, help="Max distance from center as fraction of sim radius (R3/S1R2 only)")

    args = parser.parse_args()
    ncand = max(1, min(50, args.ncand))
    identify_regions(args.density, args.snap, args.ic, args.radius, args.delta, ncand, args.rmax_frac)