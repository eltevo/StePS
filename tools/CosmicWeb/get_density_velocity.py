#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import sys
import os

# adding ../Utils/ to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from inputoutput import save_density_hdf5

_VERSION="v0.0.1.0"
_YEAR="2026"
_AUTHOR="Gabor Racz"
_DESCRIPTION="StePS Density & Velocity Field Calculator"

def apply_boundary(idx, N, mode):
    """Handles periodic ('wrap') or open ('edge') boundary indices."""
    if mode == 'wrap':
        return idx % N, np.ones_like(idx, dtype=bool)
    else:
        valid = (idx >= 0) & (idx < N)
        return np.clip(idx, 0, N - 1), valid

def assign_to_grid(pos, vals, Nx, Ny, Nz, bounds, pad_modes, scheme='CIC'):
    """Vectorized property assignment to a 3D grid using NGP, CIC, or TSC."""
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    dx, dy, dz = (x_max - x_min)/Nx, (y_max - y_min)/Ny, (z_max - z_min)/Nz
    
    u = np.clip((pos[:, 0] - x_min) / dx, 0, Nx - 1e-5)
    v = np.clip((pos[:, 1] - y_min) / dy, 0, Ny - 1e-5)
    w = np.clip((pos[:, 2] - z_min) / dz, 0, Nz - 1e-5)

    if scheme == 'NGP':
        shifts = [0]
        weights_func = lambda t, d: [np.ones_like(t)]
        u_base, v_base, w_base = np.floor(u).astype(int), np.floor(v).astype(int), np.floor(w).astype(int)
    elif scheme == 'CIC':
        shifts = [0, 1]
        u_base, v_base, w_base = np.floor(u - 0.5).astype(int), np.floor(v - 0.5).astype(int), np.floor(w - 0.5).astype(int)
        def weights_func(base, coords):
            d = coords - (base + 0.5)
            return [1.0 - d, d]
    elif scheme == 'TSC':
        shifts = [-1, 0, 1]
        u_base, v_base, w_base = np.round(u).astype(int), np.round(v).astype(int), np.round(w).astype(int)
        def weights_func(base, coords):
            d = coords - base
            return [0.5 * (0.5 - d)**2, 0.75 - d**2, 0.5 * (0.5 + d)**2]
    else:
        raise ValueError("Scheme must be NGP, CIC, or TSC")

    Wx = weights_func(u_base, u)
    Wy = weights_func(v_base, v)
    Wz = weights_func(w_base, w)

    D = vals.shape[1] if vals.ndim > 1 else 1
    grid_shape = (Nx, Ny, Nz) if D == 1 else (Nx, Ny, Nz, D)
    grid = np.zeros(grid_shape, dtype=np.float32)

    for ix, sx in enumerate(shifts):
        idx_x, val_x = apply_boundary(u_base + sx, Nx, pad_modes[0])
        for iy, sy in enumerate(shifts):
            idx_y, val_y = apply_boundary(v_base + sy, Ny, pad_modes[1])
            for iz, sz in enumerate(shifts):
                idx_z, val_z = apply_boundary(w_base + sz, Nz, pad_modes[2])
                
                valid = val_x & val_y & val_z
                weight = Wx[ix] * Wy[iy] * Wz[iz]
                
                if D == 1:
                    contribution = (vals * weight)[valid]
                    np.add.at(grid, (idx_x[valid], idx_y[valid], idx_z[valid]), contribution)
                else:
                    for dim in range(D):
                        contribution = (vals[:, dim] * weight)[valid]
                        np.add.at(grid[..., dim], (idx_x[valid], idx_y[valid], idx_z[valid]), contribution)
                        
    return grid

def load_snapshot_data(snapshot_path, calc_velocity=False):
    """Robustly reads single or multi-file split HDF5 snapshots."""
    all_pos, all_vel, all_mass = [], [], []
    header_attrs = {}

    if snapshot_path.endswith('.0.hdf5'):
        print("\tSnapshot is stored in multiple files. Stitching datasets...")
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
                if calc_velocity:
                    all_vel.append(f['PartType1/Velocities'][:])
                if 'PartType1/Masses' in f:
                    all_mass.append(f['PartType1/Masses'][:])
                elif 'MassTable' in f['Header'].attrs:
                    all_mass.append(np.ones(len(pos), dtype=np.float32) * f['Header'].attrs['MassTable'][1])
                else:
                    all_mass.append(np.ones(len(pos), dtype=np.float32))
            file_idx += 1
    else:
        print(f"\tReading single snapshot file: {snapshot_path}")
        with h5py.File(snapshot_path, 'r') as f:
            header_attrs = dict(f['Header'].attrs)
            pos = f['PartType1/Coordinates'][:]
            all_pos.append(pos)
            if calc_velocity:
                all_vel.append(f['PartType1/Velocities'][:])
            if 'PartType1/Masses' in f:
                all_mass.append(f['PartType1/Masses'][:])
            elif 'MassTable' in f['Header'].attrs:
                all_mass.append(np.ones(len(pos), dtype=np.float32) * f['Header'].attrs['MassTable'][1])
            else:
                all_mass.append(np.ones(len(pos), dtype=np.float32))

    pos = np.concatenate(all_pos, axis=0)
    vel = np.concatenate(all_vel, axis=0) if calc_velocity else None
    masses = np.concatenate(all_mass, axis=0)
    return pos, vel, masses, header_attrs

def calculate_density_field(snapshot_path, geometry, grid_res, scheme, L_arg=None, Lz_arg=None, calc_velocity=False, output_path="density.h5"):
    print(f"--- Loading snapshot data from {snapshot_path} ---")
    pos, vel, masses, header = load_snapshot_data(snapshot_path, calc_velocity)
    
    # Computing the total mass in the simulation for global density normalization
    total_sim_mass = float(np.sum(masses))
    print(f"Loaded {len(pos)} particles total. Total Simulation Mass: {total_sim_mass:.4e} (10^11 Msol)")

    def clean_attr(key):
        val = header.get(key)
        if isinstance(val, np.ndarray): return val
        return np.array([val]) if val is not None else None

    # Calculating analytical global volume based strictly on geometry limits
    if geometry == 'R3':
        R_val = clean_attr('SimulationRadius')
        if R_val is None: raise ValueError("SimulationRadius missing in header for R3.")
        R_sim = float(R_val[0])
        V_sim = (4.0 / 3.0) * np.pi * (R_sim ** 3)
        print(f"Global Layout -> R3 Sphere: Radius = {R_sim:.2f} Mpc | Total Vol = {V_sim:.2f} Mpc^3")
        
    elif geometry == 'S1R2':
        R_val = clean_attr('SimulationRadius')
        if R_val is None: raise ValueError("SimulationRadius missing in header for S1R2.")
        R_sim = float(R_val[0])
        
        Lz_val = clean_attr('Lz') if 'Lz' in header else clean_attr('BoxSize')
        if Lz_val is None: raise ValueError("Lz/BoxSize missing in header for S1R2.")
        Lz_sim = float(Lz_val[2]) if len(Lz_val) >= 3 else float(Lz_val[0])
        
        V_sim = np.pi * (R_sim ** 2) * Lz_sim
        print(f"Global Layout -> S1R2 Cylinder: Radius = {R_sim:.2f} Mpc, Length = {Lz_sim:.2f} Mpc | Total Vol = {V_sim:.2f} Mpc^3")
        
    elif geometry == 'T3':
        L_val = clean_attr('BoxSize')
        if L_val is None: raise ValueError("BoxSize missing in header for T3.")
        L_sim = float(L_val[0])
        V_sim = L_sim ** 3
        print(f"Global Layout -> T3 Periodic Cube: Side = {L_sim:.2f} Mpc | Total Vol = {V_sim:.2f} Mpc^3")
    else:
        raise ValueError("Geometry must be 'R3', 'S1R2', or 'T3'")

    # Analytical Cosmic Mean Density Reference
    rho_cosmic = total_sim_mass / V_sim
    print(f"Calculated Global Background Cosmic Density (rho_cosmic): {rho_cosmic:.6e} (10^11 Msol / Mpc^3)")

    # Defining grid bounds and padding rules based on geometry
    if geometry == 'R3':
        L = L_arg if L_arg is not None else 2.0 * R_sim
        Nx = Ny = Nz = grid_res
        bounds = ((-L/2.0, L/2.0), (-L/2.0, L/2.0), (-L/2.0, L/2.0))
        pad_modes = ['edge', 'edge', 'edge']
        
    elif geometry == 'S1R2':
        L = L_arg if L_arg is not None else 2.0 * R_sim
        Lz = Lz_arg if Lz_arg is not None else Lz_sim
        Nx = Ny = grid_res
        Nz = int(grid_res * Lz / L)
        bounds = ((-L/2.0, L/2.0), (-L/2.0, L/2.0), (0.0, Lz))
        pad_modes = ['edge', 'edge', 'wrap']
        
    elif geometry == 'T3':
        L = L_arg if L_arg is not None else L_sim
        Nx = Ny = Nz = grid_res
        bounds = ((0.0, L), (0.0, L), (0.0, L))
        pad_modes = ['wrap', 'wrap', 'wrap']

    # Filtering particles to the target sub-volume defined by bounds
    print("--- Filtering out particles outside target sub-volume ---")
    in_bounds = (
        (pos[:, 0] >= bounds[0][0]) & (pos[:, 0] <= bounds[0][1]) &
        (pos[:, 1] >= bounds[1][0]) & (pos[:, 1] <= bounds[1][1]) &
        (pos[:, 2] >= bounds[2][0]) & (pos[:, 2] <= bounds[2][1])
    )
    
    pos = pos[in_bounds]
    masses = masses[in_bounds]
    if calc_velocity:
        vel = vel[in_bounds]
        
    print(f"Kept {len(pos)} high-resolution particles inside target bounds.")
    if len(pos) == 0:
        raise ValueError("Zero particles found inside selection box bounds.")

    # Grid assignment of masses (and velocities if requested) using the specified scheme
    print(f"--- Assigning masses to grid ({Nx}x{Ny}x{Nz}) using {scheme} ---")
    mass_grid = assign_to_grid(pos, masses, Nx, Ny, Nz, bounds, pad_modes, scheme)
    
    # Calculationg the overdensity from the global cosmic density
    dx = (bounds[0][1] - bounds[0][0]) / Nx
    dy = (bounds[1][1] - bounds[1][0]) / Ny
    dz = (bounds[2][1] - bounds[2][0]) / Nz
    V_cell = dx * dy * dz
    
    # Exact mass a single cell should contain in a perfectly homogenous background universe
    expected_background_mass_per_cell = rho_cosmic * V_cell
    
    # Overdensity delta = (rho_cell / rho_cosmic) - 1.0 = (mass_cell / mass_background) - 1.0
    delta = (mass_grid / expected_background_mass_per_cell) - 1.0
    print(f"Grid Normalization -> Cell Volume: {V_cell:.4f} Mpc^3 | Unperturbed Cell Mass: {expected_background_mass_per_cell:.4f}")
    print(f"Sub-volume Overdensity Range: Min = {np.min(delta):.4f}, Max = {np.max(delta):.4f}")

    # Compute mass-weighted velocity grid if requested
    velocity_grid = None
    if calc_velocity:
        print(f"--- Computing Mass-Weighted Velocity Grid ---")
        momentum_field = vel * masses[:, np.newaxis]
        momentum_grid = assign_to_grid(pos, momentum_field, Nx, Ny, Nz, bounds, pad_modes, scheme)
        
        velocity_grid = np.zeros_like(momentum_grid)
        nonzero_mass = mass_grid > 0
        for dim in range(3):
            velocity_grid[nonzero_mass, dim] = momentum_grid[nonzero_mass, dim] / mass_grid[nonzero_mass]

    # Save metadata
    metadata = {
        'geometry': geometry,
        'grid_res_Nx': Nx, 'grid_res_Ny': Ny, 'grid_res_Nz': Nz,
        'scheme': scheme,
        'L': L,
        'Lz': Lz if geometry == 'S1R2' else L,
        'rho_cosmic': rho_cosmic
    }

    save_density_hdf5(output_path, overdensity=delta, velocity=velocity_grid, metadata=metadata)
    print("Done!")

if __name__ == "__main__":
    print(f"StePS Density Field Calculator {_VERSION} by {_AUTHOR}, {_YEAR}")
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to HDF5 snapshot (or .0.hdf5 split link)")
    parser.add_argument("--geom", type=str, choices=['R3', 'S1R2', 'T3'], required=True, help="Geometry type")
    parser.add_argument("--res", type=int, default=128, help="Grid resolution per axis")
    parser.add_argument("--scheme", type=str, choices=['NGP', 'CIC', 'TSC'], default='CIC', help="Mass assignment scheme")
    parser.add_argument("--L", type=float, default=None, help="Crop width for the target central high-res volume [Mpc]")
    parser.add_argument("--Lz", type=float, default=None, help="Explicit selection length along Z for S1R2 [Mpc]")
    parser.add_argument("--vel", action='store_true', help="Compute and save mass-weighted velocity fields")
    parser.add_argument("-o", "--out", type=str, default="density.h5", help="Output filename")

    args = parser.parse_args()
    calculate_density_field(args.input, args.geom, args.res, args.scheme, args.L, args.Lz, args.vel, args.out)