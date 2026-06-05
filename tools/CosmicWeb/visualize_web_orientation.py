#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import argparse
import sys
import os

# adding ../Utils/ to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from inputoutput import load_cosmic_web_hdf5

_VERSION="v0.0.2.0"
_YEAR="2026"
_AUTHOR="Gabor Racz"
_DESCRIPTION="StePS Cosmic Web 3D visualizer"

def compute_averaged_orientations(vectors, mask, smooth_sigma):
    """
    Solves the sign-ambiguity (+v == -v) by constructing the structure tensor,
    smoothing it, and extracting the new dominant orientation.
    """
    print(f"Smoothing orientation field (Tensor blur sigma={smooth_sigma})...")
    # Mask vectors: set to 0 outside structure to prevent bleeding
    V = vectors * mask[..., np.newaxis]
    
    # Construct 6 unique components of the symmetric 3x3 outer product tensor (v * v^T)
    Txx = V[..., 0] * V[..., 0]
    Tyy = V[..., 1] * V[..., 1]
    Tzz = V[..., 2] * V[..., 2]
    Txy = V[..., 0] * V[..., 1]
    Txz = V[..., 0] * V[..., 2]
    Tyz = V[..., 1] * V[..., 2]

    # Smooth the tensor components
    Txx = gaussian_filter(Txx, sigma=smooth_sigma)
    Tyy = gaussian_filter(Tyy, sigma=smooth_sigma)
    Tzz = gaussian_filter(Tzz, sigma=smooth_sigma)
    Txy = gaussian_filter(Txy, sigma=smooth_sigma)
    Txz = gaussian_filter(Txz, sigma=smooth_sigma)
    Tyz = gaussian_filter(Tyz, sigma=smooth_sigma)

    # Reconstruct 3x3 matrices and find the new principal eigenvector
    Nx, Ny, Nz = mask.shape
    smoothed_vectors = np.zeros_like(vectors)
    
    # Only calculate for masked regions to save time
    valid_idx = np.where(mask)
    for i, j, k in zip(*valid_idx):
        tensor = np.array([
            [Txx[i,j,k], Txy[i,j,k], Txz[i,j,k]],
            [Txy[i,j,k], Tyy[i,j,k], Tyz[i,j,k]],
            [Txz[i,j,k], Tyz[i,j,k], Tzz[i,j,k]]
        ])
        # linalg.eigh returns eigenvalues in ascending order
        w, v = np.linalg.eigh(tensor)
        # The dominant orientation is the eigenvector corresponding to the largest eigenvalue
        smoothed_vectors[i, j, k, :] = v[:, 2] 
        
    return smoothed_vectors

def trace_streamlines(vectors, mask, X, Y, Z, dx, dy, dz, num_seeds=200, max_steps=50):
    """
    Traces continuous curves through the discrete vector field.
    """
    print(f"Tracing up to {num_seeds} streamlines...")
    Nx, Ny, Nz = mask.shape
    valid_coords = np.argwhere(mask)
    
    if len(valid_coords) == 0:
        return []

    # Randomly select seed voxels from the valid mask
    np.random.seed(42)
    seed_indices = valid_coords[np.random.choice(len(valid_coords), min(num_seeds, len(valid_coords)), replace=False)]
    
    streamlines = []
    step_size = 0.5  # Step half a voxel at a time

    for seed in seed_indices:
        current_pos = seed.astype(float)
        line_x, line_y, line_z = [], [], []
        
        # Start direction
        ix, iy, iz = int(np.round(current_pos[0])), int(np.round(current_pos[1])), int(np.round(current_pos[2]))
        current_vec = vectors[ix, iy, iz]
        
        for _ in range(max_steps):
            ix, iy, iz = int(np.round(current_pos[0])), int(np.round(current_pos[1])), int(np.round(current_pos[2]))
            
            # Stop if we wander out of bounds or out of the filament mask
            if not (0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz) or not mask[ix, iy, iz]:
                break
                
            # Append physical coordinates
            line_x.append(X[ix, iy, iz])
            line_y.append(Y[ix, iy, iz])
            line_z.append(Z[ix, iy, iz])
            
            # Get local orientation
            next_vec = vectors[ix, iy, iz]
            
            # SIGN AMBIGUITY CHECK: Ensure the new vector points in the same general direction as the current heading
            if np.dot(current_vec, next_vec) < 0:
                next_vec = -next_vec
                
            # Step forward
            current_pos += next_vec * step_size
            current_vec = next_vec
            
        if len(line_x) > 3: # Only keep lines that actually went somewhere
            streamlines.append((line_x, line_y, line_z))
            
    return streamlines

def visualize_3d_orientation(h5_path, target_type='filament', mode='arrows', stride=2, slice_frac=0.2, max_streamlines=400):
    print(f"Loading data from {h5_path}...")
    try:
        data, attrs = load_cosmic_web_hdf5(h5_path)
        structure_type = data['structure_type']
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return

    Nx, Ny, Nz = structure_type.shape
    L = attrs['L']
    Lz = attrs.get('Lz', L)
    geometry = attrs.get('geometry', 'R3')
    print(f"Data loaded: Geometry={geometry}, Box Size={L} Mpc, Grid={Nx}x{Ny}x{Nz}")

    # Grid reconstruction
    x_edges, dx = np.linspace(0 if geometry=='T3' else -L/2, L if geometry=='T3' else L/2, Nx + 1, retstep=True)
    y_edges, dy = np.linspace(0 if geometry=='T3' else -L/2, L if geometry=='T3' else L/2, Ny + 1, retstep=True)
    if geometry == 'R3':
        z_edges, dz = np.linspace(-L/2, L/2, Nz + 1, retstep=True)
    else:
        z_edges, dz = np.linspace(0 if geometry!='S1R2' else 0, L if geometry=='T3' else (Lz if geometry=='S1R2' else L/2), Nz + 1, retstep=True)

    X, Y, Z = np.meshgrid(x_edges[:-1] + dx/2, y_edges[:-1] + dy/2, z_edges[:-1] + dz/2, indexing='ij')

    if target_type == 'filament':
        struct_val = 3
        vectors = data['dir_minor']
        color = '#E67E22'
    else:
        struct_val = 2
        vectors = data['dir_major']
        color = '#2E4053'

    mask = (structure_type == struct_val)

    # Apply Slicing
    z_mid = Nz // 2
    z_hw = max(1, int((Nz * slice_frac) / 2))
    slab_mask = np.zeros_like(mask, dtype=bool)
    slab_mask[:, :, z_mid-z_hw : z_mid+z_hw] = True
    mask = mask & slab_mask

    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    if mode == 'arrows':
        # Apply Tensor Smoothing and Downsampling
        smoothed_vectors = compute_averaged_orientations(vectors, mask, smooth_sigma=stride/2.0)
        
        # Now apply the stride cleanly
        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[::stride, ::stride, ::stride] = True
        final_mask = mask & stride_mask
        
        X_p, Y_p, Z_p = X[final_mask], Y[final_mask], Z[final_mask]
        U_p, V_p, W_p = smoothed_vectors[final_mask, 0], smoothed_vectors[final_mask, 1], smoothed_vectors[final_mask, 2]
        
        arrow_length = dx * stride * 0.8
        ax.quiver(X_p, Y_p, Z_p, U_p, V_p, W_p, length=arrow_length, normalize=True, 
                  color=color, arrow_length_ratio=0.3, alpha=0.8, linewidth=1.5)
        ax.set_title(f"Averaged {target_type.capitalize()} Orientations (Tensor Smoothed)", fontsize=14)

    elif mode == 'streamlines':
        # Trace continuous fibers
        lines = trace_streamlines(vectors, mask, X, Y, Z, dx, dy, dz, num_seeds=max_streamlines, max_steps=64)
        for lx, ly, lz in lines:
            ax.plot(lx, ly, lz, color=color, alpha=0.7, linewidth=2)
            
        ax.set_title(f"Continuous {target_type.capitalize()} Streamlines (Fiber Tracking)", fontsize=14)

    ax.set_xlabel(r'$x [$Mpc$]$'); ax.set_ylabel(r'$y [$Mpc$]$'); ax.set_zlabel(r'$z [$Mpc$]$')

    # Set strict axis limits
    ax.set_xlim([x_edges[0], x_edges[-1]])
    ax.set_ylim([y_edges[0], y_edges[-1]])
    ax.set_zlim([z_edges[0], z_edges[-1]])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("-i", "--input", required=True, type=str, default="cosmic_web.h5")
    parser.add_argument("--type", type=str, choices=['filament', 'sheet'], default='filament')
    parser.add_argument("--mode", type=str, choices=['arrows', 'streamlines'], default='arrows', help="'arrows' uses tensor-averaging, 'streamlines' tracks fibers.")
    parser.add_argument("--max_streamlines", type=int, default=400, help="Maximum steps for streamline tracing")
    parser.add_argument("--stride", type=int, default=8, help="Downsampling & smoothing scale for arrows")
    parser.add_argument("--slice_frac", type=float, default=1.0, help="Fraction of Z-axis to include")
    
    args = parser.parse_args()
    visualize_3d_orientation(args.input, target_type=args.type, mode=args.mode, stride=args.stride, slice_frac=args.slice_frac, max_streamlines=args.max_streamlines)