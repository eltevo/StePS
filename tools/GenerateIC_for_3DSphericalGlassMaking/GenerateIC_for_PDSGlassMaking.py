#!/usr/bin/env python3

#*******************************************************************************#
#  GenerateIC_for_PDSGlassMaking.py - IC generator for PDS StePS glasses        #
#     STEreographically Projected cosmological Simulations                      #
#    Copyright (C) 2026 Gabor Racz                                              #
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

_VERSION = "v0.0.0.1"
_YEAR    = "2026"
_NAME    = "GenerateIC_for_PDSGlassMaking"
_AUTHORS = "Gabor Racz"

import argparse
import math
from pathlib import Path

import numpy as np
import yaml


SPEED_OF_LIGHT = 299792.458  # km/s
UNIT_V = 20.738652969925447  # km/s, StePS units with G = 1
TAU = (1.0 + math.sqrt(5.0)) / 2.0
FACE_OFFSET = 1.0 - TAU / 2.0
GNOMONIC_BOUND = 2.0 * FACE_OFFSET
FACE_VECTORS = np.array([
    (a * 0.5 / TAU, b * 0.5, 0.0)
    for a in (-1, 1) for b in (-1, 1)
] + [
    (0.0, a * 0.5 / TAU, b * 0.5)
    for a in (-1, 1) for b in (-1, 1)
] + [
    (b * 0.5, 0.0, a * 0.5 / TAU)
    for a in (-1, 1) for b in (-1, 1)
])
CHUNK_SIZE = 100_000


def read_parameters(filename):
    with open(filename, encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    if not isinstance(params, dict):
        raise ValueError("The YAML file must contain a parameter mapping")

    def positive_number(key):
        value = params.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a positive number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number")
        return float(value)

    radius = positive_number("PDS_R_CURV")
    hubble = positive_number("HUBBLE_CONSTANT")
    omega_m = positive_number("OMEGA_M")
    count = params.get("NPART")
    seed = params.get("RANDSEED")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 2**31 - 1:
        raise ValueError("NPART must be an integer between 1 and 2^31 - 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("RANDSEED must be a non-negative integer")
    output = params.get("BASEOUT")
    if not isinstance(output, str) or Path(output).suffix.lower() not in (".dat", ".hdf5"):
        raise ValueError("BASEOUT must have a .dat or .hdf5 extension")
    ignore_curvature = params.get("IGNORE_CURVATURE", False)
    if not isinstance(ignore_curvature, bool):
        raise ValueError("IGNORE_CURVATURE must be a YAML boolean")

    def optional_number(key):
        value = params.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        return float(value)

    specified_k = optional_number("OMEGA_K")
    specified_l = optional_number("OMEGA_L")
    if ignore_curvature:
        if specified_l is None:
            raise ValueError("OMEGA_L is required when IGNORE_CURVATURE is true")
        omega_l = specified_l
        omega_k = 1.0 - omega_m - omega_l
    else:
        omega_k = -(SPEED_OF_LIGHT / hubble / radius) ** 2
        omega_l = 1.0 - omega_m - omega_k
        if specified_l is not None and not math.isclose(specified_l, omega_l, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"OMEGA_L disagrees with the PDS radius; expected {omega_l:.12g}")
    if specified_k is not None and not math.isclose(specified_k, omega_k, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"OMEGA_K disagrees with the selected cosmology; expected {omega_k:.12g}")
    return radius, hubble, omega_m, omega_l, omega_k, count, seed, output, ignore_curvature


def sample_positions(count, radius, rng):
    """Rejection-sample the spherical Voronoi cell in gnomonic coordinates."""
    positions = np.empty((count, 3), dtype=np.float64)
    filled = 0
    while filled < count:
        # q = (1, u)/sqrt(1 + |u|^2); the S^3 volume element is
        # R^3 (1 + |u|^2)^-2 d^3u. The 12 nearest I* elements bound the cell.
        n = min(CHUNK_SIZE, max(1024, 4 * (count - filled)))
        u = rng.uniform(-GNOMONIC_BOUND, GNOMONIC_BOUND, size=(n, 3))
        inside = np.all(u @ FACE_VECTORS.T <= FACE_OFFSET, axis=1)
        u = u[inside]
        norm2 = np.sum(u * u, axis=1)
        u = u[rng.random(len(u)) < (1.0 + norm2) ** -2]
        u = u[:count - filled]
        norm2 = np.sum(u * u, axis=1)
        # x = R q_vec/(1 + q_0), the convention in StePS.
        positions[filled:filled + len(u)] = radius * u / (1.0 + np.sqrt(1.0 + norm2))[:, None]
        filled += len(u)
    return positions


def dodecahedron_lengths(radius):
    """Lengths in the S^3 covering cell and its stereographic chart."""
    # These adjacent vertices share two of the gnomonic face planes.
    s = 2.0 * FACE_OFFSET / TAU
    vertex = np.array([s, s, s])
    adjacent = np.array([0.0, TAU * s, s / TAU])
    vertex_norm = np.linalg.norm(vertex)
    edge_delta = np.linalg.norm(vertex - adjacent)
    vertex_angle = math.atan(vertex_norm)
    face_angle = math.pi / 10.0
    edge_angle = 2.0 * math.asin(edge_delta / (2.0 * math.sqrt(1.0 + vertex_norm**2)))
    chart_vertex = radius * math.tan(vertex_angle / 2.0)
    chart_face = radius * math.tan(face_angle / 2.0)
    chart_edge = radius * edge_delta / (1.0 + math.sqrt(1.0 + vertex_norm**2))
    return edge_angle * radius, vertex_angle * radius, face_angle * radius, chart_edge, chart_vertex, chart_face


def write_ic(params):
    radius, hubble, omega_m, omega_l, omega_k, count, seed, output, ignored = params
    volume = math.pi**2 * radius**3 / 60.0
    sphere_volume = 120.0 * volume
    rho_crit = 3.0 * (hubble / UNIT_V)**2 / (8.0 * math.pi)
    mass = rho_crit * omega_m * volume / count
    if not math.isfinite(sphere_volume) or not math.isfinite(mass) or mass <= 0:
        raise ValueError("Parameters yield a non-finite or zero volume or particle mass")
    separation = (volume / count) ** (1.0 / 3.0)
    edge, vertex_radius, face_radius, chart_edge, chart_vertex, chart_face = dodecahedron_lengths(radius)
    print("\nGlass parameters:\n-----------------")
    print(f"Output file:                       {output}")
    print(f"Random seed:                       {seed}")
    print(f"Particles:                         {count}")
    print(f"S^3 curvature radius:              {radius:.12g} Mpc")
    print(f"S^3 volume:                        {sphere_volume:.12g} Mpc^3 ({sphere_volume / 1e9:.12g} Gpc^3)")
    print(f"Dodecahedron volume:               {volume:.12g} Mpc^3 ({volume / 1e9:.12g} Gpc^3)")
    print("\nDodecahedron lengths in the S^3 covering cell (geodesic):")
    print(f"Edge length:                       {edge:.12g} Mpc")
    print(f"Centre to corner (circumradius):   {vertex_radius:.12g} Mpc")
    print(f"Opposite corners (through centre): {2 * vertex_radius:.12g} Mpc")
    print(f"Centre to face (inradius):         {face_radius:.12g} Mpc")
    print(f"Opposite face centres:             {2 * face_radius:.12g} Mpc")
    print("\nStereographic coordinate lengths (Euclidean chords):")
    print(f"Edge chord:                        {chart_edge:.12g} Mpc")
    print(f"Centre to corner:                  {chart_vertex:.12g} Mpc")
    print(f"Opposite corners:                  {2 * chart_vertex:.12g} Mpc")
    print(f"Centre to face:                    {chart_face:.12g} Mpc")
    print(f"Opposite face centres:             {2 * chart_face:.12g} Mpc")
    print("\nResolution:")
    print(f"Typical particle separation:       {separation:.12g} Mpc")
    print(f"Recommended softening length:      {separation / 40.0:.12g} Mpc")
    print(f"Particle mass:                     {mass:.12g} x 10^11 solar masses")
    print("\nCosmological parameters:\n------------------------")
    print(f"Hubble constant:                   {hubble:.12g} km/s/Mpc")
    print(f"Omega_m:                           {omega_m:.12g}")
    print(f"Omega_L:                           {omega_l:.12g}")
    print(f"Omega_k:                           {omega_k:.12g}")
    print(f"Omega_total (matter + Lambda):     {omega_m + omega_l:.12g}")
    if ignored:
        print("WARNING: IGNORE_CURVATURE is true; background curvature is independent of PDS geometry.")
    rng = np.random.default_rng(seed)
    if Path(output).suffix.lower() == ".dat":
        with open(output, "w", encoding="ascii") as stream:
            for start in range(0, count, CHUNK_SIZE):
                n = min(CHUNK_SIZE, count - start)
                data = np.zeros((n, 7))
                data[:, :3] = sample_positions(n, radius, rng)
                data[:, 6] = mass
                np.savetxt(stream, data, fmt="%.17g")
    else:
        import h5py

        with h5py.File(output, "w") as snapshot:
            header = snapshot.create_group("Header")
            counts = np.array([0, count, 0, 0, 0, 0], dtype=np.uint32)
            header.attrs["NumPart_ThisFile"] = counts
            header.attrs["NumPart_Total"] = counts
            header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
            header.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
            header.attrs["Time"] = 1.0
            header.attrs["Redshift"] = 0.0
            header.attrs["BoxSize"] = 2.0 * radius
            header.attrs["NumFilesPerSnapshot"] = 1
            header.attrs["Omega0"] = omega_m
            header.attrs["OmegaLambda"] = omega_l
            header.attrs["Omega_k"] = omega_k
            header.attrs["HubbleParam"] = hubble / 100.0
            header.attrs["TopologicalManifold"] = "S^3/I*"
            header.attrs["R_curvature_Mpc"] = radius
            header.attrs["PDS_I_star_order"] = 120
            for flag in ("Sfr", "Cooling", "StellarAge", "Metals", "Feedback", "Entropy_ICs"):
                header.attrs[f"Flag_{flag}"] = 0
            particles = snapshot.create_group("PartType1")
            coords = particles.create_dataset("Coordinates", (count, 3), dtype="f8")
            velocities = particles.create_dataset("Velocities", (count, 3), dtype="f8")
            masses = particles.create_dataset("Masses", (count,), dtype="f8")
            ids = particles.create_dataset("ParticleIDs", (count,), dtype="u8")
            for start in range(0, count, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, count)
                coords[start:end] = sample_positions(end - start, radius, rng)
                velocities[start:end] = 0.0
                masses[start:end] = mass
                ids[start:end] = np.arange(start, end, dtype=np.uint64)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml_file", help="YAML parameters")
    args = parser.parse_args()
    print("+-------------------------------------------------------------------------------------------------------+\n|%s %s\t\t\t\t\t\t\t\t\t|\n| (An initial condition generator for Poincare Dodecahedral Space (S^3/I*) glass making.)\t\t|\n+-------------------------------------------------------------------------------------------------------+\n|\t\t\t\t\t\t\t\t\t\t\t\t\t|\n| %s, %s\t\t\t\t\t\t\t\t\t\t\t|\n|\tDepartment of Physics, University of Helsinki | Helsinki, Finland\t\t\t\t|\n+-------------------------------------------------------------------------------------------------------+\n\n" % (_NAME,_VERSION,_AUTHORS,_YEAR))

    try:
        write_ic(read_parameters(args.yaml_file))
    except (ValueError, OSError, yaml.YAMLError, OverflowError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
