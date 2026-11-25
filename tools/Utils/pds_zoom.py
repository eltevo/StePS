"""Shared utilities for variable-resolution spherical loads in S^3/I*."""

import csv
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq


TAU = (1 + math.sqrt(5)) / 2


def quat_mul(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.concatenate((
        (a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True)),
        a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:]),
    ), axis=-1)


def pds_group():
    generators = np.array([[.5, .5, .5, .5], [.5, -.5, -.5, -.5],
                           [TAU / 2, 1 / (2 * TAU), .5, 0],
                           [TAU / 2, -1 / (2 * TAU), -.5, 0]])
    group = [np.array([1., 0., 0., 0.])]
    for q in group:
        for g in generators:
            candidate = quat_mul(q, g)
            candidate /= np.linalg.norm(candidate)
            if not any(np.max(np.abs(candidate - other)) < 1e-9 for other in group):
                group.append(candidate)
    if len(group) != 120:
        raise RuntimeError(f"Expected 120 PDS group elements, got {len(group)}")
    return np.array(group)


def stereo_to_quat(x, radius):
    scaled = np.asarray(x) / radius
    r2 = np.sum(scaled * scaled, axis=-1, keepdims=True)
    return np.concatenate(((1 - r2) / (1 + r2), 2 * scaled / (1 + r2)), axis=-1)


def quat_to_stereo(q, radius):
    return radius * q[..., 1:] / (1 + q[..., :1])


def wrap(q, group):
    nearest = np.argmax(q @ group.T, axis=-1)
    inverse = group[nearest].copy()
    inverse[..., 1:] *= -1
    return quat_mul(inverse, q)


def azimuth_order(positions, center, radius, chunk_size):
    """Order by local azimuth of the nearest PDS image around the zoom centre."""
    qcenter = stereo_to_quat(np.asarray(center), radius)
    if np.all(qcenter[1:] == 0):
        return np.argsort(np.arctan2(positions[:, 1], positions[:, 0]))
    group = pds_group()
    inverse_center = qcenter.copy()
    inverse_center[1:] *= -1
    orbit = quat_mul(group, qcenter)
    angles = np.empty(len(positions), dtype=np.float64)
    for start in range(0, len(positions), chunk_size):
        end = min(start + chunk_size, len(positions))
        q = stereo_to_quat(positions[start:end], radius)
        nearest = np.argmax(q @ orbit.T, axis=1)
        inverse = group[nearest].copy()
        inverse[:, 1:] *= -1
        local = quat_mul(inverse_center, quat_mul(inverse, q))
        angles[start:end] = np.arctan2(local[:, 2], local[:, 1])
    return np.argsort(angles)


def volume_coordinate(chi):
    x = np.asarray(chi)
    # Avoid cancellation in chi - sin(2 chi)/2 near the centre.
    series = x**3 * (2/3 - 2*x*x/15 + 4*x**4/315 - 2*x**6/2835)
    return np.where(np.abs(x) < .02, series, x - np.sin(2*x)/2)


def ball_volume(chi, radius):
    return 2 * math.pi * radius**3 * volume_coordinate(chi)


def invert_volume(value, upper):
    if value <= 0:
        return 0.0
    if value >= volume_coordinate(upper):
        return float(upper)
    return brentq(lambda x: volume_coordinate(x) - value, 0, upper, xtol=1e-14)


def mobius(omega, scale):
    return 2 * np.arctan(scale * np.tan(np.asarray(omega) / 2))


def mobius_derivative(omega, scale):
    t = np.tan(np.asarray(omega) / 2)
    return scale * (1 + t*t) / (1 + scale*scale*t*t)


class RadialMap:
    def __init__(self, zoom, core_fraction, transition_fraction, scale, table_size):
        self.scale = scale
        self.chi_core = zoom * core_fraction
        self.chi_end = zoom * transition_fraction
        self.core = 2 * math.atan(math.tan(self.chi_core / 2) / scale)

        def core_jacobian(chi):
            return (scale * np.cos(chi / 2)**2 + np.sin(chi / 2)**2 / scale)**3

        if core_jacobian(self.chi_end) >= 1:
            raise ValueError("Mobius refinement is too strong for this zoom radius; increase lambda_mobius")

        def target_jacobian(chi):
            u = np.clip((chi - self.chi_core) / (self.chi_end - self.chi_core), 0, 1)
            smooth = u**3 * (10 + u * (-15 + 6*u))
            return core_jacobian(chi) * (1 - smooth) + smooth

        # Matching the Jacobian to one leaves an auxiliary volume offset in the buffer.
        def derivative(omega, chi):
            return [target_jacobian(chi[0]) * (math.sin(omega) / math.sin(chi[0]))**2]

        def reach_transition_end(omega, chi):
            return chi[0] - self.chi_end

        def reach_zoom(omega, chi):
            return chi[0] - zoom

        reach_zoom.terminal = True
        solution = solve_ivp(derivative, (self.core, math.pi - 1e-8), [self.chi_core],
                             events=(reach_transition_end, reach_zoom), dense_output=True,
                             rtol=2e-13, atol=1e-15,
                             max_step=(zoom - self.chi_core) / scale / 512)
        if not solution.success or len(solution.t_events[1]) != 1 or len(solution.t_events[0]) != 1:
            raise ValueError("Zoom map exceeds the available auxiliary S^3 volume; reduce refinement or zoom radius")
        self.end = solution.t_events[0][0]
        self.zoom = solution.t_events[1][0]
        ncore = max(3, min(32, table_size // 4))
        core = np.linspace(0, self.core, ncore)
        nouter = table_size - ncore + 1
        ntransition = min(nouter - 1, max(2, round((nouter - 1) * (self.end - self.core)
                                                  / (self.zoom - self.core)) + 1))
        outer = np.r_[np.linspace(self.core, self.end, ntransition)[:-1],
                      np.linspace(self.end, self.zoom, nouter - ntransition + 1)]
        values = solution.sol(outer)[0]
        derivatives = target_jacobian(values) * (np.sin(outer) / np.sin(values))**2
        self.omega = np.r_[core[:-1], outer]
        self.values = np.r_[mobius(core[:-1], scale), values]
        self.derivatives = np.r_[mobius_derivative(core[:-1], scale), derivatives]
        self.spline = CubicHermiteSpline(outer, values, derivatives, extrapolate=False)
        c = self.spline.c
        width = np.diff(outer)
        denom = 3 * c[0]
        turning = np.divide(-c[1], denom, out=np.zeros_like(denom), where=denom != 0)
        valid = (turning > 0) & (turning < width)
        minima = (3*c[0]*turning + 2*c[1])*turning + c[2]
        if np.any(self.derivatives <= 0) or np.any(minima[valid] <= 0):
            raise ValueError("Non-monotonic radial map; increase map_table_size or reduce refinement")

    def __call__(self, omega):
        x = np.asarray(omega)
        return np.where(x < self.core, mobius(x, self.scale), self.spline(x))

    def derivative(self, omega):
        x = np.asarray(omega)
        return np.where(x < self.core, mobius_derivative(x, self.scale), self.spline(x, 1))

    def jacobian(self, omega):
        x = np.asarray(omega)
        mapped = self(x)
        ratio = np.divide(np.sin(mapped), np.sin(x), out=np.full_like(x, self.scale, dtype=float), where=x != 0)
        return self.derivative(x) * ratio**2


def zoom_config(raw):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("zoom_glass must be a mapping")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("zoom_glass.enabled must be a boolean")
    if not enabled:
        return None
    allowed = {
        "enabled", "center", "lambda_mobius", "zoom_radius_fraction_of_injectivity_radius",
        "zoom_radius_mpc", "core_fraction_of_zoom_radius", "transition_end_fraction_of_zoom_radius",
        "radial_binning", "radial_shell_count", "NRBINS", "NSHELL", "angular_sampler", "random_shell_rotations",
        "minimum_angular_count", "minimum_buffer_shells", "maximum_adjacent_mass_ratio",
        "constant_resolution_radius_mpc", "constant_resolution_max_mass_ratio",
        "injectivity_safety_factor", "expert_override_safety", "map_table_size",
        "mass_match_weight", "shape_match_weight", "random_seed", "mass_tolerance",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown zoom_glass settings: {', '.join(sorted(unknown))}")
    cfg = dict(raw)
    if "NRBINS" in cfg and "radial_shell_count" in cfg:
        raise ValueError("Specify only one of zoom_glass.NRBINS and radial_shell_count")
    if "radial_shell_count" in cfg:
        warnings.warn("radial_shell_count is deprecated; use NRBINS", UserWarning)
        cfg["NRBINS"] = cfg.pop("radial_shell_count")
    if "NSHELL" not in cfg:
        raise ValueError("zoom_glass.NSHELL is required: specify the fixed number of particles per shell")
    defaults = {
        "center": "pds_center", "lambda_mobius": .5,
        "core_fraction_of_zoom_radius": .5, "transition_end_fraction_of_zoom_radius": .8,
        "radial_binning": "equal_auxiliary_volume", "NRBINS": 64,
        "angular_sampler": "random", "random_shell_rotations": True,
        "minimum_angular_count": 12, "minimum_buffer_shells": 2,
        "maximum_adjacent_mass_ratio": None, "injectivity_safety_factor": .8,
        "constant_resolution_radius_mpc": 0.0, "constant_resolution_max_mass_ratio": 1.25,
        "expert_override_safety": False, "map_table_size": 16384,
        "mass_match_weight": 2., "shape_match_weight": 1., "mass_tolerance": 1e-10,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    if "zoom_radius_mpc" in cfg and "zoom_radius_fraction_of_injectivity_radius" in cfg:
        raise ValueError("Specify only one zoom radius")
    cfg.setdefault("zoom_radius_fraction_of_injectivity_radius",
                   None if "zoom_radius_mpc" in cfg else .7)
    for key in ("lambda_mobius", "core_fraction_of_zoom_radius", "transition_end_fraction_of_zoom_radius",
                "injectivity_safety_factor", "mass_match_weight",
                "shape_match_weight", "mass_tolerance"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"zoom_glass.{key} must be a finite number")
    if not 0 < cfg["lambda_mobius"] < 1:
        raise ValueError("lambda_mobius must lie strictly between 0 and 1")
    if not 0 < cfg["core_fraction_of_zoom_radius"] < cfg["transition_end_fraction_of_zoom_radius"] < 1:
        raise ValueError("Require 0 < core fraction < transition end fraction < 1")
    if not 0 < cfg["injectivity_safety_factor"] <= 1:
        raise ValueError("injectivity_safety_factor must be in (0, 1]")
    constant_radius = cfg["constant_resolution_radius_mpc"]
    if constant_radius != "auto" and (
            isinstance(constant_radius, bool) or not isinstance(constant_radius, (int, float))
            or not math.isfinite(constant_radius) or constant_radius < 0):
        raise ValueError("constant_resolution_radius_mpc must be a non-negative number or 'auto'")
    central_ratio = cfg["constant_resolution_max_mass_ratio"]
    if (isinstance(central_ratio, bool) or not isinstance(central_ratio, (int, float))
            or not math.isfinite(central_ratio) or central_ratio <= 1):
        raise ValueError("constant_resolution_max_mass_ratio must be a finite number > 1")
    ratio_limit = cfg["maximum_adjacent_mass_ratio"]
    if ratio_limit is not None and (isinstance(ratio_limit, bool) or not isinstance(ratio_limit, (int, float))
                                    or not math.isfinite(ratio_limit) or ratio_limit < 1):
        raise ValueError("maximum_adjacent_mass_ratio must be a finite number >= 1")
    if cfg["mass_match_weight"] <= 0 or cfg["shape_match_weight"] <= 0:
        raise ValueError("Both matching weights must be positive")
    if not 0 < cfg["mass_tolerance"] < 1:
        raise ValueError("mass_tolerance must be in (0, 1)")
    for key, minimum in (("NRBINS", 3), ("NSHELL", 1), ("minimum_angular_count", 1),
                         ("minimum_buffer_shells", 2), ("map_table_size", 32)):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < minimum:
            raise ValueError(f"zoom_glass.{key} must be an integer >= {minimum}")
    for key in ("expert_override_safety", "random_shell_rotations"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"zoom_glass.{key} must be a boolean")
    if "random_shell_rotations" in raw:
        warnings.warn("random_shell_rotations has no effect with random shell sampling", UserWarning)
    if "mass_match_weight" in raw or "shape_match_weight" in raw:
        warnings.warn("mass_match_weight and shape_match_weight are obsolete; shell counts are fixed by NSHELL",
                      UserWarning)
    if "minimum_angular_count" in raw:
        warnings.warn("minimum_angular_count is obsolete; use NSHELL for the fixed shell count", UserWarning)
    if cfg["radial_binning"] not in ("equal_omega", "equal_auxiliary_volume"):
        raise ValueError("radial_binning must be equal_omega or equal_auxiliary_volume")
    if cfg["angular_sampler"] == "fibonacci":
        warnings.warn("angular_sampler=fibonacci is obsolete; using random shell sampling", UserWarning)
        cfg["angular_sampler"] = "random"
    if cfg["angular_sampler"] != "random":
        raise ValueError("Only the random angular sampler is supported")
    seed = cfg.get("random_seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("zoom_glass.random_seed must be a non-negative integer")
    return cfg


def random_shell(n, lo, hi, rng):
    """Uniform random directions and S^3 volume-uniform shell radii."""
    directions = rng.normal(size=(n, 3))
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Random shell direction has zero length")
    directions /= lengths[:, None]
    radii = np.empty(n)
    filled = 0
    while filled < n:
        candidates = rng.uniform(lo, hi, n - filled)
        accepted = candidates[rng.random(len(candidates)) * math.sin(hi)**2 <= np.sin(candidates)**2]
        radii[filled:filled + len(accepted)] = accepted
        filled += len(accepted)
    return directions, radii


def plan_zoom(cfg, radius, seed, rho, version):
    group = pds_group()
    center = cfg["center"]
    if isinstance(center, str) and center == "pds_center":
        center = np.zeros(3)
    elif isinstance(center, (list, tuple)) and len(center) == 3:
        try:
            center = np.array(center, dtype=float)
        except (ValueError, TypeError):
            raise ValueError("zoom_glass.center must contain three finite stereographic Mpc coordinates")
        if not np.all(np.isfinite(center)):
            raise ValueError("zoom_glass.center must contain three finite stereographic Mpc coordinates")
    else:
        raise ValueError("zoom_glass.center must be pds_center or three stereographic Mpc coordinates")
    qcenter = stereo_to_quat(center, radius)
    if np.max(group @ qcenter) > qcenter[0] + 1e-10:
        raise ValueError("zoom_glass.center must lie in the PDS fundamental domain")
    orbit = quat_mul(group, qcenter)
    # Left multiplication by a unit quaternion is a Clifford translation, so this
    # injectivity radius is independent of the chosen centre.
    injectivity = .5 * np.min(np.arccos(np.clip(orbit[1:] @ qcenter, -1, 1)))
    raw_zoom = cfg.get("zoom_radius_mpc")
    if raw_zoom is None:
        raw_zoom = cfg["zoom_radius_fraction_of_injectivity_radius"]
        factor = injectivity
    else:
        factor = 1 / radius
    if isinstance(raw_zoom, bool) or not isinstance(raw_zoom, (int, float)) or not math.isfinite(raw_zoom) or raw_zoom <= 0:
        raise ValueError("The zoom radius must be a finite positive number")
    zoom = raw_zoom * factor
    if zoom >= injectivity:
        raise ValueError("Zoom radius must be strictly smaller than the injectivity radius")
    if zoom > cfg["injectivity_safety_factor"] * injectivity and not cfg["expert_override_safety"]:
        raise ValueError("Zoom radius exceeds the injectivity safety limit; set expert_override_safety to override")
    radial_map = RadialMap(zoom, cfg["core_fraction_of_zoom_radius"],
                           cfg["transition_end_fraction_of_zoom_radius"],
                           cfg["lambda_mobius"], cfg["map_table_size"])

    total_volume = math.pi**2 * radius**3 / 60
    zoom_volume = float(ball_volume(zoom, radius))
    exterior_volume = total_volume - zoom_volume
    n = cfg["NSHELL"]
    def make_bins(start, count, first_chi=None):
        if cfg["radial_binning"] == "equal_omega":
            omega = np.linspace(start, radial_map.zoom, count + 1)
            auxiliary_volumes = np.diff(ball_volume(omega, radius))
        else:
            values = np.linspace(volume_coordinate(start), volume_coordinate(radial_map.zoom), count + 1)
            omega = np.array([invert_volume(v, radial_map.zoom) for v in values])
            omega[0] = start
            auxiliary_volumes = np.full(count, 2 * math.pi * radius**3 * (values[-1] - values[0]) / count)
        chi = radial_map(omega)
        if cfg["radial_binning"] == "equal_auxiliary_volume":
            buffer = omega >= radial_map.end
            offset = volume_coordinate(radial_map.chi_end) - volume_coordinate(radial_map.end)
            chi[buffer] = [invert_volume(v + offset, zoom) for v in values[buffer]]
        if first_chi is not None:
            chi[0] = first_chi
        chi[-1] = zoom
        shells = []
        for i, (lo, hi) in enumerate(zip(chi[:-1], chi[1:])):
            volume = (float(auxiliary_volumes[i]) if cfg["radial_binning"] == "equal_auxiliary_volume"
                      and omega[i] >= radial_map.end else
                      float(ball_volume(hi, radius) - ball_volume(lo, radius)))
            if not math.isfinite(volume) or volume <= 0:
                raise ValueError(f"Shell {i} has a non-positive or non-finite volume")
            midpoint = invert_volume((volume_coordinate(lo) + volume_coordinate(hi)) / 2, hi)
            omega_mid = brentq(lambda x: float(radial_map(x)) - midpoint, omega[i], omega[i+1], xtol=1e-14)
            jacobian = float(radial_map.jacobian(omega_mid))
            shells.append((lo, hi, volume, midpoint, jacobian))
        return omega, chi, shells

    omega, chi, shells = make_bins(0.0, cfg["NRBINS"])
    cutoff = cfg["constant_resolution_radius_mpc"]
    removed = 0
    if cutoff == "auto":
        volumes = np.array([shell[2] for shell in shells])
        baseline_masses = rho * volumes / n
        expected_outer = n * exterior_volume / volumes[-1]
        if not math.isfinite(expected_outer) or expected_outer > 2**31 - 1:
            raise ValueError("Estimated zoom particle count exceeds StePS's supported limit; reduce NSHELL or NRBINS")
        outer_count = math.floor(expected_outer)
        if outer_count < 1:
            raise ValueError("No exterior particles at this resolution; increase NSHELL")
        baseline_masses[-1] = rho * (exterior_volume + volumes[-1]) / (outer_count + n)
        violations = np.flatnonzero(baseline_masses[1:] / baseline_masses[:-1] >
                                    cfg["constant_resolution_max_mass_ratio"])
        if len(violations):
            removed = int(violations[-1] + 1)
            while removed < len(shells):
                central_volume = float(ball_volume(chi[removed], radius))
                central_count = math.ceil(rho * central_volume / baseline_masses[removed])
                if central_count * baseline_masses[removed] / (rho * central_volume) <= cfg["constant_resolution_max_mass_ratio"]:
                    break
                removed += 1
            if removed == len(shells):
                raise ValueError("Automatic constant-resolution region leaves no zoom shells; adjust the ratio limit")
            omega = omega[removed:]
            chi = chi[removed:]
            shells = shells[removed:]
    elif cutoff > 0:
        if cutoff >= zoom * radius:
            raise ValueError("constant_resolution_radius_mpc must be smaller than the zoom radius")
        chi_cutoff = cutoff / radius
        omega_cutoff = brentq(lambda x: float(radial_map(x)) - chi_cutoff, 0, radial_map.zoom, xtol=1e-14)
        nearest = int(np.argmin(np.abs(omega - omega_cutoff)))
        if math.isclose(omega_cutoff, omega[nearest], rel_tol=1e-12, abs_tol=1e-14):
            omega_cutoff = float(omega[nearest])
        removed = int(np.searchsorted(omega, omega_cutoff, side="right") - 1)
        omega, chi, shells = make_bins(omega_cutoff, cfg["NRBINS"] - removed, first_chi=chi_cutoff)
    buffer_count = np.count_nonzero(omega[:-1] >= radial_map.end - 1e-14)
    if buffer_count < cfg["minimum_buffer_shells"]:
        raise ValueError("Too few buffer shells; increase NRBINS, widen the buffer, or reduce the constant-resolution radius")
    central_volume = float(ball_volume(chi[0], radius)) if chi[0] > 0 else 0.0
    central_count = math.ceil(n * central_volume / shells[0][2]) if central_volume > 0 else 0
    central_mass = rho * central_volume / central_count if central_count else None
    zoom_count = central_count + len(shells) * n
    if zoom_count >= 2**31 - 1:
        raise ValueError("Zoom particle count exceeds StePS's supported limit")

    nominal_edge_mass = rho * shells[-1][2] / n
    expected_exterior = n * exterior_volume / shells[-1][2]
    if not math.isfinite(expected_exterior) or expected_exterior + zoom_count > 2**31 - 1:
        raise ValueError("Estimated zoom particle count exceeds StePS's supported limit; reduce NSHELL or NRBINS")
    exterior_count = math.floor(expected_exterior)
    if exterior_count < 1:
        raise ValueError("No exterior particles at this resolution; increase NSHELL")
    inner_volume = central_volume + math.fsum(shell[2] for shell in shells[:-1])
    exterior_mass = rho * (total_volume - inner_volume) / (exterior_count + n)
    edge_correction = exterior_mass / nominal_edge_mass - 1
    if abs(edge_correction) > .01:
        warnings.warn(f"Exterior particle-count rounding changes the edge mass by {edge_correction:.2%}; "
                      "increase NSHELL for a closer density match", UserWarning)
    rows = []
    for i, (lo, hi, volume, midpoint, jacobian) in enumerate(shells):
        mass = exterior_mass if i == len(shells) - 1 else rho * volume / n
        rows.append(dict(shell_index=removed+i, omega_lo=omega[i], omega_hi=omega[i+1], chi_lo=lo, chi_hi=hi,
                         chi_centroid=midpoint, shell_volume=volume, angular_particle_count=n,
                         particle_mass=mass, radial_thickness=radius*(hi-lo),
                         tangential_spacing_estimate=radius*math.sin(midpoint)*math.sqrt(4*math.pi/n),
                         jacobian_at_centroid=jacobian))
    masses = np.array([row["particle_mass"] for row in rows] + [exterior_mass])
    if central_count:
        masses = np.r_[central_mass, masses]
    ratios = np.maximum(masses[1:] / masses[:-1], masses[:-1] / masses[1:])
    if np.any(masses[:-1] > masses[1:] * (1 + 1e-12)):
        raise ValueError("Zoom shell masses are not monotonically increasing")
    if cutoff == "auto" and np.any(ratios > cfg["constant_resolution_max_mass_ratio"] * (1 + 1e-10)):
        raise ValueError("Automatic constant-resolution mass-ratio limit cannot be met; "
                         "increase NSHELL or adjust the ratio limit")
    if cfg["maximum_adjacent_mass_ratio"] is not None and np.any(
            ratios > cfg["maximum_adjacent_mass_ratio"] * (1 + 1e-12)):
        i = int(np.argmax(ratios))
        raise ValueError(f"Adjacent region mass ratio {ratios[i]:.6g} exceeds limit at boundary {i}; "
                         "adjust the limit, binning, shell count or refinement")
    total_mass = (math.fsum(row["particle_mass"] * n for row in rows) +
                  exterior_count * exterior_mass + (central_count * central_mass if central_count else 0))
    expected_mass = rho * total_volume
    if abs(total_mass / expected_mass - 1) > cfg["mass_tolerance"]:
        raise ValueError("Zoom load failed the total mass conservation check")
    metadata = dict(curvature_radius=radius, PDS_volume=total_volume, mean_density=rho,
                    zoom_center_stereographic_mpc=center.tolist(), injectivity_radius=injectivity*radius,
                    zoom_radius=zoom*radius, zoom_volume=zoom_volume, lambda_mobius=cfg["lambda_mobius"],
                    omega_1=radial_map.core, omega_2=radial_map.end,
                    radial_binning_mode=cfg["radial_binning"], radial_shell_count=len(rows),
                    constant_resolution_radius=chi[0]*radius, central_volume=central_volume,
                    central_particle_count=central_count, central_particle_mass=central_mass,
                    removed_shell_count=removed,
                    total_particle_count=exterior_count + zoom_count, zoom_particle_count=zoom_count,
                    exterior_particle_count=exterior_count, minimum_particle_mass=float(np.min(masses)),
                    maximum_particle_mass=float(np.max(masses)), exterior_particle_mass=exterior_mass,
                    edge_mass_relative_correction=edge_correction, total_mass=total_mass,
                    random_seed=seed, angular_seed=cfg["random_seed"] if cfg.get("random_seed") is not None else seed,
                    code_version=version,
                    zoom_config=cfg, mass_policy="last shell and exterior share a mass adjusted to conserve total mass")
    return metadata, rows, radial_map


def make_zoom(cfg, radius, count, seed, rho, sample_positions, chunk_size, version, plan=None):
    """Generate a planned fixed-shell load; count is retained for compatibility and ignored."""
    if plan is None:
        plan = plan_zoom(cfg, radius, seed, rho, version)
    metadata, rows, radial_map = plan
    group = pds_group()
    qcenter = stereo_to_quat(np.asarray(metadata["zoom_center_stereographic_mpc"]), radius)
    orbit = quat_mul(group, qcenter)
    cos_zoom = math.cos(metadata["zoom_radius"] / radius)
    exterior_count = metadata["exterior_particle_count"]
    positions = np.empty((metadata["total_particle_count"], 3), dtype=np.float64)
    particle_masses = np.empty(metadata["total_particle_count"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    filled = 0
    while filled < exterior_count:
        candidates = sample_positions(min(chunk_size, exterior_count - filled), radius, rng)
        quaternions = stereo_to_quat(candidates, radius)
        accepted = candidates[np.max(quaternions @ orbit.T, axis=1) <= cos_zoom]
        positions[filled:filled + len(accepted)] = accepted
        filled += len(accepted)
    particle_masses[:exterior_count] = metadata["exterior_particle_mass"]
    shell_rng = np.random.default_rng(metadata["angular_seed"])
    remaining = metadata["central_particle_count"]
    while remaining:
        n = min(remaining, chunk_size)
        direction, radii = random_shell(n, 0, metadata["constant_resolution_radius"] / radius, shell_rng)
        local = np.column_stack((np.cos(radii), np.sin(radii)[:, None] * direction))
        positions[filled:filled+n] = quat_to_stereo(wrap(quat_mul(qcenter, local), group), radius)
        particle_masses[filled:filled+n] = metadata["central_particle_mass"]
        filled += n
        remaining -= n
    for row in rows:
        n = row["angular_particle_count"]
        direction, radii = random_shell(n, row["chi_lo"], row["chi_hi"], shell_rng)
        local = np.column_stack((np.cos(radii), np.sin(radii)[:, None] * direction))
        quaternions = wrap(quat_mul(qcenter, local), group)
        positions[filled:filled+n] = quat_to_stereo(quaternions, radius)
        particle_masses[filled:filled+n] = row["particle_mass"]
        filled += n
    total_mass = float(np.sum(particle_masses, dtype=np.float64))
    if abs(total_mass / (rho * metadata["PDS_volume"]) - 1) > cfg["mass_tolerance"]:
        raise ValueError("Zoom load failed the total mass conservation check")
    return positions, particle_masses, metadata, rows, radial_map


def write_diagnostics(output, metadata, rows, radial_map):
    base = str(Path(output).with_suffix(""))
    with open(base + ".zoom.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with open(base + ".zoom-shells.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with open(base + ".zoom-map.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("omega", "chi", "derivative", "jacobian"))
        for omega, chi, derivative, jacobian in zip(radial_map.omega, radial_map.values,
                                                   radial_map.derivatives, radial_map.jacobian(radial_map.omega)):
            writer.writerow((omega, chi, derivative, jacobian))
