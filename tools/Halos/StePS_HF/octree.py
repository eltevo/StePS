import numpy as np
from dataclasses import dataclass

@dataclass
class Node:
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    mass: float
    com: np.ndarray
    hmax: float
    size: float
    idxs: np.ndarray  # particle indices in this node
    children: list    # length 8 or empty

def build_octree(r, m, h, max_leaf=32):
    # Compute bounding cube
    rmin = r.min(axis=0)
    rmax = r.max(axis=0)
    center = 0.5*(rmin + rmax)
    half = 0.5*np.max(rmax - rmin) * 1.0000001  # small pad for robustness
    root = _build_node(r, m, h, np.arange(len(m)), center - half, center + half, max_leaf)
    return root

def _build_node(r, m, h, idxs, bbox_min, bbox_max, max_leaf):
    mass = m[idxs].sum()
    com = (r[idxs]*m[idxs][:,None]).sum(axis=0)/mass if mass>0 else 0.5*(bbox_min+bbox_max)
    hmax = np.max(h[idxs]) if len(idxs)>0 else 0.0
    size = 0.5*np.max(bbox_max - bbox_min)

    node = Node(bbox_min, bbox_max, mass, com, hmax, size, idxs, [])
    if len(idxs) <= max_leaf:
        return node
    # split into octants
    center = 0.5*(bbox_min + bbox_max)
    child_idxs = [[] for _ in range(8)]
    for i in idxs:
        code = (r[i] > center).astype(int)
        ci = code[0] + 2*code[1] + 4*code[2]
        child_idxs[ci].append(i)

    for ci in range(8):
        if len(child_idxs[ci]) == 0:
            continue
        child_idxs[ci] = np.array(child_idxs[ci], dtype=int)
        cmin = bbox_min.copy()
        cmax = bbox_max.copy()
        for d in range(3):
            if (ci >> d) & 1:
                cmin[d] = center[d]
            else:
                cmax[d] = center[d]
        child = _build_node(r, m, h, child_idxs[ci], cmin, cmax, max_leaf)
        node.children.append(child)
    return node


def cubic_spline_kernel(r,h):
    """
    Function for calculating the Cubic spline kernel (Monaghan & Lattanzio, 1985)
    Input:
        - r: distance from the particle
        - h: softening length
    Returns:
        - cubic spline kernel value. The kernel is normalized as \int W(r,h) d^3r = 1
    """
    q = r / h
    W = np.zeros_like(q)
    factor = 8 / np.pi / h**3
    # Apply piecewise conditions
    mask1 = (q >= 0) & (q < 0.5)
    mask2 = (q >= 0.5) & (q < 1.0)
    W[mask1] = factor * (1 - 6*q[mask1]**2 + 6*q[mask1]**3)
    W[mask2] = factor * 2 * (1 - q[mask2])**3
    return W

def cubic_spline_potential(r, h):
    """
    Function for calculating the potential of Cubic spline kernel (Monaghan & Lattanzio, 1985)
    Input:
        - r: distance from the particle
        - h: softening length
    Returns:
        - cubic spline potential value (assuming unit masses and G=1 units)
    """
    q = r / h
    phi = np.zeros_like(q)
    # Apply piecewise conditions
    mask1 = (q >= 0) & (q < 0.5)
    mask2 = (q >= 0.5) & (q < 1.0)
    mask3 = (q >= 1.0)
    phi[mask1] = (16/3)*q[mask1]**2 - (48/5)*q[mask1]**4 + (32/5)*q[mask1]**5 - 14/5
    phi[mask2] = (1/(15*q[mask2])) + (32/3)*q[mask2]**2 - 16*q[mask2]**3 + (48/5)*q[mask2]**4 - (32/15)*q[mask2]**5 - 16/5
    phi[mask3] = -1/q[mask3]
    return phi/h

def potential_at_particle(i, node, r, m, h, theta=0.6, eta=1.0):
    """Return scalar potential phi at particle i from all others (unit G), isolated BC."""
    ri = r[i]
    hi = h[i]
    stack = [node]
    phi = 0.0

    while stack:
        nd = stack.pop()
        if nd.mass == 0:
            continue
        # distance from particle to node COM
        dvec = ri - nd.com
        d = np.linalg.norm(dvec) + 1e-30  # avoid zero

        # If this node is a leaf containing i alone, skip self
        if len(nd.children) == 0 and len(nd.idxs) == 1 and nd.idxs[0] == i:
            continue

        # BH + softening-aware opening criteria
        geom_open = (nd.size / d) > theta
        soft_overlap_possible = (d - nd.size) <= eta*(hi + nd.hmax)

        if len(nd.children) == 0 or (geom_open or soft_overlap_possible):
            # Leaf or must open: handle explicitly if leaf, else open children
            if len(nd.children) == 0:
                # Explicit particle loop
                js = nd.idxs
                # Exclude i; if present
                js = js[js != i]
                if js.size:
                    rij = np.linalg.norm(r[js] - ri, axis=1)
                    phi += np.sum(m[js] * cubic_spline_potential(rij, hi + h[js]))
            else:
                stack.extend(nd.children)
        else:
            # Accept node: use Newtonian monopole (softening negligible here)
            phi += - nd.mass / d
    return phi

def compute_potentials_tree(r, m, h, theta=0.6, eta=1.0, max_leaf=32):
    tree = build_octree(r, m, h, max_leaf=max_leaf)
    N = len(m)
    Phi = np.zeros(N, dtype=np.float64)
    for i in range(N):
        Phi[i] = potential_at_particle(i, tree, r, m, h, theta=theta, eta=eta)
    # Convert to energy per particle
    Epot_i = m * Phi
    return Epot_i
