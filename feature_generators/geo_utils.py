# geo_utils.py
from __future__ import annotations
from typing import Any, List, Optional
import numpy as np
from scipy.spatial import ConvexHull


def collect_object_vertices(obj: dict) -> np.ndarray:
    """
    Return all Nx3 vertices used by this CityJSON object by walking 'surfaces'.
    Works with direct coords or CityJSON 'boundaries' referencing root obj['vertices'].
    """
    root = np.asarray(obj.get("vertices", []), dtype=np.float64)
    V = []

    for srec in obj.get("geometry", [])[0].get("semantics", {}).get("surfaces", []):
        # Direct coords on the surface record
        if "vertices" in srec and is_coord_like(srec["vertices"]):
            V.append(np.asarray(srec["vertices"], dtype=np.float64))
            continue
        if "polygon" in srec and is_coord_like(srec["polygon"]):
            V.append(np.asarray(srec["polygon"], dtype=np.float64))
            continue
        if "polygon_mesh" in srec and isinstance(srec["polygon_mesh"], list) and srec["polygon_mesh"]:
            first = srec["polygon_mesh"][0]
            if is_coord_like(first):
                V.append(np.asarray(first, dtype=np.float64))
                continue
        # Boundaries → indices into root vertices
        b = srec.get("boundaries")
        if b is not None and root.ndim == 2 and root.shape[1] == 3:
            from geo_utils import first_ring_indices  # if not already imported here
            ring = first_ring_indices(b)
            if ring and len(ring) >= 3:
                if ring[0] == ring[-1]:
                    ring = ring[:-1]
                idx = np.asarray(ring, dtype=np.int64)
                ok = idx.min() >= 0 and idx.max() < root.shape[0]
                if ok:
                    V.append(root[idx, :])

    if not V:
        return np.empty((0, 3), dtype=np.float64)

    Vcat = np.vstack(V)
    # optional: unique to reduce duplicates
    try:
        Vcat = np.unique(Vcat, axis=0)
    except Exception:
        pass
    return Vcat


def is_coord_like(x: Any) -> bool:
    """
    Heuristic: True if x can be viewed as an (N, 3) array of coordinates.
    """
    try:
        arr = np.asarray(x)
        return arr.ndim == 2 and arr.shape[1] == 3
    except Exception:
        return False


def ensure_nx3(arr: np.ndarray) -> np.ndarray:
    """
    Return arr if shape is (N, 3); otherwise return an empty (0, 3) float64 array.
    """
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    return np.empty((0, 3), dtype=np.float64)


def first_ring_indices(boundaries: Any) -> Optional[List[int]]:
    """
    Extract the first ring's vertex indices from CityJSON-style nested 'boundaries'.
    Accepts common nestings like:
      - [ [i0, i1, ...] ]
      - [ [ [i0, i1, ...], [hole...] ] ]
      - [ [ [ [i0, ...] ] ] ]  # solids → shells → surfaces → rings
    Returns a flat list[int] or None if not found.
    """
    b = boundaries
    for _ in range(4):  # peel up to a safe depth
        if isinstance(b, list) and b and isinstance(b[0], list):
            b = b[0]
        else:
            break
    if isinstance(b, list) and b and all(isinstance(v, int) for v in b):
        return b
    return None


def compute_polygon_area(polygon: np.ndarray) -> float:
    """Compute area of a polygon in 3D by triangulation."""
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for i in range(1, len(polygon) - 1):
        v0, v1, v2 = polygon[0], polygon[i], polygon[i + 1]
        area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return float(area)


def compute_polygon_mesh_area(polygon_mesh: list) -> float:
    """Compute total area of a mesh of polygons."""
    return sum(compute_polygon_area(np.array(face)) for face in polygon_mesh)


def compute_polygon_mesh_volume(polygon_mesh: list) -> float:
    """Compute signed volume of a closed polygon mesh (absolute value returned)."""
    volume = 0.0
    for face in polygon_mesh:
        if len(face) < 3:
            continue
        v0 = np.array(face[0])
        for i in range(1, len(face) - 1):
            v1, v2 = np.array(face[i]), np.array(face[i + 1])
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
    return abs(volume)


def compute_eigendecomposition(vertices: np.ndarray):
    """Return eigenvalues and eigenvectors of covariance matrix of vertices."""
    if vertices.shape[0] < 2:
        return np.array([0.0, 0.0, 0.0]), np.eye(3)
    cov = np.cov(vertices, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvals, eigvecs
