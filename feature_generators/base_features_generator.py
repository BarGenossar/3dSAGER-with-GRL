
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull


class BaseFeaturesGenerator:
    """
    Base class for feature generators.
    Provides shared geometry-based feature computations,
    and a generic dispatcher that maps feature_spec entries
    to the corresponding _compute_* methods.
    """

    def __init__(self, obj: Dict[str, Any], feature_spec: Optional[List[str]] = None, logger: Any = None):
        """
        Args:
            obj (dict): The raw object dictionary from CityJSON.
            feature_spec (list, optional): List of feature names from config (JSON).
            logger (optional): Logger instance for debugging.
        """
        self.obj = obj
        self.vertices = obj.get("vertices", [])
        self.global_to_local = obj.get("global_to_local", {})
        self.local_to_global = obj.get("local_to_global", {})
        self.boundaries = obj.get("geometry", [])[0].get("boundaries", [])
        self.surfaces = obj.get("geometry", [])[0].get("semantics", {}).get("surfaces", [])
        self.feature_spec = feature_spec or []
        self.logger = logger

    # -----------------
    # Public interface
    # -----------------
    def get_feature_vector(self, element_idx: int) -> torch.Tensor:
        """
        Compute the feature vector for a given element index
        by iterating over feature_spec and calling the corresponding
        _compute_* function.

        Args:
            element_idx (int): index of the surface/element within this object

        Returns:
            torch.Tensor: feature vector of shape (dim(),)
        """
        features = []
        for feature_name in self.feature_spec:
            func = getattr(self, f"_compute_{feature_name}", None)
            if func is None:
                if self.logger:
                    self.logger.warning(f"Feature '{feature_name}' not implemented.")
                features.append(0.0)
                continue
            value = func(element_idx)
            # try:
            #     value = func(element_idx)
            # except Exception as e:
            #     if self.logger:
            #         self.logger.warning(f"Error computing {feature_name}: {e}")
            #     value = 0.0
            features.append(float(value))

        return torch.tensor(features, dtype=torch.float32)

    def dim(self) -> int:
        """Return dimensionality = number of features in the spec."""
        return len(self.feature_spec)

    # -----------------
    # Helpers
    # -----------------
    def _extract_vertices(self, obj: Dict[str, Any]) -> List[List[float]]:
        """
        Extract the global vertex list attached to the CityJSON object.
        """
        if "vertices" not in obj:
            if self.logger:
                self.logger.warning("Object missing 'vertices'. Returning empty list.")
            return []
        return obj["vertices"]

    def _resolve_indices(self, boundary_indices: List[int]) -> List[List[float]]:
        """
        Map a list of vertex indices (from a surface boundary)
        to actual 3D coordinates using the global vertices array.
        """
        coords = []
        for g in boundary_indices:
            if g in self.global_to_local:
                l = self.global_to_local[g]
                coords.append(self.vertices[l])
        return coords

    # -----------------
    # Core geometry-based feature functions
    # -----------------
    def _compute_surface_area(self, element_idx: int) -> float:
        """
        Compute area of a polygon surface (outer ring only).
        """
        surface = self.boundaries[element_idx]
        coords = self._resolve_indices(surface[0])  # outer ring
        if len(coords) < 3:
            return 0.0
        area = 0.0
        for i in range(1, len(coords) - 1):
            v0, v1, v2 = np.array(coords[0]), np.array(coords[i]), np.array(coords[i + 1])
            area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        return area

    def _compute_perimeter_length(self, element_idx: int) -> float:
        """
        Compute perimeter length of the polygon surface.
        """
        surface = self.boundaries[element_idx]
        coords = self._resolve_indices(surface[0])
        perimeter = 0.0
        n = len(coords)
        for i in range(n):
            v0, v1 = np.array(coords[i]), np.array(coords[(i + 1) % n])
            perimeter += np.linalg.norm(v1 - v0)
        return perimeter

    def _compute_height_extent(self, element_idx: int) -> float:
        """
        Vertical range of the surface (max z - min z).
        """
        surface = self.boundaries[element_idx]
        coords = self._resolve_indices(surface[0])
        if not coords:
            return 0.0
        z_coords = [v[2] for v in coords]
        return max(z_coords) - min(z_coords)

    def _compute_aspect_direction(self, element_idx: int) -> float:
        """
        Orientation of polygon normal projected on XY plane (in radians).
        """
        surface = self.boundaries[element_idx]
        coords = self._resolve_indices(surface[0])
        if len(coords) < 3:
            return 0.0
        v0, v1, v2 = np.array(coords[0]), np.array(coords[1]), np.array(coords[2])
        normal = np.cross(v1 - v0, v2 - v0)
        normal_xy = np.array([normal[0], normal[1]])
        if np.linalg.norm(normal_xy) == 0:
            return 0.0
        return float(np.arctan2(normal_xy[1], normal_xy[0]))

    def _compute_planarity_deviation(self, element_idx: int) -> float:
        """
        Deviation from best-fit plane using SVD.
        """
        surface = self.boundaries[element_idx]
        coords = self._resolve_indices(surface[0])
        if len(coords) < 3:
            return 0.0
        pts = np.array(coords)
        centroid = np.mean(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        normal = vh[-1, :]
        dists = np.dot(pts - centroid, normal)
        return float(np.std(dists))

    def _compute_compactness(self, element_idx: int) -> float:
        """
        Compactness = area / (perimeter^2).
        """
        area = self._compute_surface_area(element_idx)
        perim = self._compute_perimeter_length(element_idx)
        if perim == 0:
            return 0.0
        return area / (perim ** 2)

    def _compute_convex_hull_area(self, element_idx: int) -> float:
        """
        2D convex hull area of projection onto XY plane.
        """
        surface = self.boundaries[element_idx]
        coords = self._resolve_indices(surface[0])
        pts = np.array(coords)[:, :2]
        if pts.shape[0] < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.area

    def _compute_convex_hull_volume(self, element_idx: int) -> float:
        """
        3D convex hull volume.
        """
        surface = self.boundaries[element_idx]
        coords = self._resolve_indices(surface[0])
        pts = np.array(coords)
        if pts.shape[0] < 4:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume

    def _compute_elongation(self, element_idx: int) -> float:
        """
        Elongation = sqrt(max_eigenvalue / min_eigenvalue)
        from covariance of vertices.
        """
        surface = self.boundaries[element_idx]
        coords = self._resolve_indices(surface[0])
        pts = np.array(coords)
        if pts.shape[0] < 2:
            return 0.0
        cov = np.cov(pts.T)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-9, None)
        return float(np.sqrt(eigvals.max() / eigvals.min()))

    def _get_surface_coords(self, element_idx: int) -> np.ndarray:
        """
        Return Nx3 array of vertex coordinates for the given surface.
        Uses global_to_local mapping for safety.
        """
        coords = []

        def recurse(boundary):
            if isinstance(boundary, int):
                if boundary in self.global_to_local:
                    l = self.global_to_local[boundary]
                    coords.append(self.vertices[l])
            elif isinstance(boundary, list):
                for b in boundary:
                    recurse(b)

        recurse(self.boundaries[element_idx])
        return np.array(coords, dtype=float)
