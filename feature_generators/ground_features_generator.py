
import numpy as np
import torch
from typing import Any, Dict, Optional
from .base_features_generator import BaseFeaturesGenerator
from . import geo_utils


class GroundFeaturesGenerator(BaseFeaturesGenerator):
    """
    Feature generator for GroundSurface polygons.
    """

    def __init__(self, obj: Dict[str, Any], feature_spec: Optional[list] = None, logger: Any = None):
        super().__init__(obj=obj, feature_spec=feature_spec, logger=logger)

    # -----------------
    # Features
    # -----------------
    def _compute_surface_area(self, element_idx: int) -> float:
        return super()._compute_surface_area(element_idx)

    def _compute_perimeter_length(self, element_idx: int) -> float:
        return super()._compute_perimeter_length(element_idx)

    def _compute_elevation_mean(self, element_idx: int) -> float:
        surface = self.obj["geometry"][0]["boundaries"][element_idx]
        coords = self._resolve_indices(surface[0])
        if not coords:
            return 0.0
        z_vals = [v[2] for v in coords]
        return float(np.mean(z_vals))

    def _compute_elevation_variance(self, element_idx: int) -> float:
        surface = self.obj["geometry"][0]["boundaries"][element_idx]
        coords = self._resolve_indices(surface[0])
        if not coords:
            return 0.0
        z_vals = [v[2] for v in coords]
        return float(np.var(z_vals))

    def _compute_planarity_deviation(self, element_idx: int) -> float:
        return super()._compute_planarity_deviation(element_idx)

    def _compute_compactness(self, element_idx: int) -> float:
        return super()._compute_compactness(element_idx)

    def _compute_convex_hull_ratio_2d(self, element_idx: int) -> float:
        """Ratio between surface area and convex hull area (XY projection)."""
        surface_area = self._compute_surface_area(element_idx)
        surface = self.obj["geometry"][0]["boundaries"][element_idx]
        coords = np.array(self._resolve_indices(surface[0]))[:, :2]
        if coords.shape[0] < 3:
            return 0.0
        hull = geo_utils.ConvexHull(coords)
        hull_area = hull.area
        return float(surface_area / hull_area) if hull_area > 0 else 0.0

    def _compute_neighbor_count(self, element_idx: int) -> int:
        """
        Placeholder: number of adjacent surfaces that share edges with this ground surface.
        In practice, likely injected by pipeline (topological preprocessing).
        """
        # Without global adjacency info, default to 0
        return 0

    def _compute_footprint_alignment(self, element_idx: int) -> float:
        """
        Angle between the surface's orientation (normal projected to XY) and x-axis.
        """
        angle = super()._compute_aspect_direction(element_idx)
        # normalize angle to [0, pi/2] to measure alignment (0 = aligned with axis)
        return float(min(abs(angle), np.pi - abs(angle)))
