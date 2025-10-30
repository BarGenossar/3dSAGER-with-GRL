
import numpy as np
from typing import Any, Dict, Optional
from .base_features_generator import BaseFeaturesGenerator
from . import geo_utils


class RoofFeaturesGenerator(BaseFeaturesGenerator):
    """
    Feature generator for RoofSurface polygons.
    """

    def __init__(self, obj: Dict[str, Any], feature_spec: Optional[list] = None, logger: Any = None):
        super().__init__(obj=obj, feature_spec=feature_spec, logger=logger)
        # cache global min_z for height_above_ground
        self._global_min_z = np.min(np.array(self.vertices)[:, 2]) if len(self.vertices) else 0.0

    # -----------------
    # Features
    # -----------------
    def _compute_surface_area(self, element_idx: int) -> float:
        return super()._compute_surface_area(element_idx)

    def _compute_slope_angle(self, element_idx: int) -> float:
        """
        Angle (radians) between surface normal and XY plane.
        0 = horizontal, pi/2 = vertical.
        """
        surface = self.obj["geometry"][0]["boundaries"][element_idx]
        coords = np.array(self._resolve_indices(surface[0]))
        if coords.shape[0] < 3:
            return 0.0
        v1, v2, v3 = coords[0], coords[1], coords[2]
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        # dot with vertical axis [0,0,1]
        cos_theta = abs(np.dot(normal, np.array([0, 0, 1])))
        return float(np.arccos(cos_theta))

    def _compute_aspect_direction(self, element_idx: int) -> float:
        return super()._compute_aspect_direction(element_idx)

    def _compute_height_above_ground(self, element_idx: int) -> float:
        """Mean z of surface − min z of building."""
        surface = self.obj["geometry"][0]["boundaries"][element_idx]
        coords = np.array(self._resolve_indices(surface[0]))
        if coords.size == 0:
            return 0.0
        mean_z = float(np.mean(coords[:, 2]))
        return mean_z - self._global_min_z

    def _compute_convexity_index(self, element_idx: int) -> float:
        """Surface area / convex hull area of its footprint (XY)."""
        area = self._compute_surface_area(element_idx)
        surface = self.obj["geometry"][0]["boundaries"][element_idx]
        coords_2d = np.array(self._resolve_indices(surface[0]))[:, :2]
        if coords_2d.shape[0] < 3:
            return 0.0
        hull_area = geo_utils.ConvexHull(coords_2d).area
        return float(area / hull_area) if hull_area > 0 else 0.0

    def _compute_planarity_deviation(self, element_idx: int) -> float:
        return super()._compute_planarity_deviation(element_idx)

    def _compute_neighbor_connectivity(self, element_idx: int) -> int:
        """
        Placeholder: number of adjacent surfaces.
        Should be populated from adjacency graph in pipeline.
        """
        return 0

    def _compute_overhang_ratio(self, element_idx: int) -> float:
        """
        Ratio = roof footprint area / convex hull of footprint.
        Captures overhanging complexity.
        """
        surface = self.obj["geometry"][0]["boundaries"][element_idx]
        coords_2d = np.array(self._resolve_indices(surface[0]))[:, :2]
        if coords_2d.shape[0] < 3:
            return 0.0
        hull = geo_utils.ConvexHull(coords_2d)
        hull_area = hull.area
        footprint_area = geo_utils.compute_polygon_area(np.c_[coords_2d, np.zeros(len(coords_2d))])
        return float(footprint_area / hull_area) if hull_area > 0 else 0.0

    def _compute_roof_type_encoding(self, element_idx: int) -> float:
        """
        Placeholder categorical encoding.
        In practice: gabled, flat, hip, etc. Could come from preprocessing.
        """
        return 0.0

    def _compute_surface_fractality(self, element_idx: int) -> float:
        """
        Simple fractality proxy: perimeter^2 / area.
        Higher values = more irregular boundary.
        """
        surface_area = self._compute_surface_area(element_idx)
        perim = super()._compute_perimeter_length(element_idx)
        return float((perim ** 2) / (surface_area + 1e-9))
