# feature_generators/WallSurfaceFeaturesGenerator.py

from typing import Any, Dict, Optional
import numpy as np
from .base_features_generator import BaseFeaturesGenerator
from . import geo_utils


class WallFeaturesGenerator(BaseFeaturesGenerator):
    """
    Feature generator for WallSurface semantics.
    """

    def __init__(self, obj: Dict[str, Any], feature_spec: Optional[Dict[str, Any]] = None, logger: Any = None):
        super().__init__(obj=obj, feature_spec=feature_spec, logger=logger)

    # ----------------------------
    # Wall-specific feature methods
    # ----------------------------

    def _compute_wall_area(self, element_idx: int) -> float:
        """Alias: same as surface_area, but renamed for clarity."""
        return super()._compute_surface_area(element_idx)

    def _compute_height_extent(self, element_idx: int) -> float:
        """Vertical extent of wall surface (max Z - min Z)."""
        coords = self._get_surface_coords(element_idx)
        zs = coords[:, 2]
        return float(np.max(zs) - np.min(zs))

    def _compute_aspect_direction(self, element_idx: int) -> float:
        """Reuse base computation: orientation of normal in XY plane."""
        return super()._compute_aspect_direction(element_idx)

    def _compute_window_to_wall_ratio(self, element_idx: int) -> float:
        """
        Ratio of total window area to wall area.
        Requires that obj['surfaces'][element_idx] has 'children' pointing to windows.
        """
        wall_area = self._compute_wall_area(element_idx)
        if wall_area < 1e-9:
            return 0.0
        children = self.surfaces[element_idx].get("children", [])
        window_area = 0.0
        for child_idx in children:
            child = self.surfaces[child_idx]
            if child.get("type") == "Window":
                window_area += super()._compute_surface_area(child_idx)
        return float(window_area / wall_area)

    def _compute_door_to_wall_ratio(self, element_idx: int) -> float:
        """
        Ratio of total door area to wall area.
        Similar to window ratio but for Door children.
        """
        wall_area = self._compute_wall_area(element_idx)
        if wall_area < 1e-9:
            return 0.0
        children = self.surfaces[element_idx].get("children", [])
        door_area = 0.0
        for child_idx in children:
            child = self.surfaces[child_idx]
            if child.get("type") == "Door":
                door_area += super()._compute_surface_area(child_idx)
        return float(door_area / wall_area)

    def _compute_elongation(self, element_idx: int) -> float:
        """
        Ratio of major to minor axis in XY projection of wall.
        Uses eigen decomposition of 2D coords.
        """
        coords = self._get_surface_coords(element_idx)[:, :2]  # project XY
        if coords.shape[0] < 3:
            return 0.0
        cov = np.cov(coords, rowvar=False)
        vals, _ = np.linalg.eigh(cov)
        vals = np.sort(vals)
        if vals[0] <= 1e-12:
            return 0.0
        return float(np.sqrt(vals[-1] / vals[0]))

    def _compute_planarity_deviation(self, element_idx: int) -> float:
        """Reuse base: RMS distance of vertices to best-fit plane."""
        return super()._compute_planarity_deviation(element_idx)

    def _compute_neighbor_type_count(self, element_idx: int) -> int:
        """
        Count distinct semantic types connected to this wall.
        Placeholder: requires adjacency graph from pipeline.
        For now, returns number of unique child types.
        """
        children = self.surfaces[element_idx].get("children", [])
        types = set()
        for child_idx in children:
            types.add(self.surfaces[child_idx].get("type"))
        return len(types)

    def _compute_surface_regularity(self, element_idx: int) -> float:
        """
        Regularity proxy: ratio of wall area to convex hull area in XY.
        Higher = more regular.
        """
        area = self._compute_wall_area(element_idx)
        coords = self._get_surface_coords(element_idx)[:, :2]
        coords = np.unique(coords, axis=0)
        if coords.shape[0] < 3:
            return 0.0
        try:
            hull = geo_utils.ConvexHull(coords)
            hull_area = hull.area
        except Exception:
            return 0.0
        if hull_area < 1e-9:
            return 0.0
        return float(area / hull_area)

    def _compute_facade_segmentation(self, element_idx: int) -> int:
        """
        Count number of child elements (windows/doors) as a proxy for segmentation.
        """
        children = self.surfaces[element_idx].get("children", [])
        return len(children)
