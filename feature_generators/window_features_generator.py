# feature_generators/WindowFeaturesGenerator.py

from typing import Any, Dict, Optional
import numpy as np
from .base_features_generator import BaseFeaturesGenerator


class WindowFeaturesGenerator(BaseFeaturesGenerator):
    """
    Feature generator for Window semantics.
    """

    def __init__(self, obj: Dict[str, Any], feature_spec: Optional[Dict[str, Any]] = None, logger: Any = None):
        super().__init__(obj=obj, feature_spec=feature_spec, logger=logger)

    # ----------------------------
    # Window-specific features
    # ----------------------------

    def _compute_window_area(self, element_idx: int) -> float:
        """Alias: reuse base surface_area."""
        return super()._compute_surface_area(element_idx)

    def _compute_aspect_ratio(self, element_idx: int) -> float:
        """Aspect ratio of bounding box (width / height)."""
        coords = self._get_surface_coords(element_idx)
        if coords.shape[0] < 2:
            return 0.0
        xs, ys = coords[:, 0], coords[:, 1]
        width = xs.max() - xs.min()
        height = ys.max() - ys.min()
        if height <= 1e-9:
            return 0.0
        return float(width / height)

    def _compute_relative_wall_position(self, element_idx: int) -> float:
        """
        Vertical relative position of window centroid inside parent wall.
        Normalized [0,1]: 0=bottom, 1=top.
        """
        surface = self.surfaces[element_idx]
        parent_idx = surface.get("parent")
        if parent_idx is None:
            return 0.0
        window_coords = self._get_surface_coords(element_idx)
        wall_coords = self._get_surface_coords(parent_idx)
        z_window = np.mean(window_coords[:, 2])
        z_min, z_max = wall_coords[:, 2].min(), wall_coords[:, 2].max()
        if z_max - z_min <= 1e-9:
            return 0.0
        return float((z_window - z_min) / (z_max - z_min))

    def _compute_elevation_above_ground(self, element_idx: int) -> float:
        """Mean Z of window centroid above min ground elevation."""
        window_coords = self._get_surface_coords(element_idx)
        z_window = np.mean(window_coords[:, 2])
        ground_z = np.min(np.array(self.vertices)[:, 2])
        return float(z_window - ground_z)

    def _compute_orientation(self, element_idx: int) -> float:
        """Reuse base aspect_direction (orientation in XY)."""
        return super()._compute_aspect_direction(element_idx)

    def _compute_shape_complexity(self, element_idx: int) -> float:
        """
        Complexity proxy = perimeter^2 / area.
        Larger values = more irregular shapes.
        """
        area = super()._compute_surface_area(element_idx)
        perim = super()._compute_perimeter_length(element_idx)
        if area <= 1e-9:
            return 0.0
        return float((perim ** 2) / area)

    def _compute_density_per_wall(self, element_idx: int) -> float:
        """
        Number of windows per parent wall surface area.
        """
        surface = self.surfaces[element_idx]
        parent_idx = surface.get("parent")
        if parent_idx is None:
            return 0.0
        # count windows
        siblings = self.surfaces[parent_idx].get("children", [])
        num_windows = sum(
            1 for child_idx in siblings if self.surfaces[child_idx].get("type") == "Window"
        )
        wall_area = super()._compute_surface_area(parent_idx)
        if wall_area <= 1e-9:
            return 0.0
        return float(num_windows / wall_area)

    def _compute_symmetry_indicator(self, element_idx: int) -> float:
        """
        Symmetry proxy: ratio of width to height closeness to 1.
        Perfect square window → 1.0.
        """
        coords = self._get_surface_coords(element_idx)
        if coords.shape[0] < 2:
            return 0.0
        xs, ys = coords[:, 0], coords[:, 1]
        width = xs.max() - xs.min()
        height = ys.max() - ys.min()
        if height <= 1e-9:
            return 0.0
        ratio = width / height
        return float(min(ratio, 1.0 / ratio))

    def _compute_material_flag(self, element_idx: int) -> int:
        """
        Material flag: placeholder binary feature (1=glass).
        Could later be linked to obj['attributes'] if material info exists.
        """
        surface = self.surfaces[element_idx]
        mat = surface.get("material", "").lower()
        return int("glass" in mat) if mat else 1  # default assume glass
