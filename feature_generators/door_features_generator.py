# feature_generators/DoorFeaturesGenerator.py

from typing import Any, Dict, Optional
import numpy as np
from .base_features_generator import BaseFeaturesGenerator


class DoorFeaturesGenerator(BaseFeaturesGenerator):
    """
    Feature generator for Door semantics.
    """

    def __init__(self, obj: Dict[str, Any], feature_spec: Optional[Dict[str, Any]] = None, logger: Any = None):
        super().__init__(obj=obj, feature_spec=feature_spec, logger=logger)

    # ----------------------------
    # Door-specific features
    # ----------------------------

    def _compute_door_area(self, element_idx: int) -> float:
        """Alias for surface area."""
        return super()._compute_surface_area(element_idx)

    def _compute_aspect_ratio(self, element_idx: int) -> float:
        """Aspect ratio of bounding box (width / height)."""
        coords = self._get_surface_coords(element_idx)
        if coords.shape[0] < 2:
            return 0.0
        xs, zs = coords[:, 0], coords[:, 2]  # width vs. height
        width = xs.max() - xs.min()
        height = zs.max() - zs.min()
        if height <= 1e-9:
            return 0.0
        return float(width / height)

    def _compute_relative_wall_position(self, element_idx: int) -> float:
        """
        Vertical relative position of door centroid inside parent wall.
        Normalized [0,1].
        """
        surface = self.surfaces[element_idx]
        parent_idx = surface.get("parent")
        if parent_idx is None:
            return 0.0
        door_coords = self._get_surface_coords(element_idx)
        wall_coords = self._get_surface_coords(parent_idx)
        z_door = np.mean(door_coords[:, 2])
        z_min, z_max = wall_coords[:, 2].min(), wall_coords[:, 2].max()
        if z_max - z_min <= 1e-9:
            return 0.0
        return float((z_door - z_min) / (z_max - z_min))

    def _compute_ground_adjacency(self, element_idx: int) -> float:
        """
        Proportion of door vertices that touch the global ground elevation.
        """
        coords = self._get_surface_coords(element_idx)
        ground_z = np.min(np.array(self.vertices)[:, 2])
        num_touching = np.sum(np.isclose(coords[:, 2], ground_z, atol=1e-2))
        return float(num_touching / max(1, coords.shape[0]))

    def _compute_width_at_base(self, element_idx: int) -> float:
        """
        Width of door along X-axis at its minimum Z elevation.
        """
        coords = self._get_surface_coords(element_idx)
        z_min = coords[:, 2].min()
        base_coords = coords[np.isclose(coords[:, 2], z_min, atol=1e-2)]
        if base_coords.shape[0] < 2:
            return 0.0
        xs = base_coords[:, 0]
        return float(xs.max() - xs.min())

    def _compute_orientation(self, element_idx: int) -> float:
        """Reuse base aspect_direction."""
        return super()._compute_aspect_direction(element_idx)

    def _compute_door_to_wall_ratio(self, element_idx: int) -> float:
        """
        Ratio of this door's area to its parent wall area.
        """
        surface = self.surfaces[element_idx]
        parent_idx = surface.get("parent")
        if parent_idx is None:
            return 0.0
        door_area = self._compute_door_area(element_idx)
        wall_area = super()._compute_surface_area(parent_idx)
        if wall_area <= 1e-9:
            return 0.0
        return float(door_area / wall_area)

    def _compute_entrance_prominence(self, element_idx: int) -> float:
        """
        Proxy for prominence: door area relative to mean window area on same wall.
        >1 means door is larger than average window.
        """
        surface = self.surfaces[element_idx]
        parent_idx = surface.get("parent")
        if parent_idx is None:
            return 0.0

        siblings = self.surfaces[parent_idx].get("children", [])
        window_areas = [
            BaseFeaturesGenerator._compute_surface_area(self, c)
            for c in siblings if self.surfaces[c].get("type") == "Window"
        ]

        mean_window_area = np.mean(window_areas) if window_areas else 1.0
        door_area = self._compute_door_area(element_idx)
        return float(door_area / mean_window_area)

    def _compute_neighbor_count(self, element_idx: int) -> int:
        """
        Number of sibling elements (windows/doors) in same wall.
        """
        surface = self.surfaces[element_idx]
        parent_idx = surface.get("parent")
        if parent_idx is None:
            return 0
        siblings = self.surfaces[parent_idx].get("children", [])
        return len(siblings) - 1  # exclude the door itself
