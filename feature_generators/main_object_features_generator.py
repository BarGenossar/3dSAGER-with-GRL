from typing import Any, Dict, Optional
import numpy as np
from feature_generators.base_features_generator import BaseFeaturesGenerator
from feature_generators import geo_utils
import torch


class MainObjectFeaturesGenerator(BaseFeaturesGenerator):
    """
    Feature generator for the entire 3D object (building-level features).
    """

    def __init__(self, obj: Dict[str, Any], feature_spec: Optional[list] = None, logger: Any = None):
        super().__init__(obj=obj, feature_spec=feature_spec, logger=logger)

        # Precompute mesh, vertices, centroid, eigen decomposition
        self._polygon_mesh = self._collect_polygon_mesh()
        self._vertices_arr = np.array(self.vertices)
        self._centroid = np.mean(self._vertices_arr, axis=0) if len(self._vertices_arr) else np.zeros(3)
        self._eigvals, self._eigvecs = geo_utils.compute_eigendecomposition(self._vertices_arr)

    # -----------------
    # Mesh collection
    # -----------------
    def _collect_polygon_mesh(self):
        """Flatten boundaries into list of polygons (coords, not indices)."""
        polygons = []
        if "geometry" not in self.obj:
            return polygons
        for geom in self.obj["geometry"]:
            for boundary in geom.get("boundaries", []):
                outer = self._resolve_indices(boundary[0])
                polygons.append(outer)
        return polygons

    def get_vector(self) -> torch.Tensor:
        """
        Compute the feature vector for the whole object (not per surface).
        """
        features = []
        for feature_name in self.feature_spec:
            func = getattr(self, f"_compute_{feature_name}", None)
            if func is None:
                if self.logger:
                    self.logger.warning(f"Feature '{feature_name}' not implemented in MainObject.")
                features.append(0.0)
                continue
            try:
                value = func()   # note: no element_idx here
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error computing {feature_name} for MainObject: {e}")
                value = 0.0
            features.append(float(value))

        return torch.tensor(features, dtype=torch.float32)

    # -----------------
    # Feature functions
    # -----------------
    def _compute_area(self, _: int = 0) -> float:
        return geo_utils.compute_polygon_mesh_area(self._polygon_mesh)

    def _compute_perimeter(self, _: int = 0) -> float:
        """Sum of perimeters of all polygons in the object."""
        perim = 0.0
        for poly in self._polygon_mesh:
            n = len(poly)
            for i in range(n):
                v0, v1 = np.array(poly[i]), np.array(poly[(i + 1) % n])
                perim += np.linalg.norm(v1 - v0)
        return float(perim)

    def _compute_perimeter_index(self, _: int = 0) -> float:
        """Circularity index: 2 * sqrt(pi * area) / perimeter."""
        area = self._compute_area()
        perim = self._compute_perimeter()
        if perim == 0:
            return 0.0
        return float(2 * np.sqrt(np.pi * area) / perim)

    def _compute_volume(self, _: int = 0) -> float:
        return geo_utils.compute_polygon_mesh_volume(self._polygon_mesh)

    def _compute_bounding_box_width(self, _: int = 0) -> float:
        xs = self._vertices_arr[:, 0]
        return float(xs.max() - xs.min())

    def _compute_bounding_box_length(self, _: int = 0) -> float:
        ys = self._vertices_arr[:, 1]
        return float(ys.max() - ys.min())

    def _compute_aligned_bounding_box_width(self, _: int = 0) -> float:
        aligned = self._vertices_arr @ self._eigvecs
        return float(aligned[:, 0].max() - aligned[:, 0].min())

    def _compute_aligned_bounding_box_length(self, _: int = 0) -> float:
        aligned = self._vertices_arr @ self._eigvecs
        return float(aligned[:, 1].max() - aligned[:, 1].min())

    def _compute_aligned_bounding_box_height(self, _: int = 0) -> float:
        aligned = self._vertices_arr @ self._eigvecs
        return float(aligned[:, 2].max() - aligned[:, 2].min())

    def _compute_convex_hull_area(self, _: int = 0) -> float:
        if self._vertices_arr.shape[0] < 3:
            return 0.0
        return float(geo_utils.ConvexHull(self._vertices_arr[:, :2]).area)

    def _compute_convex_hull_volume(self, _: int = 0) -> float:
        if self._vertices_arr.shape[0] < 4:
            return 0.0
        return float(geo_utils.ConvexHull(self._vertices_arr).volume)

    def _compute_ave_centroid_distance(self, _: int = 0) -> float:
        dists = np.linalg.norm(self._vertices_arr - self._centroid, axis=1)
        return float(np.mean(dists))

    def _compute_max_height(self, _: int = 0) -> float:
        return float(self._vertices_arr[:, 2].max())

    def _compute_min_height(self, _: int = 0) -> float:
        return float(self._vertices_arr[:, 2].min())

    def _compute_height_diff(self, _: int = 0) -> float:
        return float(self._vertices_arr[:, 2].max() - self._vertices_arr[:, 2].min())

    def _compute_num_floors(self, _: int = 0) -> int:
        """Number of unique z-levels (approximation of floors)."""
        return int(len(np.unique(np.round(self._vertices_arr[:, 2], 2))))

    def _compute_axes_symmetry(self, _: int = 0) -> float:
        """Symmetry measure = mean of std(x), std(y), std(z)."""
        return float(np.mean([
            np.std(self._vertices_arr[:, 0]),
            np.std(self._vertices_arr[:, 1]),
            np.std(self._vertices_arr[:, 2])
        ]))

    def _compute_num_vertices(self, _: int = 0) -> int:
        return int(self._vertices_arr.shape[0])

    def _compute_elongation(self, _: int = 0) -> float:
        eigvals = np.clip(self._eigvals, 1e-9, None)
        return float(np.sqrt(eigvals.max() / eigvals.min()))

    def _compute_shape_index(self, _: int = 0) -> float:
        area = self._compute_area()
        perim = self._compute_perimeter()
        if area <= 0:
            return 0.0
        return perim / np.sqrt(4 * np.pi * area)

    def _compute_compactness_2d(self, _: int = 0) -> float:
        area = self._compute_area()
        hull_area = self._compute_convex_hull_area()
        return float(area / hull_area) if hull_area > 0 else 0.0

    def _compute_compactness_3d(self, _: int = 0) -> float:
        vol = self._compute_volume()
        hull_vol = self._compute_convex_hull_volume()
        return float(vol / hull_vol) if hull_vol > 0 else 0.0

    def _compute_density(self, _: int = 0) -> float:
        area = self._compute_area()
        perim = self._compute_perimeter()
        return float(area / perim) if perim > 0 else 0.0

    def _compute_hemisphericality(self, _: int = 0) -> float:
        area = self._compute_area()
        vol = self._compute_volume()
        if area <= 0:
            return 0.0
        return float(3 * np.sqrt(2) * np.sqrt(np.pi) * vol / (area ** 1.5))

    def _compute_fractality(self, _: int = 0) -> float:
        area = self._compute_area()
        vol = self._compute_volume()
        if area <= 0 or vol <= 0:
            return 0.0
        return float(1 - np.log(vol) / (1.5 * np.log(area)))

    def _compute_cubeness(self, _: int = 0) -> float:
        area = self._compute_area()
        vol = self._compute_volume()
        if area <= 0:
            return 0.0
        return float(6 * (vol ** (2.0 / 3.0)) / area)

    def _compute_circumference(self, _: int = 0) -> float:
        area = self._compute_area()
        vol = self._compute_volume()
        if area <= 0:
            return 0.0
        return float(4 * np.pi * ((3 * vol / (4 * np.pi)) ** (2.0 / 3.0)) / area)
