import numpy as np
from scipy.spatial import KDTree


class PointCloudPlanner:
    def __init__(self, safety_margin=0.5):
        self.safety_margin = float(safety_margin)
        self.tree = None
        self.points = np.empty((0, 2), dtype=np.float32)
        self.bounds = None  # (min_north, max_north, min_east, max_east)

    def update_map(self, north_east_points, bounds=None):
        """
        Rebuild spatial index after mapper accumulates new points.

        north_east_points: array-like of shape (N, 2), columns = [north, east]
        bounds: optional tuple (min_north, max_north, min_east, max_east)
        """
        pts = np.asarray(north_east_points, dtype=np.float32)

        if pts.size == 0:
            self.tree = None
            self.points = np.empty((0, 2), dtype=np.float32)
            self.bounds = bounds
            return

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("north_east_points must have shape (N, 2)")

        self.points = pts
        self.tree = KDTree(self.points)
        self.bounds = bounds

    def in_bounds(self, point):
        """Return True if point lies inside planner bounds, or if no bounds are set."""
        if self.bounds is None:
            return True

        north, east = float(point[0]), float(point[1])
        min_north, max_north, min_east, max_east = self.bounds
        return (min_north <= north <= max_north) and (min_east <= east <= max_east)

    def is_collision_free(self, point, inflation=None):
        """
        Returns True if point is safe to fly through.
        A point outside bounds is treated as not collision-free.
        """
        if not self.in_bounds(point):
            return False

        if self.tree is None:
            return True  # No obstacles mapped yet

        radius = float(inflation) if inflation is not None else self.safety_margin
        dist, _ = self.tree.query(point, k=1)
        return dist > radius

    def get_nearest_obstacle(self, point):
        """Returns (distance, coordinates) of closest obstacle."""
        if self.tree is None:
            return np.inf, None

        dist, idx = self.tree.query(point, k=1)
        return float(dist), self.points[int(idx)].copy()