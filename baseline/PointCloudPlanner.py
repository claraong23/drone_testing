import numpy as np
from scipy.spatial import KDTree

class PointCloudPlanner:
    def __init__(self, safety_margin=0.5):
        self.safety_margin = safety_margin
        self.tree = None
        self.bounds = None  # (min_north, max_north, min_east, max_east)
        
    def update_map(self, north_east_points, bounds=None):
        """Rebuild spatial index after mapper accumulates new points"""
        if len(north_east_points) == 0:
            self.tree = None
            return
        self.tree = KDTree(north_east_points)
        self.bounds = bounds
        
    def is_collision_free(self, point, inflation=None):
        """Returns True if point is safe to fly through"""
        if self.tree is None:
            return True  # No obstacles mapped yet
        
        radius = inflation if inflation is not None else self.safety_margin
        dist, _ = self.tree.query(point, k=1)
        return dist > radius
    
    def get_nearest_obstacle(self, point):
        """Returns distance and coordinates of closest obstacle"""
        if self.tree is None:
            return np.inf, None
        dist, idx = self.tree.query(point, k=1)
        return dist, self.tree.data[idx]