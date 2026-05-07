import asyncio
import numpy as np
import threading
import matplotlib.pyplot as plt

from depth_receiver import DepthReceiver
from drone_control import Drone

drone = Drone()

# =========================
# POINT CLOUD
# =========================
class PointCloud:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def convert(self, depth):
        depth = depth[::2, ::2]
        h, w = depth.shape

        i, j = np.meshgrid(np.arange(w), np.arange(h))

        z = depth
        x = (i - self.cx/2) * z / self.fx
        y = (j - self.cy/2) * z / self.fy

        mask = z > 0
        return np.stack((x[mask], y[mask], z[mask]), axis=-1)


# =========================
# GLOBAL MAP (LOG-ODDS)
# =========================
class GlobalMap:
    def __init__(self, size=400, resolution=0.1):
        self.size = size
        self.resolution = resolution
        self.log_odds = np.zeros((size, size))

        self.l_occ = 0.85
        self.l_free = -0.4
        self.l_min = -2.0
        self.l_max = 3.5

    def world_to_grid(self, x, y):
        center = self.size // 2
        gx = int(x / self.resolution) + center
        gy = int(y / self.resolution) + center
        return gx, gy

    def update(self, points):
        for x, y, z in points:
            gx, gy = self.world_to_grid(x, y)

            if 0 <= gx < self.size and 0 <= gy < self.size:
                if z < 2.0:
                    self.log_odds[gy, gx] += self.l_occ
                else:
                    self.log_odds[gy, gx] += self.l_free

        self.log_odds = np.clip(self.log_odds, self.l_min, self.l_max)

    def get_binary(self):
        return (self.log_odds > 0).astype(np.uint8)

    def get_prob(self):
        return 1 - 1/(1 + np.exp(self.log_odds))


# =========================
# TRANSFORM
# =========================
def transform_points(points, x, y, yaw):
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    world = []
    for px, py, pz in points:
        wx = x + (px * cos_y - py * sin_y)
        wy = y + (px * sin_y + py * cos_y)
        world.append([wx, wy, pz])

    return world


# =========================
# DFS PATH PLANNING
# =========================
def dfs(grid, start, goal):
    stack = [start]
    visited = set()
    parent = {}

    while stack:
        node = stack.pop()

        if node == goal:
            break

        if node in visited:
            continue

        visited.add(node)

        x, y = node
        neighbors = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]

        for nx, ny in neighbors:
            if (0 <= nx < grid.shape[1] and
                0 <= ny < grid.shape[0] and
                grid[ny, nx] == 0):

                stack.append((nx, ny))
                parent[(nx, ny)] = node

    path = []
    node = goal
    while node in parent:
        path.append(node)
        node = parent[node]

    path.reverse()
    return path


# =========================
# FRONTIER DETECTION
# =========================
def detect_frontiers(log_odds):
    frontiers = []
    h, w = log_odds.shape

    for y in range(1, h-1):
        for x in range(1, w-1):

            # free space
            if log_odds[y, x] < 0:

                neighbors = [
                    log_odds[y+1, x],
                    log_odds[y-1, x],
                    log_odds[y, x+1],
                    log_odds[y, x-1],
                ]

                # adjacent unknown
                if any(abs(n) < 0.05 for n in neighbors):
                    frontiers.append((x, y))

    return frontiers


def select_frontier(frontiers, start):
    if not frontiers:
        return None
    return min(frontiers, key=lambda f: (f[0]-start[0])**2 + (f[1]-start[1])**2)


# =========================
# CONTROL
# =========================
def compute_yaw(x, y, tx, ty):
    return np.arctan2(ty - y, tx - x)

def yaw_error(curr, target):
    err = target - curr
    return (err + np.pi) % (2*np.pi) - np.pi

def compute_command(pose, target):
    x, y, yaw = pose
    tx, ty = target

    target_yaw = compute_yaw(x, y, tx, ty)
    err = yaw_error(yaw, target_yaw)

    if abs(err) > 0.2:
        return (0.0, 0.0, 0.0, err)
    else:
        return (1.0, 0.0, 0.0, err)


# =========================
# VELOCITY SMOOTHER
# =========================
class VelocitySmoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev = np.zeros(4)

    def smooth(self, cmd):
        cmd = np.array(cmd)
        smoothed = self.alpha * cmd + (1 - self.alpha) * self.prev
        self.prev = smoothed
        return smoothed


# =========================
# VISUALIZER (THREAD)
# =========================
class Visualizer:
    def __init__(self):
        self.grid = None
        self.path = None
        self.frontiers = None
        self.lock = threading.Lock()

        # Initialize GUI in MAIN thread
        plt.ion()
        self.fig, self.ax = plt.subplots()

    def update_data(self, grid, path=None, frontiers=None):
        with self.lock:
            self.grid = grid.copy()
            self.path = path
            self.frontiers = frontiers

    def draw(self):
        with self.lock:
            if self.grid is None:
                return

            grid = self.grid.copy()
            path = self.path
            frontiers = self.frontiers

        self.ax.clear()
        self.ax.imshow(grid, cmap='gray')

        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            self.ax.plot(xs, ys)

        if frontiers:
            xs = [f[0] for f in frontiers]
            ys = [f[1] for f in frontiers]
            self.ax.scatter(xs, ys, s=2)

        plt.draw()
        plt.pause(0.001)
# =========================
# MAIN
# =========================
async def main():



    depth_topic = "/depth_camera"

    depth_cam = DepthReceiver(depth_topic)
    pc = PointCloud(320, 320, 320, 240)
    global_map = GlobalMap()

    smoother = VelocitySmoother()
    vis = Visualizer()

    await drone.connect()
    await drone.arm_and_takeoff()

    vis = Visualizer()


    while True:


        depth = depth_cam.get_frame()
        if depth is None:
            await asyncio.sleep(0.01)
            continue
        print("11")
        points = pc.convert(depth)

        x, y, z = await drone.get_position()
        yaw = await drone.get_yaw()

        world_pts = transform_points(points, x, y, yaw)
        global_map.update(world_pts)

        grid = global_map.get_binary()
        start = global_map.world_to_grid(x, y)

        frontiers = detect_frontiers(global_map.log_odds)

        # ✅ termination condition
        if len(frontiers) == 0:
            print("🎉 Exploration COMPLETE")
            await drone.send_velocity(0, 0, 0, 0)
            break

        goal = select_frontier(frontiers, start)

        path = dfs(grid, start, goal)

        if path:
            target = path[min(5, len(path)-1)]

            tx = (target[0] - global_map.size//2) * global_map.resolution
            ty = (target[1] - global_map.size//2) * global_map.resolution

            raw_cmd = compute_command((x, y, yaw), (tx, ty))
            vx, vy, vz, yaw_rate = smoother.smooth(raw_cmd)

            await drone.send_velocity(vx, vy, vz, yaw_rate)


        vis.update_data(global_map.get_prob(), path, frontiers)
        vis.draw()

        await asyncio.sleep(0.05)


if __name__ == "__main__":
    asyncio.run(main())