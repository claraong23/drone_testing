import asyncio
import time

import matplotlib.pyplot as plt
import numpy as np

from top_down import depth_to_xy_map
from depth_receiver import DepthReceiver
from drone_control import Drone
from get_position_with_task import SharedState, position_monitor_task


class GlobalMapper:
    """
    Incremental top-down occupancy mapper using NED (North-East-Down) pose.

    Coordinate conventions:
    - pose: {'north': float, 'east': float, 'down': float, 'yaw': float}
    - yaw: radians by default unless yaw_in_degrees=True
    - yaw_clockwise=True means yaw is clockwise from North (NED heading)
    - local camera coordinates from depth_to_xy_map: [X_cam(right), Z_cam(forward)]
    """

    def __init__(
        self,
        K,
        cam_height=1.0,
        obs_h_min=0.1,
        obs_h_max=1.5,
        z_min=0.2,
        z_max=15.0,
        yaw_in_degrees=False,
        yaw_clockwise=True,
        yaw_smoothing=0.7,
    ):
        self.K = K
        self.cam_height = cam_height
        self.obs_h_min = obs_h_min
        self.obs_h_max = obs_h_max
        self.z_min = z_min
        self.z_max = z_max

        self.yaw_in_degrees = yaw_in_degrees
        self.yaw_clockwise = yaw_clockwise
        self.yaw_smoothing = yaw_smoothing
        self.last_yaw_rad = 0.0
        self.first_frame = True

        # accumulated global obstacle points in [north, east]
        self.global_points = np.empty((0, 2), dtype=np.float32)

    def _local_to_ned_global(self, local_xy, north, east, yaw_rad):
        """
        Transform local camera-plane coordinates to global NED coordinates.

        local_xy[:, 0] = X_cam = right
        local_xy[:, 1] = Z_cam = forward
        """
        X_cam = local_xy[:, 0]
        Z_cam = local_xy[:, 1]

        c = np.cos(yaw_rad)
        s = np.sin(yaw_rad)

        north_global = north + Z_cam * c - X_cam * s
        east_global = east + Z_cam * s + X_cam * c
        return np.column_stack([north_global, east_global])

    def update_frame(self, depth_img, pose):
        north = float(pose.get("north", 0.0))
        east = float(pose.get("east", 0.0))
        raw_yaw = pose.get("yaw", 0.0)

        if raw_yaw is None:
            print("Warning: yaw is None, using 0.0")
            raw_yaw = 0.0

        raw_yaw = float(raw_yaw)

        # convert yaw
        if self.yaw_in_degrees:
            raw_yaw = np.deg2rad(raw_yaw)

        if not self.yaw_clockwise:
            raw_yaw = -raw_yaw

        # simple exponential smoothing
        if self.first_frame:
            self.last_yaw_rad = raw_yaw
            self.first_frame = False
        else:
            raw_yaw = self.yaw_smoothing * raw_yaw + (1.0 - self.yaw_smoothing) * self.last_yaw_rad
            self.last_yaw_rad = raw_yaw

        yaw = raw_yaw

        # extract local obstacle coordinates
        xy_obstacles = depth_to_xy_map(
            depth_img,
            self.K,
            cam_height=self.cam_height,
            obs_h_min=self.obs_h_min,
            obs_h_max=self.obs_h_max,
            z_min=self.z_min,
            z_max=self.z_max,
        )

        if xy_obstacles is None or xy_obstacles.shape[0] == 0:
            return False

        # transform and accumulate
        global_pts = self._local_to_ned_global(xy_obstacles, north, east, yaw)
        self.global_points = np.vstack([self.global_points, global_pts.astype(np.float32)])
        return True

    def get_global_points(self):
        return self.global_points.copy()

    def save_points(self, filename="global_obstacles.npy"):
        np.save(filename, self.global_points)
        print(f"Saved {len(self.global_points)} points to {filename}")


def latest_pose_from_state(state):
    """
    Safely assemble a pose dict from async telemetry state.
    Never assumes latest_position/latest_yaw already exist.
    """
    pose = {
        "north": 0.0,
        "east": 0.0,
        "down": 0.0,
        "yaw": 0.0,
    }

    if getattr(state, "latest_position", None) is not None:
        pose["north"] = float(state.latest_position.north_m)
        pose["east"] = float(state.latest_position.east_m)
        pose["down"] = float(state.latest_position.down_m)

    if getattr(state, "latest_yaw", None) is not None:
        pose["yaw"] = float(state.latest_yaw)

    return pose


async def collect_and_map_frame(receiver, mapper, state, label=""):
    pose = latest_pose_from_state(state)

    depth_img = receiver.get_frame()
    if depth_img is None:
        print(f"{label} No depth data received yet.")
        return False, pose

    updated = mapper.update_frame(depth_img, pose)
    print(
        f"{label} Pose -> "
        f"N: {pose['north']:.2f}, E: {pose['east']:.2f}, D: {pose['down']:.2f}, Yaw: {pose['yaw']:.2f}"
    )
    return updated, pose


async def run():
    K = np.array([
        [433.0, 0.0, 320.0],
        [0.0, 433.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    receiver = DepthReceiver("/depth_camera")
    time.sleep(5)

    mapper = GlobalMapper(
        K,
        cam_height=1.0,
        obs_h_min=0.1,
        obs_h_max=1.5,
        z_min=0.3,
        z_max=5.0,
        yaw_in_degrees=True,
        yaw_smoothing=1.0,
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    stop_event = asyncio.Event()
    monitor_task = None
    drone = Drone()

    try:
        # connect and wait for pre-arm readiness inside Drone wrapper
        await drone.connect()
        await drone.arm_and_takeoff()

        state = SharedState()
        monitor_task = asyncio.create_task(position_monitor_task(drone, state, stop_event))

        # give the telemetry task a moment to populate state
        await asyncio.sleep(5)

        # Step 1: map, then move north 3 m three times
        for i in range(3):
            updated, pose = await collect_and_map_frame(receiver, mapper, state, label=f"[step {i}]")
            if not updated:
                print(f"[step {i}] Frame skipped")

            await drone.send_position_setpoint(
                north=3.0,
                east=0.0,
                down=0.0,
                yaw_deg=0.0,
            )
            await asyncio.sleep(5)

        # Step 2: yaw to 90 deg and map again
        pose = latest_pose_from_state(state)
        await drone.send_position_setpoint(
            north=0.0,
            east=0.0,
            down=0.0,
            yaw_deg=90.0,
        )
        await asyncio.sleep(5)

        await collect_and_map_frame(receiver, mapper, state, label="[turn]")

        # Step 3: move east 3 m and map again
        pose = latest_pose_from_state(state)
        await drone.send_position_setpoint(
            north=0.0,
            east=3.0,
            down=0.0,
            yaw_deg=0.0,
        )
        await asyncio.sleep(5)

        _, pose = await collect_and_map_frame(receiver, mapper, state, label="[east move]")

        # Plot results
        pts = mapper.get_global_points()
        ax.clear()

        if len(pts) > 0:
            dists = np.linalg.norm(pts, axis=1)
            ax.scatter(
                pts[:, 1],  # east
                pts[:, 0],  # north
                c=dists,
                s=4,
                cmap="viridis",
                edgecolors="none",
            )

        ax.plot(pose["east"], pose["north"], "r*", markersize=12, label="Drone")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.pause(0.05)
        plt.show()

        mapper.save_points("global_obstacles.npy")

    except Exception as e:
        print(f"GlobalMapper run failed: {e}")
        raise

    finally:
        stop_event.set()

        if monitor_task is not None:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                print("Position monitor task cancelled.")

        try:
            await drone.land()
        except Exception as e:
            print(f"Landing skipped or failed: {e}")


if __name__ == "__main__":
    asyncio.run(run())