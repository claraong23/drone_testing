#!/usr/bin/env python3
import asyncio
import numpy as np
import time

from depth_receiver import DepthReceiver
from drone_control import Drone
from VelocityPlanner import VelocityPlanner


class DroneNavigation:
    def __init__(self,
                 depth_topic="/depth_camera",
                 loop_hz=20.0):

        self.loop_hz = loop_hz
        self.running = True

        # =========================
        # 🧭 GRID HEADING SYSTEM
        # =========================
        self.grid_headings = [0, 90, 180, -90]  # N, E, S, W
        self.current_heading_idx = 0
        self.target_yaw_deg = self.grid_headings[self.current_heading_idx]
        self.yaw_tolerance = 5.0

        # Camera intrinsics
        K = np.array([[433.0, 0.0, 320.0],
                      [0.0, 433.0, 240.0],
                      [0.0, 0.0, 1.0]])

        self.receiver = DepthReceiver(depth_topic)
        self.planner = VelocityPlanner(K=K,width=640,height=480,safe_distance=4.0,critical_distance=1.5)

        self.drone = Drone()

    # =========================
    # 🧠 YAW UTILS
    # =========================
    def _yaw_error(self, target, current):
        error = target - current
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        return error

    def camera_to_ned(self,vx_cam, vy_cam, yaw_deg):
        yaw = np.deg2rad(yaw_deg)
        vx_ned = vx_cam * np.cos(yaw) - vy_cam * np.sin(yaw)
        vy_ned = vx_cam * np.sin(yaw) + vy_cam * np.cos(yaw)
        return vx_ned, vy_ned

    async def align_to_grid(self):
        current_yaw = await self.drone.get_yaw()
        error = self._yaw_error(self.target_yaw_deg, current_yaw)

        if abs(error) > self.yaw_tolerance:
            print(f"🧭 Aligning to {self.target_yaw_deg}° (err={error:.2f})")
            await self.drone.rotate_to_yaw(self.target_yaw_deg)

    # =========================
    # 🔄 GRID TURNING
    # =========================
    async def rotate_next_direction(self):
        self.current_heading_idx = (self.current_heading_idx + 1) % 4
        self.target_yaw_deg = self.grid_headings[self.current_heading_idx]

        print(f"🔄 New heading: {self.target_yaw_deg}°")
        await self.drone.rotate_to_yaw(self.target_yaw_deg)

    # =========================
    # 🚁 MAIN LOOP
    # =========================
    async def run(self):
        print("\n🚁 GRID-ALIGNED AUTONOMOUS NAVIGATION\n")

        await self.drone.connect()
        await self.drone.arm_and_takeoff()

        # Initial alignment
        await self.drone.rotate_to_yaw(self.target_yaw_deg)

        try:
            while self.running:
                t_start = time.monotonic()

                depth_frame = self.receiver.get_frame()
                # vy, vx = self.planner.compute_velocity(depth_frame,goal_angle=0)
                vx,vy,info = self.planner.compute_velocity(depth_frame)
                n_vx, n_vy = self.camera_to_ned(vx_cam=vx, vy_cam=vy,yaw_deg=self.target_yaw_deg)

                # ===================================
                # THIS IS WHERE TO HAVE CODES
                # FOR MAKING HIGHER DECISION MAKING
                # NEEDS TO BE ADDED BUT NOT ADDED YET.
                # ==================================
                c = info['clearance']
                print(f"Blocked={info['blocked']} | vx={n_vx:.2f}, vy={n_vy:.2f} | left-{c['left']} center-{c['center']} right- {c['right']}")

                if info['blocked']:
                    await self.drone.send_velocity(0, 0, 0, self.target_yaw_deg)
                    await self.rotate_next_direction()
                else:
                    # Ensure alignment before motion
                    await self.align_to_grid()
                    await self.drone.send_velocity(
                        vx=n_vx,
                        vy=n_vy,
                        vz=0.0,
                        yaw_deg=self.target_yaw_deg
                    )

                # Maintain loop timing
                elapsed = time.monotonic() - t_start
                sleep_time = (1.0 / self.loop_hz) - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            print("🛑 Navigation cancelled")

        finally:
            await self.drone.send_velocity(0, 0, 0, self.target_yaw_deg)
            print("✅ Drone hovering safely")

    def stop(self):
        self.running = False


# =========================
# 🚀 ENTRY POINT
# =========================
async def main():
    nav = DroneNavigation()

    task = asyncio.create_task(nav.run())

    try:
        await task
    except KeyboardInterrupt:
        print("\n⌨️ Stopping...")
        nav.stop()
        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())