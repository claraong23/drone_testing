import asyncio
import math

from mavsdk import System
from mavsdk.action import ActionError


EARTH_RADIUS_M = 6378137.0


class Drone:
    def __init__(self, system_address="udpin://0.0.0.0:14540", takeoff_altitude=2.0):
        self.system_address = system_address
        self.takeoff_altitude = float(takeoff_altitude)
        self.drone = System()

    async def connect(self):
        await self.drone.connect(system_address=self.system_address)

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Connected")
                break

    async def wait_until_ready(self, timeout_s=30.0):
        """
        Be less strict than requiring global+home.
        Accept armable, or local position OK, or global+home.
        """
        loop = asyncio.get_running_loop()
        start = loop.time()

        async for health in self.drone.telemetry.health():
            armable = getattr(health, "is_armable", False)
            local_ok = getattr(health, "is_local_position_ok", False)
            global_ok = getattr(health, "is_global_position_ok", False)
            home_ok = getattr(health, "is_home_position_ok", False)

            print(
                f"Health: armable={armable}, "
                f"local_ok={local_ok}, global_ok={global_ok}, home_ok={home_ok}"
            )

            if armable or local_ok or (global_ok and home_ok):
                print("Vehicle readiness condition satisfied")
                return

            if loop.time() - start > timeout_s:
                raise TimeoutError(
                    "Timed out waiting for readiness "
                    f"(armable={armable}, local_ok={local_ok}, "
                    f"global_ok={global_ok}, home_ok={home_ok})"
                )

    async def arm_and_takeoff(self):
        await self.wait_until_ready()

        try:
            await self.drone.action.arm()
        except ActionError as e:
            raise RuntimeError(f"Arm failed: {e}") from e

        try:
            await self.drone.action.set_takeoff_altitude(self.takeoff_altitude)
        except Exception:
            pass

        print("Takeoff")
        try:
            await self.drone.action.takeoff()
        except ActionError as e:
            raise RuntimeError(f"Takeoff failed: {e}") from e

        await asyncio.sleep(8)

    async def _get_global_position(self):
        async for pos in self.drone.telemetry.position():
            return (
                float(pos.latitude_deg),
                float(pos.longitude_deg),
                float(pos.absolute_altitude_m),
                float(pos.relative_altitude_m),
            )
        raise RuntimeError("Failed to read global position telemetry")

    @staticmethod
    def _offset_latlon(latitude_deg, longitude_deg, north_m, east_m):
        dlat = (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
        dlon = (east_m / (EARTH_RADIUS_M * math.cos(math.radians(latitude_deg)))) * (180.0 / math.pi)
        return latitude_deg + dlat, longitude_deg + dlon

    async def send_position_setpoint(self, north, east, down, yaw_deg=0.0):
        curr_lat, curr_lon, curr_abs_alt, curr_rel_alt = await self._get_global_position()

        target_lat, target_lon = self._offset_latlon(
            curr_lat, curr_lon, float(north), float(east)
        )
        target_abs_alt = curr_abs_alt - float(down)

        try:
            await self.drone.action.goto_location(
                target_lat,
                target_lon,
                target_abs_alt,
                float(yaw_deg),
            )
        except ActionError as e:
            raise RuntimeError(f"goto_location failed: {e}") from e

    async def land(self):
        try:
            print("Land")
            await self.drone.action.land()
            await asyncio.sleep(8)
        except Exception as e:
            print(f"Landing failed: {e}")