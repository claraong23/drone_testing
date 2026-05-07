import asyncio
from mavsdk import System


async def check_drone_health():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            break

    async for battery in drone.telemetry.battery():
        print(f"Battery: {battery.remaining_percent:.1f}%")
        print(f"Voltage: {battery.voltage_v:.2f}V")
        break

    async for gps in drone.telemetry.gps_info():
        print(f"GPS satellites: {gps.num_satellites}")
        print(f"GPS fix type: {gps.fix_type}")
        break

    async for health in drone.telemetry.health():
        print(f"Global position OK: {health.is_global_position_ok}")
        print(f"Home position OK: {health.is_home_position_ok}")
        break


if __name__ == "__main__":
    asyncio.run(check_drone_health())