import asyncio
from mavsdk import System, telemetry
from mavsdk.action import ActionError

async def run():
    # 1. Initialize the MAVSDK system object
    drone = System()

    # 2. Listen for drone to broadcast and then connect to PX4 SITL over UDP (default SITL port)
    print("Connecting to PX4 SITL on udp://:14540 ...")
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    # 3. Wait until reports a successful connection
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"✅ Connected to drone")
            break

    # 4. Wait for the system to be flight-ready (GPS/Home lock, etc.)
    print("Waiting for system health checks...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("✅ System is healthy. Ready for commands.")
            break

    try:
        # 5. Takeoff to default altitude (usually 5m unless configured otherwise)
        print("-- Arming")
        await drone.action.arm()

        print("🚀 Taking off...")
        await drone.action.takeoff()
        await asyncio.sleep(10)

        # 6. Hover for 5 seconds
        print("⏳ Hovering for 5 seconds...")
        await asyncio.sleep(5)

        # 7. Initiate landing
        print("🛬 Landing...")
        await drone.action.land()

        # 8. Wait until the drone actually touches the ground
        async for landed in drone.telemetry.landed_state():
            if landed == telemetry.LandedState.ON_GROUND:
                print("✅ Landed successfully. Script finished.")
                break

    except ActionError as e:
        print(f"❌ Action failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    # Run the asynchronous event loop
    asyncio.run(run())