import asyncio
from mavsdk import System, telemetry
from mavsdk.offboard import VelocityNedYaw
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError

async def run():
    drone = System()
    print("🔌 Connecting to PX4 SITL...")
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    # Wait for connection
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"✅ Connected to drone")
            break

    # Wait for system health
    print("🩺 Waiting for system health...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("✅ System healthy. Ready.")
            break


    print("-- Arming")
    await drone.action.arm()

    # Takeoff and stabilize
    print("🚀 Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(20)  # Wait for stable hover in SITL

    # ================= OFFBOARD VELOCITY CONTROL =================
    # 1. Create velocity object (NED frame: North, East, Down, YawRate)
    vel = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
    
    # 2. PX4 requires at least one setpoint BEFORE starting offboard mode
    await drone.offboard.set_velocity_ned(vel)
    
    print("🔛 Starting Offboard mode...")
    await drone.offboard.start()

    try:
        # --- Command 1: Fly North at 1.0 m/s for 3 seconds ---
        print("⬆️  Moving North (1 m/s)...")
        for _ in range(15):  # 15 * 0.2s = 3.0s
            vel = VelocityNedYaw(north_m_s=1.0, east_m_s=0.0, down_m_s=0.0, yaw_deg=0.0)
            await drone.offboard.set_velocity_ned(vel)
            await asyncio.sleep(0.2)  # 5 Hz update rate (>2Hz required by PX4)

        # --- Command 2: Hover in place ---
        print("⏸️  Hovering...")
        for _ in range(10):
            vel = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            await drone.offboard.set_velocity_ned(vel)
            await asyncio.sleep(0.2)

        # --- Command 3: Ascend at 0.5 m/s (Negative Down = Up in NED) ---
        print("🔼  Ascending (0.5 m/s)...")
        for _ in range(10):
            vel = VelocityNedYaw(0.0, 0.0, down_m_s=-0.5, yaw_deg=0.0)
            await drone.offboard.set_velocity_ned(vel)
            await asyncio.sleep(0.2)

    except OffboardError as e:
        print(f"❌ Offboard error: {e}")
    except ActionError as e:
        print(f"❌ Action error: {e}")
    finally:
        # ⚠️ CRITICAL: Always stop offboard before landing or switching modes
        print("🛑 Stopping Offboard mode...")
        await drone.offboard.stop()

        print("🛬 Landing...")
        await drone.action.land()

        async for landed in drone.telemetry.landed_state():
            if landed == telemetry.LandedState.ON_GROUND:
                print("✅ Landed successfully. Script finished.")
                break

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted. Ensure drone lands safely in SITL window.")