import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw

async def get_position_estimate(drone: System):
    """Subscribe to PX4's fused local position (NED) and print updates."""
    print("📡 Subscribing to position_velocity_ned telemetry...")
    print("   (Values reflect EKF2-fused state, including OpenVINS if configured)")
    
    try:
        async for pos_vel in drone.telemetry.position_velocity_ned():
            # pos_vel contains: north_m, east_m, down_m, velocity_*, yaw_deg
            print(f"\rPos NED | N: {pos_vel.north_m:8.3f}m | E: {pos_vel.east_m:8.3f}m | D: {pos_vel.down_m:8.3f}m | Yaw: {pos_vel.yaw_deg:7.2f}°", end="")
            
            # ⚠️ Place your offboard control logic here if needed.
            # Example: compute trajectory error, adapt setpoints, etc.
            
    except asyncio.CancelledError:
        print("\n⏹️ Telemetry subscription cancelled.")
    except Exception as e:
        print(f"\n❌ Telemetry error: {e}")

async def run():
    # 1️⃣ Initialize & Connect
    drone = System()
    # Adjust system_address as needed:
    #   SITL/QGC: "udp://:14540"
    #   Hardware: "serial:///dev/ttyACM0:921600" or "tcp://192.168.1.2:5760"
    await drone.connect(system_address="udp://:14540")

    print("⏳ Waiting for PX4 connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"✅ Connected to PX4 (UUID: {state.uuid})")
            break

    # 2️⃣ Run telemetry subscription & offboard control concurrently
    # Offboard mode requires setpoints >2Hz. This example runs telemetry 
    # in parallel with a minimal offboard position hold loop.
    async def offboard_control():
        print("\n🚀 Starting offboard position hold (example)...")
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        await drone.action.arm()
        
        # Keep sending setpoints to prevent offboard timeout
        while True:
            await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -2.0, 0.0))
            await asyncio.sleep(0.2)

    # Run both tasks concurrently
    try:
        await asyncio.gather(
            get_position_estimate(drone),
            offboard_control()
        )
    except asyncio.CancelledError:
        print("\n🛑 Shutting down...")
    finally:
        await drone.offboard.stop()
        await drone.action.disarm()

if __name__ == "__main__":
    asyncio.run(run())