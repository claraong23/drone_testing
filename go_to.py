import asyncio
import math
from mavsdk import System, telemetry
from mavsdk.action import ActionError

# ================= HELPER FUNCTIONS =================
def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000 
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

async def explicit_system_checks(drone: System):
    print("🔍 Running explicit system checks...")

    # 1. Connection State
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"✅ Connected to drone.")
            break

    # 2. System Health
    print("⏳ Waiting for system health validation...")
    async for health in drone.telemetry.health():
        if (health.is_global_position_ok and 
            health.is_home_position_ok and 
            health.is_local_position_ok):
            print("✅ GPS, Home, and Local position are OK.")
            break

    # 3. Flight Mode Compatibility (FIXED: Use anext)
    mode = await anext(drone.telemetry.flight_mode())
    print(f"📋 Current flight mode: {mode}")
    
    # 4. Battery Check (FIXED: Use anext)
    battery = await anext(drone.telemetry.battery())
    if battery.remaining_percent < 0.30:
        raise RuntimeError(f"⚠️ Battery low ({battery.remaining_percent:.0%}). Aborting.")
    print(f"🔋 Battery: {battery.remaining_percent:.0%}")

    print("✅ All explicit checks passed.\n")

async def wait_for_arrival(drone: System, target_lat, target_lon, target_alt, 
                           pos_tolerance=1.5, alt_tolerance=1.0, timeout=60.0):
    print(f"⏳ Waiting for arrival at ({target_lat}, {target_lon})...")
    
    # Use the generator directly in the loop
    start_time = asyncio.get_event_loop().time()
    async for pos in drone.telemetry.position():
        dist_m = haversine_distance_m(pos.latitude_deg, pos.longitude_deg, target_lat, target_lon)
        alt_err = abs(pos.absolute_altitude_m - target_alt) # Usually relative is better for goto
        
        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"   📍 Dist: {dist_m:.1f}m | AltΔ: {alt_err:.1f}m | Time: {elapsed:.1f}s")

        if dist_m <= pos_tolerance and alt_err <= alt_tolerance:
            print(f"✅ Arrived!")
            return True
        
        if elapsed > timeout:
            raise TimeoutError("⏱️ Arrival timeout exceeded.")
        
        await asyncio.sleep(0.5)

# ================= MAIN FLIGHT LOGIC =================
async def run():
    drone = System()
    print("🔌 Connecting to PX4 SITL...")
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    try:
        await explicit_system_checks(drone)
        
        print("-- Arming")
        await drone.action.arm()

        print("🚀 Taking off...")
        await drone.action.set_takeoff_altitude(5.0)
        await drone.action.takeoff()
        await asyncio.sleep(8) 

        # Target coordinates
        
        TARGET_LAT, TARGET_LON, TARGET_ALT = 100, 8.545594, 8.0

        print(f"🎯 Commanding goto...")
        # Note: goto_location usually takes (lat, lon, alt_msl, yaw)
        await drone.action.goto_location(TARGET_LAT, TARGET_LON, TARGET_ALT, 0.0)

        await wait_for_arrival(drone, TARGET_LAT, TARGET_LON, TARGET_ALT)

        print("⏸️ Holding for 3 seconds...")
        await asyncio.sleep(3)

    except Exception as e:
        print(f"❌ Flight error: {e}")
    finally:
        print("🛑 Landing...")
        try:
            await drone.action.land()
            async for landed in drone.telemetry.landed_state():
                if landed == telemetry.LandedState.ON_GROUND:
                    print("✅ Landed safely.")
                    break
        except Exception as cleanup_err:
            print(f"⚠️ Cleanup warning: {cleanup_err}")

if __name__ == "__main__":
    asyncio.run(run())
