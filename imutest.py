import asyncio
from mavsdk import System

async def run():
    # Connect to the drone
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to drone!")
            break

    # Optional: Set the update rate for IMU data (e.g., 10 Hz)
    await drone.telemetry.set_rate_imu(10.0)

    # Subscribe to IMU data
    print("Starting IMU data stream...")
    async for imu in drone.telemetry.imu():
        accel = imu.acceleration_frd
        gyro = imu.angular_velocity_frd
        mag = imu.magnetic_field_frd

        print(f"--- IMU Data (Timestamp: {imu.timestamp_us}) ---")
        print(f"Accel (m/s^2): [X: {accel.forward_m_s2:.2f}, Y: {accel.right_m_s2:.2f}, Z: {accel.down_m_s2:.2f}]")
        print(f"Gyro (rad/s):  [X: {gyro.forward_rad_s:.2f}, Y: {gyro.right_rad_s:.2f}, Z: {gyro.down_rad_s:.2f}]")
        print(f"Mag (Gauss):   [X: {mag.forward_gauss:.2f}, Y: {mag.right_gauss:.2f}, Z: {mag.down_gauss:.2f}]")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(run())
