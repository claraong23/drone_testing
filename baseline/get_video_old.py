import asyncio
import cv2
import numpy as np
from mavsdk import System

async def run_drone_commands(drone):
    """Example MAVSDK commands running in parallel with video."""
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Drone discovered!")
            break

    print("Arming...")
    await drone.action.arm()
    
    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(10)
    
    print("Landing...")
    await drone.action.land()

def capture_video():
    """Captures and displays the Gazebo stream via OpenCV."""
    # Standard GStreamer pipeline for PX4 SITL Gazebo
    pipeline = (
        "udpsrc port=5600 caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264\" ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! appsink"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open video stream. Ensure Gazebo is running a camera-enabled model.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("PX4 x500 Gazebo Stream", frame)
        
        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

async def main():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    # Run MAVSDK logic in the background and OpenCV in the foreground
    loop = asyncio.get_event_loop()
    
    # Run the video capture in a separate thread to keep it smooth
    video_task = loop.run_in_executor(None, capture_video)
    
    # Run drone commands
    await run_drone_commands(drone)
    
    await video_task

if __name__ == "__main__":
    asyncio.run(main())
