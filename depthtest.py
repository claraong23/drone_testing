import numpy as np
import cv2
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node

def depth_callback(msg):
    # 1. Convert raw bytes to float32 (meters)
    depth_array = np.frombuffer(msg.data, dtype=np.float32)
    depth_img = depth_array.reshape((msg.height, msg.width))

    # 2. Handle invalid data (NaN or Inf)
    # Replaces 'too far' or 'too close' errors with 0.0
    depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Scale for visibility
    # This rescales the actual meter values (0-10m) to a 0-255 grayscale range
    visible_depth = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 4. (Optional) Apply a Colormap for better intuition
    # Makes the "black" image look like a heat map (blue=far, red=near)
    color_depth = cv2.applyColorMap(visible_depth, cv2.COLORMAP_JET)

    cv2.imshow("Corrected Depth Feed", color_depth)
    cv2.waitKey(1)

def main():
    # Initialize the Gazebo Transport Node
    node = Node()
    
    # Replace with your actual depth camera topic from 'gz topic -l'
    topic = "/depth_camera"
    
    # Subscribe to the topic
    if node.subscribe(Image, topic, depth_callback):
        print(f"Subscribed to {topic}. Press Ctrl+C to stop.")
    else:
        print(f"Failed to subscribe to {topic}.")
        return

    # Keep the script running to receive callbacks
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
