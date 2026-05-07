import time
import numpy as np
import cv2
# Import Gazebo transport and message bindings
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

def image_callback(msg: Image):
    """
    Callback function that triggers every time a new image message is received.
    """
    # Gazebo images are sent as raw bytes. Convert them to a NumPy array.
    # Typical Gazebo camera encoding is 'rgb_8' (24-bit RGB)
    frame = np.frombuffer(msg.data, dtype=np.uint8)
    
    # Reshape the flat array into (height, width, channels)
    # Note: msg.step is the number of bytes per row
    frame = frame.reshape((msg.height, msg.width, 3))
    
    # Convert RGB (Gazebo default) to BGR (OpenCV default) for correct colors
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Display the resulting frame
    cv2.imshow("Gazebo Live Feed", frame_bgr)
    cv2.waitKey(1)

def main():
    # 1. Initialize the Gazebo Transport Node
    node = Node()
    
    # 2. Define the topic name (use 'gz topic -l' to find yours)
    # Common PX4 camera topic is often /camera or /world/<world_name>/model/<model_name>/link/link/sensor/camera/image
    image_topic = "/world/roboverse/model/x500_mono_cam_0/link/camera_link/sensor/camera/image" 
    
    # 3. Subscribe to the topic
    if node.subscribe(Image, image_topic, image_callback):
        print(f"Subscribed to {image_topic}. Press Ctrl+C to stop.")
    else:
        print(f"Failed to subscribe to {image_topic}. Is Gazebo running?")
        return

    # 4. Keep the script alive to process incoming messages
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
