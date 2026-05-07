import time
import numpy as np
import cv2
import os
# Import Gazebo transport and message bindings
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

# Global counter to give each file a unique name
image_count = 0
# Directory to save images
save_dir = "captured_images"

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def image_callback(msg: Image):
    """
    Callback function that triggers every time a new image message is received.
    """
    global image_count
    
    # 1. Convert Gazebo raw bytes to a NumPy array
    frame = np.frombuffer(msg.data, dtype=np.uint8)
    
    # 2. Reshape into (height, width, 3)
    frame = frame.reshape((msg.height, msg.width, 3))
    
    # 3. Convert RGB (Gazebo) to BGR (OpenCV)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 4. Save to file
    # Filename format: captured_images/frame_0001.jpg
    filename = os.path.join(save_dir, f"frame_{image_count:04d}.jpg")
    cv2.imwrite(filename, frame_bgr)
    
    print(f"Saved: {filename}")
    image_count += 1
    
    # 5. Display the resulting frame
    cv2.imshow("Gazebo Live Feed", frame_bgr)
    cv2.waitKey(1)

def main():
    # 1. Initialize the Gazebo Transport Node
    node = Node()
    
    # 2. Define the topic name
    image_topic = "/world/roboverse/model/x500_vision_0/link/camera_link/sensor/IMX214/image" 
    
    # 3. Subscribe to the topic
    if node.subscribe(Image, image_topic, image_callback):
        print(f"Subscribed to {image_topic}. Images will be saved to '{save_dir}/'.")
        print("Press Ctrl+C to stop.")
    else:
        print(f"Failed to subscribe to {image_topic}. Is Gazebo running?")
        return

    # 4. Keep the script alive
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()