from Detector import Detector
import cv2
import time
import numpy as np
import cv2
# Import Gazebo transport and message bindings
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

class VisionApp:
    def __init__(self):
        self.detector = Detector(
            model_path="yolov10n.pt",
            confidence_threshold=0.6,
            callback=self.on_detection,
            num_workers=2,
            device="cpu",
            save_dir="./detections",
            enable_display=True,          # Spawns display thread automatically
            display_window_name="AI View"
        )

    def on_detection(self, detections, annotated_image, context):
        print(f"✅ Saved: {context['saved_path']} | Objects: {[d['class_name'] for d in detections]}")
        # UI is handled automatically by the display thread now!

    def image_callback(self,msg: Image):
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
        
        self.detector.submit_image(frame_bgr, context={"timestamp": time.time()})

    def run(self):
  # 1. Initialize the Gazebo Transport Node
        node = Node()
        
        # 2. Define the topic name (use 'gz topic -l' to find yours)
        # Common PX4 camera topic is often /camera or /world/<world_name>/model/<model_name>/link/link/sensor/camera/image
        image_topic = "/world/roboverse/model/x500_vision_0/link/camera_link/sensor/IMX214/image"
        
        # 3. Subscribe to the topic
        if node.subscribe(Image, image_topic, self.image_callback):
            print(f"Subscribed to {image_topic}. Press Ctrl+C to stop.")
        else:
            print(f"Failed to subscribe to {image_topic}. Is Gazebo running?")
            return
        try:
        
            while True:
                time.sleep(0.01)
        finally:
            self.detector.stop()

if __name__ == "__main__":
    VisionApp().run()