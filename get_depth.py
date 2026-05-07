import numpy as np
import time
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

def depth_callback(msg: Image):
    # 1️⃣ Verify encoding (Protobuf field is 'pixel_format' in msgs10)
    fmt = msg.pixel_format
    if "FLOAT32" not in fmt and "32FC1" not in fmt:
        return

    # 2️⃣ Handle row padding & decode bytes → float32 array
    row_stride = msg.step // 4  # floats per row (includes GPU alignment padding)
    depth_flat = np.frombuffer(msg.data, dtype=np.float32, count=row_stride * msg.height)
    depth_img = depth_flat.reshape((msg.height, row_stride))
    depth_img = depth_img[:, :msg.width]  # ✅ Crop to actual image width

    # 3️⃣ Example: Find nearest valid obstacle
    valid = depth_img[depth_img > 0.05]
    if valid.size > 0:
        print(f"📏 Min depth: {valid.min():.3f}m | Frame: {msg.width}x{msg.height}")

# ─────────────────────────────────────────────────────
# 🟢 Setup & Run
node = Node()
topic = "/camera/depth/image_raw"

if not node.subscribe(topic, depth_callback, Image):
    raise RuntimeError(f"❌ Failed to subscribe to '{topic}'. Is Gazebo running?")

print(f"🟢 Listening to '{topic}'... (Ctrl+C to exit)")
try:
    while True:
        time.sleep(0.1)  # Keep main thread alive; callbacks run in C++ background threads
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    node.shutdown()