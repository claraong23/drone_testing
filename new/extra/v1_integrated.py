# reactive avoidance + vision integration test
# nav is simply obstacle avoidance based on depth camera
# vision runs in parallel and logs detections 

# to improve on: more efficient nav, maybe add 2 passes to check for barrels on top and below
# to improve on: add more diverse training data to yolo model 
# (currently can detect all barrels but will mistakenly detect the red ladder as barrel)

#!/usr/bin/env python3

import asyncio
import time
import sys
import cv2
import numpy as np

from depth_receiver import DepthReceiver
from drone_control import Drone
from AvoidancePlanner import AvoidancePlanner
from get_position_with_task import SharedState, position_monitor_task

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from Detector import Detector

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEPTH_TOPIC       = "/depth_camera"
LOOP_HZ           = 20.0
STEP_SIZE         = 1.5
SAFE_DISTANCE     = 4.0
CRITICAL_DISTANCE = 2.5
TEST_DURATION     = 280    # seconds to fly before landing

K = np.array([[433.0,   0.0, 320.0],
              [  0.0, 433.0, 240.0],
              [  0.0,   0.0,   1.0]])

# ── Detection config ─────────────────────────
RGB_TOPIC       = "/world/roboverse/model/x500_vision_0/link/camera_link/sensor/IMX214/image"
YOLO_MODEL      = "my_model.pt"
YOLO_CONFIDENCE = 0.65
YOLO_INPUT_SIZE = (640, 640)
MAX_QUEUE_SIZE  = 3

detector       = None
gz_node        = None
detections_log = []


def on_detection(detections, annotated_image, context):
    for d in detections:
        print(f"\n{'✅'*10}")
        print(f"  BARREL: {d['class_name'].upper()}")
        print(f"  Conf  : {d['confidence']:.2f}")
        print(f"  Time  : {time.strftime('%H:%M:%S')}")
        detections_log.append({
            "class":      d["class_name"],
            "confidence": d["confidence"],
            "timestamp":  time.time(),
        })


def image_callback(msg: Image):
    if detector is None:
        return
    if detector.queue.qsize() >= MAX_QUEUE_SIZE:
        return

    frame = np.frombuffer(msg.data, dtype=np.uint8)
    frame = frame.reshape((msg.height, msg.width, 3))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_small = cv2.resize(frame_bgr, YOLO_INPUT_SIZE)

    detector.submit_image(frame_small, context={"timestamp": time.time()})


def print_detection_summary():
    print(f"\n{'='*40}")
    print(f"  DETECTION SUMMARY")
    print(f"{'='*40}")
    print(f"  Total: {len(detections_log)}")
    yellow = [d for d in detections_log if "yellow" in d["class"].lower()]
    red    = [d for d in detections_log if "red"    in d["class"].lower()]
    print(f"  Yellow barrels: {len(yellow)}")
    print(f"  Red barrels   : {len(red)}")
    for i, d in enumerate(detections_log):
        print(f"  [{i+1}] {d['class']:15s} conf={d['confidence']:.2f}")
    print(f"{'='*40}\n")


async def run():
    global detector, gz_node

    # ── Setup ─────────────────────────────────
    drone    = Drone()
    receiver = DepthReceiver(DEPTH_TOPIC)
    planner  = AvoidancePlanner(
        K=K, width=640, height=480,
        safe_distance=SAFE_DISTANCE,
        critical_distance=CRITICAL_DISTANCE,
    )
    state      = SharedState()
    stop_event = asyncio.Event()

    # ── Detector (init after asyncio loop is running) ──
    detector = Detector(
        model_path=YOLO_MODEL,
        confidence_threshold=YOLO_CONFIDENCE,
        callback=on_detection,
        num_workers=2,
        device="cpu",
        save_dir="./detections",
        enable_display=False,
    )
    print(f"[DETECT] Model loaded: {YOLO_MODEL}")

    gz_node = Node()
    if gz_node.subscribe(Image, RGB_TOPIC, image_callback):
        print(f"[VISION] Subscribed to {RGB_TOPIC}")
    else:
        print(f"[VISION] WARNING: Could not subscribe to {RGB_TOPIC}")

    pose = {"north": 0.0, "east": 0.0, "down": -1.5,
            "yaw": 0.0, "yaw_deg": 0.0}

    grid_headings       = [0, 90, 180, -90]
    current_heading_idx = 0
    target_yaw_deg      = 0

    # ── Connect + Takeoff ─────────────────────
    print("[INIT] Connecting...")
    await drone.connect()
    await asyncio.sleep(3)

    print("[INIT] Starting position monitor...")
    monitor_task = asyncio.create_task(
        position_monitor_task(drone, state, stop_event)
    )

    print("[INIT] Arming and taking off...")
    await drone.arm_and_takeoff()
    await drone.rotate_to_yaw(target_yaw_deg)

    print(f"[INIT] Flying for {TEST_DURATION}s. Watch the drone in Gazebo.")

    start_time = time.time()

    # ── Main Loop ─────────────────────────────
    try:
        while True:
            t_start = time.monotonic()
            elapsed = time.time() - start_time

            # Time limit
            if elapsed > TEST_DURATION:
                print("[TEST] Time up. Landing.")
                break

            # Update pose from shared state
            if state.latest_position is not None:
                pose["north"]   = state.latest_position.north_m
                pose["east"]    = state.latest_position.east_m
                pose["down"]    = state.latest_position.down_m
                pose["yaw_deg"] = state.latest_yaw or 0.0
                pose["yaw"]     = np.deg2rad(pose["yaw_deg"])

            # Get depth frame
            depth_frame = receiver.get_frame()
            if depth_frame is None:
                print("[WARN] No depth frame")
                await asyncio.sleep(0.1)
                continue

            # Avoidance planner
            north, east, _, info = planner.compute_position_ned(
                depth_frame, pose, step_size=STEP_SIZE
            )

            c = info["clearance"]
            print(f"[{elapsed:.0f}s] "
                  f"Blocked={info['blocked']} | "
                  f"L={c['left']:.1f} C={c['center']:.1f} R={c['right']:.1f} | "
                  f"N={pose['north']:.1f} E={pose['east']:.1f}")

            # Movement
            if info["blocked"]:
                await drone.send_velocity(0, 0, 0, target_yaw_deg)
                current_heading_idx = (current_heading_idx + 1) % 4
                target_yaw_deg = grid_headings[current_heading_idx]
                print(f"[BLOCKED] Rotating to {target_yaw_deg}°")
                await drone.rotate_to_yaw(target_yaw_deg)
            else:
                await drone.send_position_setpoint(
                    north=north,
                    east=east,
                    down=-1.5,
                    yaw_deg=target_yaw_deg,
                )

            # Loop timing
            elapsed_loop = time.monotonic() - t_start
            sleep_time = (1.0 / LOOP_HZ) - elapsed_loop
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        print("[TEST] Cancelled")

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("[SHUTDOWN] Stopping...")
        await drone.send_velocity(0, 0, 0, target_yaw_deg)
        await asyncio.sleep(1)
        stop_event.set()
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        if detector is not None:
            detector.stop()
        await drone.land()
        print_detection_summary()
        print("[SHUTDOWN] Done.")


if __name__ == "__main__":
    asyncio.run(run())

    