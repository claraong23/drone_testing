#!/usr/bin/env python3
"""
POTENTIAL CHANGES
- num_workers = 1 (LINE 95)
- MAX_QUEUE_SIZE = 1 OR 2 (LINE 34)
- FORWARD_SPEED_M_S = 0.8 OR 1.0 reduce it (LINE 141)
- ENABLE_DEBUG = False (LINE 28) ALREADY DONE
"""
import asyncio
import sys
import threading
import time

import cv2
import numpy as np
from mavsdk import System, telemetry
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

from depth_receiver import DepthReceiver
from Detector import Detector


MAVSDK_ADDRESS = "udpin://0.0.0.0:14540"
TAKEOFF_ALTITUDE_M = 2.1
ENABLE_DEBUG = False

# ── Vision config ────────────────────────────
RGB_TOPIC        = "/world/roboverse/model/x500_vision_0/link/camera_link/sensor/IMX214/image"
YOLO_MODEL       = "my_model.pt"
YOLO_CONFIDENCE  = 0.65
YOLO_INPUT_SIZE  = (640, 640)
MAX_QUEUE_SIZE   = 3
ANNOTATED_HOLD_S = 0.6           # how long to keep showing the annotated frame
LIVE_WINDOW_NAME = "Live Feed"

# ── Vision / display state ───────────────────
detector = None
gz_node = None
detections_log = []
_latest_annotated = None
_latest_annotated_ts = 0.0
_display_lock = threading.Lock()


def on_detection(detections, annotated_image, context):
    """Detector callback: cache annotated frame and log barrel sightings."""
    global _latest_annotated, _latest_annotated_ts
    with _display_lock:
        _latest_annotated = annotated_image
        _latest_annotated_ts = time.time()
    for d in detections:
        print(f"\n{'✅' * 10}")
        print(f"  BARREL: {d['class_name'].upper()}")
        print(f"  Conf  : {d['confidence']:.2f}")
        print(f"  Time  : {time.strftime('%H:%M:%S')}")
        detections_log.append({
            "class":      d["class_name"],
            "confidence": d["confidence"],
            "timestamp":  time.time(),
        })


def image_callback(msg: Image):
    """Gazebo RGB callback: submit to YOLO and render the live feed."""
    if detector is None:
        return

    frame = np.frombuffer(msg.data, dtype=np.uint8)
    frame = frame.reshape((msg.height, msg.width, 3))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_small = cv2.resize(frame_bgr, YOLO_INPUT_SIZE)

    if detector.queue.qsize() < MAX_QUEUE_SIZE:
        detector.submit_image(frame_small, context={"timestamp": time.time()})

    with _display_lock:
        annotated = _latest_annotated
        ts = _latest_annotated_ts
    if annotated is not None and (time.time() - ts) < ANNOTATED_HOLD_S:
        display = annotated
    else:
        display = frame_small
    cv2.imshow(LIVE_WINDOW_NAME, display)
    cv2.waitKey(1)


def start_vision():
    global detector, gz_node
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


def stop_vision():
    if detector is not None:
        detector.stop()
    cv2.destroyAllWindows()


class RightWallFollower:
    """Follows the wall on the drone's right through a corridor network."""

    # --- Front clearance (obstacle ahead) ---
    SAFE_DISTANCE_M = 2.0           # front clearance considered comfortably clear
    FRONT_BLOCKED_DISTANCE_M = 1.2  # front below this -> obstacle ahead
    CRITICAL_DISTANCE_M = 0.3       # emergency-close on every side -> stop

    # --- Right wall following ---
    RIGHT_FOLLOW_DISTANCE_M = 1.58         # standoff held from the right wall
    RIGHT_FOLLOW_GAIN_DEG_S_PER_M = 35.0  # yaw rate per metre of standoff error
    RIGHT_WALL_DISTANCE_M = 2.5           # right below this -> a wall is in range

    # --- Right opening (corridor turns / branches right) ---
    RIGHT_OPENING_DISTANCE_M = 3.5  # right beyond this -> possible opening
    RIGHT_OPENING_FRAMES = 8        # consecutive cycles before committing
                                    # (loop is ~10 Hz, so this is ~0.8 s)

    # --- Corner tracing (from try2.py) ---
    RIGHT_FOLLOW_TOLERANCE_M = 0.4  # wall "recaptured" when error within this
    CORNER_FOLLOW_MIN_S = 0.8       # trace the corner at least this long
    CORNER_FOLLOW_TIMEOUT_S = 8.0   # safety cap if the wall never reappears

    # --- Motion / timing ---
    FORWARD_SPEED_M_S = 1.2 #0.8
    YAW_RATE_DEG_S = 45.0
    CONTROL_DT_S = 0.1

    # --- Depth ROI ---
    ROI_TOP_FRACTION = 0.25
    ROI_BOTTOM_FRACTION = 0.70

    def __init__(self, depth_topic="/depth_camera"):
        self.depth_topic = depth_topic
        self.receiver = DepthReceiver(depth_topic)
        self.drone = System()
        self.running = True
        self.offboard_started = False
        self.shutdown_started = False
        self.right_wall_present = False  # latch: a wall is/was within follow range
        self.right_open_count = 0        # consecutive cycles the right read open

    # ------------------------------------------------------------------
    # Depth processing (from minimal_autonomy.py)
    # ------------------------------------------------------------------
    def _valid_depth_values(self, region):
        region = np.asarray(region, dtype=np.float32)
        return region[np.isfinite(region) & (region > 0.0)]

    def _sector_clearance(self, region):
        valid = self._valid_depth_values(region)
        if valid.size == 0:
            return 0.0
        return float(np.percentile(valid, 20))

    def compute_clearances(self, depth_frame):
        height, width = depth_frame.shape
        row_start = int(height * self.ROI_TOP_FRACTION)
        row_end = int(height * self.ROI_BOTTOM_FRACTION)
        roi = depth_frame[row_start:row_end, :]

        split = width // 3
        left = self._sector_clearance(roi[:, :split])
        center = self._sector_clearance(roi[:, split: 2 * split])
        right = self._sector_clearance(roi[:, 2 * split:])
        return left, center, right

    # ------------------------------------------------------------------
    # Heading helpers (from try2.py)
    # ------------------------------------------------------------------
    @staticmethod
    def snap_to_cardinal(yaw_deg):
        """Snap a yaw angle to the nearest of 0, 90, 180, -90."""
        cardinals = [0.0, 90.0, 180.0, -90.0]

        def angular_diff(a, b):
            d = (a - b + 180.0) % 360.0 - 180.0
            return abs(d)

        return min(cardinals, key=lambda c: angular_diff(yaw_deg, c))

    async def get_current_yaw_deg(self):
        """Read current yaw in degrees, normalised to [-180, 180]."""
        async for euler in self.drone.telemetry.attitude_euler():
            yaw = float(euler.yaw_deg)
            return ((yaw + 180.0) % 360.0) - 180.0

    async def rotate_to_yaw(self, target_yaw_deg, tolerance_deg=3.0, timeout_s=5.0):
        """Rotate in place to an absolute yaw heading."""
        loop = asyncio.get_running_loop()
        start = loop.time()

        while loop.time() - start < timeout_s:
            current = await self.get_current_yaw_deg()
            error = ((target_yaw_deg - current + 180.0) % 360.0) - 180.0

            if abs(error) <= tolerance_deg:
                await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
                return

            yaw_rate = max(-self.YAW_RATE_DEG_S, min(self.YAW_RATE_DEG_S, error * 2.0))
            await self.set_body_velocity(0.0, 0.0, 0.0, yaw_rate)
            await asyncio.sleep(self.CONTROL_DT_S)

        await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Right-opening detection (persistence counter, from try2.py "Fix 1")
    # ------------------------------------------------------------------
    def detect_right_opening(self, right):
        """
        Return True on the single cycle a real right-hand opening is confirmed.

        The right side must read open (>= RIGHT_OPENING_DISTANCE_M) for
        RIGHT_OPENING_FRAMES consecutive cycles, having been following a wall
        beforehand. The persistence count rejects brief glimpses through
        doorways or gaps that are not the actual corridor opening.
        """
        right_far = right >= self.RIGHT_OPENING_DISTANCE_M

        if right_far:
            self.right_open_count += 1
        else:
            self.right_open_count = 0
        opening_confirmed = self.right_open_count >= self.RIGHT_OPENING_FRAMES

        # An opening is a sustained transition: a wall was present on the
        # right, and now it has read open for several cycles.
        opening = opening_confirmed and self.right_wall_present

        # The "wall present" latch only clears once an opening is confirmed,
        # so it survives the cycles spent counting up.
        if right < self.RIGHT_WALL_DISTANCE_M:
            self.right_wall_present = True
        elif opening_confirmed:
            self.right_wall_present = False

        return opening

    # ------------------------------------------------------------------
    # Connection / arming / takeoff (from minimal_autonomy.py)
    # ------------------------------------------------------------------
    async def connect(self):
        print(f"Connecting to PX4 SITL on {MAVSDK_ADDRESS} ...")
        await self.drone.connect(system_address=MAVSDK_ADDRESS)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Connected")
                break

    async def wait_until_ready(self, timeout_s=30.0):
        print("Waiting for vehicle readiness...")
        loop = asyncio.get_running_loop()
        start = loop.time()

        async for health in self.drone.telemetry.health():
            armable = getattr(health, "is_armable", False)
            local_ok = getattr(health, "is_local_position_ok", False)
            global_ok = getattr(health, "is_global_position_ok", False)
            home_ok = getattr(health, "is_home_position_ok", False)

            print(
                f"Health: armable={armable}, local_ok={local_ok}, "
                f"global_ok={global_ok}, home_ok={home_ok}"
            )

            if armable or local_ok or (global_ok and home_ok):
                print("Ready for takeoff!")
                return

            if loop.time() - start > timeout_s:
                raise TimeoutError("Timed out waiting for vehicle readiness")

    async def arm_and_takeoff(self):
        await self.wait_until_ready()

        print("Arming...")
        try:
            await self.drone.action.arm()
        except ActionError as e:
            raise RuntimeError(f"Arm failed: {e}") from e

        try:
            await self.drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE_M)
        except Exception:
            pass

        print(f"Takeoff to {TAKEOFF_ALTITUDE_M:.1f} m")
        try:
            await self.drone.action.takeoff()
        except ActionError as e:
            raise RuntimeError(f"Takeoff failed: {e}") from e

        async for pos in self.drone.telemetry.position():
            alt = float(pos.relative_altitude_m)
            sys.stdout.write(
                f"\rTakeoff altitude: {alt:.2f} / {TAKEOFF_ALTITUDE_M:.2f} m   "
            )
            sys.stdout.flush()
            if alt >= TAKEOFF_ALTITUDE_M - 0.20:
                break

        print("\nTakeoff complete")
        await asyncio.sleep(2.0)

    # ------------------------------------------------------------------
    # Low-level motion (from minimal_autonomy.py)
    # ------------------------------------------------------------------
    async def set_body_velocity(self, forward_m_s, right_m_s, down_m_s, yaw_rate_deg_s):
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                forward_m_s=float(forward_m_s),
                right_m_s=float(right_m_s),
                down_m_s=float(down_m_s),
                yawspeed_deg_s=float(yaw_rate_deg_s),
            )
        )

    async def start_offboard(self):
        if self.offboard_started:
            return

        await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
        try:
            await self.drone.offboard.start()
        except OffboardError as e:
            raise RuntimeError(f"Offboard start failed: {e}") from e

        self.offboard_started = True
        print("Offboard started")

    async def hold_position(self, duration_s):
        steps = max(1, int(duration_s / self.CONTROL_DT_S))
        for _ in range(steps):
            await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
            await asyncio.sleep(self.CONTROL_DT_S)

    async def climb_to_altitude(self, target_alt_m, climb_speed=0.3):
        print(f"[ALT] Climbing to {target_alt_m}m...")
        async for pos in self.drone.telemetry.position():
            current = float(pos.relative_altitude_m)
            if current >= target_alt_m - 0.1:
                break
            await self.set_body_velocity(0.0, 0.0, -climb_speed, 0.0)
            await asyncio.sleep(0.1)
        await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
        print(f"[ALT] Reached {target_alt_m}m")

    async def yaw_in_place(self, yaw_rate_deg_s, duration_s):
        """Spin in place at a fixed yaw rate for the given duration."""
        steps = max(1, int(duration_s / self.CONTROL_DT_S))
        for _ in range(steps):
            await self.set_body_velocity(0.0, 0.0, 0.0, yaw_rate_deg_s)
            await asyncio.sleep(self.CONTROL_DT_S)
        await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Corner tracing (from try2.py)
    # ------------------------------------------------------------------
    async def follow_right_corner(self):
        """
        Trace a curved right-hand opening, then snap to the nearest cardinal
        heading so the drone exits aligned with the new corridor.

        Each tick the yaw rate is proportional to how far the right wall sits
        from RIGHT_FOLLOW_DISTANCE_M: as the wall curves away `right` grows and
        the drone yaws right to chase it, flying an arc that follows the
        corner. Ends once the wall is recaptured near the target distance.
        """
        print("Following right corner")
        loop = asyncio.get_running_loop()
        start = loop.time()

        while self.running:
            elapsed = loop.time() - start

            depth_frame = self.receiver.get_frame()
            if depth_frame is None:
                await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
                await asyncio.sleep(self.CONTROL_DT_S)
                if elapsed > self.CORNER_FOLLOW_TIMEOUT_S:
                    break
                continue

            _left, center, right = self.compute_clearances(depth_frame)

            error = right - self.RIGHT_FOLLOW_DISTANCE_M
            yaw_rate = self.RIGHT_FOLLOW_GAIN_DEG_S_PER_M * error
            yaw_rate = max(-self.YAW_RATE_DEG_S, min(self.YAW_RATE_DEG_S, yaw_rate))

            forward = self.FORWARD_SPEED_M_S
            if center < self.SAFE_DISTANCE_M:
                forward *= max(0.0, center / self.SAFE_DISTANCE_M)

            await self.set_body_velocity(forward, 0.0, 0.0, yaw_rate)

            if ENABLE_DEBUG:
                print(
                    f"  follow t={elapsed:4.1f}s R={right:.2f} "
                    f"err={error:+.2f} yaw={yaw_rate:+.1f} fwd={forward:.2f}"
                )

            wall_recaptured = abs(error) <= self.RIGHT_FOLLOW_TOLERANCE_M
            if elapsed >= self.CORNER_FOLLOW_MIN_S and wall_recaptured:
                break

            if elapsed > self.CORNER_FOLLOW_TIMEOUT_S:
                print("  corner-follow timed out")
                break

            await asyncio.sleep(self.CONTROL_DT_S)

        # Stop motion before snapping
        await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
        await asyncio.sleep(0.3)

        # Snap to nearest cardinal heading so we exit aligned
        current_yaw = await self.get_current_yaw_deg()
        target_yaw = self.snap_to_cardinal(current_yaw)
        print(f"  corner exit: yaw={current_yaw:.1f}° → snapping to {target_yaw:.1f}°")
        await self.rotate_to_yaw(target_yaw)

        # The wall has been recaptured -> re-prime the opening detector.
        self.right_wall_present = True
        self.right_open_count = 0

    # ------------------------------------------------------------------
    # Main right wall-following loop
    # ------------------------------------------------------------------
    async def task_loop(self):
        print("Right wall-following loop started")
        tick = 0
        start_time = asyncio.get_running_loop().time()
        altitude_raised = False

        while self.running:
            # After 5 minutes: stop cleanly, climb to 2.5 m, do a 360° spin,
            # then resume wall-following. Triggers exactly once.
            if not altitude_raised and (asyncio.get_running_loop().time() - start_time) >= 280:
                print("[SEQ] 5-min trigger: stop -> climb -> 360 spin")
                await self.hold_position(0.5)
                await self.climb_to_altitude(2.9)

                # record heading before spin (snapped to cardinal)
                pre_spin_yaw = await self.get_current_yaw_deg()
                target_yaw = self.snap_to_cardinal(pre_spin_yaw)

                # 20% extra time so the spin actually completes
                await self.yaw_in_place(self.YAW_RATE_DEG_S, (360.0 / self.YAW_RATE_DEG_S) * 1.2)

                # always return to where we started
                print(f"[SEQ] Returning to pre-spin heading {target_yaw:.1f}°")
                await self.rotate_to_yaw(target_yaw)
                altitude_raised = True

            depth_frame = self.receiver.get_frame()
            if depth_frame is None:
                print("HOLD | waiting for depth frame")
                await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
                await asyncio.sleep(self.CONTROL_DT_S)
                continue

            left, center, right = self.compute_clearances(depth_frame)
            opening = self.detect_right_opening(right)
            verbose = ENABLE_DEBUG and tick % 5 == 0

            front_blocked = center < self.FRONT_BLOCKED_DISTANCE_M
            right_is_open = right >= self.RIGHT_WALL_DISTANCE_M

            # Boxed in on every side -> stop and wait.
            if (
                left < self.CRITICAL_DISTANCE_M
                and center < self.CRITICAL_DISTANCE_M
                and right < self.CRITICAL_DISTANCE_M
            ):
                if verbose:
                    print(f"STOP boxed-in | L={left:.2f} C={center:.2f} R={right:.2f}")
                await self.set_body_velocity(0.0, 0.0, 0.0, 0.0)
                await asyncio.sleep(self.CONTROL_DT_S)
                tick += 1
                continue

            # Right-hand rule, top priority: a confirmed opening on the right
            # (the corridor branches right) -> trace the corner round. The
            # opening is also taken immediately when the way ahead is blocked
            # and the right is clear, where the blocked front forces it.
            if opening or (front_blocked and right_is_open):
                reason = "confirmed opening" if opening else "front blocked, right open"
                print(f"RIGHT TURN ({reason}) | R={right:.2f} -> tracing corner")
                await self.follow_right_corner()
                tick = 0
                continue

            # Way ahead blocked and no room on the right -> yaw left in place
            # so the wall stays on the drone's right (handles inside corners).
            if front_blocked:
                if verbose:
                    print(f"FRONT BLOCKED -> yaw left | C={center:.2f} R={right:.2f}")
                await self.set_body_velocity(0.0, 0.0, 0.0, -self.YAW_RATE_DEG_S)
                await asyncio.sleep(self.CONTROL_DT_S)
                tick += 1
                continue

            # Default cruise: hold a fixed standoff from the right wall with a
            # proportional yaw controller.
            if right < self.RIGHT_WALL_DISTANCE_M:
                error = right - self.RIGHT_FOLLOW_DISTANCE_M
                yaw_rate = self.RIGHT_FOLLOW_GAIN_DEG_S_PER_M * error
                yaw_rate = max(-self.YAW_RATE_DEG_S, min(self.YAW_RATE_DEG_S, yaw_rate))
                mode = "WALL FOLLOW"
            else:
                # Right is open but not a confirmed opening yet -> fly
                # straight while the opening detector counts up.
                error = 0.0
                yaw_rate = 0.0
                mode = "SEEK WALL "

            forward = self.FORWARD_SPEED_M_S
            if center < self.SAFE_DISTANCE_M:
                forward *= max(0.0, center / self.SAFE_DISTANCE_M)

            await self.set_body_velocity(forward, 0.0, 0.0, yaw_rate)

            if verbose:
                print(
                    f"{mode} | L={left:.2f} C={center:.2f} R={right:.2f} "
                    f"err={error:+.2f} yaw={yaw_rate:+.1f} fwd={forward:.2f}"
                )

            tick += 1
            await asyncio.sleep(self.CONTROL_DT_S)

    # ------------------------------------------------------------------
    # Shutdown (from minimal_autonomy.py)
    # ------------------------------------------------------------------
    async def stop_offboard(self):
        if not self.offboard_started:
            return

        print("Stopping offboard")
        try:
            await self.hold_position(0.3)
            await self.drone.offboard.stop()
        except Exception as e:
            print(f"Offboard stop skipped or failed: {e}")
        self.offboard_started = False

    async def land_and_wait(self):
        print("Landing")
        try:
            await self.drone.action.land()
        except Exception as e:
            print(f"Landing skipped or failed: {e}")
            return

        async for landed in self.drone.telemetry.landed_state():
            if landed == telemetry.LandedState.ON_GROUND:
                print("Landed")
                break

        try:
            await self.drone.action.disarm()
            print("Disarmed")
        except Exception as e:
            print(f"Disarm skipped or failed: {e}")

    async def shutdown(self):
        if self.shutdown_started:
            return
        self.shutdown_started = True

        self.running = False
        await self.stop_offboard()
        await self.land_and_wait()

    async def run(self):
        print("Starting right wall-following autonomy")
        print(f"DEBUG={'ON' if ENABLE_DEBUG else 'OFF'}")

        try:
            await self.connect()
            await self.arm_and_takeoff()
            await self.start_offboard()
            await self.task_loop()
        except asyncio.CancelledError:
            print("Wall-following autonomy cancelled")
            raise
        finally:
            await self.shutdown()

    def stop(self):
        self.running = False


async def main():
    controller = RightWallFollower()
    start_vision()
    try:
        await controller.run()
    except KeyboardInterrupt:
        controller.stop()
        await controller.shutdown()
    finally:
        stop_vision()


if __name__ == "__main__":
    asyncio.run(main())
