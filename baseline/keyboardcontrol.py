"""
PX4 Keyboard Controller using MAVSDK – VelocityBodyYawspeed
============================================================
Commands velocity in the drone body frame so movement keys work
regardless of which direction the drone is facing.

Body frame:
  +forward_m_s  = nose direction
  +right_m_s    = right of nose
  +down_m_s     = downward  (NED convention – negative = climb)

Controls:
  W / S     Throttle up / down   (climb / descend)
  A / D     Yaw CCW / CW
  U / J     Pitch forward / backward
  H / K     Roll left / right
  SPACE     Full stop hover
  T         Arm + Takeoff
  L         Land
  Q         Quit

Install:
  pip install mavsdk
"""

import asyncio
import sys
import os
import termios
import tty
import threading
import time
import select
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# ── Tunable parameters ──────────────────────────────────────────────────────
MAVSDK_ADDRESS   = "udp://:14540"
TAKEOFF_ALTITUDE = 2.5              # metres

SPEED_XY      = 1.0    # m/s  horizontal body velocity
SPEED_Z       = 1.0    # m/s  vertical velocity
YAW_RATE      = 30.0   # deg/s

KEY_HOLD_TIMEOUT = 0.12   # seconds – key considered released after this

# ── Shared state ──────────────────────────────────────────────────────────────
class State:
    forward_m_s : float = 0.0
    right_m_s   : float = 0.0
    down_m_s    : float = 0.0
    yaw_deg_s   : float = 0.0
    running         : bool = True
    takeoff         : bool = False
    land            : bool = False
    offboard_active : bool = False

state = State()

# ── Active key tracking ───────────────────────────────────────────────────────
_key_lock      = threading.Lock()
_active_key    = ''
_active_key_ts = 0.0

def _update_active_key(k: str):
    global _active_key, _active_key_ts
    with _key_lock:
        _active_key    = k
        _active_key_ts = time.monotonic()

def _get_active_key() -> str:
    with _key_lock:
        if _active_key and (time.monotonic() - _active_key_ts) < KEY_HOLD_TIMEOUT:
            return _active_key
        return ''

# Maps key -> (forward, right, down, yaw_deg_s)
VEL_MAP = {
    'u': ( SPEED_XY,  0.0,       0.0,      0.0     ),  # pitch forward
    'j': (-SPEED_XY,  0.0,       0.0,      0.0     ),  # pitch backward
    'h': ( 0.0,      -SPEED_XY,  0.0,      0.0     ),  # roll left
    'k': ( 0.0,       SPEED_XY,  0.0,      0.0     ),  # roll right
    'w': ( 0.0,       0.0,      -SPEED_Z,  0.0     ),  # throttle up (climb)
    's': ( 0.0,       0.0,       SPEED_Z,  0.0     ),  # throttle down (descend)
    'a': ( 0.0,       0.0,       0.0,     -YAW_RATE),  # yaw CCW
    'd': ( 0.0,       0.0,       0.0,      YAW_RATE),  # yaw CW
}

# ── Terminal helpers ──────────────────────────────────────────────────────────
class RawTerminal:
    def __enter__(self):
        self.fd  = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read_key(self, timeout=0.05) -> str:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return os.read(self.fd, 1).decode('utf-8', errors='ignore').lower()
        return ''

def out(msg: str):
    sys.stdout.write(msg)
    sys.stdout.flush()

def print_banner():
    out("\n" + "=" * 54 + "\n")
    out("  PX4 KEYBOARD CONTROLLER – VelocityBodyYawspeed\n")
    out("=" * 54 + "\n")
    out("  W / S       Climb / Descend\n")
    out("  A / D       Yaw CCW / CW\n")
    out("  U / J       Forward / Backward\n")
    out("  H / K       Left / Right\n")
    out("  SPACE       Full stop\n")
    out("  T           Arm + Takeoff\n")
    out("  L           Land\n")
    out("  Q           Quit\n")
    out("=" * 54 + "\n\n")

# ── Keyboard thread ───────────────────────────────────────────────────────────
def keyboard_thread():
    print_banner()
    with RawTerminal() as term:
        while state.running:
            key = term.read_key(timeout=0.05)
            if not key:
                continue

            if key in VEL_MAP:
                _update_active_key(key)
                fwd, rgt, dwn, yaw = VEL_MAP[key]
                state.forward_m_s += fwd
                state.right_m_s += rgt
                state.down_m_s += dwn
                out(f"\r[KEY] {key.upper()}  fwd={state.forward_m_s:+.1f} rgt={state.right_m_s:+.1f} "
                    f"dwn={state.down_m_s:+.1f} yaw={yaw:+.1f}   ")

            elif key == ' ':
                _update_active_key('')
                out("\n[KEY] SPACE -> Full stop\n")

            elif key == 't':
                state.takeoff = True
                out("\n[KEY] T -> Takeoff requested\n")

            elif key == 'l':
                state.land = True
                out("\n[KEY] L -> Land requested\n")

            elif key == 'q':
                state.running = False
                out("\n[KEY] Q -> Quit\n")
                break

# ── MAVSDK helpers ────────────────────────────────────────────────────────────
async def connect(drone: System):
    print(f"[MAVSDK] Connecting to {MAVSDK_ADDRESS} ...")
    await drone.connect(system_address=MAVSDK_ADDRESS)
    async for health in drone.telemetry.health():
        print(f"[HEALTH] GPS={health.is_global_position_ok}  "
              f"Home={health.is_home_position_ok}  "
              f"Arm={health.is_armable}")
        if health.is_global_position_ok and health.is_home_position_ok:
            break
    print("[MAVSDK] Connected and healthy.")


async def arm_and_takeoff(drone: System):
    print("[MAVSDK] Arming ...")
    await drone.action.arm()
    print(f"[MAVSDK] Taking off to {TAKEOFF_ALTITUDE} m ...")
    await drone.action.takeoff()
    print("[MAVSDK] Waiting for target altitude ...")
    async for pos in drone.telemetry.position():
        alt = pos.relative_altitude_m
        sys.stdout.write(f"\r[MAVSDK] Alt: {alt:.2f} / {TAKEOFF_ALTITUDE:.2f} m   ")
        sys.stdout.flush()
        if alt >= TAKEOFF_ALTITUDE - 0.20:
            break
    print(f"\n[MAVSDK] Reached {alt:.2f} m – takeoff complete.")


async def start_offboard(drone: System):
    # Bootstrap: send zero velocity setpoint before starting offboard
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    )
    try:
        await drone.offboard.start()
        state.offboard_active = True
        print("[MAVSDK] Offboard mode ACTIVE.")
    except OffboardError as e:
        print(f"[ERROR] Offboard start failed: {e._result.result}")
        raise


# ── Main control loop ─────────────────────────────────────────────────────────
async def control_loop(drone: System):
    print("[MAVSDK] Control loop running at 20 Hz ...")
    dt       = 0.05
    prev_key = ''

    while state.running:
        if state.takeoff:
            state.takeoff = False
            await arm_and_takeoff(drone)
            await start_offboard(drone)

        if state.land:
            state.land            = False
            state.offboard_active = False
            _update_active_key('')
            print("[MAVSDK] Landing ...")
            try:
                await drone.offboard.stop()
            except Exception:
                pass
            await drone.action.land()
            await asyncio.sleep(8)
            print("[MAVSDK] Landed.")

        if not state.offboard_active:
            await asyncio.sleep(dt)
            continue

        active = _get_active_key()
        fwd, rgt, dwn, yaw = VEL_MAP.get(active, (0.0, 0.0, 0.0, 0.0))

        if active != prev_key:
            if active:
                print(f"\n[CTL] '{active.upper()}' ACTIVE  "
                      f"fwd={state.forward_m_s:+.1f} rgt={state.right_m_s:+.1f} "
                      f"dwn={state.down_m_s:+.1f} yaw={yaw:+.1f}")
            else:
                print(f"\n[CTL] Released – hovering")
            prev_key = active

        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                forward_m_s  = state.forward_m_s ,
                right_m_s    = state.right_m_s ,
                down_m_s     = state.down_m_s,
                yawspeed_deg_s = yaw,
            )
        )

        await asyncio.sleep(dt)


async def shutdown(drone: System):
    print("[MAVSDK] Shutting down ...")
    state.offboard_active = False
    try:
        await drone.offboard.stop()
    except Exception:
        pass
    try:
        await drone.action.disarm()
    except Exception:
        pass
    print("[MAVSDK] Done.")


# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    drone = System()
    await connect(drone)

    kb = threading.Thread(target=keyboard_thread, daemon=True)
    kb.start()

    print("[INFO] Press T to arm & take off, then use keys to fly.\n")

    try:
        await control_loop(drone)
    except asyncio.CancelledError:
        pass
    finally:
        await shutdown(drone)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")