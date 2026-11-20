#!/usr/bin/env python3
"""
Arm mimicry (vision teleoperation) for the xArm7 - SEPARATE from the
gesture pick-and-place demo. Nothing here is used by gesture_pick_and_place.py.

A webcam watches YOU (not the table). MediaPipe Pose tracks your shoulder and
wrist in 3D; the arm's gripper copies how your wrist moves relative to your
shoulder. Optional (--gripper): make a FIST to close the gripper, OPEN your
hand to open it (or --grip pinch for a thumb-index pinch, close range only).
Optional (--twist): rotate your hand like turning a dial and the gripper spins
the same way (joint 7), relative to how your hand was when you engaged.

HOW IT MOVES (clutch / relative mapping)
  Press SPACE to engage. At that instant the script remembers where your wrist
  is and where the gripper is. From then on the gripper moves by the SAME
  offset your wrist moves (times SCALE). So engaging never makes the arm jump,
  and you can re-centre yourself by disengaging, moving, and engaging again -
  like lifting a mouse off the desk.

SAFETY LAYERS (all on by default)
  1. DRY RUN is the default: no robot connection, targets are only drawn and
     printed. Pass --live --ip <ip> to move the real arm.
  2. Every target is clamped to the WORKSPACE box below (well above the table).
  3. Each command may move at most MAX_STEP_MM from the previous one, and the
     controller is told a speed cap (ROBOT_SPEED_MM_S).
  4. Tracking is smoothed with a One Euro filter, so jitter does not become motion.
  5. Deadman: if your shoulder/wrist are not tracked well for LOST_TIMEOUT_S,
     the script DISENGAGES and stops sending targets. You must press SPACE again.
  6. 'h' = software halt (stops the current motion and disengages).
  None of this replaces the physical E-stop - keep someone on it.

BEFORE THE FIRST LIVE RUN
  - Stop the xarm_ros2 launch files. The ROS driver and this SDK script both talk
    to the controller and the driver can change the mode underneath you.
  - Check the WORKSPACE limits with your own reachability tests
    (reachability_sweep.py etc.) and get Dr. Madadi's OK for this mode.
  - Run in dry-run first and check that the AXIS_MAP signs feel right: move your
    hand right/up/forward and watch which way the target moves.
  - Online trajectory mode (mode 7) needs a reasonably recent controller
    firmware. The script checks and refuses to run if mode 7 is not accepted.

Requirements (same mediapipe pin as hand_gestures.py - needs mp.solutions):
    pip install --user "mediapipe==0.10.21" "numpy<2" opencv-python xarm-python-sdk

Usage:
    python3 arm_mimic.py                          # dry run, camera 1
    python3 arm_mimic.py --camera 0 --arm left    # dry run, other camera / arm
    python3 arm_mimic.py --live --ip 192.168.1.210 --gripper --log run1.csv
    python3 arm_mimic.py --camera 0 --gripper --twist  # + wrist twist -> gripper spin

Keys (video window focused):
    SPACE  engage / disengage
    d      toggle depth (robot X) on/off - off by default, single cameras judge depth poorly
    h      halt: stop the arm now and disengage
    q/ESC  quit (the arm stops where it is)
"""
import argparse
import csv
import math
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration - edit these for your setup
# ---------------------------------------------------------------------------

# Safe box the gripper tip may move in, robot base frame, mm.
# Floor stays above TRAVERSE_Z (153.5 mm) so it can never reach the table.
# In --live mode every corner and face centre of this box is checked with the
# controller's inverse kinematics before anything moves. If some are out of
# reach with the gripper pointing down, the box is shrunk 25 mm at a time
# (top and far side first) until every check point passes.
WORKSPACE = {
    "x": (250.0, 550.0),    # 300 mm forward/back   (was 300-500)
    "y": (-250.0, 250.0),   # 500 mm side to side   (was -200 to 200)
    "z": (170.0, 450.0),    # 280 mm up/down        (was 200-400)
}

# Gripper pointing straight down - same orientation the pick-and-place code
# uses (quaternion x=1 -> roll 180 deg).
TOOL_RPY_DEG = (180.0, 0.0, 0.0)

# Wrist twist -> gripper yaw (only with --twist). With the gripper pointing
# straight down, yaw is a rotation about the vertical, which is almost all
# joint 7. Twist is read from the line across your knuckles (index to pinky)
# as the camera sees it - keep your palm toward the camera.
TWIST_SIGN = 1.0            # flip to -1.0 if the gripper turns the opposite way to your hand
TWIST_LIMIT_DEG = 90.0      # gripper yaw stays within this of the start yaw
TWIST_MAX_STEP_DEG = 6.0    # max yaw change per command (x SEND_HZ = deg/s cap)
TWIST_DEADBAND_DEG = 4.0    # twists smaller than this are ignored (jitter)
TWIST_FILTER_BETA = 0.01    # One Euro beta for the angle (degrees, so much smaller)

# Robot mm moved per mm your wrist moves. 1.0 = one-to-one.
SCALE = 1.0

# Which human axis drives which robot axis, and the sign.
# Human axes are MediaPipe pose WORLD landmarks (metres, camera-aligned):
#   'x' = to the right in the image, 'y' = DOWN, 'z' = away from the camera.
# Robot axes: X forward from the base, Y to the robot's left, Z up.
# Default assumes the camera faces you and you face the same way as the robot
# (standing behind/beside it). Check in dry-run and flip signs as needed.
AXIS_MAP = {
    "x": ("z", -1.0),  # hand toward camera  -> robot forward
    "y": ("x", -1.0),  # hand to image-right -> robot to its right (-Y)
    "z": ("y", -1.0),  # hand up             -> robot up
}
DEPTH_AXIS = "x"            # robot axis driven by camera depth (locked by default)

SEND_HZ = 20.0              # target updates sent to the controller per second
MAX_STEP_MM = 12.0          # max distance between consecutive targets
ROBOT_SPEED_MM_S = 150.0    # speed cap passed to the controller
ROBOT_ACC_MM_S2 = 1000.0

MIN_VISIBILITY = 0.5        # pose landmark visibility needed to count as tracked
LOST_TIMEOUT_S = 0.6        # tracking lost this long -> disengage (no targets are
                            # sent while lost, so the arm holds still meanwhile)

# One Euro filter (smoothing). Lower MIN_CUTOFF = smoother but laggier;
# higher BETA = less lag on fast moves.
FILTER_MIN_CUTOFF = 1.0
FILTER_BETA = 0.4

# Gripper control from your hand (only with --gripper).
# Default "fist" mode: count extended fingers (index..pinky).
#   <= FIST_MAX_FINGERS extended -> gripper CLOSES
#   >= OPEN_MIN_FINGERS extended -> gripper OPENS
#   in between -> no change. A new state must hold GRIP_CONFIRM_FRAMES frames.
FINGER_EXTENDED_RATIO = 1.15  # tip-to-wrist / PIP-to-wrist above this = extended
FIST_MAX_FINGERS = 1
OPEN_MIN_FINGERS = 3
GRIP_CONFIRM_FRAMES = 4

# Capture resolution - higher makes your hand bigger in pixels, which the
# hand model needs when you stand back from the laptop.
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1280, 720

# Optional "pinch" mode (--grip pinch): thumb tip to index tip, divided by
# palm size, with hysteresis. Only reliable close to the camera.
PINCH_CLOSE_RATIO = 0.35
PINCH_OPEN_RATIO = 0.55
GRIPPER_OPEN_POS = 850      # xArm SDK gripper units: 850 = fully open
GRIPPER_CLOSED_POS = 300    # not fully closed - tune for what you grab
GRIPPER_SPEED = 2000

START_SPEED_MM_S = 50.0     # slow move to the start pose before mimicry
DRY_RUN_START = (400.0, 0.0, 300.0)

# MediaPipe Pose landmark indices
POSE_IDS = {
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16},
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15},
}
# MediaPipe Hands landmark indices
HAND_WRIST, HAND_THUMB_TIP, HAND_INDEX_TIP, HAND_MIDDLE_MCP = 0, 4, 8, 9
HAND_INDEX_MCP, HAND_PINKY_MCP = 5, 17
# (tip, PIP joint) for index, middle, ring, pinky
FINGER_TIP_PIP = ((8, 6), (12, 10), (16, 14), (20, 18))


# ---------------------------------------------------------------------------
# Pure helpers (no camera / robot) - covered by test_arm_mimic.py
# ---------------------------------------------------------------------------

def workspace_center(ws=WORKSPACE):
    return np.array([sum(ws[a]) / 2.0 for a in ("x", "y", "z")])


def workspace_check_points(ws=WORKSPACE):
    """8 corners + 6 face centres + centre of the box."""
    xs, ys, zs = ws["x"], ws["y"], ws["z"]
    mid = workspace_center(ws)
    pts = [(x, y, z) for x in xs for y in ys for z in zs]
    pts += [(xs[0], mid[1], mid[2]), (xs[1], mid[1], mid[2]),
            (mid[0], ys[0], mid[2]), (mid[0], ys[1], mid[2]),
            (mid[0], mid[1], zs[0]), (mid[0], mid[1], zs[1]), tuple(mid)]
    return [np.array(p, dtype=float) for p in pts]


def in_workspace(p, ws=WORKSPACE):
    return all(ws[a][0] <= p[i] <= ws[a][1] for i, a in enumerate(("x", "y", "z")))


def clamp_to_workspace(p, ws=WORKSPACE):
    """Return (clamped point, True if any axis had to be clamped)."""
    out = np.array(p, dtype=float)
    clamped = False
    for i, a in enumerate(("x", "y", "z")):
        lo, hi = ws[a]
        if out[i] < lo or out[i] > hi:
            out[i] = min(max(out[i], lo), hi)
            clamped = True
    return out, clamped


def limit_step(prev, target, max_step=MAX_STEP_MM):
    """Move from prev toward target by at most max_step mm."""
    prev = np.asarray(prev, dtype=float)
    target = np.asarray(target, dtype=float)
    d = target - prev
    n = float(np.linalg.norm(d))
    if n <= max_step or n == 0.0:
        return target
    return prev + d * (max_step / n)


def human_delta_to_robot_mm(delta_m, axis_map=AXIS_MAP, locked=(), scale=SCALE):
    """Map a human displacement (metres, keys 'x','y','z') to robot mm [X,Y,Z]."""
    out = np.zeros(3)
    for i, robot_axis in enumerate(("x", "y", "z")):
        if robot_axis in locked:
            continue
        human_axis, sign = axis_map[robot_axis]
        out[i] = sign * delta_m[human_axis] * 1000.0 * scale
    return out


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012) for a vector signal."""

    def __init__(self, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class PinchDetector:
    """Open/closed with hysteresis from a pinch ratio."""

    def __init__(self, close_ratio=PINCH_CLOSE_RATIO, open_ratio=PINCH_OPEN_RATIO):
        assert close_ratio < open_ratio
        self.close_ratio, self.open_ratio = close_ratio, open_ratio
        self.closed = False

    def update(self, ratio):
        if ratio is None:
            return self.closed  # no hand seen: keep the last state
        if not self.closed and ratio < self.close_ratio:
            self.closed = True
        elif self.closed and ratio > self.open_ratio:
            self.closed = False
        return self.closed


def wrap_deg(a):
    """Wrap an angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def hand_twist_deg(hand_landmarks, aspect=16 / 9):
    """In-image angle of the knuckle line (index MCP -> pinky MCP), degrees.
    `aspect` = frame width / height, since landmark x and y are normalised
    separately."""
    lm = hand_landmarks.landmark
    dx = (lm[HAND_PINKY_MCP].x - lm[HAND_INDEX_MCP].x) * aspect
    dy = lm[HAND_PINKY_MCP].y - lm[HAND_INDEX_MCP].y
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


class AngleUnwrapper:
    """Turns a wrapped angle stream into a continuous one (no jump at +-180)."""

    def __init__(self):
        self.prev = None

    def __call__(self, a):
        self.prev = a if self.prev is None else self.prev + wrap_deg(a - self.prev)
        return self.prev


def twist_to_yaw(twist_delta_deg, yaw_at_engage, sign=TWIST_SIGN,
                 deadband=TWIST_DEADBAND_DEG, limit=TWIST_LIMIT_DEG,
                 base_yaw=TOOL_RPY_DEG[2]):
    """Gripper yaw for a hand twist since engaging. The result is kept within
    +-limit of base_yaw (the start orientation), however often you re-engage."""
    d = sign * twist_delta_deg
    d = 0.0 if abs(d) < deadband else d - math.copysign(deadband, d)
    return min(max(yaw_at_engage + d, base_yaw - limit), base_yaw + limit)


def limit_angle_step(prev, target, max_step=TWIST_MAX_STEP_DEG):
    return prev + max(-max_step, min(max_step, target - prev))


def count_extended_fingers(hand_landmarks, ratio=FINGER_EXTENDED_RATIO):
    """Fingers (index..pinky) whose tip is clearly farther from the wrist than
    their middle joint. Uses image x/y only - MediaPipe's z is noisy."""
    lm = hand_landmarks.landmark
    w = (lm[HAND_WRIST].x, lm[HAND_WRIST].y)
    n = 0
    for tip, pip in FINGER_TIP_PIP:
        d_tip = math.dist(w, (lm[tip].x, lm[tip].y))
        d_pip = math.dist(w, (lm[pip].x, lm[pip].y))
        if d_pip > 1e-6 and d_tip / d_pip > ratio:
            n += 1
    return n


class FistDetector:
    """Open hand / fist with a dead band and a few frames of confirmation."""

    def __init__(self, fist_max=FIST_MAX_FINGERS, open_min=OPEN_MIN_FINGERS,
                 confirm=GRIP_CONFIRM_FRAMES):
        assert fist_max < open_min
        self.fist_max, self.open_min, self.confirm = fist_max, open_min, confirm
        self.closed = False
        self._candidate = None
        self._count = 0

    def update(self, fingers):
        if fingers is None:
            self._candidate, self._count = None, 0
            return self.closed  # no hand seen: keep the last state
        if fingers <= self.fist_max:
            want = True
        elif fingers >= self.open_min:
            want = False
        else:
            want = None  # ambiguous: do nothing
        if want is None or want == self.closed:
            self._candidate, self._count = None, 0
            return self.closed
        if want == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = want, 1
        if self._count >= self.confirm:
            self.closed = want
            self._candidate, self._count = None, 0
        return self.closed


def pinch_ratio(hand_landmarks):
    """Thumb-index tip distance divided by palm length (wrist to middle MCP)."""
    lm = hand_landmarks.landmark

    def dist(a, b):
        return math.dist((lm[a].x, lm[a].y, lm[a].z), (lm[b].x, lm[b].y, lm[b].z))

    palm = dist(HAND_WRIST, HAND_MIDDLE_MCP)
    if palm < 1e-6:
        return None
    return dist(HAND_THUMB_TIP, HAND_INDEX_TIP) / palm


# ---------------------------------------------------------------------------
# Robot back-ends
# ---------------------------------------------------------------------------

class DryRunRobot:
    """Pretends to be the arm. Nothing is sent anywhere."""

    live = False

    def __init__(self):
        self.pos = np.array(DRY_RUN_START, dtype=float)
        self.yaw = TOOL_RPY_DEG[2]
        self.gripper_closed = False

    def current_xyz(self):
        return self.pos.copy()

    def current_yaw(self):
        return self.yaw

    def move_to(self, xyz, yaw=TOOL_RPY_DEG[2]):
        self.pos = np.array(xyz, dtype=float)
        self.yaw = yaw
        return 0

    def set_gripper(self, closed):
        self.gripper_closed = closed

    def error(self):
        return None

    def halt(self):
        pass

    def close(self):
        pass


class XArmRobot:
    """xArm Python SDK, online trajectory planning mode (mode 7)."""

    live = True

    def __init__(self, ip, use_gripper=False, fence=False, twist=False):
        try:
            from xarm.wrapper import XArmAPI
        except ImportError:
            sys.exit("xarm-python-sdk not installed: pip install --user xarm-python-sdk")

        self.use_gripper = use_gripper
        self.fence = fence
        # IK check yaws: with --twist, every check point must also be reachable
        # with the gripper turned fully either way.
        base = TOOL_RPY_DEG[2]
        self.check_yaws = ((base - TWIST_LIMIT_DEG, base, base + TWIST_LIMIT_DEG)
                           if twist else (base,))
        self.arm = XArmAPI(ip, is_radian=False)
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        time.sleep(0.5)
        if self.arm.error_code != 0:
            code = self.arm.error_code
            self.arm.disconnect()
            sys.exit(f"Arm still reports controller error {code} after clearing. "
                     f"Code 1 usually means the emergency stop is pressed - release it "
                     f"(twist to pop it out), or check UFactory Studio, then retry.")

        self._check_workspace_reachable()
        self._move_to_start()

        self.arm.set_mode(7)
        self.arm.set_state(0)
        time.sleep(0.5)
        if self.arm.mode != 7:
            self.close()
            sys.exit(f"Controller did not accept mode 7 (online trajectory planning); "
                     f"mode is {self.arm.mode}. Update the firmware in UFactory Studio, "
                     f"or adapt this script to servo mode.")

        if fence:
            # Controller-side box as a second layer behind the software clamp.
            # SDK order: [x_max, x_min, y_max, y_min, z_max, z_min]
            ws = WORKSPACE
            self.arm.set_reduced_tcp_boundary([ws["x"][1], ws["x"][0], ws["y"][1],
                                               ws["y"][0], ws["z"][1], ws["z"][0]])
            setter = getattr(self.arm, "set_fence_mode", None) or getattr(self.arm, "set_fense_mode")
            setter(True)
            print("Controller safety fence enabled for the workspace box.")

        if use_gripper:
            self.arm.set_gripper_mode(0)
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_speed(GRIPPER_SPEED)
            self.arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
        self.gripper_closed = False

    def _unreachable_points(self):
        bad = []
        for p in workspace_check_points():
            for yaw in self.check_yaws:
                code, _ = self.arm.get_inverse_kinematics([*p, TOOL_RPY_DEG[0], TOOL_RPY_DEG[1], yaw],
                                                          input_is_radian=False,
                                                          return_is_radian=False)
                if code != 0:
                    bad.append(p)
                    break
        return bad

    def _check_workspace_reachable(self, step=25.0, max_rounds=12):
        """Ask the controller's IK (no motion) whether the whole box is reachable
        with the gripper pointing down; shrink the top and far side until it is."""
        print("Checking the WORKSPACE box with the controller's inverse kinematics (no motion)...")
        for _ in range(max_rounds + 1):
            bad = self._unreachable_points()
            if not bad:
                ws = WORKSPACE
                print(f"  All {len(workspace_check_points())} check points reachable. Using "
                      f"X {ws['x'][0]:.0f}..{ws['x'][1]:.0f}, Y {ws['y'][0]:.0f}..{ws['y'][1]:.0f}, "
                      f"Z {ws['z'][0]:.0f}..{ws['z'][1]:.0f} mm.")
                return
            for p in bad:
                print(f"  out of reach: X={p[0]:.0f} Y={p[1]:.0f} Z={p[2]:.0f}")
            # Shrink whichever limits the failing points sit on. WORKSPACE is
            # edited in place, so every helper sees the new box.
            shrunk = False
            for axis, i in (("z", 2), ("x", 0), ("y", 1)):
                lo, hi = WORKSPACE[axis]
                if any(abs(p[i] - hi) < 1e-6 for p in bad) and hi - step > lo + 100:
                    WORKSPACE[axis] = (lo, hi - step)
                    shrunk = True
                if axis == "y" and any(abs(p[i] - lo) < 1e-6 for p in bad) and lo + step < hi - 100:
                    WORKSPACE[axis] = (WORKSPACE[axis][0] + step, WORKSPACE[axis][1])
                    shrunk = True
            if not shrunk:
                break
            print(f"  shrinking -> X max {WORKSPACE['x'][1]:.0f}, "
                  f"Y {WORKSPACE['y'][0]:.0f}..{WORKSPACE['y'][1]:.0f}, Z max {WORKSPACE['z'][1]:.0f}")
        self.arm.disconnect()
        sys.exit("Could not find a reachable box automatically. Set WORKSPACE by hand and retry.")

    def _move_to_start(self):
        code, pos = self.arm.get_position()
        if code != 0:
            sys.exit(f"Could not read arm position (code {code}).")
        start = workspace_center()
        print(f"\nArm is at X={pos[0]:.1f} Y={pos[1]:.1f} Z={pos[2]:.1f} "
              f"R={pos[3]:.1f} P={pos[4]:.1f} Yaw={pos[5]:.1f}")
        print(f"Start pose = centre of the workspace box, gripper down: "
              f"X={start[0]:.1f} Y={start[1]:.1f} Z={start[2]:.1f}")
        ans = input(f"Move there now at {START_SPEED_MM_S:.0f} mm/s? Clear the area. [y/N] ").strip().lower()
        if ans != "y":
            self.arm.disconnect()
            sys.exit("Not moving. Exiting.")
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.5)
        code = self.arm.set_position(*start, *TOOL_RPY_DEG, speed=START_SPEED_MM_S,
                                     mvacc=500, wait=True)
        if code != 0:
            self.close()
            sys.exit(f"Move to start failed (code {code}).")

    def current_xyz(self):
        code, pos = self.arm.get_position()
        if code != 0:
            raise RuntimeError(f"get_position failed (code {code})")
        return np.array(pos[:3], dtype=float)

    def current_yaw(self):
        code, pos = self.arm.get_position()
        if code != 0:
            raise RuntimeError(f"get_position failed (code {code})")
        return float(pos[5])

    def move_to(self, xyz, yaw=TOOL_RPY_DEG[2]):
        return self.arm.set_position(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                                     roll=TOOL_RPY_DEG[0], pitch=TOOL_RPY_DEG[1],
                                     yaw=float(yaw), speed=ROBOT_SPEED_MM_S,
                                     mvacc=ROBOT_ACC_MM_S2, wait=False)

    def set_gripper(self, closed):
        if not self.use_gripper or closed == self.gripper_closed:
            return
        self.arm.set_gripper_position(GRIPPER_CLOSED_POS if closed else GRIPPER_OPEN_POS, wait=False)
        self.gripper_closed = closed

    def error(self):
        if self.arm.error_code != 0:
            return f"controller error {self.arm.error_code}"
        if not self.arm.connected:
            return "lost connection to the controller"
        return None

    def halt(self):
        # State 4 stops the current motion; then re-arm mode 7 so we can engage again.
        self.arm.set_state(4)
        time.sleep(0.2)
        self.arm.set_mode(7)
        self.arm.set_state(0)

    def close(self):
        try:
            if self.fence:
                setter = getattr(self.arm, "set_fence_mode", None) or getattr(self.arm, "set_fense_mode")
                setter(False)
            self.arm.set_state(4)
            time.sleep(0.2)
            self.arm.set_mode(0)
            self.arm.set_state(0)
        finally:
            self.arm.disconnect()


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

class ArmTracker:
    def __init__(self, side, use_hands):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            sys.exit('This mediapipe build has no mp.solutions. Install the same pin as '
                     'hand_gestures.py: pip install --user "mediapipe==0.10.21" "numpy<2"')
        self.mp = mp
        self.ids = POSE_IDS[side]
        self.pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True,
                                           min_detection_confidence=0.6,
                                           min_tracking_confidence=0.6)
        self.hands = (mp.solutions.hands.Hands(max_num_hands=2, model_complexity=1,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
                      if use_hands else None)
        self.draw = mp.solutions.drawing_utils

    def process(self, frame_bgr):
        """Returns dict with 'delta' (wrist - shoulder, metres, or None),
        'pinch' (ratio or None), 'fingers' (count or None) and raw results for drawing."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        pose_res = self.pose.process(rgb)
        out = {"delta": None, "pinch": None, "fingers": None, "twist": None,
               "pose": pose_res, "hand": None}

        if pose_res.pose_world_landmarks and pose_res.pose_landmarks:
            w = pose_res.pose_world_landmarks.landmark
            img = pose_res.pose_landmarks.landmark
            sh, wr = self.ids["shoulder"], self.ids["wrist"]
            if min(img[sh].visibility, img[wr].visibility) >= MIN_VISIBILITY:
                out["delta"] = {
                    "x": w[wr].x - w[sh].x,
                    "y": w[wr].y - w[sh].y,
                    "z": w[wr].z - w[sh].z,
                }

        if self.hands is not None and pose_res.pose_landmarks:
            hand_res = self.hands.process(rgb)
            if hand_res.multi_hand_landmarks:
                # Use the detected hand closest to the tracked arm's wrist.
                pw = pose_res.pose_landmarks.landmark[self.ids["wrist"]]
                best = min(hand_res.multi_hand_landmarks,
                           key=lambda h: (h.landmark[0].x - pw.x) ** 2 + (h.landmark[0].y - pw.y) ** 2)
                out["hand"] = best
                out["pinch"] = pinch_ratio(best)
                out["fingers"] = count_extended_fingers(best)
                h, w = frame_bgr.shape[:2]
                out["twist"] = hand_twist_deg(best, aspect=w / h)
        return out

    def draw_overlay(self, frame, result):
        if result["pose"].pose_landmarks:
            self.draw.draw_landmarks(frame, result["pose"].pose_landmarks,
                                     self.mp.solutions.pose.POSE_CONNECTIONS)
        if result["hand"] is not None:
            self.draw.draw_landmarks(frame, result["hand"],
                                     self.mp.solutions.hands.HAND_CONNECTIONS)

    def close(self):
        self.pose.close()
        if self.hands is not None:
            self.hands.close()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def put(frame, text, row, color=(255, 255, 255)):
    cv2.putText(frame, text, (10, 28 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 28 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="xArm7 arm mimicry (dry run by default)")
    ap.add_argument("--camera", type=int, default=1, help="camera facing YOU (default 1)")
    ap.add_argument("--arm", choices=("right", "left"), default="right", help="which of your arms to copy")
    ap.add_argument("--live", action="store_true", help="actually move the robot")
    ap.add_argument("--ip", default=None, help="xArm controller IP (required with --live)")
    ap.add_argument("--gripper", action="store_true", help="control the gripper with your hand")
    ap.add_argument("--grip", choices=("fist", "pinch"), default="fist",
                    help="fist = fist closes / open hand opens (default); pinch = thumb+index pinch")
    ap.add_argument("--twist", action="store_true",
                    help="rotate your hand (palm to camera) to spin the gripper (joint 7)")
    ap.add_argument("--fence", action="store_true",
                    help="also enable the controller's own TCP boundary fence (live only)")
    ap.add_argument("--log", default=None, help="write a CSV of tracking and targets")
    args = ap.parse_args()

    if args.live and not args.ip:
        ap.error("--live needs --ip (e.g. --ip 192.168.1.210)")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}.")

    tracker = ArmTracker(args.arm, use_hands=args.gripper or args.twist)
    robot = (XArmRobot(args.ip, args.gripper, args.fence, args.twist) if args.live
             else DryRunRobot())

    log_file = open(args.log, "w", newline="") if args.log else None
    logger = csv.writer(log_file) if log_file else None
    if logger:
        logger.writerow(["t", "engaged", "hx_m", "hy_m", "hz_m", "tx_mm", "ty_mm", "tz_mm",
                         "yaw_deg", "hand_twist_deg", "clamped", "gripper_closed", "event"])

    filt = OneEuroFilter()
    grip = FistDetector() if args.grip == "fist" else PinchDetector()
    locked = {DEPTH_AXIS}
    unwrap = AngleUnwrapper()
    twist_filt = OneEuroFilter(beta=TWIST_FILTER_BETA)

    engaged = False
    origin_h = None          # filtered human delta at engage
    origin_r = None          # robot position at engage
    last_cmd = None          # last target actually sent
    last_yaw = TOOL_RPY_DEG[2]  # last gripper yaw actually sent
    origin_yaw = None        # gripper yaw at engage
    origin_twist = None      # filtered hand twist at engage (set when the hand is first seen)
    last_send = 0.0
    last_tracked = 0.0
    clamped = False
    message, message_until = "", 0.0
    t0 = time.monotonic()
    fps, prev_frame = 0.0, None

    def say(text, secs=2.5):
        nonlocal message, message_until
        print(text)
        message, message_until = text, time.monotonic() + secs

    def disengage(reason):
        nonlocal engaged
        if engaged:
            engaged = False
            say(f"DISENGAGED: {reason}")
            if logger:
                logger.writerow([f"{time.monotonic() - t0:.3f}", 0, "", "", "", "", "", "",
                                 "", "", "", int(grip.closed), f"disengage: {reason}"])

    mode = "LIVE" if robot.live else "DRY RUN"
    print(f"\n[{mode}] tracking your {args.arm} arm on camera {args.camera}.")
    if args.twist:
        print("Wrist twist ON: palm toward the camera, turn your hand like a dial.")
    print("SPACE engage/disengage, d depth on/off, h halt, q quit.\n")

    window = f"arm mimic [{mode}]"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                say("Camera read failed.")
                break
            now = time.monotonic()

            result = tracker.process(frame)
            human = None
            if result["delta"] is not None:
                last_tracked = now
                raw = np.array([result["delta"]["x"], result["delta"]["y"], result["delta"]["z"]])
                f = filt(raw, now)
                human = {"x": f[0], "y": f[1], "z": f[2]}

            if args.gripper:
                grip.update(result["fingers"] if args.grip == "fist" else result["pinch"])

            twist = None
            if args.twist and result["twist"] is not None:
                twist = float(twist_filt(np.array([unwrap(result["twist"])]), now)[0])

            err = robot.error()
            if err:
                disengage(err)

            if engaged and now - last_tracked > LOST_TIMEOUT_S:
                disengage("tracking lost")

            target = None
            if engaged and human is not None:
                delta_h = {k: human[k] - origin_h[k] for k in ("x", "y", "z")}
                desired = origin_r + human_delta_to_robot_mm(delta_h, locked=locked)
                desired, clamped = clamp_to_workspace(desired)
                target = limit_step(last_cmd, desired)

                yaw_target = last_yaw  # no hand seen: hold the current twist
                if args.twist and twist is not None:
                    if origin_twist is None:
                        origin_twist, origin_yaw = twist, last_yaw
                    yaw_target = limit_angle_step(
                        last_yaw, twist_to_yaw(twist - origin_twist, origin_yaw))

                if now - last_send >= 1.0 / SEND_HZ:
                    code = robot.move_to(target, yaw_target)
                    if code != 0:
                        disengage(f"set_position returned {code}")
                    else:
                        last_cmd = target
                        last_yaw = yaw_target
                        last_send = now

                if args.gripper:
                    robot.set_gripper(grip.closed)

                if logger:
                    logger.writerow([f"{now - t0:.3f}", 1, f"{human['x']:.4f}", f"{human['y']:.4f}",
                                     f"{human['z']:.4f}", f"{target[0]:.1f}", f"{target[1]:.1f}",
                                     f"{target[2]:.1f}", f"{yaw_target:.1f}",
                                     "" if twist is None else f"{twist:.1f}",
                                     int(clamped), int(grip.closed), ""])

            # ---- overlay
            tracker.draw_overlay(frame, result)
            state_col = (0, 220, 0) if engaged else (0, 200, 255)
            put(frame, f"{mode} | {'ENGAGED' if engaged else 'disengaged (SPACE to engage)'}", 0, state_col)
            fps = 0.9 * fps + 0.1 / max(now - prev_frame, 1e-3) if prev_frame else 0.0
            prev_frame = now
            put(frame, f"tracking: {'OK' if human is not None else 'NO'} | depth: "
                       f"{'off' if DEPTH_AXIS in locked else 'ON'} | {fps:.0f} fps", 1)
            pos = last_cmd if last_cmd is not None else robot.current_xyz()
            yaw_txt = ""
            if args.twist:
                yaw_txt = f" yaw={last_yaw:+.0f}" + ("" if result["twist"] is not None else " (no hand)")
            put(frame, f"target X={pos[0]:.0f} Y={pos[1]:.0f} Z={pos[2]:.0f} mm{yaw_txt}"
                       + ("  CLAMPED" if engaged and clamped else ""), 2,
                (0, 0, 255) if engaged and clamped else (255, 255, 255))
            if args.gripper:
                if args.grip == "fist":
                    seen = (f"fingers up={result['fingers']}" if result["fingers"] is not None
                            else "no hand seen")
                else:
                    seen = (f"pinch={result['pinch']:.2f}" if result["pinch"] is not None
                            else "no hand seen")
                put(frame, f"gripper: {'CLOSED' if grip.closed else 'open'}  ({seen})", 3,
                    (0, 200, 255) if grip.closed else (255, 255, 255))
            if message and now < message_until:
                put(frame, message, 4, (0, 200, 255))

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                if engaged:
                    disengage("by operator")
                elif human is None:
                    say("Can't engage: your shoulder and wrist are not both tracked.")
                else:
                    start = robot.current_xyz()
                    base = TOOL_RPY_DEG[2]
                    yaw_now = base + wrap_deg(robot.current_yaw() - base)
                    if not in_workspace(start):
                        say("Can't engage: arm is outside the WORKSPACE box.")
                    elif args.twist and abs(yaw_now - base) > TWIST_LIMIT_DEG + 1:
                        say(f"Can't engage: gripper yaw {yaw_now:+.0f} is outside the twist limit.")
                    else:
                        origin_h = dict(human)
                        origin_r = start
                        last_cmd = start
                        last_yaw = yaw_now
                        origin_yaw = last_yaw
                        origin_twist = twist  # None -> captured when the hand is first seen
                        engaged = True
                        say("ENGAGED - the arm now follows your wrist.")
            elif key == ord("d"):
                if engaged:
                    say("Disengage before changing depth mode.")
                else:
                    locked = set() if locked else {DEPTH_AXIS}
                    say(f"Depth (robot {DEPTH_AXIS.upper()}) {'ON' if not locked else 'off'}.")
            elif key == ord("h"):
                robot.halt()
                disengage("halt key")
                last_cmd = None
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down - stopping the arm.")
        robot.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
