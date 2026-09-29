#!/usr/bin/env python3
"""
Hand-signal controlled pick-and-place for the xArm7.

Hand pose: palm DOWN toward the table, knuckles up.

Commands:
  index + middle          (hold ~1 s) - grab the box in the workspace, send it HOME
                                        (HOME = the drop location in detect_and_move.py)
  thumb + index + middle  (hold ~1 s) - bring the box from HOME back to the workspace,
                                        to where it was last picked from (or the
                                        centre of the calibrated grid)
  FIST                    (instant)   - STOP: halts the arm mid-move and pauses
  OPEN HAND, flat         (hold ~1 s) - RESUME the paused task

After a command, relax your hand (no command shape) before the next one.

After a STOP the task is PAUSED:
  - open hand resumes it. If the gripper is holding the box it finishes the
    carry; if not, it redoes the pick (send-home re-checks where the box is).
  - while holding the box, new commands are refused - resume first.
  - while not holding, a new command replaces the paused task.

Keyboard backup (workspace window focused): g = send home, b = bring back,
s = STOP, r = resume, c = cancel a queued command, q = quit.

Safety:
  - The fist stop is a SOFTWARE stop (asks MoveIt to halt the trajectory).
    It is not a substitute for the E-stop.
  - Commands are refused when they don't make sense: send-home with no box
    seen (or outside the calibrated area); bring-back while a box is in the
    workspace.
  - With one camera for both workspace and gestures, your hand is over the
    table, so a command only starts once your hand has left the view for 1 s.
    Give stop signals from the edge of the view, away from the arm's path.

Usage:
    python3 gesture_pick_and_place.py <workspace_cam> [gesture_cam]
    workspace_cam must be the index calibrate_camera.py was run with.
    gesture_cam defaults to the workspace camera.

Run under: ros2 launch xarm_planner xarm7_planner_realmove.launch.py
           robot_ip:=192.168.1.210 kinematics_suffix:=calibrated
"""
import json
import math
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from xarm_msgs.srv import PlanPose, PlanExec, PlanSingleStraight

import detect_and_move as dm
from pixel_to_world import PixelToWorld
from hand_gestures import (HandGestureDetector, GestureTrigger, draw_status,
                           GESTURE_TWO, GESTURE_THREE, GESTURE_FIST, GESTURE_OPEN)

# Where to put the box when bringing it back and nothing has been picked yet
# this session: centre of the calibrated grid (calibration point 5 was
# reached at X=531.31, Y=-173.62).
WORKSPACE_DEFAULT_X_MM = 530.0
WORKSPACE_DEFAULT_Y_MM = -180.0

BOUNDS_MARGIN_MM = 20.0   # how far outside the calibrated points a box may be
HAND_CLEAR_SEC = 1.0      # same-camera mode: hand must be gone this long before moving

# MoveIt's move_group halts the running trajectory when "stop" is published here.
MOVEIT_EVENT_TOPIC = "/trajectory_execution_event"

SEND_HOME = "send_home"
BRING_BACK = "bring_back"
COMMAND_FOR_GESTURE = {GESTURE_TWO: SEND_HOME, GESTURE_THREE: BRING_BACK}
COMMAND_FOR_KEY = {ord("g"): SEND_HOME, ord("b"): BRING_BACK}


class Stopped(Exception):
    pass


class MoveFailed(Exception):
    pass


class Task:
    def __init__(self, command, from_xy, from_z, to_xy, to_z):
        self.command = command
        self.from_xy = from_xy
        self.from_z = from_z
        self.to_xy = to_xy
        self.to_z = to_z


class Worker:
    """Runs one pick-and-place task in a background thread, so the camera loop
    keeps reading gestures (and can stop the arm) while it moves."""

    def __init__(self, node, plan_client, exec_client):
        self.node = node
        self.plan_client = plan_client
        self.exec_client = exec_client
        self.stop_event = threading.Event()
        self.holding = False   # True while the gripper is closed on the box
        self._thread = None
        self._result = None

    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, task, cur_xy):
        self.stop_event.clear()
        self._result = None
        self._thread = threading.Thread(target=self._run, args=(task, cur_xy), daemon=True)
        self._thread.start()

    def take_result(self):
        """Returns (task, status, arm_xy) once a task has finished, else None.
        status is 'done', 'stopped' or 'failed'."""
        if self._thread is not None and not self._thread.is_alive():
            result, self._result, self._thread = self._result, None, None
            return result
        return None

    def join(self, timeout):
        if self._thread is not None:
            self._thread.join(timeout)

    # --- steps; each checks for a stop before and after ---
    def _check(self):
        if self.stop_event.is_set():
            raise Stopped()

    def _move(self, x, y, z):
        self._check()
        ok = dm.move_to_pose(self.node, self.plan_client, self.exec_client, x, y, z)
        self._check()
        if not ok:
            raise MoveFailed()

    def _traverse(self, fx, fy, tx, ty):
        """Same curved route around the base as dm.traverse, but checks for a
        stop between segments."""
        r0, a0 = math.hypot(fx, fy), math.atan2(fy, fx)
        r1, a1 = math.hypot(tx, ty), math.atan2(ty, tx)
        da = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
        n = max(1, math.ceil(abs(math.degrees(da)) / dm.MAX_SEGMENT_DEG))
        for i in range(1, n + 1):
            t = i / n
            r, a = r0 + (r1 - r0) * t, a0 + da * t
            x, y = (tx, ty) if i == n else (r * math.cos(a), r * math.sin(a))
            if n > 1:
                self.node.get_logger().info(f"  traverse segment {i}/{n} -> X={x:.1f}, Y={y:.1f}")
            self._move(x, y, dm.TRAVERSE_Z_MM)

    def _run(self, task, cur_xy):
        log = self.node.get_logger()
        cx, cy = cur_xy
        status = "failed"
        try:
            if not self.holding:
                log.info("=== Moving to pick location ===")
                self._move(cx, cy, dm.TRAVERSE_Z_MM)
                self._traverse(cx, cy, *task.from_xy)

                log.info("=== Opening gripper above pick location ===")
                self._check()
                dm.move_gripper(self.node, dm.GRIPPER_OPEN_REAL)

                log.info("=== Descending to grip ===")
                self._move(*task.from_xy, task.from_z)

                log.info("=== Closing gripper ===")
                self._check()
                dm.move_gripper(self.node, dm.GRIPPER_GRAB_BOX_REAL)
                self.holding = True

                log.info("=== Lifting ===")
                self._move(*task.from_xy, dm.TRAVERSE_Z_MM)
                carry_from = task.from_xy
            else:
                log.info("=== Resuming carry: lifting to traverse height ===")
                self._move(cx, cy, dm.TRAVERSE_Z_MM)
                carry_from = (cx, cy)

            log.info("=== Moving to place location ===")
            self._traverse(*carry_from, *task.to_xy)

            log.info("=== Descending to release ===")
            self._move(*task.to_xy, task.to_z)

            log.info("=== Opening gripper ===")
            self._check()
            dm.move_gripper(self.node, dm.GRIPPER_OPEN_REAL)
            self.holding = False

            log.info("=== Lifting clear ===")
            self._move(*task.to_xy, dm.TRAVERSE_Z_MM)
            status = "done"
        except Stopped:
            status = "stopped"
            log.warn("=== STOPPED by operator ===")
        except MoveFailed:
            log.error("Move failed.")
        except Exception as e:  # keep the camera loop alive whatever happens
            log.error(f"Task error: {e}")

        arm_xy = task.to_xy if status == "done" else dm.read_current_xy(self.node)
        self._result = (task, status, arm_xy)


def detect_box(frame, converter):
    """Same detection as detect_and_move.py. Returns (world_xy_or_None, display_frame)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, dm.HSV_LOWER, dm.HSV_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    display = frame.copy()
    if not contours:
        return None, display
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= dm.MIN_CONTOUR_AREA:
        return None, display

    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w // 2, y + h // 2
    wx, wy = converter.convert(cx, cy)
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(display, f"world=({wx:.1f}, {wy:.1f}) mm", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return (wx, wy), display


def load_bounds(path="camera_calibration.json"):
    with open(path) as f:
        pts = np.array(json.load(f)["world_points"], dtype=float)
    return pts.min(axis=0) - BOUNDS_MARGIN_MM, pts.max(axis=0) + BOUNDS_MARGIN_MM


def in_bounds(xy, bounds):
    lo, hi = bounds
    return lo[0] <= xy[0] <= hi[0] and lo[1] <= xy[1] <= hi[1]


def check_new_command(command, box_xy, bounds):
    """Returns None if the command makes sense now, else the reason it doesn't."""
    if command == SEND_HOME:
        if box_xy is None:
            return "no box detected in workspace"
        if not in_bounds(box_xy, bounds):
            return "box outside calibrated area"
    elif command == BRING_BACK and box_xy is not None:
        return "a box is already in the workspace"
    return None


def build_task(command, box_xy, last_pick_xy):
    home = (dm.DROP_X_MM, dm.DROP_Y_MM)
    if command == SEND_HOME:
        return Task(SEND_HOME, box_xy, dm.GRIP_Z_MM, home, dm.DROP_Z_MM)
    target = last_pick_xy or (WORKSPACE_DEFAULT_X_MM, WORKSPACE_DEFAULT_Y_MM)
    return Task(BRING_BACK, home, dm.DROP_Z_MM, target, dm.GRIP_Z_MM)


def main():
    ws_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gesture_index = int(sys.argv[2]) if len(sys.argv) > 2 else ws_index
    same_camera = gesture_index == ws_index

    ws_cap = cv2.VideoCapture(ws_index)
    if not ws_cap.isOpened():
        print(f"Could not open workspace camera {ws_index}.")
        sys.exit(1)
    gesture_cap = ws_cap
    if not same_camera:
        gesture_cap = cv2.VideoCapture(gesture_index)
        if not gesture_cap.isOpened():
            print(f"Could not open gesture camera {gesture_index}.")
            sys.exit(1)

    try:
        converter = PixelToWorld("camera_calibration.json")
        bounds = load_bounds("camera_calibration.json")
    except FileNotFoundError:
        print("camera_calibration.json not found. Run calibrate_camera.py first.")
        sys.exit(1)

    rclpy.init()
    node = Node("gesture_pick_and_place")
    plan_client = node.create_client(PlanPose, dm.ARM_POSE_PLAN_SERVICE)
    exec_client = node.create_client(PlanExec, dm.ARM_EXEC_SERVICE)
    for client, name in ((plan_client, dm.ARM_POSE_PLAN_SERVICE), (exec_client, dm.ARM_EXEC_SERVICE)):
        if not client.wait_for_service(timeout_sec=5.0):
            print(f"Service not available: {name}")
            sys.exit(1)

    straight = node.create_client(PlanSingleStraight, dm.ARM_STRAIGHT_PLAN_SERVICE)
    dm.straight_client = straight if straight.wait_for_service(timeout_sec=5.0) else None

    stop_pub = node.create_publisher(String, MOVEIT_EVENT_TOPIC, 10)
    for _ in range(30):  # give discovery up to 3 s to find move_group
        if stop_pub.get_subscription_count() > 0:
            break
        time.sleep(0.1)
    stop_available = stop_pub.get_subscription_count() > 0

    print("Reading current arm position...")
    start = dm.read_current_xy(node)
    if start is None:
        print("Could not read current robot position - check robot_states is publishing.")
        sys.exit(1)
    cur_xy = start
    print(f"Starting position: X={cur_xy[0]:.2f} mm, Y={cur_xy[1]:.2f} mm")
    print(f"HOME (drop) = ({dm.DROP_X_MM}, {dm.DROP_Y_MM}); TRAVERSE_Z={dm.TRAVERSE_Z_MM}; "
          f"straight-line planning {'ON' if dm.straight_client else 'OFF'}")
    if stop_available:
        print("Fist stop: ON (MoveIt is listening on /trajectory_execution_event)")
    else:
        print("WARNING: nothing is listening on /trajectory_execution_event - the fist "
              "will NOT stop the arm. Use the E-stop.")
    print(f"Workspace camera {ws_index}, gesture camera {gesture_index}"
          f"{' (same camera - moves start once your hand leaves the view)' if same_camera else ''}")
    print("2 fingers = send home | thumb+2 = bring back | FIST = stop | open hand = resume | "
          "keys: g, b, s=stop, r=resume, c=cancel, q=quit\n")

    detector = HandGestureDetector()
    trigger = GestureTrigger()
    worker = Worker(node, plan_client, exec_client)

    last_pick_xy = None   # where the box was last picked from in the workspace
    paused_task = None    # task that was stopped (or failed while holding the box)
    pending = None        # ("new", command) or ("resume", task), waiting to start
    no_hand_since = None
    message = None

    def say(text):
        nonlocal message
        message = text
        print(text)

    try:
        while True:
            ok, ws_frame = ws_cap.read()
            if not ok:
                print("Failed to read workspace camera.")
                break
            box_xy, ws_display = detect_box(ws_frame, converter)

            if same_camera:
                g_frame = ws_display
            else:
                ok, g_frame = gesture_cap.read()
                if not ok:
                    print("Failed to read gesture camera.")
                    break

            gesture, extended = detector.process(g_frame)
            fired, progress = trigger.update(gesture)
            now = time.monotonic()
            no_hand_since = (no_hand_since or now) if extended is None else None

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # --- STOP: checked first, works at any time ---
            if fired == GESTURE_FIST or key == ord("s"):
                if worker.busy():
                    worker.stop_event.set()
                    say("STOP - halting arm")
                elif pending:
                    pending = None
                    say("Stop - queued command cancelled")
                else:
                    say("Stop (arm not moving)")
            if worker.busy() and worker.stop_event.is_set():
                stop_pub.publish(String(data="stop"))  # repeat until the task has ended

            # --- a task just finished ---
            result = worker.take_result()
            if result:
                task, status, arm_xy = result
                if arm_xy is not None:
                    cur_xy = arm_xy
                if status == "done":
                    if task.command == SEND_HOME:
                        last_pick_xy = task.from_xy
                    paused_task = None
                    say(f"{task.command}: done")
                elif status == "stopped":
                    paused_task = task
                    say(f"{task.command} PAUSED{' (holding box)' if worker.holding else ''}"
                        f" - open hand to resume")
                else:
                    paused_task = task if worker.holding else None
                    if paused_task:
                        say(f"{task.command} FAILED while holding the box - open hand to retry")
                    else:
                        say(f"{task.command} FAILED - task cancelled")
                    print("If moves keep failing, check `ros2 control list_controllers`.")
                trigger.disarm()

            if key == ord("c") and pending:
                pending = None
                say("Queued command cancelled")

            # --- resume ---
            if fired == GESTURE_OPEN or key == ord("r"):
                if worker.busy():
                    say("Already moving")
                elif not paused_task:
                    say("Nothing to resume")
                else:
                    pending = ("resume", paused_task)
                    say(f"Resuming {paused_task.command}" +
                        (" - remove hand from view" if same_camera else ""))

            # --- new command ---
            command = COMMAND_FOR_GESTURE.get(fired) or COMMAND_FOR_KEY.get(key)
            if command:
                if worker.busy():
                    say("Busy - make a fist to stop")
                elif paused_task and worker.holding:
                    say(f"Refused: holding the box - open hand to resume {paused_task.command}")
                    trigger.disarm()
                else:
                    reason = check_new_command(command, box_xy, bounds)
                    if reason:
                        say(f"Refused {command}: {reason}")
                        trigger.disarm()
                    else:
                        if paused_task:
                            print(f"Paused {paused_task.command} discarded.")
                            paused_task = None
                        pending = ("new", command)
                        say(f"Queued {command}" + (" - remove hand from view" if same_camera else ""))

            # --- start a queued command when it's safe ---
            ready = (pending and not worker.busy() and
                     (not same_camera or
                      (no_hand_since is not None and now - no_hand_since >= HAND_CLEAR_SEC)))
            if ready:
                kind, value = pending
                pending = None
                if kind == "resume":
                    task = value
                    if task.command == SEND_HOME and not worker.holding:
                        # box hasn't been picked yet - use where it is now
                        reason = check_new_command(SEND_HOME, box_xy, bounds)
                        task = None if reason else build_task(SEND_HOME, box_xy, last_pick_xy)
                        if reason:
                            say(f"Can't resume send_home: {reason} (task still paused)")
                    if task:
                        paused_task = None
                else:
                    reason = check_new_command(value, box_xy, bounds)  # re-check with hand out of view
                    task = None if reason else build_task(value, box_xy, last_pick_xy)
                    if reason:
                        say(f"Dropped {value}: {reason}")
                if task:
                    say(f"Moving: {task.command} (fist = stop)")
                    print(f"  from X={task.from_xy[0]:.2f}, Y={task.from_xy[1]:.2f} "
                          f"to X={task.to_xy[0]:.2f}, Y={task.to_xy[1]:.2f}")
                    worker.start(task, cur_xy)

            draw_status(g_frame, gesture, extended, progress, trigger.armed, message,
                        detector.last_thumb_ratio)
            cv2.imshow("Workspace", ws_display)
            if not same_camera:
                cv2.imshow("Gestures", g_frame)

    finally:
        if worker.busy():
            print("Quitting - stopping the arm first...")
            worker.stop_event.set()
            for _ in range(30):
                if not worker.busy():
                    break
                stop_pub.publish(String(data="stop"))
                worker.join(0.1)
        detector.close()
        ws_cap.release()
        if not same_camera:
            gesture_cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
