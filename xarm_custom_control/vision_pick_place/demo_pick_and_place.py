#!/usr/bin/env python3
"""
Demo: vision-guided pick and place using the camera-to-robot homography.

What the audience sees in the window:
  - the 9 calibration points and the calibrated workspace (green area)
  - the robot's own X/Y grid drawn onto the camera image (every 50 mm), so the
    pixel -> robot mapping is visible
  - the detected box with its pixel AND robot coordinates
  - move the mouse over the image to read the robot X/Y of any pixel
  - side panel: calibration accuracy (residual + leave-one-out, worked out
    live from camera_calibration.json) and a results table

Modes (keyboard, camera window focused):
  p      PICK & PLACE  - pick the box from the workspace, put it at HOME
  b      BRING BACK    - pick the box from HOME, put it at the workspace centre
  s      SEQUENCE      - pick the box and move it through SEQUENCE_TARGETS.
                         After every place the arm parks out of view, the
                         camera re-measures the box, and the next pick uses
                         that fresh measurement. Each step's placement error
                         (where we told the arm to put it vs. where the camera
                         sees it) goes into the results table and a CSV.
  SPACE  STOP          - software stop (asks MoveIt to halt). NOT the E-stop.
  r      RESUME        - continue a stopped/failed step
  c      CANCEL        - drop the rest of the sequence
  x      gripper is empty (tell the program after you take the box out by hand)
  q      quit

Usage:
    python3 demo_pick_and_place.py [camera_index] [--dry-run]
    camera_index defaults to the one saved in camera_calibration.json.
    --dry-run: camera + overlays only, no robot connection (for rehearsing).

Run under: ros2 launch xarm_planner xarm7_planner_realmove.launch.py
           robot_ip:=192.168.1.210 kinematics_suffix:=calibrated
"""
import csv
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

import detect_and_move as dm
import gesture_pick_and_place as gp
from pixel_to_world import PixelToWorld

CALIB_FILE = "camera_calibration.json"

# Sequence placement targets (robot X, Y in mm). All inside the calibrated
# 3x3 grid (X 417-661, Y -88 to -264), pulled ~40 mm in from its edges so
# the box stays inside the camera's calibrated area.
SEQUENCE_TARGETS = [
    (610.0, -130.0),
    (610.0, -225.0),
    (470.0, -225.0),
    (470.0, -130.0),
    (540.0, -178.0),   # finish in the centre
]
WORKSPACE_CENTRE = (540.0, -178.0)
HOME = (dm.DROP_X_MM, dm.DROP_Y_MM)

# After placing in the workspace the arm moves here (at traverse height) so
# it is not blocking the camera when the box is re-measured. Default is HOME
# because it is already known to be reachable and out of the workspace.
# If HOME is slow to reach, set a closer spot that is out of the camera view.
PARK_XY = HOME

SETTLE_SEC = 0.8     # wait after the arm parks before measuring
MEASURE_SEC = 1.0    # how long to collect detections for one measurement
MIN_DETECTIONS = 5   # need at least this many detections in MEASURE_SEC
GRID_STEP_MM = 50.0
PANEL_W = 400
RESULTS_DIR = "demo_results"

GREEN, YELLOW, RED, WHITE, GREY, CYAN = (0, 200, 0), (0, 255, 255), (0, 0, 255), (255, 255, 255), (150, 150, 150), (255, 200, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- calibration
def load_calibration(path=CALIB_FILE):
    with open(path) as f:
        d = json.load(f)
    px = np.array(d["pixel_points"], dtype=np.float64)
    wd = np.array(d["world_points"], dtype=np.float64)
    H = np.array(d["homography"], dtype=np.float64)
    return d, px, wd, H


def apply_h(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def calibration_stats(px, wd, H):
    """Fit residuals and leave-one-out errors (mm), same fit as calibrate_camera.py."""
    resid = np.linalg.norm(apply_h(H, px) - wd, axis=1)
    loo = []
    for i in range(len(px)):
        keep = np.arange(len(px)) != i
        Hi, _ = cv2.findHomography(px[keep], wd[keep], method=0)
        loo.append(np.linalg.norm(apply_h(Hi, px[i:i + 1])[0] - wd[i]))
    return resid, np.array(loo)


# ------------------------------------------------------------------- drawing
def draw_workspace(img, px, wd, H_inv, bounds):
    """Calibrated area, robot X/Y grid and calibration points."""
    hull = cv2.convexHull(px.astype(np.int32))
    overlay = img.copy()
    cv2.fillPoly(overlay, [hull], (0, 120, 0))
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.polylines(img, [hull], True, GREEN, 2)

    (x0, y0), (x1, y1) = bounds
    xs = np.arange(math.ceil(x0 / GRID_STEP_MM) * GRID_STEP_MM, x1 + 1e-6, GRID_STEP_MM)
    ys = np.arange(math.ceil(y0 / GRID_STEP_MM) * GRID_STEP_MM, y1 + 1e-6, GRID_STEP_MM)
    for x in xs:   # lines of constant robot X
        line = apply_h(H_inv, [(x, y) for y in np.linspace(y0, y1, 20)]).astype(np.int32)
        cv2.polylines(img, [line], False, GREY, 1, cv2.LINE_AA)
        cv2.putText(img, f"X{x:.0f}", tuple(line[0] + (3, -3)), FONT, 0.35, GREY, 1)
    for y in ys:   # lines of constant robot Y
        line = apply_h(H_inv, [(x, y) for x in np.linspace(x0, x1, 20)]).astype(np.int32)
        cv2.polylines(img, [line], False, GREY, 1, cv2.LINE_AA)
        cv2.putText(img, f"Y{y:.0f}", tuple(line[-1] + (3, 10)), FONT, 0.35, GREY, 1)

    for i, (u, v) in enumerate(px.astype(int), start=1):
        cv2.circle(img, (u, v), 5, GREEN, -1)
        cv2.putText(img, str(i), (u + 6, v - 6), FONT, 0.45, GREEN, 1)


def cross(img, p, color, size=9, t=2):
    u, v = int(p[0]), int(p[1])
    cv2.line(img, (u - size, v), (u + size, v), color, t)
    cv2.line(img, (u, v - size), (u, v + size), color, t)


def draw_targets(img, H_inv, targets, results, active_idx):
    for i, t in enumerate(targets):
        p = apply_h(H_inv, [t])[0]
        color = CYAN if i == active_idx else (WHITE if i >= len(results) else GREY)
        cross(img, p, color)
        cv2.putText(img, f"T{i + 1}", (int(p[0]) + 8, int(p[1]) + 16), FONT, 0.45, color, 1)
    for r in results:   # measured position + error line
        if r["measured"] is None:
            continue
        pt = apply_h(H_inv, [r["target"]])[0].astype(int)
        pm = apply_h(H_inv, [r["measured"]])[0].astype(int)
        cv2.line(img, tuple(pt), tuple(pm), YELLOW, 1)
        cv2.circle(img, tuple(pm), 4, YELLOW, -1)


def draw_panel(h, lines):
    panel = np.full((h, PANEL_W, 3), 30, np.uint8)
    y = 24
    for text, color, scale in lines:
        if text == "---":
            cv2.line(panel, (10, y - 8), (PANEL_W - 10, y - 8), (80, 80, 80), 1)
            y += 8
            continue
        cv2.putText(panel, text, (12, y), FONT, scale, color, 1, cv2.LINE_AA)
        y += int(30 * scale) + 6
    return panel


# ----------------------------------------------------------------- detection
def detect(frame, converter):
    """Box detection from detect_and_move.py. Returns ((u, v), (X, Y), bbox) or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, dm.HSV_LOWER, dm.HSV_UPPER)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(cv2.erode(mask, k, iterations=1), k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) <= dm.MIN_CONTOUR_AREA:
        return None
    x, y, w, h = cv2.boundingRect(c)
    u, v = x + w // 2, y + h // 2
    return (u, v), converter.convert(u, v), (x, y, w, h)


# ---------------------------------------------------------------------- robot
class DemoWorker(gp.Worker):
    """gp.Worker (background moves + software stop), plus a park move after a
    place so the camera can see the box."""
    park_xy = None

    def _run(self, task, cur_xy):
        super()._run(task, cur_xy)
        _, status, arm_xy = self._result
        if status != "done" or self.park_xy is None or math.dist(task.to_xy, self.park_xy) < 1.0:
            return
        self.node.get_logger().info("=== Parking out of the camera view ===")
        try:
            self._traverse(*task.to_xy, *self.park_xy)
        except gp.Stopped:
            status = "stopped"
        except Exception as e:
            self.node.get_logger().error(f"Park failed: {e}")
            status = "failed"
        self._spin(0.2)
        self._result = (task, status, self._pose[:2] if self._pose else arm_xy)


def connect_robot():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from xarm_msgs.srv import PlanPose, PlanExec, PlanSingleStraight

    rclpy.init()
    node = Node("demo_pick_and_place")
    plan = node.create_client(PlanPose, dm.ARM_POSE_PLAN_SERVICE)
    exe = node.create_client(PlanExec, dm.ARM_EXEC_SERVICE)
    for client, name in ((plan, dm.ARM_POSE_PLAN_SERVICE), (exe, dm.ARM_EXEC_SERVICE)):
        if not client.wait_for_service(timeout_sec=5.0):
            sys.exit(f"Service not available: {name} - is the planner launch running?")
    straight = node.create_client(PlanSingleStraight, dm.ARM_STRAIGHT_PLAN_SERVICE)
    dm.straight_client = straight if straight.wait_for_service(timeout_sec=5.0) else None

    stop_pub = node.create_publisher(String, gp.MOVEIT_EVENT_TOPIC, 10)
    for _ in range(30):
        if stop_pub.get_subscription_count() > 0:
            break
        time.sleep(0.1)
    stop_ok = stop_pub.get_subscription_count() > 0

    if dm.read_current_xy(node) is None:
        sys.exit("Could not read /xarm/robot_states - is the arm connected?")
    worker = DemoWorker(node, plan, exe, stop_pub)
    worker.park_xy = PARK_XY
    return rclpy, node, worker, stop_ok


# ----------------------------------------------------------------------- main
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv

    try:
        calib, px, wd, H = load_calibration()
    except FileNotFoundError:
        sys.exit(f"{CALIB_FILE} not found - run from the folder that has it.")
    H_inv = np.linalg.inv(H)
    converter = PixelToWorld(CALIB_FILE)
    bounds = gp.load_bounds(CALIB_FILE)
    tight = (wd.min(axis=0), wd.max(axis=0))
    resid, loo = calibration_stats(px, wd, H)
    print(f"Calibration: {len(px)} points, residual mean {resid.mean():.1f} / max {resid.max():.1f} mm, "
          f"leave-one-out mean {loo.mean():.1f} / max {loo.max():.1f} mm")
    for t in SEQUENCE_TARGETS + [WORKSPACE_CENTRE]:
        if not gp.in_bounds(t, tight):
            print(f"WARNING: target {t} is outside the calibrated grid.")

    cam = int(args[0]) if args else int(calib.get("camera_index", 0))
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {cam}.")

    rclpy = node = worker = None
    stop_ok = False
    if not dry_run:
        rclpy, node, worker, stop_ok = connect_robot()
        print(f"Robot connected. Straight-line planning {'ON' if dm.straight_client else 'OFF'}. "
              f"Software stop {'ON' if stop_ok else 'OFF - use the E-stop'}.")

    # state: idle | moving | settling | measuring | stopped
    state, message = "idle", "Ready."
    steps = []            # remaining targets in the current run
    step_targets = []     # all targets of the current run (for drawing)
    current = None        # (task, measure_after) of the running/stopped step
    results = []          # {"step", "target", "measured", "error"}
    recent = deque(maxlen=10)   # recent world detections, for a steady reading
    samples, phase_t = [], 0.0
    remeasure = False     # True when a stop/miss happened during measuring
    mouse = {"pt": None}
    csv_path = None

    cv2.namedWindow("xArm7 demo")
    cv2.setMouseCallback("xArm7 demo", lambda e, x, y, f, p: mouse.__setitem__("pt", (x, y)))

    def steady_box():
        if len(recent) < 5:
            return None
        return tuple(np.median(np.array(recent), axis=0))

    def start_task(task, measure_after):
        nonlocal state, current, message
        current = (task, measure_after)
        if dry_run:
            message = f"[dry run] would move {fmt(task.from_xy)} -> {fmt(task.to_xy)}"
            print(message)
            state = "idle"
            return
        worker.start(task, None)
        state = "moving"

    def start_next_step(box_xy):
        target = steps.pop(0)
        start_task(gp.Task("seq", box_xy, dm.GRIP_Z_MM, target, dm.GRIP_Z_MM), True)

    def record(target, measured):
        nonlocal csv_path
        err = None if measured is None else math.dist(target, measured)
        results.append(dict(step=len(results) + 1, target=target, measured=measured, error=err))
        if csv_path is None:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            csv_path = os.path.join(RESULTS_DIR, time.strftime("run_%Y%m%d_%H%M%S.csv"))
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "target_x_mm", "target_y_mm",
                                        "measured_x_mm", "measured_y_mm", "error_mm"])
        with open(csv_path, "a", newline="") as f:
            m = measured or (None, None)
            csv.writer(f).writerow([results[-1]["step"], *target, *m,
                                    "" if err is None else f"{err:.2f}"])
        return err

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            det = detect(frame, converter)
            if det is not None and gp.in_bounds(det[1], bounds):
                recent.append(det[1])
            elif state == "idle":
                recent.clear()

            # ---- worker finished a task
            if worker is not None and state == "moving":
                res = worker.take_result()
                if res is not None:
                    task, status, _ = res
                    if status == "done" and current[1]:
                        state, phase_t, message = "settling", time.monotonic(), "Placed. Measuring box..."
                    elif status == "done":
                        state, message, current = "idle", "Done.", None
                    else:
                        state = "stopped"
                        message = (f"Step {status}. r = resume, c = cancel"
                                   + (" (gripper HOLDING the box)" if worker.holding else ""))

            # ---- re-measure the box after a place
            now = time.monotonic()
            if state == "settling" and now - phase_t > SETTLE_SEC:
                state, phase_t, samples = "measuring", now, []
            if state == "measuring":
                if det is not None:
                    samples.append(det[1])
                if now - phase_t > MEASURE_SEC:
                    target = current[0].to_xy
                    if len(samples) < MIN_DETECTIONS:
                        state, remeasure = "stopped", True
                        message = "Box not seen after place. r = measure again, c = cancel"
                    else:
                        measured = tuple(np.median(np.array(samples), axis=0))
                        err = record(target, measured)
                        print(f"Step {len(results)}: target {fmt(target)}  measured {fmt(measured)}  "
                              f"error {err:.1f} mm")
                        recent.clear()
                        recent.extend([measured] * 5)
                        current = None
                        if steps:
                            message = f"Error {err:.1f} mm. Next step..."
                            start_next_step(measured)   # next pick uses the fresh measurement
                        else:
                            state, message = "idle", f"Run finished. Last error {err:.1f} mm."

            # ---- drawing
            view = frame.copy()
            draw_workspace(view, px, wd, H_inv, tight)
            active = None
            if step_targets:
                active = len(step_targets) - len(steps) - 1 if state != "idle" else None
                draw_targets(view, H_inv, step_targets, results[-len(step_targets):] if results else [], active)
            if det is not None:
                (u, v), (X, Y), (bx, by, bw, bh) = det
                inside = gp.in_bounds((X, Y), bounds)
                cv2.rectangle(view, (bx, by), (bx + bw, by + bh), GREEN if inside else RED, 2)
                cv2.circle(view, (u, v), 5, RED, -1)
                cv2.putText(view, f"px ({u},{v}) -> X {X:.0f}, Y {Y:.0f} mm", (bx, max(15, by - 8)),
                            FONT, 0.5, YELLOW, 2)
            if mouse["pt"] is not None and mouse["pt"][0] < frame.shape[1]:
                mu, mv = mouse["pt"]
                mx, my = converter.convert(mu, mv)
                cross(view, (mu, mv), CYAN, 6, 1)
                cv2.putText(view, f"({mu},{mv}) -> X {mx:.0f}, Y {my:.0f}", (mu + 8, mv - 8), FONT, 0.45, CYAN, 1)

            lines = [("xArm7 Vision-Guided Pick & Place", WHITE, 0.6),
                     ("pixel -> robot X/Y via planar homography", GREY, 0.45),
                     ("---", None, 0),
                     (f"Calibration: {len(px)} points, robot as its own ruler", WHITE, 0.45),
                     (f"  fit residual   mean {resid.mean():.1f}  max {resid.max():.1f} mm", GREEN, 0.45),
                     (f"  leave-one-out  mean {loo.mean():.1f}  max {loo.max():.1f} mm", GREEN, 0.45),
                     ("---", None, 0),
                     (f"Mode: {state.upper()}{'  (DRY RUN)' if dry_run else ''}", YELLOW, 0.5),
                     (message[:52], WHITE, 0.42)]
            if worker is not None:
                lines.append((f"Gripper: {'HOLDING box' if worker.holding else 'empty'}   "
                              f"Stop: {'software ON' if stop_ok else 'E-STOP ONLY'}",
                              WHITE if stop_ok else RED, 0.42))
            box = steady_box()
            lines.append((f"Box: {'X %.1f, Y %.1f mm' % box if box else 'not in workspace'}", WHITE, 0.45))
            lines.append(("---", None, 0))
            lines.append(("Step  target (X,Y)    measured      err", GREY, 0.42))
            for r in results[-7:]:
                m = fmt(r["measured"]) if r["measured"] else "  --"
                e = f"{r['error']:.1f}" if r["error"] is not None else "--"
                lines.append((f" {r['step']:<3} {fmt(r['target']):<15} {m:<13} {e}", WHITE, 0.42))
            errs = [r["error"] for r in results if r["error"] is not None]
            if errs:
                lines.append((f"Placement error: mean {np.mean(errs):.1f}  max {max(errs):.1f} mm", YELLOW, 0.45))
            lines += [("---", None, 0),
                      ("p pick->home  b home->centre  s sequence", GREY, 0.42),
                      ("SPACE stop  r resume  c cancel  x empty  q quit", GREY, 0.42)]
            cv2.imshow("xArm7 demo", np.hstack([view, draw_panel(view.shape[0], lines)]))

            # ---- keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if worker is not None and state == "moving":
                    worker.stop_event.set()
                    message = "Stopping..."
                elif state in ("settling", "measuring"):
                    state, remeasure = "stopped", True
                    message = "Stopped before measuring. r = measure, c = cancel"
            elif key == ord("c") and state in ("idle", "stopped"):
                steps, current, state, remeasure = [], None, "idle", False
                message = "Cancelled." + (" Gripper still HOLDING the box!" if worker and worker.holding else "")
            elif key == ord("x") and worker is not None and state != "moving":
                worker.holding = False
                message = "Marked gripper as empty."
            elif key == ord("r") and state == "stopped" and current is not None:
                task, measure_after = current
                if remeasure:
                    remeasure = False
                    state, phase_t, message = "settling", time.monotonic(), "Measuring again..."
                else:
                    if worker is not None and not worker.holding and task.from_xy != HOME:
                        box = steady_box()
                        if box is None:
                            message = "Can't resume: box not seen in workspace."
                            continue
                        task.from_xy = box   # box may have been nudged - pick where it is now
                    message = "Resuming..."
                    start_task(task, measure_after)
            elif key in (ord("p"), ord("b"), ord("s")) and state == "idle":
                box = steady_box()
                if worker is not None and worker.holding:
                    message = "Gripper is holding something - press x once it's empty."
                elif key == ord("b"):
                    if box is not None:
                        message = "Box is already in the workspace."
                    else:
                        steps, step_targets = [], [WORKSPACE_CENTRE]
                        message = "Bringing box from HOME to the centre."
                        start_task(gp.Task("bring", HOME, dm.DROP_Z_MM, WORKSPACE_CENTRE, dm.GRIP_Z_MM), True)
                elif box is None:
                    message = "No box detected inside the calibrated area."
                elif key == ord("p"):
                    steps, step_targets = [], []
                    message = f"Picking box at {fmt(box)} -> HOME."
                    start_task(gp.Task("pick", box, dm.GRIP_Z_MM, HOME, dm.DROP_Z_MM), False)
                else:
                    steps, step_targets = list(SEQUENCE_TARGETS), list(SEQUENCE_TARGETS)
                    results.clear()
                    message = f"Sequence of {len(steps)} placements started."
                    if dry_run:
                        message = f"[dry run] sequence: {' -> '.join(fmt(t) for t in steps)}"
                        print(message)
                        steps = []
                    else:
                        start_next_step(box)
    finally:
        if worker is not None and worker.busy():
            worker.stop_event.set()
            worker.join(5.0)
        cap.release()
        cv2.destroyAllWindows()
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()
        if csv_path:
            print(f"Results saved to {csv_path}")


def fmt(xy):
    return "--" if xy is None else f"({xy[0]:.0f},{xy[1]:.0f})"


if __name__ == "__main__":
    main()
