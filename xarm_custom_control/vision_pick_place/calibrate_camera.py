#!/usr/bin/env python3
"""
Camera-to-robot calibration tool (homography version - flexible point count).

Workflow for each point:
  1. Move the arm's gripper tip to hover just above a spot on the table.
  2. Read the arm's real X/Y position - from
     `ros2 topic echo /xarm/robot_states --once` (look at the `pose` field,
     first two numbers are X, Y in mm).
  3. Click that same physical spot in the camera window.
  4. Enter the X, Y you read from the robot when prompted in the terminal.
     (Empty input re-prompts; type 's' to skip the point.)
  5. Repeat for as many points as you want (4 minimum, more is better -
     spread across the whole area, both edges and interior).

Keys (with the camera window focused):
  s  - solve homography, report per-point error, and save
  u  - undo the last point
  q  - quit (collected points are already autosaved)

Every point is autosaved to calibration_points_partial.json as soon as it is
entered, and reloaded on the next run - so a crash or force-quit never loses
work. Delete that file to start fresh.

Usage:
    python3 calibrate_camera.py [camera_index]

Output:
    Saves camera_calibration.json in the same folder - used by
    pixel_to_world.py to convert any future detected pixel into a real
    robot X/Y position.
"""
import os
import sys
import json
import cv2
import numpy as np

MIN_POINTS_REQUIRED = 4
PARTIAL_FILE = "calibration_points_partial.json"
OUTPUT_FILE = "camera_calibration.json"

pixel_points = []
world_points = []

# The mouse callback only records a click here. All terminal input happens in
# the main loop, so the OpenCV window never blocks inside a callback.
pending_click = None
accepting_clicks = True


def on_click(event, x, y, flags, param):
    global pending_click
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if not accepting_clicks or pending_click is not None:
        return  # ignore clicks while a point is being entered
    pending_click = (x, y)


def save_partial():
    with open(PARTIAL_FILE, "w") as f:
        json.dump({"pixel_points": pixel_points, "world_points": world_points}, f, indent=2)


def load_partial():
    if not os.path.exists(PARTIAL_FILE):
        return
    try:
        with open(PARTIAL_FILE) as f:
            data = json.load(f)
        pixel_points.extend(data.get("pixel_points", []))
        world_points.extend(data.get("world_points", []))
        print(f"Resumed {len(pixel_points)} saved point(s) from {PARTIAL_FILE}.")
        for i, (p, w) in enumerate(zip(pixel_points, world_points), start=1):
            print(f"  {i}: pixel={tuple(p)} -> world={tuple(w)}")
        print()
    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not read {PARTIAL_FILE} ({e}); starting fresh.\n")


def ask_float(prompt):
    """Re-prompt until a number is entered. Returns None if the user skips."""
    while True:
        s = input(prompt).strip()
        if s.lower() in ("s", "skip"):
            return None
        if s == "":
            continue  # stray Enter - just ask again
        try:
            return float(s)
        except ValueError:
            print("    Not a number. Enter a value in mm, or 's' to skip this point.")


def drain_gui_events(n=5):
    """Process queued window events (e.g. extra clicks made while typing)
    without accepting them as new points."""
    global accepting_clicks, pending_click
    accepting_clicks = False
    for _ in range(n):
        cv2.waitKey(10)
    pending_click = None
    accepting_clicks = True


def handle_pending_click():
    global pending_click
    x, y = pending_click
    print(f"\nClicked pixel: ({x}, {y})")
    wx = ask_float("  Enter robot X (mm): ")
    wy = ask_float("  Enter robot Y (mm): ") if wx is not None else None

    if wx is None or wy is None:
        print("  Skipped this point.")
    else:
        pixel_points.append([x, y])
        world_points.append([wx, wy])
        save_partial()
        print(f"  Saved point {len(pixel_points)}: pixel=({x},{y}) -> world=({wx},{wy})")

    pending_click = None
    drain_gui_events()


def solve_and_save(camera_index):
    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(world_points, dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        print("Homography computation failed - check your points aren't collinear.")
        return False

    # Per-point reprojection error: how far each pixel lands from the robot
    # position you entered, after applying H.
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)

    print("\nPer-point error (mm):")
    for i, e in enumerate(errors, start=1):
        flag = "   <-- check this point" if e > max(5.0, 3 * np.median(errors)) else ""
        print(f"  {i:2d}: {e:6.2f}{flag}")
    print(f"  mean = {errors.mean():.2f} mm, max = {errors.max():.2f} mm")
    if len(pixel_points) == 4:
        print("  (With exactly 4 points the error is always ~0 - add more points to validate the fit.)")

    calibration_data = {
        "homography": H.tolist(),
        "pixel_points": pixel_points,
        "world_points": world_points,
        "camera_index": camera_index,
        "transform_type": "homography",
        "mean_error_mm": float(errors.mean()),
        "max_error_mm": float(errors.max()),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"\nSaved homography using {len(pixel_points)} points to {OUTPUT_FILE}")
    print("Homography matrix:")
    print(H)
    return True


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    load_partial()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera at index {camera_index}.")
        sys.exit(1)

    window_name = "Calibration - click a point, then enter its robot X/Y in the terminal"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)

    print("Click points in the window, enter their robot X/Y when prompted.")
    print(f"Need at least {MIN_POINTS_REQUIRED} points - more is better, spread across the whole area.")
    print("Keys (window focused): 's' solve & save, 'u' undo last point, 'q' quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        display = frame.copy()
        for i, (px, py) in enumerate(pixel_points, start=1):
            cv2.circle(display, (int(px), int(py)), 6, (0, 255, 0), -1)
            cv2.putText(display, str(i), (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if pending_click is not None:
            cv2.circle(display, pending_click, 8, (0, 255, 255), 2)
            status = "Enter robot X/Y in the terminal..."
        else:
            status = f"Points collected: {len(pixel_points)}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if pending_click is not None:
            cv2.waitKey(1)  # let the yellow marker draw before blocking on input
            handle_pending_click()
            continue

        if key == ord("q"):
            print(f"Quit. {len(pixel_points)} point(s) kept in {PARTIAL_FILE}.")
            break
        elif key == ord("u"):
            if pixel_points:
                p, w = pixel_points.pop(), world_points.pop()
                save_partial()
                print(f"Removed last point: pixel={tuple(p)} -> world={tuple(w)}. {len(pixel_points)} left.")
            else:
                print("No points to undo.")
        elif key == ord("s"):
            if len(pixel_points) < MIN_POINTS_REQUIRED:
                print(f"Need at least {MIN_POINTS_REQUIRED} points, only have {len(pixel_points)}.")
                continue
            if solve_and_save(camera_index):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()