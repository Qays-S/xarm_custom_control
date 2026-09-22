#!/usr/bin/env python3
"""
Camera-to-robot calibration tool (homography version - flexible point count).

Workflow for each point:
  1. Move the arm's gripper tip to hover just above a spot on the table.
  2. Read the arm's real X/Y position - from
     `ros2 topic echo /xarm/robot_states --once` (look at the `pose` field,
     first two numbers are X, Y in mm).
  3. Click that same physical spot in the camera window that pops up.
  4. Enter the X, Y you read from the robot when prompted in the terminal.
  5. Repeat for as many points as you want (4 minimum, more is better -
     spread across the whole area, both edges and interior).
  6. Press 's' to compute and save the homography once you have at least
     4 points. Press 'q' to quit without saving.

Usage:
    python3 calibrate_camera.py [camera_index]

Output:
    Saves camera_calibration.json in the same folder - used by
    pixel_to_world.py to convert any future detected pixel into a real
    robot X/Y position.
"""
import sys
import json
import cv2
import numpy as np

MIN_POINTS_REQUIRED = 4

pixel_points = []
world_points = []
latest_frame = None


def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    print(f"\nClicked pixel: ({x}, {y})")
    try:
        wx = float(input("  Enter robot X (mm): ").strip())
        wy = float(input("  Enter robot Y (mm): ").strip())
    except ValueError:
        print("  Invalid number, discarding this point.")
        return

    pixel_points.append([x, y])
    world_points.append([wx, wy])
    print(f"  Saved point {len(pixel_points)}: pixel=({x},{y}) -> world=({wx},{wy})")


def main():
    global latest_frame

    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera at index {camera_index}.")
        sys.exit(1)

    window_name = "Calibration - click a point, then enter its robot X/Y in the terminal"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)

    print("Click points in the window, enter their robot X/Y when prompted.")
    print(f"Need at least {MIN_POINTS_REQUIRED} points - more is better, spread across the whole area.")
    print("Press 's' to solve and save once you have enough points. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        latest_frame = frame.copy()

        for i, (px, py) in enumerate(pixel_points, start=1):
            cv2.circle(latest_frame, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(
                latest_frame, str(i), (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        cv2.putText(
            latest_frame, f"Points collected: {len(pixel_points)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        cv2.imshow(window_name, latest_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit without saving.")
            break
        elif key == ord("s"):
            if len(pixel_points) < MIN_POINTS_REQUIRED:
                print(f"Need at least {MIN_POINTS_REQUIRED} points, only have {len(pixel_points)}.")
                continue

            src = np.array(pixel_points, dtype=np.float32)
            dst = np.array(world_points, dtype=np.float32)

            H, mask = cv2.findHomography(src, dst, method=0)

            if H is None:
                print("Homography computation failed - check your points aren't collinear.")
                continue

            calibration_data = {
                "homography": H.tolist(),
                "pixel_points": pixel_points,
                "world_points": world_points,
                "camera_index": camera_index,
                "transform_type": "homography",
            }

            with open("camera_calibration.json", "w") as f:
                json.dump(calibration_data, f, indent=2)

            print(f"\nSaved homography using {len(pixel_points)} points to camera_calibration.json")
            print("Homography matrix:")
            print(H)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
