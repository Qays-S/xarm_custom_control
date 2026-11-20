#!/usr/bin/env python3
"""
Combines box_detector.py's HSV detection with pixel_to_world.py's
calibration to find the box and print its real-world robot X/Y position
live.

Uses the HSV range you already calibrated. Edit HSV_LOWER/HSV_UPPER below
if you need to re-tune it.

Usage:
    python3 detect_and_locate.py [camera_index]

Controls:
    - Press 'q' to quit.
    - Press 'p' to print the current detected world position to the
      terminal (useful for grabbing a value to paste into a script).
"""
import sys
import cv2
import numpy as np
from pixel_to_world import PixelToWorld

# Calibrated HSV range from box_detector.py
HSV_LOWER = np.array([55, 63, 91])
HSV_UPPER = np.array([120, 170, 255])

MIN_CONTOUR_AREA = 500


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera at index {camera_index}.")
        sys.exit(1)

    try:
        converter = PixelToWorld("camera_calibration.json")
    except FileNotFoundError:
        print("camera_calibration.json not found. Run calibrate_camera.py first.")
        sys.exit(1)

    print("Detecting box and converting to robot coordinates.")
    print("Press 'p' to print the current position. Press 'q' to quit.\n")

    last_world_pos = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display = frame.copy()
        last_world_pos = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(largest)
                cx, cy = x + w // 2, y + h // 2

                world_x, world_y = converter.convert(cx, cy)
                last_world_pos = (world_x, world_y)

                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    f"pixel=({cx},{cy})",
                    (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    f"world=({world_x:.1f}, {world_y:.1f}) mm",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("Box Detection + World Position", display)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            if last_world_pos:
                print(f"World position: X={last_world_pos[0]:.2f} mm, Y={last_world_pos[1]:.2f} mm")
            else:
                print("No box detected right now.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
