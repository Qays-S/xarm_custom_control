#!/usr/bin/env python3
"""
Standalone box detector - color/contour based, with live-tunable HSV trackbars.

Run this first to find the right HSV range for your box under your lighting,
then reuse those values once you wire this into the ROS 2 pipeline.

Controls:
    - Drag the trackbars in the "Trackbars" window until the "Mask" window
      shows ONLY the box as solid white, everything else black.
    - Press 'q' to quit.
    - Press 's' to print the current HSV range to the terminal (copy these
      values once they look right - you'll want them for the real pipeline).

Usage:
    python3 box_detector.py [camera_index]

If you're not sure which camera index to use, check available cameras first:
    ls /dev/video*
(usually /dev/video0 is index 0, /dev/video2 is index 1, etc. - USB cameras
often show up as two device nodes per camera, so try a couple of indices
if 0 doesn't show your feed)
"""
import sys
import cv2
import numpy as np

MIN_CONTOUR_AREA = 500  # ignore small noise blobs - raise this if you see false positives


def nothing(x):
    pass


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera at index {camera_index}.")
        print("Try a different index, e.g.: python3 box_detector.py 1")
        sys.exit(1)

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("H min", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("H max", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("S min", "Trackbars", 50, 255, nothing)
    cv2.createTrackbar("S max", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V min", "Trackbars", 50, 255, nothing)
    cv2.createTrackbar("V max", "Trackbars", 255, 255, nothing)

    print("Adjust the trackbars until the Mask window shows only the box in white.")
    print("Press 's' to print the current HSV range. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H min", "Trackbars")
        h_max = cv2.getTrackbarPos("H max", "Trackbars")
        s_min = cv2.getTrackbarPos("S min", "Trackbars")
        s_max = cv2.getTrackbarPos("S max", "Trackbars")
        v_min = cv2.getTrackbarPos("V min", "Trackbars")
        v_max = cv2.getTrackbarPos("V max", "Trackbars")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

        # Clean up noise: erode then dilate (removes small speckles, fills small gaps)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display = frame.copy()

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(largest)
                cx, cy = x + w // 2, y + h // 2

                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    f"({cx}, {cy})  area={int(area)}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Camera Feed", display)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            print(f"\nCurrent HSV range:")
            print(f"  lower = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"  upper = np.array([{h_max}, {s_max}, {v_max}])\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
