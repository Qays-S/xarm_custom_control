#!/usr/bin/env python3
"""
Converts a pixel coordinate into a real-world robot X/Y position, using the
homography saved by calibrate_camera.py.

Use as a module:
    from pixel_to_world import PixelToWorld
    converter = PixelToWorld("camera_calibration.json")
    x_mm, y_mm = converter.convert(px, py)

Or run directly to test a single pixel from the command line:
    python3 pixel_to_world.py <pixel_x> <pixel_y>
"""
import sys
import json
import numpy as np
import cv2


class PixelToWorld:
    def __init__(self, calibration_file="camera_calibration.json"):
        with open(calibration_file, "r") as f:
            data = json.load(f)
        self.homography = np.array(data["homography"], dtype=np.float64)

    def convert(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Convert a pixel (x, y) to a real-world (x_mm, y_mm) robot position."""
        pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(pt, self.homography)
        x_mm, y_mm = world_pt[0][0]
        return float(x_mm), float(y_mm)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 pixel_to_world.py <pixel_x> <pixel_y>")
        sys.exit(1)

    px, py = float(sys.argv[1]), float(sys.argv[2])

    converter = PixelToWorld()
    x_mm, y_mm = converter.convert(px, py)

    print(f"Pixel ({px}, {py}) -> Robot position (X={x_mm:.2f} mm, Y={y_mm:.2f} mm)")


if __name__ == "__main__":
    main()
