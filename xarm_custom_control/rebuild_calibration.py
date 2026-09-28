#!/usr/bin/env python3
"""
Reconstructs camera_calibration.json from the 8 points recorded in-session
today (2026-09-28), after calibrate_camera.py was killed before 's' was
pressed. Run this on the lab laptop from the same directory calibrate_camera.py
normally saves into (~/ros2_ws/src/xarm_custom_control/xarm_custom_control/).

This does NOT move the arm or touch the camera - it's a pure offline fit
from the pixel/world pairs you already collected and confirmed on-screen.

It also writes calibration_points_partial.json, so the patched
calibrate_camera.py can resume from these 8 points if you add more later.
"""
import json
import cv2
import numpy as np

CAMERA_INDEX = 0  # confirmed working index today

pixel_points = [
    [518, 164],
    [466, 199],
    [417, 233],
    [366, 163],
    [417, 137],
    [462, 99],
    [411, 41],
    [367, 68],
]

world_points = [
    [660.91, -90.63],
    [660.97, -180.03],
    [660.94, -283.58],
    [530.15, -283.60],
    [530.14, -182.13],
    [530.05, -89.59],
    [405.68, -89.82],
    [398.42, -181.71],
]

src = np.array(pixel_points, dtype=np.float32)
dst = np.array(world_points, dtype=np.float32)

H, _ = cv2.findHomography(src, dst, method=0)
if H is None:
    print("Homography fit FAILED - points may be degenerate.")
    raise SystemExit(1)

# Fit error: reproject each pixel point through H, compare to recorded world point.
predicted = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
fit_err = np.linalg.norm(predicted - dst, axis=1)

# Leave-one-out error: refit without each point and predict it. This is a
# more honest estimate of accuracy on a point the fit hasn't seen.
loo_err = []
for i in range(len(src)):
    keep = np.arange(len(src)) != i
    Hi, _ = cv2.findHomography(src[keep], dst[keep], method=0)
    p = cv2.perspectiveTransform(src[i:i + 1].reshape(-1, 1, 2), Hi).reshape(2)
    loo_err.append(float(np.linalg.norm(p - dst[i])))
loo_err = np.array(loo_err)

print("Point  pixel        recorded world        predicted world      fit err  held-out err")
for i, ((px, py), (wx, wy), (pwx, pwy)) in enumerate(zip(pixel_points, world_points, predicted), start=1):
    print(f"  {i}   ({px:3d},{py:3d})   ({wx:7.2f},{wy:8.2f})   ({pwx:7.2f},{pwy:8.2f})"
          f"   {fit_err[i-1]:5.2f}mm   {loo_err[i-1]:5.2f}mm")

print(f"\nFit error:      mean {fit_err.mean():.2f} mm, max {fit_err.max():.2f} mm")
print(f"Held-out error: mean {loo_err.mean():.2f} mm, max {loo_err.max():.2f} mm")
if fit_err.max() > 15.0:
    print("WARNING: error is fairly high for a homography fit of a flat plane.")
    print("Points may not be coplanar/consistent - consider redoing calibration physically.")

calibration_data = {
    "homography": H.tolist(),
    "pixel_points": pixel_points,
    "world_points": world_points,
    "camera_index": CAMERA_INDEX,
    "transform_type": "homography",
    "mean_error_mm": float(fit_err.mean()),
    "max_error_mm": float(fit_err.max()),
}
with open("camera_calibration.json", "w") as f:
    json.dump(calibration_data, f, indent=2)

with open("calibration_points_partial.json", "w") as f:
    json.dump({"pixel_points": pixel_points, "world_points": world_points}, f, indent=2)

print("\nSaved camera_calibration.json from the 8 recorded points.")
print("Also wrote calibration_points_partial.json so calibrate_camera.py can resume from them.")
print("Next: run detect_and_locate.py and sanity-check a few live readings")
print("against where the box actually is before trusting detect_and_move.py.")
