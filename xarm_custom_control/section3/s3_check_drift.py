#!/usr/bin/env python3
"""
Check that the camera has not moved since calibration (paper step: "rigidly mount
the RGB camera and do not move it during a calibrated session").

Compares a live frame (or an image file) with the reference frame saved by
s3_collect.py, using ORB features on the static scene. Reports the median pixel
shift and the shift the calibration homography would imply in mm on the table.
Run it at the start and end of every recording session and log the result.

--dots CALIB (recommended): also finds each calibration dot (post-it) of CALIB in the
live frame by template matching against the reference, and decides OK / MOVED on the
dots alone. The whole-scene ORB number is still printed, but it can be fooled by things
that move on their own (e.g. the parked arm in a slightly different pose), while the dots
are exactly what the calibration depends on. CALIB's pixel positions must belong to the
reference image (e.g. now_E5_median.png with calib_cam3d.json).

Usage
  python3 s3_check_drift.py calib_session_reference_frame.png [--camera 0 | --image now.png]
                            [--calib camera_calibration_s3.json] [--max-px 1.0]
                            [--dots calib/calib_cam3d.json] [--save now.png]
Exit code 0 = OK, 1 = camera moved (recalibrate), 2 = could not decide.
"""
import argparse
import sys

import cv2
import numpy as np

from s3_common import CAM_H, CAM_W, open_camera


def shift(ref, cur):
    orb = cv2.ORB_create(3000)
    g1, g2 = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (ref, cur))
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(m) < 30:
        return None
    p1 = np.float32([k1[x.queryIdx].pt for x in m])
    p2 = np.float32([k2[x.trainIdx].pt for x in m])
    a, inl = cv2.estimateAffinePartial2D(p1, p2, ransacReprojThreshold=2.0)
    if a is None or inl.sum() < 20:
        return None
    inl = inl.ravel().astype(bool)
    d = np.linalg.norm(p2[inl] - p1[inl], axis=1)  # static-scene matches dominate the inliers
    return float(np.median(d)), int(inl.sum()), a


def dot_shifts(ref, cur, points, half=25, search0=14):
    """Sub-pixel shift of each dot patch from ref to cur (template matching)."""
    g1, g2 = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for im in (ref, cur))
    out = []
    for p in points:
        ui, vi = int(round(p["u"])), int(round(p["v"]))
        H, W_ = g2.shape
        search = min(search0, ui - half, vi - half, W_ - 1 - ui - half, H - 1 - vi - half)  # dots near the edge
        if search < 3:
            continue
        T = g1[vi - half:vi + half + 1, ui - half:ui + half + 1]
        W = g2[vi - half - search:vi + half + search + 1, ui - half - search:ui + half + search + 1]
        if T.shape != (2 * half + 1,) * 2 or W.shape != (2 * (half + search) + 1,) * 2:
            continue
        r = cv2.matchTemplate(W, T, cv2.TM_CCOEFF_NORMED)
        _, sc, _, (x, y) = cv2.minMaxLoc(r)

        def sub(a, b, c):
            d = a - 2 * b + c
            return 0.0 if d == 0 else 0.5 * (a - c) / d
        dx = x - search + (sub(r[y, x - 1], r[y, x], r[y, x + 1]) if 0 < x < r.shape[1] - 1 else 0)
        dy = y - search + (sub(r[y - 1, x], r[y, x], r[y + 1, x]) if 0 < y < r.shape[0] - 1 else 0)
        out.append((p["id"], float(dx), float(dy), float(sc)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=CAM_W)
    ap.add_argument("--height", type=int, default=CAM_H)
    ap.add_argument("--image")
    ap.add_argument("--calib")
    ap.add_argument("--max-px", type=float, default=1.0)
    ap.add_argument("--dots", help="calibration json whose dot pixel positions belong to the reference image")
    ap.add_argument("--save", help="also save the live frame to this file")
    a = ap.parse_args()
    ref = cv2.imread(a.reference)
    if a.image:
        cur = cv2.imread(a.image)
    else:
        cap = open_camera(a.camera, a.width, a.height)  # also lets auto-exposure settle
        ok, cur = cap.read()
        cap.release()
    if a.save and cur is not None:
        cv2.imwrite(a.save, cur)
    if ref is None or cur is None or ref.shape != cur.shape:
        sys.exit("Could not read both images at the same resolution "
                 f"(reference {None if ref is None else ref.shape[1::-1]}, camera "
                 f"{None if cur is None else cur.shape[1::-1]}). Use --width/--height to match the reference.")
    r = shift(ref, cur)
    if a.dots:
        import json
        ds = [d for d in dot_shifts(ref, cur, json.load(open(a.dots))["points"]) if d[3] >= 0.6]
        if len(ds) < 4:
            print(f"UNDECIDED: only {len(ds)} dots found clearly (is anything covering the table?)")
            sys.exit(2)
        mag = [float(np.hypot(dx, dy)) for _, dx, dy, _ in ds]
        med_d = float(np.median(mag))
        scene = f"; whole scene {r[0]:.2f} px, for information only" if r else ""
        msg = (f"dots moved {med_d:.2f} px (median of {len(ds)} dots, max {max(mag):.2f}; "
               f"dx {np.median([d[1] for d in ds]):+.2f}, dy {np.median([d[2] for d in ds]):+.2f}){scene}")
        if med_d > a.max_px:
            print(f"MOVED: {msg}. Recalibrate (or re-register the dots) before recording.")
            sys.exit(1)
        print(f"OK: {msg}.")
        return
    if r is None:
        print("UNDECIDED: too few matching features (is the scene lit and unobstructed?)")
        sys.exit(2)
    med, ninl, _ = r
    msg = f"median shift {med:.2f} px over {ninl} static features"
    if a.calib:
        import json
        hm = np.array(json.load(open(a.calib))["homography"])
        c = np.array([[ref.shape[1] / 2, ref.shape[0] / 2, 1.0], [ref.shape[1] / 2 + med, ref.shape[0] / 2, 1.0]])
        w = (hm @ c.T).T
        w = w[:, :2] / w[:, 2:]
        msg += f" (~{np.linalg.norm(w[1] - w[0]):.1f} mm on the table near the image centre)"
    if med > a.max_px:
        print(f"MOVED: {msg}. Recalibrate before recording.")
        sys.exit(1)
    print(f"OK: {msg}.")


if __name__ == "__main__":
    main()
