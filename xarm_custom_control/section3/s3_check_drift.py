#!/usr/bin/env python3
"""
Check that the camera has not moved since calibration (paper step: "rigidly mount
the RGB camera and do not move it during a calibrated session").

Compares a live frame (or an image file) with the reference frame saved by
s3_collect.py, using ORB features on the static scene. Reports the median pixel
shift and the shift the calibration homography would imply in mm on the table.
Run it at the start and end of every recording session and log the result.

Usage
  python3 s3_check_drift.py calib_session_reference_frame.png [--camera 0 | --image now.png]
                            [--calib camera_calibration_s3.json] [--max-px 1.0]
Exit code 0 = OK, 1 = camera moved (recalibrate), 2 = could not decide.
"""
import argparse
import sys

import cv2
import numpy as np


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image")
    ap.add_argument("--calib")
    ap.add_argument("--max-px", type=float, default=1.0)
    a = ap.parse_args()
    ref = cv2.imread(a.reference)
    if a.image:
        cur = cv2.imread(a.image)
    else:
        cap = cv2.VideoCapture(a.camera)
        for _ in range(10):  # let auto-exposure settle
            ok, cur = cap.read()
        cap.release()
    if ref is None or cur is None or ref.shape != cur.shape:
        sys.exit("Could not read both images at the same resolution.")
    r = shift(ref, cur)
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
