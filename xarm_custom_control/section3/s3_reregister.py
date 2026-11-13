#!/usr/bin/env python3
"""
Re-register a calibration after a small camera shift, instead of touching all the dots again.

Finds every calibration dot in a fresh camera frame (same template matching as s3_check_drift.py),
checks that the dots all moved together (a plain shift, no rotation or zoom), moves the clicked
pixel positions by that shift, and saves the fresh frame as the new reference. The robot X, Y, Z
of each dot do not change. The old files are kept with an _before_rr suffix.

  python3 s3_reregister.py calib/calib_live.json [--camera 0] [--max-spread 1.0]
Afterwards rerun s3_calib_analyze.py and s3_check_drift.py (it should then say OK).
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

from s3_check_drift import dot_shifts
from s3_common import CAM_H, CAM_W, open_camera


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calib")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--max-spread", type=float, default=1.0,
                    help="largest allowed difference (px) between one dot's shift and the median shift")
    a = ap.parse_args()
    session = json.load(open(a.calib))
    folder = os.path.dirname(a.calib)
    ref_path = os.path.join(folder, session.get("reference_frame", ""))
    ref = cv2.imread(ref_path)
    cap = open_camera(a.camera, CAM_W, CAM_H)
    ok, cur = cap.read()
    cap.release()
    if ref is None or not ok or ref.shape != cur.shape:
        sys.exit(f"Could not read the reference ({ref_path}) and a camera frame at the same size.")

    ds = [d for d in dot_shifts(ref, cur, session["points"]) if d[3] >= 0.6]
    if len(ds) < 6:
        sys.exit(f"Only {len(ds)} dots found clearly; clear the table (move the arm away) and try again.")
    shifts = np.array([[dx, dy] for _, dx, dy, _ in ds])
    med = np.median(shifts, axis=0)
    spread = np.linalg.norm(shifts - med, axis=1)
    print(f"{len(ds)} dots found; median shift dx {med[0]:+.2f} px, dy {med[1]:+.2f} px; "
          f"largest difference from the median {spread.max():.2f} px")
    if spread.max() > a.max_spread:
        sys.exit("The dots did not all move together (the camera turned or tilted, not just shifted). "
                 "Recalibrate instead.")

    stem = os.path.splitext(a.calib)[0]
    shutil.copy(a.calib, stem + "_before_rr.json")
    shutil.copy(ref_path, os.path.splitext(ref_path)[0] + "_before_rr.png")
    for p in session["points"]:
        p["u"] = round(p["u"] + float(med[0]), 2)
        p["v"] = round(p["v"] + float(med[1]), 2)
    session["reregistered"] = {"dx_px": round(float(med[0]), 3), "dy_px": round(float(med[1]), 3),
                               "dots_used": len(ds), "max_spread_px": round(float(spread.max()), 3)}
    cv2.imwrite(ref_path, cur)
    json.dump(session, open(a.calib, "w"), indent=1)
    print(f"Updated {a.calib} and {ref_path} (old copies saved with _before_rr).")


if __name__ == "__main__":
    main()
