#!/usr/bin/env python3
"""
Redo the camera calibration after the camera moved: click 4 dots, no arm needed.

The dots (post-its) and the robot have not moved, so each dot's robot X, Y, Z is taken from an
earlier calibration. Only where the dots appear in the new camera picture has changed. You click
the 4 corner dots the program asks for; it works out where the other dots must be, snaps every
mark onto its dot in the picture, and shows the result for you to check.

Window keys
  left click  (step 1) mark the dot the program asks for
              (step 2) move the nearest mark onto the dot, if a mark is off
  u           undo the last click (step 1)
  s           save (step 2)
  q           quit without saving

Usage (from reliability_hmp/h3; move the arm out of the camera view first)
  python3 ../calib_from_clicks.py --from calib/calib_live2.json --out calib/calib_live3.json --camera 0
  python3 ../calib_analyze.py calib/calib_live3.json --holdout 4
"""
import argparse
import os

import cv2
import numpy as np

from cam_common import CAM_H, CAM_W, load_json, open_camera, save_json, stamp
from calib_drift_check import dot_shifts

MIN_SCORE = 0.6   # template-match score below this = the snap is not trusted (mark shown red)


def corner_dots(points):
    """The 4 dots at the corners of the grid (by robot X, Y), in a fixed order."""
    xy = np.array([[p["X"], p["Y"]] for p in points])
    picks = []
    for s in (np.array([1, 1]), np.array([1, -1]), np.array([-1, -1]), np.array([-1, 1])):
        order = np.argsort(-(xy @ s))
        picks.append(next(int(i) for i in order if int(i) not in picks))
    return picks


def locate(old_ref, cur, old_pts, corner_idx, clicks):
    """Where every dot is in the new picture.
    The 4 clicks give an old-picture -> new-picture mapping (the table is flat, so 4 points fix it).
    The old reference picture is warped into the new view, and each dot is then matched in a small
    window around where the mapping puts it. Returns a list of (u, v, score) in the order of old_pts."""
    src = np.float32([[old_pts[i]["u"], old_pts[i]["v"]] for i in corner_idx])
    dst = np.float32(clicks)
    g = cv2.getPerspectiveTransform(src, dst)
    h, w = cur.shape[:2]
    warped = cv2.warpPerspective(old_ref, g, (w, h))
    pred = cv2.perspectiveTransform(np.float32([[[p["u"], p["v"]]] for p in old_pts]), g).reshape(-1, 2)
    fake = [{"id": k, "u": float(u), "v": float(v)} for k, (u, v) in enumerate(pred)]
    found = {k: (dx, dy, sc) for k, dx, dy, sc in dot_shifts(warped, cur, fake, half=20, search0=14)}
    out = []
    for k, (u, v) in enumerate(pred):
        dx, dy, sc = found.get(k, (0.0, 0.0, 0.0))
        out.append((float(u + dx), float(v + dy), float(sc)) if sc >= MIN_SCORE else (float(u), float(v), float(sc)))
    return out


def draw(frame, msg, marks=None, clicks=(), asked=None):
    view = frame.copy()
    if marks is not None:
        for (u, v, sc, label) in marks:
            col = (0, 200, 0) if sc >= MIN_SCORE else (0, 0, 255)
            cv2.circle(view, (int(round(u)), int(round(v))), 5, col, 1)
            cv2.putText(view, label, (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    for i, (u, v) in enumerate(clicks):
        cv2.drawMarker(view, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
    mx, my = state["mouse"]
    h, w = frame.shape[:2]
    r = 20
    x0, y0 = max(0, min(w - 2 * r, mx - r)), max(0, min(h - 2 * r, my - r))
    patch = cv2.resize(frame[y0:y0 + 2 * r, x0:x0 + 2 * r], (8 * r, 8 * r), interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(patch, ((mx - x0) * 4, (my - y0) * 4), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
    view[0:8 * r, w - 8 * r:w] = patch
    cv2.putText(view, msg, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return view


state = {"mouse": (0, 0), "click": None}


def on_mouse(event, x, y, flags, param):
    state["mouse"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        state["click"] = (x, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, help="earlier calibration (dots' robot X, Y, Z)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} already exists; delete it first.")

    old = load_json(a.src)
    old_pts = sorted(old["points"], key=lambda p: p["id"])
    old_ref = cv2.imread(os.path.join(os.path.dirname(a.src), old["reference_frame"]))
    if old_ref is None:
        raise SystemExit("Could not read the earlier calibration's reference picture.")
    corners = corner_dots(old_pts)

    cap = open_camera(a.camera, CAM_W, CAM_H)
    ok, ref = cap.read()
    if not ok:
        raise SystemExit(f"Could not read camera {a.camera}")
    h, w = ref.shape[:2]
    if [w, h] != old["image_size"] or old_ref.shape[:2] != (h, w):
        raise SystemExit(f"Camera resolution {w}x{h} differs from {a.src}'s {old['image_size']}.")

    win = "calib_from_clicks"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__.split("Window keys")[1].split("Usage")[0])
    print("Click these dots, in this order: " + ", ".join(str(old_pts[i]["id"]) for i in corners))
    clicks, marks = [], None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if marks is None:
                msg = f"click dot {old_pts[corners[len(clicks)]]['id']}  ({len(clicks)}/4, u = undo)"
                cv2.imshow(win, draw(frame, msg, clicks=clicks))
            else:
                bad = [m[3] for m in marks if m[2] < MIN_SCORE]
                msg = ("check every circle sits on its dot (click to move one), s = save" if not bad else
                       f"red = not found, click to place: dots {', '.join(bad)}; s = save")
                cv2.imshow(win, draw(frame, msg, marks=marks))
            k = cv2.waitKey(15) & 0xFF
            if k == ord("q"):
                print("Quit without saving.")
                return
            if state["click"] is not None:
                c, state["click"] = state["click"], None
                if marks is None:
                    clicks.append(c)
                    if len(clicks) == 4:
                        found = locate(old_ref, ref, old_pts, corners, clicks)
                        marks = [(u, v, sc, str(p["id"])) for (u, v, sc), p in zip(found, old_pts)]
                        n_ok = sum(sc >= MIN_SCORE for _, _, sc in found)
                        print(f"Snapped {n_ok} of {len(found)} dots onto the picture.")
                else:   # move the nearest mark to the click
                    i = int(np.argmin([np.hypot(c[0] - u, c[1] - v) for u, v, _, _ in marks]))
                    marks[i] = (float(c[0]), float(c[1]), 1.0, marks[i][3])
                    print(f"Dot {marks[i][3]} placed by hand.")
            if k == ord("u") and marks is None and clicks:
                clicks.pop()
            if k == ord("s") and marks is not None:
                if any(m[2] < MIN_SCORE for m in marks):
                    print("Some dots are still red: click on each of them first.")
                    continue
                ref_path = os.path.splitext(a.out)[0] + "_reference_frame.png"
                cv2.imwrite(ref_path, ref)
                pts = [{"id": p["id"], "u": round(u, 2), "v": round(v, 2), "X": p["X"], "Y": p["Y"], "Z": p["Z"],
                        "t": stamp()} for (u, v, _, _), p in zip(marks, old_pts)]
                save_json(a.out, {"mode": "calib", "created": stamp(), "camera_index": a.camera,
                                  "image_size": [w, h],
                                  "arm_source": f"robot X, Y, Z reused from {a.src}; 4 dots clicked, rest snapped",
                                  "points": pts, "current_location": 0,
                                  "reference_frame": os.path.basename(ref_path)})
                print(f"Saved {a.out} and {ref_path}")
                return
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
