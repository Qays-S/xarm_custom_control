#!/usr/bin/env python3
"""
Collect pixel <-> robot correspondences for calibration (mode "calib") or for the
height/parallax test (mode "parallax"). This script NEVER moves the arm; it only
reads the TCP position after you have jogged the tip into place yourself.

calib    : touch the gripper tip to the table at 12-20 spots spread over the whole
           area people will reach into (edges, corners and interior), one point each.
parallax : pick >= 3 table spots. At each spot first touch the table (height 0),
           then raise the tip straight up to known heights (e.g. 50, 100, 150, 200,
           250 mm) without changing X/Y, recording a point at every height.

Window keys
  left click  mark the tip in the image (a zoomed inset helps you click precisely)
  r / Enter   record: reads the arm (or asks you to type X Y Z) for the marked pixel
  n           parallax only: start the next table spot
  u           undo the last point
  q           quit (everything is autosaved after every point)

Usage
  python3 s3_collect.py calib    --out calib_session.json    [--camera 0] [--arm sdk|manual]
  python3 s3_collect.py parallax --out parallax_session.json [--camera 0] [--arm sdk|manual]
Re-running with the same --out resumes the session.
"""
import argparse
import os

import cv2
import numpy as np

from s3_common import ARM_IP_DEFAULT, ArmReader, load_json, save_json, stamp

click = {"pt": None, "mouse": (0, 0)}


def on_mouse(event, x, y, flags, param):
    click["mouse"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        click["pt"] = (x, y)


def draw(frame, session):
    view = frame.copy()
    for p in session["points"]:
        u, v = int(round(p["u"])), int(round(p["v"]))
        cv2.circle(view, (u, v), 4, (0, 200, 0), -1)
        label = str(p["id"]) if session["mode"] == "calib" else f"L{p['location']}"
        cv2.putText(view, label, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    if click["pt"]:
        cv2.drawMarker(view, click["pt"], (0, 0, 255), cv2.MARKER_CROSS, 18, 1)
    # zoom inset (4x) around the mouse so the tip can be clicked to the pixel
    mx, my = click["mouse"]
    h, w = frame.shape[:2]
    r = 20
    x0, y0 = max(0, min(w - 2 * r, mx - r)), max(0, min(h - 2 * r, my - r))
    patch = cv2.resize(frame[y0:y0 + 2 * r, x0:x0 + 2 * r], (8 * r, 8 * r), interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(patch, ((mx - x0) * 4, (my - y0) * 4), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
    view[0:8 * r, w - 8 * r:w] = patch
    n = len(session["points"])
    txt = f"{session['mode']}: {n} points"
    if session["mode"] == "parallax":
        txt += f" | spot L{session['current_location']}"
    elif n < 12:
        txt += f" (need 12-20, {12 - n} more)"
    cv2.putText(view, txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return view


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["calib", "parallax"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=0, help="request a capture width (0 = camera default)")
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--arm", choices=["sdk", "manual"], default="sdk")
    ap.add_argument("--ip", default=ARM_IP_DEFAULT)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.camera)
    if a.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    if a.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read camera {a.camera}")
    h, w = frame.shape[:2]

    if os.path.exists(a.out):
        session = load_json(a.out)
        print(f"Resuming {a.out}: {len(session['points'])} points")
        if session["image_size"] != [w, h]:
            raise SystemExit(f"Camera resolution {w}x{h} differs from the session's {session['image_size']}.")
    else:
        session = {"mode": a.mode, "created": stamp(), "camera_index": a.camera, "image_size": [w, h],
                   "arm_source": a.arm, "points": [], "current_location": 0}
        ref = os.path.splitext(a.out)[0] + "_reference_frame.png"
        cv2.imwrite(ref, frame)  # used by s3_check_drift.py to prove the camera did not move
        session["reference_frame"] = os.path.basename(ref)
        save_json(a.out, session)
        print(f"Saved reference frame {ref}")

    arm = ArmReader(a.arm, a.ip)
    win = f"s3_collect {a.mode}"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print(__doc__.split("Window keys")[1].split("Usage")[0])
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imshow(win, draw(frame, session))
            k = cv2.waitKey(15) & 0xFF
            if k == ord("q"):
                break
            if k == ord("u") and session["points"]:
                gone = session["points"].pop()
                save_json(a.out, session)
                print(f"Removed point {gone['id']}")
            if k == ord("n") and a.mode == "parallax":
                session["current_location"] += 1
                save_json(a.out, session)
                print(f"Next table spot: L{session['current_location']}. Touch the table there first.")
            if k in (ord("r"), 13, 10):
                if click["pt"] is None:
                    print("Click the tip in the image first.")
                    continue
                xyz = arm.read_xyz()
                if xyz is None:
                    continue
                p = {"id": (session["points"][-1]["id"] + 1) if session["points"] else 1,
                     "u": float(click["pt"][0]), "v": float(click["pt"][1]),
                     "X": xyz[0], "Y": xyz[1], "Z": xyz[2], "t": stamp()}
                if a.mode == "parallax":
                    p["location"] = session["current_location"]
                    same = [q for q in session["points"] if q["location"] == p["location"]]
                    if same and np.hypot(p["X"] - same[0]["X"], p["Y"] - same[0]["Y"]) > 2.0:
                        print("  Warning: X/Y moved > 2 mm from this spot's first point; raise the tip straight up.")
                else:
                    zs = [q["Z"] for q in session["points"]]
                    if zs and abs(p["Z"] - np.median(zs)) > 3.0:
                        print(f"  Warning: Z={p['Z']:.1f} differs from the other points by > 3 mm; "
                              f"is the tip really touching the table?")
                session["points"].append(p)
                save_json(a.out, session)
                print(f"  #{p['id']}: pixel ({p['u']:.0f}, {p['v']:.0f}) -> X={p['X']:.1f} Y={p['Y']:.1f} Z={p['Z']:.1f}")
                click["pt"] = None
    finally:
        arm.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Saved {len(session['points'])} points to {a.out}")


if __name__ == "__main__":
    main()
