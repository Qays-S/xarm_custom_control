#!/usr/bin/env python3
"""
Camera latency ("glass-to-glass") test for the end-to-end perception latency.

The laptop screen flips a window between black and white at random intervals while
the camera watches the screen. Latency = time the first camera frame shows the new
brightness - time the screen was told to flip. This includes the screen's own refresh
delay, so it is a slight over-estimate (state that in the paper). Adding the
detector's processing time from s3_perception_analyze.py gives the end-to-end
perception latency (event in the world -> keypoints available).

IMPORTANT: never move the calibrated camera to do this. Bring the laptop screen into the
camera's view instead (e.g. lay the laptop on the table under the camera with the lid opened
back so the screen faces up), without touching the camera or its tripod. Use the same
resolution / camera settings as for recording.

Aiming: the window shows the live camera view. Click two opposite corners of the laptop
screen as it appears in that view (r = redo), then press SPACE. The window goes full screen
and flips between black and white; only the area you marked is measured.

Usage
  python3 s3_latency_test.py --out logs/latency01 [--camera 0] [--flips 40]
       [--width 1280 --height 720]
Writes flips.csv, frames.csv, summary.json (and roi.png, the camera view with the marked area).
"""
import argparse
import csv
import json
import os
import random
import threading
import time

import cv2
import numpy as np

from s3_common import CAM_H, CAM_W, open_camera


class Grabber(threading.Thread):
    """Reads frames as fast as the camera delivers them and timestamps each one."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.rows, self.running, self.last = cap, [], True, None
        self.roi = None            # (x0, y0, x1, y1) in camera pixels; None = centre of the image
        self.record = False

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            t = time.monotonic()
            if ok:
                self.last = f
                if self.record:
                    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    h, w = g.shape
                    x0, y0, x1, y1 = self.roi or (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
                    self.rows.append((t, float(g[y0:y1, x0:x1].mean())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=CAM_W)
    ap.add_argument("--height", type=int, default=CAM_H)
    ap.add_argument("--flips", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cap = open_camera(a.camera, a.width, a.height)
    g = Grabber(cap)
    g.start()

    win = "s3 latency test: click 2 corners of the laptop screen in the camera view, r = redo, SPACE = start, q = quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    corners = []

    def on_mouse(ev, x, y, fl, p):
        if ev == cv2.EVENT_LBUTTONDOWN and len(corners) < 2:
            corners.append((x, y))
    cv2.setMouseCallback(win, on_mouse)
    while True:  # aiming phase: live camera view, mark the screen
        if g.last is None:
            cv2.waitKey(30); continue
        view = g.last.copy()
        for c in corners:
            cv2.circle(view, c, 6, (0, 0, 255), -1)
        if len(corners) == 2:
            cv2.rectangle(view, corners[0], corners[1], (0, 0, 255), 2)
        msg = "click 2 opposite corners of the laptop screen" if len(corners) < 2 else "SPACE to start (r to redo)"
        cv2.putText(view, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("r"):
            corners.clear()
        if k == ord(" ") and len(corners) == 2:
            break
        if k == ord("q"):
            g.running = False
            return
    (xa, ya), (xb, yb) = corners
    m = 0.15  # shrink the box a little so only the screen itself is measured
    x0, x1, y0, y1 = sorted((xa, xb)) + sorted((ya, yb))
    dx, dy = int((x1 - x0) * m), int((y1 - y0) * m)
    g.roi = (x0 + dx, y0 + dy, x1 - dx, y1 - dy)
    snap = g.last.copy(); cv2.rectangle(snap, g.roi[:2], g.roi[2:], (0, 0, 255), 2)
    cv2.imwrite(os.path.join(a.out, "roi.png"), snap)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    white = np.full((720, 1280, 3), 255, np.uint8)
    black = np.zeros_like(white)
    g.record = True

    flips, state = [], True
    cv2.imshow(win, white); cv2.waitKey(500)
    for i in range(a.flips):
        state = not state
        cv2.imshow(win, white if state else black)
        cv2.waitKey(1)  # pushes the image to the screen
        t_show = time.monotonic()
        flips.append((t_show, "white" if state else "black"))
        hold = random.uniform(0.6, 1.2)  # random spacing avoids locking to the frame rate
        t_end = t_show + hold
        while time.monotonic() < t_end:
            cv2.waitKey(5)
    g.running = False
    g.join(timeout=1)
    cv2.destroyAllWindows()
    cap.release()

    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_frame", "roi_brightness"]); w.writerows(g.rows)
    lat = analyse(flips, g.rows)
    with open(os.path.join(a.out, "flips.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_show", "to", "latency_ms"])
        for (t, to), l in zip(flips, lat):
            w.writerow([round(t, 6), to, "" if l is None else round(l, 2)])
    good = np.array([l for l in lat if l is not None])
    summary = {"flips": len(flips), "detected": int(len(good)),
               "latency_ms": {} if not len(good) else {
                   "mean": float(good.mean()), "median": float(np.median(good)),
                   "p95": float(np.percentile(good, 95)), "min": float(good.min()), "max": float(good.max())},
               "camera_fps_measured": float(1 / np.median(np.diff([r[0] for r in g.rows]))) if len(g.rows) > 2 else None,
               "roi": list(map(int, g.roi)),
               "note": "camera + screen latency (upper bound for the camera's share)"}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


def analyse(flips, rows):
    """First frame after each flip whose brightness crosses halfway between the levels."""
    if len(rows) < 10:
        return [None] * len(flips)
    t = np.array([r[0] for r in rows]); b = np.array([r[1] for r in rows])
    lo, hi = np.percentile(b, 5), np.percentile(b, 95)
    if hi - lo < 20:
        print("Brightness barely changes: is the camera looking at the window?")
        return [None] * len(flips)
    mid = (lo + hi) / 2
    out = []
    for i, (ts, to) in enumerate(flips):
        t_next = flips[i + 1][0] if i + 1 < len(flips) else ts + 1.5
        m = (t > ts) & (t < t_next)
        hit = np.flatnonzero((b[m] > mid) if to == "white" else (b[m] < mid))
        out.append(None if not len(hit) else float((t[m][hit[0]] - ts) * 1000))
    return out


if __name__ == "__main__":
    main()
