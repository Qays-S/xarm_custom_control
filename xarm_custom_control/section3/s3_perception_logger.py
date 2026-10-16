#!/usr/bin/env python3
"""
Record what the camera and pose detector see, frame by frame, for Section III
(and later as the raw input for the forecasting datasets).

For every frame it saves:
  * the raw RGB frame (frames/000123.jpg, or .png with --lossless)
  * pixel coordinates (u, v) and detector confidence for the left and right
    shoulder, elbow, wrist and hand, plus a missing flag per keypoint
  * timing: when the read started/finished, when detection finished, and the
    camera driver's own timestamp if it provides one

Keypoints: MediaPipe Pose (shoulder 11/12, elbow 13/14, wrist 15/16; confidence =
landmark visibility). "Hand" = MediaPipe Hands landmark 9 (base of the middle
finger) of the hand nearest each pose wrist; confidence = hand detection score.
Pose smoothing is OFF by default so jitter and latency are measured honestly
(--smooth turns it on, matching arm_mimic.py, and is recorded in meta.json).

Keys while recording (window focused)
  s   start/stop a STATIC HOLD segment (keep the arm still; used to measure jitter)
  o   start/stop an OCCLUSION segment (you block the arm on purpose)
  m   drop a generic marker (e.g. "target switch cue")
  q   stop
Every key press is written to events.csv with its frame index and time.

Usage
  python3 s3_perception_logger.py --out logs/session01 [--camera 0] [--duration 120]
       [--width 1280 --height 720] [--lossless] [--no-view] [--conf 0.5] [--smooth]
Requires: pip install --user "mediapipe==0.10.21" "numpy<2" opencv-python
"""
import argparse
import csv
import json
import os
import time

import cv2

POSE_IDS = {"L_shoulder": 11, "R_shoulder": 12, "L_elbow": 13, "R_elbow": 14, "L_wrist": 15, "R_wrist": 16}
KEYPOINTS = ["L_shoulder", "L_elbow", "L_wrist", "L_hand", "R_shoulder", "R_elbow", "R_wrist", "R_hand"]
HAND_CENTER = 9


def detect(pose, hands, rgb, w, h, conf_th):
    out = {k: (float("nan"), float("nan"), 0.0) for k in KEYPOINTS}
    pr = pose.process(rgb)
    pose_found = pr.pose_landmarks is not None
    if pose_found:
        lm = pr.pose_landmarks.landmark
        for k, i in POSE_IDS.items():
            out[k] = (lm[i].x * w, lm[i].y * h, float(lm[i].visibility))
    if hands is not None:
        hr = hands.process(rgb)
        if hr.multi_hand_landmarks:
            cands = []
            for hl, hd in zip(hr.multi_hand_landmarks, hr.multi_handedness):
                c = hl.landmark[HAND_CENTER]
                cands.append((c.x * w, c.y * h, float(hd.classification[0].score)))
            for side in "LR":
                wu, wv, _ = out[f"{side}_wrist"]
                if pose_found and wu == wu:  # wrist known: take the closest hand
                    best = min(cands, key=lambda c: (c[0] - wu) ** 2 + (c[1] - wv) ** 2)
                    if (best[0] - wu) ** 2 + (best[1] - wv) ** 2 < (0.15 * max(w, h)) ** 2:
                        out[f"{side}_hand"] = best
    missing = {k: int(not (out[k][2] >= conf_th and out[k][0] == out[k][0])) for k in KEYPOINTS}
    return out, missing, pose_found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", default="0", help="camera index, or a video file to replay")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--duration", type=float, default=0, help="seconds (0 = until q)")
    ap.add_argument("--lossless", action="store_true", help="save PNG instead of JPEG (quality 95)")
    ap.add_argument("--no-frames", action="store_true", help="do not save frames (NOT for paper data)")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--conf", type=float, default=0.5, help="confidence below this counts as missing")
    ap.add_argument("--model-complexity", type=int, default=1)
    a = ap.parse_args()

    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise SystemExit('Install the mediapipe pin used elsewhere: pip install --user "mediapipe==0.10.21" "numpy<2"')
    pose = mp.solutions.pose.Pose(model_complexity=a.model_complexity, smooth_landmarks=a.smooth,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = None if a.no_hands else mp.solutions.hands.Hands(max_num_hands=2, model_complexity=1,
                                                             min_detection_confidence=0.5,
                                                             min_tracking_confidence=0.5)

    src = int(a.camera) if a.camera.isdigit() else a.camera
    cap = cv2.VideoCapture(src)
    if a.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    if a.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {a.camera}")

    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)
    ext = ".png" if a.lossless else ".jpg"
    fcsv = open(os.path.join(a.out, "frames.csv"), "w", newline="")
    ecsv = open(os.path.join(a.out, "events.csv"), "w", newline="")
    fw, ew = csv.writer(fcsv), csv.writer(ecsv)
    cols = ["frame", "t_read_start", "t_read_end", "t_detect_end", "cam_ms", "pose_found"]
    for k in KEYPOINTS:
        cols += [f"{k}_u", f"{k}_v", f"{k}_conf", f"{k}_missing"]
    fw.writerow(cols)
    ew.writerow(["frame", "t", "event"])

    meta = {"started_wall": time.strftime("%Y-%m-%d %H:%M:%S"), "camera": a.camera,
            "requested_size": [a.width, a.height], "reported_fps": cap.get(cv2.CAP_PROP_FPS),
            "mediapipe": mp.__version__, "opencv": cv2.__version__, "pose_smoothing": a.smooth,
            "model_complexity": a.model_complexity, "hands": not a.no_hands, "conf_threshold": a.conf,
            "frames_saved": not a.no_frames, "frame_format": ext, "clock": "time.monotonic (s)",
            "hand_point": "MediaPipe Hands landmark 9 (middle-finger MCP)"}
    state = {"static": False, "occlusion": False}
    win = "s3 perception logger (s=static hold, o=occlusion, m=marker, q=quit)"
    t0, n = time.monotonic(), 0
    try:
        while True:
            t_rs = time.monotonic()
            ok, frame = cap.read()
            t_re = time.monotonic()
            if not ok:
                if isinstance(src, str):
                    break  # end of replayed file
                ew.writerow([n, round(t_re, 6), "read_failed"])
                continue
            h, w = frame.shape[:2]
            if n == 0:
                meta["actual_size"] = [w, h]
            cam_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            kp, miss, found = detect(pose, hands, rgb, w, h, a.conf)
            t_de = time.monotonic()
            if not a.no_frames:
                cv2.imwrite(os.path.join(a.out, "frames", f"{n:06d}{ext}"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == ".jpg" else [])
            row = [n, round(t_rs, 6), round(t_re, 6), round(t_de, 6), round(cam_ms, 3), int(found)]
            for k in KEYPOINTS:
                u, v, c = kp[k]
                row += [round(u, 2) if u == u else "", round(v, 2) if v == v else "", round(c, 4), miss[k]]
            fw.writerow(row)
            if not a.no_view:
                view = frame.copy()
                for k in KEYPOINTS:
                    u, v, c = kp[k]
                    if u == u:
                        col = (0, 200, 0) if not miss[k] else (0, 0, 255)
                        cv2.circle(view, (int(u), int(v)), 5, col, -1)
                        cv2.putText(view, k, (int(u) + 6, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
                flags = " ".join(s.upper() for s, on in state.items() if on)
                cv2.putText(view, f"frame {n}  {flags}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(win, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                for ch, name in (("s", "static"), ("o", "occlusion")):
                    if key == ord(ch):
                        state[name] = not state[name]
                        ew.writerow([n, round(t_de, 6), f"{name}_{'start' if state[name] else 'end'}"])
                if key == ord("m"):
                    ew.writerow([n, round(t_de, 6), "marker"])
            n += 1
            if n % 30 == 0:
                fcsv.flush(); ecsv.flush()
            if a.duration and t_de - t0 >= a.duration:
                break
    finally:
        for name, on in state.items():  # close any open segment
            if on:
                ew.writerow([n, round(time.monotonic(), 6), f"{name}_end"])
        meta["frames"] = n
        meta["duration_s"] = round(time.monotonic() - t0, 3)
        with open(os.path.join(a.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        fcsv.close(); ecsv.close(); cap.release()
        cv2.destroyAllWindows()
        print(f"Recorded {n} frames in {meta['duration_s']} s to {a.out}")


if __name__ == "__main__":
    main()
