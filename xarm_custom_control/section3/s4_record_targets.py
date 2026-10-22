#!/usr/bin/env python3
"""
Record target-to-target reaching for blueprint Section IV (E1-E4). The robot stays
PARKED the whole time; nothing here ever commands the arm.

You reach between taped calibration dots (the targets). The screen tells you which
dot to touch next, and the tool records, frame by frame:
  * the raw RGB frame and the arm keypoints (same columns as s3_perception_logger.py)
    plus the index fingertip, which is what touches the dot
  * every cue: when it appeared, which target, and the instruction shown
  * when your hand left the start dot and when it arrived at the goal
  * the robot's joint angles/velocities and TCP pose (read-only, from ROS 2)

Modes (blueprint step numbers in brackets)
  E1  [19] one reach at a time, random target, comfortable speed, short hold at each dot
  E2  [20] continuous: the next target appears the moment you arrive; each reach also
           asks for a speed (slow/normal/fast) and a path (straight/curved)
  E3  [21] ambiguous: TWO candidate dots light up; the real one is revealed only after
           your hand has started moving
  E4  [22] abrupt switches: on about half the reaches the goal changes shortly after
           you start moving (sometimes back to where you came from). The cue time is
           logged here; the physical change onset is measured afterwards from the
           keypoints (s4_dataset_summary.py), so the two are kept separate.

Arrival is detected automatically: the fingertip (or the hand point, if the
fingertip is missing) stays within --radius pixels of the goal dot for --dwell
seconds. If detection misses a touch, press SPACE to mark the arrival yourself.

Keys (camera window focused)
  SPACE  force "arrived" for the current goal
  x      mark the current/last reach as bad (kept in the data, flagged invalid)
  o      start/stop an OCCLUSION segment     m  generic marker
  p      pause/resume cueing                  q  stop (everything is saved as you go)

Usage
  python3 s4_record_targets.py --mode E1 --out logs/E1_s01 --targets 1 4 9 12
  python3 s4_record_targets.py --mode E4 --out logs/E4_s01 --reaches 40 --arm ros
  python3 s4_record_targets.py --mode E2 --out /tmp/test --sim        # no camera: dry run
Targets are dot ids from --calib-session (default: the day-2 touch calibration), so
every target is known both in pixels and in robot millimetres.
"""
import argparse
import csv
import json
import math
import os
import random
import threading
import time

import cv2
import numpy as np

POSE_IDS = {"L_shoulder": 11, "R_shoulder": 12, "L_elbow": 13, "R_elbow": 14, "L_wrist": 15, "R_wrist": 16}
KEYPOINTS = ["L_shoulder", "L_elbow", "L_wrist", "L_hand", "L_tip",
             "R_shoulder", "R_elbow", "R_wrist", "R_hand", "R_tip"]
HAND_CENTER, INDEX_TIP = 9, 8
SPEEDS, PATHS = ["slow", "normal", "fast"], ["straight", "curved"]


# --------------------------------------------------------------- perception
def detect(pose, hands, rgb, w, h, conf_th):
    """Same keypoints as s3_perception_logger.py, plus the index fingertip per side."""
    out = {k: (float("nan"), float("nan"), 0.0) for k in KEYPOINTS}
    pr = pose.process(rgb)
    found = pr.pose_landmarks is not None
    if found:
        lm = pr.pose_landmarks.landmark
        for k, i in POSE_IDS.items():
            out[k] = (lm[i].x * w, lm[i].y * h, float(lm[i].visibility))
    hr = hands.process(rgb)
    if hr.multi_hand_landmarks:
        cands = []
        for hl, hd in zip(hr.multi_hand_landmarks, hr.multi_handedness):
            c, t = hl.landmark[HAND_CENTER], hl.landmark[INDEX_TIP]
            s = float(hd.classification[0].score)
            cands.append(((c.x * w, c.y * h, s), (t.x * w, t.y * h, s)))
        for side in "LR":
            wu, wv, _ = out[f"{side}_wrist"]
            if found and wu == wu:
                best = min(cands, key=lambda c: (c[0][0] - wu) ** 2 + (c[0][1] - wv) ** 2)
                if (best[0][0] - wu) ** 2 + (best[0][1] - wv) ** 2 < (0.15 * max(w, h)) ** 2:
                    out[f"{side}_hand"], out[f"{side}_tip"] = best
            elif not found and len(cands) == 1 and side == "R":  # pose lost but a hand is visible
                out["R_hand"], out["R_tip"] = cands[0]
    missing = {k: int(not (out[k][2] >= conf_th and out[k][0] == out[k][0])) for k in KEYPOINTS}
    return out, missing, found


class SimPerson:
    """--sim: a fake hand that reacts to cues, for testing the tool without a camera."""

    def __init__(self, start, rng):
        self.p = np.array(start, float)
        self.goal, self.cue_t, self.rng = None, None, rng

    def set_goal(self, g, t):
        self.goal, self.cue_t = np.array(g, float), t

    def step(self, t, dt):
        if self.goal is not None and t - self.cue_t > 0.3:  # reaction time
            d = self.goal - self.p
            n = np.linalg.norm(d)
            if n > 1:
                self.p += d / n * min(n, 260 * dt)
        return self.p + self.rng.normal(0, 1.0, 2)


# --------------------------------------------------------------- robot (read-only)
class RosRobotLog(threading.Thread):
    """Subscribes to the xArm driver's joint states and robot state; never publishes."""

    def __init__(self, path, joint_topic, state_topic):
        super().__init__(daemon=True)
        import rclpy
        from sensor_msgs.msg import JointState
        self.rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node("s4_record_targets_reader")
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "source"] + [f"q{i}" for i in range(1, 8)] + [f"v{i}" for i in range(1, 8)]
                        + ["x", "y", "z", "roll", "pitch", "yaw", "state", "err"])
        self.lock, self.n, self.last = threading.Lock(), 0, {}
        self.node.create_subscription(JointState, joint_topic, self._joints, 10)
        try:
            from xarm_msgs.msg import RobotMsg
            self.node.create_subscription(RobotMsg, state_topic, self._state, 10)
        except ImportError:
            print("  xarm_msgs not importable (did you source the workspace?); logging joint states only")

    def _row(self, src, q=(), v=(), pose=(), state="", err=""):
        q = list(q)[:7] + [""] * (7 - len(list(q)[:7]))
        v = list(v)[:7] + [""] * (7 - len(list(v)[:7]))
        pose = list(pose)[:6] + [""] * (6 - len(list(pose)[:6]))
        with self.lock:
            self.w.writerow([round(time.monotonic(), 6), src] + q + v + pose + [state, err])
            self.n += 1

    def _joints(self, m):
        self.last["q"] = list(m.position)
        self._row("joint_states", m.position, m.velocity)

    def _state(self, m):
        self.last["pose"] = list(m.pose)
        self._row("robot_states", m.angle, (), m.pose, m.state, m.err)

    def run(self):
        while self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.n and self.n % 200 == 0:
                with self.lock:
                    self.f.flush()

    def close(self):
        with self.lock:
            self.f.close()
        self.node.destroy_node()
        self.rclpy.try_shutdown()


# --------------------------------------------------------------- cue logic
class Cuer:
    def __init__(self, mode, targets, rng, a):
        self.mode, self.T, self.rng, self.a = mode, targets, rng, a
        self.ids = list(targets)

    def pause_after_arrival(self):
        return {"E1": self.rng.uniform(1.0, 2.0), "E2": 0.0,
                "E3": self.rng.uniform(1.0, 2.0), "E4": self.rng.uniform(0.6, 1.2)}[self.mode]

    def direction(self, a, b):
        (ua, va), (ub, vb) = self.T[a]["px"], self.T[b]["px"]
        return math.atan2(vb - va, ub - ua)

    def next_reach(self, cur):
        others = [i for i in self.ids if i != cur]
        goal = self.rng.choice(others)
        r = {"goal": goal, "instruction": "comfortable speed", "decoy": "", "switch_to": "",
             "switch_delay": "", "resolve_delay": ""}
        if self.mode == "E2":
            r["instruction"] = f"{self.rng.choice(SPEEDS)}, {self.rng.choice(PATHS)}"
        if self.mode == "E3" and cur is not None and len(others) >= 2:
            # decoy = the other target whose direction from the start is closest to the goal's
            rest = [i for i in others if i != goal]
            d0 = self.direction(cur, goal)
            r["decoy"] = min(rest, key=lambda i: abs(math.remainder(self.direction(cur, i) - d0, 2 * math.pi)))
            r["resolve_delay"] = round(self.rng.uniform(0.10, 0.30), 3)
            r["instruction"] = "one of the two lit dots: start moving, the real one appears"
        if self.mode == "E4" and self.rng.random() < self.a.switch_prob:
            if cur is not None and self.rng.random() < 0.5:
                r["switch_to"] = cur  # reversal: back to where you came from
            else:
                r["switch_to"] = self.rng.choice([i for i in others if i != goal] or others)
            r["switch_delay"] = round(self.rng.uniform(0.10, 0.35), 3)
        return r


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["E1", "E2", "E3", "E4"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", type=int, nargs="+", default=[1, 4, 9, 12])
    ap.add_argument("--calib-session", default="calib/calib_session_touch_day2.json")
    ap.add_argument("--reaches", type=int, default=40, help="valid reaches to record, then stop")
    ap.add_argument("--side", default="R", choices=["L", "R"], help="the arm you reach with")
    ap.add_argument("--participant", default="P01")
    ap.add_argument("--camera", default="0")
    ap.add_argument("--radius", type=float, default=30, help="arrival radius around a dot (px)")
    ap.add_argument("--dwell", type=float, default=0.2, help="seconds inside the radius to count as arrived")
    ap.add_argument("--timeout", type=float, default=6.0, help="seconds before an unfinished reach is re-cued")
    ap.add_argument("--switch-prob", type=float, default=0.5, help="E4: share of reaches with a switch")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--arm", default="ros", choices=["ros", "none"],
                    help="ros: log joint states/pose from the running xArm driver (read-only)")
    ap.add_argument("--joint-topic", default="/xarm/joint_states")
    ap.add_argument("--state-topic", default="/xarm/robot_states")
    ap.add_argument("--gripper", default="closed", help="gripper state while parked (written to meta.json)")
    ap.add_argument("--lossless", action="store_true")
    ap.add_argument("--sim", action="store_true", help="no camera: a simulated hand follows the cues")
    a = ap.parse_args()

    calib = json.load(open(a.calib_session))
    dots = {p["id"]: p for p in calib["points"]}
    missing_ids = [i for i in a.targets if i not in dots]
    if missing_ids:
        raise SystemExit(f"Dots {missing_ids} are not in {a.calib_session}")
    T = {i: {"px": (float(dots[i]["u"]), float(dots[i]["v"])),
             "robot_mm": [dots[i]["X"], dots[i]["Y"], dots[i]["Z"]]} for i in a.targets}
    seed = a.seed if a.seed is not None else int(time.time()) % 100000
    rng = random.Random(seed)
    cuer = Cuer(a.mode, T, rng, a)

    if a.sim:
        cap, pose, hands = None, None, None
        sim = SimPerson(T[a.targets[0]]["px"], np.random.default_rng(seed))
        W, H = 640, 480
    else:
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise SystemExit('Install the pinned mediapipe: pip install --user "mediapipe==0.10.21" "numpy<2"')
        pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=False,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=1,
                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(int(a.camera) if a.camera.isdigit() else a.camera)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera {a.camera}")

    if os.path.exists(os.path.join(a.out, "frames.csv")):
        raise SystemExit(f"{a.out} already has a recording; use a new --out (e.g. ..._s02)")
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)
    robot = None
    if a.arm == "ros" and not a.sim:
        try:
            robot = RosRobotLog(os.path.join(a.out, "robot.csv"), a.joint_topic, a.state_topic)
            robot.start()
        except ImportError:
            raise SystemExit("rclpy not found: run `source ~/ros2_ws/install/setup.bash` first, or use --arm none")

    ext = ".png" if a.lossless else ".jpg"
    fcsv, ecsv, rcsv = (open(os.path.join(a.out, n), "w", newline="") for n in ("frames.csv", "events.csv", "reaches.csv"))
    fw, ew, rw = csv.writer(fcsv), csv.writer(ecsv), csv.writer(rcsv)
    cols = ["frame", "t_read_start", "t_read_end", "t_detect_end", "cam_ms", "pose_found"]
    for k in KEYPOINTS:
        cols += [f"{k}_u", f"{k}_v", f"{k}_conf", f"{k}_missing"]
    fw.writerow(cols + ["goal", "phase"])
    ew.writerow(["frame", "t", "event", "reach", "target", "detail"])
    RCOLS = ["reach", "mode", "start", "goal_cued", "decoy", "instruction", "t_cue", "t_depart",
             "t_resolve", "switch_to", "t_switch_cue", "goal_final", "t_arrive", "arrival", "valid", "note"]
    rw.writerow(RCOLS)

    meta = {"started_wall": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": a.mode, "participant": a.participant,
            "side": a.side, "seed": seed, "targets": {str(k): v for k, v in T.items()},
            "calib_session": a.calib_session, "radius_px": a.radius, "dwell_s": a.dwell,
            "timeout_s": a.timeout, "switch_prob": a.switch_prob if a.mode == "E4" else None,
            "robot": "parked; read-only logging" if robot else "parked; not logged",
            "gripper_state": a.gripper, "sim": a.sim, "conf_threshold": a.conf,
            "keypoint_source": "MediaPipe Pose 11-16, Hands landmark 9 (hand) and 8 (index tip); smoothing off",
            "clock": "time.monotonic (s)", "frame_format": ext}

    def log(n, t, ev, reach="", target="", detail=""):
        ew.writerow([n, round(t, 6), ev, reach, target, detail])

    # state
    cur = None                       # dot the hand is at (None before the setup touch)
    R, rid, valid_n = None, 0, 0     # current reach record
    phase, phase_until = "setup", 0.0
    inside_since, paused, occl = None, False, False
    goal = a.targets[0]
    t0, n = time.monotonic(), 0
    win = f"s4 {a.mode}  (SPACE=arrived  x=bad reach  o=occlusion  m=marker  p=pause  q=quit)"

    def new_reach(t):
        nonlocal R, rid, goal, phase
        rid += 1
        R = {k: "" for k in RCOLS}
        R.update(cuer.next_reach(cur))
        R.update({"reach": rid, "mode": a.mode, "start": cur, "goal_cued": R["goal"], "t_cue": round(t, 6),
                  "valid": 1})
        goal = R["goal"]
        phase = "cued"
        log(n, t, "cue", rid, goal, R["instruction"] + (f"; decoy {R['decoy']}" if R["decoy"] else ""))
        if a.sim:
            sim.set_goal(T[goal]["px"] if not R["decoy"] else
                         ((np.array(T[goal]["px"]) + T[R["decoy"]]["px"]) / 2), t)

    def finish(t, how):
        nonlocal cur, phase, phase_until, valid_n, R
        R.update({"goal_final": goal, "t_arrive": round(t, 6), "arrival": how})
        rw.writerow([R[k] for k in RCOLS]); rcsv.flush()
        log(n, t, "arrive", R["reach"], goal, how)
        if R["valid"]:
            valid_n += 1
        cur = goal
        phase, phase_until, R = "hold", t + cuer.pause_after_arrival(), None

    try:
        while True:
            t_rs = time.monotonic()
            if a.sim:
                frame = np.full((H, W, 3), 90, np.uint8)
                ok, cam_ms = True, (t_rs - t0) * 1000
                time.sleep(1 / 30)
            else:
                ok, frame = cap.read()
                cam_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t_re = time.monotonic()
            if not ok:
                log(n, t_re, "read_failed")
                continue
            h, w = frame.shape[:2]
            if a.sim:
                kp = {k: (float("nan"), float("nan"), 0.0) for k in KEYPOINTS}
                u, v = sim.step(t_re, 1 / 30)
                kp[f"{a.side}_tip"] = (u, v, 0.95)
                kp[f"{a.side}_hand"] = (u + 12, v + 18, 0.95)
                miss = {k: int(kp[k][2] < a.conf) for k in KEYPOINTS}
                found = True
            else:
                kp, miss, found = detect(pose, hands, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), w, h, a.conf)
            t = time.monotonic()

            # effector point: fingertip, else hand point
            eff = None
            for k in (f"{a.side}_tip", f"{a.side}_hand"):
                if not miss[k]:
                    eff = kp[k][:2]
                    break

            # ---- cue state machine
            if not paused:
                if phase == "setup":
                    pass  # waits for the first touch of the first target
                elif phase == "hold" and t >= phase_until:
                    if valid_n >= a.reaches:
                        log(n, t, "session_complete")
                        break
                    new_reach(t)
                elif phase in ("cued", "moving") and R is not None:
                    if phase == "cued" and cur is not None and eff is not None and \
                            math.dist(eff, T[cur]["px"]) > 1.5 * a.radius:
                        R["t_depart"] = round(t, 6)
                        phase = "moving"
                        log(n, t, "depart", R["reach"], cur)
                    if phase == "moving" and R["decoy"] != "" and R["t_resolve"] == "" and \
                            t >= R["t_depart"] + R["resolve_delay"]:
                        R["t_resolve"] = round(t, 6)
                        log(n, t, "resolve", R["reach"], goal, f"decoy {R['decoy']} dropped")
                        if a.sim:
                            sim.set_goal(T[goal]["px"], t - 0.3)
                    if phase == "moving" and R["switch_to"] != "" and R["t_switch_cue"] == "" and \
                            t >= R["t_depart"] + R["switch_delay"]:
                        R["t_switch_cue"] = round(t, 6)
                        goal = R["switch_to"]
                        log(n, t, "switch_cue", R["reach"], goal, "reversal" if goal == R["start"] else "new target")
                        if a.sim:
                            sim.set_goal(T[goal]["px"], t)
                    if t - R["t_cue"] > a.timeout:
                        R.update({"valid": 0, "note": "timeout"})
                        R.update({"goal_final": goal, "arrival": "timeout"})
                        rw.writerow([R[k] for k in RCOLS]); rcsv.flush()
                        log(n, t, "timeout", R["reach"], goal)
                        R, phase, phase_until = None, "hold", t + 0.5
                # arrival (also used for the setup touch)
                if phase in ("setup", "cued", "moving"):
                    resolved = R is None or R["decoy"] == "" or R["t_resolve"] != ""
                    near = eff is not None and math.dist(eff, T[goal]["px"]) <= a.radius
                    if near and resolved and (phase != "cued" or cur is None or goal != cur):
                        inside_since = inside_since or t
                        if t - inside_since >= a.dwell:
                            if phase == "setup":
                                cur, phase, phase_until = goal, "hold", t + 1.0
                                log(n, t, "setup_touch", "", goal)
                            else:
                                finish(t, "auto")
                            inside_since = None
                    else:
                        inside_since = None

            # ---- save
            cv2.imwrite(os.path.join(a.out, "frames", f"{n:06d}{ext}"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == ".jpg" else [])
            row = [n, round(t_rs, 6), round(t_re, 6), round(t, 6), round(cam_ms, 3), int(found)]
            for k in KEYPOINTS:
                u, v, c = kp[k]
                row += [round(u, 2) if u == u else "", round(v, 2) if v == v else "", round(c, 4), miss[k]]
            fw.writerow(row + [goal if phase != "hold" else "", phase])

            # ---- display
            view = frame.copy()
            lit = {goal} if phase in ("setup", "cued", "moving") else set()
            if R is not None and R["decoy"] != "" and R["t_resolve"] == "":
                lit = {goal, R["decoy"]}
            for i, d in T.items():
                c = (0, 255, 255) if i in lit else (160, 160, 160)
                cv2.circle(view, tuple(int(x) for x in d["px"]), int(a.radius), c, 3 if i in lit else 1)
                cv2.putText(view, str(i), (int(d["px"][0]) + 8, int(d["px"][1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            if eff is not None:
                cv2.circle(view, (int(eff[0]), int(eff[1])), 6, (0, 200, 0), -1)
            if phase == "setup":
                big = f"TOUCH {goal} TO START"
            elif phase == "hold":
                big = "HOLD"
            elif len(lit) == 2:
                big = f"{min(lit)} OR {max(lit)} ?"
            else:
                big = f"GO TO {goal}"
            if paused:
                big = "PAUSED (p)"
            # banner at the bottom: the top of the image has dots (e.g. #9) that must stay visible
            cv2.rectangle(view, (0, h - 70), (w, h), (0, 0, 0), -1)
            cv2.putText(view, big, (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
            sub = (R["instruction"] if R is not None and a.mode == "E2" else "") + \
                  f"   reaches {valid_n}/{a.reaches}" + ("   OCCLUSION" if occl else "")
            cv2.putText(view, sub, (12, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log(n, time.monotonic(), "quit")
                break
            if key == ord(" ") and phase in ("setup", "cued", "moving"):
                if phase == "setup":
                    cur, phase, phase_until = goal, "hold", t + 1.0
                    log(n, t, "setup_touch", "", goal, "manual")
                else:
                    finish(t, "manual")
            if key == ord("x"):
                target_r = R
                if target_r is not None:
                    target_r.update({"valid": 0, "note": "marked bad"})
                log(n, t, "marked_bad", target_r["reach"] if target_r else rid)
            if key == ord("o"):
                occl = not occl
                log(n, t, f"occlusion_{'start' if occl else 'end'}")
            if key == ord("m"):
                log(n, t, "marker")
            if key == ord("p"):
                paused = not paused
                log(n, t, "pause" if paused else "resume")
            n += 1
            if n % 30 == 0:
                fcsv.flush(); ecsv.flush()
    finally:
        if R is not None:
            R.update({"valid": 0, "note": (R["note"] + "; " if R["note"] else "") + "session ended mid-reach",
                      "goal_final": goal})
            rw.writerow([R[k] for k in RCOLS])
        meta.update({"frames": n, "duration_s": round(time.monotonic() - t0, 3), "valid_reaches": valid_n,
                     "reaches_cued": rid, "actual_size": [w, h] if n else None})
        if robot:
            meta["robot_rows"] = robot.n
            meta["robot_last"] = robot.last
            robot.close()
        with open(os.path.join(a.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        for f in (fcsv, ecsv, rcsv):
            f.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(f"{a.mode}: {valid_n} valid reaches of {rid} cued, {n} frames, {meta['duration_s']} s -> {a.out}")


if __name__ == "__main__":
    main()
