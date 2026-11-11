#!/usr/bin/env python3
"""
H3 live trials: the planner drives the real xArm7 along the pick-and-place loop while a person reaches.

Used by s4_record_targets.py --live METHOD (the recorder keeps the cues and the logging; this file
adds the robot). Every 0.0625 s the chosen planner picks a speed (0, 0.25, 0.5 or 1 x 150 mm/s) with
exactly the same code as the offline replay (s7_replay.py), and the arm is moved that far along the
fixed loop with the SDK's online trajectory mode (mode 7).

Safety, for EVERY planner (these are not part of any method and are logged as overrides):
  - no arm keypoint seen for 0.25 s, or no new camera step for 0.3 s      -> stop
  - the real arm (from its joint angles) within --hard-stop-mm of a seen keypoint -> stop
  - controller error, lost connection, or 'h' pressed                       -> stop
  after any of these the arm waits until 3 steps in a row are clear before it moves again
  - speed capped at 150 mm/s in software AND in the controller (reduced mode), collision
    detection on. None of this replaces the physical E-stop: someone holds it at all times.
The default robot is "dry" (no arm: the loop is simulated). --robot xarm moves the real arm.
"""
import csv
import json
import math
import os
import threading
import time

import numpy as np

import s7_replay as R
from s9_online import DT, OnlineGrid, OnlineModels

TOOL_RPY = (180.0, 0.0, 0.0)          # gripper pointing straight down (as in arm_mimic.py)
START_SPEED = 50.0                    # mm/s for the move to the start pose
ACC = 1000.0                          # mm/s^2 for mode 7 moves
LOST_STEPS = 4                        # 4 steps = 0.25 s without any keypoint -> stop
STALE_S = 0.3                         # no new 16 Hz step for this long -> stop
METHODS = list(R.METHODS)             # reactive, cv, gru, mgru, sts, uncert, gate, full, full_nofb, full2


def cached_path(calib, cache_dir="logs/cache"):
    """s7_replay.Path for this calibration, saved after the first build (it runs IK for every point,
    about a minute on a laptop). The cache key is the calibration's points and the loop settings."""
    import hashlib
    import pickle
    key = hashlib.sha256(json.dumps([calib["points"], R.PICK, R.PLACE, R.Z_UP, R.Z_DOWN, R.DWELL_S, R.V_NOM,
                                     R.H_BAND], sort_keys=True).encode()).hexdigest()[:16]
    f = os.path.join(cache_dir, f"path_{key}.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    print("Building the pick-and-place loop for this calibration (about a minute, only the first time)...")
    path = R.Path(calib)
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(path, open(f, "wb"))
    return path


# ------------------------------------------------------------------ the planner (mirrors s7_replay.simulate)
class LivePlanner:
    """One planning step at a time, with the same rules and settings as the offline replay."""

    def __init__(self, path, method, p, delay):
        self.path, self.method, self.p, self.D = path, method, p, delay
        self.s, self.a_prev, self.low, self.high, self.fb = 0.0, 1.0, 0, 0, False
        self.mid_l, self.mid_d = (p.get("l_fix", (p.get("l_min", 1) + p.get("l_max", 1)) / 2),
                                  p.get("d_fix", p.get("d_min", 0) + p.get("dd", 0) / 2))

    def step(self, F):
        """F: the newest record (s9_online.OnlineModels.record or one anchor of s7_replay.forecasts).
        Returns the chosen speed and what went into it. Does not advance s (the caller does)."""
        method, p, D = self.method, self.p, self.D
        rkey = "r_s" if method == "full2" else "r"
        r = F.get(rkey, np.nan)
        r = r if r == r else 0.0
        if method in ("reactive", "cv", "gru", "sts", "full2"):
            src, pi = F["gru" if method == "full2" else method][None], np.ones(1)
        else:
            src, pi = F["modes"], F["pi"]
        if D > 0 and method != "reactive":
            traj, now_in = src[:, D - 1:D - 1 + R.K], None
        else:
            traj, now_in = src[:, D:D + R.K], F["now"]
        lam, dreq = self.mid_l, self.mid_d
        if method in ("full", "full_nofb", "uncert", "full2"):
            ru = F.get("r_u", np.nan)
            rr = r if method != "uncert" else (ru if ru == ru else 0.0)
            lam = p["l_min"] + (1 - rr) * (p["l_max"] - p["l_min"])
            dreq = p["d_min"] + (1 - rr) * p["dd"]
        if method == "gate":
            lam, dreq = p["l_fix"], p["d_fix"]
        if method in ("full", "gate", "full2"):
            self.low = self.low + 1 if r < p["tau"] else 0
            self.high = self.high + 1 if r >= p["tau"] else 0
            if not self.fb and self.low >= R.N_HYST:
                self.fb = True
            elif self.fb and self.high >= R.N_HYST:
                self.fb = False
            if not F["visible"]:
                self.fb = True
        ai, J = R.choose(self.path, self.s, self.a_prev, pi, traj, lam, dreq, now_in)
        fb = self.fb and method in ("full", "gate", "full2")
        a = 0.0 if fb else float(R.SPEEDS[ai])
        return {"a": a, "fallback": float(fb), "lam": lam, "dreq": dreq, "r": r}

    def advance(self, a, v_scale=1.0):
        self.s += a * R.V_NOM * DT * v_scale
        self.a_prev = a


# ------------------------------------------------------------------ robots
class DryRobot:
    """No arm: the gripper is wherever the commanded path position says."""

    live = False

    def __init__(self, path):
        self.path, self.cmd = path, path.xyz[0]

    def command(self, xyz, speed):
        self.cmd = xyz

    def hold(self, xyz):
        self.cmd = xyz

    def read(self):
        return self.cmd, None

    def error(self):
        return None

    def gripper(self, closed):
        pass

    def halt(self):
        pass

    def resume(self):
        return None

    def close(self):
        pass


class XArmRobot:
    """The real xArm7 through the Python SDK, online trajectory mode (mode 7), as in arm_mimic.py."""

    live = True

    def __init__(self, ip, path, use_gripper=False):
        from xarm.wrapper import XArmAPI
        self.path, self.use_gripper = path, use_gripper
        self.arm = XArmAPI(ip, is_radian=False)
        self.arm.clean_warn(); self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        time.sleep(0.5)
        if self.arm.error_code != 0:
            code = self.arm.error_code
            self.arm.disconnect()
            raise SystemExit(f"Arm reports controller error {code} after clearing "
                             f"(1 usually means the E-stop is pressed). Fix it in UFactory Studio and retry.")
        self._check_reachable()
        # controller-side limits, a second layer behind the software cap
        for call, args in (("set_collision_sensitivity", (3,)), ("set_self_collision_detection", (True,)),
                           ("set_reduced_max_tcp_speed", (R.V_NOM,)), ("set_reduced_mode", (True,))):
            try:
                code = getattr(self.arm, call)(*args)
                print(f"  {call}{args}: {'ok' if code == 0 else f'code {code}'}")
            except Exception as e:                     # older firmware: report, keep the software limits
                print(f"  {call}{args}: not available ({e})")
        self._move_to_start()
        self.arm.set_mode(7); self.arm.set_state(0)
        time.sleep(0.5)
        if self.arm.mode != 7:
            self.close()
            raise SystemExit(f"Controller did not accept mode 7 (mode is {self.arm.mode}); update the firmware.")
        if use_gripper:
            self.arm.set_gripper_mode(0); self.arm.set_gripper_enable(True)
            self.arm.set_gripper_position(850, wait=True)
        self.closed, self.halted = False, False

    def _check_reachable(self):
        """Ask the controller's IK (no motion) about every 10th point of the loop."""
        bad = []
        for p in self.path.xyz[::10]:
            code, _ = self.arm.get_inverse_kinematics([*p, *TOOL_RPY], input_is_radian=False,
                                                      return_is_radian=False)
            if code != 0:
                bad.append(p)
        if bad:
            self.arm.disconnect()
            raise SystemExit(f"{len(bad)} loop points are out of reach, e.g. {np.round(bad[0], 1)}; "
                             f"check the calibration file (dots 6 and 8).")
        print(f"  all {len(self.path.xyz[::10])} checked loop points are reachable")

    def _move_to_start(self):
        start = self.path.xyz[0]
        code, pos = self.arm.get_position()
        print(f"\nArm is at X={pos[0]:.1f} Y={pos[1]:.1f} Z={pos[2]:.1f}; "
              f"loop start (150 mm above dot 6) is X={start[0]:.1f} Y={start[1]:.1f} Z={start[2]:.1f}")
        if input(f"Move there now at {START_SPEED:.0f} mm/s? Clear the area. [y/N] ").strip().lower() != "y":
            self.arm.disconnect()
            raise SystemExit("Not moving. Exiting.")
        self.arm.set_mode(0); self.arm.set_state(0)
        time.sleep(0.5)
        code = self.arm.set_position(*start, *TOOL_RPY, speed=START_SPEED, mvacc=500, wait=True)
        if code != 0:
            self.close()
            raise SystemExit(f"Move to start failed (code {code}).")

    def command(self, xyz, speed):
        return self.arm.set_position(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                                     roll=TOOL_RPY[0], pitch=TOOL_RPY[1], yaw=TOOL_RPY[2],
                                     speed=float(max(speed, 1.0)), mvacc=ACC, wait=False)

    def hold(self, xyz):
        """Planner chose to wait: stop at the commanded point (a few mm ahead at most)."""
        return self.command(xyz, 0.5 * R.V_NOM)

    def halt(self):
        """Safety stop: the controller stops the arm where it is (state 4)."""
        if not self.halted:
            self.arm.set_state(4)
            self.halted = True

    def resume(self):
        """After a safety stop: re-arm mode 7 and return the real TCP so the planner can re-sync."""
        if not self.halted:
            return None
        self.arm.set_mode(7); self.arm.set_state(0)
        time.sleep(0.1)
        self.halted = False
        return np.array(self.arm.position[:3], float)

    def read(self):
        """Cached TCP position (mm) and joint angles (degrees) from the controller's report stream."""
        return np.array(self.arm.position[:3], float), np.array(self.arm.angles[:7], float)

    def error(self):
        if self.arm.error_code != 0:
            return f"controller error {self.arm.error_code}"
        if not self.arm.connected:
            return "lost connection to the controller"
        return None

    def gripper(self, closed):
        if self.use_gripper and closed != self.closed:
            self.arm.set_gripper_position(0 if closed else 850, wait=False)
            self.closed = closed

    def close(self):
        try:
            self.arm.set_state(4)
            time.sleep(0.2)
            self.arm.set_mode(0); self.arm.set_state(0)
        finally:
            self.arm.disconnect()


# ------------------------------------------------------------------ real-arm separation
def arm_separation(q_deg, pts_mm, table_z):
    """Planar distance (mm) from the real arm's low capsules to the nearest of pts_mm (J,2), the same
    measure as the replay (s7_replay.Path.dist) but from the measured joint angles."""
    from s7_robot import capsules
    pts = pts_mm[np.isfinite(pts_mm).all(1)]
    if q_deg is None or not len(pts):
        return np.nan
    caps = [R.clip_low(c, table_z) for c in capsules(np.radians(q_deg))]
    best = np.inf
    for a, b, r in caps:
        d = b - a
        L2 = max(float(d @ d), 1e-9)
        t = np.clip(((pts - a) @ d) / L2, 0, 1)
        best = min(best, float((np.linalg.norm(pts - (a + t[:, None] * d), axis=1) - r).min()))
    return best


# ------------------------------------------------------------------ the 16 Hz control loop
class LiveController(threading.Thread):
    """Runs beside the recorder: reads the newest 16 Hz step, plans, moves the arm, logs one row per step."""

    COLS = ["t", "tick", "anchor", "data_age_s", "method", "a_planner", "a", "override", "fallback", "lam",
            "dreq", "r", "r_u", "r_s", "visible", "s_mm", "cycles", "cmd_x", "cmd_y", "cmd_z", "tcp_x",
            "tcp_y", "tcp_z"] + [f"q{i}" for i in range(1, 8)] + ["sep_real_mm", "sep_cmd_mm", "compute_ms"]

    def __init__(self, out_dir, calib, method, side, W, H, robot="dry", ip=None, results=None,
                 reliability=None, replay_summary=None, manicast_dir=None, prepared_train=None,
                 delay=3, hard_stop_mm=20.0, v_scale=1.0, use_gripper=False):
        super().__init__(daemon=True)
        if method not in METHODS:
            raise SystemExit(f"--live must be one of {METHODS}")
        self.path = cached_path(calib)
        settings = json.load(open(replay_summary)) if replay_summary and os.path.exists(replay_summary) else {}
        chosen = settings.get("chosen_on_val", {})
        if method not in chosen:
            if method != "reactive":
                raise SystemExit(f"No tuned settings for {method} in {replay_summary}; run s7_replay.py first.")
            chosen[method] = {"d_fix": 60.0, "l_fix": 2.5, "tau": 0.0}      # Phase A default (replay's choice)
        p = dict(chosen[method])
        if method in ("full", "gate"):
            p["tau"] = settings.get("tau_from_cal", p.get("tau"))
        if method == "full2":
            p["tau"] = settings.get("tau_smoothed_from_cal", p.get("tau"))
        self.params, self.method, self.v_scale = p, method, v_scale
        if method != "reactive" and robot == "xarm" and not os.path.exists("logs/live_freeze.json"):
            print("\n  NOTE: logs/live_freeze.json is missing. Run s9_freeze.py before Phase C sessions,\n"
                  "  so the paper can show the models and settings were fixed before the comparison.\n")
        self.grid = OnlineGrid(side, W, H)
        need = (("sts",) if method == "sts" else ()) + \
               (("rel",) if method in ("uncert", "gate", "full", "full_nofb", "full2") else ())
        self.models = OnlineModels(calib, results, reliability, manicast_dir, prepared_train, side, need)
        needs = {"cv": "cv", "gru": "gru", "mgru": "modes", "sts": "sts", "full2": "gru"}
        if method != "reactive" and not self.models.have_models:
            raise SystemExit(f"{method} needs trained models in {results}")
        if method == "sts" and self.models.sts is None:
            raise SystemExit("sts needs --manicast-dir and the fine-tuned STS-GCN")
        if method in ("uncert", "gate", "full", "full_nofb", "full2") and self.models.rel is None:
            raise SystemExit(f"{method} needs the reliability model in {reliability}")
        self.needs = needs.get(method)
        self.planner = LivePlanner(self.path, method, p, delay)
        self.robot = XArmRobot(ip, self.path, use_gripper) if robot == "xarm" else DryRobot(self.path)
        self.hard_stop_mm = hard_stop_mm
        self.running, self.moving, self.held = True, False, False      # moving: cueing has started
        self.status = {"a": 0.0, "override": "waiting", "r": np.nan}
        self.f = open(os.path.join(out_dir, "live.csv"), "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(self.COLS)
        self.lost, self.tick, self.rec, self.t_new = 0, 0, None, time.monotonic()
        self.clear = R.N_HYST                 # steps without a safety stop (resume only after 3 in a row)
        self.dwells = self._dwell_ranges()
        self.info = {"method": method, "params": p, "robot": robot, "delay_steps": delay,
                     "hard_stop_mm": hard_stop_mm, "v_nom_mm_s": R.V_NOM * v_scale, "cycle_mm": self.path.length,
                     "ik_ok": self.path.ik_ok, "models": bool(self.models.have_models)}

    def _dwell_ranges(self):
        """Path sample ranges of the pick and place pauses (repeated points)."""
        same = np.r_[False, (np.abs(np.diff(self.path.xyz, axis=0)).sum(1) == 0)]
        idx = np.flatnonzero(same)
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []
        return [(g[0] - 1, g[-1]) for g in groups]

    # called by the recorder
    def add_frame(self, t, kp, miss, phase, goal):
        if self.grid.add_frame(t, kp, miss, phase, goal):
            self.t_new = time.monotonic()

    def add_cue(self, t):
        self.grid.add_cue(t)

    def run(self):
        # one thread each for torch and scikit-learn: with the camera loop running beside this one,
        # their default thread pools fight for the CPU and each step takes several times longer
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(1)
        self._limits = threadpool_limits(1)
        t_next = time.monotonic()
        while self.running:
            t_next += DT
            try:
                self.step()
            except Exception as e:             # anything unexpected: stop the arm and end the loop
                self.status["override"] = f"ERROR {e}"
                print(f"\nLIVE CONTROLLER ERROR, arm stopped: {e}")
                self.robot.hold(self.path.xyz[self.path.idx(self.planner.s)])
                self.running = False
                break
            time.sleep(max(0.0, t_next - time.monotonic()))

    def step(self):
        t = time.monotonic()
        c0 = time.perf_counter()
        steps, frames = self.grid.snapshot()
        if steps and (self.rec is None or steps[-1]["anchor"] != self.rec["anchor"]):
            rec = self.models.record(steps, frames)
            if rec is not None:
                self.rec = rec
        compute_ms = (time.perf_counter() - c0) * 1000
        rec = self.rec
        override, plan = "", {"a": 0.0, "fallback": 0.0, "lam": np.nan, "dreq": np.nan, "r": np.nan}
        if not self.moving:
            override = "waiting"
        elif self.held:
            override = "hold key"
        elif rec is None:
            override = "no data"
        else:
            plan = self.planner.step(rec)
            self.lost = 0 if rec["visible"] else self.lost + 1
            if self.lost >= LOST_STEPS:
                override = "no keypoints"
            elif t - self.t_new > STALE_S:
                override = "camera stalled"
        err = self.robot.error()
        if err:
            override = err
        tcp, q = self.robot.read()
        sep_real = arm_separation(q, rec["now"], self.path.table_z) if (rec is not None and q is not None) else np.nan
        sep_cmd = (float(self.path.dist(rec["now"][None], np.array([self.planner.s]))[0].min())
                   if rec is not None and np.isfinite(rec["now"]).all() else np.nan)
        if not override and sep_real == sep_real and sep_real < self.hard_stop_mm:
            override = "hard stop"
        safety_stop = override not in ("", "waiting")
        self.clear = 0 if safety_stop else self.clear + 1
        if not override and self.clear < R.N_HYST:
            override = "resuming"                 # clear again, but wait 3 steps before moving
        a = 0.0 if override else plan["a"]
        if override and override != "waiting":
            self.robot.halt()                     # safety stop: stop where the arm is
        else:
            back = self.robot.resume()            # after a safety stop: continue from where the arm stopped
            if back is not None:
                self.resync(back)
        if self.moving and not self.held and rec is not None:
            self.planner.advance(a, self.v_scale)
        s = self.planner.s
        i = self.path.idx(s)
        cmd = self.path.xyz[i]
        if not override or override == "waiting":
            if a > 0:
                ahead = self.path.xyz[self.path.idx(s + a * R.V_NOM * DT * self.v_scale)]   # one step ahead
                self.robot.command(ahead, a * R.V_NOM * self.v_scale)
            else:
                self.robot.hold(cmd)
        for k_, (lo, hi) in enumerate(self.dwells[:2]):      # close at the pick pause, open at the place pause
            if lo <= i <= hi:
                self.robot.gripper(k_ == 0)
        if err:
            self.running = False
        self.status = {"a": a, "override": override, "r": plan["r"], "fallback": plan["fallback"]}
        qq = list(q) if q is not None else [""] * 7
        tc = list(tcp) if tcp is not None else ["", "", ""]
        self.w.writerow([round(t, 6), self.tick, rec["anchor"] if rec else "",
                         round(t - rec["t"], 4) if rec else "", self.method, plan["a"], a, override,
                         plan["fallback"], plan["lam"], plan["dreq"],
                         rec.get("r", "") if rec else "", rec.get("r_u", "") if rec else "",
                         rec.get("r_s", "") if rec else "", int(rec["visible"]) if rec else "",
                         round(s, 2), round(s / self.path.length, 4), *np.round(cmd, 2), *tc, *qq,
                         round(sep_real, 2) if sep_real == sep_real else "",
                         round(sep_cmd, 2) if sep_cmd == sep_cmd else "", round(compute_ms, 2)])
        self.tick += 1
        if self.tick % 32 == 0:
            self.f.flush()

    def resync(self, tcp):
        """Move the planner's path position to the path point nearest the real TCP (within 40 mm of
        where it thought it was), so a safety stop never makes the arm jump."""
        i0 = int(self.planner.s / self.path.ds)
        cand = np.arange(i0 - 20, i0 + 21)
        d = np.linalg.norm(self.path.xyz[cand % self.path.S] - tcp, axis=1)
        self.planner.s = float(cand[int(np.argmin(d))] * self.path.ds)

    def close(self):
        self.running = False
        time.sleep(2 * DT)
        try:
            self.robot.hold(self.path.xyz[self.path.idx(self.planner.s)])
        finally:
            self.robot.close()
            self.f.close()
        self.info["ticks"] = self.tick
        self.info["cycles"] = round(self.planner.s / self.path.length, 3)
        return self.info
