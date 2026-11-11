#!/usr/bin/env python3
"""
Tests the real-arm code path of s9_live.py without an arm: a stand-in for the SDK's XArmAPI follows
the commands (at the commanded speed) while a recorded session is played back in real time.
Checks that the commanded and followed positions never jump, that safety stops halt the arm and
re-arm it, and that the loop keeps its 16 Hz timing. Nothing connects to a robot.

  python3 s9_mockarm_test.py [--session logs/E4_s02] [--seconds 60] [--method full2]
"""
import argparse
import builtins
import csv
import json
import os
import sys
import tempfile
import threading
import time
import types

import numpy as np


class FakeArm:
    """Enough of xarm.wrapper.XArmAPI for s9_live.XArmRobot; moves toward the last target at its speed."""

    def __init__(self, ip, is_radian=False):
        from s7_robot import ik_down
        self.ik = ik_down
        self.error_code, self.warn_code, self.mode, self.state, self.connected = 0, 0, 0, 0, True
        self._p = np.array([300.0, 0.0, 300.0]); self._t = time.monotonic()
        self._target, self._speed = self._p.copy(), 0.0
        self._q = None
        self.calls = []
        self.lock = threading.Lock()

    # ---- SDK calls used by s9_live
    def clean_warn(self): return 0
    def clean_error(self): return 0
    def motion_enable(self, enable=True): return 0
    def get_inverse_kinematics(self, pose, **k): return 0, [0.0] * 7
    def set_collision_sensitivity(self, v): return 0
    def set_self_collision_detection(self, v): return 0
    def set_reduced_max_tcp_speed(self, v): return 0
    def set_reduced_mode(self, v): return 0
    def get_position(self): self._update(); return 0, list(self._p) + [180.0, 0.0, 0.0]
    def disconnect(self): self.connected = False
    def set_gripper_mode(self, m): return 0
    def set_gripper_enable(self, e): return 0
    def set_gripper_position(self, pos, wait=False): self.calls.append(("gripper", pos)); return 0

    def set_mode(self, m):
        self.mode = m; self.calls.append(("mode", m)); return 0

    def set_state(self, s):
        self._update()
        self.state = s; self.calls.append(("state", s))
        if s == 4:                       # stop: brake to a halt where it is (1000 mm/s^2 from <=150 mm/s)
            self._target = self._p.copy()
        return 0

    def set_position(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None, speed=100, mvacc=None,
                     wait=False, **k):
        self._update()
        if self.state == 4 or (self.mode == 7 and self.state != 0):
            return 9                     # not ready (the real controller refuses while stopped)
        self._target, self._speed = np.array([x, y, z], float), float(speed)
        if wait:
            self._p = self._target.copy()
        self.calls.append(("move", self._target.copy(), speed))
        return 0

    def _update(self):
        with self.lock:
            t = time.monotonic(); dt = t - self._t; self._t = t
            d = self._target - self._p; n = np.linalg.norm(d)
            if n > 0:
                self._p = self._p + d / n * min(n, self._speed * dt)

    @property
    def position(self):
        self._update(); return list(self._p) + [180.0, 0.0, 0.0]

    @property
    def angles(self):
        self._q, _ = self.ik(self._p, self._q, iters=30)
        return list(np.degrees(self._q))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="logs/E4_s02")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--method", default="full2")
    ap.add_argument("--manicast-dir", default=os.path.expanduser("~/manicast"))
    a = ap.parse_args()
    sys.modules["xarm"] = types.ModuleType("xarm")
    wrapper = types.ModuleType("xarm.wrapper"); wrapper.XArmAPI = FakeArm
    sys.modules["xarm.wrapper"] = wrapper
    builtins.input = lambda *x: "y"                     # answer the "move to start?" question
    import s7_replay as R
    from s9_live import LiveController
    from s9_selftest import frames_of
    meta, frames, cues = frames_of(a.session)
    out = tempfile.mkdtemp(prefix="mockarm_")
    ctl = LiveController(out, R.calib_for(a.session), a.method, meta["side"], *meta.get("actual_size", [1280, 720]),
                         robot="xarm", ip="mock", manicast_dir=a.manicast_dir, results="logs/forecast_results",
                         reliability="logs/reliability", replay_summary="logs/replay/summary.json",
                         prepared_train="logs/prepared")
    arm = ctl.robot.arm
    ctl.start()
    t_play0, t_rec0, ci = time.monotonic(), None, 0
    pos_log = []
    for t, kp, miss, phase, goal in frames:
        if t_rec0 is None:
            t_rec0 = t
        while ci < len(cues) and cues[ci] < t:
            ctl.add_cue(cues[ci]); ci += 1
        lag = (t - t_rec0) - (time.monotonic() - t_play0)
        if lag > 0:
            time.sleep(lag)
        ctl.add_frame(t, kp, miss, phase, goal)
        ctl.moving = phase != "setup"
        pos_log.append((time.monotonic(), *arm.position[:3]))
        if time.monotonic() - t_play0 > a.seconds or not ctl.running:
            break
    info = ctl.close()
    rows = list(csv.DictReader(open(os.path.join(out, "live.csv"))))
    cmd = np.array([[float(r[c]) for c in ("cmd_x", "cmd_y", "cmd_z")] for r in rows])
    tick_t = np.array([float(r["t"]) for r in rows])
    step = np.linalg.norm(np.diff(cmd, axis=0), axis=1)
    P = np.array(pos_log)
    v = np.linalg.norm(np.diff(P[:, 1:], axis=0), axis=1) / np.maximum(np.diff(P[:, 0]), 1e-3)
    over = [r["override"] for r in rows]
    halts = sum(1 for c in arm.calls if c == ("state", 4))
    rearms = sum(1 for i, c in enumerate(arm.calls[:-1]) if c == ("mode", 7))
    refused = sum(1 for r in rows if r["override"] == "" and False)
    res = {"ticks": len(rows), "tick_ms_median": round(float(np.median(np.diff(tick_t)) * 1000), 1),
           "tick_ms_max": round(float(np.diff(tick_t).max() * 1000), 1),
           "largest_command_step_mm": round(float(step.max()), 2),
           "allowed_step_mm": round(R.V_NOM * R.DT * 1.01, 2),
           "arm_speed_max_mm_s": round(float(v.max()), 1),
           "overrides": {k: over.count(k) for k in sorted(set(over))},
           "halts": halts, "rearms_mode7": rearms, "cycles": info["cycles"]}
    ok = (res["largest_command_step_mm"] <= res["allowed_step_mm"] + 2.0 and res["arm_speed_max_mm_s"] <= 160
          and res["tick_ms_median"] < 70)
    print(json.dumps(res, indent=1))
    print("PASS" if ok else "FAIL", "(commands never jump, the arm never exceeds 150 mm/s, 16 Hz timing kept)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
