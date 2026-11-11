#!/usr/bin/env python3
"""
H3: results of the live sessions (the planners on the real arm), with the replay's metrics.

Near-miss = the arm is moving (commanded speed > 0, or the measured gripper still moving faster
than 10 mm/s while it brakes) while the planar distance from its low links (from the logged joint
angles) to the person's elbow, wrist or hand is under 50 mm. The person's position at each step is
taken from the camera frames captured at that moment (read time minus the measured camera latency),
not from the delayed data the planner had.

Usage
  python3 s9_analyze_live.py logs/C_P01_* logs/C_P02_* [--out logs/live_results] [--cam-latency 0.114]
Writes per_session.csv, by_method.csv (per participant and condition too) and summary.json.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

import s7_replay as R
from s9_live import arm_separation, cached_path

FKP = ["elbow", "wrist", "hand"]
MOVING_MM_S = 10.0


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def person_mm(session, meta, t_query, cam_latency, H):
    """Elbow, wrist and hand (mm on the table) at the given times; NaN if not seen within 0.25 s."""
    side = meta["side"]
    fr = [f for f in csv.DictReader(open(os.path.join(session, "frames.csv"))) if f["phase"] != "setup"]
    t_cap = np.array([fnum(f["t_read_end"]) for f in fr]) - cam_latency          # when the frame was taken
    out = np.full((len(t_query), len(FKP), 2), np.nan)
    for j, k in enumerate(FKP):
        u = np.array([fnum(f[f"{side}_{k}_u"]) for f in fr]); v = np.array([fnum(f[f"{side}_{k}_v"]) for f in fr])
        ok = np.array([f[f"{side}_{k}_missing"] == "0" for f in fr]) & ~np.isnan(u)
        to = t_cap[ok]
        if len(to) < 2:
            continue
        i = np.searchsorted(to, t_query, side="right")
        good = (i > 0) & (i < len(to))
        ip, inx = np.clip(i - 1, 0, len(to) - 1), np.clip(i, 0, len(to) - 1)
        good &= (to[inx] - to[ip]) <= 0.25
        px = np.stack([np.interp(t_query, to, u[ok]), np.interp(t_query, to, v[ok])], -1)
        mm = R.to_mm(px / R.SIZE, H)
        out[good, j] = mm[good]
    return out


_PATHS = {}


def path_for(calib_file):
    """The pick-and-place loop for a calibration (built once: it runs IK for every point)."""
    if calib_file not in _PATHS:
        calib = json.load(open(calib_file))
        _PATHS[calib_file] = (cached_path(calib), R.homography(calib))
    return _PATHS[calib_file]


def session_metrics(session, cam_latency):
    meta = json.load(open(os.path.join(session, "meta.json")))
    live = meta.get("live", {})
    rows = list(csv.DictReader(open(os.path.join(session, "live.csv"))))
    rows = [r for r in rows if r["override"] != "waiting"]                  # from the first cue on
    if not rows:
        return None
    path, H = path_for(meta["calib_session"])
    t = np.array([fnum(r["t"]) for r in rows])
    a = np.array([fnum(r["a"]) for r in rows])
    s = np.array([fnum(r["s_mm"]) for r in rows])
    q = np.array([[fnum(r[f"q{i}"]) for i in range(1, 8)] for r in rows])
    tcp = np.array([[fnum(r[c]) for c in ("tcp_x", "tcp_y", "tcp_z")] for r in rows])
    over = np.array([r["override"] for r in rows])
    fb = np.array([fnum(r["fallback"]) for r in rows])
    hum = person_mm(session, meta, t, cam_latency, H)
    real = np.isfinite(q).all(1)
    sep = np.full(len(t), np.nan)
    for i in range(len(t)):
        if not np.isfinite(hum[i]).any():
            continue
        if real[i]:
            sep[i] = arm_separation(q[i], hum[i], path.table_z)
        else:                                                    # dry run: the commanded path position
            vis = np.isfinite(hum[i]).all(1)
            sep[i] = float(path.dist(hum[i][vis][None], np.array([s[i]]))[0].min())
    speed = np.r_[0, np.linalg.norm(np.diff(tcp, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)]
    moving = (a > 0) | (np.nan_to_num(speed) > MOVING_MM_S)
    close = np.nan_to_num(sep, nan=1e9) < R.D_SAFE
    viol = close & moving
    minutes = (t[-1] - t[0] + R.DT) / 60
    sep_m = np.where(moving, sep, np.nan)
    episodes = int(((np.diff(fb) == 1).sum() + fb[0]))
    safety = (over != "") & (over != "waiting")
    ldec = R.W_G * (1 - a) + R.W_W * (a == 0) + R.W_P * viol
    age = np.array([fnum(r["data_age_s"]) for r in rows])
    comp = np.array([fnum(r["compute_ms"]) for r in rows])
    return {"session": os.path.basename(os.path.normpath(session)), "participant": meta["participant"],
            "condition": meta["mode"], "method": live.get("method", ""), "robot": live.get("robot", ""),
            "minutes": round(minutes, 2), "cycles_per_min": round((s[-1] - s[0]) / path.length / minutes, 3),
            "mean_speed": round(float(a.mean()), 3), "wait_share": round(float((a == 0).mean()), 3),
            "fallback_share": round(float(fb.mean()), 3), "fallback_per_min": round(episodes / minutes, 2),
            "safety_stop_share": round(float(safety.mean()), 3),
            "safety_stops": {k: int((over == k).sum()) for k in sorted(set(over[safety]))},
            "min_sep_mm": round(float(np.nanmin(sep_m)), 1) if np.isfinite(sep_m).any() else None,
            "sep_p5_mm": round(float(np.nanpercentile(sep_m, 5)), 1) if np.isfinite(sep_m).any() else None,
            "near_misses_per_min": round(float(viol.sum() / minutes), 2),
            "near_miss_share": round(float(viol.mean()), 4), "decision_loss": round(float(ldec.mean()), 4),
            "person_seen_share": round(float(np.isfinite(sep).mean()), 3),
            "data_age_s_median": round(float(np.nanmedian(age)), 3),
            "compute_ms_p95": round(float(np.nanpercentile(comp, 95)), 1),
            "valid_reaches": meta.get("valid_reaches"), "_raw": (minutes, viol.sum(), (s[-1] - s[0]) / path.length,
                                                                 (a == 0).sum(), len(a))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", default="logs/live_results")
    ap.add_argument("--cam-latency", type=float, default=0.114, help="s, camera latency (s3_latency_test.py)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    per = [m for m in (session_metrics(s, a.cam_latency) for s in a.sessions) if m]
    keys = [k for k in per[0] if not k.startswith("_")]
    with open(os.path.join(a.out, "per_session.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for m in per:
            w.writerow([json.dumps(m[k]) if isinstance(m[k], dict) else m[k] for k in keys])
    # pooled per method (all), and per method x participant x condition: rates from summed counts
    groups = defaultdict(list)
    for m in per:
        for key in ((m["method"], "all", "all"), (m["method"], m["participant"], "all"),
                    (m["method"], "all", m["condition"]), (m["method"], m["participant"], m["condition"])):
            groups[key].append(m["_raw"])
    rows = []
    for (mth, p, c), raws in sorted(groups.items()):
        mins = sum(r[0] for r in raws); nm = sum(r[1] for r in raws); cyc = sum(r[2] for r in raws)
        wait = sum(r[3] for r in raws) / sum(r[4] for r in raws)
        rows.append([R.METHODS.get(mth, mth), p, c, len(raws), round(mins, 2), round(cyc / mins, 3),
                     round(nm / mins, 2), round(wait, 3)])
    with open(os.path.join(a.out, "by_method.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "participant", "condition", "sessions", "minutes", "cycles_per_min",
                    "near_misses_per_min", "wait_share"])
        w.writerows(rows)
    json.dump({"sessions": [{k: v for k, v in m.items() if not k.startswith("_")} for m in per],
               "cam_latency_s": a.cam_latency, "d_safe_mm": R.D_SAFE},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(f"{'method':<40}{'who':<6}{'cond':<6}{'min':>6}{'cyc/min':>9}{'near/min':>10}{'wait':>7}")
    for r in rows:
        if r[1] == "all" or r[2] == "all":
            print(f"{r[0]:<40}{r[1]:<6}{r[2]:<6}{r[4]:>6.1f}{r[5]:>9.2f}{r[6]:>10.2f}{r[7]:>7.2f}")
    print(f"\nSaved {a.out}/per_session.csv, by_method.csv, summary.json")


if __name__ == "__main__":
    main()
