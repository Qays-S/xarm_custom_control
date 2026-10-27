#!/usr/bin/env python3
"""
Summarise recordings from s4_record_targets.py for blueprint Section IV.

Gives the "required outputs" that can be computed before any model is trained:
  * counts and durations by participant and condition (E1-E4)
  * frame rate, dropped/failed reads, and missing-keypoint rates (overall and
    during occlusion segments) for the reaching arm
  * reach timing: reaction time (cue -> leaving the start dot) and movement time
  * E3: time from leaving the start dot to the target being revealed
  * E4: switch cue time vs the PHYSICAL change onset, measured from the keypoints
    (first frame after the cue where the fingertip's direction of motion turns
    more than --turn-deg away from the original goal), kept separate as required
  * the dataset schema (every column written by the recorder)

Usage
  python3 s4_dataset_summary.py logs/E1_s01 logs/E2_s01 logs/E4_s01 [--out logs/summary]
"""
import argparse
import csv
import json
import math
import os

import numpy as np


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    frames = list(csv.DictReader(open(os.path.join(d, "frames.csv"))))
    events = list(csv.DictReader(open(os.path.join(d, "events.csv"))))
    reaches = list(csv.DictReader(open(os.path.join(d, "reaches.csv"))))
    # optional corrections.json written after a session: {"invalid_reaches": [..], "reason": "..", "calib": ".."}
    cp = os.path.join(d, "corrections.json")
    if os.path.exists(cp):
        corr = json.load(open(cp))
        for r in reaches:
            if int(r["reach"]) in corr.get("invalid_reaches", []):
                r["valid"] = "0"
                r["note"] = (r.get("note") or "") + "; corrected: " + corr.get("reason", "")
    return meta, frames, events, reaches


def effector_track(frames, side):
    t = np.array([fnum(f["t_detect_end"]) for f in frames])
    xy = np.full((len(frames), 2), np.nan)
    for i, f in enumerate(frames):
        for k in (f"{side}_tip", f"{side}_hand"):
            if f[f"{k}_missing"] == "0":
                xy[i] = fnum(f[f"{k}_u"]), fnum(f[f"{k}_v"])
                break
    return t, xy


def switch_onset(t, xy, t_cue, old_px, turn_deg, win=1.5, smooth=3):
    """First time after t_cue at which the direction of motion is > turn_deg away from old goal."""
    m = (t >= t_cue - 0.2) & (t <= t_cue + win)
    tt, p = t[m], xy[m]
    ok = ~np.isnan(p[:, 0])
    tt, p = tt[ok], p[ok]
    if len(p) < smooth + 3:
        return None
    k = np.ones(smooth) / smooth
    ps = np.c_[np.convolve(p[:, 0], k, "valid"), np.convolve(p[:, 1], k, "valid")]
    ts = tt[smooth - 1:]
    for i in range(1, len(ps)):
        if ts[i] < t_cue:
            continue
        v = ps[i] - ps[i - 1]
        if np.linalg.norm(v) < 0.5:  # px per frame: not really moving
            continue
        g = np.array(old_px) - ps[i]
        ang = math.degrees(math.acos(np.clip(v @ g / (np.linalg.norm(v) * np.linalg.norm(g) + 1e-9), -1, 1)))
        if ang > turn_deg:
            return float(ts[i])
    return None


def q(a, p):
    a = [x for x in a if x == x]
    return round(float(np.percentile(a, p)), 3) if a else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", default="logs/summary")
    ap.add_argument("--turn-deg", type=float, default=45.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    by_cond, per_session, onset_rows, schema = {}, [], [], {}
    for d in a.sessions:
        meta, frames, events, reaches = load(d)
        side, mode = meta["side"], meta["mode"]
        T = {int(k): v for k, v in meta["targets"].items()}
        t, xy = effector_track(frames, side)
        dur = meta.get("duration_s") or (t[-1] - t[0] if len(t) else 0)
        fails = sum(e["event"] == "read_failed" for e in events)
        # occlusion mask
        occ = np.zeros(len(frames), bool)
        start = None
        for e in events:
            if e["event"] == "occlusion_start":
                start = int(e["frame"])
            elif e["event"] == "occlusion_end" and start is not None:
                occ[start:int(e["frame"]) + 1] = True
                start = None
        miss = {}
        for k in ("shoulder", "elbow", "wrist", "hand", "tip"):
            # only frames after the first touch: before it the participant may still be walking to the seat
            col = np.array([f[f"{side}_{k}_missing"] == "1" for f in frames if f.get("phase", "") != "setup"])
            miss[k] = {"all": round(float(col.mean()), 4) if len(col) else None,
                       "occlusion": round(float(np.array([f[f"{side}_{k}_missing"] == "1" for f in frames])[occ].mean()), 4)
                       if occ.any() else None}
        valid = [r for r in reaches if r["valid"] == "1"]
        rt = [fnum(r["t_depart"]) - fnum(r["t_cue"]) for r in valid]
        mt = [fnum(r["t_arrive"]) - fnum(r["t_depart"]) for r in valid]
        res = [fnum(r["t_resolve"]) - fnum(r["t_depart"]) for r in valid if r["t_resolve"]]
        for r in valid:
            if r["t_switch_cue"]:
                tc = fnum(r["t_switch_cue"])
                on = switch_onset(t, xy, tc, T[int(r["goal_cued"])]["px"], a.turn_deg)
                onset_rows.append({"session": d, "reach": r["reach"], "kind": "reversal" if r["switch_to"] == r["start"] else "new target",
                                   "t_switch_cue": tc, "t_physical_onset": on,
                                   "delay_s": round(on - tc, 3) if on is not None else None})
        s = {"session": d, "participant": meta["participant"], "condition": mode, "duration_s": round(dur, 1),
             "frames": len(frames), "fps": round(len(frames) / dur, 2) if dur else None, "read_failures": fails,
             "reaches_cued": len(reaches), "reaches_valid": len(valid),
             "arrivals_manual": sum(r["arrival"] == "manual" for r in valid),
             "timeouts": sum(r["arrival"] == "timeout" for r in reaches),
             "reaction_s_median": q(rt, 50), "movement_s_median": q(mt, 50),
             "e3_resolve_after_depart_s_median": q(res, 50) if res else None,
             "side": side, "missing_rate": miss, "robot_logged": os.path.exists(os.path.join(d, "robot.csv"))}
        per_session.append(s)
        c = by_cond.setdefault((meta["participant"], mode), {"sessions": 0, "duration_s": 0, "frames": 0, "reaches_valid": 0})
        c["sessions"] += 1; c["duration_s"] += dur; c["frames"] += len(frames); c["reaches_valid"] += len(valid)
        for fn in ("frames.csv", "events.csv", "reaches.csv"):
            schema[fn] = next(csv.reader(open(os.path.join(d, fn))))
        if os.path.exists(os.path.join(d, "robot.csv")):
            schema["robot.csv"] = next(csv.reader(open(os.path.join(d, "robot.csv"))))

    # ---- write
    with open(os.path.join(a.out, "counts_by_condition.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant", "condition", "sessions", "duration_min", "frames", "valid_reaches"])
        for (p, m), c in sorted(by_cond.items()):
            w.writerow([p, m, c["sessions"], round(c["duration_s"] / 60, 2), c["frames"], c["reaches_valid"]])
    if onset_rows:
        with open(os.path.join(a.out, "e4_switch_onsets.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, onset_rows[0].keys()); w.writeheader(); w.writerows(onset_rows)
    json.dump({"sessions": per_session, "schema": schema}, open(os.path.join(a.out, "summary.json"), "w"), indent=2)

    print(f"{'participant':<12}{'cond':<6}{'sessions':>9}{'minutes':>9}{'frames':>8}{'reaches':>9}")
    for (p, m), c in sorted(by_cond.items()):
        print(f"{p:<12}{m:<6}{c['sessions']:>9}{c['duration_s'] / 60:>9.1f}{c['frames']:>8}{c['reaches_valid']:>9}")
    for s in per_session:
        mr = ", ".join(f"{k} {v['all'] * 100:.1f}%" for k, v in s["missing_rate"].items() if v["all"] is not None)
        print(f"\n{s['session']} ({s['condition']}): {s['fps']} fps, {s['read_failures']} read failures, "
              f"{s['reaches_valid']}/{s['reaches_cued']} valid reaches ({s['arrivals_manual']} marked by hand), "
              f"reaction {s['reaction_s_median']} s, movement {s['movement_s_median']} s (medians)")
        print(f"  missing ({'right' if s['side'] == 'R' else 'left'} arm): {mr}")
    if onset_rows:
        d = [r["delay_s"] for r in onset_rows if r["delay_s"] is not None]
        print(f"\nE4: {len(onset_rows)} switches, physical onset found for {len(d)}; "
              f"cue-to-onset delay median {q(d, 50)} s (p90 {q(d, 90)} s)")
    print(f"\nOutputs in {a.out}/")


if __name__ == "__main__":
    main()
