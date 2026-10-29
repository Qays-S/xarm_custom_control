#!/usr/bin/env python3
"""
Blueprint step 17, the timing and jitter part: camera timing, detection time, keypoint jitter
and the end-to-end perception latency, from recordings that already exist plus one latency test.

From the E1-E4 recordings (no new recording needed):
  * frame interval and frame rate (time between frame captures)
  * camera read time (t_read_end - t_read_start) and detection time (t_detect_end - t_read_end,
    MediaPipe Pose + Hands on the 640-wide copy)
  * keypoint jitter: while the hand rests on a target in E1 ('hold' phase, at least 10 frames),
    the RMS distance of each keypoint from its mean position in that hold. This includes the
    small real movements of a resting hand, so it is an upper bound on detector jitter.
From the latency test (s3_latency_test.py):
  * camera latency: screen flip -> first frame that shows it (includes the screen's refresh)
From the forecasting results:
  * forecast time per window (GRU and STS-GCN, one CPU thread)
End-to-end estimate = camera latency + detection time + forecast time (medians, and p95s).

Usage
  python3 s3_timing_summary.py logs/E1_s01 ... logs/E4_s02 [--latency logs/latency01] [--out logs/timing]
"""
import argparse
import csv
import json
import os

import numpy as np

KP = ["shoulder", "elbow", "wrist", "hand", "tip"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def stats(a):
    a = np.asarray([x for x in a if x == x], float)
    return {"n": int(len(a)), "median": round(float(np.median(a)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "mean": round(float(a.mean()), 2), "max": round(float(a.max()), 2)} if len(a) else {"n": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--latency", help="folder written by s3_latency_test.py")
    ap.add_argument("--results", default="logs/forecast_results")
    ap.add_argument("--side", default="R")
    ap.add_argument("--out", default="logs/timing")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    interval, read, detect, jit, fails, nframes = [], [], [], {k: [] for k in KP}, 0, 0
    for d in a.sessions:
        fr = [r for r in csv.DictReader(open(os.path.join(d, "frames.csv"))) if r.get("phase", "") != "setup"]
        ev = list(csv.DictReader(open(os.path.join(d, "events.csv")))) if os.path.exists(os.path.join(d, "events.csv")) else []
        fails += sum(e["event"] == "read_failed" for e in ev)
        nframes += len(fr)
        tr = np.array([fnum(r["t_read_end"]) for r in fr])
        interval += list(np.diff(tr) * 1000)
        read += list((tr - np.array([fnum(r["t_read_start"]) for r in fr])) * 1000)
        detect += list((np.array([fnum(r["t_detect_end"]) for r in fr]) - tr) * 1000)
        meta = json.load(open(os.path.join(d, "meta.json")))
        if meta.get("mode") != "E1":
            continue
        ph = [r.get("phase", "") for r in fr]
        i = 0
        while i < len(ph):
            if ph[i] != "hold":
                i += 1; continue
            j = i
            while j < len(ph) and ph[j] == "hold":
                j += 1
            if j - i >= 10:
                for k in KP:
                    col = f"{a.side}_{k}_u"
                    if col not in fr[0]:
                        continue
                    P = np.array([[fnum(fr[q][col]), fnum(fr[q][f"{a.side}_{k}_v"])] for q in range(i + 2, j)
                                  if fr[q][f"{a.side}_{k}_missing"] == "0"])
                    if len(P) >= 6:
                        jit[k].append(float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())))
            i = j

    out = {"frames": nframes, "read_failures": fails,
           "frame_interval_ms": stats(interval), "fps_from_median_interval": round(1000 / np.median(interval), 2),
           "camera_read_ms": stats(read), "detection_ms": stats(detect),
           "jitter_rms_px_during_holds": {k: stats(v) for k, v in jit.items() if v}}
    cam = None
    if a.latency:
        cam = json.load(open(os.path.join(a.latency, "summary.json")))
        lat = [fnum(r["latency_ms"]) for r in csv.DictReader(open(os.path.join(a.latency, "flips.csv")))]
        out["camera_latency_ms"] = stats(lat)
        out["camera_latency_note"] = "screen flip to first camera frame showing it; includes the screen's refresh"
    fc = {}
    for name, path in (("GRU", "gru/gru_inference_time.json"), ("STS-GCN", "stsgcn/stsgcn_inference_time.json")):
        p = os.path.join(a.results, path)
        if os.path.exists(p):
            fc[name] = json.load(open(p))
    out["forecast_ms"] = fc
    if cam is not None:
        for name, t in fc.items():
            out[f"end_to_end_ms_{name}"] = {
                "median": round(out["camera_latency_ms"]["median"] + out["detection_ms"]["median"] + t["median_ms"], 1),
                "p95_sum": round(out["camera_latency_ms"]["p95"] + out["detection_ms"]["p95"] + t["p95_ms"], 1)}
    json.dump(out, open(os.path.join(a.out, "timing_summary.json"), "w"), indent=2)

    print(f"{nframes} frames, {fails} camera read failures")
    print(f"frame interval   median {out['frame_interval_ms']['median']:.1f} ms (p95 {out['frame_interval_ms']['p95']:.1f})  -> {out['fps_from_median_interval']} fps")
    print(f"camera read      median {out['camera_read_ms']['median']:.1f} ms (p95 {out['camera_read_ms']['p95']:.1f})")
    print(f"detection        median {out['detection_ms']['median']:.1f} ms (p95 {out['detection_ms']['p95']:.1f})")
    if cam is not None:
        c = out["camera_latency_ms"]
        print(f"camera latency   median {c['median']:.1f} ms (p95 {c['p95']:.1f}), {c['n']} flips detected")
    for name, t in fc.items():
        line = f"forecast {name:<8}median {t['median_ms']:.2f} ms (p95 {t['p95_ms']:.2f})"
        if cam is not None:
            e = out[f"end_to_end_ms_{name}"]
            line += f"   -> end-to-end about {e['median']:.0f} ms (p95 sum {e['p95_sum']:.0f})"
        print(line)
    print("\nkeypoint jitter while the hand rests on a target (E1 holds), RMS px:")
    for k, v in out["jitter_rms_px_during_holds"].items():
        print(f"  {k:<9} median {v['median']:.2f}  p95 {v['p95']:.2f}  ({v['n']} holds)")
    if cam is None:
        print("\n(no --latency folder given: run s3_latency_test.py for the camera latency and end-to-end total)")
    print(f"\nSaved {a.out}/timing_summary.json")


if __name__ == "__main__":
    main()
