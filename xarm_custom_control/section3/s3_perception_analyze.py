#!/usr/bin/env python3
"""
Section III perception statistics from one or more s3_perception_logger.py sessions:
frame rate, dropped frames, detection failures, per-keypoint visibility/missingness
(overall and inside occlusion segments), keypoint jitter during static holds, and
the processing and end-to-end latency distributions.

Usage
  python3 s3_perception_analyze.py logs/session01 [logs/session02 ...]
       [--side R] [--latency logs/latency01] [--out logs/perception_report]
--side picks the reaching arm (R or L) for the headline numbers; both are in the CSVs.
--latency adds the camera latency from s3_latency_test.py so the end-to-end latency
(camera + detection) can be reported.
"""
import argparse
import csv
import json
import os

import numpy as np

KP = ["shoulder", "elbow", "wrist", "hand"]


def load_session(d):
    with open(os.path.join(d, "frames.csv")) as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(d, "events.csv")) as f:
        ev = list(csv.DictReader(f))
    meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
    return rows, ev, meta


def segments(ev, name):
    segs, start = [], None
    for e in ev:
        if e["event"] == f"{name}_start":
            start = int(e["frame"])
        elif e["event"] == f"{name}_end" and start is not None:
            segs.append((start, int(e["frame"]))); start = None
    return segs


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--side", choices=["R", "L"], default="R")
    ap.add_argument("--latency")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.sessions[0])), "perception_report")
    os.makedirs(out, exist_ok=True)

    report = {"side": a.side, "sessions": {}}
    all_int, all_proc, all_jit = [], [], {k: [] for k in KP}
    tot = {"frames": 0, "pose_missing": 0, "dropped": 0, "miss": {k: 0 for k in KP},
           "occ_frames": 0, "occ_miss": {k: 0 for k in KP}}
    for d in a.sessions:
        rows, ev, meta = load_session(d)
        n = len(rows)
        if n < 3:
            print(f"{d}: too few frames, skipped"); continue
        t = np.array([float(r["t_read_end"]) for r in rows])
        dt = np.diff(t) * 1000
        period = float(np.median(dt))
        gaps = dt[dt > 1.5 * period]
        dropped = int(np.sum(np.round(gaps / period) - 1))
        proc = np.array([(float(r["t_detect_end"]) - float(r["t_read_end"])) * 1000 for r in rows])
        pose_miss = sum(r["pose_found"] == "0" for r in rows)
        miss = {k: sum(r[f"{a.side}_{k}_missing"] == "1" for r in rows) for k in KP}
        occ = segments(ev, "occlusion")
        occ_mask = np.zeros(n, bool)
        for s0, s1 in occ:
            occ_mask[s0:s1 + 1] = True
        occ_miss = {k: int(sum(rows[i][f"{a.side}_{k}_missing"] == "1" for i in np.flatnonzero(occ_mask))) for k in KP}
        # jitter: spread of each keypoint while the arm is held still
        jit = {}
        for s0, s1 in segments(ev, "static"):
            seg = rows[s0 + 3:s1 - 3]  # trim the key-press transitions
            for k in KP:
                uv = np.array([[float(r[f"{a.side}_{k}_u"]), float(r[f"{a.side}_{k}_v"])] for r in seg
                               if r[f"{a.side}_{k}_missing"] == "0"])
                if len(uv) >= 10:
                    sd = float(np.sqrt(np.mean(np.sum((uv - uv.mean(0)) ** 2, axis=1))))   # RMS radius (px)
                    step = float(np.sqrt(np.mean(np.sum(np.diff(uv, axis=0) ** 2, axis=1))))  # frame-to-frame RMS (px)
                    jit.setdefault(k, []).append({"frames": len(uv), "rms_radius_px": sd, "frame_to_frame_rms_px": step})
                    all_jit[k].append(sd)
        report["sessions"][d] = {
            "frames": n, "duration_s": float(t[-1] - t[0]), "fps_median": 1000 / period,
            "frame_interval_ms": {"median": period, "p95": pct(dt, 95), "max": float(dt.max())},
            "dropped_frames_est": dropped, "pose_not_found_frames": int(pose_miss),
            "missing_fraction": {k: miss[k] / n for k in KP},
            "occlusion_frames": int(occ_mask.sum()),
            "missing_fraction_in_occlusion": {k: (occ_miss[k] / occ_mask.sum()) if occ_mask.sum() else None for k in KP},
            "processing_ms": {"median": pct(proc, 50), "p95": pct(proc, 95), "p99": pct(proc, 99)},
            "jitter_static_px": jit, "meta": meta}
        all_int += dt.tolist(); all_proc += proc.tolist()
        tot["frames"] += n; tot["pose_missing"] += pose_miss; tot["dropped"] += dropped
        tot["occ_frames"] += int(occ_mask.sum())
        for k in KP:
            tot["miss"][k] += miss[k]; tot["occ_miss"][k] += occ_miss[k]
        # per-frame visibility table for the paper's statistics
        with open(os.path.join(out, f"visibility_{os.path.basename(os.path.normpath(d))}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "in_occlusion"] + [f"{a.side}_{k}_missing" for k in KP] + [f"{a.side}_{k}_conf" for k in KP])
            for i, r in enumerate(rows):
                w.writerow([i, int(occ_mask[i])] + [r[f"{a.side}_{k}_missing"] for k in KP] + [r[f"{a.side}_{k}_conf"] for k in KP])

    all_int, all_proc = np.array(all_int), np.array(all_proc)
    overall = {"frames": tot["frames"], "fps_median": 1000 / float(np.median(all_int)),
               "frame_interval_ms": {"median": pct(all_int, 50), "p95": pct(all_int, 95), "p99": pct(all_int, 99)},
               "dropped_frames_est": tot["dropped"], "dropped_fraction": tot["dropped"] / (tot["frames"] + tot["dropped"]),
               "detection_failure_fraction": tot["pose_missing"] / tot["frames"],
               "missing_fraction": {k: tot["miss"][k] / tot["frames"] for k in KP},
               "missing_fraction_in_occlusion": {k: (tot["occ_miss"][k] / tot["occ_frames"]) if tot["occ_frames"] else None for k in KP},
               "processing_ms": {"median": pct(all_proc, 50), "p95": pct(all_proc, 95), "p99": pct(all_proc, 99)},
               "jitter_static_rms_radius_px_median": {k: (float(np.median(v)) if v else None) for k, v in all_jit.items()}}
    if a.latency:
        lat = json.load(open(os.path.join(a.latency, "summary.json")))
        cam = lat.get("latency_ms", {})
        if cam:
            overall["camera_latency_ms"] = cam
            overall["end_to_end_latency_ms_estimate"] = {
                "median": cam["median"] + overall["processing_ms"]["median"],
                "p95": cam["p95"] + overall["processing_ms"]["p95"],
                "note": "camera+screen latency + detection time; percentiles added as a conservative bound"}
    report["overall"] = overall
    json.dump(report, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
        ax[0].hist(np.clip(all_int, 0, 5 * np.median(all_int)), bins=60)
        ax[0].set_xlabel("frame interval (ms)"); ax[0].set_ylabel("frames"); ax[0].set_title(f"fps ≈ {overall['fps_median']:.1f}")
        ax[1].hist(all_proc, bins=60, color="tab:orange")
        ax[1].set_xlabel("detection time per frame (ms)"); ax[1].set_title(f"median {overall['processing_ms']['median']:.0f} ms, p95 {overall['processing_ms']['p95']:.0f} ms")
        mf = [overall["missing_fraction"][k] * 100 for k in KP]
        ax[2].bar(KP, mf, label="all frames")
        if tot["occ_frames"]:
            ax[2].bar(KP, [overall["missing_fraction_in_occlusion"][k] * 100 for k in KP], alpha=.5, label="occlusion segments")
        ax[2].set_ylabel("% frames missing"); ax[2].set_title(f"{a.side} arm keypoint missingness"); ax[2].legend()
        fig.tight_layout(); fig.savefig(os.path.join(out, "perception_stats.png"), dpi=150)
    except ImportError:
        pass

    o = overall
    print(f"Frames {o['frames']}, fps {o['fps_median']:.1f}, dropped ~{o['dropped_frames_est']} ({o['dropped_fraction']*100:.1f}%)")
    print(f"Pose not found in {o['detection_failure_fraction']*100:.1f}% of frames")
    print("Missing (" + a.side + "): " + ", ".join(f"{k} {v*100:.1f}%" for k, v in o["missing_fraction"].items()))
    print(f"Detection time: median {o['processing_ms']['median']:.1f} ms, p95 {o['processing_ms']['p95']:.1f} ms")
    print("Static-hold jitter (RMS px): " + ", ".join(f"{k} {v:.2f}" if v is not None else f"{k} n/a"
                                                    for k, v in o["jitter_static_rms_radius_px_median"].items()))
    if "end_to_end_latency_ms_estimate" in o:
        print(f"End-to-end latency ≈ median {o['end_to_end_latency_ms_estimate']['median']:.0f} ms, "
              f"p95 {o['end_to_end_latency_ms_estimate']['p95']:.0f} ms")
    print(f"Outputs in {out}/")


if __name__ == "__main__":
    main()
