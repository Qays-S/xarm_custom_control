#!/usr/bin/env python3
"""
Steps 25-27: turn the E1-E4 recordings into 16 Hz keypoint tracks and a frozen split manifest,
plus a leakage audit.

Split (one participant, so by session and time, never by random windows)
  round 1 (*_s01) -> train;  round 2 (*_s02) split by reach into thirds: val, cal, test
  E5 -> its own held-out set "e5" (prepare it into its own folder, see below)

Usage
  python3 s5_prepare.py logs/E1_s01 logs/E2_s01 logs/E3_s01 logs/E4_s01 \\
                        logs/E1_s02 logs/E2_s02 logs/E3_s02 logs/E4_s02
  python3 s5_prepare.py logs/E5_s01 logs/E5_s02 --out logs/prepared_e5
"""
import argparse
import csv
import hashlib
import json
import os

import numpy as np

KP = ["shoulder", "elbow", "wrist", "hand", "tip"]


def fnum(x):
    """float(x), or NaN for blanks."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def sha(path):
    """SHA-256 of a file, saved in the manifest so we can prove the data never changed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resample_session(d, dt, max_gap):
    """Resample one session onto a fixed time grid (every dt s), using the frame capture time.

    Returns per step: t (s), y (T,5,2) u/W and v/H (NaN = missing), conf, miss (1 = missing),
    reach (0 = before the first cue), goal (dot shown on screen) and phase (cued / moving / hold).
    """
    meta = json.load(open(os.path.join(d, "meta.json")))
    side = meta["side"]
    W, H = meta.get("actual_size", [1280, 720])
    frames = [f for f in csv.DictReader(open(os.path.join(d, "frames.csv"))) if f["phase"] != "setup"]
    reaches = list(csv.DictReader(open(os.path.join(d, "reaches.csv"))))
    tr = np.array([fnum(f["t_read_end"]) for f in frames])
    tg = np.arange(tr[0], tr[-1], dt)
    T = len(tg)
    y = np.full((T, len(KP), 2), np.nan)
    conf = np.zeros((T, len(KP)))
    miss = np.ones((T, len(KP)), np.uint8)
    for j, k in enumerate(KP):
        u = np.array([fnum(f[f"{side}_{k}_u"]) for f in frames]) / W     # normalise by image size
        v = np.array([fnum(f[f"{side}_{k}_v"]) for f in frames]) / H
        c = np.array([fnum(f[f"{side}_{k}_conf"]) for f in frames])
        obs = np.array([f[f"{side}_{k}_missing"] == "0" for f in frames]) & ~np.isnan(u)
        conf[:, j] = np.interp(tg, tr, np.nan_to_num(c))
        to = tr[obs]
        if len(to) < 2:
            continue
        # Interpolate between two detections only if they are at most max_gap s apart;
        # otherwise the step stays missing. Nothing is filled across long gaps.
        i = np.searchsorted(to, tg, side="right")          # first observed sample after tg
        prev_ok = i > 0
        nxt_ok = i < len(to)
        ip, inx = np.clip(i - 1, 0, len(to) - 1), np.clip(i, 0, len(to) - 1)
        exact = prev_ok & (to[ip] == tg)
        gap = to[inx] - to[ip]
        ok = exact | (prev_ok & nxt_ok & (gap <= max_gap))
        y[ok, j, 0] = np.interp(tg[ok], to, u[obs])
        y[ok, j, 1] = np.interp(tg[ok], to, v[obs])
        miss[ok, j] = 0
    # per-step labels from the nearest preceding raw frame
    ir = np.clip(np.searchsorted(tr, tg, side="right") - 1, 0, len(tr) - 1)
    goal = np.array([int(fnum(frames[i]["goal"])) if frames[i]["goal"] else -1 for i in ir])
    phase = np.array([frames[i]["phase"] for i in ir])
    cues = np.array([fnum(r["t_cue"]) for r in reaches])
    reach = np.searchsorted(cues, tg, side="right")        # 0 before the first cue, then 1..n
    return dict(t=tg, y=y, conf=conf, miss=miss, reach=reach, goal=goal, phase=phase,
                W=W, H=H, side=side, mode=meta["mode"], n_reaches=len(reaches),
                calib=meta.get("calib_session"), participant=meta["participant"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", default="logs/prepared")
    ap.add_argument("--dt", type=float, default=0.0625)       # 16 Hz, close to the camera's 14-18 fps
    ap.add_argument("--max-gap", type=float, default=0.25)
    ap.add_argument("--history-s", type=float, default=1.0, help="L in seconds (used to size the split guard)")
    ap.add_argument("--horizon-s", type=float, default=1.5, help="longest K in seconds (used to size the split guard)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L, K = round(a.history_s / a.dt), round(a.horizon_s / a.dt)

    manifest = {"version": 1, "dt": a.dt, "max_gap_s": a.max_gap, "history_steps": L, "max_horizon_steps": K,
                "keypoints": KP, "coords": "u/W, v/H (normalised image coordinates)",
                "rule": "round 1 (_s01) = train; round 2 (_s02) split by reach into thirds: val, cal, test; "
                        "a window is used only if history and future lie inside one block",
                "sessions": {}}
    audit = []
    for d in a.sessions:
        name = os.path.basename(os.path.normpath(d))
        s = resample_session(d, a.dt, a.max_gap)
        np.savez_compressed(os.path.join(a.out, name + ".npz"), **{k: s[k] for k in
                            ("t", "y", "conf", "miss", "reach", "goal", "phase")})
        T, n = len(s["t"]), s["n_reaches"]
        # block per step
        if s["mode"] == "E5":
            blocks = [("e5", 0, n)]
        elif name.endswith("_s01"):
            blocks = [("train", 0, n)]
        else:
            b1, b2 = round(n / 3), round(2 * n / 3)
            blocks = [("val", 0, b1), ("cal", b1, b2), ("test", b2, n)]
        step_set = np.empty(T, object)
        for setname, lo, hi in blocks:          # reaches lo+1..hi (reach 0 = before first cue goes to the first block)
            m = (s["reach"] > lo) & (s["reach"] <= hi)
            if lo == 0:
                m |= s["reach"] == 0
            step_set[m] = setname
        # a window anchored at step i uses steps i-L+1 .. i+K; keep it only if all lie in the same block
        usable = {}
        for setname, _, _ in blocks:
            inb = (step_set == setname).astype(int)
            c = np.concatenate([[0], np.cumsum(inb)])
            idx = np.arange(L - 1, T - K)
            full = (c[idx + K + 1] - c[idx - L + 1]) == (L + K)
            usable[setname] = idx[full].tolist()
        manifest["sessions"][name] = {
            "path": d, "condition": s["mode"], "participant": s["participant"], "calib": s["calib"],
            "image_size": [s["W"], s["H"]], "side": s["side"], "steps": T,
            "duration_s": round(float(s["t"][-1] - s["t"][0]), 1),
            "frames_csv_sha256": sha(os.path.join(d, "frames.csv")),
            "blocks": [{"set": b, "reaches": [lo + 1, hi]} for b, lo, hi in blocks],
            "anchors": {k: [v[0], v[-1], len(v)] if v else [None, None, 0] for k, v in usable.items()},
            "anchor_lists": usable,
            "missing_rate": {k: round(float(s["miss"][:, j].mean()), 4) for j, k in enumerate(KP)},
        }
        # leakage audit: windows from different sets of the same recording must not share any step
        spans = {k: set() for k in usable}
        for k, v in usable.items():
            for i in v:
                spans[k].update(range(i - L + 1, i + K + 1))
        names = list(spans)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ov = len(spans[names[i]] & spans[names[j]])
                audit.append((name, names[i], names[j], ov))

    json.dump(manifest, open(os.path.join(a.out, "split_manifest.json"), "w"), indent=1)
    with open(os.path.join(a.out, "leakage_audit.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "set_a", "set_b", "shared_steps"])
        w.writerows(audit)

    tot = {}
    print(f"dt = {a.dt} s, history L = {L} steps, longest horizon K = {K} steps\n")
    print(f"{'session':<9}{'cond':<5}{'steps':>7}  windows per set")
    for name, m in manifest["sessions"].items():
        parts = ", ".join(f"{k} {v[2]}" for k, v in m["anchors"].items())
        for k, v in m["anchors"].items():
            tot[k] = tot.get(k, 0) + v[2]
        print(f"{name:<9}{m['condition']:<5}{m['steps']:>7}  {parts}")
    print("\nTotal windows: " + ", ".join(f"{k} {v}" for k, v in tot.items()))
    bad = [r for r in audit if r[3] > 0]
    print("Leakage audit: " + ("PASS (no window shares a step with a window from another set)" if not bad
                               else f"FAIL {bad}"))
    print(f"Outputs in {a.out}/")


if __name__ == "__main__":
    main()
