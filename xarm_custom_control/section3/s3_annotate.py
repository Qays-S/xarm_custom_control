#!/usr/bin/env python3
"""
Blueprint step 15 / Section III output "manually annotated keypoint subset":
click the true shoulder, elbow, wrist, hand and index fingertip on sampled recorded frames,
then score the detector (MediaPipe) against your clicks.

Annotate (one or more sessions; frames are sampled evenly over time with a fixed seed,
skipping the pre-start 'setup' frames, and the sample is saved before you start):
  python3 s3_annotate.py annotate logs/E1_s01 logs/E2_s01 ... --n 15 [--side R] [--repeat 0.1]

  For each frame click, in order (on the reaching arm):
    SHOULDER  centre of the shoulder joint (where the upper arm meets the body), not the shirt edge
    ELBOW     centre of the elbow joint (middle of the arm's width at the bend), not the bony tip
    WRIST     centre of the wrist joint (middle of its width, at the crease where the hand meets
              the forearm), not its side; with a watch on, click where the wrist bends, not the watch
    HAND      base knuckle of the middle finger (MediaPipe Hands landmark 9)
    TIP       tip of the index finger (landmark 8)
  A zoomed view follows the mouse (top-right) so you can place the click accurately.
  Keys:  n = this point is not visible   b = back one point   s = skip frame (too blurry to judge)
         q = save and quit (re-running resumes where you stopped)
  The detector's points are NOT drawn while you annotate, so they cannot bias your clicks.
  With --repeat 0.1, about 10% of the frames come back a second time at the end (you won't be
  told which); comparing your two clicks measures your own click consistency.

Evaluate (all sessions together and per session):
  python3 s3_annotate.py evaluate logs/E1_s01 logs/E2_s01 ... [--out logs/keypoint_eval]
    Pixel error per keypoint (median, mean, p90), PCK at 5/10/20 px and at 5% of the image
    diagonal, visibility agreement (detector missing a visible point, or reporting a hidden one),
    and your own click consistency on the repeated frames.
"""
import argparse
import csv
import json
import os

import numpy as np

KP = ["shoulder", "elbow", "wrist", "hand", "tip"]
BAR = 44   # height of the instruction bar above the image, px
HINT = {"shoulder": "centre of shoulder joint", "elbow": "centre of elbow joint",
        "wrist": "centre of wrist (where it bends)", "hand": "base knuckle of middle finger",
        "tip": "tip of index finger"}


def usable_frames(session):
    with open(os.path.join(session, "frames.csv")) as f:
        rows = list(csv.DictReader(f))
    files = set(int(os.path.splitext(x)[0]) for x in os.listdir(os.path.join(session, "frames")))
    return [int(r["frame"]) for r in rows if r.get("phase", "") != "setup" and int(r["frame"]) in files]


def sample(frames, n, seed):
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, len(frames), n + 1).astype(int)
    return sorted(frames[int(rng.integers(lo, max(lo + 1, hi)))] for lo, hi in zip(edges[:-1], edges[1:]) if lo < len(frames))


def kp_list(session):
    with open(os.path.join(session, "frames.csv")) as f:
        header = next(csv.reader(f))
    return [k for k in KP if any(h.endswith(f"_{k}_u") for h in header)]


def annotate(a):
    import cv2
    out = a.out
    os.makedirs(out, exist_ok=True)
    plan_p = os.path.join(out, "annotation_plan.json")
    if os.path.exists(plan_p):
        plan = json.load(open(plan_p))
        print(f"Using the saved sample in {plan_p}")
    else:
        rng = np.random.default_rng(a.seed)
        items = []
        for i, s in enumerate(a.sessions):
            items += [[s, fr, 0] for fr in sample(usable_frames(s), a.n, a.seed + i)]
        rep = [list(x[:2]) + [1] for x in
               [items[j] for j in rng.choice(len(items), int(round(a.repeat * len(items))), replace=False)]] if a.repeat > 0 else []
        order = [items[j] for j in rng.permutation(len(items))] + rep     # repeats come at the end, mixed
        plan = {"items": order, "seed": a.seed, "side": a.side, "n_per_session": a.n, "repeat": a.repeat}
        json.dump(plan, open(plan_p, "w"), indent=1)
    ann_p = os.path.join(out, "annotations.json")
    ann = json.load(open(ann_p)) if os.path.exists(ann_p) else {}
    side = plan["side"]

    state = {"pt": None, "mouse": (0, 0)}

    def on_mouse(ev, x, y, fl, p):
        state["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x, y)
    win = "s3 annotate  |  click the point named at the top   n=not visible  b=back  s=skip frame  q=save+quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    todo = [it for it in plan["items"] if f"{it[0]}|{it[1]}|{it[2]}" not in ann]
    total = len(plan["items"])
    print(f"{total - len(todo)} of {total} done, {len(todo)} to go")
    for n_done, (session, fr, rep) in enumerate(todo, start=total - len(todo) + 1):
        img = cv2.imread(os.path.join(session, "frames", f"{fr:06d}.jpg"))
        if img is None:
            continue
        kps = kp_list(session)
        cur, i = {}, 0
        H, W = img.shape[:2]
        while i < len(kps):
            view = img.copy()
            for k, v in cur.items():
                if v:
                    cv2.circle(view, tuple(int(c) for c in v), 4, (0, 255, 0), -1)
                    cv2.putText(view, k, (int(v[0]) + 6, int(v[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            # zoom box (x4) around the mouse, drawn top-right
            mx, my = state["mouse"]
            my -= BAR                     # mouse y is measured on the window, which has the 44 px text bar on top
            mx, my = int(np.clip(mx, 0, W - 1)), int(np.clip(my, 0, H - 1))
            r = 30
            x0, y0 = int(np.clip(mx - r, 0, W - 2 * r)), int(np.clip(my - r, 0, H - 2 * r))
            z = cv2.resize(img[y0:y0 + 2 * r, x0:x0 + 2 * r], (8 * r, 8 * r), interpolation=cv2.INTER_NEAREST)
            c = 4 * r + (mx - x0 - r) * 4, 4 * r + (my - y0 - r) * 4
            cv2.drawMarker(z, (int(c[0]), int(c[1])), (0, 255, 255), cv2.MARKER_CROSS, 24, 1)
            zx = 10 if (mx > W - 30 - 8 * r and my < 30 + 8 * r) else W - 10 - 8 * r   # move the box out of the way
            view[10:10 + 8 * r, zx:zx + 8 * r] = z
            cv2.rectangle(view, (zx - 1, 9), (zx + 8 * r + 1, 11 + 8 * r), (255, 255, 255), 1)
            bar = np.full((BAR, W, 3), 30, np.uint8)
            cv2.putText(bar, f"[{n_done}/{total}]  click {('RIGHT' if side == 'R' else 'LEFT')} {kps[i].upper()}: {HINT[kps[i]]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(win, np.vstack([bar, view]))
            k = cv2.waitKey(20) & 0xFF
            if state["pt"]:
                x, y = state["pt"]; state["pt"] = None
                y -= bar.shape[0]
                if 0 <= y < H:
                    cur[kps[i]] = [float(x), float(y)]; i += 1
            elif k == ord("n"):
                cur[kps[i]] = None; i += 1
            elif k == ord("b") and i > 0:
                i -= 1; cur.pop(kps[i], None)
            elif k == ord("s"):
                cur = "skipped"; break
            elif k == ord("q"):
                json.dump(ann, open(ann_p, "w"), indent=1)
                cv2.destroyAllWindows()
                print(f"Saved {len(ann)} of {total}; re-run the same command to continue.")
                return
        ann[f"{session}|{fr}|{rep}"] = cur
        json.dump(ann, open(ann_p, "w"), indent=1)
    cv2.destroyAllWindows()
    print(f"All {len(ann)} frames annotated. Now run the same command with 'evaluate' instead of 'annotate'.")


def score(pairs, kps, diag):
    out = {}
    for k in kps:
        e = np.array([p["err"] for p in pairs if p["kp"] == k and p["err"] is not None])
        v = {"vis_det": 0, "vis_missed": 0, "invis_det": 0, "invis_ok": 0}
        for p in pairs:
            if p["kp"] != k:
                continue
            v["vis_det" if (p["gt"] and not p["miss"]) else "vis_missed" if p["gt"] else "invis_det" if not p["miss"] else "invis_ok"] += 1
        out[k] = {"n_scored": int(len(e)),
                  "pixel_error": {} if not len(e) else {"median": float(np.median(e)), "mean": float(e.mean()),
                                                        "p90": float(np.percentile(e, 90)), "max": float(e.max())},
                  "PCK": {} if not len(e) else {**{f"{t}px": float((e <= t).mean()) for t in (5, 10, 20)},
                                                "5pct_diag": float((e <= 0.05 * diag).mean())},
                  "visibility": v,
                  "miss_rate_when_visible": v["vis_missed"] / max(1, v["vis_det"] + v["vis_missed"]),
                  "false_detection_rate_when_not_visible": v["invis_det"] / max(1, v["invis_det"] + v["invis_ok"])}
    return out


def evaluate(a):
    ann = json.load(open(os.path.join(a.out, "annotations.json")))
    plan = json.load(open(os.path.join(a.out, "annotation_plan.json")))
    side = plan.get("side", a.side)
    rows, sizes = {}, {}
    for s in set(k.split("|")[0] for k in ann):
        with open(os.path.join(s, "frames.csv")) as f:
            rows[s] = {int(r["frame"]): r for r in csv.DictReader(f)}
        mp = os.path.join(s, "meta.json")
        sizes[s] = json.load(open(mp)).get("actual_size", [1280, 720]) if os.path.exists(mp) else [1280, 720]
    w, h = next(iter(sizes.values()))
    diag = float(np.hypot(w, h))
    pairs, csv_rows, first = [], [], {}
    for key, a_kp in ann.items():
        s, fr, rep = key.split("|"); fr = int(fr)
        if a_kp == "skipped":
            continue
        if rep == "1":
            first.setdefault((s, fr), {})["repeat"] = a_kp
            continue
        first.setdefault((s, fr), {})["first"] = a_kp
        r = rows[s][fr]
        for k in a_kp:
            gt = a_kp.get(k)
            miss = r[f"{side}_{k}_missing"] == "1"
            du, dv = r[f"{side}_{k}_u"], r[f"{side}_{k}_v"]
            err = float(np.hypot(float(du) - gt[0], float(dv) - gt[1])) if (gt and not miss) else None
            pairs.append({"session": s, "kp": k, "gt": gt, "miss": miss, "err": err, "phase": r.get("phase", "")})
            csv_rows.append([s, fr, k, "" if not gt else gt[0], "" if not gt else gt[1], du, dv,
                             r[f"{side}_{k}_conf"], int(miss), "" if err is None else round(err, 2), r.get("phase", "")])
    kps = [k for k in KP if any(p["kp"] == k for p in pairs)]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "annotations_eval.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["session", "frame", "keypoint", "gt_u", "gt_v", "det_u", "det_v", "det_conf", "det_missing", "error_px", "phase"])
        wr.writerows(csv_rows)
    # your own click consistency on repeated frames
    cons = {k: [] for k in kps}
    for d in first.values():
        if "first" in d and "repeat" in d and d["repeat"] != "skipped":
            for k in kps:
                p, q = d["first"].get(k), d["repeat"].get(k)
                if p and q:
                    cons[k].append(float(np.hypot(p[0] - q[0], p[1] - q[1])))
    summary = {"side": side, "n_frames": sum(1 for d in first.values() if "first" in d),
               "n_skipped": sum(v == "skipped" for v in ann.values()), "image_diag_px": diag,
               "all": score(pairs, kps, diag),
               "by_phase": {ph: score([p for p in pairs if p["phase"] == ph], kps, diag) for ph in ("cued", "moving", "hold")},
               "by_session": {s: score([p for p in pairs if p["session"] == s], kps, diag) for s in sorted(rows)},
               "annotator_consistency_px": {k: ({"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))} if v else {})
                                            for k, v in cons.items()}}
    json.dump(summary, open(os.path.join(a.out, "annotations_summary.json"), "w"), indent=2)
    print(f"{summary['n_frames']} frames scored ({summary['n_skipped']} skipped), {len(rows)} sessions\n")
    print("Error = distance in pixels between your click and MediaPipe's point (1280x720 image)")
    print(f"{'keypoint':<10}{'n':>5}{'median':>9}{'p90':>8}{'PCK@5':>8}{'PCK@10':>8}{'PCK@20':>8}{'missed':>9}{'you vs you':>12}")
    for k in kps:
        s = summary["all"][k]; pe = s["pixel_error"]; c = summary["annotator_consistency_px"][k]
        print(f"{k:<10}{s['n_scored']:>5}" + (f"{pe['median']:>9.1f}{pe['p90']:>8.1f}{s['PCK']['5px']*100:>7.0f}%{s['PCK']['10px']*100:>7.0f}%{s['PCK']['20px']*100:>7.0f}%"
                                             if pe else " " * 46) + f"{s['miss_rate_when_visible']*100:>8.0f}%"
              + (f"{c['median']:>10.1f}px" if c else f"{'':>12}"))
    print("\nmedian error (px) while the hand is moving vs holding still:")
    for k in kps:
        m, hd = summary["by_phase"]["moving"][k]["pixel_error"], summary["by_phase"]["hold"][k]["pixel_error"]
        print(f"  {k:<10} moving {m.get('median', float('nan')):5.1f}   hold {hd.get('median', float('nan')):5.1f}")
    print(f"\nSaved {a.out}/annotations_summary.json and annotations_eval.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["annotate", "evaluate"])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--n", type=int, default=15, help="frames per session")
    ap.add_argument("--side", choices=["R", "L"], default="R")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=float, default=0.1, help="share of frames shown twice, to measure your own consistency")
    ap.add_argument("--out", default="logs/keypoint_eval")
    a = ap.parse_args()
    (annotate if a.cmd == "annotate" else evaluate)(a)


if __name__ == "__main__":
    main()
