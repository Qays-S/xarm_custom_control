#!/usr/bin/env python3
"""
Manually annotated keypoint subset: click the true shoulder, elbow, wrist and hand
on randomly sampled recorded frames, then score the detector against your clicks.

Annotate (frames are sampled evenly over time with a fixed seed; the list is saved
first so the sample is fixed before you look at detector output):
  python3 s3_annotate.py annotate logs/session01 --n 100 [--side R] [--seed 0]
    For each frame click, in order: shoulder, elbow, wrist, hand (the centre of the
    knuckles, matching the detector's hand point). Keys:
      n  this keypoint is not visible (occluded / out of view)
      b  go back one keypoint
      s  skip this frame (e.g. motion blur makes it impossible to judge)
      q  save and quit (re-running resumes where you stopped)
    Detector output is NOT drawn while you annotate, to avoid biasing your clicks.

Evaluate:
  python3 s3_annotate.py evaluate logs/session01 [--side R]
    Pixel error per keypoint, PCK at 5/10/20 px and at 5% of the image diagonal,
    and visibility agreement (detector missing a visible keypoint, or reporting
    one you marked not visible). Writes annotations_eval.csv and annotations_summary.json.
"""
import argparse
import csv
import json
import os

import numpy as np

KP = ["shoulder", "elbow", "wrist", "hand"]


def sample(n_frames, n, seed):
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, n_frames, n + 1).astype(int)
    return sorted(int(rng.integers(lo, max(lo + 1, hi))) for lo, hi in zip(edges[:-1], edges[1:]))


def annotate(a):
    import cv2
    with open(os.path.join(a.session, "frames.csv")) as f:
        n_frames = sum(1 for _ in f) - 1
    plan_p = os.path.join(a.session, "annotation_plan.json")
    if os.path.exists(plan_p):
        plan = json.load(open(plan_p))
    else:
        plan = {"frames": sample(n_frames, a.n, a.seed), "seed": a.seed, "side": a.side}
        json.dump(plan, open(plan_p, "w"), indent=2)
    ann_p = os.path.join(a.session, "annotations.json")
    ann = json.load(open(ann_p)) if os.path.exists(ann_p) else {}
    frames_dir = os.path.join(a.session, "frames")
    files = {int(os.path.splitext(x)[0]): x for x in os.listdir(frames_dir)}
    click = {"pt": None}

    def on_mouse(ev, x, y, fl, p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (x, y)
    win = "s3 annotate (click: shoulder, elbow, wrist, hand | n=not visible b=back s=skip q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    todo = [fr for fr in plan["frames"] if str(fr) not in ann]
    print(f"{len(plan['frames']) - len(todo)} done, {len(todo)} to go")
    for fr in todo:
        img = cv2.imread(os.path.join(frames_dir, files[fr]))
        cur, i = {}, 0
        while i < len(KP):
            view = img.copy()
            for k, v in cur.items():
                if v:
                    cv2.circle(view, tuple(int(c) for c in v), 4, (0, 255, 0), -1)
                    cv2.putText(view, k, (int(v[0]) + 5, int(v[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(view, f"frame {fr}: click {plan['side']} {KP[i].upper()}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win, view)
            k = cv2.waitKey(20) & 0xFF
            if click["pt"]:
                cur[KP[i]] = [float(click["pt"][0]), float(click["pt"][1])]; click["pt"] = None; i += 1
            elif k == ord("n"):
                cur[KP[i]] = None; i += 1
            elif k == ord("b") and i > 0:
                i -= 1; cur.pop(KP[i], None)
            elif k == ord("s"):
                cur = "skipped"; break
            elif k == ord("q"):
                json.dump(ann, open(ann_p, "w"), indent=1)
                cv2.destroyAllWindows()
                print(f"Saved {len(ann)} frames; re-run to continue.")
                return
        ann[str(fr)] = cur
        json.dump(ann, open(ann_p, "w"), indent=1)
    cv2.destroyAllWindows()
    print(f"All {len(ann)} sampled frames annotated. Now run: evaluate {a.session}")


def evaluate(a):
    ann = json.load(open(os.path.join(a.session, "annotations.json")))
    plan = json.load(open(os.path.join(a.session, "annotation_plan.json")))
    side = plan.get("side", a.side)
    with open(os.path.join(a.session, "frames.csv")) as f:
        rows = {int(r["frame"]): r for r in csv.DictReader(f)}
    meta_p = os.path.join(a.session, "meta.json")
    w, h = json.load(open(meta_p)).get("actual_size", [1280, 720]) if os.path.exists(meta_p) else (1280, 720)
    diag = float(np.hypot(w, h))
    out_rows, err = [], {k: [] for k in KP}
    vis = {k: {"vis_det": 0, "vis_missed": 0, "invis_det": 0, "invis_ok": 0} for k in KP}
    for fr, a_kp in ann.items():
        if a_kp == "skipped":
            continue
        r = rows[int(fr)]
        for k in KP:
            gt = a_kp.get(k)
            det_missing = r[f"{side}_{k}_missing"] == "1"
            du, dv = r[f"{side}_{k}_u"], r[f"{side}_{k}_v"]
            e = ""
            if gt is not None and not det_missing:
                e = float(np.hypot(float(du) - gt[0], float(dv) - gt[1])); err[k].append(e); vis[k]["vis_det"] += 1
            elif gt is not None:
                vis[k]["vis_missed"] += 1
            elif not det_missing:
                vis[k]["invis_det"] += 1
            else:
                vis[k]["invis_ok"] += 1
            out_rows.append([fr, k, "" if gt is None else gt[0], "" if gt is None else gt[1],
                             du, dv, r[f"{side}_{k}_conf"], int(det_missing), "" if e == "" else round(e, 2)])
    with open(os.path.join(a.session, "annotations_eval.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "keypoint", "gt_u", "gt_v", "det_u", "det_v", "det_conf", "det_missing", "error_px"])
        wr.writerows(out_rows)
    summary = {"side": side, "n_frames": sum(v != "skipped" for v in ann.values()),
               "n_skipped": sum(v == "skipped" for v in ann.values()), "image_diag_px": diag, "keypoints": {}}
    for k in KP:
        e = np.array(err[k])
        v = vis[k]
        summary["keypoints"][k] = {
            "n_scored": int(len(e)),
            "pixel_error": {} if not len(e) else {"mean": float(e.mean()), "median": float(np.median(e)),
                                                  "p90": float(np.percentile(e, 90)), "max": float(e.max())},
            "PCK": {} if not len(e) else {f"{t}px": float((e <= t).mean()) for t in (5, 10, 20)} | {
                "5pct_diag": float((e <= 0.05 * diag).mean())},
            "visibility": v,
            "miss_rate_when_visible": v["vis_missed"] / max(1, v["vis_det"] + v["vis_missed"]),
            "false_detection_rate_when_not_visible": v["invis_det"] / max(1, v["invis_det"] + v["invis_ok"])}
    json.dump(summary, open(os.path.join(a.session, "annotations_summary.json"), "w"), indent=2)
    for k, s in summary["keypoints"].items():
        pe = s["pixel_error"]
        print(f"{k:9s} n={s['n_scored']:3d}  " + (f"median {pe['median']:.1f}px  PCK@10px {s['PCK']['10px']*100:.0f}%  " if pe else "")
              + f"missed-when-visible {s['miss_rate_when_visible']*100:.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["annotate", "evaluate"])
    ap.add_argument("session")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--side", choices=["R", "L"], default="R")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (annotate if a.cmd == "annotate" else evaluate)(a)


if __name__ == "__main__":
    main()
