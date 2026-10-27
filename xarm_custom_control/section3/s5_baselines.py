#!/usr/bin/env python3
"""
Blueprint Section V, step 28 (first half): non-learned baselines, scored on D_val only.

  current pose      the arm stays where it is now (the reference every forecaster must beat)
  constant velocity the arm keeps the velocity of the last n steps; n is chosen on D_val

Scores are ADE and FDE in pixels of the 1280x720 image, at 0.25, 0.5, 1.0 and 1.5 s, for the
elbow, wrist and hand and their mean. Every model is scored on the same windows: those whose
current pose and whole future up to that horizon were observed. The test set is not touched.

Usage
  python3 s5_baselines.py                 # after s5_prepare.py
  python3 s5_baselines.py --set train     # e.g. to compare with val
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from s5_data import FORECAST_KP, HORIZONS_S, ade_fde_px, eval_mask, load, windows


def current_pose(w, K):
    return np.repeat(w["hist"][:, -1:], K, axis=1)


def const_vel(w, K, n):
    """Velocity from the last n steps; if y_{t-n} is missing, use the largest observed n' < n; else zero."""
    h = w["hist"]
    cur = h[:, -1]
    v = np.zeros_like(cur)
    done = np.zeros(cur.shape[:2], bool)
    for nn in range(n, 0, -1):
        past = h[:, -1 - nn]
        ok = ~np.isnan(past[..., 0]) & ~done
        v[ok] = (cur[ok] - past[ok]) / nn
        done |= ok
    steps = np.arange(1, K + 1)[None, :, None, None]
    return cur[:, None] + steps * v[:, None]


def score(pred, w, ks):
    rows = {}
    for k in ks:
        m = eval_mask(w, k)
        a, f = ade_fde_px(pred[m], {"fut": w["fut"][m]}, k)
        rows[k] = dict(n=int(m.sum()), ade=a.mean(0), fde=f.mean(0), ade_all=a.mean(), fde_all=f.mean())
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", default="logs/prepared")
    ap.add_argument("--set", default="val", choices=["train", "val"])
    ap.add_argument("--out", default="logs/forecast_results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m, sess = load(a.prepared)
    dt = m["dt"]
    ks = [round(h / dt) for h in HORIZONS_S]
    K = max(ks)
    w = windows(m, sess, a.set)
    print(f"{a.set}: {len(w['cond'])} windows, dt {dt} s, horizons {HORIZONS_S} s = {ks} steps\n")

    # choose n for constant velocity on this set (val) by mean ADE over all horizons
    cand = {}
    for n in (1, 2, 3, 4, 6, 8):
        r = score(const_vel(w, K, n), w, ks)
        cand[n] = np.mean([r[k]["ade_all"] for k in ks])
    best = min(cand, key=cand.get)
    print("constant velocity, mean ADE (px) by velocity window n: "
          + ", ".join(f"n={n}: {v:.1f}" for n, v in cand.items()) + f"  -> n = {best}\n")

    models = {"current pose": lambda: current_pose(w, K), f"constant velocity (n={best})": lambda: const_vel(w, K, best)}
    results, timing = {}, {}
    for name, f in models.items():
        t0 = time.perf_counter()
        p = f()
        timing[name] = (time.perf_counter() - t0) / len(p) * 1e6
        results[name] = (p, score(p, w, ks))

    hdr = f"{'model':<28}{'horizon':>8}{'n':>7}" + "".join(f"{k + ' ADE':>12}" for k in FORECAST_KP) + f"{'mean ADE':>10}{'mean FDE':>10}"
    print(hdr)
    rows = []
    for name, (_, r) in results.items():
        for h, k in zip(HORIZONS_S, ks):
            x = r[k]
            print(f"{name:<28}{h:>7}s{x['n']:>7}" + "".join(f"{v:>12.1f}" for v in x["ade"]) + f"{x['ade_all']:>10.1f}{x['fde_all']:>10.1f}")
            rows.append([a.set, name, h, x["n"]] + [round(float(v), 2) for v in x["ade"]] + [round(float(v), 2) for v in x["fde"]]
                        + [round(float(x["ade_all"]), 2), round(float(x["fde_all"]), 2)])
    # by condition and by phase, hand only, 1.0 s
    k1 = ks[HORIZONS_S.index(1.0)]
    print(f"\nHand ADE (px) at 1.0 s by condition and phase of the current step:")
    groups = [("cond", c) for c in sorted(set(w["cond"]))] + [("phase", p) for p in ("cued", "moving", "hold")]
    print(f"{'':<28}" + "".join(f"{g[1]:>9}" for g in groups))
    for name, (p, _) in results.items():
        line = f"{name:<28}"
        for key, val in groups:
            sel = (w[key] == val) & eval_mask(w, k1)
            a_, _ = ade_fde_px(p[sel], {"fut": w["fut"][sel]}, k1)
            line += f"{a_[:, 2].mean():>9.1f}"
        print(line)
    print("\nInference time per window: " + ", ".join(f"{k} {v:.2f} us" for k, v in timing.items()))

    with open(os.path.join(a.out, f"baselines_{a.set}.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["set", "model", "horizon_s", "n_windows"] + [f"{k}_ade_px" for k in FORECAST_KP]
                    + [f"{k}_fde_px" for k in FORECAST_KP] + ["mean_ade_px", "mean_fde_px"])
        cw.writerows(rows)
    json.dump({"cv_window_search_mean_ade_px": {str(k): round(float(v), 3) for k, v in cand.items()},
               "cv_window_chosen": best, "chosen_on": a.set},
              open(os.path.join(a.out, f"baselines_{a.set}.json"), "w"), indent=1)
    print(f"Saved {a.out}/baselines_{a.set}.csv")


if __name__ == "__main__":
    main()
