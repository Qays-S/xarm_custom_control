#!/usr/bin/env python3
"""
Offline replay (E6), trade-off between caution and task progress: instead of one weighted decision loss,
sweep each planner's caution setting and record near-misses per minute against pick-and-place cycles per
minute. A planner is better if its curve has fewer near-misses at the same task progress.

Caution setting swept (everything else as chosen on D_val in s7_replay.py, perception delay 0.19 s):
  fixed-margin planners (reactive, constant velocity, GRU, multimodal GRU, ManiCast-style):
      margin d in {20, 40, 60, 90, 120, 180, 250} mm
  uncertainty-only adaptation: d_min in {20, 40, 60, 90, 120} mm
  reliability-guided (v1, v2) and binary gate: the declared false-trust budget used to set tau on D_cal,
      in {5, 10, 15, 20, 30}% (tau re-chosen on D_cal for each budget; v2 on the smoothed score), plus no fallback
Near-miss = the arm moves while the planar separation proxy is below 50 mm (s7_replay.py).
Results on D_val (where the settings were chosen) and E5 (held out). D_test stays locked.

Usage
  python3 s7_tradeoff.py --manicast-dir ~/manicast
Outputs in logs/replay/: tradeoff.csv, tradeoff_matched.csv, tradeoff.png
"""
import argparse
import csv
import json
import os

import numpy as np

import s7_replay as R
from s6_reliability import choose_tau

FIXED = ["reactive", "cv", "gru", "mgru", "sts"]
D_SWEEP = [20.0, 40.0, 60.0, 90.0, 120.0, 180.0, 250.0]
DMIN_SWEEP = [20.0, 40.0, 60.0, 90.0, 120.0]
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30]


def taus_from_cal(budgets, l_max):
    dev = np.load("logs/reliability/features_dev.npz", allow_pickle=True)
    sc = np.load("logs/reliability/scores_cal.npz", allow_pickle=True)
    cal = dev["set"] == "cal"
    L = dev["L_future"][cal]
    assert len(L) == len(sc["r_combined"])
    ok = ~np.isnan(L)
    z = (L[ok] <= l_max).astype(int)
    r = sc["r_combined"][ok]
    return {b: choose_tau(r, z, b) for b in budgets}


def matched(points, levels):
    """Near-misses per minute interpolated at given cycles-per-minute levels (NaN outside the curve)."""
    pts = sorted(points)
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    out = []
    for lv in levels:
        out.append(float(np.interp(lv, x, y)) if x.min() <= lv <= x.max() else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manicast-dir", default=os.path.expanduser("~/manicast"))
    ap.add_argument("--results", default="logs/forecast_results")
    ap.add_argument("--out", default="logs/replay")
    a = ap.parse_args()
    summ = json.load(open(os.path.join(a.out, "summary.json")))
    chosen = summ["chosen_on_val"]
    R.DELAY = summ.get("delay_steps", 3)
    rel = json.load(open("logs/reliability/summary.json"))
    taus = taus_from_cal(BUDGETS, rel["l_max_px"])
    print("tau on D_cal per false-trust budget:", {f"{int(b * 100)}%": round(t, 3) for b, t in taus.items()})

    path = R.Path(json.load(open("calib/calib_cam3d.json")))
    sets = {"val": R.forecasts("logs/prepared", "val", a.results, a.manicast_dir, "logs/reliability/scores_val.npz"),
            "E5": R.forecasts("logs/prepared_e5", "e5", a.results, a.manicast_dir, "logs/reliability/scores_e5.npz")}
    has_sts = all(F["sts"] is not None for Fs in sets.values() for F in Fs.values())
    runs = []
    for mth in FIXED:
        if mth == "sts" and not has_sts:
            continue
        for d in D_SWEEP:
            runs.append((mth, f"d={d:.0f}", dict(chosen[mth], d_fix=d)))
    for d in DMIN_SWEEP:
        runs.append(("uncert", f"d_min={d:.0f}", dict(chosen["uncert"], d_min=d)))
    for mth in ("full", "gate"):
        for b in BUDGETS:
            runs.append((mth, f"budget={int(b * 100)}%", dict(chosen[mth], tau=taus[b])))
    runs.append(("full", "no fallback", dict(chosen["full"], tau=-1.0)))
    for b in BUDGETS:
        runs.append(("full2", f"budget={int(b * 100)}%", dict(chosen["full2"], tau=R.tau_smoothed(b, rel["l_max_px"]))))

    rows, curves = [], {}
    for setname, Fs in sets.items():
        for mth, label, p in runs:
            sm = R.summarise(R.run_set(path, Fs, mth, p), path)
            rows.append([setname, R.METHODS[mth], label, sm["cycles_per_min"], sm["violations_per_min"], sm["wait_share"],
                         sm["fallback_share"], sm["sep_p5_mm"]])
            curves.setdefault((setname, mth), []).append((float(sm["cycles_per_min"]), float(sm["violations_per_min"])))
            print(f"{setname:<4}{R.METHODS[mth]:<34}{label:<14}{sm['cycles_per_min']:>7.2f} cycles/min{sm['violations_per_min']:>8.2f} near-misses/min", flush=True)
    with open(os.path.join(a.out, "tradeoff.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["set", "method", "setting", "cycles_per_min", "near_misses_per_min", "wait_share", "fallback_share", "sep_p5_mm"])
        cw.writerows(rows)

    levels = [2.0, 3.0, 4.0, 5.0]
    mrows = []
    print(f"\nNear-misses per minute at matched task progress (interpolated along each curve)")
    print(f"{'set':<5}{'method':<36}" + "".join(f"{f'{lv:g} cyc/min':>12}" for lv in levels))
    for (setname, mth), pts in curves.items():
        v = matched(pts, levels)
        mrows.append([setname, R.METHODS[mth]] + [round(x, 2) if x == x else "" for x in v])
        print(f"{setname:<5}{R.METHODS[mth]:<36}" + "".join(f"{x:>12.2f}" if x == x else f"{'-':>12}" for x in v))
    with open(os.path.join(a.out, "tradeoff_matched.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["set", "method"] + [f"near_misses_at_{lv:g}_cycles_per_min" for lv in levels]); cw.writerows(mrows)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
        style = {"reactive": ("k", "o"), "cv": ("tab:gray", "v"), "gru": ("tab:blue", "s"), "mgru": ("tab:cyan", "D"),
                 "sts": ("tab:purple", "^"), "uncert": ("tab:orange", "P"), "gate": ("tab:green", "X"), "full": ("tab:red", "*"),
                 "full2": ("darkred", "*")}
        for i, setname in enumerate(sets):
            for (sn, mth), pts in curves.items():
                if sn != setname:
                    continue
                pts = sorted(pts)
                c, mk = style[mth]
                ax[i].plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=c, lw=1.2 if not mth.startswith("full") else 2.2,
                           ms=5 if not mth.startswith("full") else 9, ls="--" if mth == "full2" else "-", label=R.METHODS[mth])
            ax[i].set(xlabel="pick-and-place cycles per minute (higher is better)", ylabel="near-misses per minute (lower is better)",
                      title=("Validation (settings chosen here)" if setname == "val" else "E5, degraded view (held out)"))
            ax[i].grid(alpha=0.3)
        ax[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "tradeoff.png"), dpi=150)
    except ImportError:
        pass
    print(f"\nSaved in {a.out}/")


if __name__ == "__main__":
    main()
