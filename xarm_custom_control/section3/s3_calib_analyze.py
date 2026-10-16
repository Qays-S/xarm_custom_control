#!/usr/bin/env python3
"""
Fit the camera-to-table homography H_CB robustly and report held-out error.

Protocol (matches the paper's Section III):
  1. Before fitting, a held-out set is reserved (--holdout N points, chosen at random
     with a fixed --seed, or listed explicitly with --holdout-ids). The split is
     written to the output so it can be reported and never silently changed.
  2. H is fitted on the remaining points with RANSAC + DLT refit (outliers are
     reported, not hidden).
  3. Headline result: error of that H on the held-out points (mm).
  4. Secondary: leave-one-out error over every point.
  5. The deployed H (for pixel_to_world.py) is refitted on all inlier points; the
     held-out numbers therefore describe a slightly smaller fit and are conservative.

Usage
  python3 s3_calib_analyze.py calib_session.json --holdout 4 [--seed 0] [--ransac-mm 10]
  python3 s3_calib_analyze.py calib_session.json --holdout-ids 3 8 14 17
Writes <session>_analysis/ : calibration_error.csv, summary.json, error_map.png,
error_ecdf.png, and camera_calibration_s3.json (same keys pixel_to_world.py reads).
"""
import argparse
import csv
import os

import numpy as np

from s3_common import errors_mm, fit_homography, load_json, robust_fit, save_json, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--holdout-ids", type=int, nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ransac-mm", type=float, default=10.0)
    a = ap.parse_args()

    s = load_json(a.session)
    pts = s["points"]
    ids = np.array([p["id"] for p in pts])
    pix = np.array([[p["u"], p["v"]] for p in pts], float)
    world = np.array([[p["X"], p["Y"]] for p in pts], float)
    n = len(pts)
    if n < 8:
        raise SystemExit(f"Only {n} points; collect 12-20 so that some can be held out.")

    if a.holdout_ids:
        hold = np.isin(ids, a.holdout_ids)
        if hold.sum() != len(a.holdout_ids):
            raise SystemExit("Some --holdout-ids are not in the session.")
    else:
        rng = np.random.default_rng(a.seed)
        hold = np.zeros(n, bool)
        hold[rng.choice(n, a.holdout, replace=False)] = True
    train = ~hold
    if train.sum() < 6:
        raise SystemExit("Fewer than 6 training points left; hold out fewer.")

    h_train, inl = robust_fit(pix[train], world[train], a.ransac_mm, seed=a.seed)
    train_ids = ids[train]
    outliers = train_ids[~inl].tolist()
    e_train = errors_mm(h_train, pix[train][inl], world[train][inl])
    e_hold = errors_mm(h_train, pix[hold], world[hold])

    # leave-one-out over every inlier point (secondary diagnostic)
    keep = np.ones(n, bool)
    keep[np.isin(ids, outliers)] = False
    loo = np.full(n, np.nan)
    for i in np.flatnonzero(keep):
        m = keep.copy()
        m[i] = False
        loo[i] = errors_mm(fit_homography(pix[m], world[m]), pix[i:i + 1], world[i:i + 1])[0]

    h_all = fit_homography(pix[keep], world[keep])
    e_all = errors_mm(h_all, pix[keep], world[keep])

    out = os.path.splitext(a.session)[0] + "_analysis"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "calibration_error.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "role", "u", "v", "X_mm", "Y_mm", "Z_mm", "error_trainfit_mm", "error_loo_mm"])
        e_tr_all = errors_mm(h_train, pix, world)
        for i, p in enumerate(pts):
            role = "heldout" if hold[i] else ("outlier" if p["id"] in outliers else "train")
            wr.writerow([p["id"], role, p["u"], p["v"], p["X"], p["Y"], p["Z"],
                         round(float(e_tr_all[i]), 3), "" if np.isnan(loo[i]) else round(float(loo[i]), 3)])

    zs = np.array([p["Z"] for p in pts])
    summary = {
        "session": os.path.basename(a.session), "n_points": n,
        "split": {"heldout_ids": ids[hold].tolist(), "train_ids": train_ids.tolist(),
                  "seed": a.seed, "explicit": bool(a.holdout_ids)},
        "ransac_threshold_mm": a.ransac_mm, "ransac_outlier_ids": outliers,
        "error_mm": {"heldout": stats(e_hold), "train_in_sample": stats(e_train),
                     "leave_one_out": stats(loo[~np.isnan(loo)]), "deployed_fit_in_sample": stats(e_all)},
        "workspace_mm": {"X": [float(world[:, 0].min()), float(world[:, 0].max())],
                         "Y": [float(world[:, 1].min()), float(world[:, 1].max())]},
        "table_Z_spread_mm": float(np.ptp(zs)),
        "image_size": s["image_size"],
    }
    save_json(os.path.join(out, "summary.json"), summary)
    save_json(os.path.join(out, "camera_calibration_s3.json"), {
        "homography": h_all.tolist(), "pixel_points": pix[keep].tolist(), "world_points": world[keep].tolist(),
        "camera_index": s["camera_index"], "transform_type": "homography",
        "mean_error_mm": float(e_all.mean()), "max_error_mm": float(e_all.max()),
        "heldout_mean_error_mm": summary["error_mm"]["heldout"]["mean"],
        "heldout_max_error_mm": summary["error_mm"]["heldout"]["max"],
        "source_session": os.path.basename(a.session)})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        role_style = {"train": ("o", "tab:blue"), "heldout": ("s", "tab:orange"), "outlier": ("x", "tab:red")}
        for role, (mk, col) in role_style.items():
            m = np.array([("heldout" if hold[i] else ("outlier" if pts[i]["id"] in outliers else "train")) == role
                          for i in range(n)])
            if m.any():
                ax.scatter(world[m, 1], world[m, 0], marker=mk, c=col, label=role)
        e_plot = errors_mm(h_train, pix, world)
        for i in range(n):
            ax.annotate(f"{e_plot[i]:.1f}", (world[i, 1], world[i, 0]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("robot Y (mm)"); ax.set_ylabel("robot X (mm)")
        ax.set_title("Calibration points (labels: error of training fit, mm)")
        ax.legend(); ax.set_aspect("equal"); fig.tight_layout()
        fig.savefig(os.path.join(out, "error_map.png"), dpi=150)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for lab, e in (("held-out", e_hold), ("leave-one-out", loo[~np.isnan(loo)]), ("in-sample", e_train)):
            e = np.sort(e)
            ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post", label=f"{lab} (n={len(e)})")
        ax.set_xlabel("reprojection error (mm)"); ax.set_ylabel("fraction of points ≤ error")
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(out, "error_ecdf.png"), dpi=150)
    except ImportError:
        print("matplotlib not installed; skipped plots")

    hs = summary["error_mm"]["heldout"]
    print(f"Points: {n} ({int(hold.sum())} held out, ids {ids[hold].tolist()}); RANSAC outliers: {outliers or 'none'}")
    print(f"HELD-OUT error: mean {hs['mean']:.2f} mm, median {hs['median']:.2f}, max {hs['max']:.2f}")
    lo = summary["error_mm"]["leave_one_out"]
    print(f"Leave-one-out:  mean {lo['mean']:.2f} mm, median {lo['median']:.2f}, max {lo['max']:.2f}")
    print(f"In-sample (train): mean {summary['error_mm']['train_in_sample']['mean']:.2f} mm")
    print(f"Table Z spread across points: {summary['table_Z_spread_mm']:.1f} mm (should be a few mm at most)")
    print(f"Outputs in {out}/")


if __name__ == "__main__":
    main()
