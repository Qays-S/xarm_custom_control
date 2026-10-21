#!/usr/bin/env python3
"""
Height / parallax sensitivity of the table homography.

For every point recorded by `s3_collect.py parallax`, the tip's true table position is
the arm's (X, Y); its height above the table is Z minus the Z of that spot's table
touch (the lowest Z recorded at that spot). Projecting the tip's pixel through H
treats it as if it were on the table, so the error ||H(u,v) - (X,Y)|| shows how wrong
planar projection becomes for an elbow, wrist or hand held above the table.

Usage
  python3 s3_parallax_analyze.py parallax_session.json --calib camera_calibration_s3.json
                                 [--calib-error-mm 7.5]
Writes <session>_analysis/: parallax_points.csv, parallax_by_height.csv, summary.json,
parallax_vs_height.png, parallax_vectors.png.
--calib-error-mm (default: held-out mean stored in the calibration file) is used to
report the height at which parallax error exceeds the calibration error itself.
"""
import argparse
import csv
import os

import numpy as np

from s3_common import apply_h, load_json, save_json, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--calib-error-mm", type=float)
    ap.add_argument("--bin-mm", type=float, default=25.0, help="height bin width for the summary table")
    a = ap.parse_args()

    s = load_json(a.session)
    cal = load_json(a.calib)
    hm = np.array(cal["homography"], float)
    cal_err = a.calib_error_mm or cal.get("heldout_mean_error_mm") or cal.get("mean_error_mm")

    rows = []
    for loc in sorted({p["location"] for p in s["points"]}):
        pts = [p for p in s["points"] if p["location"] == loc]
        # a spot's table Z is its touch reading; a session may override it per spot
        # (e.g. when the touch point itself was excluded) via "z_table": {"<location>": Z}
        z_table = s.get("z_table", {}).get(str(loc), min(p["Z"] for p in pts))
        for p in pts:
            proj = apply_h(hm, [[p["u"], p["v"]]])[0]
            d = proj - np.array([p["X"], p["Y"]])
            rows.append({"location": loc, "id": p["id"], "height_mm": p["Z"] - z_table, "u": p["u"], "v": p["v"],
                         "X": p["X"], "Y": p["Y"], "projX": proj[0], "projY": proj[1],
                         "dX": d[0], "dY": d[1], "error_mm": float(np.hypot(*d))})
    if len({r["location"] for r in rows}) < 3:
        print("Warning: fewer than 3 table spots; parallax depends on position, so use >= 3 spread-out spots.")

    out = os.path.splitext(a.session)[0] + "_analysis"
    os.makedirs(out, exist_ok=True)
    keys = list(rows[0].keys())
    with open(os.path.join(out, "parallax_points.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])

    h = np.array([r["height_mm"] for r in rows])
    e = np.array([r["error_mm"] for r in rows])
    above = h > 5
    # least-squares slope through the origin per spot (a first-order summary; the
    # by-height table and plot show the full, possibly non-linear, relationship)
    e0 = {loc: next(r["error_mm"] for r in rows if r["location"] == loc and r["height_mm"] <= 5)
          if any(r["location"] == loc and r["height_mm"] <= 5 for r in rows) else 0.0
          for loc in {r["location"] for r in rows}}
    slopes = []
    for loc in e0:
        m = np.array([r["location"] == loc for r in rows]) & above
        if m.sum():
            slopes.append(float(np.sum(h[m] * e[m]) / np.sum(h[m] ** 2)))
    slope = float(np.median(slopes)) if slopes else float("nan")

    bins = np.arange(0, h.max() + a.bin_mm, a.bin_mm)
    by = []
    for lo in bins:
        m = (h >= lo - a.bin_mm / 2) & (h < lo + a.bin_mm / 2)
        if m.any():
            st = stats(e[m])
            by.append({"height_mm": float(lo), **{k: round(v, 2) if isinstance(v, float) else v for k, v in st.items()}})
    with open(os.path.join(out, "parallax_by_height.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, by[0].keys())
        wr.writeheader()
        wr.writerows(by)

    # height where the mean error first exceeds the calibration error (interpolated
    # between height bins; parallax is not exactly linear for a nearby camera)
    h_exceed = None
    if cal_err and by:
        hb = np.array([b["height_mm"] for b in by]); eb = np.array([b["mean"] for b in by])
        over = np.flatnonzero(eb > cal_err)
        if len(over):
            i = over[0]
            h_exceed = float(hb[0]) if i == 0 else float(np.interp(cal_err, eb[i - 1:i + 1], hb[i - 1:i + 1]))
    summary = {"n_points": len(rows), "n_spots": len(e0), "max_height_mm": float(h.max()),
               "error_at_table_mm": stats(e[~above]), "error_above_table_mm": stats(e[above]),
               "slope_mm_error_per_mm_height": {"median": slope, "per_spot": slopes},
               "calibration_error_reference_mm": cal_err,
               "height_where_parallax_exceeds_calibration_error_mm": h_exceed,
               "by_height": by}
    save_json(os.path.join(out, "summary.json"), summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for loc in sorted(e0):
            m = np.array([r["location"] == loc for r in rows])
            o = np.argsort(h[m])
            ax.plot(h[m][o], e[m][o], "o-", label=f"spot L{loc}")
        if cal_err:
            ax.axhline(cal_err, ls="--", c="gray", label=f"calibration error {cal_err:.1f} mm")
        ax.set_xlabel("height above table (mm)"); ax.set_ylabel("planar projection error (mm)")
        ax.set_title("Parallax: error of projecting elevated points through H")
        ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(out, "parallax_vs_height.png"), dpi=150)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for r in rows:
            ax.annotate("", (r["projY"], r["projX"]), (r["Y"], r["X"]),
                        arrowprops=dict(arrowstyle="->", color=plt.cm.viridis(r["height_mm"] / max(h.max(), 1))))
        ax.scatter([r["Y"] for r in rows], [r["X"] for r in rows], c="k", s=10, label="true table position")
        ax.set_xlabel("robot Y (mm)"); ax.set_ylabel("robot X (mm)"); ax.set_aspect("equal")
        ax.set_title("Where elevated points land after projection\n(arrow colour = height)")
        ax.autoscale(); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(out, "parallax_vectors.png"), dpi=150)
    except ImportError:
        print("matplotlib not installed; skipped plots")

    print(f"{len(rows)} points at {len(e0)} spots, heights up to {h.max():.0f} mm")
    print(f"Error at table level: mean {summary['error_at_table_mm'].get('mean', float('nan')):.1f} mm")
    print(f"First-order slope: ~{slope:.2f} mm of error per mm of height (median over spots)")
    for b in by:
        print(f"  height ~{b['height_mm']:.0f} mm: mean {b['mean']:.1f} mm, max {b['max']:.1f} mm (n={b['n']})")
    if h_exceed is not None:
        print(f"Parallax exceeds the calibration error ({cal_err:.1f} mm) above ~{h_exceed:.0f} mm height")
    print(f"Outputs in {out}/")


if __name__ == "__main__":
    main()
