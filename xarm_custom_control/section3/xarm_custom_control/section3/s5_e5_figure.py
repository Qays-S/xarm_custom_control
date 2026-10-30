#!/usr/bin/env python3
"""
Figure for Section 6.6 (replaces the old Table 8): right-wrist error 1.0 s ahead while the hand is moving,
on the validation set and in each E5 condition, for every forecaster. Errors are ADE in pixels of the
1280 x 720 image (1 px is about 1.5 mm at the table). The multimodal GRU is scored on its most likely mode.

Reads the numbers s5_eval_e5.py already saved; computes nothing new.
  logs/forecast_results/e5/e5_results.csv      (validation, moving phase, 1.0 s)
  logs/forecast_results/e5/e5_by_condition.csv (each E5 condition)
Bars taller than the plot are cut at the top and their value is printed above them.

Usage
  python3 s5_e5_figure.py
Output: logs/forecast_results/e5/e5_by_condition.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "logs/forecast_results/e5"
MODELS = [("current pose", "Current pose"), ("constant velocity (n=2)", "Constant velocity"), ("GRU", "GRU"),
          ("multimodal GRU (top mode)", "Multimodal GRU"), ("STS-GCN fine-tuned", "STS-GCN, fine-tuned")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CONDS = [("none", "normal", "Normal reach"), ("cover", "normal", "Hand covered"), ("head", "normal", "Paper above head"),
         ("none", "dim", "Normal reach"), ("cover", "dim", "Hand covered"), ("head", "dim", "Paper above head")]
Y_MAX = 200.0


def main():
    val, n_val = {}, None
    for r in csv.DictReader(open(os.path.join(D, "e5_results.csv"))):
        if r["set"] == "val" and r["phase"] == "moving" and float(r["horizon_s"]) == 1.0:
            val[r["model"]] = float(r["wrist_ade_px"]); n_val = int(r["n_windows"])
    cond, n_cond = {}, {}
    for r in csv.DictReader(open(os.path.join(D, "e5_by_condition.csv"))):
        cond[(r["model"], r["perturb"], r["lighting"])] = float(r["wrist_ade_1s_px"])
        n_cond[(r["perturb"], r["lighting"])] = int(r["n_moving_windows"])

    groups = [("Validation", n_val, {m: val[m] for m, _ in MODELS})]
    for p, l, name in CONDS:
        groups.append((name, n_cond[(p, l)], {m: cond[(m, p, l)] for m, _ in MODELS}))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w, gap = 0.16, 0.6
    x0 = [0.0]
    for g in range(1, len(groups)):
        x0.append(x0[-1] + len(MODELS) * w + gap + (0.5 if g in (1, 4) else 0))   # extra space between blocks
    for i, (key, label) in enumerate(MODELS):
        xs = [x + i * w for x in x0]
        ys = [g[2][key] for g in groups]
        ax.bar(xs, [min(y, Y_MAX) for y in ys], width=w - 0.02, color=COLORS[i], label=label, zorder=2)
        for xx, y in zip(xs, ys):
            if y > Y_MAX:
                ax.text(xx, Y_MAX + 3, f"{y:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333333")
                for yy in (Y_MAX - 14, Y_MAX - 8):          # white break marks: this bar is cut off
                    ax.plot([xx - w / 2, xx + w / 2], [yy - 3, yy + 3], color="white", lw=2, zorder=3)
    centres = [x + (len(MODELS) - 1) * w / 2 for x in x0]
    ax.set_xticks(centres, [f"{g[0]}\n({g[1]} windows)" for g in groups])
    ax.set_ylim(0, Y_MAX + 18)
    ax.set_ylabel("Right-wrist error 1.0 s ahead (px)\n1 px is about 1.5 mm at the table")
    ax.yaxis.grid(True, color="#e2e2e2", zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    # block labels under the ticks
    for (a, b), txt in (((1, 3), "E5, room lights at half"), ((4, 6), "E5, all lights off")):
        ax.annotate(txt, xy=((centres[a] + centres[b]) / 2, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -34), textcoords="offset points", ha="center", va="top", fontsize=9, fontweight="bold")
    ax.legend(ncol=5, loc="upper left", frameon=False, fontsize=8.5, bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    out = os.path.join(D, "e5_by_condition.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
