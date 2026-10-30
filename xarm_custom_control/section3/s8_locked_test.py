#!/usr/bin/env python3
"""
Blueprint Algorithm 1 step 8 and Section IX steps 54-56: ONE evaluation on the locked test set D_test.

Everything was frozen first (logs/locked_test/freeze.json holds the SHA-256 of every model, setting and script
used). Nothing in this script is tuned: models, reliability estimators, calibration, tau and planner settings
are all loaded as they are. D_test = the last third of the reaches of every round-2 session (E1-E4).

  A. Forecasting (Section V/IX): ADE/FDE of current pose, constant velocity, GRU, multimodal GRU (most likely
     mode; plus NLL, best-of-modes oracle and calibration), fine-tuned STS-GCN.
  B. Reliability (Section VI): the four estimators (fitted on D_val, calibrated on D_cal, tau from D_cal):
     AUROC, AUPRC, Brier, log loss, ECE, Spearman with the realised loss, false-trust rate at tau, per
     condition, and detection delay after E4 target switches.
  C. Planning (Section VII, E6 offline replay): every planner with its D_val settings and the 0.19 s
     perception delay, plus the same without delay, plus the caution sweep (near-misses at matched progress).

Usage
  python3 s8_locked_test.py --manicast-dir ~/manicast
Outputs in logs/locked_test/
"""
import argparse
import csv
import json
import os
import pickle

import numpy as np
import torch
from scipy.stats import spearmanr

from s5_baselines import const_vel, current_pose
from s5_data import FORECAST_KP, HORIZONS_S, ade_fde_px, eval_mask, load, windows
from s5_gru import GRUForecaster, features, predict
from s5_mgru import MixtureGRU, hpd_coverage, nll_px, run as mrun, to_positions

OUT = "logs/locked_test"


def part_a(manicast_dir, res="logs/forecast_results"):
    m, sess = load("logs/prepared")
    w = windows(m, sess, "test")
    x, cur = features(w)
    ks = [round(h / m["dt"]) for h in HORIZONS_S]
    cvn = json.load(open(os.path.join(res, "baselines_val.json")))["cv_window_chosen"]
    gc = json.load(open(os.path.join(res, "gru", "gru_best.json")))
    gru = GRUForecaster(gc["n_in"], gc["config"]["hidden"], gc["config"]["layers"], gc["K"])
    gru.load_state_dict(torch.load(os.path.join(res, "gru", "gru_best.pt"), map_location="cpu"))
    mc = json.load(open(os.path.join(res, "mgru", "mgru_best.json")))
    mg = MixtureGRU(mc["n_in"], mc["config"]["hidden"], mc["config"]["layers"], mc["K"], mc["M"])
    mg.load_state_dict(torch.load(os.path.join(res, "mgru", "mgru_best.pt"), map_location="cpu"))
    out = mrun(mg, x)
    pi, pos = to_positions(out, cur)
    top = pos[np.arange(len(pos)), pi.argmax(1)]
    preds = {"current pose": current_pose(w, 24), f"constant velocity (n={cvn})": const_vel(w, 24, cvn),
             "GRU": predict(gru, x, cur), "multimodal GRU (most likely mode)": top}
    if manicast_dir and os.path.exists(os.path.join(manicast_dir, "model", "manicast.py")):
        from s5_eval_e5 import StsGcn
        m0, s0 = load("logs/prepared")
        sts = StsGcn(manicast_dir, m0, s0, os.path.join(res, "stsgcn", "stsgcn_finetuned_best.pt")).run(m, sess, "test", w)
        preds["STS-GCN fine-tuned"] = np.concatenate([sts, np.repeat(sts[:, -1:], 8, 1)], 1)
    rows = []
    print(f"A. Forecasting on D_test: {len(w['cond'])} windows")
    print(f"{'model':<36}{'horizon':>8}{'n':>6}{'elbow':>8}{'wrist':>8}{'hand':>8}{'mean':>8}{'FDE':>8}")
    for name, p in preds.items():
        for h, k in zip(HORIZONS_S, ks):
            if name.startswith("STS") and h > 1.0:
                continue
            mk = eval_mask(w, k)
            ad, fd = ade_fde_px(p[mk], {"fut": w["fut"][mk]}, k)
            a_ = ad.mean(0); f_ = fd.mean(0)
            mean = float(np.nanmean(a_)) if not name.startswith("STS") else float(a_[:2].mean())
            rows.append(["test", name, h, int(mk.sum())] + [round(float(v), 2) for v in a_] + [round(float(v), 2) for v in f_] + [round(mean, 2)])
            print(f"{name:<36}{h:>7}s{mk.sum():>6}" + "".join(f"{v:>8.1f}" for v in a_) + f"{mean:>8.1f}{np.nanmean(f_):>8.1f}")
    nll = {str(h): round(nll_px(out, w, cur, k)[0], 3) for h, k in zip(HORIZONS_S, ks)}
    per_mode = np.stack([ade_fde_px(pos[eval_mask(w, 16)][:, q], {"fut": w["fut"][eval_mask(w, 16)]}, 16)[0].mean(1)
                         for q in range(mc["M"])], 1)
    cal = hpd_coverage(out, w, cur, ks)
    k1 = 16
    cond = {}
    for name, p in preds.items():
        cond[name] = {}
        for c in sorted(set(w["cond"])) + ["moving", "hold"]:
            sel = ((w["cond"] == c) if c.startswith("E") else (w["phase"] == c)) & eval_mask(w, k1)
            ad, _ = ade_fde_px(p[sel], {"fut": w["fut"][sel]}, k1)
            cond[name][c] = round(float(ad[:, 1].mean()), 2)          # wrist (all models have it)
    with open(os.path.join(OUT, "forecast_test.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["set", "model", "horizon_s", "n_windows"] + [f"{k}_ade_px" for k in FORECAST_KP]
                    + [f"{k}_fde_px" for k in FORECAST_KP] + ["mean_ade_px"])
        cw.writerows(rows)
    with open(os.path.join(OUT, "mgru_calibration_test.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["horizon_step", "nominal", "n", "elbow", "wrist", "hand", "mean"]); cw.writerows(cal)
    print(f"multimodal GRU on D_test: NLL {nll}, best-of-modes mean ADE at 1.0 s {per_mode.min(1).mean():.1f} px")
    print("calibration (mean coverage): " + ", ".join(f"step {c[0]} {c[1]:.0%}->{c[-1]:.0%}" for c in cal))
    print("wrist ADE at 1.0 s by condition/phase: " + json.dumps(cond))
    return {"nll": nll, "min_ade_1s": round(float(per_mode.min(1).mean()), 2), "calibration": cal, "wrist_1s_by_condition": cond,
            "n_windows": int(len(w["cond"]))}


def part_b():
    from s6_features import build
    from s6_reliability import X, metrics, sub, boot_spearman, SETS
    model = pickle.load(open("logs/reliability/reliability_model.pkl", "rb"))
    l_max = model["l_max_px"]
    T_all = build("logs/prepared", ["test"])
    np.savez_compressed(os.path.join(OUT, "features_test.npz"), **T_all)
    T = sub(T_all, ~np.isnan(T_all["L_future"]))
    T["z"] = (T["L_future"] <= l_max).astype(int)
    print(f"\nB. Reliability on D_test: {len(T['z'])} labelled windows, {1 - T['z'].mean():.0%} unreliable (l_max {l_max} px)")
    rows, scores = [], {}
    for name, (mdl, cal, tau, cols) in model["estimators"].items():
        r = cal.predict(mdl.predict_proba(X(T, cols))[:, 1])
        mt = metrics(r, T["z"], T["L_future"], tau)
        rows.append([name] + [mt[k] for k in ("AUROC", "AUPRC", "Brier", "log_loss", "ECE", "spearman_r_vs_loss", "trusted_share", "FTR")] + [round(tau, 3)])
        print(f"  {name:<24} AUROC {mt['AUROC']:.3f}  AUPRC {mt['AUPRC']:.3f}  Brier {mt['Brier']:.3f}  ECE {mt['ECE']:.3f}  "
              f"rho {mt['spearman_r_vs_loss']:+.3f}  trusted {mt['trusted_share']:.2f}  FTR {mt['FTR'] if mt['FTR'] is not None else float('nan'):.3f}")
        scores[name] = cal.predict(mdl.predict_proba(X(T_all, cols))[:, 1])
    with open(os.path.join(OUT, "reliability_test.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["estimator", "AUROC", "AUPRC", "Brier", "log_loss", "ECE", "spearman_r_vs_loss", "trusted_share", "FTR", "tau"])
        cw.writerows(rows)
    np.savez_compressed(os.path.join(OUT, "scores_test.npz"), session=T_all["session"], anchor=T_all["anchor"],
                        r_combined=scores["combined (o+u+e)"], r_uncertainty=scores["uncertainty-only (u)"])
    mdl, cal, tau, cols = model["estimators"]["combined (o+u+e)"]
    r = cal.predict(mdl.predict_proba(X(T, cols))[:, 1])
    crow = []
    for c in sorted(set(T["cond"])) + ["moving", "hold"]:
        m = (T["cond"] == c) if c.startswith("E") else (T["phase"] == c)
        mt = metrics(r[m], T["z"][m], T["L_future"][m], tau)
        crow.append([c, mt["n"], mt["unreliable_share"], mt["AUROC"], mt["trusted_share"], mt["FTR"]])
    with open(os.path.join(OUT, "reliability_test_by_condition.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["condition", "n", "unreliable_share", "AUROC", "trusted_share", "FTR"]); cw.writerows(crow)
    print("  combined, per condition: " + "; ".join(f"{c[0]} AUROC {c[3]} FTR {c[5]}" for c in crow))
    g = np.array([f"{s}_{q}" for s, q in zip(T["session"], T["reach"])])
    assoc = {"spearman": round(float(spearmanr(r, T["L_future"]).correlation), 3), "ci95_reach_bootstrap": boot_spearman(r, T["L_future"], g)}
    sw = switches(T_all, scores["combined (o+u+e)"], tau)
    print(f"  r_t vs realised loss: {assoc}; E4 switches in D_test: {sw['n']}, detected {sw['detected']}, delay median {sw['median']} s")
    return {"estimators": rows, "by_condition": crow, "association": assoc, "switch_detection": sw,
            "n_labelled": int(len(T["z"])), "unreliable_share": round(float(1 - T["z"].mean()), 3)}


def switches(T, r_all, tau, win=1.5):
    from s4_dataset_summary import effector_track, load as load_session, switch_onset
    rows, delays = [], []
    for s in [x for x in np.unique(T["session"]) if x.startswith("E4")]:
        tgrid = np.load(os.path.join("logs", "prepared", s + ".npz"))["t"]
        meta, frames, _, reaches = load_session(os.path.join("logs", s))
        t, xy = effector_track(frames, meta["side"])
        tg = {int(k): v for k, v in meta["targets"].items()}
        m = T["session"] == s
        ta, rr = tgrid[T["anchor"][m]], r_all[m]
        mine = set(T["reach"][m].tolist())
        for q in reaches:
            if q["valid"] != "1" or not q["t_switch_cue"] or int(q["reach"]) not in mine:
                continue
            on = switch_onset(t, xy, float(q["t_switch_cue"]), tg[int(q["goal_cued"])]["px"], 45.0)
            if on is None:
                continue
            hit = np.where((ta >= on) & (ta <= on + win) & (rr < tau))[0]
            d = float(ta[hit[0]] - on) if len(hit) else None
            rows.append([s, q["reach"], round(on, 3), round(d, 3) if d is not None else ""])
            if d is not None:
                delays.append(d)
    with open(os.path.join(OUT, "switch_detection_test.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["session", "reach", "t_physical_onset", "delay_s"]); cw.writerows(rows)
    q = lambda p: round(float(np.percentile(delays, p)), 3) if delays else None
    return {"n": len(rows), "detected": len(delays), "median": q(50), "p90": q(90)}


def part_c(manicast_dir):
    import s7_replay as R
    import s7_tradeoff as TT
    summ = json.load(open("logs/replay/summary.json"))
    chosen, tau, tau2 = summ["chosen_on_val"], summ["tau_from_cal"], summ["tau_smoothed_from_cal"]
    rel = json.load(open("logs/reliability/summary.json"))
    path = R.Path(json.load(open("calib/calib_cam3d.json")))
    F = R.forecasts("logs/prepared", "test", "logs/forecast_results", manicast_dir, os.path.join(OUT, "scores_test.npz"))
    methods = [mth for mth in R.METHODS if mth != "sts" or all(f["sts"] is not None for f in F.values())]
    res = {}
    print(f"\nC. Offline replay on D_test ({sum(len(f['anchor']) for f in F.values()) * R.DT / 60:.1f} min of recordings)")
    print(f"{'delay':<7}{'method':<40}{'cycles/min':>11}{'wait':>7}{'fallbk':>8}{'viol/min':>9}{'L_dec':>8}{'regret':>8}")
    rows = []
    for delay in (3, 0):
        R.DELAY = delay
        for mth in methods:
            p = dict(chosen[mth], tau=tau2 if mth == "full2" else tau)
            sm = R.summarise(R.run_set(path, F, mth, p), path)
            res[f"delay{delay}/{mth}"] = sm
            rows.append([delay, R.METHODS[mth]] + [sm[k] for k in ("cycles_per_min", "wait_share", "fallback_share", "fallback_per_min",
                                                                  "min_sep_mm", "sep_p5_mm", "violations_per_min", "decision_loss", "regret")])
            print(f"{delay * 0.0625:<7.2f}{R.METHODS[mth]:<40}{sm['cycles_per_min']:>11.2f}{sm['wait_share']:>7.2f}{sm['fallback_share']:>8.2f}"
                  f"{sm['violations_per_min']:>9.2f}{sm['decision_loss']:>8.3f}{sm['regret']:>8.3f}", flush=True)
    with open(os.path.join(OUT, "replay_test.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["delay_steps", "method", "cycles_per_min", "wait_share", "fallback_share", "fallback_per_min", "min_sep_mm",
                     "sep_p5_mm", "violations_per_min", "decision_loss", "regret"])
        cw.writerows(rows)
    # caution sweep with delay, same grid as s7_tradeoff.py
    R.DELAY = 3
    taus = TT.taus_from_cal(TT.BUDGETS, rel["l_max_px"])
    runs = []
    for mth in TT.FIXED:
        if mth in methods:
            runs += [(mth, f"d={d:.0f}", dict(chosen[mth], d_fix=d)) for d in TT.D_SWEEP]
    runs += [("uncert", f"d_min={d:.0f}", dict(chosen["uncert"], d_min=d)) for d in TT.DMIN_SWEEP]
    for mth in ("full", "gate"):
        runs += [(mth, f"budget={int(b * 100)}%", dict(chosen[mth], tau=taus[b])) for b in TT.BUDGETS]
    runs.append(("full", "no fallback", dict(chosen["full"], tau=-1.0)))
    runs += [("full2", f"budget={int(b * 100)}%", dict(chosen["full2"], tau=R.tau_smoothed(b, rel["l_max_px"]))) for b in TT.BUDGETS]
    curves, trows = {}, []
    for mth, label, p in runs:
        sm = R.summarise(R.run_set(path, F, mth, p), path)
        trows.append([R.METHODS[mth], label, sm["cycles_per_min"], sm["violations_per_min"]])
        curves.setdefault(mth, []).append((float(sm["cycles_per_min"]), float(sm["violations_per_min"])))
    with open(os.path.join(OUT, "tradeoff_test.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["method", "setting", "cycles_per_min", "near_misses_per_min"]); cw.writerows(trows)
    levels = [2.0, 3.0, 4.0, 5.0]
    mrows = []
    print(f"\nNear-misses per minute at matched progress on D_test (0.19 s delay)")
    print(f"{'method':<40}" + "".join(f"{f'{lv:g} cyc/min':>12}" for lv in levels))
    for mth, pts in curves.items():
        v = TT.matched(pts, levels)
        mrows.append([R.METHODS[mth]] + [round(x, 2) if x == x else "" for x in v])
        print(f"{R.METHODS[mth]:<40}" + "".join(f"{x:>12.2f}" if x == x else f"{'-':>12}" for x in v))
    with open(os.path.join(OUT, "tradeoff_matched_test.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["method"] + [f"near_misses_at_{lv:g}_cycles_per_min" for lv in levels]); cw.writerows(mrows)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        style = {"reactive": ("k", "o"), "cv": ("tab:gray", "v"), "gru": ("tab:blue", "s"), "mgru": ("tab:cyan", "D"),
                 "sts": ("tab:purple", "^"), "uncert": ("tab:orange", "P"), "gate": ("tab:green", "X"), "full": ("tab:red", "*"),
                 "full2": ("darkred", "*")}
        for mth, pts in curves.items():
            pts = sorted(pts); c, mk = style[mth]
            ax.plot([q[0] for q in pts], [q[1] for q in pts], marker=mk, color=c, lw=2.2 if mth.startswith("full") else 1.2,
                    ms=9 if mth.startswith("full") else 5, ls="--" if mth == "full2" else "-", label=R.METHODS[mth])
        ax.set(xlabel="pick-and-place cycles per minute (higher is better)", ylabel="near-misses per minute (lower is better)",
               title="Locked test set, 0.19 s perception delay")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "tradeoff_test.png"), dpi=150)
    except ImportError:
        pass
    return {"replay": res, "matched": mrows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manicast-dir", default=os.path.expanduser("~/manicast"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.join(OUT, "freeze.json")):
        raise SystemExit("logs/locked_test/freeze.json missing: freeze everything before opening D_test")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    out = {"forecasting": part_a(a.manicast_dir), "reliability": part_b(), "planning": part_c(a.manicast_dir)}
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
    print(f"\nSaved in {OUT}/")


if __name__ == "__main__":
    main()
