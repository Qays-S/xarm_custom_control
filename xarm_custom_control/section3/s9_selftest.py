#!/usr/bin/env python3
"""
Checks that the live code (s9_online.py, s9_live.py) gives exactly what the offline pipeline gives,
by feeding a recorded session to it frame by frame. Run it on the lab laptop before the first live
session, and again after retraining. Nothing moves; no camera is needed.

  A. 16 Hz steps: OnlineGrid fed frame by frame == s5_prepare.py --causal on the whole session
  B. forecasts and reliability: OnlineModels.record == s7_replay.forecasts on causally prepared data
  C. planners: LivePlanner == s7_replay.simulate for every method, on the replay's own inputs

Usage
  python3 s9_selftest.py --manicast-dir ~/manicast [--session logs/E4_s02] [--every 5]
Writes logs/selftest/selftest.json; prints PASS/FAIL per check and the live compute time per step.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import s7_replay as R
from s9_live import LivePlanner
from s9_online import KP, RAW_KP, OnlineGrid, OnlineModels


def frames_of(session):
    """Recorded frames as the recorder would hand them to the live controller."""
    meta = json.load(open(os.path.join(session, "meta.json")))
    side = meta["side"]
    names = [f"{side}_{k}" for k in KP] + RAW_KP
    out = []
    for f in csv.DictReader(open(os.path.join(session, "frames.csv"))):
        num = lambda x: float(x) if x not in ("", None) else float("nan")
        kp = {n: (num(f[f"{n}_u"]), num(f[f"{n}_v"]), num(f[f"{n}_conf"])) for n in names}
        miss = {n: int(f[f"{n}_missing"]) for n in names}
        out.append((float(f["t_read_end"]), kp, miss, f["phase"], f["goal"]))
    cues = sorted(float(r["t_cue"]) for r in csv.DictReader(open(os.path.join(session, "reaches.csv"))))
    return meta, out, cues


def feed(grid, frames, cues, on_step=None):
    """Play frames into the grid in time order, cues as they happen; call on_step after new steps."""
    ci = 0
    for t, kp, miss, phase, goal in frames:
        while ci < len(cues) and cues[ci] < t:              # live, a cue is known before the next frame
            grid.add_cue(cues[ci]); ci += 1
        n = grid.add_frame(t, kp, miss, phase, goal)
        if n and on_step:
            on_step()


def close(a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False
    same_nan = np.isnan(a) == np.isnan(b)
    return bool(same_nan.all() and np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=tol))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="logs/E4_s02")
    ap.add_argument("--manicast-dir", default=os.path.expanduser("~/manicast"))
    ap.add_argument("--results", default="logs/forecast_results")
    ap.add_argument("--reliability", default="logs/reliability")
    ap.add_argument("--calib", default="calib/calib_cam3d.json")
    ap.add_argument("--every", type=int, default=1, help="compare B on every n-th step (all steps are run)")
    ap.add_argument("--out", default="logs/selftest")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report, ok_all = {}, True
    name = os.path.basename(os.path.normpath(a.session))
    meta, frames, cues = frames_of(a.session)
    W, H = meta.get("actual_size", [1280, 720])

    # ---- A. 16 Hz steps
    from s5_prepare import resample_session
    off = resample_session(a.session, 0.0625, 0.25, causal=True)
    old = resample_session(a.session, 0.0625, 0.25, causal=False)
    g = OnlineGrid(meta["side"], W, H)
    g.steps = type(g.steps)(maxlen=None)          # keep every step for this check
    feed(g, frames, cues)
    st = list(g.steps)
    n = min(len(st), len(off["t"]))
    yo = np.stack([q["y"] for q in st[:n]])
    checks = {"times": close([q["t"] for q in st[:n]], off["t"][:n], 1e-9),
              "positions": close(yo, off["y"][:n], 1e-9),
              "missing": bool((np.stack([q["miss"] for q in st[:n]]) == off["miss"][:n]).all()),
              "confidence": close(np.stack([q["conf"] for q in st[:n]]), off["conf"][:n], 1e-9),
              "phase": bool((np.array([q["phase"] for q in st[:n]]) == off["phase"][:n]).all()),
              "reach": bool((np.array([q["reach"] for q in st[:n]]) == off["reach"][:n]).all()),
              "step count": len(off["t"]) - len(st) in (0, 1)}
    differ = float((old["miss"][:n] != off["miss"][:n]).any(1).mean())
    report["A_steps"] = {"steps": n, **checks,
                         "share_of_steps_where_causal_rule_differs_from_original": round(differ, 4)}
    ok = all(checks.values()); ok_all &= ok
    print(f"A. 16 Hz steps ({n} steps): {'PASS' if ok else 'FAIL'}  {checks}")
    print(f"   the causal rule changes {differ:.1%} of steps compared with the original preparation")

    # ---- B. forecasts and reliability on causally prepared data
    prep = os.path.join(a.out, "prepared_causal")
    shutil.rmtree(prep, ignore_errors=True)
    subprocess.run([sys.executable, "s5_prepare.py", a.session, "--out", prep, "--causal"], check=True,
                   stdout=subprocess.DEVNULL)
    from s5_data import load
    from s6_features import build
    import pickle
    from s6_reliability import X
    m, _ = load(prep)
    setname = next(k for k, v in m["sessions"][name]["anchors"].items() if v[2] > 0)
    T = build(prep, [setname], a.results)
    est = pickle.load(open(os.path.join(a.reliability, "reliability_model.pkl"), "rb"))["estimators"]
    sc = {}
    for key, nm in (("r_combined", "combined (o+u+e)"), ("r_uncertainty", "uncertainty-only (u)")):
        mdl, cal, tau, cols = est[nm]
        sc[key] = cal.predict(mdl.predict_proba(X(T, cols))[:, 1])
    sfile = os.path.join(a.out, "scores_selftest.npz")
    np.savez_compressed(sfile, session=T["session"], anchor=T["anchor"], **sc)
    have_sts = os.path.exists(os.path.join(a.manicast_dir, "model", "manicast.py"))
    F = R.forecasts(prep, setname, a.results, a.manicast_dir if have_sts else None, sfile)[name]
    want = {int(t): i for i, t in enumerate(F["anchor"])}
    calib = R.calib_for(a.session)                  # the session's own calibration, as s7_replay uses
    models = OnlineModels(calib, a.results, a.reliability, a.manicast_dir if have_sts else None, "logs/prepared",
                          meta["side"])
    g = OnlineGrid(meta["side"], W, H)
    fields = ["now", "reactive", "gru", "cv", "pi", "modes", "r", "r_u", "r_s"] + (["sts"] if have_sts else [])
    bad = {k: 0 for k in fields + ["visible"]}
    worst = {k: 0.0 for k in fields}
    times, count = [], [0]

    def on_step():
        steps, fr = g.snapshot()
        anc = steps[-1]["anchor"]
        if anc not in want:
            return
        t0 = time.perf_counter()
        rec = models.record(steps, fr)                  # every step, as live (keeps r_s exact)
        times.append((time.perf_counter() - t0) * 1000)
        if anc % a.every:
            return
        i = want[anc]
        count[0] += 1
        bad["visible"] += rec["visible"] != bool(F["visible"][i])
        for k in fields:
            ref = F[k][i]
            tol = 1e-6 if k in ("r", "r_u", "r_s", "pi") else 0.01          # mm fields: 0.01 mm
            if not close(rec[k], ref, tol):
                bad[k] += 1
                worst[k] = max(worst[k], float(np.nanmax(np.abs(np.asarray(rec[k], float) - np.asarray(ref, float)))))
    feed(g, frames, cues, on_step)
    okB = all(v == 0 for v in bad.values()) and count[0] > 0
    ok_all &= okB
    report["B_models"] = {"set": setname, "steps_checked": count[0], "mismatches": bad, "largest_difference": worst,
                          "sts_checked": have_sts,
                          "compute_ms_per_step": {"median": round(float(np.median(times)), 1),
                                                  "p95": round(float(np.percentile(times, 95)), 1)}}
    print(f"B. forecasts + reliability ({count[0]} steps of {setname}): {'PASS' if okB else 'FAIL'}  "
          f"mismatches {bad}")
    print(f"   live compute per step: median {np.median(times):.1f} ms, p95 {np.percentile(times, 95):.1f} ms "
          f"(budget 62.5 ms)")

    # ---- C. planners on the replay's own inputs
    path = R.Path(json.load(open(a.calib)))
    summ = json.load(open("logs/replay/summary.json"))
    chosen, tau, tau2 = summ["chosen_on_val"], summ["tau_from_cal"], summ["tau_smoothed_from_cal"]
    Fs = R.forecasts("logs/prepared", "val", a.results, a.manicast_dir if have_sts else None,
                     "logs/reliability/scores_val.npz")
    sname = sorted(Fs)[0]
    Fv = Fs[sname]
    resC = {}
    for mth in R.METHODS:
        if mth == "sts" and Fv["sts"] is None:
            continue
        p = dict(chosen[mth], tau=tau2 if mth == "full2" else tau)
        ref = R.simulate(path, Fv, mth, p)["a"]
        lp = LivePlanner(path, mth, p, R.DELAY)
        got = []
        for i in range(len(Fv["anchor"])):
            j = max(0, i - R.DELAY)
            rec = {k: (v[j] if isinstance(v, np.ndarray) else v) for k, v in Fv.items() if v is not None}
            a_ = lp.step(rec)["a"]
            lp.advance(a_)
            got.append(a_)
        resC[mth] = bool(np.array_equal(np.array(got), ref))
    okC = all(resC.values()); ok_all &= okC
    report["C_planners"] = {"session": sname, "steps": len(Fv["anchor"]), "identical": resC}
    print(f"C. planners ({len(Fv['anchor'])} steps of {sname}): {'PASS' if okC else 'FAIL'}  {resC}")

    report["all_pass"] = bool(ok_all)
    json.dump(report, open(os.path.join(a.out, "selftest.json"), "w"), indent=2, default=float)
    print(f"\n{'ALL CHECKS PASS' if ok_all else 'SOME CHECKS FAILED'}; saved {a.out}/selftest.json")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
