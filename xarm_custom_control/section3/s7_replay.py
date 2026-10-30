#!/usr/bin/env python3
"""
Blueprint Sections VII-VIII (steps 41-47, 50-51) and E6: reliability-guided planning, tested by OFFLINE REPLAY.

The recorded human reaches are played back at 16 Hz. A simulated xArm 7 (s7_robot.py) runs pick-and-place
cycles between dot 6 (pick) and dot 8 (place) along a fixed path. At every step each planner chooses how fast
to move along the path for the next 1.0 s (receding horizon; only the first step is executed). Nobody is near
a moving arm: the person was recorded with the arm parked, so the person does not react to the simulated
robot (open-loop replay, a limitation stated in the paper).

Geometry (step 51): the arm is a chain of capsules (s7_robot.py). The human-robot "distance" is a planar
proxy: human keypoints (elbow, wrist, hand) are mapped to the table with the calibration homography and
compared with the table footprint of the arm capsules that come within H_BAND = 200 mm of the table (the
height band where a seated person's forearm and hand are while reaching on the table). Higher parts of
the arm, for example the upper arm passing over a resting hand, are left out; without this band the
footprint of the whole arm would cover the reaching area all the time. Points above the table are placed
wrongly by about 0.4 mm per mm of height (Section 4.3), so this is a proxy, not a 3-D distance and not a
collision measure.

Perception delay (steps 46, 49): the planner at time t only has the camera data of time t - D. D = 3 steps
(0.19 s) by default, the measured 167 ms from an event to a forecast (Section 4.6) rounded up to the 16 Hz
grid. Every input (observed pose, forecast, r_t) is therefore taken from t - D, and a forecast is read D steps
further ahead, so it describes the same future moments as the robot's plan. Outcomes are always scored on
the person's real position at time t. --delay 0 gives the idealised version without delay.

Robot action modes (step 41): continue (speed x1), slow (x0.5, x0.25), wait (x0); fallback = wait.
Planner (Section VII), for candidate speed a in {0, 0.25, 0.5, 1}:
    J(a) = lg*(1-a) + ls*|a - a_prev| + lw*[a == 0] + lambda_t * a * J_risk(a)
    J_risk(a) = sum_m pi_m * max_k softplus(kappa * (d_req - d_mk(a)))        (mixture-aware)
    d_mk(a)   pairs the person's position at step k-1 (k = 1: the position observed now; then the forecast)
              with the arm's position after k steps at speed a, so the arm keeps away from where the person is
              now as well as where they are going; the worst step of the horizon counts. (A first version
              averaged the risk over the forecast steps only; on D_val that let forecast-based planners move
              while the person was still close, so it was replaced before any held-out evaluation.)
    The risk is scaled by the speed a, as in speed-and-separation monitoring (ISO/TS 15066): moving slower
    near a person is less risky, and a stopped arm adds no risk.
    d_mk(a)   = planar proxy distance between the forecast keypoints of mode m at step k and the arm at the
                position it would reach at step k with speed a
    lambda_t  = l_min + (1 - r_t)(l_max - l_min),   d_req = d_min + (1 - r_t) * dd        (continuous)
    fallback  = wait, when r_t < tau for 3 steps in a row (and until r_t >= tau for 3 steps), or when no
                keypoint is visible. The 3-step rule is declared here because r_t flickers (Section VI).
Methods (steps 45, 56; single-mode forecasts use pi = 1):
    reactive            current pose held for 1 s, fixed margin
    constant velocity   fixed margin
    GRU                 deterministic GRU, fixed margin
    multimodal GRU      mixture risk, fixed margin                         (= A6/A7 without gating)
    ManiCast-style      fine-tuned STS-GCN (cost-aware recipe), fixed margin (A9; elbow and wrist only)
    uncertainty-only    multimodal GRU, lambda/d_req driven by the uncertainty-only reliability score (A5);
                        no fallback, because that score cannot meet the false-trust budget on D_cal
    binary gate         multimodal GRU, smallest margin when trusted, fallback otherwise (A6)
    reliability-guided  multimodal GRU, continuous adaptation to r_t plus fallback (full method)
    reliability, no fallback   the same without the fallback (ablation A11, offline only)
    reliability-guided v2      one declared revision after the first results (2026-10-30): r_t smoothed by a
                        causal exponential moving average (time constant 0.25 s) and the deterministic GRU as the
                        planner's forecast; tau from D_cal for the same 10% budget on the smoothed score.
                        Success criterion fixed before running: on E5 at 3 cycles/min, fewer near-misses than
                        both the GRU with a fixed margin and the reactive planner (s7_tradeoff.py).
    Every method gets its own settings, chosen on D_val by the same rule (lowest mean decision loss):
    fixed-margin methods: margin d in {40, 60, 90, 120, 180} mm x risk weight in {2.5, 5.5, 10};
    adaptive methods: d_min in {40, 60}, dd in {60, 120} mm, l_max in {4, 10} (l_min = 1);
    binary gate: margin in {40, 60, 90} mm x weight in {1, 2.5, 5.5} while trusted.

Evaluation (Section IX), on the realised human positions:
    task progress (pick-and-place cycles per minute), waiting share, fallback count and share,
    minimum planar separation proxy while moving, proxy-threshold violations = steps where the arm MOVES while
    the separation proxy is below 50 mm (lab-defined), per minute. The person was recorded without a robot,
    so they come close to wherever the arm is; closeness to a stopped arm is reported separately.
    decision loss L_dec = 1*(1-a) + 0.5*[a == 0] + 20*[violation] per step (weights fixed here,
    before any testing), and one-step decision regret against an oracle planner that sees the realised
    future (same cost, realised positions, margin d_min).
Data: D_val for choosing the ranges (step 43; r_t there comes from cross-validation folds, never fitted on
the step it scores), tau from D_cal (Section VI), E5 as a stress test. D_test stays locked.

Usage
  python3 s7_replay.py --manicast-dir ~/manicast            # tune on val, then report val and E5
Outputs in logs/replay/: tuning.csv, results.csv, per_condition.csv, summary.json, trace_example.png
"""
import argparse
import csv
import json
import os
import pickle

import cv2
import numpy as np

from s5_baselines import const_vel, current_pose
from s5_data import FORECAST_KP, load, windows
from s5_gru import GRUForecaster, SIZE, features, ffill, predict
from s5_mgru import run as mrun, to_positions
from s6_features import load_mgru
from s7_robot import capsules, ik_down

DT = 0.0625
K = 16
SPEEDS = np.array([0.0, 0.25, 0.5, 1.0])
V_NOM = 150.0             # mm/s along the path
PICK, PLACE = 6, 8
Z_UP, Z_DOWN = 150.0, 30.0
DWELL_S = 0.5
D_SAFE = 50.0
W_G, W_W, W_P = 1.0, 0.5, 20.0
KAPPA = 0.1               # 1/mm
LG, LS, LW = 1.0, 0.2, 0.3
N_HYST = 3
EMA_TAU_S = 0.25
EMA_A = 1 - np.exp(-DT / EMA_TAU_S)
DELAY = 3                 # steps of perception delay (0.19 s); see the docstring
ORACLE_L, ORACLE_D = 5.5, 60.0   # oracle for regret: realised future, fixed weight and margin
H_BAND = 200.0            # mm above the table
FAR = 1e6


# ------------------------------------------------------------------ robot path and footprints
class Path:
    def __init__(self, calib):
        P = {p["id"]: p for p in calib["points"]}
        a = np.array([P[PICK]["X"], P[PICK]["Y"]]); b = np.array([P[PLACE]["X"], P[PLACE]["Y"]])
        zt = lambda i: P[i]["Z"]
        wps = [(*a, zt(PICK) + Z_UP), (*a, zt(PICK) + Z_DOWN), "dwell", (*a, zt(PICK) + Z_UP),
               (*b, zt(PLACE) + Z_UP), (*b, zt(PLACE) + Z_DOWN), "dwell", (*b, zt(PLACE) + Z_UP), (*a, zt(PICK) + Z_UP)]
        pts, last = [], None
        for w in wps:
            if w == "dwell":
                n = int(DWELL_S * V_NOM / 2.0)
                pts += [last] * n
                continue
            w = np.array(w, float)
            if last is not None:
                n = max(1, int(np.linalg.norm(w - last) / 2.0))
                pts += [last + (w - last) * (i + 1) / n for i in range(n)]
            else:
                pts.append(w)
            last = w
        self.xyz = np.array(pts)                      # one sample every 2 mm of path (or dwell)
        self.table_z = float(np.mean([p["Z"] for p in calib["points"]]))
        self.ds = 2.0
        self.S = len(self.xyz)
        self.length = self.S * self.ds
        q = None
        A, B, R, ok = [], [], [], 0
        for p in self.xyz:
            q, good = ik_down(p, q)
            ok += good
            caps = [clip_low(c, self.table_z) for c in capsules(q)]
            A.append([c[0] for c in caps]); B.append([c[1] for c in caps]); R.append([c[2] for c in caps])
        self.A, self.B, self.R = np.array(A), np.array(B), np.array(R)          # (S,nseg,2), (S,nseg)
        self.ik_ok = ok / self.S

    def idx(self, s):
        return (np.asarray(s) / self.ds).astype(int) % self.S

    def dist(self, pts, s):
        """pts (...,2) mm and path positions s broadcastable to pts[...,0]: planar proxy distance (mm)."""
        i = self.idx(s)
        A, B, R = self.A[i], self.B[i], self.R[i]                          # (...,nseg,2)
        p = pts[..., None, :]
        d = B - A
        L2 = (d ** 2).sum(-1)
        t = np.clip(((p - A) * d).sum(-1) / np.maximum(L2, 1e-9), 0, 1)
        q = A + t[..., None] * d
        return (np.linalg.norm(p - q, axis=-1) - R).min(-1)


def clip_low(cap, table_z):
    """Part of a capsule (metres) whose surface comes within H_BAND of the table, as a planar segment in mm.
    A capsule entirely above the band is moved far away (it never counts)."""
    a, b, r = cap[0] * 1000, cap[1] * 1000, cap[2] * 1000
    top = table_z + H_BAND + r                     # the axis may be up to r above the band
    za, zb = a[2], b[2]
    if za > top and zb > top:
        return np.array([FAR, FAR]), np.array([FAR, FAR]), 0.0
    if za > top or zb > top:
        t = (top - za) / (zb - za)
        cut = a + t * (b - a)
        a, b = (cut, b) if za > top else (a, cut)
    return a[:2], b[:2], r


# ------------------------------------------------------------------ human forecasts per session, in mm
def homography(calib):
    uv = np.array([[p["u"], p["v"]] for p in calib["points"]], np.float32)
    xy = np.array([[p["X"], p["Y"]] for p in calib["points"]], np.float32)
    H, _ = cv2.findHomography(uv, xy, 0)
    return H


def to_mm(norm_xy, H):
    px = norm_xy * SIZE
    sh = px.shape
    p = cv2.perspectiveTransform(np.nan_to_num(px).reshape(-1, 1, 2).astype(np.float64), H).reshape(sh)
    return np.where(np.isnan(norm_xy), np.nan, p)


def calib_for(session_dir):
    notes = os.path.join(session_dir, "session_notes.json")
    path = json.load(open(notes))["calib_session_corrected"] if os.path.exists(notes) else \
        json.load(open(os.path.join(session_dir, "meta.json")))["calib_session"]
    return json.load(open(path))


def forecasts(prepared, setname, results, manicast_dir, scores_file):
    """Per session: dict of arrays over the set's anchors (in time order)."""
    m, sess = load(prepared)
    w = windows(m, sess, setname)
    x, cur = features(w)
    mg, _ = load_mgru(results)
    pi, pos = to_positions(mrun(mg, x), cur)                       # (N,M), (N,M,K,J,2) normalised
    gc = json.load(open(os.path.join(results, "gru", "gru_best.json")))
    gru = GRUForecaster(gc["n_in"], gc["config"]["hidden"], gc["config"]["layers"], gc["K"])
    import torch
    gru.load_state_dict(torch.load(os.path.join(results, "gru", "gru_best.pt"), map_location="cpu"))
    g = predict(gru, x, cur)
    cvn = json.load(open(os.path.join(results, "baselines_val.json")))["cv_window_chosen"]
    wf = dict(w); wf["hist"] = ffill(w["hist"], w["hmiss"])
    cvp = const_vel(wf, 24, cvn)
    rea = current_pose(wf, 24)
    sts = None
    if manicast_dir and os.path.exists(os.path.join(manicast_dir, "model", "manicast.py")):
        from s5_eval_e5 import StsGcn
        m0, s0 = load("logs/prepared")
        S = StsGcn(manicast_dir, m0, s0, os.path.join(results, "stsgcn", "stsgcn_finetuned_best.pt"))
        sts = S.run(m, sess, setname, w)
        sts[:, :, 2] = sts[:, :, 1]                                  # no hand joint: use the wrist
    sc = np.load(scores_file)
    rmap = {(s, int(a)): (rc, ru) for s, a, rc, ru in zip(sc["session"], sc["anchor"], sc["r_combined"], sc["r_uncertainty"])}
    out = {}
    for name, info in m["sessions"].items():
        sel = np.where(w["session"] == name)[0]
        if not len(sel):
            continue
        sel = sel[np.argsort(w["anchor"][sel])]
        H = homography(calib_for(info["path"]))
        mm = lambda a: to_mm(a, H)
        rr = np.array([rmap.get((name, int(t)), (np.nan, np.nan)) for t in w["anchor"][sel]])
        out[name] = {
            "anchor": w["anchor"][sel], "cond": info["condition"], "path": info["path"],
            "reach": sess[name]["reach"][w["anchor"][sel]],
            "now": mm(wf["hist"][sel, -1]),                                          # (n,J,2) last seen
            "visible": (w["hmiss"][sel, -1] == 0).any(1),
            "fut_real": mm(np.where(np.isnan(w["fut"][sel, :K]), np.repeat(wf["hist"][sel, -1:], K, 1), w["fut"][sel, :K])),
            "pi": pi[sel], "modes": mm(pos[sel]),
            "gru": mm(g[sel]), "cv": mm(cvp[sel]), "reactive": mm(rea[sel]),
            "sts": mm(np.concatenate([sts[sel], np.repeat(sts[sel][:, -1:], 8, 1)], 1)) if sts is not None else None,
            "r": rr[:, 0], "r_u": rr[:, 1], "r_s": ema(rr[:, 0])}
    return out


def ema(r):
    """Causal exponential moving average of a score series (missing values hold the last average)."""
    out, m = np.zeros(len(r)), None
    for i, v in enumerate(r):
        if v == v:
            m = v if m is None else m + EMA_A * (v - m)
        out[i] = m if m is not None else np.nan
    return out


def tau_smoothed(budget, l_max):
    """tau on D_cal for the smoothed score (each cal session smoothed in time order)."""
    from s6_reliability import choose_tau
    dev = np.load("logs/reliability/features_dev.npz", allow_pickle=True)
    sc = np.load("logs/reliability/scores_cal.npz", allow_pickle=True)
    cal = dev["set"] == "cal"
    L, ses, anc = dev["L_future"][cal], sc["session"], sc["anchor"]
    rs = np.zeros(len(L))
    for sname in np.unique(ses):
        m = np.where(ses == sname)[0]
        o = m[np.argsort(anc[m])]
        rs[o] = ema(sc["r_combined"][o])
    ok = ~np.isnan(L) & ~np.isnan(rs)
    return choose_tau(rs[ok], (L[ok] <= l_max).astype(int), budget)


# ------------------------------------------------------------------ planner
def softplus(x):
    return np.logaddexp(0, x)


def choose(path, s, a_prev, pi, traj, lam, dreq, now=None):
    """traj (M,K,J,2) mm forecast, now (J,2) mm observed, pi (M,). Returns chosen speed index and all costs."""
    if now is not None and np.isfinite(now).all():
        traj = np.concatenate([np.repeat(now[None, None], len(traj), 0), traj[:, :K - 1]], 1)    # (M,K,J,2)
    k = np.arange(1, K + 1)
    sk = s + SPEEDS[:, None] * V_NOM * DT * k[None]                       # (A,K)
    d = path.dist(traj[None, :, :, :, :], sk[:, None, :, None])          # (A,M,K,J)
    d = d.min(-1)                                                          # nearest keypoint
    risk = (pi[None, :] * softplus(KAPPA * (dreq - d)).max(-1)).sum(-1)   # (A,) worst step, mixture-weighted
    J = LG * (1 - SPEEDS) + LS * np.abs(SPEEDS - a_prev) + LW * (SPEEDS == 0) + lam * SPEEDS * risk
    return int(np.argmin(J)), J


def simulate(path, F, method, p, delay=None):
    """Replay one session with one method. Returns per-step records."""
    D = DELAY if delay is None else delay
    n = len(F["anchor"])
    s, a_prev, low, high, fb = 0.0, 1.0, 0, 0, False
    rec = {k: np.zeros(n) for k in ("a", "sep", "fallback", "regret", "progress", "lam", "dreq")}
    mid_l, mid_d = p.get("l_fix", (p.get("l_min", 1) + p.get("l_max", 1)) / 2), p.get("d_fix", p.get("d_min", 0) + p.get("dd", 0) / 2)
    for i in range(n):
        j = max(0, i - D)                                 # the newest data the planner has at step i
        rkey = "r_s" if method == "full2" else "r"
        r = F[rkey][j] if F[rkey][j] == F[rkey][j] else 0.0
        # human positions paired with the arm after 1..K steps: times t .. t+K-1. Forecast methods with a delay
        # take time t from their own forecast; without delay (or for the reactive planner) time t is the
        # newest observation, followed by the forecast.
        if method in ("reactive", "cv", "gru", "sts", "full2"):
            src, pi = F["gru" if method == "full2" else method][j][None], np.ones(1)
        else:
            src, pi = F["modes"][j], F["pi"][j]
        if D > 0 and method != "reactive":
            traj, now_in = src[:, D - 1:D - 1 + K], None
        else:
            traj, now_in = src[:, D:D + K], F["now"][j]
        lam, dreq = mid_l, mid_d
        if method in ("full", "full_nofb", "uncert", "full2"):
            rr = r if method != "uncert" else (F["r_u"][j] if F["r_u"][j] == F["r_u"][j] else 0.0)
            lam = p["l_min"] + (1 - rr) * (p["l_max"] - p["l_min"])
            dreq = p["d_min"] + (1 - rr) * p["dd"]
        if method == "gate":
            lam, dreq = p["l_fix"], p["d_fix"]
        # fallback with hysteresis (full and gate)
        if method in ("full", "gate", "full2"):
            low = low + 1 if r < p["tau"] else 0
            high = high + 1 if r >= p["tau"] else 0
            if not fb and low >= N_HYST:
                fb = True
            elif fb and high >= N_HYST:
                fb = False
            if not F["visible"][j]:
                fb = True
        ai, _ = choose(path, s, a_prev, pi, traj, lam, dreq, now_in)
        a = 0.0 if (fb and method in ("full", "gate", "full2")) else SPEEDS[ai]
        # realised outcome and one-step regret against an oracle that sees the realised future
        sep = float(path.dist(F["now"][i][None], np.array([s]))[0].min()) if np.isfinite(F["now"][i]).all() else np.nan
        _, Jtrue = choose(path, s, a_prev, np.ones(1), F["fut_real"][i][None], ORACLE_L, ORACLE_D, F["now"][i])
        rec["regret"][i] = Jtrue[list(SPEEDS).index(a)] - Jtrue.min()
        rec["a"][i], rec["sep"][i], rec["fallback"][i], rec["lam"][i], rec["dreq"][i] = a, sep, float(fb and method in ("full", "gate", "full2")), lam, dreq
        s += a * V_NOM * DT
        rec["progress"][i] = s
        a_prev = a
    return rec


def summarise(recs, path):
    a = np.concatenate([r["a"] for r in recs]); sep = np.concatenate([r["sep"] for r in recs])
    fb = np.concatenate([r["fallback"] for r in recs]); reg = np.concatenate([r["regret"] for r in recs])
    minutes = len(a) * DT / 60
    cycles = sum(r["progress"][-1] for r in recs) / path.length
    close = np.nan_to_num(sep, nan=1e9) < D_SAFE
    viol = close & (a > 0)
    sep_moving = np.where(a > 0, sep, np.nan)
    episodes = sum(int(((np.diff(r["fallback"]) == 1).sum() + r["fallback"][0])) for r in recs)
    ldec = W_G * (1 - a) + W_W * (a == 0) + W_P * viol
    return {"minutes": round(minutes, 2), "cycles_per_min": round(cycles / minutes, 3), "mean_speed": round(float(a.mean()), 3),
            "wait_share": round(float((a == 0).mean()), 3), "fallback_share": round(float(fb.mean()), 3),
            "fallback_per_min": round(episodes / minutes, 2),
            "min_sep_mm": round(float(np.nanmin(sep_moving)), 1), "sep_p5_mm": round(float(np.nanpercentile(sep_moving, 5)), 1),
            "close_while_stopped_share": round(float((close & (a == 0)).mean()), 4),
            "violations_per_min": round(float(viol.sum() / minutes), 2), "violation_share": round(float(viol.mean()), 4),
            "decision_loss": round(float(ldec.mean()), 4), "regret": round(float(reg.mean()), 4)}


METHODS = {"reactive": "reactive", "cv": "constant velocity", "gru": "GRU", "mgru": "multimodal GRU",
           "sts": "ManiCast-style (STS-GCN)", "uncert": "uncertainty-only adaptation", "gate": "binary gate",
           "full": "reliability-guided (ours)", "full_nofb": "reliability-guided, no fallback (A11)",
           "full2": "reliability-guided v2 (smoothed, GRU)"}

FIXED_GRID = [dict(d_fix=d, l_fix=l) for d in (40.0, 60.0, 90.0, 120.0, 180.0) for l in (2.5, 5.5, 10.0)]
ADAPT_GRID = [dict(d_min=dm, dd=dd, l_min=1.0, l_max=lm) for dm in (40.0, 60.0) for dd in (60.0, 120.0) for lm in (4.0, 10.0)]
GATE_GRID = [dict(d_fix=d, l_fix=l) for d in (40.0, 60.0, 90.0) for l in (1.0, 2.5, 5.5)]


def run_set(path, Fs, method, p):
    return [simulate(path, F, method, p) for F in Fs.values()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manicast-dir", default=os.path.expanduser("~/manicast"))
    ap.add_argument("--results", default="logs/forecast_results")
    ap.add_argument("--out", default="logs/replay")
    ap.add_argument("--delay", type=int, default=3, help="perception delay in 16 Hz steps (0 = idealised)")
    a = ap.parse_args()
    global DELAY
    DELAY = a.delay
    if a.delay != 3 and a.out == "logs/replay":
        a.out = f"logs/replay_delay{a.delay}"
    os.makedirs(a.out, exist_ok=True)
    rel = json.load(open("logs/reliability/summary.json"))
    tau = rel["tau"]["combined (o+u+e)"]
    tau2 = tau_smoothed(rel["ftr_budget_on_cal"], rel["l_max_px"])
    print(f"tau on D_cal (10% false-trust budget): raw score {tau:.3f}, smoothed score (v2) {tau2:.3f}")

    path = Path(json.load(open("calib/calib_cam3d.json")))
    print(f"robot path: {path.length / 10:.0f} cm per cycle (pick dot {PICK} -> place dot {PLACE}), "
          f"IK solved at {path.ik_ok:.0%} of path samples, {path.length / V_NOM:.1f} s per cycle at full speed")
    Fv = forecasts("logs/prepared", "val", a.results, a.manicast_dir, "logs/reliability/scores_val.npz")
    Fe = forecasts("logs/prepared_e5", "e5", a.results, a.manicast_dir, "logs/reliability/scores_e5.npz")
    has_sts = all(F["sts"] is not None for F in list(Fv.values()) + list(Fe.values()))
    methods = [m for m in METHODS if m != "sts" or has_sts]

    # ---- step 43: each method's settings on D_val (lowest mean decision loss)
    best, tune = {}, []
    for mth in methods:
        grid = ADAPT_GRID if mth in ("full", "full_nofb", "uncert", "full2") else GATE_GRID if mth == "gate" else FIXED_GRID
        scored = []
        for g in grid:
            p = dict(g, tau=tau2 if mth == "full2" else tau)
            sm = summarise(run_set(path, Fv, mth, p), path)
            scored.append((sm["decision_loss"], p))
            tune.append([METHODS[mth], json.dumps(g), sm["decision_loss"], sm["violations_per_min"], sm["cycles_per_min"]])
        best[mth] = min(scored, key=lambda t: t[0])[1]
        print(f"  {METHODS[mth]:<40} chosen {json.dumps({k: v for k, v in best[mth].items() if k != 'tau'})}", flush=True)
    with open(os.path.join(a.out, "tuning.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["method", "settings", "decision_loss", "violations_per_min", "cycles_per_min"]); cw.writerows(tune)

    rows, cond_rows, results = [], [], {}
    for setname, Fs in (("val", Fv), ("E5", Fe)):
        for mth in methods:
            recs = run_set(path, Fs, mth, best[mth])
            sm = summarise(recs, path)
            results[(setname, mth)] = (recs, sm)
            rows.append([setname, METHODS[mth]] + [sm[k] for k in ("cycles_per_min", "wait_share", "fallback_share", "fallback_per_min",
                                                                  "min_sep_mm", "sep_p5_mm", "violations_per_min", "decision_loss", "regret")])
            # per condition
            groups = {}
            for (name, F), rc in zip(Fs.items(), recs):
                if setname == "val":
                    groups.setdefault(F["cond"], []).append(rc)
            for c, rc in sorted(groups.items()):
                s2 = summarise(rc, path)
                cond_rows.append([setname, c, METHODS[mth], s2["cycles_per_min"], s2["violations_per_min"], s2["decision_loss"], s2["regret"]])
    hdr = ["set", "method", "cycles_per_min", "wait_share", "fallback_share", "fallback_per_min", "min_sep_mm", "sep_p5_mm",
           "violations_per_min", "decision_loss", "regret"]
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(hdr); cw.writerows(rows)
    with open(os.path.join(a.out, "per_condition.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["set", "condition", "method", "cycles_per_min", "violations_per_min", "decision_loss", "regret"])
        cw.writerows(cond_rows)
    print(f"\n{'set':<5}{'method':<40}{'cycles/min':>11}{'wait':>7}{'fallbk':>8}{'min sep':>9}{'sep p5':>8}{'viol/min':>9}{'L_dec':>8}{'regret':>8}")
    for r in rows:
        print(f"{r[0]:<5}{r[1]:<40}{r[2]:>11.2f}{r[3]:>7.2f}{r[4]:>8.2f}{r[6]:>9.1f}{r[7]:>8.1f}{r[8]:>9.2f}{r[9]:>8.3f}{r[10]:>8.3f}")
    json.dump({"delay_steps": DELAY, "delay_s": DELAY * DT, "path": {"pick_dot": PICK, "place_dot": PLACE, "cycle_mm": path.length, "v_nom_mm_s": V_NOM, "ik_ok": path.ik_ok},
               "chosen_on_val": best, "tau_from_cal": tau, "tau_smoothed_from_cal": tau2, "d_safe_mm": D_SAFE, "loss_weights": [W_G, W_W, W_P],
               "results": {f"{k[0]}/{k[1]}": v[1] for k, v in results.items()}},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    pickle.dump({k: v[0] for k, v in results.items()}, open(os.path.join(a.out, "traces.pkl"), "wb"))
    print(f"\nSaved in {a.out}/")


if __name__ == "__main__":
    main()
