#!/usr/bin/env python3
"""
H3 live trials: online perception -> 16 Hz steps -> forecasts -> reliability, while the arm runs.

Matches the offline pipeline (s5_prepare, s5_gru, s5_mgru, s5_stsgcn, s6_features) step for step,
so a model trained on prepared recordings sees the same inputs live. The one rule that differs from
the original s5_prepare.py: a 16 Hz step may only use detections up to the first camera frame after
it (causal), since live there is no later frame yet. Live recordings are prepared with
`s5_prepare.py --causal`, which uses causal_grid() below, so training and live inputs are identical.
Checked against the offline code by s9_selftest.py.
"""
import json
import os
import threading
from collections import deque

import numpy as np
import torch

DT = 0.0625            # s per step (16 Hz)
MAX_GAP = 0.25         # s; gaps longer than this stay missing
KP = ["shoulder", "elbow", "wrist", "hand", "tip"]     # same order as s5_prepare.KP
RAW_KP = ["L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist"]   # read by STS-GCN
KEEP_STEPS = 40        # 16 Hz steps kept: 1.0 s history + 1.0 s of older forecasts for e_t + margin
KEEP_FRAMES_S = 2.0    # raw frames kept (s)


# ------------------------------------------------------------------ the causal resampling rule
def causal_grid(tr, U, V, C, OBS, tg, max_gap=MAX_GAP):
    """Detections -> 16 Hz steps, using only frames up to the first frame after each step.

    tr (F,) frame times; U, V, C (F,J) pixels and confidence; OBS (F,J) observed flags; every tg < tr[-1].
    Same as s5_prepare.resample_session, except a missing point is only filled between two
    detections if the second one is in the very next frame. Returns u, v (NaN = missing), conf, miss.
    """
    T, J = len(tg), U.shape[1]
    u, v = np.full((T, J), np.nan), np.full((T, J), np.nan)
    conf, miss = np.zeros((T, J)), np.ones((T, J), np.uint8)
    nf = np.searchsorted(tr, tg, side="right")             # index of the first frame after each step
    for j in range(J):
        conf[:, j] = np.interp(tg, tr, np.nan_to_num(C[:, j]))
        oi = np.flatnonzero(OBS[:, j])
        if len(oi) < 2:
            continue
        to = tr[oi]
        i = np.searchsorted(to, tg, side="right")            # first detection after the step
        prev_ok = i > 0
        ip, inx = np.clip(i - 1, 0, len(oi) - 1), np.clip(i, 0, len(oi) - 1)
        exact = prev_ok & (to[ip] == tg)
        nxt_ok = (i < len(oi)) & (oi[inx] == nf)              # the next detection is in the next frame
        ok = exact | (prev_ok & nxt_ok & (to[inx] - to[ip] <= max_gap))
        u[ok, j] = np.interp(tg[ok], to, U[oi, j])
        v[ok, j] = np.interp(tg[ok], to, V[oi, j])
        miss[ok, j] = 0
    return u, v, conf, miss


def causal_at(to, u, v, tq, cut, max_gap=MAX_GAP):
    """One raw keypoint at times tq (STS-GCN input), using only detections at or before `cut`
    (broadcastable to tq) to fill a gap. Same as s5_stsgcn.RawTracks.at otherwise."""
    out = np.full(tq.shape + (2,), np.nan)
    if len(to) < 2:
        return out
    i = np.searchsorted(to, tq, side="right")
    ip, inx = np.clip(i - 1, 0, len(to) - 1), np.clip(i, 0, len(to) - 1)
    exact = (i > 0) & (to[ip] == tq)
    ok = exact | ((i > 0) & (i < len(to)) & (to[inx] - to[ip] <= max_gap) & (to[inx] <= cut))
    out[ok, 0] = np.interp(tq[ok], to, u)
    out[ok, 1] = np.interp(tq[ok], to, v)
    return out


# ------------------------------------------------------------------ the live 16 Hz buffer
class OnlineGrid:
    """Collects detections from the recorder (one call per camera frame) and turns them into
    16 Hz steps as soon as each step is final (the first frame after it has arrived)."""

    def __init__(self, side, W, H):
        self.side, self.W, self.H = side, W, H
        self.lock = threading.Lock()
        self.fr = deque()            # raw frames: (t, u(5), v(5), c(5), obs(5), raw{name: (u, v, obs)}, phase, goal)
        self.t0, self.k = None, 0    # first frame time (grid origin) and the next step index
        self.steps = deque(maxlen=KEEP_STEPS)
        self.cues = []
        self.n_steps = 0             # steps made so far (the anchor index of the newest step + 1)

    def add_cue(self, t):
        with self.lock:
            self.cues.append(round(t, 6))

    def add_frame(self, t, kp, miss, phase, goal):
        """kp {name: (u, v, conf)} in full-size pixels, miss {name: 0/1}, the phase and goal written
        to frames.csv for this frame. Rounded exactly as frames.csv stores them."""
        if phase == "setup":          # s5_prepare drops the frames before the first touch
            return 0
        t = round(t, 6)

        def val(name):
            u_, v_, c_ = kp[name]
            u_ = round(u_, 2) if u_ == u_ else np.nan
            v_ = round(v_, 2) if v_ == v_ else np.nan
            return u_, v_, round(c_, 4), (miss[name] == 0) and u_ == u_
        cols = [val(f"{self.side}_{k}") for k in KP]
        raw = {n: val(n) for n in RAW_KP}
        with self.lock:
            self.fr.append((t, np.array([c[0] for c in cols]), np.array([c[1] for c in cols]),
                            np.array([c[2] for c in cols]), np.array([c[3] for c in cols]),
                            raw, phase, goal))
            while self.fr and self.fr[0][0] < t - KEEP_FRAMES_S:
                self.fr.popleft()
            if self.t0 is None:
                self.t0 = t
            tg = []
            while self.t0 + self.k * DT < t:
                tg.append(self.t0 + self.k * DT)
                self.k += 1
            if not tg:
                return 0
            tg = np.array(tg)
            tr = np.array([f[0] for f in self.fr])
            U, V, C, O = (np.stack([f[i] for f in self.fr]) for i in (1, 2, 3, 4))
            u, v, conf, m = causal_grid(tr, U, V, C, O.astype(bool), tg)
            ir = np.clip(np.searchsorted(tr, tg, side="right") - 1, 0, len(tr) - 1)
            reach = np.searchsorted(np.array(self.cues), tg, side="right")
            for q in range(len(tg)):
                f = self.fr[ir[q]]
                self.steps.append({"t": float(tg[q]), "y": np.stack([u[q] / self.W, v[q] / self.H], -1),
                                   "conf": conf[q], "miss": m[q], "phase": f[6],
                                   "goal": int(f[7]) if f[7] not in ("", None) else -1,
                                   "reach": int(reach[q]), "cut": t, "anchor": self.n_steps})
                self.n_steps += 1
            return len(tg)

    def snapshot(self):
        """Copies of the kept steps and raw frames (for the model thread)."""
        with self.lock:
            return list(self.steps), list(self.fr)


# ------------------------------------------------------------------ forecasts and reliability, live
class OnlineModels:
    """Every forecaster and both reliability scores for the newest 16 Hz step, in robot mm.

    Without trained models (Phase A, before anything is trained) only the current pose is
    available, which is all the reactive planner needs."""

    def __init__(self, calib, results=None, reliability=None, manicast_dir=None, prepared_train=None,
                 side="R", need=("sts", "rel")):
        from s7_replay import homography
        self.H = homography(calib)
        self.side = side
        self.L, self.K = 16, 24
        self.jidx = [1, 2, 3]                           # elbow, wrist, hand in KP
        self.have_models = bool(results) and os.path.exists(os.path.join(results, "gru", "gru_best.pt"))
        self.ema, self.last_anchor = None, -1
        self.rel = None
        self.sts = None
        if not self.have_models:
            return
        from s5_gru import GRUForecaster
        from s6_features import load_mgru
        self.mg, c = load_mgru(results)
        self.mg.eval()
        self.L = 16
        self.K = c["K"]
        gc = json.load(open(os.path.join(results, "gru", "gru_best.json")))
        self.gru = GRUForecaster(gc["n_in"], gc["config"]["hidden"], gc["config"]["layers"], gc["K"])
        self.gru.load_state_dict(torch.load(os.path.join(results, "gru", "gru_best.pt"), map_location="cpu"))
        self.gru.eval()
        self.cvn = json.load(open(os.path.join(results, "baselines_val.json")))["cv_window_chosen"]
        if "rel" in need and reliability and os.path.exists(os.path.join(reliability, "reliability_model.pkl")):
            import pickle
            self.rel = pickle.load(open(os.path.join(reliability, "reliability_model.pkl"), "rb"))
        if "sts" in need and manicast_dir and os.path.exists(os.path.join(manicast_dir, "model", "manicast.py")):
            from s5_data import load
            from s5_eval_e5 import StsGcn
            m0, s0 = load(prepared_train)
            self.sts = StsGcn(manicast_dir, m0, s0, os.path.join(results, "stsgcn", "stsgcn_finetuned_best.pt"))
            self.sts.net.eval()

    def mm(self, a):
        from s7_replay import to_mm
        return to_mm(a, self.H)

    def record(self, steps, frames):
        """The planner's inputs for the newest step: the same fields s7_replay.forecasts() gives for
        one anchor (now, visible, pi, modes, gru, cv, reactive, sts, r, r_u, r_s), plus its time."""
        from s5_baselines import const_vel, current_pose
        from s5_gru import ffill
        if len(steps) < self.L:
            return None
        n = len(steps)
        s = {"t": np.array([q["t"] for q in steps]), "y": np.stack([q["y"] for q in steps]),
             "conf": np.stack([q["conf"] for q in steps]), "miss": np.stack([q["miss"] for q in steps]),
             "phase": np.array([q["phase"] for q in steps]), "reach": np.array([q["reach"] for q in steps])}
        last = steps[-1]
        # the single window ending at the newest step, as s5_data.windows builds it
        hi = np.arange(n - self.L, n)
        w = {"hist": s["y"][hi][:, self.jidx][None], "hmiss": s["miss"][hi][:, self.jidx][None],
             "hconf": s["conf"][hi][:, self.jidx][None]}
        wf = dict(w); wf["hist"] = ffill(w["hist"], w["hmiss"])
        rec = {"t": last["t"], "anchor": last["anchor"], "cut": last["cut"],
               "now": self.mm(wf["hist"][0, -1]), "visible": bool((w["hmiss"][0, -1] == 0).any()),
               "reactive": self.mm(current_pose(wf, 24)[0])}
        if not self.have_models:
            return rec
        from s5_gru import features, predict
        from s5_mgru import run as mrun, to_positions
        with torch.no_grad():
            x, cur = features(w)
            rec["gru"] = self.mm(predict(self.gru, x, cur)[0])
            pi, pos = to_positions(mrun(self.mg, x), cur)
        rec["pi"], rec["modes"] = pi[0], self.mm(pos[0])
        rec["cv"] = self.mm(const_vel(wf, 24, self.cvn)[0])
        if self.sts is not None:
            rec["sts"] = self.mm(self.sts_one(last, frames))
        if self.rel is not None:
            self._reliability(rec, s, steps)
        return rec

    def _reliability(self, rec, s, steps):
        """r (combined), r_u (uncertainty only) and r_s (smoothed r, for v2) for the newest step,
        from s6_features.session_table run over the kept steps (it only looks back in time)."""
        from s6_features import session_table
        from s6_reliability import X
        T = len(s["t"])
        glob = [q["anchor"] for q in steps]
        new = [i for i in range(self.L - 1, T) if glob[i] > self.last_anchor] or [T - 1]
        # only the new steps and the steps their recent errors look back to (4, 8, 16 steps);
        # every feature only looks back in time, so this gives the same values, much faster
        allowed = {i - j: "live" for i in new for j in (0, 4, 8, 16) if i - j >= self.L - 1}
        with torch.no_grad():
            f, meta = session_table(self.mg, s, self.L, self.K, self.jidx, allowed)
        row = {int(a_): k for k, a_ in enumerate(meta["anchor"])}
        out = {}
        for key, name in (("r", "combined (o+u+e)"), ("r_u", "uncertainty-only (u)")):
            mdl, cal, tau, cols = self.rel["estimators"][name]
            out[key] = cal.predict(mdl.predict_proba(X(f, cols))[:, 1])
        # smoothed score: update once per new step, in order (as s7_replay.ema does offline)
        from s7_replay import EMA_A
        for i in new:
            v = out["r"][row[i]]
            if v == v:
                self.ema = v if self.ema is None else self.ema + EMA_A * (v - self.ema)
            self.last_anchor = glob[i]
        rec["r"], rec["r_u"] = float(out["r"][row[T - 1]]), float(out["r_u"][row[T - 1]])
        rec["r_s"] = float(self.ema) if self.ema is not None else np.nan

    def sts_one(self, last, frames):
        """Fine-tuned STS-GCN forecast for the newest step, (24,3,2) normalised (hand = wrist)."""
        import s5_stsgcn as S
        offs = np.arange(-(S.N_IN - 1), S.N_OUT + 1) * S.DT
        tq = last["t"] + offs[:S.N_IN]
        fr = [f for f in frames if f[0] <= last["cut"]]
        p = {}
        for name in RAW_KP:
            to = np.array([f[0] for f in fr if f[5][name][3]])
            uu = np.array([f[5][name][0] for f in fr if f[5][name][3]])
            vv = np.array([f[5][name][1] for f in fr if f[5][name][3]])
            p[name] = causal_at(to, uu, vv, tq, last["cut"])
        bt = (p["L_shoulder"] + p["R_shoulder"]) / 2
        hist = np.stack([bt] + [p[n] for n in S.OURS[1:]], axis=-2)              # (10,7,2)
        P = np.concatenate([hist, np.full((S.N_OUT, 7, 2), np.nan)])[None]       # future unknown
        X, _, _, origin, usable = S.tensors(self.sts.fr, P)
        cur = np.nan_to_num(self.sts.fr.fill_history(P)[:, -1])
        with torch.no_grad():
            q = S.to_eval16(self.sts.fr, S.forward(self.sts.net, X), origin, cur)
        if not usable[0]:
            q[:, :, :2] = np.repeat((cur[:, [S.R_ELBOW, S.R_WRIST]] / S.SIZE)[:, None], 16, 1)
        q[:, :, 2] = q[:, :, 1]                                                   # no hand joint: use the wrist
        return np.concatenate([q[0], np.repeat(q[0][-1:], 8, 0)], 0)
