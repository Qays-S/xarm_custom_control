"""
Blueprint Section VI, steps 33-36: online reliability features and offline labels.

For every session, the multimodal GRU (s5_mgru.py, frozen) is run at EVERY time step, so that at time t
we also have the forecasts that were issued at t-4, t-8 and t-16 (0.25, 0.5 and 1.0 s earlier).

Online features at time t (only information available at t):
  o_t  observation quality (step 33), over the elbow, wrist and hand
       conf_now_mean, conf_now_min   detector confidence at t
       conf_recent_mean              mean confidence over the last 4 steps (0.25 s)
       miss_now                      number of the three keypoints missing at t
       miss_hist                     share of missing points over the 1.0 s history
       steps_since_miss              steps since any keypoint was last missing (16 = not in the history)
       jitter_px                     RMS second difference over the last 5 steps (median over keypoints)
       max_jump_px                   largest single-step move over the last 4 steps (catches jumps/swaps)
  u_t  predictive uncertainty (step 34), from the forecast issued at t
       top_prob, mode_entropy        mode probabilities
       within_std_wrist/hand_px      spread of the most likely mode at 1.0 s (geometric mean of the 2-D std)
       between_wrist/hand_px         probability-weighted spread of the mode means at 1.0 s
  e_t  retrospective consistency (step 35), from earlier forecasts now that their time has come
       resid_j_px   distance between the most likely mode of the forecast issued at t-j and the
                    observed position at t (mean over observed keypoints), j = 4, 8, 16
       nll_j        mixture negative log predictive density of the observed position at t under the
                    forecast issued at t-j (mixture-aware, no collapse to one Gaussian), nats per point, px units
       An earlier forecast is used only if its own window lies in the same split block (so no history
       from another block leaks in); otherwise the feature is missing (NaN).
  Robot and task state are not used: the robot was parked and the goal is not known to the robot.

Offline label (step 36), from the realised future only, never an online input:
  L_future  mean ADE (px) of the most likely mode over elbow, wrist and hand for the next 1.0 s,
            defined only when the current pose and that whole future were observed
  z_rel     1 if L_future <= l_max (l_max is fixed in s6_reliability.py before any testing)
"""
import json
import math
import os

import numpy as np
import torch

from s5_data import FORECAST_KP, load
from s5_gru import SIZE, features
from s5_mgru import LN100, MixtureGRU, point_logpdf, run

LAGS = (4, 8, 16)
K_LABEL = 16          # 1.0 s
O_COLS = ["conf_now_mean", "conf_now_min", "conf_recent_mean", "miss_now", "miss_hist", "steps_since_miss",
          "jitter_px", "max_jump_px"]
U_COLS = ["top_prob", "mode_entropy", "within_std_wrist_px", "within_std_hand_px", "between_wrist_px", "between_hand_px"]
E_COLS = [f"resid_{j}_px" for j in LAGS] + [f"nll_{j}" for j in LAGS]


def load_mgru(results="logs/forecast_results"):
    c = json.load(open(os.path.join(results, "mgru", "mgru_best.json")))
    m = MixtureGRU(c["n_in"], c["config"]["hidden"], c["config"]["layers"], c["K"], c["M"])
    m.load_state_dict(torch.load(os.path.join(results, "mgru", "mgru_best.pt"), map_location="cpu"))
    return m, c


def session_windows(s, L, K, jidx):
    """Windows anchored at every step L-1 .. T-1 of one session (future padded with NaN/missing)."""
    T = len(s["t"])
    y, miss, conf = s["y"][:, jidx], s["miss"][:, jidx].astype(int), s["conf"][:, jidx]
    yp = np.concatenate([y, np.full((K,) + y.shape[1:], np.nan)])
    mp = np.concatenate([miss, np.ones((K,) + miss.shape[1:], int)])
    idx = np.arange(L - 1, T)
    hi = idx[:, None] + np.arange(-L + 1, 1)[None]
    fi = idx[:, None] + np.arange(1, K + 1)[None]
    return dict(hist=y[hi], hmiss=miss[hi], hconf=conf[hi], fut=yp[fi], fmiss=mp[fi], anchor=idx,
                phase=s["phase"][idx], reach=s["reach"][idx])


def session_table(model, s, L, K, jidx, allowed):
    """Features and labels for every anchor of one session. allowed: anchor -> set name (only those rows kept)."""
    w = session_windows(s, L, K, jidx)
    x, cur = features(w)
    out = run(model, x)
    logpi, mu, a, b, off = (t.numpy() for t in out)
    pi = np.exp(logpi)
    N = len(x)
    pos = cur[:, None, None] + mu * 100.0 / SIZE                       # (N,M,K,J,2) normalised
    top = pos[np.arange(N), pi.argmax(1)]                               # (N,K,J,2)
    hist_px = np.where(np.isnan(w["hist"]), np.nan, w["hist"]) * SIZE
    from s5_gru import ffill
    hf = ffill(w["hist"], w["hmiss"]) * SIZE

    f = {}
    # ---- o_t
    f["conf_now_mean"] = w["hconf"][:, -1].mean(1)
    f["conf_now_min"] = w["hconf"][:, -1].min(1)
    f["conf_recent_mean"] = w["hconf"][:, -4:].mean((1, 2))
    f["miss_now"] = w["hmiss"][:, -1].sum(1).astype(float)
    f["miss_hist"] = w["hmiss"].mean((1, 2))
    anym = w["hmiss"].any(2)                                            # (N,L)
    last = np.where(anym.any(1), L - 1 - (L - 1 - np.argmax(anym[:, ::-1], 1)), -1)
    f["steps_since_miss"] = np.where(anym.any(1), (L - 1) - last, L).astype(float)
    d2 = np.diff(hf[:, -5:], n=2, axis=1)                               # (N,3,J,2)
    f["jitter_px"] = np.median(np.sqrt((np.linalg.norm(d2, axis=-1) ** 2).mean(1)), 1)
    f["max_jump_px"] = np.linalg.norm(np.diff(hf[:, -5:], axis=1), axis=-1).max((1, 2))
    # ---- u_t
    f["top_prob"] = pi.max(1)
    f["mode_entropy"] = -(pi * np.log(pi + 1e-12)).sum(1)
    k = K_LABEL - 1
    it = pi.argmax(1)
    gstd = np.sqrt(a[np.arange(N), it, k] * b[np.arange(N), it, k]) * 100.0   # (N,J) px
    f["within_std_wrist_px"], f["within_std_hand_px"] = gstd[:, 1], gstd[:, 2]
    mean_k = (pi[:, :, None, None] * pos[:, :, k]).sum(1)               # (N,J,2)
    dev = ((pos[:, :, k] - mean_k[:, None]) * SIZE) ** 2                # (N,M,J,2)
    spread = np.sqrt((pi[:, :, None] * dev.sum(-1)).sum(1))            # (N,J)
    f["between_wrist_px"], f["between_hand_px"] = spread[:, 1], spread[:, 2]
    # ---- e_t (earlier forecasts vs the position observed now)
    ycur = w["hist"][:, -1]                                             # (N,J,2) normalised, NaN if missing
    obs = w["hmiss"][:, -1] == 0
    anchors = w["anchor"]
    row = {int(t): i for i, t in enumerate(anchors)}
    setof = np.array([allowed.get(int(t), "") for t in anchors])
    for j in LAGS:
        rj, nj = np.full(N, np.nan), np.full(N, np.nan)
        for i, t in enumerate(anchors):
            p = row.get(int(t) - j)
            if p is None or setof[i] == "" or setof[p] != setof[i] or not obs[i].any():
                continue
            o = obs[i]
            rj[i] = np.linalg.norm((top[p, j - 1][o] - ycur[i][o]) * SIZE, axis=-1).mean()
            dsp = torch.from_numpy(((ycur[i] - cur[p]) * SIZE / 100.0)[None, None, None].astype(np.float32))  # (1,1,1,J,2)
            lp = point_logpdf(dsp, torch.from_numpy(mu[p:p + 1, :, j - 1:j]), torch.from_numpy(a[p:p + 1, :, j - 1:j]),
                              torch.from_numpy(b[p:p + 1, :, j - 1:j]), torch.from_numpy(off[p:p + 1, :, j - 1:j]))[0, :, 0]
            mix = torch.logsumexp(torch.from_numpy(logpi[p])[:, None] + lp, 0).numpy()     # (J,)
            nj[i] = float(-(mix[o]).mean() + 2 * LN100)
        f[f"resid_{j}_px"], f[f"nll_{j}"] = rj, nj
    # ---- label
    ok = (w["hmiss"][:, -1].sum(1) == 0) & (w["fmiss"][:, :K_LABEL].sum((1, 2)) == 0)
    err = np.linalg.norm((top[:, :K_LABEL] - w["fut"][:, :K_LABEL]) * SIZE, axis=-1)     # (N,k,J)
    f["L_future"] = np.where(ok, err.mean((1, 2)), np.nan)
    f["L_hand"] = np.where(ok, err[:, :, 2].mean(1), np.nan)
    keep = setof != ""
    meta = dict(anchor=anchors[keep], set=setof[keep], phase=w["phase"][keep], reach=w["reach"][keep])
    return {k: v[keep] for k, v in f.items()}, meta


def build(prepared, sets, results="logs/forecast_results"):
    """One table (dict of columns) for the given sets of a prepared folder."""
    model, c = load_mgru(results)
    m, sess = load(prepared)
    L, K = m["history_steps"], c["K"]
    jidx = [m["keypoints"].index(k) for k in FORECAST_KP]
    cols = {}
    for name, info in m["sessions"].items():
        allowed = {}
        for sn in sets:
            for t in info["anchor_lists"].get(sn, []):
                allowed[int(t)] = sn
        if not allowed:
            continue
        f, meta = session_table(model, sess[name], L, K, jidx, allowed)
        f.update(meta)
        f["session"] = np.array([name] * len(meta["anchor"]))
        f["cond"] = np.array([info["condition"]] * len(meta["anchor"]))
        for k, v in f.items():
            cols.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in cols.items()}
