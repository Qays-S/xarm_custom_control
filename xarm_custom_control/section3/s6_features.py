"""
Section VI, steps 33-36: build the reliability features and labels.

Runs the frozen multimodal GRU (s5_mgru.py) at every time step of every session and
turns its output into one row per time step: the online features o_t, u_t, e_t
(things the robot could know at time t) and the offline label (how good the
forecast actually turned out to be). Used by s6_reliability.py and s8_locked_test.py.
"""
import json
import math
import os

import numpy as np
import torch

from s5_data import FORECAST_KP, load
from s5_gru import SIZE, features
from s5_mgru import LN100, MixtureGRU, point_logpdf, run

# Earlier forecasts we look back at, in 16 Hz steps: 4 = 0.25 s, 8 = 0.5 s, 16 = 1.0 s ago.
LAGS = (4, 8, 16)
K_LABEL = 16          # the label looks 1.0 s (16 steps) into the future

# Column names, grouped the same way as in the paper.
# o_t: observation quality (how good the camera's view is right now)
O_COLS = ["conf_now_mean", "conf_now_min", "conf_recent_mean", "miss_now", "miss_hist",
          "steps_since_miss", "jitter_px", "max_jump_px"]
# u_t: forecast uncertainty (how unsure the forecast itself is)
U_COLS = ["top_prob", "mode_entropy", "within_std_wrist_px", "within_std_hand_px",
          "between_wrist_px", "between_hand_px"]
# e_t: recent errors (how far off the forecasts made 0.25, 0.5 and 1.0 s ago turned out)
E_COLS = [f"resid_{j}_px" for j in LAGS] + [f"nll_{j}" for j in LAGS]


def load_mgru(results="logs/forecast_results"):
    """Load the trained multimodal GRU and its settings (it is frozen from here on)."""
    c = json.load(open(os.path.join(results, "mgru", "mgru_best.json")))
    m = MixtureGRU(c["n_in"], c["config"]["hidden"], c["config"]["layers"], c["K"], c["M"])
    m.load_state_dict(torch.load(os.path.join(results, "mgru", "mgru_best.pt"),
                                 map_location="cpu"))
    return m, c


def session_windows(s, L, K, jidx):
    """Windows anchored at every step L-1 .. T-1 of one session (future padded with NaN/missing)."""
    T = len(s["t"])
    y, miss, conf = s["y"][:, jidx], s["miss"][:, jidx].astype(int), s["conf"][:, jidx]
    # Pad the end with K "missing" steps so the last anchors still get a full-length future.
    yp = np.concatenate([y, np.full((K,) + y.shape[1:], np.nan)])
    mp = np.concatenate([miss, np.ones((K,) + miss.shape[1:], int)])
    idx = np.arange(L - 1, T)                          # one anchor per time step
    hi = idx[:, None] + np.arange(-L + 1, 1)[None]     # history indices (L steps up to the anchor)
    fi = idx[:, None] + np.arange(1, K + 1)[None]      # future indices (the K steps after it)
    return dict(hist=y[hi], hmiss=miss[hi], hconf=conf[hi], fut=yp[fi], fmiss=mp[fi],
                anchor=idx, phase=s["phase"][idx], reach=s["reach"][idx])


def session_table(model, s, L, K, jidx, allowed):
    """Features and labels for every anchor of one session. allowed: anchor -> set name (only those rows kept)."""
    w = session_windows(s, L, K, jidx)
    x, cur = features(w)
    out = run(model, x)
    logpi, mu, a, b, off = (t.numpy() for t in out)
    pi = np.exp(logpi)                                 # mode probabilities, (N, M)
    N = len(x)
    # Forecast positions of every mode, back in normalised image units.
    pos = cur[:, None, None] + mu * 100.0 / SIZE                       # (N,M,K,J,2) normalised
    top = pos[np.arange(N), pi.argmax(1)]                               # (N,K,J,2) most likely mode
    hist_px = np.where(np.isnan(w["hist"]), np.nan, w["hist"]) * SIZE
    from s5_gru import ffill
    hf = ffill(w["hist"], w["hmiss"]) * SIZE           # history in px, gaps filled with the last seen point

    f = {}
    # ---- o_t: observation quality, over the elbow, wrist and hand (step 33)
    f["conf_now_mean"] = w["hconf"][:, -1].mean(1)               # detector confidence now
    f["conf_now_min"] = w["hconf"][:, -1].min(1)
    f["conf_recent_mean"] = w["hconf"][:, -4:].mean((1, 2))      # mean over the last 0.25 s
    f["miss_now"] = w["hmiss"][:, -1].sum(1).astype(float)       # how many of the 3 points are missing now
    f["miss_hist"] = w["hmiss"].mean((1, 2))                     # share missing over the 1.0 s history
    # Steps since any keypoint was last missing (L = never missing in the history).
    anym = w["hmiss"].any(2)                                            # (N,L)
    last = np.where(anym.any(1), L - 1 - (L - 1 - np.argmax(anym[:, ::-1], 1)), -1)
    f["steps_since_miss"] = np.where(anym.any(1), (L - 1) - last, L).astype(float)
    # Jitter: RMS second difference over the last 5 steps (median over the keypoints).
    d2 = np.diff(hf[:, -5:], n=2, axis=1)                               # (N,3,J,2)
    f["jitter_px"] = np.median(np.sqrt((np.linalg.norm(d2, axis=-1) ** 2).mean(1)), 1)
    # Largest single-step move in the last 4 steps; catches jumps and left/right swaps.
    f["max_jump_px"] = np.linalg.norm(np.diff(hf[:, -5:], axis=1), axis=-1).max((1, 2))

    # ---- u_t: forecast uncertainty, from the forecast issued now (step 34)
    f["top_prob"] = pi.max(1)                                    # probability of the most likely mode
    f["mode_entropy"] = -(pi * np.log(pi + 1e-12)).sum(1)        # high = the modes disagree
    k = K_LABEL - 1                                              # look at the 1.0 s step
    it = pi.argmax(1)
    # Spread within the most likely mode (geometric mean of the 2-D std), in px.
    gstd = np.sqrt(a[np.arange(N), it, k] * b[np.arange(N), it, k]) * 100.0   # (N,J) px
    f["within_std_wrist_px"], f["within_std_hand_px"] = gstd[:, 1], gstd[:, 2]
    # Spread between the modes: probability-weighted distance of each mode from their mean.
    mean_k = (pi[:, :, None, None] * pos[:, :, k]).sum(1)               # (N,J,2)
    dev = ((pos[:, :, k] - mean_k[:, None]) * SIZE) ** 2                # (N,M,J,2)
    spread = np.sqrt((pi[:, :, None] * dev.sum(-1)).sum(1))            # (N,J)
    f["between_wrist_px"], f["between_hand_px"] = spread[:, 1], spread[:, 2]

    # ---- e_t: recent errors (step 35). The forecasts issued j steps ago are now due,
    # so compare them with where the arm actually is now.
    ycur = w["hist"][:, -1]                                             # (N,J,2) normalised, NaN if missing
    obs = w["hmiss"][:, -1] == 0
    anchors = w["anchor"]
    row = {int(t): i for i, t in enumerate(anchors)}
    setof = np.array([allowed.get(int(t), "") for t in anchors])
    for j in LAGS:
        rj, nj = np.full(N, np.nan), np.full(N, np.nan)
        for i, t in enumerate(anchors):
            p = row.get(int(t) - j)                    # row of the forecast issued j steps ago
            # Only use it if it is in the same split block (no leaks across blocks)
            # and at least one keypoint is visible now; otherwise leave NaN.
            if p is None or setof[i] == "" or setof[p] != setof[i] or not obs[i].any():
                continue
            o = obs[i]
            # resid_j: distance between that forecast's most likely mode and the arm now (px).
            rj[i] = np.linalg.norm((top[p, j - 1][o] - ycur[i][o]) * SIZE, axis=-1).mean()
            # nll_j: how unlikely the arm's position now was under the whole mixture
            # (all modes, not just the top one), in nats per point.
            dsp = torch.from_numpy(((ycur[i] - cur[p]) * SIZE / 100.0)[None, None, None]
                                   .astype(np.float32))  # (1,1,1,J,2)
            lp = point_logpdf(dsp,
                              torch.from_numpy(mu[p:p + 1, :, j - 1:j]),
                              torch.from_numpy(a[p:p + 1, :, j - 1:j]),
                              torch.from_numpy(b[p:p + 1, :, j - 1:j]),
                              torch.from_numpy(off[p:p + 1, :, j - 1:j]))[0, :, 0]
            mix = torch.logsumexp(torch.from_numpy(logpi[p])[:, None] + lp, 0).numpy()     # (J,)
            nj[i] = float(-(mix[o]).mean() + 2 * LN100)
        f[f"resid_{j}_px"], f[f"nll_{j}"] = rj, nj

    # ---- label (step 36): only from the real future, never used as an input.
    # L_future = mean error (px) of the most likely mode over the next 1.0 s, averaged over
    # elbow, wrist and hand. Defined only when the current pose and that whole future were seen.
    # The 0/1 label is L_future <= l_max, applied in s6_reliability.py.
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
        # Which time steps of this session belong to the requested sets (train/val/...).
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
