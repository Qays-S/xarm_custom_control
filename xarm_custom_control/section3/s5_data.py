"""Shared loading for Section V: windows from logs/prepared (made by s5_prepare.py)."""
import json
import os

import numpy as np

FORECAST_KP = ["elbow", "wrist", "hand"]          # y_t in the paper (blueprint Section III)
HORIZONS_S = [0.25, 0.5, 1.0, 1.5]                 # blueprint step 27


def load(prepared="logs/prepared"):
    m = json.load(open(os.path.join(prepared, "split_manifest.json")))
    sess = {k: dict(np.load(os.path.join(prepared, k + ".npz"), allow_pickle=False)) for k in m["sessions"]}
    return m, sess


def windows(m, sess, setname, L=None, K=None):
    """All windows of one set.

    Returns a dict of arrays, one row per window:
      hist  (N,L,3,2)  normalised u,v of elbow, wrist, hand (NaN = missing)
      hmiss (N,L,3)    missingness of the history
      hconf (N,L,3)    detector confidence of the history
      fut   (N,K,3,2)  future positions (NaN = missing)
      fmiss (N,K,3)
      cond, session, phase (N,) labels of the anchor step; anchor (N,) its step index
    """
    L = L or m["history_steps"]
    K = K or m["max_horizon_steps"]
    j = [m["keypoints"].index(k) for k in FORECAST_KP]
    out = {k: [] for k in ("hist", "hmiss", "hconf", "fut", "fmiss", "cond", "session", "phase", "anchor")}
    for name, info in m["sessions"].items():
        idx = np.array(info["anchor_lists"].get(setname, []), int)
        if len(idx) == 0:
            continue
        s = sess[name]
        y, miss, conf = s["y"][:, j], s["miss"][:, j], s["conf"][:, j]
        hi = idx[:, None] + np.arange(-L + 1, 1)[None]
        fi = idx[:, None] + np.arange(1, K + 1)[None]
        out["hist"].append(y[hi]); out["hmiss"].append(miss[hi]); out["hconf"].append(conf[hi])
        out["fut"].append(y[fi]); out["fmiss"].append(miss[fi])
        out["cond"] += [info["condition"]] * len(idx)
        out["session"] += [name] * len(idx)
        out["phase"].append(s["phase"][idx]); out["anchor"].append(idx)
    return {k: (np.concatenate(v) if k not in ("cond", "session") else np.array(v)) for k, v in out.items()}


def eval_mask(w, k):
    """Windows that can be scored at horizon k steps: current pose and future 1..k all observed."""
    return (w["hmiss"][:, -1].sum(1) == 0) & (w["fmiss"][:, :k].sum((1, 2)) == 0)


def ade_fde_px(pred, w, k, size=(1280, 720)):
    """Per-window ADE and FDE in pixels (full-resolution image) over horizon k, per keypoint -> (N,3) each."""
    s = np.array(size, float)
    e = np.linalg.norm((pred[:, :k] - w["fut"][:, :k]) * s, axis=-1)      # (N,k,3)
    return e.mean(1), e[:, -1]
