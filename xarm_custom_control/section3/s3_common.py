"""Shared helpers for the Section III (geometry and perception) tools.

Nothing in this folder ever commands the arm to move. When a tool reads the arm,
it only calls get_position(); you jog the arm yourself (teach mode, UFactory
Studio, or your own scripts).
"""
import json
import os
import time

import numpy as np

ARM_IP_DEFAULT = "192.168.1.210"


# ----------------------------------------------------------------- arm (read-only)
class ArmReader:
    """Read the TCP pose of the xArm without ever enabling or moving it.

    source="sdk":    xarm-python-sdk get_position() (stop other programs that hold
                     the SDK connection first if reads fail).
    source="manual": you type X Y Z (mm) yourself, e.g. copied from
                     `ros2 topic echo /xarm/robot_states --once` (pose field).
    """

    def __init__(self, source="sdk", ip=ARM_IP_DEFAULT):
        self.source = source
        self.arm = None
        if source == "sdk":
            try:
                from xarm.wrapper import XArmAPI
            except ImportError:
                raise SystemExit("xarm-python-sdk not installed (pip install --user xarm-python-sdk), "
                                 "or run with --arm manual")
            self.arm = XArmAPI(ip, is_radian=False)  # connect only; no enable/mode/state calls

    def read_xyz(self):
        if self.source == "manual":
            while True:
                raw = input("  Enter arm X Y Z in mm (space separated), or 's' to skip: ").strip()
                if raw.lower() == "s":
                    return None
                try:
                    x, y, z = (float(v) for v in raw.replace(",", " ").split())
                    return [x, y, z]
                except ValueError:
                    print("  Need three numbers, e.g. 512.3 -140.0 102.5")
        reads = []
        for _ in range(5):  # average a few reads to reject a single bad sample
            code, pos = self.arm.get_position()
            if code == 0:
                reads.append(pos[:3])
            time.sleep(0.02)
        if not reads:
            print("  get_position failed; is the arm connected and not held by another program?")
            return None
        reads = np.array(reads, float)
        if np.ptp(reads, axis=0).max() > 0.5:
            print("  Warning: the arm was moving while reading (spread > 0.5 mm). Hold still and retry.")
            return None
        return reads.mean(0).round(3).tolist()

    def close(self):
        if self.arm is not None:
            self.arm.disconnect()


# ----------------------------------------------------------------- homography
def _norm(pts):
    m = pts.mean(0)
    s = np.sqrt(2) / max(np.mean(np.linalg.norm(pts - m, axis=1)), 1e-12)
    return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1.0]])


def fit_homography(pix, world):
    """Normalized DLT (same result as cv2.findHomography(method=0))."""
    pix, world = np.asarray(pix, float), np.asarray(world, float)
    if len(pix) < 4:
        raise ValueError("need at least 4 points")
    tp, tw = _norm(pix), _norm(world)
    ph = (tp @ np.c_[pix, np.ones(len(pix))].T).T
    wh = (tw @ np.c_[world, np.ones(len(world))].T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(ph, wh):
        rows += [[-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]]
    h = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    h = np.linalg.inv(tw) @ h @ tp
    return h / h[2, 2]


def apply_h(h, pix):
    pix = np.atleast_2d(np.asarray(pix, float))
    q = (h @ np.c_[pix, np.ones(len(pix))].T).T
    return q[:, :2] / q[:, 2:3]


def robust_fit(pix, world, thresh_mm=10.0, iters=2000, seed=0):
    """RANSAC over 4-point samples, then a DLT refit on the inliers.
    Returns (H, inlier_mask)."""
    pix, world = np.asarray(pix, float), np.asarray(world, float)
    n = len(pix)
    rng = np.random.default_rng(seed)
    best = np.ones(n, bool)
    best_count, best_err = -1, np.inf
    if n > 5:
        for _ in range(iters):
            idx = rng.choice(n, 4, replace=False)
            try:
                h = fit_homography(pix[idx], world[idx])
            except (ValueError, np.linalg.LinAlgError):
                continue
            err = np.linalg.norm(apply_h(h, pix) - world, axis=1)
            inl = err < thresh_mm
            if inl.sum() > best_count or (inl.sum() == best_count and err[inl].mean() < best_err):
                best, best_count, best_err = inl, inl.sum(), err[inl].mean()
    return fit_homography(pix[best], world[best]), best


def errors_mm(h, pix, world):
    return np.linalg.norm(apply_h(h, pix) - np.asarray(world, float), axis=1)


def stats(e):
    e = np.asarray(e, float)
    if len(e) == 0:
        return {}
    return {"n": int(len(e)), "mean": float(e.mean()), "median": float(np.median(e)),
            "p90": float(np.percentile(e, 90)), "max": float(e.max()), "min": float(e.min()),
            "rmse": float(np.sqrt((e ** 2).mean()))}


# ----------------------------------------------------------------- io
def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")
