"""
Blueprint Section VIII, step 48: xArm 7 + gripper kinematics and simple collision geometry, for offline replay.

Kinematics: the joint origins (x, y, z, roll, pitch, yaw) of UFACTORY's published xarm7 default kinematics
(xarm_ros2, xarm_description/config/kinematics/default/xarm7_default_kinematics.yaml, BSD licence; the
numbers are restated here, no file is copied). Every joint turns about its local z axis, as in the URDF.
Joint limits are the URDF's (xarm7_robot_macro.xacro). Tool: the xArm gripper, TCP 172 mm along the flange z.
Units: metres and radians inside, millimetres at the interface (robot base frame, as the arm reports it).

Collision geometry: each link is a capsule (a segment with a radius) between consecutive joint origins, plus
the gripper from the flange to the TCP. The radii are conservative round numbers from the arm's outer
dimensions, not a certified model. For the planar separation proxy the capsules are projected on the table.

IK: damped least squares on TCP position with the tool pointing straight down, using the 7th degree of
freedom to stay close to a comfortable rest posture. Only used to turn a Cartesian path into joint angles.
"""
import math

import numpy as np

# (x, y, z, roll, pitch, yaw) of joint i relative to the previous joint frame, metres / radians
JOINTS = [(0, 0, 0.267, 0, 0, 0),
          (0, 0, 0, -1.5708, 0, 0),
          (0, -0.293, 0, 1.5708, 0, 0),
          (0.0525, 0, 0, 1.5708, 0, 0),
          (0.0775, -0.3425, 0, 1.5708, 0, 0),
          (0, 0, 0, 1.5708, 0, 0),
          (0.076, 0.097, 0, -1.5708, 0, 0)]
LOWER = np.array([-math.pi * 0.99, -2.18, -math.pi * 0.99, -0.11, -math.pi * 0.99, -1.75, -math.pi * 0.99])
UPPER = np.array([math.pi * 0.99, 2.18, math.pi * 0.99, math.pi * 0.99, math.pi * 0.99, math.pi * 0.99, math.pi * 0.99])
TCP_OFFSET = 0.172
# capsule radius (m) of the segment that ends at each frame: base->j2 ... j7->flange, flange->TCP
RADII = [0.07, 0.07, 0.06, 0.06, 0.055, 0.05, 0.045, 0.045]
REST = np.array([0.0, -0.5, 0.0, 1.2, 0.0, 1.7, 0.0])          # a comfortable elbow-up posture


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


_FIXED = []
for x, y, z, r, p, yw in JOINTS:
    T = np.eye(4); T[:3, :3] = _rpy(r, p, yw); T[:3, 3] = (x, y, z)
    _FIXED.append(T)


def fk(q):
    """Frames of joints 1..7 (after their rotation) and the TCP, as a list of 4x4 matrices (metres)."""
    T = np.eye(4)
    out = []
    for i, qi in enumerate(q):
        c, s = math.cos(qi), math.sin(qi)
        Rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        T = T @ _FIXED[i] @ Rz
        out.append(T)
    tcp = T.copy(); tcp[:3, 3] = T[:3, 3] + T[:3, 2] * TCP_OFFSET
    out.append(tcp)
    return out


def flange_mm(q):
    return fk(q)[6][:3, 3] * 1000.0


def tcp_mm(q):
    return fk(q)[7][:3, 3] * 1000.0


def capsules(q):
    """Segments (start, end) in metres and their radii: base -> joint frames -> flange -> TCP."""
    F = fk(q)
    pts = [np.zeros(3)] + [f[:3, 3] for f in F]
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    # drop zero-length segments (coincident joint origins), keep radii aligned
    keep = [(a, b, RADII[min(i, len(RADII) - 1)]) for i, (a, b) in enumerate(segs) if np.linalg.norm(b - a) > 1e-6]
    return keep


def ik_down(target_mm, q0=None, iters=200, tol_mm=0.5):
    """Joint angles that put the TCP at target_mm with the tool pointing straight down (-z)."""
    q = (REST if q0 is None else q0).astype(float).copy()
    target = np.asarray(target_mm, float) / 1000.0
    for _ in range(iters):
        F = fk(q)
        p, z = F[7][:3, 3], F[7][:3, 2]
        err = np.r_[target - p, 0.5 * np.cross(z, np.array([0, 0, -1.0]))]
        if np.linalg.norm(err[:3]) * 1000 < tol_mm and np.linalg.norm(err[3:]) < 1e-3:
            return q, True
        J = np.zeros((6, 7)); eps = 1e-5
        for i in range(7):
            dq = q.copy(); dq[i] += eps
            G = fk(dq)
            J[:3, i] = (G[7][:3, 3] - p) / eps
            J[3:, i] = 0.5 * np.cross(G[7][:3, 2], np.array([0, 0, -1.0])) / eps * -1 + 0.5 * np.cross(z, np.array([0, 0, -1.0])) / eps
        lam = 0.05
        Jp = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(6))
        null = (np.eye(7) - Jp @ J) @ (0.1 * (REST - q))
        q = np.clip(q + Jp @ err + null, LOWER, UPPER)
    F = fk(q)
    return q, np.linalg.norm(F[7][:3, 3] - target) * 1000 < 2.0


def point_capsule_xy_mm(points_xy_mm, q):
    """Planar proxy: for each table-plane point (mm), the smallest distance to the footprint of any capsule
    (segment projected on the table, minus its radius). Returns (N,) mm; negative means inside a footprint."""
    P = np.asarray(points_xy_mm, float) / 1000.0
    best = np.full(len(P), np.inf)
    for a, b, r in capsules(q):
        a2, b2 = a[:2], b[:2]
        d = b2 - a2
        L2 = float(d @ d)
        if L2 < 1e-12:
            dist = np.linalg.norm(P - a2, axis=1)
        else:
            t = np.clip(((P - a2) @ d) / L2, 0, 1)
            dist = np.linalg.norm(P - (a2 + t[:, None] * d), axis=1)
        best = np.minimum(best, dist - r)
    return best * 1000.0
