#!/usr/bin/env python3
"""Offline tests for arm_mimic.py's safety and mapping helpers.
No camera, robot or mediapipe needed:  python3 -m pytest test_arm_mimic.py -q"""
import numpy as np

import arm_mimic as am


def test_clamp_keeps_points_inside_and_flags():
    inside = am.workspace_center()
    p, c = am.clamp_to_workspace(inside)
    assert not c and np.allclose(p, inside)

    p, c = am.clamp_to_workspace([0.0, 999.0, -50.0])
    assert c
    assert am.in_workspace(p)
    assert p[0] == am.WORKSPACE["x"][0]
    assert p[1] == am.WORKSPACE["y"][1]
    assert p[2] == am.WORKSPACE["z"][0]


def test_workspace_never_reaches_table():
    # grasp height in detect_and_move.py is 58 mm, traverse 153.5 mm
    assert am.WORKSPACE["z"][0] > 153.5


def test_limit_step_caps_distance():
    a = np.array([400.0, 0.0, 300.0])
    far = a + np.array([100.0, 0.0, 0.0])
    s = am.limit_step(a, far, max_step=12.0)
    assert np.isclose(np.linalg.norm(s - a), 12.0)
    near = a + np.array([3.0, 4.0, 0.0])
    assert np.allclose(am.limit_step(a, near, 12.0), near)


def test_mapping_signs_and_lock():
    d = {"x": 0.10, "y": -0.05, "z": -0.20}  # right 10 cm, up 5 cm, 20 cm toward camera
    out = am.human_delta_to_robot_mm(d, locked=())
    assert np.allclose(out, [200.0, -100.0, 50.0])
    out_locked = am.human_delta_to_robot_mm(d, locked={"x"})
    assert out_locked[0] == 0.0 and np.allclose(out_locked[1:], [-100.0, 50.0])


def test_one_euro_smooths_jitter():
    f = am.OneEuroFilter()
    rng = np.random.default_rng(0)
    raw = [np.array([0.3, 0.0, 0.0]) + rng.normal(0, 0.01, 3) for _ in range(120)]
    out = [f(x, i / 30.0) for i, x in enumerate(raw)]
    assert np.std(np.diff(out[20:], axis=0)) < np.std(np.diff(raw[20:], axis=0)) / 2


def test_pinch_hysteresis():
    p = am.PinchDetector(0.35, 0.55)
    assert p.update(0.5) is False      # between thresholds, starts open
    assert p.update(0.3) is True       # closes
    assert p.update(0.5) is True       # stays closed in the dead band
    assert p.update(None) is True      # hand lost: keep state
    assert p.update(0.6) is False      # opens


def test_engage_then_move_is_continuous():
    """Simulate a session: engaging must not jump, and a sudden 1 m hand
    glitch must still be clamped and step-limited."""
    origin_r = am.workspace_center()
    origin_h = {"x": 0.2, "y": 0.1, "z": -0.1}
    last = origin_r.copy()
    for human in ({"x": 0.2, "y": 0.1, "z": -0.1}, {"x": 1.2, "y": -0.9, "z": -0.1}):
        delta = {k: human[k] - origin_h[k] for k in "xyz"}
        desired, _ = am.clamp_to_workspace(origin_r + am.human_delta_to_robot_mm(delta, locked={"x"}))
        nxt = am.limit_step(last, desired)
        assert np.linalg.norm(nxt - last) <= am.MAX_STEP_MM + 1e-9
        assert am.in_workspace(nxt)
        last = nxt


class _P:
    def __init__(self, x, y):
        self.x, self.y, self.z = x, y, 0.0


class _Hand:
    def __init__(self, pts):
        self.landmark = [_P(*p) for p in pts]


def _hand(extended):
    """Fake hand, wrist at (0.5, 0.9), fingers pointing up; `extended` = which
    of index..pinky are straight (tip far above PIP) vs curled (tip below PIP)."""
    pts = [(0.5, 0.9)] + [(0.5, 0.85)] * 20
    for k, (tip, pip) in enumerate(am.FINGER_TIP_PIP):
        x = 0.44 + 0.04 * k
        pts[pip] = (x, 0.70)
        pts[tip] = (x, 0.55) if extended[k] else (x, 0.80)
    return _Hand(pts)


def test_count_extended_fingers():
    assert am.count_extended_fingers(_hand([1, 1, 1, 1])) == 4
    assert am.count_extended_fingers(_hand([0, 0, 0, 0])) == 0
    assert am.count_extended_fingers(_hand([1, 1, 0, 0])) == 2


def test_fist_detector_confirms_and_ignores_flicker():
    g = am.FistDetector(fist_max=1, open_min=3, confirm=4)
    for _ in range(3):
        assert g.update(0) is False       # not confirmed yet
    assert g.update(0) is True            # 4th frame of fist -> closed
    assert g.update(4) is True            # one open frame is not enough
    assert g.update(0) is True            # flicker back resets the count
    assert g.update(2) is True            # ambiguous count changes nothing
    assert g.update(None) is True         # hand lost keeps state
    for _ in range(4):
        g.update(4)
    assert g.closed is False              # held open -> opens


def test_workspace_check_points_cover_box():
    pts = am.workspace_check_points()
    assert len(pts) == 15
    assert all(am.in_workspace(p) for p in pts)


def _knuckles(angle_deg, aspect=16 / 9):
    """Fake hand whose index->pinky knuckle line sits at angle_deg on screen."""
    import math
    pts = [(0.5, 0.5)] * 21
    r = 0.05
    pts = list(pts)
    pts[am.HAND_INDEX_MCP] = (0.5, 0.5)
    pts[am.HAND_PINKY_MCP] = (0.5 + r * math.cos(math.radians(angle_deg)) / aspect,
                              0.5 + r * math.sin(math.radians(angle_deg)))
    return _Hand(pts)


def test_hand_twist_angle_and_unwrap():
    for a in (0, 30, -60, 170):
        assert abs(am.hand_twist_deg(_knuckles(a)) - a) < 1e-6
    u = am.AngleUnwrapper()
    seq = [u(a) for a in (170, 179, -175, -165)]  # crossing +-180 is continuous
    assert np.allclose(seq, [170, 179, 185, 195])


def test_twist_to_yaw_deadband_and_limit():
    assert am.twist_to_yaw(2.0, 0.0, deadband=4) == 0.0          # jitter ignored
    assert am.twist_to_yaw(30.0, 0.0, deadband=4) == 26.0        # beyond deadband
    assert am.twist_to_yaw(-30.0, 0.0, sign=-1, deadband=4) == 26.0
    assert am.twist_to_yaw(500.0, 0.0, limit=90) == 90.0         # hard limit
    # re-engaging repeatedly can't wind the gripper past the limit
    assert am.twist_to_yaw(60.0, 80.0, limit=90) == 90.0
    assert am.limit_angle_step(0.0, 50.0, 6.0) == 6.0
    assert am.limit_angle_step(0.0, -3.0, 6.0) == -3.0
