#!/usr/bin/env python3
"""
Hand-signal recognition for the xArm pick-and-place.

Hand pose: palm facing DOWN toward the table, knuckles toward the sky (so the
overhead camera sees the back of the hand).

Gestures:
  TWO   - index + middle out, thumb/ring/pinky folded  -> grab and send home
  THREE - thumb + index + middle out, ring/pinky folded -> bring from home to workspace
  FIST  - all four fingers folded (thumb ignored)       -> STOP / pause
  OPEN  - all four fingers out, hand flat (thumb ignored) -> RESUME a paused task

TWO, THREE and OPEN only fire after being held steadily for HOLD_SEC, and after
firing the hand must drop (show no command) for REARM_SEC before the next
one - so one held gesture = one action.

FIST is handled separately: it fires fast (STOP_HOLD_SEC), works at any time
(including while the arm is moving), and fires once per fist.

Standalone test (no robot, safe to run anytime):
    python3 hand_gestures.py [camera_index]
The overlay shows the thumb value - use it to tune THUMB_OUT_RATIO.

Requires: pip install --user "mediapipe==0.10.21" "numpy<2"
(newer mediapipe versions removed the mp.solutions.hands API used here)
"""
import math
import sys
import time

import cv2

GESTURE_TWO = "TWO"
GESTURE_THREE = "THREE"
GESTURE_FIST = "FIST"
GESTURE_OPEN = "OPEN"

GESTURE_LABELS = {
    GESTURE_TWO: "2 fingers: grab + send home",
    GESTURE_THREE: "thumb+2: bring to workspace",
    GESTURE_FIST: "FIST: STOP",
    GESTURE_OPEN: "Open hand: resume",
}

HOLD_SEC = 1.0        # TWO / THREE / OPEN must be held this long to fire
REARM_SEC = 0.5       # hand must show no command this long before the next one
STOP_HOLD_SEC = 0.25  # fist must be held this long to fire (short, but not a single frame)

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
MIDDLE_MCP = 9

# (pip, tip) landmark indices for each non-thumb finger
FINGERS = {
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}

# A finger counts as extended if its tip is this much farther from the wrist
# than its middle joint. Measured on the operator's hand (palm down, 2026-09-29):
# extended 1.26-1.33, folded 0.60-0.76.
EXTENDED_RATIO = 1.1

# The thumb counts as out if its tip is this far from the index knuckle,
# measured in palm lengths (wrist -> middle knuckle). Thumb stuck out to the
# side reads ~0.8-1.0; tucked ~0.2. Measured on the operator's hand: thumb out
# 0.91, tucked 0.19-0.23. Check the value in the test window if it misreads.
THUMB_OUT_RATIO = 0.7


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def analyze(points):
    """points: 21 (x, y) landmark positions in pixels.
    Returns (set of extended fingers incl. 'thumb', thumb_ratio).
    Uses distances, not up/down position, so it works at any hand rotation."""
    wrist = points[WRIST]
    extended = set()
    for name, (pip, tip) in FINGERS.items():
        d_pip = _dist(wrist, points[pip])
        if d_pip > 0 and _dist(wrist, points[tip]) > EXTENDED_RATIO * d_pip:
            extended.add(name)

    palm = _dist(wrist, points[MIDDLE_MCP])
    thumb_ratio = _dist(points[THUMB_TIP], points[INDEX_MCP]) / palm if palm > 0 else 0.0
    if thumb_ratio > THUMB_OUT_RATIO:
        extended.add("thumb")
    return extended, thumb_ratio


def classify(extended):
    fingers = extended - {"thumb"}
    if not fingers:
        return GESTURE_FIST
    if fingers == {"index", "middle", "ring", "pinky"}:
        return GESTURE_OPEN
    if extended == {"index", "middle"}:
        return GESTURE_TWO
    if extended == {"thumb", "index", "middle"}:
        return GESTURE_THREE
    return None


class GestureTrigger:
    """Debounces per-frame gestures into single commands."""

    def __init__(self, hold_sec=HOLD_SEC, rearm_sec=REARM_SEC, stop_hold_sec=STOP_HOLD_SEC):
        self.hold_sec = hold_sec
        self.rearm_sec = rearm_sec
        self.stop_hold_sec = stop_hold_sec
        self.armed = True
        self._current = None
        self._since = None
        self._clear_since = None
        self._fist_since = None
        self._fist_fired = False

    def update(self, gesture, now=None):
        """Feed one frame's gesture (or None).
        Returns (fired_gesture_or_None, hold_progress_0_to_1)."""
        now = time.monotonic() if now is None else now

        # FIST: always active, fast, once per fist. Also blocks TWO/THREE/OPEN
        # until the hand has been relaxed for rearm_sec.
        if gesture == GESTURE_FIST:
            self._current = self._since = self._clear_since = None
            if self._fist_since is None:
                self._fist_since = now
            if not self._fist_fired and now - self._fist_since >= self.stop_hold_sec:
                self._fist_fired = True
                self.armed = False
                return GESTURE_FIST, 0.0
            return None, 0.0
        self._fist_since = None
        self._fist_fired = False

        if gesture is None:
            self._current = self._since = None
            if not self.armed:
                if self._clear_since is None:
                    self._clear_since = now
                elif now - self._clear_since >= self.rearm_sec:
                    self.armed = True
                    self._clear_since = None
            return None, 0.0

        self._clear_since = None
        if not self.armed:
            return None, 0.0

        if gesture != self._current:
            self._current = gesture
            self._since = now

        held = now - self._since
        if held >= self.hold_sec:
            self.disarm()
            return gesture, 1.0
        return None, held / self.hold_sec

    def disarm(self):
        """Block TWO/THREE/OPEN until the hand has shown no command for rearm_sec."""
        self.armed = False
        self._current = self._since = self._clear_since = None


class HandGestureDetector:
    """Wraps MediaPipe Hands. process() returns (gesture_or_None, extended_set_or_None);
    extended is None when no hand is visible. last_thumb_ratio holds the latest thumb value."""

    def __init__(self, min_detection_confidence=0.7):
        import mediapipe as mp
        self._mp_hands = mp.solutions.hands
        self._draw = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,  # fastest model; plenty for finger counting
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self.last_thumb_ratio = None

    def process(self, frame_bgr, draw=True):
        h, w = frame_bgr.shape[:2]
        result = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            self.last_thumb_ratio = None
            return None, None

        hand = result.multi_hand_landmarks[0]
        if draw:
            self._draw.draw_landmarks(frame_bgr, hand, self._mp_hands.HAND_CONNECTIONS)

        points = [(lm.x * w, lm.y * h) for lm in hand.landmark]
        extended, self.last_thumb_ratio = analyze(points)
        return classify(extended), extended

    def close(self):
        self._hands.close()


def draw_status(frame, gesture, extended, progress, armed, extra_line=None, thumb_ratio=None):
    """Overlay the current gesture, hold progress bar and armed state."""
    if extended is None:
        text, color = "No hand", (255, 255, 255)
    elif gesture is None:
        text, color = f"Hand: {', '.join(sorted(extended))} (no command)", (255, 255, 255)
    else:
        text = GESTURE_LABELS[gesture]
        color = (0, 0, 255) if gesture == GESTURE_FIST else (255, 255, 255)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not armed and gesture != GESTURE_FIST:
        cv2.putText(frame, "Relax hand to re-arm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    elif progress > 0:
        w = frame.shape[1] - 20
        cv2.rectangle(frame, (10, 45), (10 + w, 60), (255, 255, 255), 1)
        cv2.rectangle(frame, (10, 45), (10 + int(w * progress), 60), (0, 255, 0), -1)

    if extra_line:
        cv2.putText(frame, extra_line, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if thumb_ratio is not None:
        cv2.putText(frame, f"thumb={thumb_ratio:.2f} (out if > {THUMB_OUT_RATIO})",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


def main():
    """Standalone gesture test - no robot involved."""
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"Could not open camera {cam}.")
        sys.exit(1)

    detector = HandGestureDetector()
    trigger = GestureTrigger()
    last_fired = None
    print("Palm down. Hold 2 fingers, thumb+2 or an open hand for 1 s; make a fist to test STOP. 'q' quits.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gesture, extended = detector.process(frame)
            fired, progress = trigger.update(gesture)
            if fired:
                last_fired = f"FIRED: {GESTURE_LABELS[fired]}"
                print(last_fired)
            draw_status(frame, gesture, extended, progress, trigger.armed, last_fired,
                        detector.last_thumb_ratio)
            cv2.imshow("Hand gesture test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
