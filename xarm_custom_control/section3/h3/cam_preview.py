"""Live camera view with body tracking, for placing the camera. Nothing touches the arm.
Run from section3/h3:  python3 cam_preview.py [camera index, default 0]   (q to quit)"""
import sys
sys.path.insert(0, '..')
import cv2
import mediapipe as mp
from s3_common import open_camera

cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = open_camera(cam)
pose = mp.solutions.pose.Pose(model_complexity=1)
draw, P = mp.solutions.drawing_utils, mp.solutions.pose.PoseLandmark
want = [("head", P.NOSE), ("L shoulder", P.LEFT_SHOULDER), ("R shoulder", P.RIGHT_SHOULDER),
        ("L elbow", P.LEFT_ELBOW), ("R elbow", P.RIGHT_ELBOW),
        ("L wrist", P.LEFT_WRIST), ("R wrist", P.RIGHT_WRIST)]
bad = 0
while True:
    ok, f = cap.read()
    if not ok:                       # camera dropped out: say so instead of closing silently
        bad += 1
        if bad > 30:
            print("Camera stopped sending frames (unplugged or USB dropped). Check the cable, then rerun.")
            break
        continue
    bad = 0
    r = pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if r.pose_landmarks:
        draw.draw_landmarks(f, r.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    for i, (name, lm) in enumerate(want):
        seen = r.pose_landmarks and r.pose_landmarks.landmark[lm].visibility > 0.5
        cv2.putText(f, name, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 200, 0) if seen else (0, 0, 255), 2)
    h, w = f.shape[:2]
    cv2.line(f, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)
    cv2.line(f, (0, h // 2), (w, h // 2), (200, 200, 200), 1)
    cv2.imshow("camera preview (q to quit)", f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
