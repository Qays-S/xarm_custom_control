# Section III tools: experimental geometry and perception

These scripts produce every Section III requirement of the blueprint. **None of them moves
the arm.** Where the arm is involved, you jog it yourself (UFactory Studio, teach mode or your
own scripts) and the tool only *reads* its position. The perception recordings are done with
the robot parked.

| Blueprint step / required output | Tool | What you get |
|---|---|---|
| 12. Camera rigidly mounted, not moved during a session | `s3_check_drift.py` | OK / MOVED check against the calibration's reference frame (run at start and end of every session) |
| 13. 12–20 planar fiducials, several held out | `s3_collect.py calib` | point list + reference frame |
| 14. Robust H_CB, held-out metric error → **calibration-error distribution** | `s3_calib_analyze.py` | held-out, leave-one-out and in-sample errors, ECDF plot, error map, `camera_calibration_s3.json` |
| 16. Markers at known heights → **height/parallax sensitivity** | `s3_collect.py parallax` + `s3_parallax_analyze.py` | error vs height table and plots, height at which parallax exceeds calibration error |
| 15. Elbow/wrist/hand (+shoulder), raw frames, pixels, confidence, missingness | `s3_perception_logger.py` | `frames/`, `frames.csv`, `events.csv`, `meta.json` |
| 17. Frame rate, dropped frames, detection failures, jitter → **visibility/missingness statistics** | `s3_perception_analyze.py` | `summary.json`, per-frame visibility CSV, plots |
| 17. End-to-end perception latency → **latency distribution** | `s3_latency_test.py` (+ analyze `--latency`) | camera latency distribution + detection time = end-to-end estimate |
| **Manually annotated keypoint subset** | `s3_annotate.py` | your clicks on ~100 frames, pixel error, PCK, missed-when-visible rates |

Install (same pins as `arm_mimic.py`):
```
pip install --user "mediapipe==0.10.21" "numpy<2" opencv-python matplotlib xarm-python-sdk
```
Run everything from this folder. Keep the camera resolution the same for every step
(pass the same `--width/--height`).

## Order of work (your steps on the robot)

**A. Latency test: do this FIRST, before calibrating.** After calibration the camera must not
move. Put the laptop screen in front of the camera, then:
```
python3 s3_latency_test.py --out logs/latency01 --flips 40
```
Aim the camera so the window fills much of the image, press SPACE, and keep everything still.
The result includes the screen's refresh delay, so it slightly overestimates the camera's latency.

**B. Mount the camera rigidly** in its final reach-view position. Do not touch it again this session.

**C. Calibration points (12–20).**
```
python3 s3_collect.py calib --out calib/calib_session.json        # add --arm manual if the SDK can't connect
```
- Keep the gripper pointing straight down (roll 180, pitch 0) so the tip is directly below the TCP.
- For each point: jog the tip until it **touches the table**, click the tip in the image (use the
  zoom inset), then press `r`. The script reads the arm's X/Y/Z. It warns if Z drifts more than
  3 mm from the other points, which means the tip isn't on the table.
- Cover the **whole area people reach into**: corners, edges and interior, spread evenly. Now that
  the arm can move, avoid the dead space of the 9-point session.
- Collect 16 if you can. 12 is the minimum.

Then:
```
python3 s3_calib_analyze.py calib/calib_session.json --holdout 4
```
The 4 held-out points are chosen at random with `--seed 0` before fitting and written to
`summary.json`. Report the **held-out** numbers as the headline, and leave-one-out as secondary.
To use the new calibration in your other scripts, copy
`calib_session_analysis/camera_calibration_s3.json` over `camera_calibration.json`. It has the same keys.

**D. Drift check at the start and end of every recording session:**
```
python3 s3_check_drift.py calib/calib_session_reference_frame.png --calib calib/calib_session_analysis/camera_calibration_s3.json
```
If it says MOVED, recalibrate before using that session's data.

**E. Parallax (known heights).**
```
python3 s3_collect.py parallax --out calib/parallax_session.json
```
- Pick at least 3 table spots spread over the workspace (near, middle, far from the camera).
- At each spot, first touch the table and record (`r`). Then raise the tip **straight up**,
  changing Z only, to e.g. 50, 100, 150, 200, 250 mm, recording at each height. Cover the
  heights where elbows, wrists and hands actually are during reaches.
- Press `n` before moving to the next spot.

Then:
```
python3 s3_parallax_analyze.py calib/parallax_session.json --calib calib/calib_session_analysis/camera_calibration_s3.json
```

**F. Perception recordings (robot parked).**
```
python3 s3_perception_logger.py --out logs/session01 --duration 300
```
- Do normal reaches with the arm you'll study (the right arm unless you choose otherwise).
- Include at least 3 **static holds** of 5–10 s with the arm still: press `s` to start and `s`
  to stop. These measure keypoint jitter.
- Do a few deliberate **occlusions** (`o` to start and stop), and press `m` at any cue you want
  marked.
- Pose smoothing is OFF on purpose, so the jitter and latency numbers are honest.

**G. Manual annotation (~100 frames):**
```
python3 s3_annotate.py annotate logs/session01 --n 100 --side R
python3 s3_annotate.py evaluate logs/session01
```
Frames are sampled evenly over the recording with a fixed seed, and the sample is saved before
you start. Click shoulder, elbow, wrist, then hand (centre of the knuckles). Press `n` if a
keypoint isn't visible.

**H. Statistics:**
```
python3 s3_perception_analyze.py logs/session01 [logs/session02 ...] --side R --latency logs/latency01
```

## What goes in the paper (Section III outputs)
- **Calibration-error distribution:** held-out mean/median/max, LOO, the ECDF figure, and the
  number and spread of fiducials.
- **Height/parallax sensitivity:** error-vs-height table and figure, plus the height above which
  planar projection error exceeds the calibration error. This justifies treating elevated arm
  points as image-space only.
- **Manually annotated subset:** frame count, per-keypoint pixel error, PCK@10 px, and
  missed-when-visible rate.
- **Visibility/missingness:** missing fraction per keypoint (overall and during occlusion),
  detection-failure fraction, dropped frames, fps.
- **Latency:** camera latency distribution, detection time, and their sum as an end-to-end
  estimate (median, p95).

All tools were tested on simulated data (a synthetic camera with known geometry and a
synthetic video of a person). Real values come only from your recordings.

## Section IV recordings (E1–E4): moving between targets

The robot stays **parked** for all of these; the tools only read its state. The targets are
taped calibration dots (default #1, #4, #9, #12), so each target is known in pixels and in
robot millimetres. Run from `section3`, with the xArm driver running and the workspace sourced
(`source ~/ros2_ws/install/setup.bash`) so the robot's joint states are logged.

**Before every session:** `python3 s3_check_drift.py calib/reference_cam3b.png --camera 0` must say OK.

| Set | Command | What you do |
|---|---|---|
| E1 | `python3 s4_record_targets.py --mode E1 --out logs/E1_s01` | Touch the dot shown ("GO TO 9") at a comfortable speed, hold until the next cue |
| E2 | `python3 s4_record_targets.py --mode E2 --out logs/E2_s01` | Keep moving: the next dot appears the moment you arrive. Follow the speed/path hint at the bottom |
| E3 | `python3 s4_record_targets.py --mode E3 --out logs/E3_s01` | Two dots light up ("4 OR 12 ?"). Start moving toward both; the real one is revealed once you are moving |
| E4 | `python3 s4_record_targets.py --mode E4 --out logs/E4_s01` | Go to the dot shown; on about half the reaches the goal changes just after you start moving. Follow the new one |

- Each session stops after 40 valid reaches (`--reaches`), about 2–4 minutes.
- Start: the screen says "TOUCH 1 TO START". Touch it with your index fingertip.
- Arrival is detected from the fingertip. If a touch isn't picked up, press SPACE.
- Press `x` if a reach went wrong (it stays in the data, flagged invalid), `o` to mark an
  occlusion, `p` to pause, `q` to stop. A new session needs a new `--out` folder (`_s02`, ...).
- Dry run without the camera: add `--sim`.

**Summary (counts, durations, missing keypoints, timing, E4 switch onsets):**
```
python3 s4_dataset_summary.py logs/E1_s01 logs/E2_s01 logs/E3_s01 logs/E4_s01
```

## Section V: forecasting (runs on the laptop, no camera or arm needed)

**1. Prepare the tracks and freeze the split** (once, after recording):
```
python3 s5_prepare.py logs/E1_s01 logs/E2_s01 logs/E3_s01 logs/E4_s01 logs/E1_s02 logs/E2_s02 logs/E3_s02 logs/E4_s02
```
Resamples every session to 16 Hz (0.0625 s steps), normalises pixels by the image size, and
writes `logs/prepared/` with one `.npz` per session, `split_manifest.json` and
`leakage_audit.csv`. Split: round 1 = train; each round 2 session is cut by reach into
thirds = val, cal, test. A window is used only if it lies inside one block, so the audit
must say PASS. Commit `logs/prepared/` so the split stays frozen.

**2. Baselines on the validation set** (current pose and constant velocity):
```
python3 s5_baselines.py
```
ADE/FDE in pixels at 0.25, 0.5, 1.0 and 1.5 s, saved in `logs/forecast_results/`. The test
set is not used.

**3. Deterministic GRU** (needs `pip install torch`; about 5 min on a laptop CPU):
```
python3 s5_gru.py            # small grid search, chosen on val; --quick for one config
```
Saves the learning curves, the chosen model (`gru_best.pt`), val ADE/FDE (`gru_val.csv`) and
the one-window inference time in `logs/forecast_results/gru/`. The test set is not used.
