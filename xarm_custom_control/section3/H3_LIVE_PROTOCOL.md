# H3 live trials: session protocol

Everything is recorded with the xArm7 actually running pick-and-place (dot 6 to dot 8) while the
participant reaches. Nothing is recorded with the arm parked.

| Phase | Who | What | Time |
|---|---|---|---|
| 0. Setup and checks | Qays | calibration, self-tests, dry runs, slow runs | about 1.5 h |
| A. Training data | Qays (P01) | E1–E4 with the reactive planner running the arm | about 30 min of reaching |
| B. Train and freeze | code | retrain everything on Phase A only, then freeze | a few hours of computer time |
| C. Planner comparison | P01 and P02 | 10 planners on E4 | about 60 min of reaching each |

P02 never appears in Phases A or B, so their Phase C results test the method on a new person.

## Safety rules for every session with the arm moving

- Someone other than the participant holds the E-stop for the whole session and watches the arm.
- The xArm ROS driver is NOT running (the SDK script controls the arm). Close UFactory Studio's
  control page too.
- Built into `s9_live.py` (all planners): the arm stops if no arm keypoint is seen for 0.25 s, if the
  camera stalls for 0.3 s, if the real arm comes within 20 mm of a keypoint, on any controller error,
  or when `h` is pressed; it waits 3 clear steps before moving again. Speed is capped at 150 mm/s in
  software and in the controller (reduced mode), and collision detection is on.
- First time with a new setup: a run with nobody at the table, then a run at one third of the speed
  (`--speed-scale 0.33`, 50 mm/s) with the participant, before any full-speed run.
- The participant can stop at any time by pulling their arm back, and by saying so.

## 0. Setup (once, and again if the camera or table moves)

Work in a separate folder so the original (parked-arm) results are never overwritten:

```bash
cd ~/ros2_ws/src/xarm_custom_control/xarm_custom_control/section3
mkdir -p h3/calib h3/logs && cd h3
```

All commands below are run in `section3/h3`, as `python3 ../script.py`.

1. Calibrate the camera to the table. Touch the dots **in order 1 to 12**, so that each id matches
   the dot number (6 = pick, 8 = place):
   ```bash
   python3 ../s3_collect.py calib --out calib/calib_live.json
   python3 ../s3_calib_analyze.py calib/calib_live.json --holdout 4
   ```
2. Check the code on this laptop (nothing moves). From `section3` (it uses the original data):
   ```bash
   cd .. && python3 s9_selftest.py --manicast-dir ~/manicast && python3 s9_mockarm_test.py && cd h3
   ```
   Both must print PASS. The self-test also prints the compute time per step; it must stay well under 62.5 ms.
3. Dry run with the camera, robot not connected (`--robot dry` is the default):
   ```bash
   python3 ../s4_record_targets.py --mode E2 --out logs/dry_01 --reaches 10 --live reactive \
       --calib-session calib/calib_live.json
   ```
   The banner shows `ROBOT reactive 1.00` while moving and the stop reason in brackets.
4. Real arm, nobody reaching. The arm only moves while it can see a person's arm, so sit back with your
   forearm visible to the camera and do not reach (use `x` to skip reaches):
   ```bash
   python3 ../s4_record_targets.py --mode E2 --out logs/empty_01 --reaches 5 --live reactive \
       --robot xarm --speed-scale 0.33 --calib-session calib/calib_live.json
   ```
   It asks before moving to the start pose (150 mm above dot 6). Watch one full loop.
5. Real arm, participant, slow: the same with `--out logs/slow_01 --reaches 20`.

## A. Training data (P01, reactive planner, full speed)

Drift check at the start and end of each sitting:
`python3 ../s3_check_drift.py calib/calib_live_reference_frame.png --dots calib/calib_live.json`

Common options: `--live reactive --robot xarm --calib-session calib/calib_live.json --participant P01`

| Session | Extra options |
|---|---|
| `logs/A_P01_E1_s01` | `--mode E1` |
| `logs/A_P01_E2_s01` | `--mode E2` |
| `logs/A_P01_E3_s01` | `--mode E3` |
| `logs/A_P01_E4_s01` | `--mode E4` |
| `logs/A_P01_E1_s02` ... `E4_s02` | same as round 1 |

40 reaches each (the default). Round 1 (`_s01`) is training; round 2 is split in halves into
validation and calibration.

E5 (covering the hand with paper, dimming the lights) was dropped on 2026-11-12: it was hard to run
and not a realistic workspace condition. Degraded views are studied instead where they happen on their
own, e.g. when the robot arm passes between the camera and the hand (`s6_reliability.py` reports
"clear view" vs "partly hidden" moments). A stray attempt is kept as `logs/X_A_P01_E5_s01_nopaper`;
names starting with `X_` are never used.

Before every session: charger plugged in, `powerprofilesctl set performance`, RViz and the ROS
driver closed (the camera should run at about 14 frames per second). If the drift check says MOVED
by a small, uniform shift, `python3 ../s3_reregister.py calib/calib_live.json` fixes the calibration
without touching the dots again (it refuses if the dots did not all move together).

## B. Train and freeze (in `section3/h3`)

```bash
python3 ../s5_prepare.py logs/A_P01_E[1-4]_s0[12] --causal --split live
python3 ../s5_baselines.py
python3 ../s5_gru.py
python3 ../s5_mgru.py
python3 ../s5_stsgcn.py --manicast-dir ~/manicast
python3 ../s6_reliability.py
python3 ../s7_replay.py --manicast-dir ~/manicast --calib calib/calib_live.json
python3 ../s9_selftest.py --manicast-dir ~/manicast --session logs/A_P01_E4_s02 --calib calib/calib_live.json
python3 ../s9_freeze.py
```

`--causal` makes training use exactly what the robot sees live (a step only uses frames already
seen). The self-test must pass before freezing. After `s9_freeze.py`, nothing in `logs/prepared`,
`logs/forecast_results`, `logs/reliability` or `logs/replay` may change.

## C. Planner comparison (P01 and P02) — to be updated now that E5 is dropped

Each session: one planner, one condition. Common options:
`--robot xarm --calib-session calib/calib_live.json --participant P0x --live <planner>`

- Normal view: `--mode E4` (40 reaches, about 3 min), out `logs/C_P0x_<planner>_E4`
- Degraded view: `--mode E5 --reaches 20 --dim-at 11 --bright-at 99` (about 3 min), out `logs/C_P0x_<planner>_E5`

Planner order (counterbalanced, so tiredness and learning do not favour any planner):

| Block | Order |
|---|---|
| P01, E4 | reactive, cv, full2, gru, full_nofb, mgru, full, sts, gate, uncert |
| P02, E4 | cv, gru, reactive, mgru, full2, sts, full_nofb, uncert, full, gate |
| P01, E5 | gru, mgru, cv, sts, reactive, uncert, full2, gate, full_nofb, full |
| P02, E5 | mgru, sts, gru, uncert, cv, gate, reactive, full, full2, full_nofb |

Suggested days: E4 block one day, E5 block another; a short break every 3 sessions. The participant
is not told which planner is running. Write down anything unusual (a stop, a near-touch, a lighting
change) with the session name.

P02 first does 5 slow practice reaches (`--speed-scale 0.33 --live reactive --out logs/practice_P02`)
to get used to the arm moving; practice sessions are not analysed.

## Results

```bash
python3 ../s9_analyze_live.py logs/C_P0*_* --out logs/live_results
```

Near-misses use the real arm pose (logged joint angles) and where the person actually was at each
step, not the delayed data the planner had. Forecast and reliability accuracy on Phase C can be
recomputed offline from the recorded frames with the frozen models
(`python3 ../s5_prepare.py logs/C_* --causal --split live --out logs/prepared_c`).
