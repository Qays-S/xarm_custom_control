# Which calibration analysis the paper uses

The paper (Section 4.2, Figure 2a) reports the analysis in `calib_cam3_orig_ids_analysis/`
(held-out 1.64 mm, leave-one-out 1.85 mm, in-sample 0.99 mm), made with

    python3 s3_calib_analyze.py calib/calib_cam3_orig_ids.json --holdout 3

`calib_cam3_orig_ids.json` is the final camera position with the dots renumbered to the original
fiducial numbering. `calib_cam3_analysis/` is the same session analysed earlier with the order the dots
were clicked in, so the random held-out points differ (held-out 0.89 mm). It is kept for the record;
quote the `orig_ids` results.
