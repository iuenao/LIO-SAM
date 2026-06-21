# Reliability Analysis

## LiDAR Factor Relative-Pose Consistency

`compute_factor_nees.py` compares the actual relative LiDAR measurement inserted into
GTSAM with synchronized ground-truth relative motion. It reports factor-level NEES,
Gaussian NLL, chi-square coverage, reliability/error correlations, and reliability
strata. These are LiDAR factor consistency metrics, not full-state consistency.

The analysis requires keyframe diagnostics generated after Phase 9. Older CSV files do
not contain the complete mapped covariance or inserted relative measurement and must not
be used to approximate factor NEES from the final optimized trajectory.

The user must explicitly pass `--confirm-same-body-frame`. Do this only after verifying
that the factor measurement and ground-truth trajectory describe the same physical body
frame. If GT is for another sensor frame, transform it first using the calibrated rigid
extrinsic.

Example:

```bash
ros2 run lio_sam compute_factor_nees.py \
  --factors /path/to/run_keyframe_factors.csv \
  --gt /path/to/gt.txt \
  --dataset M2UD \
  --sequence plaza01 \
  --method post_split_only_norm \
  --run-id plaza01_post_split_only_norm \
  --confirm-same-body-frame
```

Outputs are written to a `factor_confidence/` directory beside the diagnostics CSV by
default.

## M2UD Ground-Truth Frame Sensitivity

The M2UD indoor documentation does not identify whether its offline-fusion trajectory
describes the robot, LiDAR, or IMU origin. The four-wheel calibration gives a
`lidar_to_robot` translation of `[0.42, 0.0, 0.34]` m but does not define the transform
direction. `compare_m2ud_gt_frames.py` therefore evaluates three explicit hypotheses:

- unchanged ground truth;
- `T_gt_factor` translation `+[0.42, 0.0, 0.34]`;
- `T_gt_factor` translation `-[0.42, 0.0, 0.34]`.

The convention is
`T_world_factor = T_world_gt * T_gt_factor`. The offset consequently rotates with each
GT pose; it is not a constant world-frame translation. Each hypothesis also exports a
transformed TUM trajectory that can be supplied to EVO.

```bash
ros2 run lio_sam compare_m2ud_gt_frames.py \
  --factors /path/to/run_keyframe_factors.csv \
  --gt /path/to/gt.txt \
  --sequence plaza01 \
  --method proposed_split_norm \
  --run-id plaza01_proposed_split_norm
```

This is a sensitivity analysis, not evidence that the hypothesis with the lowest error
is the dataset's authoritative frame. Final claims still require confirmation of the
ground-truth body convention from the dataset authors or generator metadata.
