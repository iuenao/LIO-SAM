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

### ECEF Ground Truth With Local Attitude

Some RTK/INS trajectories, including M2DGR Walk01, store positions in ECEF while their
quaternions describe body attitude in a local ENU navigation frame. Convert only the
positions before computing relative factors:

```bash
ros2 run lio_sam compute_factor_nees.py \
  --factors /path/to/run_keyframe_factors.csv \
  --gt /path/to/ecef_gt.txt \
  --gt-position-coordinates ecef \
  --gt-orientation-coordinates local \
  --gt-to-factor-translation -0.15905 -0.00067 0.16824 \
  --gt-to-factor-rpy-deg 0 0 0 \
  --frame-hypothesis m2dgr_xsens_enu_to_lidar \
  --confirm-same-body-frame
```

The first GT position defines the local ENU origin unless `--ecef-origin X Y Z` is
provided. The selected origin and derived latitude/longitude are recorded in both
summary formats. Use `--gt-orientation-coordinates ecef` only when the quaternion is
known to express attitude in ECEF; the tool then rotates it into ENU as well.

## Per-Correspondence Covariance Feasibility Logs

Set `enablePerCorrespondenceCSV: true` only for short offline covariance experiments.
The default is `false`. For each accepted keyframe, map optimization writes:

- `<prefix>_correspondences.csv`: source/map coordinates, raw and LIO-SAM-scaled
  residuals, exact factor weight, reliability terms, and the six-entry LM Jacobian row;
- `<prefix>_lm_to_factor_jacobian.csv`: the exact 6x6 finite-difference Jacobian used to
  map LM covariance into the GTSAM BetweenFactor tangent frame.

`scaled_residual = lio_sam_base_scale * raw_residual`. Use this scaled residual with
`j0..j5` when constructing score vectors for a sandwich covariance. The ablation runner
enables both files with `--enable-per-correspondence-csv`.

The expected data volume is large. Replay at `0.5x` for the controlled Walk01 and
Plaza01 probes so CSV I/O does not cause dropped sensor messages:

```bash
ros2 run lio_sam run_ablation_bags.py \
  --dataset M2DGR \
  --sequence walk01_correspondence_120s \
  --bag /mnt/sdcard/rosbags/M2DGR/walk_01 \
  --config /home/unitree/ros2_workspaces/lio_ws/src/LIO-SAM/config/M2DGR.yaml \
  --launch run_M2DGR.launch.py \
  --methods raw_hessian proposed_split \
  --workspace /home/unitree/ros2_workspaces/lio_ws \
  --play-args="--clock --rate 0.5" \
  --bag-duration 240 \
  --settle-time 20 \
  --enable-per-correspondence-csv
```

Use the same command for Plaza01 with dataset `M2UD`, sequence
`plaza01_correspondence_120s`, bag `/mnt/sdcard/rosbags/M2UD/plaza_01`, configuration
`config/M2UD.yaml`, and launch file `run_M2UD.launch.py`.

Before testing a new covariance formulation, reconstruct the current weighted Hessian
and mapped covariance from the logs and compare them numerically with the corresponding
`*_keyframe_factors.csv` row. This establishes that column ordering, weighting, scale,
eigenvalue clamping, and tangent-frame mapping are reproduced exactly.

## Offline Cluster-Robust Covariance Evaluation

`evaluate_cluster_robust_covariance.py` streams a correspondence CSV one keyframe at a
time, reconstructs the production covariance, and evaluates independent-point and
feature-separated spatial-voxel sandwich covariances. It reports full, rotation,
translation, and per-axis NEES together with NLL, coverage, cluster counts, covariance
eigenvalues, and production-reconstruction error.

Example for Plaza01 raw Hessian:

```bash
ros2 run lio_sam evaluate_cluster_robust_covariance.py \
  --correspondences /path/to/plaza_raw_hessian_correspondences.csv \
  --jacobians /path/to/plaza_raw_hessian_lm_to_factor_jacobian.csv \
  --factors /path/to/plaza_raw_hessian_keyframe_factors.csv \
  --gt /home/unitree/ros2_workspaces/lio_ws/trajectory_save/plaza_01/gt.txt \
  --output-dir /path/to/raw_hessian/cluster_covariance \
  --dataset M2UD \
  --sequence plaza01_correspondence_120s \
  --method raw_hessian \
  --frame-hypothesis identity \
  --confirm-same-body-frame
```

For Walk01, additionally use `--gt-position-coordinates ecef`,
`--gt-orientation-coordinates local`,
`--gt-to-factor-translation -0.15905 -0.00067 0.16824`, and frame hypothesis
`m2dgr_xsens_enu_to_lidar`.

Outputs are `cluster_covariance_per_factor.csv`, `cluster_covariance_summary.csv`, and
`cluster_covariance_summary.json`. The command fails rather than reporting results when
production-covariance reconstruction exceeds the configured relative-error tolerance.

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
