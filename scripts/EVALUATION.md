# Reproducible Trajectory Evaluation

## Common Timestamp Grid

`evaluate_common_grid.py` finds the interval shared by ground truth and every requested
method, creates a fixed-rate timestamp grid, and interpolates all trajectories onto the
same timestamps. Linear interpolation is used for translation and quaternion SLERP for
orientation. Samples that cross an excessive GT or estimator gap are removed from every
trajectory, preserving an identical comparison set.

```bash
ros2 run lio_sam evaluate_common_grid.py \
  --tum-dir /path/to/run/tum_trajectories \
  --gt /path/to/gt.txt \
  --methods fixed residual_only proposed_split \
  --grid-rate 10 \
  --max-trajectory-gap 5 \
  --prefix sequence_repeat1
```

The command writes synchronized files and metadata under `tum_trajectories/common_grid/`,
then invokes the existing EVO evaluator. Use `--t-start` and `--t-end` for a predefined
evaluation window. Use `--resample-only` to skip EVO.

## Repeated Runs

After common-grid evaluation has been run for every repeat:

```bash
ros2 run lio_sam aggregate_repeated_ape.py \
  --run-dirs /path/to/repeat1 /path/to/repeat2 /path/to/repeat3 \
  --methods fixed residual_only proposed_split \
  --output-dir /path/to/repeated_summary
```

The aggregator reports every run, excludes methods with missing results or fatal launch
errors, and produces per-method RMSE mean, sample standard deviation, median, range, and
Student-t confidence intervals. Partial trajectories from crashed estimators are not
silently ranked. Pass `--include-unhealthy` only for debugging, never for paper tables.

For short-window bag experiments, use a slower replay rate and a proportionally longer
wall-clock limit so computationally expensive methods process the same bag interval.
Keep estimator configuration fixed across repeats.

## Correspondence-Level Covariance Probe

For the logging-only cluster/sandwich covariance feasibility experiment, pass
`--enable-per-correspondence-csv` to `run_ablation_bags.py`. This produces final-pose
correspondence rows and the exact LM-to-factor Jacobian for every accepted adaptive
factor. Keep this flag off for ordinary ablations because the CSV output is large. The
schema, reconstruction rule, and controlled Walk01/Plaza01 commands are documented in
`scripts/reliability_analysis/README.md`.

After collecting those logs, use `evaluate_cluster_robust_covariance.py` for the offline
IID/voxel sandwich ablation. This command does not alter or rerun LIO-SAM; it evaluates
candidate factor covariances against synchronized GT and preserves production covariance
as the baseline. See the reliability-analysis README for frame-specific commands.

## Range-Aware Factor Covariance

The `range_aware` ablation uses raw-Hessian factor covariance with the per-sensor
`factorRangeNoiseAlpha` from the selected dataset YAML. M2UD uses `0.0`; M2DGR uses
`0.17`. The `raw_hessian` method always forces alpha to zero so it remains a clean
classical-Hessian baseline.

Run the comparison with:

```bash
ros2 run lio_sam run_ablation_bags.py \
  --dataset DATASET \
  --sequence SEQUENCE \
  --bag /path/to/bag \
  --config /path/to/LIO-SAM/config/DATASET.yaml \
  --launch run_DATASET.launch.py \
  --methods fixed raw_hessian range_aware \
  --workspace /home/unitree/ros2_workspaces/lio_ws \
  --play-args=--clock
```

Ordinary online validation needs aggregate diagnostics but not the large
per-correspondence CSV. Confirm `factor_range_noise_alpha` and `mean_range_weight` in
the keyframe-factor CSV before computing block-wise NEES.
