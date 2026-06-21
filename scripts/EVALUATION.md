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
