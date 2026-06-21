# Plaza01 Repeated Runs

## Frozen protocol

- Dataset: M2UD Plaza01
- Playback rate: `1.0`
- Wall-clock bag limit: `120 s`
- Evaluation window: `1733480143.945709229` to `1733480252.8310242`
- Common-grid rate: `10 Hz`
- Methods: fixed, Huber, Cauchy, geometry registration, residual-only covariance,
  proposed split

Do not change YAML parameters between repeats.

## Repeat 2

```bash
source /home/unitree/ros2_workspaces/lio_ws/install/setup.bash

ros2 run lio_sam run_ablation_bags.py \
  --dataset M2UD \
  --sequence plaza01_repeat2_120s \
  --bag /mnt/sdcard/rosbags/M2UD/plaza_01 \
  --config /home/unitree/ros2_workspaces/lio_ws/src/LIO-SAM/config/M2UD.yaml \
  --launch run_M2UD.launch.py \
  --methods fixed huber cauchy geometry_registration residual_only proposed_split \
  --workspace /home/unitree/ros2_workspaces/lio_ws \
  --play-args="--clock --rate 1.0" \
  --bag-duration 120 \
  --settle-time 15
```

## Repeat 3

```bash
ros2 run lio_sam run_ablation_bags.py \
  --dataset M2UD \
  --sequence plaza01_repeat3_120s \
  --bag /mnt/sdcard/rosbags/M2UD/plaza_01 \
  --config /home/unitree/ros2_workspaces/lio_ws/src/LIO-SAM/config/M2UD.yaml \
  --launch run_M2UD.launch.py \
  --methods fixed huber cauchy geometry_registration residual_only proposed_split \
  --workspace /home/unitree/ros2_workspaces/lio_ws \
  --play-args="--clock --rate 1.0" \
  --bag-duration 120 \
  --settle-time 15
```

## Common-grid evaluation

For repeat 1, use `plaza01_phase9_absolute_120s` as `RUN`. For repeats 2 and 3,
use their corresponding repeat directory:

```bash
RUN=plaza01_repeat2_120s
PREFIX=plaza01_repeat2_common
ros2 run lio_sam evaluate_common_grid.py \
  --tum-dir "/home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD/${RUN}/tum_trajectories" \
  --gt /home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD/plaza01_phase9_confidence_120s/tum_trajectories/gt.txt \
  --methods fixed huber cauchy geometry_registration residual_only proposed_split \
  --grid-rate 10 \
  --max-trajectory-gap 5 \
  --t-start 1733480143.945709229 \
  --t-end 1733480252.8310242 \
  --prefix "${PREFIX}"
```

## Aggregate all three runs

```bash
ros2 run lio_sam aggregate_repeated_ape.py \
  --run-dirs \
    /home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD/plaza01_phase9_absolute_120s \
    /home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD/plaza01_repeat2_120s \
    /home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD/plaza01_repeat3_120s \
  --methods fixed huber cauchy geometry_registration residual_only proposed_split \
  --output-dir /home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD/plaza01_repeated_summary
```

## Completed results

All evaluations contain 1089 identical 10 Hz samples over 108.8 seconds. RMSE is
translation APE after SE(3) Umeyama alignment.

| method | repeat 1 | repeat 2 | repeat 3 | mean +/- sample std (m) |
|---|---:|---:|---:|---:|
| fixed | 0.067076 | 0.074994 | 0.073071 | 0.071714 +/- 0.004130 |
| Huber | 0.066848 | 0.068029 | 0.065146 | 0.066674 +/- 0.001449 |
| Cauchy | 0.065470 | 0.071757 | 0.075102 | 0.070776 +/- 0.004890 |
| geometry registration | 0.066793 | 0.070198 | 0.072897 | 0.069963 +/- 0.003059 |
| residual only | 0.073583 | 0.067012 | 0.079256 | 0.073284 +/- 0.006127 |
| proposed split | **0.059762** | **0.064926** | 0.066297 | **0.063662 +/- 0.003446** |

The proposed method has the lowest three-run mean, 4.5% below Huber and 11.2% below
fixed. With only three repeats, the 95% Student-t confidence intervals overlap.

Initial repeat-2 fixed/residual-only runs and the initial repeat-3 geometry-registration
run were rejected after IMU preintegration raised `IndeterminantLinearSystemException`.
Each affected method was rerun cleanly before evaluation.

Aggregate CSV: `lio_sam_logs/M2UD/plaza01_repeated_summary/repeated_ape_summary.csv`.
