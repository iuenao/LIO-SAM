# Plaza02 First-200-Second Repeated Runs

## Frozen protocol

- Dataset: M2UD Plaza02
- Playback rate: `1.0`
- Wall-clock bag limit: `200 s`
- Evaluation window: `1733809830.675062418` to `1733810014.3725405`
- Common-grid rate: `10 Hz`
- Methods: fixed, Huber, Cauchy, geometry registration, residual-only covariance,
  proposed split

## Repeat 2

```bash
source /home/unitree/ros2_workspaces/lio_ws/install/setup.bash

ros2 run lio_sam run_ablation_bags.py \
  --dataset M2UD \
  --sequence plaza02_repeat2_200s \
  --bag /mnt/sdcard/rosbags/M2UD/plaza_02 \
  --config /home/unitree/ros2_workspaces/lio_ws/src/LIO-SAM/config/M2UD.yaml \
  --launch run_M2UD.launch.py \
  --methods fixed huber cauchy geometry_registration residual_only proposed_split \
  --workspace /home/unitree/ros2_workspaces/lio_ws \
  --play-args="--clock --rate 1.0" \
  --bag-duration 200 \
  --settle-time 15
```

## Repeat 3

```bash
ros2 run lio_sam run_ablation_bags.py \
  --dataset M2UD \
  --sequence plaza02_repeat3_200s \
  --bag /mnt/sdcard/rosbags/M2UD/plaza_02 \
  --config /home/unitree/ros2_workspaces/lio_ws/src/LIO-SAM/config/M2UD.yaml \
  --launch run_M2UD.launch.py \
  --methods fixed huber cauchy geometry_registration residual_only proposed_split \
  --workspace /home/unitree/ros2_workspaces/lio_ws \
  --play-args="--clock --rate 1.0" \
  --bag-duration 200 \
  --settle-time 15
```

## Common-grid evaluation

For repeat 1, use `plaza02_absolute_first200s` as `RUN`. For repeats 2 and 3,
use their corresponding repeat directory:

```bash
RUN=plaza02_repeat2_200s
PREFIX=plaza02_repeat2_common
ros2 run lio_sam evaluate_common_grid.py \
  --tum-dir "/home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD/${RUN}/tum_trajectories" \
  --gt /home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD_old/plaza02_200s/tum_trajectories/gt.txt \
  --methods fixed huber cauchy geometry_registration residual_only proposed_split \
  --grid-rate 10 \
  --max-trajectory-gap 5 \
  --t-start 1733809830.675062418 \
  --t-end 1733810014.3725405 \
  --prefix "${PREFIX}"
```

## Aggregate all three runs

```bash
ros2 run lio_sam aggregate_repeated_ape.py \
  --run-dirs \
    /home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD/plaza02_absolute_first200s \
    /home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD/plaza02_repeat2_200s \
    /home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD/plaza02_repeat3_200s \
  --methods fixed huber cauchy geometry_registration residual_only proposed_split \
  --output-dir /home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2UD/plaza02_repeated_summary
```

## Completed results

All evaluations contain 1837 identical 10 Hz samples over 183.6 seconds. RMSE is
translation APE after SE(3) Umeyama alignment.

| method | repeat 1 | repeat 2 | repeat 3 | mean +/- sample std (m) |
|---|---:|---:|---:|---:|
| fixed | 0.127788 | 0.132577 | 0.131582 | 0.130649 +/- 0.002527 |
| Huber | 0.090479 | **0.082528** | **0.082235** | **0.085081 +/- 0.004677** |
| Cauchy | 0.110026 | 0.112552 | 0.098762 | 0.107113 +/- 0.007342 |
| geometry registration | **0.088550** | 0.087354 | 0.096393 | 0.090765 +/- 0.004910 |
| residual only | 0.110857 | 0.109006 | 0.121775 | 0.113879 +/- 0.006900 |
| proposed split | 0.089344 | 0.090478 | 0.084057 | 0.087960 +/- 0.003427 |

Huber has the lowest three-run mean. Proposed split is 3.4% above Huber, but 32.7%
below fixed and 3.1% below geometry registration. The result does not support universal
dominance over robust kernels.

The old repeat-1 proposed trajectory was rejected because IMU preintegration raised
`IndeterminantLinearSystemException`. It was rerun cleanly at rate 1.0 and the complete
repeat-1 table was reevaluated before aggregation.

Aggregate CSV: `lio_sam_logs/M2UD/plaza02_repeated_summary/repeated_ape_summary.csv`.
