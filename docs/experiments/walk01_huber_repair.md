# Walk01 Huber Repair Run

The first Huber trajectory ended after approximately 86 seconds without a fatal process
error. Rerun only Huber in the existing sequence directory before using the nine-method
Walk01 table.

```bash
source /home/unitree/ros2_workspaces/lio_ws/install/setup.bash

ros2 run lio_sam run_ablation_bags.py \
  --dataset M2DGR \
  --sequence walk01_absolute_120s \
  --bag /mnt/sdcard/rosbags/M2DGR/walk_01 \
  --config /home/unitree/ros2_workspaces/lio_ws/src/LIO-SAM/config/M2DGR.yaml \
  --launch run_M2DGR.launch.py \
  --methods huber \
  --workspace /home/unitree/ros2_workspaces/lio_ws \
  --play-args="--clock --rate 1.0" \
  --bag-duration 130 \
  --settle-time 20
```

Re-evaluate the nine-method common grid:

```bash
ros2 run lio_sam evaluate_common_grid.py \
  --tum-dir /home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2DGR/walk01_absolute_120s/tum_trajectories \
  --gt /home/unitree/ros2_workspaces/lio_ws/trajectory_save/M2DGR/walk_01/gt.txt \
  --methods fixed huber cauchy geometry_registration raw_hessian residual_only geometry_cov_only post_split_only proposed_split \
  --grid-rate 10 \
  --max-trajectory-gap 5 \
  --t-start 1628059253.253469944 \
  --t-end 1628059359.658053875 \
  --prefix walk01_full_common_repaired
```

## Completed result

The repaired Huber trajectory contains 83 poses over 125.847 seconds and has no fatal
launch-log signature. The nine-method common grid contains 1065 samples over 106.4
seconds.

| method | RMSE (m) |
|---|---:|
| fixed | 0.096188 |
| Huber | 0.110963 |
| Cauchy | 0.101049 |
| geometry registration | 0.094510 |
| raw Hessian | 0.106412 |
| residual only | 0.104181 |
| geometry covariance only | 0.098887 |
| post split only | 0.099774 |
| proposed split | **0.094396** |

Proposed split is best, although its margin over geometry registration is only 0.1% and
its margin over fixed is 1.9%.
