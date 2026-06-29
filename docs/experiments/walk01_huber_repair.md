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

## Corrected factor NEES

Walk01 GT positions are ECEF while the Xsens attitude is treated as local ENU. Run this
command for each covariance method by changing `METHOD`:

```bash
METHOD=proposed_split
ros2 run lio_sam compute_factor_nees.py \
  --factors "/home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2DGR/walk01_absolute_120s/${METHOD}/walk01_absolute_120s_${METHOD}_keyframe_factors.csv" \
  --gt /home/unitree/ros2_workspaces/lio_ws/trajectory_save/M2DGR/walk_01/gt.txt \
  --output-dir "/home/unitree/ros2_workspaces/lio_ws/lio_sam_logs/M2DGR/walk01_absolute_120s/${METHOD}/factor_confidence_enu_xsens" \
  --dataset M2DGR \
  --sequence walk01_absolute_120s \
  --method "${METHOD}" \
  --run-id "walk01_absolute_120s_${METHOD}_enu_xsens" \
  --gt-position-coordinates ecef \
  --gt-orientation-coordinates local \
  --gt-to-factor-translation -0.15905 -0.00067 0.16824 \
  --gt-to-factor-rpy-deg 0 0 0 \
  --frame-hypothesis m2dgr_xsens_enu_to_lidar \
  --confirm-same-body-frame
```

| method | mean NEES | normalized NEES | mean NLL | upper 95% coverage | translation error (m) |
|---|---:|---:|---:|---:|---:|
| fixed | **104.17** | **17.36** | **23.06** | 17.95% | 0.0590 |
| raw Hessian | 276.14 | 46.02 | 108.95 | 11.27% | 0.0622 |
| residual only | 170.33 | 28.39 | 56.71 | 12.82% | 0.0606 |
| geometry covariance only | 220.53 | 36.75 | 81.77 | 14.29% | 0.0641 |
| post split only | 179.90 | 29.98 | 62.04 | **22.37%** | 0.0635 |
| proposed split | 148.87 | 24.81 | 46.56 | 20.00% | **0.0587** |

All methods are strongly overconfident. Fixed has the best mean NEES/NLL; proposed has
the lowest translation disagreement but does not provide the best calibrated covariance.
