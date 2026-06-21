# M2UD Reliability-Aware Ablation Profiles

These files record the method settings used by `scripts/run_ablation_bags.py`.
They are lightweight ROS 2 parameter overlays, not full dataset configs.

Main paper table:

- `fixed`
- `huber`
- `cauchy`
- `raw_hessian`
- `residual_only`
- `geometry_cov_only`
- `post_split_only`
- `proposed_split_norm`

Scale-attribution variants:

- `raw_hessian_norm`
- `post_split_only_norm`
- `post_split_only_norm_floor`
- `proposed_split`
- `proposed_split_norm_blend80`
- `proposed_split_norm_blend50`
- `proposed_split_norm_floor`
- `proposed_split_norm_blend80_floor`
- `proposed_split_norm_blend50_floor`

Failure-analysis table:

- `geometry_registration`
- `legacy_residual_solve`
- `legacy_combined_solve`

Registration/covariance factorial comparison:

| method | registration weighting | reliability covariance |
| --- | --- | --- |
| `fixed` | no | no |
| `geometry_registration` | yes | no |
| `post_split_only_norm_floor` | no | yes |
| `proposed_split_norm_floor` | yes | yes |
