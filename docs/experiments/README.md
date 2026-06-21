# Conference Experiment Runbooks

- `plaza01_repeated_runs.md`: completed three-run synchronized evaluation and commands.
- `plaza02_repeated_runs.md`: completed three-run first-200-second evaluation and commands.
- `walk01_huber_repair.md`: completed Huber repair and corrected nine-method evaluation.

All commands use playback rate `1.0` to match the existing experiments. Generated run
directories are unique except for the intentional Walk01 Huber repair. Do not modify
method YAML parameters between repeats.

Before accepting any result, inspect launch logs for fatal estimator exits. The repeated
APE aggregator performs this check automatically and excludes unhealthy method runs.
