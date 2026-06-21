#!/usr/bin/env python3

import csv
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load_module("evaluate_common_grid", ROOT / "scripts/evaluate_common_grid.py")
AGGREGATE = load_module("aggregate_repeated_ape", ROOT / "scripts/aggregate_repeated_ape.py")


def write_linear_tum(path: Path, timestamps, offset: float = 0.0):
    with path.open("w", encoding="utf-8") as stream:
        for timestamp in timestamps:
            stream.write(f"{timestamp:.9f} {timestamp + offset:.9f} 0 0 0 0 0 1\n")


class CommonGridTest(unittest.TestCase):
    def test_resamples_every_trajectory_to_identical_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gt = root / "gt.txt"
            fixed = root / "sequence_fixed_tum.txt"
            proposed = root / "sequence_proposed_split_tum.txt"
            write_linear_tum(gt, np.arange(0.0, 5.1, 0.1))
            write_linear_tum(fixed, [0.0, 2.0, 4.0], offset=1.0)
            write_linear_tum(proposed, [0.5, 2.5, 4.5], offset=2.0)

            output = root / "common"
            metadata = COMMON.create_common_grid(
                gt,
                {"fixed": fixed, "proposed_split": proposed},
                output,
                grid_rate=2.0,
                max_trajectory_gap=2.1,
                max_gt_gap=0.2,
                requested_start=None,
                requested_end=None,
            )

            gt_times, gt_positions, _ = COMMON.load_tum(output / "gt.txt")
            fixed_times, fixed_positions, _ = COMMON.load_tum(output / fixed.name)
            proposed_times, proposed_positions, _ = COMMON.load_tum(output / proposed.name)
            self.assertEqual(metadata["common_samples"], 8)
            np.testing.assert_allclose(gt_times, fixed_times)
            np.testing.assert_allclose(gt_times, proposed_times)
            np.testing.assert_allclose(fixed_positions[:, 0], gt_positions[:, 0] + 1.0)
            np.testing.assert_allclose(proposed_positions[:, 0], gt_positions[:, 0] + 2.0)

    def test_rejects_interpolation_across_large_gap(self):
        timestamps = np.array([0.0, 4.0])
        translations = np.zeros((2, 3))
        quaternions = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        _, _, valid = COMMON.interpolate_trajectory(
            timestamps, translations, quaternions, np.array([2.0]), max_gap=1.0
        )
        self.assertFalse(valid[0])


class RepeatedAggregationTest(unittest.TestCase):
    def test_find_summary_uses_newest_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results = root / "tum_trajectories/common_grid/ape_results"
            results.mkdir(parents=True)
            alphabetically_last = results / "z_old_ape_summary.csv"
            newest = results / "a_new_ape_summary.csv"
            alphabetically_last.write_text("old\n", encoding="utf-8")
            newest.write_text("new\n", encoding="utf-8")
            os.utime(alphabetically_last, ns=(1, 1))
            os.utime(newest, ns=(2, 2))

            self.assertEqual(AGGREGATE.find_summary(root), newest)

    def create_run(self, root: Path, name: str, fixed_rmse: float, proposed_rmse: float, fatal=False):
        run = root / name
        result_dir = run / "tum_trajectories/common_grid/ape_results"
        result_dir.mkdir(parents=True)
        fields = ["", "rmse", "mean", "median", "std", "min", "max", "sse"]
        with (result_dir / f"{name}_ape_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for method, rmse in (("fixed", fixed_rmse), ("proposed_split", proposed_rmse)):
                writer.writerow(
                    {
                        "": str(result_dir / f"{name}_{method}_ape.zip"),
                        "rmse": rmse,
                        "mean": rmse,
                        "median": rmse,
                        "std": 0.0,
                        "min": rmse,
                        "max": rmse,
                        "sse": rmse * rmse,
                    }
                )
        for method in ("fixed", "proposed_split"):
            method_dir = run / method
            method_dir.mkdir(parents=True)
            text = "terminate called\n" if fatal and method == "proposed_split" else "clean\n"
            (method_dir / "launch.log").write_text(text, encoding="utf-8")
        return run

    def test_excludes_unhealthy_method_runs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run1 = self.create_run(root, "run1", 0.1, 0.08)
            run2 = self.create_run(root, "run2", 0.2, 0.09, fatal=True)
            _, summary = AGGREGATE.aggregate_runs(
                [run1, run2],
                ["fixed", "proposed_split"],
                root / "aggregate",
                confidence=0.95,
                include_unhealthy=False,
            )
            by_method = {row["method"]: row for row in summary}
            self.assertEqual(by_method["fixed"]["runs_included"], 2)
            self.assertAlmostEqual(by_method["fixed"]["mean_rmse"], 0.15)
            self.assertEqual(by_method["proposed_split"]["runs_included"], 1)
            self.assertEqual(by_method["proposed_split"]["failures_or_missing"], 1)


if __name__ == "__main__":
    unittest.main()
