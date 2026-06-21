#!/usr/bin/env python3

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "reliability_analysis" / "compute_factor_nees.py"
SPEC = importlib.util.spec_from_file_location("compute_factor_nees", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FactorNeesTest(unittest.TestCase):
    def test_se3_log_translation(self):
        error = MODULE.se3_log(np.eye(3), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(error, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0])

    def test_body_offset_rotates_with_ground_truth_pose(self):
        rotation_world_gt = MODULE.rpy_degrees_to_matrix((0.0, 0.0, 90.0))
        rotation_world_factor, translation_world_factor = MODULE.compose_body_transform(
            rotation_world_gt,
            np.array([1.0, 2.0, 3.0]),
            np.eye(3),
            np.array([0.42, 0.0, 0.34]),
        )
        np.testing.assert_allclose(rotation_world_factor, rotation_world_gt, atol=1e-12)
        np.testing.assert_allclose(
            translation_world_factor, [1.0, 2.42, 3.34], atol=1e-12
        )

    def test_matrix_quaternion_round_trip(self):
        rotation = MODULE.rpy_degrees_to_matrix((12.0, -8.0, 37.0))
        quaternion = MODULE.matrix_to_quaternion(rotation)
        np.testing.assert_allclose(MODULE.quaternion_to_matrix(quaternion), rotation, atol=1e-12)

    def test_exact_relative_measurements_have_zero_nees(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gt_path = root / "gt.txt"
            gt_path.write_text(
                "0.0 0 0 0 0 0 0 1\n"
                "1.0 1 0 0 0 0 0 1\n"
                "2.0 2 0 0 0 0 0 1\n",
                encoding="utf-8",
            )

            factor_path = root / "factors.csv"
            fieldnames = sorted(MODULE.REQUIRED_COLUMNS | {"mean_geometry_reliability"})
            rows = []
            for index in [1, 2]:
                row = {name: "0" for name in fieldnames}
                row.update(
                    {
                        "timestamp": str(float(index)),
                        "previous_keyframe_timestamp": str(float(index - 1)),
                        "keyframe_index": str(index),
                        "mapped_covariance_available": "1",
                        "factor_covariance_used_adaptive": "1",
                        "factor_covariance_mode": "4",
                        "used_fixed_fallback": "0",
                        "factor_rel_tx": "1",
                        "factor_rel_qw": "1",
                        "mean_combined_reliability": "0.9",
                        "mean_geometry_reliability": "0.95",
                        "effective_correspondence_ratio": "0.9",
                    }
                )
                for matrix_index in range(6):
                    row[f"mapped_cov_{matrix_index}_{matrix_index}"] = (
                        "1e-6" if matrix_index < 3 else "1e-4"
                    )
                rows.append(row)

            with factor_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            output_dir = root / "output"
            arguments = [
                str(SCRIPT),
                "--factors",
                str(factor_path),
                "--gt",
                str(gt_path),
                "--output-dir",
                str(output_dir),
                "--confirm-same-body-frame",
            ]
            with patch.object(sys, "argv", arguments):
                self.assertEqual(MODULE.main(), 0)

            summary = json.loads(
                (output_dir / "factor_confidence_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["factors_evaluated"], 2)
            self.assertAlmostEqual(summary["nees"]["mean"], 0.0)
            self.assertAlmostEqual(summary["inside_upper_confidence_pct"], 100.0)

    def test_gt_to_factor_transform_changes_relative_translation_during_rotation(self):
        rotation0 = np.eye(3)
        rotation1 = MODULE.rpy_degrees_to_matrix((0.0, 0.0, 90.0))
        offset = np.array([1.0, 0.0, 0.0])
        factor_rotation0, factor_translation0 = MODULE.compose_body_transform(
            rotation0, np.zeros(3), np.eye(3), offset
        )
        factor_rotation1, factor_translation1 = MODULE.compose_body_transform(
            rotation1, np.zeros(3), np.eye(3), offset
        )
        relative_rotation, relative_translation = MODULE.relative_pose(
            factor_rotation0,
            factor_translation0,
            factor_rotation1,
            factor_translation1,
        )
        np.testing.assert_allclose(relative_rotation, rotation1, atol=1e-12)
        np.testing.assert_allclose(relative_translation, [-1.0, 1.0, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
