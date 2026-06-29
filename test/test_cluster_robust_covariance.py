#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_DIR = Path(__file__).parents[1] / "scripts" / "reliability_analysis"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "evaluate_cluster_robust_covariance.py"
SPEC = importlib.util.spec_from_file_location("evaluate_cluster_robust_covariance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def arguments():
    return SimpleNamespace(
        information_damping=0.0,
        information_eigenvalue_min=1.0e-12,
        information_eigenvalue_max=1.0e12,
    )


class ClusterRobustCovarianceTest(unittest.TestCase):
    def test_summary_orders_named_baselines_before_voxels(self):
        rows = []
        for variant, voxel in (("voxel_0.5m", 0.5), ("production", ""), ("iid_sandwich", "")):
            rows.append(
                {
                    "variant": variant,
                    "voxel_size_m": voxel,
                    "regularized_eigenvalues": 0,
                    "inside_two_sided": 1,
                    "inside_upper": 1,
                    "nees": 6.0,
                    "rotation_nees": 3.0,
                    "translation_nees": 3.0,
                    "rotation_x_nees": 1.0,
                    "rotation_y_nees": 1.0,
                    "rotation_z_nees": 1.0,
                    "translation_x_nees": 1.0,
                    "translation_y_nees": 1.0,
                    "translation_z_nees": 1.0,
                    "nll": 0.0,
                    "covariance_trace": 1.0,
                    "rotation_covariance_trace": 0.5,
                    "translation_covariance_trace": 0.5,
                    "cluster_count": 10,
                    "cluster_ratio": 0.5,
                    "finite_sample_correction": 1.0,
                }
            )
        summaries = MODULE.summarize(rows, 0.95)
        self.assertEqual(
            [row["variant"] for row in summaries],
            ["production", "iid_sandwich", "voxel_0.5m"],
        )

    def test_spatial_clusters_are_separated_by_feature_type(self):
        scores = np.arange(24, dtype=float).reshape(4, 6)
        points = np.array(
            [[0.10, 0.10, 0.10], [0.20, 0.20, 0.20], [0.15, 0.15, 0.15], [1.10, 0.10, 0.10]]
        )
        features = np.array([1, 1, 2, 1])

        separated = MODULE.cluster_score_matrix(scores, points, features, 1.0, True)
        combined = MODULE.cluster_score_matrix(scores, points, features, 1.0, False)

        self.assertEqual(len(separated), 3)
        self.assertEqual(len(combined), 2)
        np.testing.assert_allclose(np.sum(separated, axis=0), np.sum(scores, axis=0))
        np.testing.assert_allclose(np.sum(combined, axis=0), np.sum(scores, axis=0))

    def test_coherent_clusters_increase_sandwich_covariance(self):
        jacobians = np.vstack((np.eye(6), np.eye(6)))
        weights = np.ones(12)
        residuals = np.ones(12)
        points = np.vstack((np.zeros((6, 3)), np.full((6, 3), 0.1)))
        features = np.ones(12, dtype=np.int64)
        information = jacobians.T @ jacobians

        iid, iid_clusters, _, _ = MODULE.sandwich_covariance(
            information,
            jacobians,
            weights,
            residuals,
            points,
            features,
            None,
            True,
            False,
            arguments(),
        )
        clustered, spatial_clusters, _, _ = MODULE.sandwich_covariance(
            information,
            jacobians,
            weights,
            residuals,
            points,
            features,
            1.0,
            True,
            False,
            arguments(),
        )

        self.assertEqual(iid_clusters, 12)
        self.assertEqual(spatial_clusters, 1)
        self.assertGreater(np.trace(clustered), np.trace(iid))


if __name__ == "__main__":
    unittest.main()
