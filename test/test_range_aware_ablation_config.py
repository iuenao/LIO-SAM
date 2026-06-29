#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_ablation_bags.py"
SPEC = importlib.util.spec_from_file_location("run_ablation_bags", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RangeAwareAblationConfigTest(unittest.TestCase):
    def generated_params(self, base_config: str, method: str):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_config = root / f"{method}.yaml"
            MODULE.write_method_config(
                ROOT / "config" / base_config,
                output_config,
                "test_dataset",
                "test_sequence",
                method,
                root / "test_bag",
                root / method,
            )
            return yaml.safe_load(output_config.read_text(encoding="utf-8"))["/**"][
                "ros__parameters"
            ]

    def test_range_aware_inherits_m2dgr_sensor_alpha(self):
        params = self.generated_params("M2DGR.yaml", "range_aware")

        self.assertEqual(params["factorCovarianceMode"], 1)
        self.assertEqual(params["registrationWeightMode"], 0)
        self.assertAlmostEqual(params["factorRangeNoiseAlpha"], 0.17)

    def test_raw_hessian_disables_range_weight_for_clean_baseline(self):
        params = self.generated_params("M2DGR.yaml", "raw_hessian")

        self.assertEqual(params["factorCovarianceMode"], 1)
        self.assertEqual(params["factorRangeNoiseAlpha"], 0.0)

    def test_m2ud_range_aware_reduces_to_raw_hessian(self):
        params = self.generated_params("M2UD.yaml", "range_aware")

        self.assertEqual(params["factorCovarianceMode"], 1)
        self.assertEqual(params["factorRangeNoiseAlpha"], 0.0)


if __name__ == "__main__":
    unittest.main()
