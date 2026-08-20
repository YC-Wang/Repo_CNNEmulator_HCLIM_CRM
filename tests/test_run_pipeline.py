from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_pipeline


class RunPipelineTests(unittest.TestCase):
    def test_wrapper_stops_before_inference_if_training_fails(self) -> None:
        config = {
            "metadata": {"experiment_id": "pr_test"},
            "paths": {
                "predictor_file": "../predictors.nc",
                "target_file": "../target.nc",
                "output_root": "../outputs",
            },
            "experiment": {
                "variable": "pr",
                "target_scale": 86400.0,
                "std_epsilon": 1.0e-6,
                "predictor_time_offset_hours": 3,
                "dates": {
                    "train": ["2000-01-01", "2000-01-02"],
                    "val": ["2000-01-03", "2000-01-03"],
                    "test": ["2000-01-04", "2000-01-04"],
                },
                "downscale_variables": ["var_a"],
            },
            "model": {
                "hidden_layer_dense": 256,
                "kernel_size": 5,
                "layer_filters": [16, 32, 64],
                "dropout": 0.6,
                "dense_activation": "selu",
                "cnn_activation": "selu",
                "padding": "valid",
                "use_bn": True,
                "use_pooling": True,
            },
            "training": {
                "learning_rate": 0.0001,
                "batch_size": 64,
                "epochs": 300,
                "loss": "mse",
                "metrics": "mse",
            },
            "inference": {
                "batch_size": 32,
                "calculate_test_metrics": True,
                "clip_negative_rainfall": False,
                "saved_output_units": "scaled",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)

            with mock.patch("run_pipeline.run_command", side_effect=[2]) as run_command:
                with mock.patch.object(sys, "argv", ["run_pipeline.py", str(config_path), "--stage", "all"]):
                    exit_code = run_pipeline.main()

        self.assertEqual(exit_code, 2)
        self.assertEqual(run_command.call_count, 1)


if __name__ == "__main__":
    unittest.main()
