from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_utils import get_inference_runs, resolve_config_paths


class PipelineUtilsTests(unittest.TestCase):
    def test_resolve_config_paths_supports_legacy_temperature_config(self) -> None:
        config = {
            "metadata": {
                "dataset_version": "FPS_HCLIM43_ITERIM",
            },
            "paths": {
                "work_dir": "./",
                "data_train_dir": "data_sample/train_data/",
                "data_infer_dir": "data_sample/infer_data/",
            },
            "experiment": {
                "variable": "tas",
                "x_filename": "combined_12km_6hr_20000101-20091231_swapped_2003_2009.nc",
                "y_filename_template": "{variable}_3km_6hr_200001010000-200912311800_swapped_2003_2009.nc",
                "dates": {
                    "train": ["2000-01-01", "2007-12-31"],
                    "val": ["2008-01-01", "2008-12-31"],
                    "test": ["2009-01-01", "2009-12-31"],
                },
                "downscale_variables": ["phi500"],
            },
            "training": {
                "batch_size": 64,
                "loss": "mse",
                "metrics": "mse",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "exp1_temperature.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(config, config_file)

        self.assertEqual(resolved["metadata"]["experiment_id"], "exp1_temperature")
        self.assertTrue(Path(resolved["paths"]["predictor_file"]).is_absolute())
        self.assertTrue(Path(resolved["paths"]["target_file"]).is_absolute())
        self.assertTrue(Path(resolved["paths"]["output_root"]).is_absolute())
        self.assertEqual(resolved["experiment"]["target_scale"], 1.0)
        self.assertEqual(resolved["experiment"]["predictor_time_offset_hours"], 0)
        self.assertEqual(resolved["experiment"]["std_epsilon"], 1.0e-6)
        self.assertEqual(resolved["inference"]["batch_size"], 64)
        self.assertTrue(resolved["paths"]["target_file"].endswith("tas_3km_6hr_200001010000-200912311800_swapped_2003_2009.nc"))

    def test_resolve_config_paths_updates_nested_segment_and_run_files(self) -> None:
        config = {
            "paths": {
                "predictor_file": "./predictors.nc",
                "target_file": "./target.nc",
                "output_root": "./outputs",
            },
            "training": {
                "segments": [
                    {
                        "name": "hist",
                        "predictor_file": "./hist_predictors.nc",
                        "target_file": "./hist_target.nc",
                        "dates": ["2001-01-01", "2004-12-31"],
                    }
                ],
            },
            "inference": {
                "runs": [
                    {
                        "name": "future",
                        "predictor_file": "./future_predictors.nc",
                        "target_file": "./future_target.nc",
                        "dates": ["2050-01-01", "2050-12-31"],
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "exp.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(config, config_file)

        self.assertTrue(Path(resolved["paths"]["predictor_file"]).is_absolute())
        self.assertTrue(Path(resolved["training"]["segments"][0]["predictor_file"]).is_absolute())
        self.assertTrue(Path(resolved["inference"]["runs"][0]["target_file"]).is_absolute())

    def test_get_inference_runs_falls_back_to_default_single_run(self) -> None:
        config = {
            "paths": {
                "predictor_file": "/tmp/predictors.nc",
                "target_file": "/tmp/target.nc",
            },
            "experiment": {
                "dates": {
                    "test": ["2005-01-01", "2005-12-31"],
                }
            },
            "inference": {},
        }

        runs = get_inference_runs(config)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["name"], "default")
        self.assertEqual(runs[0]["dates"], ["2005-01-01", "2005-12-31"])


if __name__ == "__main__":
    unittest.main()
