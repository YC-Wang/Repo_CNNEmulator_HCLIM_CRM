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
