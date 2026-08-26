from __future__ import annotations

import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_utils import (
    build_output_paths,
    ensure_can_write_inference_outputs,
    ensure_can_write_training_outputs,
    get_config_warnings,
    get_experiment_dir,
    get_inference_dates,
    get_inference_run,
    get_inference_runs,
    resolve_config_paths,
    write_netcdf_with_time_validation,
)


def load_inference_module():
    script_path = REPO_ROOT / "scripts" / "inference_ncp_mse.py"
    spec = importlib.util.spec_from_file_location("inference_ncp_mse_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PipelineUtilsTests(unittest.TestCase):
    def canonical_config(self, output_root: str = "./outputs") -> dict:
        return {
            "metadata": {
                "dataset_version": "FPS_HCLIM43_ITERIM",
            },
            "paths": {
                "predictor_file": "./predictors.nc",
                "target_file": "./target.nc",
                "output_root": output_root,
            },
            "experiment": {
                "variable": "tas",
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
                "model_type": "cnn",
                "overwrite_existing": False,
            },
            "inference": {
                "run_name": "test",
                "batch_size": 32,
                "calculate_test_metrics": True,
                "prediction_filename": "prediction.nc",
                "metrics_filename": "evaluation_metrics.csv",
                "overwrite_existing": False,
            },
        }

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
                "log_root": "../output/logs/",
                "model_root": "../output/models/",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "exp1_temperature.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(config, config_file)
            warnings = get_config_warnings(config, resolved, config_file)

        self.assertEqual(resolved["metadata"]["experiment_id"], "exp1_temperature")
        self.assertTrue(Path(resolved["paths"]["predictor_file"]).is_absolute())
        self.assertTrue(Path(resolved["paths"]["target_file"]).is_absolute())
        self.assertTrue(Path(resolved["paths"]["output_root"]).is_absolute())
        self.assertEqual(resolved["experiment"]["target_scale"], 1.0)
        self.assertEqual(resolved["experiment"]["predictor_time_offset_hours"], 0)
        self.assertEqual(resolved["experiment"]["std_epsilon"], 1.0e-6)
        self.assertEqual(resolved["inference"]["batch_size"], 32)
        self.assertTrue(any("Deprecated config fields detected" in message for message in warnings))

    def test_relative_output_root_is_resolved_from_config_directory(self) -> None:
        config = self.canonical_config(output_root="./outputs")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "example.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(config, config_file)

        self.assertEqual(Path(resolved["paths"]["output_root"]), (config_file.parent / "outputs").resolve())

    def test_absolute_output_root_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            absolute_output = str((Path(tmpdir) / "shared_outputs").resolve())
            config = self.canonical_config(output_root=absolute_output)
            config_file = Path(tmpdir) / "configs" / "example.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(config, config_file)

        self.assertEqual(resolved["paths"]["output_root"], absolute_output)

    def test_default_and_explicit_experiment_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "derived_name.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)

            default_resolved = resolve_config_paths(self.canonical_config(), config_file)
            explicit_config = self.canonical_config()
            explicit_config["metadata"]["experiment_id"] = "custom_exp"
            explicit_resolved = resolve_config_paths(explicit_config, config_file)

        self.assertEqual(default_resolved["metadata"]["experiment_id"], "derived_name")
        self.assertEqual(explicit_resolved["metadata"]["experiment_id"], "custom_exp")

    def test_build_output_paths_creates_canonical_directory_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "example.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(self.canonical_config(), config_file)
            paths = build_output_paths(resolved)

            for key in (
                "experiment_dir",
                "config_dir",
                "models_dir",
                "normalization_dir",
                "logs_dir",
                "bootstrap_log_dir",
                "tensorboard_dir",
                "inference_logs_dir",
                "inference_root_dir",
                "inference_run_dir",
            ):
                self.assertTrue(paths[key].is_dir(), msg=key)

    def test_canonical_training_and_inference_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "example.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(self.canonical_config(), config_file)
            paths = build_output_paths(resolved, run_name="eval_2009")

        experiment_dir = get_experiment_dir(resolved)
        self.assertEqual(paths["experiment_dir"], experiment_dir)
        self.assertEqual(paths["model_file"], experiment_dir / "models" / "best_model.h5")
        self.assertEqual(paths["tensorboard_dir"], experiment_dir / "logs" / "tensorboard")
        self.assertEqual(paths["training_log_file"], experiment_dir / "logs" / "training.log")
        self.assertEqual(paths["inference_run_dir"], experiment_dir / "inference" / "eval_2009")
        self.assertEqual(paths["inference_log_file"], experiment_dir / "logs" / "inference" / "eval_2009.log")

    def test_default_inference_model_resolution_and_override(self) -> None:
        inference_module = load_inference_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "example.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(self.canonical_config(), config_file)
            paths = build_output_paths(resolved)
            paths["model_file"].write_text("stub", encoding="utf-8")
            external_model = Path(tmpdir) / "external_model.h5"
            external_model.write_text("external", encoding="utf-8")

            default_path = inference_module.resolve_inference_model_path(
                Namespace(model_file=None),
                resolved,
                config_file,
                paths,
            )
            override_path = inference_module.resolve_inference_model_path(
                Namespace(model_file=str(external_model)),
                resolved,
                config_file,
                paths,
            )

        self.assertEqual(default_path, paths["model_file"])
        self.assertEqual(override_path, external_model)

    def test_inference_dates_fallback_and_explicit_value(self) -> None:
        default_config = self.canonical_config()
        explicit_config = self.canonical_config()
        explicit_config["inference"]["dates"] = ["2010-01-01", "2010-12-31"]

        self.assertEqual(get_inference_dates(default_config), ["2009-01-01", "2009-12-31"])
        self.assertEqual(get_inference_dates(explicit_config), ["2010-01-01", "2010-12-31"])

    def test_prediction_and_metrics_filenames_are_configurable(self) -> None:
        config = self.canonical_config()
        config["inference"]["prediction_filename"] = "predictions_custom.nc"
        config["inference"]["metrics_filename"] = "metrics_custom.csv"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "example.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(config, config_file)
            paths = build_output_paths(resolved)

        self.assertEqual(paths["prediction_file"].name, "predictions_custom.nc")
        self.assertEqual(paths["metrics_file"].name, "metrics_custom.csv")

    def test_get_inference_runs_and_named_run_selection(self) -> None:
        config = self.canonical_config()
        config["inference"]["runs"] = [
            {
                "name": "test_2009",
                "predictor_file": "./predictors_2009.nc",
                "target_file": "./target_2009.nc",
                "dates": ["2009-01-01", "2009-12-31"],
            },
            {
                "name": "future_2050",
                "predictor_file": "./predictors_2050.nc",
                "target_file": "./target_2050.nc",
                "dates": ["2050-01-01", "2050-12-31"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "example.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(config, config_file)

        runs = get_inference_runs(resolved)
        selected = get_inference_run(resolved, "future_2050")
        self.assertEqual(len(runs), 2)
        self.assertEqual(selected["name"], "future_2050")
        self.assertTrue(Path(selected["predictor_file"]).is_absolute())

    def test_training_and_inference_overwrite_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "example.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(self.canonical_config(), config_file)
            paths = build_output_paths(resolved)
            paths["model_file"].write_text("model", encoding="utf-8")
            paths["prediction_file"].write_text("prediction", encoding="utf-8")
            paths["metrics_file"].write_text("metrics", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                ensure_can_write_training_outputs(paths, overwrite_existing=False)
            with self.assertRaises(FileExistsError):
                ensure_can_write_inference_outputs(paths, overwrite_existing=False, calculate_test_metrics=True)

            ensure_can_write_training_outputs(paths, overwrite_existing=True)
            ensure_can_write_inference_outputs(paths, overwrite_existing=True, calculate_test_metrics=True)

    def test_write_netcdf_with_time_validation_preserves_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "prediction.nc"
            times = pd.date_range("2009-01-01 00:00:00", periods=4, freq="6h")
            dataset = xr.Dataset(
                {
                    "tas": (("time", "y", "x"), np.arange(16, dtype=float).reshape(4, 2, 2)),
                },
                coords={"time": times, "y": [0, 1], "x": [0, 1]},
            )

            summary = write_netcdf_with_time_validation(dataset, output_file)
            reopened = xr.open_dataset(output_file, decode_times=True)
            reopened_times = reopened["time"].values.copy()
            reopened.close()

        self.assertEqual(summary["first"], str(times.values[0]))
        self.assertEqual(summary["last"], str(times.values[-1]))
        self.assertEqual(summary["count"], 4)
        self.assertTrue(np.array_equal(times.values, reopened_times))

    def test_training_and_inference_share_the_same_experiment_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "configs" / "example.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            resolved = resolve_config_paths(self.canonical_config(), config_file)
            training_paths = build_output_paths(resolved, run_name="test")
            inference_paths = build_output_paths(resolved, run_name="future")

        self.assertEqual(training_paths["experiment_dir"], inference_paths["experiment_dir"])
        self.assertEqual(training_paths["model_file"], inference_paths["model_file"])


if __name__ == "__main__":
    unittest.main()
