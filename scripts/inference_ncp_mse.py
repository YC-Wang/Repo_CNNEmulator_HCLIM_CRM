#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_utils import add_file_handler, setup_logging
from src.pipeline_utils import get_git_commit_sha, get_inference_dates, load_yaml_config


LOGGER_NAME = "inference_ncp_mse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML inference with YAML config")
    parser.add_argument(
        "config_file",
        type=str,
        nargs="?",
        default="config.yaml",
        help="Path to the yaml configuration file",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        help="Path to the trained model .h5 file. Overrides inference.model_file in YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for inference artifacts. Overrides inference.output_dir in YAML.",
    )
    parser.add_argument(
        "--smoke-test-imports",
        action="store_true",
        help="Validate script/module imports without loading data or running inference.",
    )
    return parser.parse_args()


def resolve_cli_path(path_argument: str) -> Path:
    path = Path(path_argument)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def resolve_path_from_config_dir(config_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def resolve_inference_model_path(raw_cfg: dict, config_dir: Path, args: argparse.Namespace) -> Path:
    if args.model_file:
        model_path = resolve_cli_path(args.model_file)
    else:
        model_file = raw_cfg.get("inference", {}).get("model_file")
        if model_file:
            model_path = resolve_path_from_config_dir(config_dir, model_file)
        else:
            raise ValueError("Inference requires --model-file or inference.model_file in the YAML config.")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    return model_path


def resolve_inference_output_dir(raw_cfg: dict, config_dir: Path, args: argparse.Namespace, variable: str) -> Path:
    if args.output_dir:
        return resolve_cli_path(args.output_dir)

    configured = raw_cfg.get("inference", {}).get("output_dir")
    if configured:
        return resolve_path_from_config_dir(config_dir, configured)

    return (config_dir / "output" / "inference" / variable).resolve()


def load_inference_dependencies() -> dict[str, object]:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import xarray as xr

    from src.models import predict
    from src.prepare_data import create_test_train_split, prepare_training_dataset

    return {
        "np": np,
        "pd": pd,
        "tf": tf,
        "xr": xr,
        "predict": predict,
        "create_test_train_split": create_test_train_split,
        "prepare_training_dataset": prepare_training_dataset,
    }


def build_experiment_configuration(raw_cfg: dict, config_dir: Path) -> tuple[dict, dict]:
    work_dir = resolve_path_from_config_dir(config_dir, raw_cfg["paths"]["work_dir"])
    data_train_dir_raw = Path(raw_cfg["paths"]["data_train_dir"])
    data_infer_dir_raw = Path(raw_cfg["paths"]["data_infer_dir"])
    variable = raw_cfg["experiment"]["variable"]

    if data_train_dir_raw.is_absolute():
        data_train_dir = data_train_dir_raw
    else:
        data_train_dir = (work_dir / data_train_dir_raw).resolve()

    if data_infer_dir_raw.is_absolute():
        data_infer_dir = data_infer_dir_raw
    else:
        data_infer_dir = resolve_path_from_config_dir(config_dir, raw_cfg["paths"]["data_infer_dir"])

    y_file = raw_cfg["experiment"]["y_filename_template"].format(variable=variable)
    x_file = raw_cfg["experiment"]["x_filename"]

    exp_config = {
        "y": str((data_infer_dir / y_file).resolve()),
        "X": str((data_train_dir / x_file).resolve()),
        "train_start": raw_cfg["experiment"]["dates"]["train"][0],
        "train_end": raw_cfg["experiment"]["dates"]["train"][1],
        "val_start": raw_cfg["experiment"]["dates"]["val"][0],
        "val_end": raw_cfg["experiment"]["dates"]["val"][1],
        "test_start": raw_cfg["experiment"]["dates"]["test"][0],
        "test_end": raw_cfg["experiment"]["dates"]["test"][1],
        "output_var": [variable],
        "downscale_variables": raw_cfg["experiment"]["downscale_variables"],
    }
    resolved_paths = {
        "work_dir": work_dir,
        "data_train_dir": data_train_dir,
        "data_infer_dir": data_infer_dir,
        "predictor_file": Path(exp_config["X"]),
        "target_file": Path(exp_config["y"]),
    }
    return exp_config, resolved_paths


def build_inference_output_paths(
    raw_cfg: dict,
    config_dir: Path,
    args: argparse.Namespace,
    variable: str,
) -> dict[str, Path | str]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    inference_cfg = raw_cfg.get("inference", {})
    run_name = inference_cfg.get("run_name", f"{variable}_test")
    output_dir = resolve_inference_output_dir(raw_cfg, config_dir, args, variable)
    log_root = resolve_path_from_config_dir(config_dir, raw_cfg["training"]["log_root"])
    diagnostic_log_file = (log_root / "diagnostic" / "inference" / run_name / f"inference_{timestamp}.log").resolve()
    prediction_file = (output_dir / inference_cfg.get("prediction_filename", "prediction.nc")).resolve()
    metrics_file = (output_dir / inference_cfg.get("metrics_filename", "evaluation_metrics.csv")).resolve()
    config_backup_path = (output_dir / "config_backup.yaml").resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_log_file.parent.mkdir(parents=True, exist_ok=True)

    return {
        "timestamp": timestamp,
        "run_name": run_name,
        "output_dir": output_dir,
        "diagnostic_log_file": diagnostic_log_file,
        "prediction_file": prediction_file,
        "metrics_file": metrics_file,
        "config_backup_path": config_backup_path,
    }


def summarize_time_axis(data) -> tuple[str, str, int]:
    time_values = data["time"].values
    return str(time_values[0]), str(time_values[-1]), int(data.sizes["time"])


def count_missing_values(data) -> int:
    if hasattr(data, "data_vars"):
        return int(sum(int(data[name].isnull().sum().item()) for name in data.data_vars))
    return int(data.isnull().sum().item())


def log_runtime_context(logger: logging.Logger, config_path: Path | None, tf_module=None) -> None:
    logger.info("Command line: %s", subprocess.list2cmdline(sys.argv))
    logger.info("Current working directory: %s", Path.cwd())
    logger.info("Resolved script directory: %s", SCRIPT_DIR)
    logger.info("Resolved project root: %s", PROJECT_ROOT)
    logger.info("Python executable: %s", sys.executable)
    logger.info("Python version: %s", sys.version.replace("\n", " "))
    logger.info("Hostname: %s", platform.node())
    logger.info("Git commit SHA: %s", get_git_commit_sha(PROJECT_ROOT) or "unavailable")
    logger.info("SLURM job ID: %s", os.getenv("SLURM_JOB_ID", "not set"))
    logger.info("SLURM array task ID: %s", os.getenv("SLURM_ARRAY_TASK_ID", "not set"))
    if config_path is not None:
        logger.info("Configuration path: %s", config_path)
    if tf_module is not None:
        logger.info("TensorFlow version: %s", getattr(tf_module, "__version__", "unknown"))
        try:
            gpu_devices = tf_module.config.list_physical_devices("GPU")
            logger.info("Visible GPU devices: %s", [device.name for device in gpu_devices] or [])
        except Exception:
            logger.warning("Could not list TensorFlow GPU devices.", exc_info=True)


def run_import_smoke_test(logger: logging.Logger) -> int:
    load_inference_dependencies()
    logger.info("Inference smoke test imports completed successfully.")
    return 0


def build_time_encoding(prediction_time_values, variable_name: str) -> dict:
    first_timestamp = str(prediction_time_values[0]).replace("T", " ").split(".")[0]
    return {
        "time": {
            "dtype": "float64",
            "units": f"hours since {first_timestamp}",
            "calendar": "proleptic_gregorian",
        },
        variable_name: {},
    }


def main() -> int:
    args = parse_args()
    bootstrap_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bootstrap_log_file = (PROJECT_ROOT / "output" / "logs" / "bootstrap" / f"inference_bootstrap_{bootstrap_timestamp}.log").resolve()
    logger = setup_logging(bootstrap_log_file, logger_name=LOGGER_NAME)
    logger.info("Bootstrap log file: %s", bootstrap_log_file)

    if args.smoke_test_imports:
        return run_import_smoke_test(logger)

    config_path = resolve_cli_path(args.config_file)
    raw_cfg, _ = load_yaml_config(str(config_path))

    dependencies = load_inference_dependencies()
    np = dependencies["np"]
    pd = dependencies["pd"]
    tf = dependencies["tf"]
    xr = dependencies["xr"]
    predict = dependencies["predict"]
    create_test_train_split = dependencies["create_test_train_split"]
    prepare_training_dataset = dependencies["prepare_training_dataset"]

    tf.random.set_seed(2)

    exp_config, resolved_paths = build_experiment_configuration(raw_cfg, config_path.parent)
    variable = raw_cfg["experiment"]["variable"]
    model_file = resolve_inference_model_path(raw_cfg, config_path.parent, args)
    output_paths = build_inference_output_paths(raw_cfg, config_path.parent, args, variable)
    add_file_handler(logger, output_paths["diagnostic_log_file"])

    shutil.copy2(config_path, output_paths["config_backup_path"])
    logger.info("Configuration backup saved to %s", output_paths["config_backup_path"])
    log_runtime_context(logger, config_path, tf_module=tf)
    logger.info("Inference run name: %s", output_paths["run_name"])
    logger.info("Target variable: %s", variable)
    logger.info("Inference dates: %s", get_inference_dates(raw_cfg))
    logger.info("Predictor variable list and order: %s", raw_cfg["experiment"]["downscale_variables"])
    logger.info("Resolved predictor path: %s", resolved_paths["predictor_file"])
    logger.info("Resolved target path: %s", resolved_paths["target_file"])
    logger.info("Resolved model file: %s", model_file)
    logger.info("Resolved output directory: %s", output_paths["output_dir"])
    logger.info("Resolved prediction file: %s", output_paths["prediction_file"])
    logger.info("Resolved metrics file: %s", output_paths["metrics_file"])
    logger.info("Resolved diagnostic log path: %s", output_paths["diagnostic_log_file"])

    x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(exp_config)
    x_test_first, x_test_last, x_test_count = summarize_time_axis(x_test)
    y_test_first, y_test_last, y_test_count = summarize_time_axis(y_test)
    logger.info("Predictor test timestamps: first=%s last=%s", x_test_first, x_test_last)
    logger.info("Target test timestamps: first=%s last=%s", y_test_first, y_test_last)
    logger.info(
        "Split sample counts: train=%s val=%s test=%s",
        x_train.sizes["time"],
        x_val.sizes["time"],
        x_test.sizes["time"],
    )
    logger.info("Aligned predictor test samples: %s", x_test_count)
    logger.info("Aligned target test samples: %s", y_test_count)
    logger.info("Predictor missing values: %s", count_missing_values(x_test))
    logger.info("Target missing values: %s", count_missing_values(y_test))

    outscale = 1.0
    y_train = y_train * outscale
    y_val = y_val * outscale
    y_test = y_test * outscale

    train_mean = y_train.mean(dim="time")
    train_std = y_train.std(dim="time")

    _x_train_ready, x_test_ready, _x_val_ready, _y_train_ready, y_test_ready, _y_val_ready = prepare_training_dataset(
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
    )
    logger.info("Prepared predictor array shape: %s", x_test_ready.shape)
    logger.info("Prepared target array shape: %s", y_test_ready.shape)
    logger.info("Input channel count: %s", x_test_ready.shape[-1])
    logger.info("Flattened target grid points: %s", y_test_ready.z.size)

    x_test_values = x_test_ready.values if isinstance(x_test_ready, xr.DataArray) else x_test_ready

    logger.info("Loading model from %s", model_file)
    model = tf.keras.models.load_model(str(model_file))
    model.summary(print_fn=logger.info)

    prediction_ds = predict(
        model,
        x_test_values,
        y_test_ready,
        batch_size=int(raw_cfg.get("inference", {}).get("batch_size", 32)),
        key=variable,
        pred_name=variable,
        loss="mse",
        thres=0.5,
    ).unstack()

    train_mean_aligned = train_mean.broadcast_like(prediction_ds[variable])
    train_std_aligned = train_std.broadcast_like(prediction_ds[variable])
    prediction_ds[variable] = (prediction_ds[variable] * train_std_aligned) + train_mean_aligned

    time_encoding = build_time_encoding(prediction_ds.time.values, variable)
    prediction_ds.to_netcdf(output_paths["prediction_file"], encoding=time_encoding)
    logger.info("Prediction saved to %s", output_paths["prediction_file"])

    ground_truth = y_test_ready.unstack()
    prediction_array = prediction_ds[variable]
    correlation = xr.corr(ground_truth, prediction_array, dim="time")
    mae = np.abs(ground_truth - prediction_array).mean(dim="time")
    rmse = np.sqrt(((ground_truth - prediction_array) ** 2).mean(dim="time"))
    std_dev_truth = ground_truth.std(dim="time")
    std_dev_pred = prediction_array.std(dim="time")

    mean_dims = [dim for dim in ("x", "y") if dim in correlation.dims]
    if not mean_dims:
        raise ValueError("Could not determine spatial dimensions for inference metrics.")

    metrics_df = pd.DataFrame(
        {
            "metric": ["correlation", "mae", "rmse", "std_dev_truth", "std_dev_pred"],
            "value": [
                correlation.mean(mean_dims).item(),
                mae.mean(mean_dims).item(),
                rmse.mean(mean_dims).item(),
                std_dev_truth.mean(mean_dims).item(),
                std_dev_pred.mean(mean_dims).item(),
            ],
        }
    )
    metrics_df.to_csv(output_paths["metrics_file"], index=False)
    logger.info("Metrics saved to %s", output_paths["metrics_file"])
    logger.info("Inference completed successfully.")
    return 0


if __name__ == "__main__":
    logger = logging.getLogger(LOGGER_NAME)
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Inference failed")
        raise
