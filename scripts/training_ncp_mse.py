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
from src.pipeline_utils import get_git_commit_sha, load_yaml_config


LOGGER_NAME = "training_ncp_mse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML Experiment with YAML config")
    parser.add_argument(
        "config_file",
        type=str,
        nargs="?",
        default="config.yaml",
        help="Path to the yaml configuration file",
    )
    parser.add_argument(
        "--smoke-test-imports",
        action="store_true",
        help="Validate script/module imports without loading data or starting training.",
    )
    return parser.parse_args()


def resolve_cli_config_path(config_argument: str) -> Path:
    config_path = Path(config_argument)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    return config_path


def resolve_path_from_config_dir(config_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def load_training_dependencies() -> dict[str, object]:
    import numpy as np
    import tensorflow as tf
    import xarray as xr
    from tensorflow.keras.optimizers import legacy

    from src.models import simple_conv, simple_dense, train_model
    from src.prepare_data import create_test_train_split, prepare_training_dataset

    return {
        "np": np,
        "tf": tf,
        "xr": xr,
        "legacy": legacy,
        "simple_conv": simple_conv,
        "simple_dense": simple_dense,
        "train_model": train_model,
        "create_test_train_split": create_test_train_split,
        "prepare_training_dataset": prepare_training_dataset,
    }


def count_missing_values(data) -> int:
    if hasattr(data, "data_vars"):
        return int(sum(int(data[name].isnull().sum().item()) for name in data.data_vars))
    return int(data.isnull().sum().item())


def count_zero_std_points(data, np_module) -> int:
    if hasattr(data, "data_vars"):
        total = 0
        for name in data.data_vars:
            std = data[name].std(dim="time")
            total += int(np_module.isclose(std.fillna(-9999).values, 0.0).sum())
        return total
    std = data.std(dim="time")
    return int(np_module.isclose(std.fillna(-9999).values, 0.0).sum())


def summarize_time_axis(data) -> tuple[str, str, int]:
    time_values = data["time"].values
    return str(time_values[0]), str(time_values[-1]), int(data.sizes["time"])


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


def build_output_paths(raw_cfg: dict, config_dir: Path, variable: str) -> dict[str, Path | str]:
    training_cfg = raw_cfg["training"]
    model_cfg = raw_cfg["model"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    model_type = training_cfg["model_type"]
    tag = training_cfg["experiment_tag"]
    exp_id = f"{timestamp}_{model_type}_{variable}_{tag}"
    log_root = resolve_path_from_config_dir(config_dir, training_cfg["log_root"])
    model_root = resolve_path_from_config_dir(config_dir, training_cfg["model_root"])
    tensorboard_log_dir = (log_root / model_type / exp_id).resolve()
    diagnostic_log_file = (log_root / "diagnostic" / model_type / exp_id / "training.log").resolve()
    model_weights_path = (model_root / f"{exp_id}.h5").resolve()
    config_backup_path = (tensorboard_log_dir / "config_backup.yaml").resolve()

    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_log_file.parent.mkdir(parents=True, exist_ok=True)
    model_weights_path.parent.mkdir(parents=True, exist_ok=True)

    return {
        "timestamp": timestamp,
        "exp_id": exp_id,
        "model_type": model_type,
        "tensorboard_log_dir": tensorboard_log_dir,
        "diagnostic_log_file": diagnostic_log_file,
        "model_weights_path": model_weights_path,
        "config_backup_path": config_backup_path,
        "dense_activation": model_cfg["dense_activation"],
        "cnn_activation": model_cfg["cnn_activation"],
    }


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
    load_training_dependencies()
    logger.info("Smoke test imports completed successfully.")
    return 0


def main() -> int:
    args = parse_args()
    bootstrap_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bootstrap_log_file = (PROJECT_ROOT / "output" / "logs" / "bootstrap" / f"training_bootstrap_{bootstrap_timestamp}.log").resolve()
    logger = setup_logging(bootstrap_log_file, logger_name=LOGGER_NAME)
    logger.info("Bootstrap log file: %s", bootstrap_log_file)

    if args.smoke_test_imports:
        return run_import_smoke_test(logger)

    config_path = resolve_cli_config_path(args.config_file)
    raw_cfg, _ = load_yaml_config(str(config_path))

    dependencies = load_training_dependencies()
    np = dependencies["np"]
    tf = dependencies["tf"]
    xr = dependencies["xr"]
    legacy = dependencies["legacy"]
    simple_conv = dependencies["simple_conv"]
    simple_dense = dependencies["simple_dense"]
    train_model = dependencies["train_model"]
    create_test_train_split = dependencies["create_test_train_split"]
    prepare_training_dataset = dependencies["prepare_training_dataset"]

    tf.random.set_seed(2)

    exp_config, resolved_paths = build_experiment_configuration(raw_cfg, config_path.parent)
    variable = raw_cfg["experiment"]["variable"]
    output_paths = build_output_paths(raw_cfg, config_path.parent, variable)
    add_file_handler(logger, output_paths["diagnostic_log_file"])

    shutil.copy2(config_path, output_paths["config_backup_path"])
    logger.info("Configuration backup saved to %s", output_paths["config_backup_path"])
    log_runtime_context(logger, config_path, tf_module=tf)
    logger.info("Experiment ID: %s", output_paths["exp_id"])
    logger.info("Experiment tag: %s", raw_cfg["training"]["experiment_tag"])
    logger.info("Target variable: %s", variable)
    logger.info("Predictor variable list and order: %s", raw_cfg["experiment"]["downscale_variables"])
    logger.info("Training period: %s", raw_cfg["experiment"]["dates"]["train"])
    logger.info("Validation period: %s", raw_cfg["experiment"]["dates"]["val"])
    logger.info("Test period: %s", raw_cfg["experiment"]["dates"]["test"])
    logger.info("Resolved predictor path: %s", resolved_paths["predictor_file"])
    logger.info("Resolved target path: %s", resolved_paths["target_file"])
    logger.info("Resolved model-output path: %s", output_paths["model_weights_path"])
    logger.info("Resolved TensorBoard log path: %s", output_paths["tensorboard_log_dir"])
    logger.info("Resolved diagnostic log path: %s", output_paths["diagnostic_log_file"])

    x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(exp_config)
    x_train_first, x_train_last, aligned_train = summarize_time_axis(x_train)
    y_train_first, y_train_last, _ = summarize_time_axis(y_train)
    logger.info("Predictor timestamps: first=%s last=%s", x_train_first, x_train_last)
    logger.info("Target timestamps: first=%s last=%s", y_train_first, y_train_last)
    logger.info("Aligned training samples: %s", aligned_train)
    logger.info(
        "Split sample counts: train=%s val=%s test=%s",
        x_train.sizes["time"],
        x_val.sizes["time"],
        x_test.sizes["time"],
    )
    logger.info("Predictor missing values: %s", count_missing_values(x_train))
    logger.info("Target missing values: %s", count_missing_values(y_train))

    outscale = 1.0
    y_train = y_train * outscale
    y_val = y_val * outscale
    y_test = y_test * outscale

    train_mean = y_train.mean(dim="time")
    train_std = y_train.std(dim="time")
    logger.info("Target zero-standard-deviation grid points: %s", count_zero_std_points(y_train, np))
    logger.info("Predictor zero-standard-deviation grid points: %s", count_zero_std_points(x_train, np))

    y_train = (y_train - train_mean) / train_std
    y_val = (y_val - train_mean) / train_std
    y_test = (y_test - train_mean) / train_std

    x_train, x_test, x_val, y_train, y_test, y_val = prepare_training_dataset(
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
    )
    logger.info("Predictor array shape after preparation: %s", x_train.shape)
    logger.info("Target array shape after preparation: %s", y_train.shape)

    input_shape = x_train.shape[1:]
    output_shape = y_train.z.size
    logger.info("Input channel count: %s", input_shape[-1])
    logger.info("Flattened target grid points: %s", output_shape)

    optimizer = legacy.Adam(lr=raw_cfg["training"]["learning_rate"])
    simple_cnn = simple_conv(
        layer_filters=raw_cfg["model"]["layer_filters"],
        bn=raw_cfg["model"]["use_bn"],
        padding=raw_cfg["model"]["padding"],
        kernel_size=(raw_cfg["model"]["kernel_size"], raw_cfg["model"]["kernel_size"]),
        pooling=raw_cfg["model"]["use_pooling"],
        dense_layers=[raw_cfg["model"]["hidden_layer_dense"], output_shape],
        dense_activation=raw_cfg["model"]["dense_activation"],
        input_shape=input_shape,
        dropout=raw_cfg["model"]["dropout"],
        activation=raw_cfg["model"]["cnn_activation"],
    )
    simple_linear = simple_dense(
        dense_layers=[raw_cfg["model"]["hidden_layer_dense"], output_shape],
        dense_activation=raw_cfg["model"]["dense_activation"],
        input_shape=input_shape,
        dropout=raw_cfg["model"]["dropout"],
    )

    x_train = x_train.values if isinstance(x_train, xr.DataArray) else x_train
    y_train = y_train.values if isinstance(y_train, xr.DataArray) else y_train
    y_train = y_train.to_array().values if isinstance(y_train, xr.Dataset) else y_train
    x_val_values = x_val.values if isinstance(x_val, xr.DataArray) else x_val
    y_val_values = y_val.values if isinstance(y_val, xr.DataArray) else y_val
    y_val_values = y_val_values.to_array().values if isinstance(y_val_values, xr.Dataset) else y_val_values

    model_type = raw_cfg["training"]["model_type"]
    if model_type == "linear":
        active_model = simple_linear
    elif model_type == "cnn":
        active_model = simple_cnn
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected 'linear' or 'cnn'.")

    logger.info("Model type: %s", model_type)
    logger.info("Loss: %s", raw_cfg["training"]["loss"])
    logger.info("Metric: %s", raw_cfg["training"]["metrics"])
    logger.info("Learning rate: %s", raw_cfg["training"]["learning_rate"])
    logger.info("Batch size: %s", raw_cfg["training"]["batch_size"])
    logger.info("Maximum epochs: %s", raw_cfg["training"]["epochs"])
    logger.info("Model checkpoint path: %s", output_paths["model_weights_path"])

    history, trained_model = train_model(
        active_model,
        x_train,
        y_train,
        x_val=x_val_values,
        y_val=y_val_values,
        loss=raw_cfg["training"]["loss"],
        epochs=raw_cfg["training"]["epochs"],
        batch_size=raw_cfg["training"]["batch_size"],
        optimizer=optimizer,
        model_weights_name=str(output_paths["model_weights_path"]),
        logdir=str(output_paths["tensorboard_log_dir"]),
        metrics=raw_cfg["training"]["metrics"],
    )

    logger.info("Training completed: yes")
    logger.info("Epochs completed: %s", len(history.history.get("loss", [])))
    if history.history.get("loss"):
        logger.info("Final training loss: %s", history.history["loss"][-1])
    if history.history.get("val_loss"):
        logger.info("Final validation loss: %s", history.history["val_loss"][-1])
    logger.info("Best model path: %s", output_paths["model_weights_path"])
    return 0


if __name__ == "__main__":
    logger = logging.getLogger(LOGGER_NAME)
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Training failed")
        raise

