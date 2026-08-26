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
from src.pipeline_utils import (
    build_failed_bootstrap_log_path,
    build_legacy_split_config,
    build_output_paths,
    ensure_can_write_training_outputs,
    get_config_warnings,
    get_git_commit_sha,
    get_experiment_dir,
    load_yaml_config,
    resolve_config_paths,
    summarize_time_values,
    write_yaml,
)


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
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing canonical best_model.h5 for this experiment.",
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


def summarize_time_axis(data) -> dict[str, str | int]:
    return summarize_time_values(data["time"].values)


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


def bootstrap_log_for_smoke_test(timestamp: str) -> Path:
    path = (PROJECT_ROOT / "outputs" / "_smoke_test" / "logs" / "bootstrap" / f"training_smoke_test_{timestamp}.log").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.smoke_test_imports:
        logger = setup_logging(bootstrap_log_for_smoke_test(timestamp), logger_name=LOGGER_NAME)
        return run_import_smoke_test(logger)

    config_path = resolve_cli_config_path(args.config_file)
    try:
        raw_cfg, _ = load_yaml_config(str(config_path))
    except Exception:
        logger = setup_logging(build_failed_bootstrap_log_path(config_path, "training", timestamp), logger_name=LOGGER_NAME)
        logger.exception("Failed to parse configuration.")
        raise

    resolved_cfg = resolve_config_paths(raw_cfg, config_path)
    output_paths = build_output_paths(resolved_cfg)
    logger = setup_logging(output_paths["bootstrap_log_dir"] / f"training_{timestamp}.log", logger_name=LOGGER_NAME)
    add_file_handler(logger, output_paths["training_log_file"])

    for warning_message in get_config_warnings(raw_cfg, resolved_cfg, config_path):
        logger.warning(warning_message)

    overwrite_existing = args.overwrite or bool(resolved_cfg["training"].get("overwrite_existing", False))
    ensure_can_write_training_outputs(output_paths, overwrite_existing)

    shutil.copy2(config_path, output_paths["input_config_backup"])
    write_yaml(output_paths["resolved_config_file"], resolved_cfg)

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

    log_runtime_context(logger, config_path, tf_module=tf)
    logger.info("Experiment ID: %s", resolved_cfg["metadata"]["experiment_id"])
    logger.info("Experiment directory: %s", get_experiment_dir(resolved_cfg))
    logger.info("Target variable: %s", resolved_cfg["experiment"]["variable"])
    logger.info("Predictor variable list and order: %s", resolved_cfg["experiment"]["downscale_variables"])
    logger.info("Training period: %s", resolved_cfg["experiment"]["dates"]["train"])
    logger.info("Validation period: %s", resolved_cfg["experiment"]["dates"]["val"])
    logger.info("Test period: %s", resolved_cfg["experiment"]["dates"]["test"])
    logger.info("Resolved predictor path: %s", resolved_cfg["paths"]["predictor_file"])
    logger.info("Resolved target path: %s", resolved_cfg["paths"]["target_file"])
    logger.info("Resolved config input backup: %s", output_paths["input_config_backup"])
    logger.info("Resolved config output backup: %s", output_paths["resolved_config_file"])
    logger.info("Resolved TensorBoard path: %s", output_paths["tensorboard_dir"])
    logger.info("Resolved training diagnostic log: %s", output_paths["training_log_file"])
    logger.info("Resolved bootstrap log directory: %s", output_paths["bootstrap_log_dir"])
    logger.info("Resolved normalization mean file: %s", output_paths["normalization_mean_file"])
    logger.info("Resolved normalization std file: %s", output_paths["normalization_std_file"])
    logger.info("Resolved model file: %s", output_paths["model_file"])

    exp_config = build_legacy_split_config(resolved_cfg, dates=resolved_cfg["experiment"]["dates"]["test"])
    x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(exp_config)

    x_train_summary = summarize_time_axis(x_train)
    y_train_summary = summarize_time_axis(y_train)
    logger.info("Predictor timestamps: first=%s last=%s timestep=%s count=%s", x_train_summary["first"], x_train_summary["last"], x_train_summary["timestep"], x_train_summary["count"])
    logger.info("Target timestamps: first=%s last=%s timestep=%s count=%s", y_train_summary["first"], y_train_summary["last"], y_train_summary["timestep"], y_train_summary["count"])
    logger.info(
        "Split sample counts: train=%s val=%s test=%s",
        x_train.sizes["time"],
        x_val.sizes["time"],
        x_test.sizes["time"],
    )
    logger.info("Predictor missing values: %s", count_missing_values(x_train))
    logger.info("Target missing values: %s", count_missing_values(y_train))

    outscale = float(resolved_cfg["experiment"]["target_scale"])
    y_train = y_train * outscale
    y_val = y_val * outscale
    y_test = y_test * outscale

    train_mean = y_train.mean(dim="time")
    train_std = y_train.std(dim="time")
    train_mean.to_netcdf(output_paths["normalization_mean_file"])
    train_std.to_netcdf(output_paths["normalization_std_file"])
    logger.info("Saved training mean to %s", output_paths["normalization_mean_file"])
    logger.info("Saved training std to %s", output_paths["normalization_std_file"])
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

    optimizer = legacy.Adam(lr=resolved_cfg["training"]["learning_rate"])
    simple_cnn = simple_conv(
        layer_filters=resolved_cfg["model"]["layer_filters"],
        bn=resolved_cfg["model"]["use_bn"],
        padding=resolved_cfg["model"]["padding"],
        kernel_size=(resolved_cfg["model"]["kernel_size"], resolved_cfg["model"]["kernel_size"]),
        pooling=resolved_cfg["model"]["use_pooling"],
        dense_layers=[resolved_cfg["model"]["hidden_layer_dense"], output_shape],
        dense_activation=resolved_cfg["model"]["dense_activation"],
        input_shape=input_shape,
        dropout=resolved_cfg["model"]["dropout"],
        activation=resolved_cfg["model"]["cnn_activation"],
    )
    simple_linear = simple_dense(
        dense_layers=[resolved_cfg["model"]["hidden_layer_dense"], output_shape],
        dense_activation=resolved_cfg["model"]["dense_activation"],
        input_shape=input_shape,
        dropout=resolved_cfg["model"]["dropout"],
    )

    x_train = x_train.values if isinstance(x_train, xr.DataArray) else x_train
    y_train = y_train.values if isinstance(y_train, xr.DataArray) else y_train
    y_train = y_train.to_array().values if isinstance(y_train, xr.Dataset) else y_train
    x_val_values = x_val.values if isinstance(x_val, xr.DataArray) else x_val
    y_val_values = y_val.values if isinstance(y_val, xr.DataArray) else y_val
    y_val_values = y_val_values.to_array().values if isinstance(y_val_values, xr.Dataset) else y_val_values

    model_type = resolved_cfg["training"]["model_type"]
    if model_type == "linear":
        active_model = simple_linear
    elif model_type == "cnn":
        active_model = simple_cnn
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected 'linear' or 'cnn'.")

    logger.info("Model type: %s", model_type)
    logger.info("Loss: %s", resolved_cfg["training"]["loss"])
    logger.info("Metric: %s", resolved_cfg["training"]["metrics"])
    logger.info("Learning rate: %s", resolved_cfg["training"]["learning_rate"])
    logger.info("Batch size: %s", resolved_cfg["training"]["batch_size"])
    logger.info("Maximum epochs: %s", resolved_cfg["training"]["epochs"])
    logger.info("Model checkpoint path: %s", output_paths["model_file"])

    history, _trained_model = train_model(
        active_model,
        x_train,
        y_train,
        x_val=x_val_values,
        y_val=y_val_values,
        loss=resolved_cfg["training"]["loss"],
        epochs=resolved_cfg["training"]["epochs"],
        batch_size=resolved_cfg["training"]["batch_size"],
        optimizer=optimizer,
        model_weights_name=str(output_paths["model_file"]),
        logdir=str(output_paths["tensorboard_dir"]),
        metrics=resolved_cfg["training"]["metrics"],
    )

    logger.info("Training completed: yes")
    logger.info("Epochs completed: %s", len(history.history.get("loss", [])))
    if history.history.get("loss"):
        logger.info("Final training loss: %s", history.history["loss"][-1])
    if history.history.get("val_loss"):
        logger.info("Final validation loss: %s", history.history["val_loss"][-1])
    logger.info("Best model path: %s", output_paths["model_file"])
    return 0


if __name__ == "__main__":
    logger = logging.getLogger(LOGGER_NAME)
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Training failed")
        raise
