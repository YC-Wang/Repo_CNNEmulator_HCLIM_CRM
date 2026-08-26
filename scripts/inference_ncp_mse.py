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
    ensure_can_write_inference_outputs,
    get_config_warnings,
    get_git_commit_sha,
    get_inference_run,
    get_experiment_dir,
    load_yaml_config,
    resolve_config_paths,
    resolve_path,
    summarize_time_values,
    write_netcdf_with_time_validation,
    write_yaml,
)


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
        help="Path to a trained model file. Overrides the canonical experiment model.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Select a named inference run when inference.runs is configured.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing outputs for the selected inference run.",
    )
    parser.add_argument(
        "--smoke-test-imports",
        action="store_true",
        help="Validate script/module imports without loading data or running inference.",
    )
    return parser.parse_args()


def resolve_cli_config_path(config_argument: str) -> Path:
    config_path = Path(config_argument)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    return config_path


def load_inference_dependencies() -> dict[str, object]:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import xarray as xr

    from src.prepare_data import create_test_train_split, format_features, prepare_training_dataset

    return {
        "np": np,
        "pd": pd,
        "tf": tf,
        "xr": xr,
        "create_test_train_split": create_test_train_split,
        "format_features": format_features,
        "prepare_training_dataset": prepare_training_dataset,
    }


def count_missing_values(data) -> int:
    if hasattr(data, "data_vars"):
        return int(sum(int(data[name].isnull().sum().item()) for name in data.data_vars))
    return int(data.isnull().sum().item())


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
    load_inference_dependencies()
    logger.info("Inference smoke test imports completed successfully.")
    return 0


def bootstrap_log_for_smoke_test(timestamp: str) -> Path:
    path = (PROJECT_ROOT / "outputs" / "_smoke_test" / "logs" / "bootstrap" / f"inference_smoke_test_{timestamp}.log").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_inference_model_path(
    args: argparse.Namespace,
    resolved_cfg: dict,
    config_path: Path,
    output_paths: dict[str, Path],
) -> Path:
    if args.model_file:
        model_path = resolve_path(Path.cwd(), args.model_file)
    else:
        configured_model = resolved_cfg.get("inference", {}).get("model_file")
        if configured_model:
            model_path = resolve_path(config_path.parent, configured_model)
        else:
            model_path = output_paths["model_file"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    return model_path


def load_saved_normalization(xr_module, output_paths: dict[str, Path]):
    if not output_paths["normalization_mean_file"].exists() or not output_paths["normalization_std_file"].exists():
        return None, None
    return (
        xr_module.open_dataarray(output_paths["normalization_mean_file"]).load(),
        xr_module.open_dataarray(output_paths["normalization_std_file"]).load(),
    )


def transpose_feature_array(feature_array):
    for dims in (("time", "y", "x", "feature"), ("time", "latitude", "longitude", "feature"), ("time", "lat", "lon", "feature")):
        if all(dim in feature_array.dims for dim in dims):
            return feature_array.transpose(*dims)
    raise ValueError(f"Unsupported predictor dimensions for feature formatting: {feature_array.dims}")


def stack_spatial_template(data_array):
    for dims in (("y", "x"), ("latitude", "longitude"), ("lat", "lon")):
        if all(dim in data_array.dims for dim in dims):
            return data_array.stack(z=list(dims)).dropna("z")
    raise ValueError(f"Unsupported target dimensions for spatial stacking: {data_array.dims}")


def load_predictor_only_slices(run: dict, config: dict, xr_module, pd_module):
    predictor_ds = xr_module.open_dataset(run["predictor_file"], chunks={"time": 3000})[config["experiment"]["downscale_variables"]]
    predictor_offset_hours = int(config["experiment"]["predictor_time_offset_hours"])
    predictor_ds["time"] = pd_module.to_datetime(predictor_ds.time.dt.strftime("%Y-%m-%d %H:00:00")) + pd_module.Timedelta(hours=predictor_offset_hours)

    if not predictor_ds.time.to_index().is_unique:
        predictor_ds = predictor_ds.sel(time=~predictor_ds.time.to_index().duplicated())

    train_dates = config["experiment"]["dates"]["train"]
    inference_dates = run["dates"]
    x_train = predictor_ds.sel(time=slice(train_dates[0], train_dates[1])).load()
    x_infer = predictor_ds.sel(time=slice(inference_dates[0], inference_dates[1])).load()
    return x_train, x_infer


def prepare_predictors_only(x_train, x_infer, format_features):
    means = x_train.mean()
    stds = x_train.std()
    x_infer_norm = (x_infer - means) / stds
    return transpose_feature_array(format_features(x_infer_norm))


def build_prediction_dataset(predictions, template_stacked, variable_name: str, xr_module):
    prediction_values = predictions.squeeze()
    prediction_array = xr_module.DataArray(
        prediction_values,
        coords=template_stacked.coords,
        dims=template_stacked.dims,
        name=variable_name,
    )
    return prediction_array.to_dataset(name=variable_name).unstack()


def maybe_clip_prediction(prediction_ds, variable_name: str, config: dict) -> None:
    if variable_name == "pr" and config["inference"]["clip_negative_rainfall"]:
        prediction_ds[variable_name] = prediction_ds[variable_name].clip(min=0.0)


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
        logger = setup_logging(build_failed_bootstrap_log_path(config_path, "inference", timestamp), logger_name=LOGGER_NAME)
        logger.exception("Failed to parse configuration.")
        raise

    resolved_cfg = resolve_config_paths(raw_cfg, config_path)
    selected_run = get_inference_run(resolved_cfg, args.run_name)
    output_paths = build_output_paths(resolved_cfg, run_name=selected_run["name"])
    logger = setup_logging(output_paths["bootstrap_log_dir"] / f"inference_{timestamp}.log", logger_name=LOGGER_NAME)
    add_file_handler(logger, output_paths["inference_log_file"])

    for warning_message in get_config_warnings(raw_cfg, resolved_cfg, config_path):
        logger.warning(warning_message)

    overwrite_existing = args.overwrite or bool(resolved_cfg["inference"].get("overwrite_existing", False))
    ensure_can_write_inference_outputs(
        output_paths,
        overwrite_existing=overwrite_existing,
        calculate_test_metrics=bool(resolved_cfg["inference"]["calculate_test_metrics"]),
    )

    shutil.copy2(config_path, output_paths["input_config_backup"])
    write_yaml(output_paths["resolved_config_file"], resolved_cfg)

    dependencies = load_inference_dependencies()
    np = dependencies["np"]
    pd = dependencies["pd"]
    tf = dependencies["tf"]
    xr = dependencies["xr"]
    create_test_train_split = dependencies["create_test_train_split"]
    format_features = dependencies["format_features"]
    prepare_training_dataset = dependencies["prepare_training_dataset"]

    tf.random.set_seed(2)

    model_file = resolve_inference_model_path(args, resolved_cfg, config_path, output_paths)
    variable = resolved_cfg["experiment"]["variable"]
    calculate_test_metrics = bool(resolved_cfg["inference"]["calculate_test_metrics"])
    target_file = selected_run.get("target_file")

    log_runtime_context(logger, config_path, tf_module=tf)
    logger.info("Experiment ID: %s", resolved_cfg["metadata"]["experiment_id"])
    logger.info("Experiment directory: %s", get_experiment_dir(resolved_cfg))
    logger.info("Inference run name: %s", selected_run["name"])
    logger.info("Target variable: %s", variable)
    logger.info("Inference dates: %s", selected_run["dates"])
    logger.info("Predictor variable list and order: %s", resolved_cfg["experiment"]["downscale_variables"])
    logger.info("Resolved predictor path: %s", selected_run["predictor_file"])
    logger.info("Resolved target path: %s", target_file or "not configured")
    logger.info("Resolved model file: %s", model_file)
    logger.info("Resolved config input backup: %s", output_paths["input_config_backup"])
    logger.info("Resolved config output backup: %s", output_paths["resolved_config_file"])
    logger.info("Resolved bootstrap log directory: %s", output_paths["bootstrap_log_dir"])
    logger.info("Resolved inference diagnostic log: %s", output_paths["inference_log_file"])
    logger.info("Resolved prediction file: %s", output_paths["prediction_file"])
    logger.info("Resolved metrics file: %s", output_paths["metrics_file"])

    outscale = float(resolved_cfg["experiment"]["target_scale"])
    train_mean, train_std = load_saved_normalization(xr, output_paths)
    ground_truth = None

    if target_file:
        exp_config = build_legacy_split_config(
            resolved_cfg,
            predictor_file=selected_run["predictor_file"],
            target_file=target_file,
            dates=selected_run["dates"],
        )
        x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(exp_config)
        x_test_summary = summarize_time_axis(x_test)
        y_test_summary = summarize_time_axis(y_test)
        logger.info("Predictor timestamps: first=%s last=%s timestep=%s count=%s", x_test_summary["first"], x_test_summary["last"], x_test_summary["timestep"], x_test_summary["count"])
        logger.info("Target timestamps: first=%s last=%s timestep=%s count=%s", y_test_summary["first"], y_test_summary["last"], y_test_summary["timestep"], y_test_summary["count"])
        logger.info(
            "Split sample counts: train=%s val=%s test=%s",
            x_train.sizes["time"],
            x_val.sizes["time"],
            x_test.sizes["time"],
        )
        logger.info("Predictor missing values: %s", count_missing_values(x_test))
        logger.info("Target missing values: %s", count_missing_values(y_test))

        y_train = y_train * outscale
        y_val = y_val * outscale
        y_test = y_test * outscale

        if train_mean is None or train_std is None:
            logger.warning("Saved normalization files were not found. Falling back to training-period target statistics from the current target data.")
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
        template_stacked = y_test_ready
        ground_truth = y_test_ready.unstack() if calculate_test_metrics else None
    else:
        if calculate_test_metrics:
            raise ValueError("inference.calculate_test_metrics is true, but no target_file is configured for the selected run.")
        if train_mean is None or train_std is None:
            raise FileNotFoundError(
                "Saved normalization files are required for target-free inference. "
                f"Expected {output_paths['normalization_mean_file']} and {output_paths['normalization_std_file']}."
            )

        x_train, x_test = load_predictor_only_slices(selected_run, resolved_cfg, xr, pd)
        x_test_summary = summarize_time_axis(x_test)
        logger.info("Predictor timestamps: first=%s last=%s timestep=%s count=%s", x_test_summary["first"], x_test_summary["last"], x_test_summary["timestep"], x_test_summary["count"])
        logger.info("Predictor missing values: %s", count_missing_values(x_test))
        x_test_ready = prepare_predictors_only(x_train, x_test, format_features)
        template_stacked = stack_spatial_template(train_mean)
        template_stacked = template_stacked.expand_dims(time=x_test_ready["time"])
        template_stacked = template_stacked.transpose("time", "z")

    logger.info("Prepared predictor array shape: %s", x_test_ready.shape)
    logger.info("Input channel count: %s", x_test_ready.shape[-1])
    logger.info("Flattened target grid points: %s", template_stacked.z.size)

    x_test_values = x_test_ready.values if isinstance(x_test_ready, xr.DataArray) else x_test_ready

    logger.info("Loading model from %s", model_file)
    model = tf.keras.models.load_model(str(model_file))
    model.summary(print_fn=logger.info)

    predictions = model.predict(
        x_test_values,
        verbose=1,
        batch_size=int(resolved_cfg["inference"]["batch_size"]),
    )
    prediction_ds = build_prediction_dataset(predictions, template_stacked, variable, xr)

    train_mean_aligned = train_mean.broadcast_like(prediction_ds[variable])
    train_std_aligned = train_std.broadcast_like(prediction_ds[variable])
    prediction_ds[variable] = (prediction_ds[variable] * train_std_aligned) + train_mean_aligned
    maybe_clip_prediction(prediction_ds, variable, resolved_cfg)

    time_summary = write_netcdf_with_time_validation(prediction_ds, output_paths["prediction_file"])
    logger.info("Prediction saved to %s", output_paths["prediction_file"])
    logger.info("Prediction timestamps verified after NetCDF round trip: first=%s last=%s timestep=%s count=%s", time_summary["first"], time_summary["last"], time_summary["timestep"], time_summary["count"])

    if calculate_test_metrics:
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
    else:
        logger.info("Inference metrics disabled; skipping metrics-file generation.")

    logger.info("Inference completed successfully.")
    return 0


if __name__ == "__main__":
    logger = logging.getLogger(LOGGER_NAME)
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Inference failed")
        raise
