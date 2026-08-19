#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_utils import (
    get_experiment_dir,
    get_inference_runs,
    load_yaml_config,
    read_resolved_config,
    resolve_config_paths,
)
from prepare_data import (
    align_predictors_and_target,
    get_spatial_dims,
    load_predictor_normalization,
    load_target_normalization,
    normalize_with_training_stats,
    reconstruct_grid,
    save_target_normalization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run precipitation CNN inference.")
    parser.add_argument("config_file", help="Path to the YAML configuration file.")
    return parser.parse_args()


def _apply_unit_choice(
    predictions_scaled: xr.DataArray,
    target_scale: float,
    saved_output_units: str,
) -> xr.DataArray:
    if saved_output_units == "scaled":
        return predictions_scaled
    if saved_output_units == "original":
        return predictions_scaled / target_scale
    raise ValueError("inference.saved_output_units must be either 'scaled' or 'original'.")


def _compute_metrics(
    truth: xr.DataArray,
    prediction: xr.DataArray,
) -> pd.DataFrame:
    spatial_dims = get_spatial_dims(truth)
    correlation = xr.corr(truth, prediction, dim="time").mean(spatial_dims).item()
    mae = np.abs(truth - prediction).mean(dim="time").mean(spatial_dims).item()
    rmse = np.sqrt(((truth - prediction) ** 2).mean(dim="time")).mean(spatial_dims).item()
    std_truth = truth.std(dim="time").mean(spatial_dims).item()
    std_pred = prediction.std(dim="time").mean(spatial_dims).item()
    return pd.DataFrame(
        {
            "metric": ["correlation", "mae", "rmse", "std_dev_truth", "std_dev_pred"],
            "value": [correlation, mae, rmse, std_truth, std_pred],
        }
    )


def _verify_saved_time(output_file: Path, in_memory: xr.DataArray, evaluation_target: xr.DataArray | None) -> None:
    reopened = xr.open_dataset(output_file).load()[in_memory.name]
    if not np.array_equal(reopened["time"].values, in_memory["time"].values):
        raise ValueError("Saved prediction timestamps do not match in-memory timestamps.")
    if evaluation_target is not None and not np.array_equal(
        reopened["time"].values,
        evaluation_target["time"].values,
    ):
        raise ValueError("Saved prediction timestamps do not match evaluation target timestamps.")


def _run_single_inference(
    experiment_dir: Path,
    resolved_config: dict,
    predictor_mean: xr.Dataset,
    predictor_std_safe: xr.Dataset,
    predictor_order: list[str],
    target_mean: xr.DataArray,
    target_std_safe: xr.DataArray,
    target_valid_mask: xr.DataArray,
    run_config: dict,
    output_file: Path,
    metrics_file: Path | None,
) -> None:
    predictors, target = align_predictors_and_target(
        predictor_file=Path(run_config["predictor_file"]),
        target_file=Path(run_config["target_file"]),
        predictor_variables=predictor_order,
        target_variable=resolved_config["experiment"]["variable"],
        predictor_time_offset_hours=resolved_config["experiment"]["predictor_time_offset_hours"],
    )
    predictors = predictors.sel(time=slice(run_config["dates"][0], run_config["dates"][1])).load()
    target = target.sel(time=slice(run_config["dates"][0], run_config["dates"][1])).load()
    if predictors.sizes["time"] == 0:
        raise ValueError(f"Inference period '{run_config['name']}' is empty after alignment.")
    if predictors.sizes["time"] != target.sizes["time"]:
        raise ValueError(f"Inference predictor and target counts differ after alignment for run '{run_config['name']}'.")

    for variable_name in predictor_order:
        if variable_name not in predictors.data_vars:
            raise KeyError(f"Inference predictors are missing required variable '{variable_name}'.")
    predictors = predictors[predictor_order]

    normalized_predictors = normalize_with_training_stats(
        predictors,
        predictor_mean[predictor_order],
        predictor_std_safe[predictor_order],
    )
    spatial_dims = get_spatial_dims(target_valid_mask)
    normalized_predictors = xr.concat(
        [normalized_predictors[name] for name in predictor_order],
        dim="feature",
    ).assign_coords(feature=("feature", predictor_order)).transpose("time", *spatial_dims, "feature")

    if not np.isfinite(normalized_predictors.values).all():
        raise ValueError("Non-finite inference predictors were produced after normalization.")

    model = tf.keras.models.load_model(experiment_dir / "model.h5")
    expected_input_shape = model.input_shape[1:]
    if tuple(normalized_predictors.shape[1:]) != tuple(expected_input_shape):
        raise ValueError(
            f"Model input shape mismatch. Expected {expected_input_shape}, got {tuple(normalized_predictors.shape[1:])}."
        )

    predictions_normalized = model.predict(
        normalized_predictors.values,
        batch_size=int(resolved_config["inference"]["batch_size"]),
        verbose=1,
    )
    if predictions_normalized.shape[-1] != int(target_valid_mask.sum().item()):
        raise ValueError("Model output length does not match the saved valid target points.")

    predictions_scaled_flat = (
        predictions_normalized * target_std_safe.stack(z=spatial_dims).where(target_valid_mask.stack(z=spatial_dims), drop=True).values
        + target_mean.stack(z=spatial_dims).where(target_valid_mask.stack(z=spatial_dims), drop=True).values
    )

    prediction_scaled = reconstruct_grid(
        predictions=predictions_scaled_flat,
        timestamps=predictors["time"],
        target_valid_mask=target_valid_mask,
        variable_name=resolved_config["experiment"]["variable"],
        spatial_reference=target.isel(time=0, drop=True),
    )

    saved_output_units = resolved_config["inference"].get("saved_output_units", "scaled")
    print(f"Source target units: {resolved_config['metadata'].get('source_units', 'missing')}")
    print(f"Saved output units mode: {saved_output_units}")
    prediction = _apply_unit_choice(
        prediction_scaled,
        float(resolved_config["experiment"]["target_scale"]),
        saved_output_units,
    )

    clip_negative = bool(resolved_config["inference"].get("clip_negative_rainfall", False))
    if clip_negative:
        prediction = prediction.clip(min=0.0)

    source_units = resolved_config["metadata"].get("source_units")
    working_units = resolved_config["metadata"].get("working_units")
    if saved_output_units == "scaled":
        prediction.attrs["units"] = working_units
    else:
        prediction.attrs["units"] = source_units
    prediction.attrs["target_scale"] = float(resolved_config["experiment"]["target_scale"])
    prediction.attrs["source_units"] = source_units or "missing"
    prediction.attrs["working_units"] = working_units or "unspecified"
    prediction.attrs["cell_methods"] = target.attrs.get("cell_methods", "")
    prediction.attrs["accumulation_interval"] = target.attrs.get("accumulation_interval", "")
    prediction.attrs["timestamp_convention"] = target.attrs.get("timestamp_convention", "")
    prediction.attrs["model_checkpoint"] = str(experiment_dir / "model.h5")
    prediction.attrs["experiment_id"] = resolved_config["metadata"]["experiment_id"]
    prediction.attrs["inference_run_name"] = run_config["name"]
    prediction.attrs["predictor_list"] = ",".join(predictor_order)
    prediction.attrs["git_commit_sha"] = resolved_config["metadata"].get("git_commit_sha", "")
    prediction.attrs["negative_rainfall_clipping_applied"] = int(clip_negative)
    prediction.name = resolved_config["experiment"]["variable"]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_dataset(name=prediction.name).to_netcdf(output_file)

    evaluation_target = None
    if resolved_config["inference"].get("calculate_test_metrics", False) and metrics_file is not None:
        target_scale = float(resolved_config["experiment"]["target_scale"])
        if saved_output_units == "scaled":
            evaluation_target = target * target_scale
        else:
            evaluation_target = target
        metrics_df = _compute_metrics(evaluation_target, prediction)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_file, index=False)

    _verify_saved_time(output_file, prediction, evaluation_target)


def main() -> int:
    args = parse_args()
    raw_config, config_path = load_yaml_config(args.config_file)
    config = resolve_config_paths(raw_config, config_path)
    experiment_dir = get_experiment_dir(config)
    resolved_config = read_resolved_config(experiment_dir)

    predictor_mean, predictor_std, predictor_std_safe, predictor_order = load_predictor_normalization(
        experiment_dir / "normalization"
    )
    target_mean, target_std, target_std_safe, target_valid_mask, target_zero_std_mask = load_target_normalization(
        experiment_dir / "normalization"
    )

    inference_runs = get_inference_runs(resolved_config)
    multi_run_mode = len(inference_runs) > 1 or resolved_config.get("inference", {}).get("runs")

    for run_config in inference_runs:
        run_name = run_config["name"]
        print(f"Running inference target: {run_name}")
        if multi_run_mode:
            output_file = experiment_dir / "inference_runs" / run_name / "prediction.nc"
            metrics_file = experiment_dir / "inference_runs" / run_name / "metrics.csv"
        else:
            output_file = experiment_dir / "prediction.nc"
            metrics_file = experiment_dir / "metrics.csv"

        _run_single_inference(
            experiment_dir=experiment_dir,
            resolved_config=resolved_config,
            predictor_mean=predictor_mean,
            predictor_std_safe=predictor_std_safe,
            predictor_order=predictor_order,
            target_mean=target_mean,
            target_std_safe=target_std_safe,
            target_valid_mask=target_valid_mask,
            run_config=run_config,
            output_file=output_file,
            metrics_file=metrics_file,
        )

    print(f"Inference artifacts saved under {experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
