#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import simple_conv, train_model
from pipeline_utils import (
    ensure_output_directories,
    get_experiment_dir,
    get_experiment_id,
    get_git_commit_sha,
    load_yaml_config,
    resolve_config_paths,
    write_yaml,
)
from prepare_data import (
    compute_training_stats,
    create_training_split_from_segments,
    create_test_train_split,
    load_segmented_split,
    normalize_with_training_stats,
    prepare_training_dataset,
    save_predictor_normalization,
    save_target_normalization,
    validate_prepared_arrays,
)


tf.random.set_seed(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the precipitation CNN pipeline.")
    parser.add_argument("config_file", help="Path to the YAML configuration file.")
    return parser.parse_args()


def _working_units(config: dict, source_units: str | None) -> str:
    working_units = config["experiment"].get("working_units")
    if working_units:
        return working_units
    return source_units or "unspecified"


def _build_default_split_config(config: dict, predictor_file: Path, target_file: Path, variable: str) -> dict:
    return {
        "X": str(predictor_file),
        "y": str(target_file),
        "train_start": config["experiment"]["dates"]["train"][0],
        "train_end": config["experiment"]["dates"]["train"][1],
        "val_start": config["experiment"]["dates"]["val"][0],
        "val_end": config["experiment"]["dates"]["val"][1],
        "test_start": config["experiment"]["dates"]["test"][0],
        "test_end": config["experiment"]["dates"]["test"][1],
        "output_var": [variable],
        "downscale_variables": config["experiment"]["downscale_variables"],
        "predictor_time_offset_hours": config["experiment"]["predictor_time_offset_hours"],
    }


def _empty_like_time(data):
    return data.isel(time=slice(0, 0))


def main() -> int:
    args = parse_args()
    raw_config, config_path = load_yaml_config(args.config_file)
    config = resolve_config_paths(raw_config, config_path)

    experiment_id = get_experiment_id(config)
    experiment_dir = get_experiment_dir(config)
    output_dirs = ensure_output_directories(experiment_dir)

    predictor_file = Path(config["paths"]["predictor_file"])
    target_file = Path(config["paths"]["target_file"])
    variable = config["experiment"]["variable"]

    print(f"Experiment ID: {experiment_id}")
    print(f"Configuration: {config_path}")
    print(f"Predictor file: {predictor_file}")
    print(f"Target file: {target_file}")

    training_segments = config.get("training", {}).get("segments")
    validation_segments = config.get("training", {}).get("validation_segments")
    uses_segmented_training = bool(training_segments)
    sequential_training = bool(config.get("training", {}).get("sequential_segments", False))

    if uses_segmented_training:
        x_train, x_val, y_train, y_val = create_training_split_from_segments(
            train_segments=training_segments,
            predictor_variables=config["experiment"]["downscale_variables"],
            target_variable=variable,
            predictor_time_offset_hours=config["experiment"]["predictor_time_offset_hours"],
            validation_segments=validation_segments,
        )
        x_test = None
        y_test = None
    else:
        split_config = _build_default_split_config(config, predictor_file, target_file, variable)
        x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(split_config)

    source_units = y_train.attrs.get("units")
    target_scale = float(config["experiment"]["target_scale"])
    working_units = _working_units(config, source_units)
    print(f"Source target units: {source_units or 'missing'}")
    print(f"Applied target scale: {target_scale}")
    print(f"Working target units: {working_units}")

    y_train_scaled = y_train * target_scale
    y_val_scaled = y_val * target_scale if y_val is not None else None
    y_test_scaled = y_test * target_scale if y_test is not None else None

    std_epsilon = float(config["experiment"]["std_epsilon"])
    predictor_mean, predictor_std, predictor_std_safe, predictor_valid_mask, predictor_zero_std_mask = compute_training_stats(
        x_train[config["experiment"]["downscale_variables"]],
        std_epsilon,
    )
    target_mean, target_std, target_std_safe, target_valid_mask, target_zero_std_mask = compute_training_stats(
        y_train_scaled,
        std_epsilon,
    )

    zero_variance_training = normalize_with_training_stats(
        y_train_scaled,
        target_mean,
        target_std_safe,
    ).where(target_zero_std_mask)
    if zero_variance_training.count() and not np.allclose(
        zero_variance_training.fillna(0.0).values,
        0.0,
    ):
        raise ValueError("Zero-variance normalized training targets must evaluate to zero.")

    x_train_ready, x_val_ready, x_test_ready, y_train_ready, y_val_ready, y_test_ready = prepare_training_dataset(
        x_train=x_train,
        x_val=x_val if x_val is not None else x_train.isel(time=slice(0, 0)),
        x_test=x_test if x_test is not None else x_train.isel(time=slice(0, 0)),
        y_train=y_train_scaled,
        y_val=y_val_scaled if y_val_scaled is not None else y_train_scaled.isel(time=slice(0, 0)),
        y_test=y_test_scaled if y_test_scaled is not None else y_train_scaled.isel(time=slice(0, 0)),
        predictor_mean=predictor_mean,
        predictor_std_safe=predictor_std_safe,
        target_mean=target_mean,
        target_std_safe=target_std_safe,
        target_valid_mask=target_valid_mask,
        variable_order=config["experiment"]["downscale_variables"],
    )

    x_arrays = {"train": x_train_ready}
    y_arrays = {"train": y_train_ready}
    if x_val is not None and y_val is not None:
        x_arrays["val"] = x_val_ready
        y_arrays["val"] = y_val_ready
    if x_test is not None and y_test is not None:
        x_arrays["test"] = x_test_ready
        y_arrays["test"] = y_test_ready
    validate_prepared_arrays(x_arrays, y_arrays)

    resolved_config = json.loads(json.dumps(config))
    resolved_config["metadata"]["source_units"] = source_units
    resolved_config["metadata"]["working_units"] = working_units
    resolved_config["metadata"]["git_commit_sha"] = get_git_commit_sha(REPO_ROOT)
    resolved_config["metadata"]["predictor_order"] = config["experiment"]["downscale_variables"]
    resolved_config["metadata"]["target_spatial_dims"] = list(target_valid_mask.dims)
    resolved_config["metadata"]["target_masks_saved"] = True
    resolved_config["metadata"]["predictor_stats_saved"] = True
    resolved_config["metadata"]["uses_segmented_training"] = uses_segmented_training
    resolved_config["metadata"]["sequential_training_segments"] = sequential_training
    resolved_config["paths"]["experiment_dir"] = str(experiment_dir)
    resolved_config["artifacts"] = {
        "model": "model.h5",
        "history": "history.csv",
        "prediction": "prediction.nc",
        "metrics": "metrics.csv",
        "normalization_dir": "normalization",
    }

    save_predictor_normalization(
        predictor_mean=predictor_mean,
        predictor_std=predictor_std,
        predictor_std_safe=predictor_std_safe,
        variable_order=config["experiment"]["downscale_variables"],
        normalization_dir=output_dirs["normalization"],
    )
    save_target_normalization(
        target_mean=target_mean,
        target_std=target_std,
        target_std_safe=target_std_safe,
        target_valid_mask=target_valid_mask,
        target_zero_std_mask=target_zero_std_mask,
        normalization_dir=output_dirs["normalization"],
    )
    write_yaml(experiment_dir / "config_resolved.yaml", resolved_config)

    input_shape = x_train_ready.shape[1:]
    output_shape = y_train_ready.sizes["z"]

    model = simple_conv(
        layer_filters=config["model"]["layer_filters"],
        bn=config["model"]["use_bn"],
        padding=config["model"]["padding"],
        kernel_size=(config["model"]["kernel_size"], config["model"]["kernel_size"]),
        pooling=config["model"]["use_pooling"],
        dense_layers=[config["model"]["hidden_layer_dense"], output_shape],
        dense_activation=config["model"]["dense_activation"],
        input_shape=input_shape,
        dropout=config["model"]["dropout"],
        activation=config["model"]["cnn_activation"],
    )

    history_frames: list[pd.DataFrame] = []
    if uses_segmented_training and sequential_training:
        if x_val is not None and y_val is not None:
            x_val_values = x_val_ready.values
            y_val_values = y_val_ready.values
        else:
            x_val_values = None
            y_val_values = None

        for phase_index, segment in enumerate(training_segments, start=1):
            phase_name = segment.get("name", f"segment_{phase_index}")
            print(f"Sequential training phase {phase_index}: {phase_name}")
            x_phase, y_phase = load_segmented_split(
                segments=[segment],
                predictor_variables=config["experiment"]["downscale_variables"],
                target_variable=variable,
                predictor_time_offset_hours=config["experiment"]["predictor_time_offset_hours"],
                label=f"train-phase-{phase_name}",
            )
            x_phase_ready, _, _, y_phase_ready, _, _ = prepare_training_dataset(
                x_train=x_phase,
                x_val=_empty_like_time(x_phase),
                x_test=_empty_like_time(x_phase),
                y_train=y_phase * target_scale,
                y_val=_empty_like_time(y_phase),
                y_test=_empty_like_time(y_phase),
                predictor_mean=predictor_mean,
                predictor_std_safe=predictor_std_safe,
                target_mean=target_mean,
                target_std_safe=target_std_safe,
                target_valid_mask=target_valid_mask,
                variable_order=config["experiment"]["downscale_variables"],
            )
            history, _ = train_model(
                model=model,
                x_train=x_phase_ready.values,
                y_train=y_phase_ready.values,
                x_val=x_val_values,
                y_val=y_val_values,
                loss=config["training"]["loss"],
                model_weights_name=str(experiment_dir / "model.h5"),
                logdir=str(output_dirs["logs"] / f"phase_{phase_index:02d}_{phase_name}"),
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                optimizer=Adam(learning_rate=float(config["training"]["learning_rate"])),
                metrics=[config["training"]["metrics"]],
            )
            history_df = pd.DataFrame(history.history)
            history_df.insert(0, "phase_name", phase_name)
            history_df.insert(0, "phase_index", phase_index)
            history_frames.append(history_df)
    else:
        history, _ = train_model(
            model=model,
            x_train=x_train_ready.values,
            y_train=y_train_ready.values,
            x_val=x_val_ready.values if x_val is not None and y_val is not None else None,
            y_val=y_val_ready.values if x_val is not None and y_val is not None else None,
            loss=config["training"]["loss"],
            model_weights_name=str(experiment_dir / "model.h5"),
            logdir=str(output_dirs["logs"]),
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            optimizer=Adam(learning_rate=float(config["training"]["learning_rate"])),
            metrics=[config["training"]["metrics"]],
        )
        history_df = pd.DataFrame(history.history)
        history_df.insert(0, "phase_name", "combined")
        history_df.insert(0, "phase_index", 1)
        history_frames.append(history_df)

    pd.concat(history_frames, ignore_index=True).to_csv(experiment_dir / "history.csv", index=False)
    print(f"Training artifacts saved under {experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
