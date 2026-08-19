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
    create_test_train_split,
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

    split_config = {
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

    x_train, x_val, x_test, y_train, y_val, y_test = create_test_train_split(split_config)

    source_units = y_train.attrs.get("units")
    target_scale = float(config["experiment"]["target_scale"])
    working_units = _working_units(config, source_units)
    print(f"Source target units: {source_units or 'missing'}")
    print(f"Applied target scale: {target_scale}")
    print(f"Working target units: {working_units}")

    y_train_scaled = y_train * target_scale
    y_val_scaled = y_val * target_scale
    y_test_scaled = y_test * target_scale

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
        x_val=x_val,
        x_test=x_test,
        y_train=y_train_scaled,
        y_val=y_val_scaled,
        y_test=y_test_scaled,
        predictor_mean=predictor_mean,
        predictor_std_safe=predictor_std_safe,
        target_mean=target_mean,
        target_std_safe=target_std_safe,
        target_valid_mask=target_valid_mask,
        variable_order=config["experiment"]["downscale_variables"],
    )

    validate_prepared_arrays(
        {
            "train": x_train_ready,
            "val": x_val_ready,
            "test": x_test_ready,
        },
        {
            "train": y_train_ready,
            "val": y_val_ready,
            "test": y_test_ready,
        },
    )

    resolved_config = json.loads(json.dumps(config))
    resolved_config["metadata"]["source_units"] = source_units
    resolved_config["metadata"]["working_units"] = working_units
    resolved_config["metadata"]["git_commit_sha"] = get_git_commit_sha(REPO_ROOT)
    resolved_config["metadata"]["predictor_order"] = config["experiment"]["downscale_variables"]
    resolved_config["metadata"]["target_spatial_dims"] = list(target_valid_mask.dims)
    resolved_config["metadata"]["target_masks_saved"] = True
    resolved_config["metadata"]["predictor_stats_saved"] = True
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

    history, _ = train_model(
        model=model,
        x_train=x_train_ready.values,
        y_train=y_train_ready.values,
        x_val=x_val_ready.values,
        y_val=y_val_ready.values,
        loss=config["training"]["loss"],
        model_weights_name=str(experiment_dir / "model.h5"),
        logdir=str(output_dirs["logs"]),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        optimizer=Adam(learning_rate=float(config["training"]["learning_rate"])),
        metrics=[config["training"]["metrics"]],
    )

    pd.DataFrame(history.history).to_csv(experiment_dir / "history.csv", index=False)
    print(f"Training artifacts saved under {experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
