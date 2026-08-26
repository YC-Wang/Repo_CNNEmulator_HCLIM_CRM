from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


FAILED_EXPERIMENT_ID = "_failed"
DEFAULT_OUTPUT_ROOT = "./outputs"
DEFAULT_MODEL_FILENAME = "best_model.h5"
DEFAULT_PREDICTION_FILENAME = "prediction.nc"
DEFAULT_METRICS_FILENAME = "evaluation_metrics.csv"
DEFAULT_RUN_NAME = "test"


def repo_root_from_file(file_path: str) -> Path:
    return Path(file_path).resolve().parents[1]


def load_yaml_config(config_path: str) -> tuple[dict[str, Any], Path]:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, config_file


def resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _default_target_scale(variable: str) -> float:
    return 86400.0 if variable == "pr" else 1.0


def _default_predictor_time_offset_hours(variable: str) -> int:
    return 3 if variable == "pr" else 0


def _legacy_root_candidate(path: Path, marker: str) -> Path:
    parts = list(path.parts)
    if marker in parts:
        marker_index = parts.index(marker)
        return Path(*parts[:marker_index]) if marker_index > 0 else path.parent
    return path.parent


def _derive_output_root(resolved: dict[str, Any], base_dir: Path) -> Path:
    paths = resolved.setdefault("paths", {})
    training = resolved.setdefault("training", {})
    inference = resolved.setdefault("inference", {})

    configured_root = paths.get("output_root")
    if configured_root:
        return resolve_path(base_dir, configured_root)

    candidates: list[Path] = []
    if training.get("log_root"):
        candidates.append(_legacy_root_candidate(resolve_path(base_dir, training["log_root"]), "logs"))
    if training.get("model_root"):
        candidates.append(_legacy_root_candidate(resolve_path(base_dir, training["model_root"]), "models"))
    if inference.get("output_dir"):
        candidates.append(_legacy_root_candidate(resolve_path(base_dir, inference["output_dir"]), "inference"))

    if candidates:
        return candidates[0].resolve()

    return resolve_path(base_dir, DEFAULT_OUTPUT_ROOT)


def _resolve_legacy_predictor_file(paths: dict[str, Any], experiment: dict[str, Any], base_dir: Path) -> str:
    x_filename = experiment.get("x_filename")
    if not x_filename:
        raise KeyError("paths.predictor_file or experiment.x_filename is required.")

    work_dir = resolve_path(base_dir, paths.get("work_dir", "."))
    data_train_dir = Path(paths.get("data_train_dir", "."))
    predictor_dir = data_train_dir if data_train_dir.is_absolute() else (work_dir / data_train_dir).resolve()
    return str((predictor_dir / x_filename).resolve())


def _resolve_legacy_target_file(paths: dict[str, Any], experiment: dict[str, Any], base_dir: Path) -> str:
    variable = experiment.get("variable")
    y_filename_template = experiment.get("y_filename_template")
    if not variable or not y_filename_template:
        raise KeyError("paths.target_file or experiment.variable and experiment.y_filename_template are required.")

    target_name = y_filename_template.format(variable=variable)
    data_infer_dir = Path(paths.get("data_infer_dir", "."))
    target_dir = data_infer_dir if data_infer_dir.is_absolute() else resolve_path(base_dir, data_infer_dir)
    return str((target_dir / target_name).resolve())


def resolve_config_paths(config: dict[str, Any], config_file: Path) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    base_dir = config_file.parent
    metadata = resolved.setdefault("metadata", {})
    paths = resolved.setdefault("paths", {})
    experiment = resolved.setdefault("experiment", {})
    training = resolved.setdefault("training", {})
    inference = resolved.setdefault("inference", {})

    metadata.setdefault("experiment_id", config_file.stem)

    variable = experiment.get("variable")
    if variable:
        experiment.setdefault("target_scale", _default_target_scale(variable))
        experiment.setdefault("predictor_time_offset_hours", _default_predictor_time_offset_hours(variable))
    experiment.setdefault("std_epsilon", 1.0e-6)

    if "predictor_file" not in paths:
        paths["predictor_file"] = _resolve_legacy_predictor_file(paths, experiment, base_dir)
    if "target_file" not in paths:
        paths["target_file"] = _resolve_legacy_target_file(paths, experiment, base_dir)

    paths["output_root"] = str(_derive_output_root(resolved, base_dir))
    paths["predictor_file"] = str(resolve_path(base_dir, paths["predictor_file"]))
    paths["target_file"] = str(resolve_path(base_dir, paths["target_file"]))

    training.setdefault("batch_size", 32)
    training.setdefault("overwrite_existing", False)
    inference.setdefault("run_name", DEFAULT_RUN_NAME)
    inference.setdefault("batch_size", 32)
    inference.setdefault("dates", experiment.get("dates", {}).get("test"))
    inference.setdefault("calculate_test_metrics", True)
    inference.setdefault("clip_negative_rainfall", False)
    inference.setdefault("saved_output_units", "scaled")
    inference.setdefault("prediction_filename", DEFAULT_PREDICTION_FILENAME)
    inference.setdefault("metrics_filename", DEFAULT_METRICS_FILENAME)
    inference.setdefault("overwrite_existing", False)

    for segment_name in ("segments", "validation_segments"):
        for segment in training.get(segment_name, []):
            for key in ("predictor_file", "target_file"):
                if key in segment:
                    segment[key] = str(resolve_path(base_dir, segment[key]))

    for run in inference.get("runs", []):
        run.setdefault("name", inference["run_name"])
        if "predictor_file" in run:
            run["predictor_file"] = str(resolve_path(base_dir, run["predictor_file"]))
        if "target_file" in run:
            run["target_file"] = str(resolve_path(base_dir, run["target_file"]))

    return resolved


def get_config_warnings(raw_config: dict[str, Any], resolved_config: dict[str, Any], config_file: Path) -> list[str]:
    warnings: list[str] = []
    metadata = raw_config.get("metadata", {})
    paths = raw_config.get("paths", {})
    training = raw_config.get("training", {})
    inference = raw_config.get("inference", {})

    if "experiment_id" not in metadata:
        warnings.append(
            f"metadata.experiment_id is not set in {config_file.name}; defaulting to '{resolved_config['metadata']['experiment_id']}'."
        )

    deprecated_fields = []
    for field_path, value in (
        ("training.log_root", training.get("log_root")),
        ("training.model_root", training.get("model_root")),
        ("inference.output_dir", inference.get("output_dir")),
        ("inference.model_file", inference.get("model_file")),
    ):
        if value is not None:
            deprecated_fields.append(field_path)

    if deprecated_fields:
        warnings.append(
            "Deprecated config fields detected: "
            + ", ".join(deprecated_fields)
            + f". Canonical outputs now resolve under paths.output_root={resolved_config['paths']['output_root']}."
        )
    elif "output_root" not in paths:
        warnings.append(
            f"paths.output_root is not set in {config_file.name}; defaulting to {resolved_config['paths']['output_root']}."
        )

    return warnings


def get_experiment_id(config: dict[str, Any]) -> str:
    return config["metadata"]["experiment_id"]


def get_experiment_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["output_root"]) / get_experiment_id(config)


def ensure_output_directories(experiment_dir: Path, run_name: str = DEFAULT_RUN_NAME) -> dict[str, Path]:
    directories = {
        "experiment_dir": experiment_dir,
        "config_dir": experiment_dir / "config",
        "models_dir": experiment_dir / "models",
        "normalization_dir": experiment_dir / "normalization",
        "logs_dir": experiment_dir / "logs",
        "bootstrap_log_dir": experiment_dir / "logs" / "bootstrap",
        "tensorboard_dir": experiment_dir / "logs" / "tensorboard",
        "inference_logs_dir": experiment_dir / "logs" / "inference",
        "inference_root_dir": experiment_dir / "inference",
        "inference_run_dir": experiment_dir / "inference" / run_name,
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def build_output_paths(config: dict[str, Any], run_name: str | None = None) -> dict[str, Path]:
    selected_run_name = run_name or config.get("inference", {}).get("run_name", DEFAULT_RUN_NAME)
    experiment_dir = get_experiment_dir(config)
    directories = ensure_output_directories(experiment_dir, run_name=selected_run_name)
    inference_cfg = config.get("inference", {})

    return {
        **directories,
        "input_config_backup": directories["config_dir"] / "config_input.yaml",
        "resolved_config_file": directories["config_dir"] / "config_resolved.yaml",
        "model_file": directories["models_dir"] / DEFAULT_MODEL_FILENAME,
        "normalization_mean_file": directories["normalization_dir"] / "target_mean.nc",
        "normalization_std_file": directories["normalization_dir"] / "target_std.nc",
        "training_log_file": directories["logs_dir"] / "training.log",
        "inference_log_file": directories["inference_logs_dir"] / f"{selected_run_name}.log",
        "prediction_file": directories["inference_run_dir"] / inference_cfg.get("prediction_filename", DEFAULT_PREDICTION_FILENAME),
        "metrics_file": directories["inference_run_dir"] / inference_cfg.get("metrics_filename", DEFAULT_METRICS_FILENAME),
    }


def build_failed_bootstrap_log_path(config_path: Path, stage: str, timestamp: str) -> Path:
    fallback_root = config_path.parent / "outputs" / FAILED_EXPERIMENT_ID / "logs" / "bootstrap"
    fallback_root.mkdir(parents=True, exist_ok=True)
    return (fallback_root / f"{stage}_{timestamp}.log").resolve()


def build_legacy_split_config(
    config: dict[str, Any],
    predictor_file: str | None = None,
    target_file: str | None = None,
    dates: list[str] | None = None,
) -> dict[str, Any]:
    selected_dates = dates or config["experiment"]["dates"]["test"]
    return {
        "X": predictor_file or config["paths"]["predictor_file"],
        "y": target_file or config["paths"]["target_file"],
        "train_start": config["experiment"]["dates"]["train"][0],
        "train_end": config["experiment"]["dates"]["train"][1],
        "val_start": config["experiment"]["dates"]["val"][0],
        "val_end": config["experiment"]["dates"]["val"][1],
        "test_start": selected_dates[0],
        "test_end": selected_dates[1],
        "output_var": [config["experiment"]["variable"]],
        "downscale_variables": config["experiment"]["downscale_variables"],
    }


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def read_resolved_config(experiment_dir: Path) -> dict[str, Any]:
    with (experiment_dir / "config" / "config_resolved.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_git_commit_sha(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def get_inference_dates(config: dict[str, Any]) -> list[str]:
    inference_dates = config.get("inference", {}).get("dates")
    if inference_dates:
        return inference_dates
    return config["experiment"]["dates"]["test"]


def get_inference_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    runs = config.get("inference", {}).get("runs")
    if runs:
        return runs

    return [
        {
            "name": config.get("inference", {}).get("run_name", DEFAULT_RUN_NAME),
            "predictor_file": config["paths"]["predictor_file"],
            "target_file": config["paths"]["target_file"],
            "dates": get_inference_dates(config),
        }
    ]


def get_inference_run(config: dict[str, Any], run_name: str | None = None) -> dict[str, Any]:
    runs = get_inference_runs(config)
    requested_name = run_name or config.get("inference", {}).get("run_name", DEFAULT_RUN_NAME)
    for run in runs:
        if run["name"] == requested_name:
            return run

    if len(runs) == 1 and run_name is None:
        return runs[0]

    available_names = ", ".join(run["name"] for run in runs)
    raise ValueError(f"Inference run '{requested_name}' was not found. Available runs: {available_names}")


def ensure_can_write_training_outputs(output_paths: dict[str, Path], overwrite_existing: bool) -> None:
    if output_paths["model_file"].exists() and not overwrite_existing:
        raise FileExistsError(
            f"Model file already exists: {output_paths['model_file']}. "
            "Set training.overwrite_existing: true or pass --overwrite to replace it."
        )


def ensure_can_write_inference_outputs(
    output_paths: dict[str, Path],
    overwrite_existing: bool,
    calculate_test_metrics: bool,
) -> None:
    blocking_files = [output_paths["prediction_file"]]
    if calculate_test_metrics:
        blocking_files.append(output_paths["metrics_file"])

    existing_files = [path for path in blocking_files if path.exists()]
    if existing_files and not overwrite_existing:
        raise FileExistsError(
            "Inference output already exists: "
            + ", ".join(str(path) for path in existing_files)
            + ". Set inference.overwrite_existing: true or pass --overwrite to replace it."
        )


def summarize_time_values(time_values: Any) -> dict[str, str | int]:
    values = list(time_values)
    if not values:
        raise ValueError("Time coordinate is empty.")
    timestep = "n/a"
    if len(values) > 1:
        timestep = str(values[1] - values[0])
    return {
        "first": str(values[0]),
        "last": str(values[-1]),
        "count": len(values),
        "timestep": timestep,
    }


def write_netcdf_with_time_validation(dataset: Any, output_file: Path) -> dict[str, str | int]:
    import numpy as np
    import xarray as xr

    output_file.parent.mkdir(parents=True, exist_ok=True)
    original_times = dataset["time"].values.copy()
    dataset.to_netcdf(output_file)

    with xr.open_dataset(output_file, decode_times=True) as reopened:
        reopened_times = reopened["time"].values.copy()

    if original_times.shape != reopened_times.shape or not np.array_equal(original_times, reopened_times):
        raise ValueError(
            f"Time-coordinate mismatch after writing NetCDF to {output_file}. "
            "Decoded timestamps do not match the in-memory timestamps."
        )

    return summarize_time_values(original_times)
