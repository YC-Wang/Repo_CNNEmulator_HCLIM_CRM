from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def repo_root_from_file(file_path: str) -> Path:
    return Path(file_path).resolve().parents[1]


def load_yaml_config(config_path: str) -> tuple[dict[str, Any], Path]:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, config_file


def _default_target_scale(variable: str) -> float:
    return 86400.0 if variable == "pr" else 1.0


def _default_predictor_time_offset_hours(variable: str) -> int:
    return 3 if variable == "pr" else 0


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
        if not variable:
            raise KeyError("experiment.variable is required when paths.predictor_file is not provided.")
        x_filename = experiment.get("x_filename")
        if x_filename:
            work_dir = Path(paths.get("work_dir", "."))
            data_train_dir = Path(paths.get("data_train_dir", "."))
            paths["predictor_file"] = str((base_dir / work_dir / data_train_dir / x_filename).resolve())

    if "target_file" not in paths:
        if not variable:
            raise KeyError("experiment.variable is required when paths.target_file is not provided.")
        y_filename_template = experiment.get("y_filename_template")
        if y_filename_template:
            data_infer_dir = Path(paths.get("data_infer_dir", "."))
            target_name = y_filename_template.format(variable=variable)
            paths["target_file"] = str((base_dir / data_infer_dir / target_name).resolve())

    paths.setdefault("output_root", "./outputs")

    training.setdefault("batch_size", 32)
    inference.setdefault("batch_size", training["batch_size"])
    inference.setdefault("calculate_test_metrics", True)
    inference.setdefault("clip_negative_rainfall", False)
    inference.setdefault("saved_output_units", "scaled")

    for key in ("predictor_file", "target_file", "output_root"):
        if key in paths:
            paths[key] = str((base_dir / paths[key]).resolve())

    for segment in resolved.get("training", {}).get("segments", []):
        for key in ("predictor_file", "target_file"):
            if key in segment:
                segment[key] = str((base_dir / segment[key]).resolve())

    for segment in resolved.get("training", {}).get("validation_segments", []):
        for key in ("predictor_file", "target_file"):
            if key in segment:
                segment[key] = str((base_dir / segment[key]).resolve())

    for run in resolved.get("inference", {}).get("runs", []):
        for key in ("predictor_file", "target_file"):
            if key in run:
                run[key] = str((base_dir / run[key]).resolve())
    return resolved


def get_experiment_id(config: dict[str, Any]) -> str:
    return config["metadata"]["experiment_id"]


def get_experiment_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["output_root"]) / get_experiment_id(config)


def ensure_output_directories(experiment_dir: Path) -> dict[str, Path]:
    directories = {
        "experiment": experiment_dir,
        "normalization": experiment_dir / "normalization",
        "logs": experiment_dir / "logs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def read_resolved_config(experiment_dir: Path) -> dict[str, Any]:
    with (experiment_dir / "config_resolved.yaml").open("r", encoding="utf-8") as handle:
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
            "name": "default",
            "predictor_file": config["paths"]["predictor_file"],
            "target_file": config["paths"]["target_file"],
            "dates": get_inference_dates(config),
        }
    ]
