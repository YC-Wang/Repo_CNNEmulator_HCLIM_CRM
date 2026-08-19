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


def resolve_config_paths(config: dict[str, Any], config_file: Path) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    base_dir = config_file.parent
    paths = resolved["paths"]
    for key in ("predictor_file", "target_file", "output_root"):
        paths[key] = str((base_dir / paths[key]).resolve())
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
