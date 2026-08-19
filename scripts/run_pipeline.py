#!/usr/bin/env python

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_utils import get_experiment_id, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training, inference, or both.")
    parser.add_argument("config_file", help="Path to the YAML configuration file.")
    parser.add_argument(
        "--stage",
        choices=("train", "infer", "all"),
        default="all",
        help="Which stage to execute.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> int:
    result = subprocess.run(command)
    return result.returncode


def main() -> int:
    args = parse_args()
    raw_config, config_path = load_yaml_config(args.config_file)
    experiment_id = get_experiment_id(raw_config)

    train_cmd = [sys.executable, str(Path(__file__).with_name("training_ncp_mse.py")), str(config_path)]
    infer_cmd = [sys.executable, str(Path(__file__).with_name("inference_ncp_mse.py")), str(config_path)]

    print(f"Experiment ID: {experiment_id}")
    print(f"Configuration: {config_path}")

    if args.stage in ("train", "all"):
        print(f"Running: {' '.join(train_cmd)}")
        train_code = run_command(train_cmd)
        if train_code != 0:
            return train_code

    if args.stage in ("infer", "all"):
        print(f"Running: {' '.join(infer_cmd)}")
        infer_code = run_command(infer_cmd)
        if infer_code != 0:
            return infer_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
