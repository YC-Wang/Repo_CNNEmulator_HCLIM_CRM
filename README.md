# CNN Emulator for HCLIM Convection-Permitting Dataset

This repository contains the CNN-based temperature and precipitation emulator workflow used for HCLIM-CRM downscaling experiments. The current `main` branch uses YAML-driven training and inference entry points under `scripts/`, with a canonical experiment directory rooted at `paths.output_root`.

## Reference

- Rampal, N., et al. (2022). High-resolution downscaling with interpretable deep learning: Rainfall extremes over New Zealand. *Weather and Climate Extremes*, 38, 100525. [https://doi.org/10.1016/j.wace.2022.100525](https://doi.org/10.1016/j.wace.2022.100525)
- Wang, F., Y.-C. Wang, H. E. Kourabbaslou, K. Krus, A. Aldama-Campino, G. Nikulin, R. Doscher, P. Lind, S. Mirjalili, C. Lennard, and F. Schenk, 2026: Emulating land-atmosphere feedbacks in convection-permitting regional climate models using machine learning. *Artificial Intelligence for the Earth Systems*, submitted.

## Main Entry Points

- Training: `scripts/training_ncp_mse.py`
- Inference: `scripts/inference_ncp_mse.py`

Commands remain:

```bash
python scripts/training_ncp_mse.py config.yaml
python scripts/inference_ncp_mse.py config.yaml
```

Both scripts can also be launched from inside `scripts/` by passing a relative path such as `../config.yaml`.

## Canonical Configuration

The example [config.yaml](C:/Users/Yi-Chi/Documents/ChatGPT/paper-revision-CNN/config.yaml) now uses the canonical fields:

- `metadata.experiment_id`
- `paths.output_root`
- `training.overwrite_existing`
- `inference.run_name`
- `inference.dates`
- `inference.batch_size`
- `inference.calculate_test_metrics`
- `inference.clip_negative_rainfall`
- `inference.saved_output_units`
- `inference.prediction_filename`
- `inference.metrics_filename`
- `inference.overwrite_existing`

Path handling rules:

- the CLI config argument is resolved relative to the current working directory
- relative paths inside the YAML are resolved from the YAML file location
- absolute predictor, target, and output paths are preserved
- when `metadata.experiment_id` is absent, the YAML filename stem is used and a warning is logged

## Canonical Output Layout

All artifacts for one experiment live under:

```text
outputs/<experiment_id>/
  config/
    config_input.yaml
    config_resolved.yaml
  models/
    best_model.h5
  normalization/
    target_mean.nc
    target_std.nc
  logs/
    bootstrap/
    training.log
    inference/
      <run_name>.log
    tensorboard/
  inference/
    <run_name>/
      prediction.nc
      evaluation_metrics.csv
```

Meaning of the main paths:

- `config/config_input.yaml`: exact input YAML used for the run
- `config/config_resolved.yaml`: resolved configuration after defaults and path expansion
- `models/best_model.h5`: canonical training checkpoint and default inference model
- `logs/tensorboard/`: TensorBoard event directory
- `logs/training.log`: training diagnostic log
- `logs/inference/<run_name>.log`: inference diagnostic log
- `logs/bootstrap/`: early bootstrap logs written after the config is minimally resolved
- `inference/<run_name>/prediction.nc`: NetCDF predictions
- `inference/<run_name>/evaluation_metrics.csv`: evaluation metrics when enabled

Training and inference for the same experiment always resolve to the same experiment directory.

## Training

Run training:

```bash
python scripts/training_ncp_mse.py config.yaml
```

Optional overwrite override:

```bash
python scripts/training_ncp_mse.py config.yaml --overwrite
```

Training behavior:

- saves the best model to `outputs/<experiment_id>/models/best_model.h5`
- saves TensorBoard data to `outputs/<experiment_id>/logs/tensorboard/`
- saves training diagnostics to `outputs/<experiment_id>/logs/training.log`
- saves target normalization files to `outputs/<experiment_id>/normalization/`
- refuses to overwrite an existing `best_model.h5` unless `training.overwrite_existing: true` or `--overwrite` is used

## Inference

Run inference with the model produced by the same experiment:

```bash
python scripts/inference_ncp_mse.py config.yaml
```

Use an external model explicitly:

```bash
python scripts/inference_ncp_mse.py config.yaml --model-file /path/to/model.h5
```

Select a named inference run when `inference.runs` is present:

```bash
python scripts/inference_ncp_mse.py config.yaml --run-name test_2009
```

Optional overwrite override:

```bash
python scripts/inference_ncp_mse.py config.yaml --overwrite
```

Inference behavior:

- loads `outputs/<experiment_id>/models/best_model.h5` by default
- writes predictions under `outputs/<experiment_id>/inference/<run_name>/`
- writes diagnostics to `outputs/<experiment_id>/logs/inference/<run_name>.log`
- writes metrics only when `inference.calculate_test_metrics` is true
- refuses to overwrite an existing `prediction.nc` or `evaluation_metrics.csv` unless `inference.overwrite_existing: true` or `--overwrite` is used

Default inference data selection:

- predictor file: `paths.predictor_file`
- target file: `paths.target_file`
- dates: `inference.dates`, falling back to `experiment.dates.test`

Named-run example:

```yaml
inference:
  runs:
    - name: "test_2009"
      predictor_file: "/path/to/predictors.nc"
      target_file: "/path/to/target.nc"
      dates: ["2009-01-01", "2009-12-31"]
```

## Time Handling

Inference preserves the aligned timestamps from the actual run data:

- no year or reference timestamp is hard-coded
- predictor time offset support remains configurable with `experiment.predictor_time_offset_hours`
- NetCDF writing uses the in-memory timestamps and reopens the saved file to verify an exact decoded time-coordinate round trip
- the scripts log first time, last time, timestep, and timestamp count
- inference fails if writing the NetCDF introduces an additional time shift

For the temperature example:

- `experiment.predictor_time_offset_hours: 0`

Future precipitation configs can still set:

- `experiment.predictor_time_offset_hours: 3`

## Legacy Compatibility

Older YAML files remain supported, including configs that still use:

- `training.log_root`
- `training.model_root`
- `inference.output_dir`
- `inference.model_file`

These fields are deprecated. When they are present:

- the scripts log a deprecation warning
- `paths.output_root` is preferred when provided
- if only legacy fields are present, a conservative experiment root is derived from them so one run stays under one root

The actual experiment config [exp1_normal_withT.yaml](C:/Users/Yi-Chi/Documents/ChatGPT/paper-revision-CNN/exp1_normal_withT.yaml) remains usable through this compatibility path.

## Environment

The project environment is defined in `environment.yml`. For lightweight local validation, the non-TensorFlow checks only need:

- `PyYAML`
- `numpy`
- `pandas`
- `xarray`
- `pytest`
- `netCDF4`

TensorFlow-dependent training or inference runs should still be executed in the full project Conda environment or the Freja runtime environment.

## Validation

Lightweight validation commands:

```bash
python -m compileall scripts src tests
python -m pytest -q -p no:cacheprovider tests/test_logging_utils.py tests/test_training_script.py tests/test_inference_script.py tests/test_pipeline_utils.py
```

TensorFlow-dependent model construction, checkpoint loading, callback behavior, and real HCLIM training or inference runs are not covered by the lightweight test suite alone.

## Git Tracking

Generated runtime artifacts are not source files and should not remain tracked by Git. The repository ignore rules now exclude:

- `output/`
- `outputs/*` except `outputs/.gitkeep`
- `scripts/log_dir/`
- `slurm-*.out`
- model files, NetCDF files, TensorBoard events, and Python cache directories
