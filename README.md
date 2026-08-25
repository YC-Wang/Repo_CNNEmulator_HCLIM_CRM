# CNN Emulator for HCLIM Convection-Permitting Dataset

This repository contains the CNN-based temperature and precipitation emulator workflow used for HCLIM-CRM downscaling experiments. The current `main` branch uses YAML-driven training and inference entry points under `scripts/`.

## Reference

- Rampal, N., et al. (2022). High-resolution downscaling with interpretable deep learning: Rainfall extremes over New Zealand. *Weather and Climate Extremes*, 38, 100525. [https://doi.org/10.1016/j.wace.2022.100525](https://doi.org/10.1016/j.wace.2022.100525)
- Wang, F., Y.-C. Wang, H. E. Kourabbaslou, K. Krus, A. Aldama-Campino, G. Nikulin, R. Doscher, P. Lind, S. Mirjalili, C. Lennard, and F. Schenk, 2026: Emulating land-atmosphere feedbacks in convection-permitting regional climate models using machine learning. *Artificial Intelligence for the Earth Systems*, submitted.

## Main Entry Points

- Training: `scripts/training_ncp_mse.py`
- Inference: `scripts/inference_ncp_mse.py`

Both scripts accept a positional YAML config path and can be launched either from the repository root or from the `scripts/` directory.

## Environment

The project environment is defined in `environment.yml`. For lightweight local validation, the non-TensorFlow checks only need:

- `PyYAML`
- `numpy`
- `pandas`
- `xarray`
- `pytest`
- `netCDF4`

TensorFlow-dependent training or inference runs should be executed in the project Conda environment or the Freja runtime environment.

## Configuration

The codebase is config-driven. Existing experiment YAML files such as `exp1_normal_withT.yaml` remain the source of truth for:

- dataset paths
- train/validation/test periods
- predictor ordering
- model hyperparameters
- output locations

Path handling follows the current scripts:

- the CLI config argument is resolved relative to the current working directory
- relative paths inside the YAML are resolved from the YAML file location
- absolute HPC paths are preserved as absolute paths
- inference model paths can be supplied by `--model-file` or `inference.model_file`

## Training

Run training with a config file:

```bash
python scripts/training_ncp_mse.py exp1_normal_withT.yaml
```

From inside `scripts/`:

```bash
python training_ncp_mse.py ../exp1_normal_withT.yaml
```

Import-only smoke test:

```bash
python scripts/training_ncp_mse.py --smoke-test-imports
```

Training writes:

- a bootstrap log under `output/logs/bootstrap/`
- a run-specific diagnostic log under the configured `training.log_root`
- a TensorBoard log directory under the configured `training.log_root`
- model weights under the configured `training.model_root`
- a copied config file alongside the run outputs

## Inference

Run inference with the same YAML configuration style used by training:

```bash
python scripts/inference_ncp_mse.py exp1_normal_withT.yaml --model-file /path/to/model.h5
```

If the YAML includes `inference.model_file`, the CLI override is optional:

```bash
python scripts/inference_ncp_mse.py exp1_normal_withT.yaml
```

From inside `scripts/`:

```bash
python inference_ncp_mse.py ../exp1_normal_withT.yaml --model-file /path/to/model.h5
```

Import-only smoke test:

```bash
python scripts/inference_ncp_mse.py --smoke-test-imports
```

Inference writes:

- a bootstrap log under `output/logs/bootstrap/`
- a run-specific diagnostic log under `training.log_root/diagnostic/inference/`
- a prediction NetCDF file
- an evaluation metrics CSV
- a copied config file in the inference output directory

By default, inference output goes to `output/inference/<variable>/` next to the config file unless `--output-dir` or `inference.output_dir` is set.

## Notes on the Current Temperature Pipeline

- The temperature path on `main` preserves the legacy `exp1*.yaml` inventory.
- Temperature inference uses the training-period target mean and standard deviation to invert normalized predictions before writing NetCDF output.
- The current Freja bootstrap log added on `main` documents a real data issue where `tas` was requested in `downscale_variables` but was absent from the predictor dataset. That failure is a dataset/config mismatch, not an entry-point import problem.

## Validation

Available lightweight checks on this branch:

```bash
python -m compileall scripts src tests
python -m pytest -q -p no:cacheprovider tests/test_logging_utils.py tests/test_training_script.py tests/test_inference_script.py tests/test_pipeline_utils.py
```

TensorFlow-dependent model build, save/load, and end-to-end training validation should still be run in the full project environment.
