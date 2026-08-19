# Precipitation CNN Pipeline

This repository trains and runs a precipitation (`pr`) convolutional neural network with the existing HCLIM scientific setup preserved:

- the current CNN architecture from [src/models.py](/C:/Users/Yi-Chi/Documents/ChatGPT/paper-revision-CNN/src/models.py)
- ordinary MSE loss and MSE metric
- standard normalization with mean and standard deviation computed from the training period only
- separate training and inference entrypoints
- a lightweight wrapper that can run training, inference, or both
- configurable `+3 hour` predictor alignment for rainfall timestamps

## Environment

The original project environment is described in [environment.yml](/C:/Users/Yi-Chi/Documents/ChatGPT/paper-revision-CNN/environment.yml). The training and inference scripts require at least:

- Python with `tensorflow`
- `xarray`, `numpy`, `pandas`, `pyyaml`
- NetCDF IO support through `netcdf4` or `h5netcdf`

On Freja, activate the existing environment first:

```bash
mamba activate high_res_env
```

## Configuration

Use one YAML file for both training and inference. Start from [configs/pr_example.yaml](/C:/Users/Yi-Chi/Documents/ChatGPT/paper-revision-CNN/configs/pr_example.yaml) and copy it to `configs/pr01.yaml` through `configs/pr08.yaml` once the final periods and predictor combinations are defined.

Key fields:

- `metadata.experiment_id`: stable experiment directory name under `outputs/`
- `paths.predictor_file` and `paths.target_file`: NetCDF inputs
- `paths.output_root`: parent directory for experiment outputs
- `experiment.target_scale`: multiplier applied before target normalization
- `experiment.working_units`: explicit scaled target units
- `experiment.std_epsilon`: threshold for replacing only finite near-zero standard deviations with `1.0`
- `experiment.predictor_time_offset_hours`: rainfall predictor timestamp shift applied once before intersection
- `experiment.downscale_variables`: predictor ordering used for both training and inference
- `inference.saved_output_units`: `scaled` or `original`

`target_scale: 86400.0` is only appropriate when the source precipitation variable is a flux in `kg m-2 s-1` and the working unit should be `mm day-1`. The code prints the source target units at runtime and does not infer alternative units from the variable name.

## Commands

Train only:

```bash
python scripts/training_ncp_mse.py configs/pr01.yaml
```

Infer only:

```bash
python scripts/inference_ncp_mse.py configs/pr01.yaml
```

Run both stages:

```bash
python scripts/run_pipeline.py configs/pr01.yaml --stage all
```

Wrapper stage selection:

```bash
python scripts/run_pipeline.py configs/pr01.yaml --stage train
python scripts/run_pipeline.py configs/pr01.yaml --stage infer
```

## Output Layout

Each experiment uses a stable directory keyed by `metadata.experiment_id`:

```text
outputs/<experiment_id>/
  model.h5
  config_resolved.yaml
  history.csv
  prediction.nc
  metrics.csv
  normalization/
    predictor_mean.nc
    predictor_std.nc
    predictor_std_safe.nc
    target_normalization.nc
  logs/
```

The saved normalization files allow inference to load training-derived statistics directly instead of recalculating them.

## Normalization Rules

Predictor normalization:

- mean and standard deviation are computed over the training period only
- statistics are computed per predictor variable and grid cell
- finite standard deviations below `std_epsilon` are replaced with `1.0`
- missing statistics remain missing

Target normalization:

- `target_scale` is applied before target statistics are computed
- mean and standard deviation are computed over training time only
- statistics remain spatially varying over the target grid
- finite near-zero target standard deviations are replaced with `1.0`
- zero-variance normalized training targets are verified to be zero

Training, validation, and test targets share one saved training-derived target mask and one deterministic stacking order. Inference reconstructs the full output grid from that saved mask instead of deriving a new one.

## Rainfall Timestamps And NetCDF Time

For precipitation, predictor timestamps are shifted by `experiment.predictor_time_offset_hours` once before intersecting with target timestamps. The target timestamps are not shifted by the code. The aligned predictor timestamps are preserved through inference and saved directly to NetCDF.

The previous hard-coded `hours since 2009-01-01 ...` encoding has been removed. `prediction.nc` is written with the actual in-memory timestamps, then reopened immediately to verify that decoded timestamps still match exactly and that no extra three-hour shift was introduced.

## Freja Verification Before Running All Eight Experiments

Before launching the final `pr01` to `pr08` configurations on Freja, verify:

- predictor and target files expose the expected variable names
- source precipitation units justify the configured `target_scale`
- target metadata clarify accumulation interval and timestamp convention
- aligned timestamps match the intended rainfall semantics after the configurable `+3 hour` shift
- TensorFlow and any required GPU or CUDA stack are available in the Freja runtime
- the full HCLIM datasets fit the expected predictor and target spatial dimensions

## GitHub Actions For Eight Experiments

The repository includes a sample matrix workflow at [.github/workflows/run-eight-experiments.yml](/C:/Users/Yi-Chi/Documents/ChatGPT/paper-revision-CNN/.github/workflows/run-eight-experiments.yml). It is designed for a self-hosted runner on the same environment that can access the existing `/nobackup/...` predictor and target files.

Recommended setup:

- map each table row to one config file: `configs/pr01.yaml` through `configs/pr08.yaml`
- set `metadata.experiment_id` in each file to a stable output directory name
- keep the workflow as a thin orchestrator and let `scripts/run_pipeline.py` remain the single training and inference entrypoint
- use `workflow_dispatch` so you can rerun `train`, `infer`, or `all` without editing the workflow
- keep `max-parallel` conservative unless Freja capacity clearly supports more concurrent TensorFlow jobs

If Freja is only reachable through Slurm and cannot host a GitHub runner directly, keep the same matrix layout but replace the training step with an `sbatch` submission wrapper instead of running Python inline.
