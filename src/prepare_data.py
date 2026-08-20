from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar


TIME_DIM = "time"
SUPPORTED_SPATIAL_DIMS = (
    ("y", "x"),
    ("latitude", "longitude"),
    ("lat", "lon"),
)


def format_features(x_data: xr.Dataset, variable_order: Iterable[str] | None = None) -> xr.DataArray:
    ordered_names = list(variable_order or x_data.data_vars)
    features = xr.concat([x_data[name] for name in ordered_names], dim="feature")
    features = features.assign_coords(feature=("feature", ordered_names))
    features.name = "stacked_features"
    return features


def get_spatial_dims(data: xr.Dataset | xr.DataArray) -> tuple[str, str]:
    for dims in SUPPORTED_SPATIAL_DIMS:
        if all(dim in data.dims for dim in dims):
            return dims
    raise ValueError(
        f"Expected one of {SUPPORTED_SPATIAL_DIMS} in {tuple(data.dims)}."
    )


def get_time_frequency(time_coord: xr.DataArray) -> str:
    index = pd.DatetimeIndex(pd.to_datetime(time_coord.values))
    if len(index) < 3:
        return "undetectable"
    freq = pd.infer_freq(index)
    return freq or "undetectable"


def _print_time_range(label: str, time_coord: xr.DataArray) -> None:
    index = pd.DatetimeIndex(pd.to_datetime(time_coord.values))
    if index.empty:
        print(f"{label}: no timestamps")
        return
    print(
        f"{label}: first={index[0]}, last={index[-1]}, "
        f"freq={get_time_frequency(time_coord)}"
    )


def _coerce_datetime_hourly(time_coord: xr.DataArray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(time_coord.values)).floor("h")


def _drop_duplicate_times(
    data: xr.Dataset | xr.DataArray,
    label: str,
) -> xr.Dataset | xr.DataArray:
    index = _coerce_datetime_hourly(data[TIME_DIM])
    duplicate_mask = index.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        print(f"{label}: dropping {duplicate_count} duplicate timestamps")
        data = data.isel({TIME_DIM: ~duplicate_mask})
        index = _coerce_datetime_hourly(data[TIME_DIM])
    if index.has_duplicates:
        raise ValueError(f"{label} still has duplicate timestamps after deduplication.")
    return data.assign_coords({TIME_DIM: index})


def _validate_requested_variables(dataset: xr.Dataset, required_variables: Iterable[str], label: str) -> None:
    missing = [name for name in required_variables if name not in dataset.data_vars]
    if missing:
        raise KeyError(f"{label} is missing required variables: {missing}")


def align_predictors_and_target(
    predictor_file: Path,
    target_file: Path,
    predictor_variables: list[str],
    target_variable: str,
    predictor_time_offset_hours: int,
    load_into_memory: bool = False,
) -> tuple[xr.Dataset, xr.DataArray]:
    predictors_opened = xr.open_dataset(predictor_file, chunks={TIME_DIM: 3000})
    target_opened = xr.open_dataset(target_file, chunks={TIME_DIM: 3000})

    _validate_requested_variables(predictors_opened, predictor_variables, "Predictor dataset")
    _validate_requested_variables(target_opened, [target_variable], "Target dataset")

    predictors = predictors_opened[predictor_variables]
    target = target_opened[target_variable]

    original_predictor_time = _coerce_datetime_hourly(predictors[TIME_DIM])
    adjusted_predictor_time = original_predictor_time + pd.to_timedelta(
        predictor_time_offset_hours,
        unit="h",
    )
    target_time = _coerce_datetime_hourly(target[TIME_DIM])

    print("Time alignment summary before intersection")
    _print_time_range("predictor-original", xr.DataArray(original_predictor_time, dims=TIME_DIM))
    _print_time_range("predictor-adjusted", xr.DataArray(adjusted_predictor_time, dims=TIME_DIM))
    _print_time_range("target", xr.DataArray(target_time, dims=TIME_DIM))

    predictors = predictors.assign_coords({TIME_DIM: adjusted_predictor_time})
    target = target.assign_coords({TIME_DIM: target_time})

    predictors = _drop_duplicate_times(predictors, "Predictor dataset")
    target = _drop_duplicate_times(target, "Target dataset")

    common_times = predictors.indexes[TIME_DIM].intersection(target.indexes[TIME_DIM])
    if common_times.empty:
        raise ValueError("No overlapping timestamps remain after predictor-target alignment.")

    predictors = predictors.sel({TIME_DIM: common_times})
    target = target.sel({TIME_DIM: common_times})

    if predictors.sizes[TIME_DIM] != target.sizes[TIME_DIM]:
        raise ValueError("Predictor and target sample counts differ after alignment.")

    aligned_time = pd.DatetimeIndex(predictors.indexes[TIME_DIM])
    unique_hours = sorted(set(aligned_time.hour.tolist()))
    print("Time alignment summary after intersection")
    print(
        f"common timestamps={len(aligned_time)}, first={aligned_time[0]}, "
        f"last={aligned_time[-1]}, unique_hours={unique_hours}"
    )

    if load_into_memory:
        predictors = predictors.load()
        target = target.load()
        predictors_opened.close()
        target_opened.close()

    return predictors, target


def split_aligned_data(
    predictors: xr.Dataset,
    target: xr.DataArray,
    dates: dict[str, list[str]],
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.DataArray, xr.DataArray, xr.DataArray]:
    splits = {}
    for name in ("train", "val", "test"):
        start, end = dates[name]
        x_part = predictors.sel({TIME_DIM: slice(start, end)})
        y_part = target.sel({TIME_DIM: slice(start, end)})
        if x_part.sizes.get(TIME_DIM, 0) == 0 or y_part.sizes.get(TIME_DIM, 0) == 0:
            raise ValueError(f"The configured {name} period {start} to {end} is empty.")
        if x_part.sizes[TIME_DIM] != y_part.sizes[TIME_DIM]:
            raise ValueError(f"Predictor/target mismatch inside the {name} split.")
        splits[name] = (x_part, y_part)

    print(
        "Split sample counts: "
        f"train={splits['train'][0].sizes[TIME_DIM]}, "
        f"val={splits['val'][0].sizes[TIME_DIM]}, "
        f"test={splits['test'][0].sizes[TIME_DIM]}"
    )

    with ProgressBar():
        loaded = {
            name: (x_part.load(), y_part.load())
            for name, (x_part, y_part) in splits.items()
        }

    return (
        loaded["train"][0],
        loaded["val"][0],
        loaded["test"][0],
        loaded["train"][1],
        loaded["val"][1],
        loaded["test"][1],
    )


def slice_aligned_data(
    predictors: xr.Dataset,
    target: xr.DataArray,
    date_range: list[str],
    label: str,
) -> tuple[xr.Dataset, xr.DataArray]:
    start, end = date_range
    x_part = predictors.sel({TIME_DIM: slice(start, end)})
    y_part = target.sel({TIME_DIM: slice(start, end)})
    if x_part.sizes.get(TIME_DIM, 0) == 0 or y_part.sizes.get(TIME_DIM, 0) == 0:
        raise ValueError(f"The configured {label} period {start} to {end} is empty.")
    if x_part.sizes[TIME_DIM] != y_part.sizes[TIME_DIM]:
        raise ValueError(f"Predictor/target mismatch inside the {label} split.")
    return x_part, y_part


def concatenate_time_segments(
    x_segments: list[xr.Dataset],
    y_segments: list[xr.DataArray],
    label: str,
) -> tuple[xr.Dataset, xr.DataArray]:
    if not x_segments or not y_segments:
        raise ValueError(f"No {label} segments were provided.")

    with ProgressBar():
        x_loaded = [segment.load() for segment in x_segments]
        y_loaded = [segment.load() for segment in y_segments]

    predictors = xr.concat(x_loaded, dim=TIME_DIM)
    target = xr.concat(y_loaded, dim=TIME_DIM)

    time_index = pd.DatetimeIndex(pd.to_datetime(predictors[TIME_DIM].values))
    if time_index.has_duplicates:
        duplicates = time_index[time_index.duplicated()].unique().tolist()
        raise ValueError(f"{label} segments contain duplicate timestamps: {duplicates[:5]}")

    predictors = predictors.assign_coords({TIME_DIM: time_index})
    target = target.assign_coords({TIME_DIM: time_index})

    print(f"{label} sample count after concatenation: {predictors.sizes[TIME_DIM]}")
    return predictors, target


def load_segmented_split(
    segments: list[dict],
    predictor_variables: list[str],
    target_variable: str,
    predictor_time_offset_hours: int,
    label: str,
) -> tuple[xr.Dataset, xr.DataArray]:
    x_segments: list[xr.Dataset] = []
    y_segments: list[xr.DataArray] = []

    for index, segment in enumerate(segments, start=1):
        segment_label = segment.get("name", f"{label}-{index}")
        predictors, target = align_predictors_and_target(
            predictor_file=Path(segment["predictor_file"]),
            target_file=Path(segment["target_file"]),
            predictor_variables=predictor_variables,
            target_variable=target_variable,
            predictor_time_offset_hours=predictor_time_offset_hours,
        )
        x_part, y_part = slice_aligned_data(
            predictors,
            target,
            segment["dates"],
            f"{label} segment {segment_label}",
        )
        x_segments.append(x_part)
        y_segments.append(y_part)

    return concatenate_time_segments(x_segments, y_segments, label)


def compute_training_stats(
    data: xr.Dataset | xr.DataArray,
    std_epsilon: float,
) -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
    mean = data.mean(dim=TIME_DIM)
    std = data.std(dim=TIME_DIM)
    valid_mask = xr.apply_ufunc(np.isfinite, mean) & xr.apply_ufunc(np.isfinite, std)
    zero_std_mask = valid_mask & (std < std_epsilon)
    std_safe = std.where(~zero_std_mask, other=1.0)
    return mean, std, std_safe, valid_mask, zero_std_mask


def normalize_with_training_stats(
    data: xr.Dataset | xr.DataArray,
    mean: xr.Dataset | xr.DataArray,
    std_safe: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    return (data - mean) / std_safe


def _stack_with_mask(
    data: xr.DataArray,
    valid_mask: xr.DataArray,
    spatial_dims: tuple[str, str],
) -> xr.DataArray:
    valid_stack = valid_mask.stack(z=spatial_dims)
    if data.sizes.get(TIME_DIM, 0) == 0:
        empty = xr.DataArray(
            np.empty((0, int(valid_stack.sum().item())), dtype=data.dtype),
            dims=(TIME_DIM, "z"),
            coords={
                TIME_DIM: data[TIME_DIM].values,
                "z": valid_stack.where(valid_stack, drop=True).coords["z"].values,
            },
        )
        return empty
    stacked = data.stack(z=spatial_dims).transpose(TIME_DIM, "z")
    return stacked.where(valid_stack, drop=True)


def prepare_training_dataset(
    x_train: xr.Dataset,
    x_val: xr.Dataset,
    x_test: xr.Dataset,
    y_train: xr.DataArray,
    y_val: xr.DataArray,
    y_test: xr.DataArray,
    predictor_mean: xr.Dataset,
    predictor_std_safe: xr.Dataset,
    target_mean: xr.DataArray,
    target_std_safe: xr.DataArray,
    target_valid_mask: xr.DataArray,
    variable_order: list[str],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    spatial_dims = get_spatial_dims(y_train)

    x_train_norm = normalize_with_training_stats(x_train[variable_order], predictor_mean[variable_order], predictor_std_safe[variable_order])
    x_val_norm = normalize_with_training_stats(x_val[variable_order], predictor_mean[variable_order], predictor_std_safe[variable_order])
    x_test_norm = normalize_with_training_stats(x_test[variable_order], predictor_mean[variable_order], predictor_std_safe[variable_order])

    x_train_norm = format_features(x_train_norm, variable_order).transpose(TIME_DIM, *spatial_dims, "feature")
    x_val_norm = format_features(x_val_norm, variable_order).transpose(TIME_DIM, *spatial_dims, "feature")
    x_test_norm = format_features(x_test_norm, variable_order).transpose(TIME_DIM, *spatial_dims, "feature")

    y_train_norm = normalize_with_training_stats(y_train, target_mean, target_std_safe)
    y_val_norm = normalize_with_training_stats(y_val, target_mean, target_std_safe)
    y_test_norm = normalize_with_training_stats(y_test, target_mean, target_std_safe)

    y_train_norm = _stack_with_mask(y_train_norm, target_valid_mask, spatial_dims)
    y_val_norm = _stack_with_mask(y_val_norm, target_valid_mask, spatial_dims)
    y_test_norm = _stack_with_mask(y_test_norm, target_valid_mask, spatial_dims)

    return x_train_norm, x_val_norm, x_test_norm, y_train_norm, y_val_norm, y_test_norm


def validate_prepared_arrays(
    x_arrays: dict[str, xr.DataArray],
    y_arrays: dict[str, xr.DataArray],
) -> None:
    output_size = None
    for name, y_array in y_arrays.items():
        current_size = y_array.sizes["z"]
        if output_size is None:
            output_size = current_size
        elif current_size != output_size:
            raise ValueError("Training, validation, and test targets do not share the same output size.")
        if not np.isfinite(y_array.values).all():
            raise ValueError(f"Non-finite target values found in {name}.")

    for name, x_array in x_arrays.items():
        if not np.isfinite(x_array.values).all():
            raise ValueError(f"Non-finite predictor values found in {name}.")


def reconstruct_grid(
    predictions: np.ndarray,
    timestamps: xr.DataArray,
    target_valid_mask: xr.DataArray,
    variable_name: str,
    spatial_reference: xr.DataArray | None = None,
) -> xr.DataArray:
    spatial_dims = get_spatial_dims(target_valid_mask)
    full_shape = (len(timestamps),) + tuple(target_valid_mask.sizes[dim] for dim in spatial_dims)
    filled = np.full(full_shape, np.nan, dtype=np.float32)

    template = xr.DataArray(
        filled,
        dims=(TIME_DIM, *spatial_dims),
        coords={
            TIME_DIM: timestamps.values,
            spatial_dims[0]: target_valid_mask.coords[spatial_dims[0]].values,
            spatial_dims[1]: target_valid_mask.coords[spatial_dims[1]].values,
        },
        name=variable_name,
    )

    stacked = template.stack(z=spatial_dims)
    valid_stack = target_valid_mask.stack(z=spatial_dims)
    stacked.loc[{ "z": valid_stack[valid_stack].z }] = predictions
    reconstructed = stacked.unstack("z")

    if spatial_reference is not None:
        for coord_name, coord in spatial_reference.coords.items():
            if coord_name == TIME_DIM:
                continue
            if all(dim in reconstructed.dims for dim in coord.dims):
                reconstructed = reconstructed.assign_coords({coord_name: coord})

    return reconstructed


def save_predictor_normalization(
    predictor_mean: xr.Dataset,
    predictor_std: xr.Dataset,
    predictor_std_safe: xr.Dataset,
    variable_order: list[str],
    normalization_dir: Path,
) -> None:
    metadata = {
        "predictor_order": json.dumps(variable_order),
    }
    predictor_mean = predictor_mean.assign_attrs(metadata)
    predictor_std = predictor_std.assign_attrs(metadata)
    predictor_std_safe = predictor_std_safe.assign_attrs(metadata)
    predictor_mean.to_netcdf(normalization_dir / "predictor_mean.nc")
    predictor_std.to_netcdf(normalization_dir / "predictor_std.nc")
    predictor_std_safe.to_netcdf(normalization_dir / "predictor_std_safe.nc")


def save_target_normalization(
    target_mean: xr.DataArray,
    target_std: xr.DataArray,
    target_std_safe: xr.DataArray,
    target_valid_mask: xr.DataArray,
    target_zero_std_mask: xr.DataArray,
    normalization_dir: Path,
) -> None:
    spatial_dims = get_spatial_dims(target_mean)
    target_ds = xr.Dataset(
        {
            "target_mean": target_mean,
            "target_std": target_std,
            "target_std_safe": target_std_safe,
            "target_valid_mask": target_valid_mask.astype(np.int8),
            "target_zero_std_mask": target_zero_std_mask.astype(np.int8),
        }
    )
    target_ds.attrs["target_spatial_dims"] = json.dumps(list(spatial_dims))
    target_ds.attrs["stacking_convention"] = "xarray.DataArray.stack(z=target_spatial_dims)"
    target_ds.to_netcdf(normalization_dir / "target_normalization.nc")


def load_predictor_normalization(normalization_dir: Path) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, list[str]]:
    predictor_mean_ds = xr.open_dataset(normalization_dir / "predictor_mean.nc")
    predictor_std_ds = xr.open_dataset(normalization_dir / "predictor_std.nc")
    predictor_std_safe_ds = xr.open_dataset(normalization_dir / "predictor_std_safe.nc")
    predictor_mean = predictor_mean_ds.load()
    predictor_std = predictor_std_ds.load()
    predictor_std_safe = predictor_std_safe_ds.load()
    predictor_mean_ds.close()
    predictor_std_ds.close()
    predictor_std_safe_ds.close()
    variable_order = json.loads(predictor_mean.attrs["predictor_order"])
    return predictor_mean, predictor_std, predictor_std_safe, variable_order


def load_target_normalization(normalization_dir: Path) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    target_opened = xr.open_dataset(normalization_dir / "target_normalization.nc")
    target_ds = target_opened.load()
    target_opened.close()
    return (
        target_ds["target_mean"],
        target_ds["target_std"],
        target_ds["target_std_safe"],
        target_ds["target_valid_mask"].astype(bool),
        target_ds["target_zero_std_mask"].astype(bool),
    )


def create_test_train_split(config: dict) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.DataArray, xr.DataArray, xr.DataArray]:
    predictors, target = align_predictors_and_target(
        predictor_file=Path(config["X"]),
        target_file=Path(config["y"]),
        predictor_variables=config["downscale_variables"],
        target_variable=config["output_var"][0],
        predictor_time_offset_hours=config.get("predictor_time_offset_hours", 0),
    )
    dates = {
        "train": [config["train_start"], config["train_end"]],
        "val": [config["val_start"], config["val_end"]],
        "test": [config["test_start"], config["test_end"]],
    }
    return split_aligned_data(predictors, target, dates)


def create_training_split_from_segments(
    train_segments: list[dict],
    predictor_variables: list[str],
    target_variable: str,
    predictor_time_offset_hours: int,
    validation_segments: list[dict] | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None, xr.DataArray, xr.DataArray | None]:
    x_train, y_train = load_segmented_split(
        segments=train_segments,
        predictor_variables=predictor_variables,
        target_variable=target_variable,
        predictor_time_offset_hours=predictor_time_offset_hours,
        label="train",
    )

    if validation_segments:
        x_val, y_val = load_segmented_split(
            segments=validation_segments,
            predictor_variables=predictor_variables,
            target_variable=target_variable,
            predictor_time_offset_hours=predictor_time_offset_hours,
            label="validation",
        )
        return x_train, x_val, y_train, y_val

    return x_train, None, y_train, None
