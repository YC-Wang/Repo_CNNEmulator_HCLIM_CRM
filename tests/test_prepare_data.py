from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prepare_data import (
    align_predictors_and_target,
    compute_training_stats,
    load_predictor_normalization,
    load_target_normalization,
    normalize_with_training_stats,
    prepare_training_dataset,
    reconstruct_grid,
    save_predictor_normalization,
    save_target_normalization,
)


def build_target_array(values: np.ndarray) -> xr.DataArray:
    times = np.array(
        [
            "2000-01-01T03:00:00",
            "2000-01-02T03:00:00",
            "2000-01-03T03:00:00",
            "2000-01-04T03:00:00",
        ],
        dtype="datetime64[ns]",
    )
    return xr.DataArray(
        values,
        dims=("time", "y", "x"),
        coords={"time": times, "y": [0, 1], "x": [0, 1]},
        name="pr",
        attrs={"units": "kg m-2 s-1"},
    )


def build_predictor_dataset(a_values: np.ndarray, b_values: np.ndarray) -> xr.Dataset:
    times = np.array(
        [
            "2000-01-01T00:00:00",
            "2000-01-02T00:00:00",
            "2000-01-03T00:00:00",
            "2000-01-04T00:00:00",
        ],
        dtype="datetime64[ns]",
    )
    coords = {"time": times, "y": [0, 1], "x": [0, 1]}
    return xr.Dataset(
        {
            "var_b": xr.DataArray(b_values, dims=("time", "y", "x"), coords=coords),
            "var_a": xr.DataArray(a_values, dims=("time", "y", "x"), coords=coords),
        }
    )


class PrepareDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = build_target_array(
            np.array(
                [
                    [[1.0, 2.0], [np.nan, 10.0]],
                    [[3.0, 4.0], [np.nan, 10.0]],
                    [[5.0, 6.0], [np.nan, 10.0]],
                    [[7.0, 8.0], [np.nan, 10.0]],
                ]
            )
        )
        self.predictors = build_predictor_dataset(
            a_values=np.array(
                [
                    [[1.0, 5.0], [np.nan, 1.0]],
                    [[3.0, 5.0], [np.nan, 1.0]],
                    [[9.0, 5.0], [np.nan, 2.0]],
                    [[11.0, 5.0], [np.nan, 3.0]],
                ]
            ),
            b_values=np.array(
                [
                    [[10.0, 0.0], [1.0, 7.0]],
                    [[12.0, 0.0], [1.0, 7.0]],
                    [[30.0, 0.0], [2.0, 9.0]],
                    [[32.0, 0.0], [3.0, 11.0]],
                ]
            ),
        )
        self.x_train = self.predictors.isel(time=slice(0, 2))
        self.x_val = self.predictors.isel(time=slice(2, 3))
        self.x_test = self.predictors.isel(time=slice(3, 4))
        self.y_train = self.target.isel(time=slice(0, 2))
        self.y_val = self.target.isel(time=slice(2, 3))
        self.y_test = self.target.isel(time=slice(3, 4))

    def test_training_stats_use_training_only_and_reuse_for_val_test(self) -> None:
        target_mean, target_std, target_std_safe, _, _ = compute_training_stats(self.y_train, 1.0e-6)
        predictor_mean, _, predictor_std_safe, _, _ = compute_training_stats(self.x_train, 1.0e-6)

        self.assertTrue(np.allclose(target_mean.sel(y=0, x=0).item(), 2.0))
        self.assertTrue(np.allclose(target_std.sel(y=0, x=0).item(), 1.0))
        self.assertTrue(np.allclose(predictor_mean["var_a"].sel(y=0, x=0).item(), 2.0))

        y_val_norm = normalize_with_training_stats(self.y_val, target_mean, target_std_safe)
        x_val_norm = normalize_with_training_stats(self.x_val, predictor_mean, predictor_std_safe)

        self.assertTrue(np.allclose(y_val_norm.sel(time=self.y_val.time[0], y=0, x=0).item(), 3.0))
        self.assertTrue(np.allclose(x_val_norm["var_a"].sel(time=self.x_val.time[0], y=0, x=0).item(), 7.0))

    def test_safe_std_preserves_missing_and_replaces_only_finite_zero(self) -> None:
        predictor_mean, predictor_std, predictor_std_safe, predictor_valid_mask, predictor_zero_std_mask = compute_training_stats(
            self.x_train,
            1.0e-6,
        )
        target_mean, target_std, target_std_safe, target_valid_mask, target_zero_std_mask = compute_training_stats(
            self.y_train,
            1.0e-6,
        )

        self.assertTrue(np.isnan(target_std.sel(y=1, x=0).item()))
        self.assertTrue(np.isnan(target_std_safe.sel(y=1, x=0).item()))
        self.assertTrue(target_zero_std_mask.sel(y=1, x=1).item())
        self.assertEqual(target_std_safe.sel(y=1, x=1).item(), 1.0)

        self.assertTrue(np.isnan(predictor_std["var_a"].sel(y=1, x=0).item()))
        self.assertTrue(np.isnan(predictor_std_safe["var_a"].sel(y=1, x=0).item()))
        self.assertTrue(predictor_zero_std_mask["var_b"].sel(y=0, x=1).item())
        self.assertEqual(predictor_std_safe["var_b"].sel(y=0, x=1).item(), 1.0)
        self.assertFalse(bool(predictor_valid_mask["var_a"].sel(y=1, x=0).item()))
        self.assertFalse(bool(target_valid_mask.sel(y=1, x=0).item()))

    def test_prepare_training_dataset_uses_common_mask_and_config_order(self) -> None:
        predictor_mean, _, predictor_std_safe, _, _ = compute_training_stats(self.x_train, 1.0e-6)
        target_mean, _, target_std_safe, target_valid_mask, target_zero_std_mask = compute_training_stats(
            self.y_train,
            1.0e-6,
        )

        x_train_ready, x_val_ready, x_test_ready, y_train_ready, y_val_ready, y_test_ready = prepare_training_dataset(
            x_train=self.x_train,
            x_val=self.x_val,
            x_test=self.x_test,
            y_train=self.y_train,
            y_val=self.y_val,
            y_test=self.y_test,
            predictor_mean=predictor_mean,
            predictor_std_safe=predictor_std_safe,
            target_mean=target_mean,
            target_std_safe=target_std_safe,
            target_valid_mask=target_valid_mask,
            variable_order=["var_b", "var_a"],
        )

        self.assertEqual(tuple(x_train_ready.feature.values.tolist()), ("var_b", "var_a"))
        self.assertEqual(y_train_ready.sizes["z"], y_val_ready.sizes["z"])
        self.assertEqual(y_train_ready.sizes["z"], y_test_ready.sizes["z"])
        self.assertEqual(y_train_ready.sizes["z"], 3)
        self.assertTrue(np.allclose(y_train_ready[:, -1].values, 0.0))
        self.assertTrue(target_zero_std_mask.sel(y=1, x=1).item())
        self.assertEqual(x_val_ready.shape[-1], 2)
        self.assertEqual(x_test_ready.shape[-1], 2)

    def test_inverse_normalization_recovers_original_values(self) -> None:
        target_mean, _, target_std_safe, _, _ = compute_training_stats(self.y_train, 1.0e-6)
        normalized = normalize_with_training_stats(self.y_val, target_mean, target_std_safe)
        recovered = normalized * target_std_safe + target_mean
        self.assertTrue(np.allclose(recovered.fillna(-999).values, self.y_val.fillna(-999).values))

    def test_reconstruct_grid_restores_cells_and_leaves_masked_points_missing(self) -> None:
        target_mean, _, _, target_valid_mask, _ = compute_training_stats(self.y_train, 1.0e-6)
        prediction = reconstruct_grid(
            predictions=np.array([[100.0, 200.0, 300.0]], dtype=np.float32),
            timestamps=self.y_val["time"],
            target_valid_mask=target_valid_mask,
            variable_name="pr",
        )

        self.assertEqual(prediction.sel(time=self.y_val.time[0], y=0, x=0).item(), 100.0)
        self.assertEqual(prediction.sel(time=self.y_val.time[0], y=0, x=1).item(), 200.0)
        self.assertTrue(np.isnan(prediction.sel(time=self.y_val.time[0], y=1, x=0).item()))
        self.assertEqual(prediction.sel(time=self.y_val.time[0], y=1, x=1).item(), 300.0)

    def test_artifacts_can_be_saved_and_loaded(self) -> None:
        predictor_mean, predictor_std, predictor_std_safe, _, _ = compute_training_stats(self.x_train, 1.0e-6)
        target_mean, target_std, target_std_safe, target_valid_mask, target_zero_std_mask = compute_training_stats(
            self.y_train,
            1.0e-6,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            normalization_dir = Path(tmpdir)
            save_predictor_normalization(
                predictor_mean,
                predictor_std,
                predictor_std_safe,
                ["var_b", "var_a"],
                normalization_dir,
            )
            save_target_normalization(
                target_mean,
                target_std,
                target_std_safe,
                target_valid_mask,
                target_zero_std_mask,
                normalization_dir,
            )

            loaded_mean, loaded_std, loaded_std_safe, predictor_order = load_predictor_normalization(normalization_dir)
            loaded_target_mean, loaded_target_std, loaded_target_std_safe, loaded_valid_mask, loaded_zero_mask = load_target_normalization(
                normalization_dir
            )

        self.assertEqual(predictor_order, ["var_b", "var_a"])
        self.assertTrue(loaded_mean.equals(predictor_mean))
        self.assertTrue(loaded_std.equals(predictor_std))
        self.assertTrue(loaded_std_safe.equals(predictor_std_safe))
        self.assertTrue(loaded_target_mean.equals(target_mean))
        self.assertTrue(loaded_target_std.equals(target_std))
        self.assertTrue(loaded_target_std_safe.equals(target_std_safe))
        self.assertTrue(loaded_valid_mask.equals(target_valid_mask))
        self.assertTrue(loaded_zero_mask.equals(target_zero_std_mask))

    def test_alignment_applies_offset_once_and_time_roundtrip_is_stable(self) -> None:
        predictors = build_predictor_dataset(
            a_values=np.ones((4, 2, 2)),
            b_values=np.ones((4, 2, 2)),
        )
        target = build_target_array(np.ones((4, 2, 2)))

        with tempfile.TemporaryDirectory() as tmpdir:
            predictor_file = Path(tmpdir) / "predictors.nc"
            target_file = Path(tmpdir) / "target.nc"
            predictors.to_netcdf(predictor_file)
            target.to_dataset(name="pr").to_netcdf(target_file)

            aligned_predictors, aligned_target = align_predictors_and_target(
                predictor_file=predictor_file,
                target_file=target_file,
                predictor_variables=["var_b", "var_a"],
                target_variable="pr",
                predictor_time_offset_hours=3,
                load_into_memory=True,
            )

            self.assertTrue(
                np.array_equal(
                    aligned_predictors.time.values,
                    aligned_target.time.values,
                )
            )
            self.assertEqual(str(aligned_predictors.time.values[0]), "2000-01-01T03:00:00.000000000")
            self.assertEqual(str(aligned_predictors.time.values[-1]), "2000-01-04T03:00:00.000000000")

            prediction = xr.DataArray(
                np.ones((aligned_predictors.sizes["time"], 2, 2)),
                dims=("time", "y", "x"),
                coords={
                    "time": aligned_predictors.time.values,
                    "y": [0, 1],
                    "x": [0, 1],
                },
                name="pr",
            )
            output_file = Path(tmpdir) / "prediction.nc"
            prediction.to_dataset(name="pr").to_netcdf(output_file)
            reopened_ds = xr.open_dataset(output_file)
            reopened = reopened_ds.load()["pr"]
            if hasattr(aligned_predictors, "close"):
                aligned_predictors.close()
            if hasattr(aligned_target, "close"):
                aligned_target.close()
            reopened_ds.close()

        self.assertTrue(np.array_equal(reopened.time.values, prediction.time.values))

    def test_inference_uses_saved_training_statistics(self) -> None:
        predictor_mean, predictor_std, predictor_std_safe, _, _ = compute_training_stats(self.x_train, 1.0e-6)
        shifted_inference = self.x_test.copy(deep=True)
        shifted_inference["var_a"] = shifted_inference["var_a"] + 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            normalization_dir = Path(tmpdir)
            save_predictor_normalization(
                predictor_mean,
                predictor_std,
                predictor_std_safe,
                ["var_b", "var_a"],
                normalization_dir,
            )
            loaded_mean, _, loaded_std_safe, predictor_order = load_predictor_normalization(normalization_dir)

        normalized = normalize_with_training_stats(
            shifted_inference[predictor_order],
            loaded_mean[predictor_order],
            loaded_std_safe[predictor_order],
        )
        self.assertTrue(
            np.allclose(
                normalized["var_a"].sel(time=shifted_inference.time[0], y=0, x=0).item(),
                109.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
