"""Walk-forward ridge model for cross-sectional return forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_research_lab.features import MODEL_FEATURE_COLUMNS


@dataclass
class FittedRidgeModel:
    feature_columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame[self.feature_columns].to_numpy(dtype=float)
        standardized = (features - self.mean) / self.scale
        design = np.column_stack([np.ones(len(frame)), standardized])
        return design @ self.coefficients


def _fit_ridge(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    ridge_lambda: float,
    sample_weight: np.ndarray,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> FittedRidgeModel:
    features = train_frame[feature_columns].to_numpy(dtype=float)
    target = train_frame[target_column].to_numpy(dtype=float)

    mean = np.nanmean(features, axis=0)
    scale = np.nanstd(features, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(train_frame)), standardized])

    sqrt_weight = np.sqrt(sample_weight).reshape(-1, 1)
    weighted_design = design * sqrt_weight
    weighted_target = target * sqrt_weight.ravel()

    penalty = np.eye(design.shape[1]) * ridge_lambda
    penalty[0, 0] = 0.0
    left = weighted_design.T @ weighted_design + penalty
    right = weighted_design.T @ weighted_target

    try:
        coefficients = np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(left) @ right

    return FittedRidgeModel(
        feature_columns=feature_columns,
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        train_start=train_start,
        train_end=train_end,
    )


def walk_forward_predictions(
    feature_frame: pd.DataFrame,
    config: dict[str, Any],
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate out-of-sample predictions with purged walk-forward retraining."""

    model_config = config["model"]
    feature_columns = feature_columns or MODEL_FEATURE_COLUMNS
    target_column = str(model_config["target"])
    train_window = int(model_config["train_window"])
    purge_days = int(model_config["purge_days"])
    retrain_every = int(model_config["retrain_every"])
    ridge_lambda = float(model_config["ridge_lambda"])
    half_life_days = float(model_config["half_life_days"])
    min_train_observations = int(model_config["min_train_observations"])

    frame = feature_frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    dates = pd.Index(sorted(frame["date"].unique()))
    date_to_index = {date: index for index, date in enumerate(dates)}

    prediction_frames: list[pd.DataFrame] = []
    coefficient_records: list[dict[str, object]] = []
    fitted_model: FittedRidgeModel | None = None
    last_retrain_index = -10**9

    first_test_index = train_window + purge_days
    for test_index in range(first_test_index, len(dates)):
        test_date = dates[test_index]
        train_end_index = test_index - purge_days
        train_start_index = max(0, train_end_index - train_window)
        train_dates = dates[train_start_index:train_end_index]

        should_retrain = (
            fitted_model is None
            or test_index - last_retrain_index >= retrain_every
        )
        if should_retrain:
            train_mask = frame["date"].isin(train_dates)
            train_frame = frame.loc[train_mask].copy()
            if len(train_frame) < min_train_observations:
                continue

            train_frame["_date_index"] = train_frame["date"].map(date_to_index)
            age = train_end_index - train_frame["_date_index"].to_numpy(dtype=float)
            sample_weight = np.power(0.5, age / half_life_days)

            fitted_model = _fit_ridge(
                train_frame=train_frame,
                feature_columns=feature_columns,
                target_column=target_column,
                ridge_lambda=ridge_lambda,
                sample_weight=sample_weight,
                train_start=pd.Timestamp(dates[train_start_index]),
                train_end=pd.Timestamp(dates[train_end_index - 1]),
            )
            last_retrain_index = test_index

            coefficient_record: dict[str, object] = {
                "train_start": fitted_model.train_start,
                "train_end": fitted_model.train_end,
                "intercept": fitted_model.coefficients[0],
            }
            for name, coefficient in zip(
                feature_columns,
                fitted_model.coefficients[1:],
                strict=True,
            ):
                coefficient_record[name] = coefficient
            coefficient_records.append(coefficient_record)

        if fitted_model is None:
            continue

        test_frame = frame.loc[frame["date"] == test_date].copy()
        if test_frame.empty:
            continue

        test_frame["prediction"] = fitted_model.predict(test_frame)
        test_frame["model_train_start"] = fitted_model.train_start
        test_frame["model_train_end"] = fitted_model.train_end
        prediction_frames.append(test_frame)

    if not prediction_frames:
        raise ValueError("Walk-forward model produced no predictions.")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = pd.DataFrame.from_records(coefficient_records)
    return predictions, coefficients
