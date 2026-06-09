"""Feature engineering for cross-sectional equity alpha research."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "ret_5",
    "ret_21",
    "ret_63",
    "momentum_21_5",
    "reversal_5",
    "vol_21",
    "vol_63",
    "range_5",
    "volume_shock",
    "liquidity",
    "quality_score",
    "value_score",
    "distance_to_high_63",
    "beta_63",
]

MODEL_FEATURE_COLUMNS = [f"z_{column}" for column in RAW_FEATURE_COLUMNS]


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator.replace(0, np.nan)
    return np.log(ratio.replace([np.inf, -np.inf], np.nan))


def build_features(market_data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Create point-in-time features and next-day targets."""

    frame = market_data.sort_values(["symbol", "date"]).copy()
    grouped = frame.groupby("symbol", sort=False)

    frame["ret_1"] = grouped["close"].pct_change()
    frame["ret_5"] = grouped["ret_1"].transform(
        lambda series: series.rolling(5, min_periods=5).sum()
    )
    frame["ret_21"] = grouped["ret_1"].transform(
        lambda series: series.rolling(21, min_periods=21).sum()
    )
    frame["ret_63"] = grouped["ret_1"].transform(
        lambda series: series.rolling(63, min_periods=63).sum()
    )
    frame["momentum_21_5"] = frame["ret_21"] - frame["ret_5"]
    frame["reversal_5"] = -frame["ret_5"]
    frame["vol_21"] = grouped["ret_1"].transform(
        lambda series: series.rolling(21, min_periods=21).std()
    )
    frame["vol_63"] = grouped["ret_1"].transform(
        lambda series: series.rolling(63, min_periods=45).std()
    )

    frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"]
    frame["range_5"] = grouped["range_pct"].transform(
        lambda series: series.rolling(5, min_periods=5).mean()
    )
    frame["dollar_volume"] = frame["close"] * frame["volume"]
    frame["adv_20"] = grouped["dollar_volume"].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    frame["volume_ma_20"] = grouped["volume"].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    frame["volume_shock"] = _safe_log_ratio(frame["volume"], frame["volume_ma_20"])
    frame["liquidity"] = np.log1p(frame["adv_20"])
    frame["high_63"] = grouped["close"].transform(
        lambda series: series.rolling(63, min_periods=63).max()
    )
    frame["distance_to_high_63"] = frame["close"] / frame["high_63"] - 1.0

    betas: list[pd.Series] = []
    for _, asset_frame in frame.groupby("symbol", sort=False):
        covariance = asset_frame["ret_1"].rolling(63, min_periods=45).cov(
            asset_frame["market_return"]
        )
        variance = asset_frame["market_return"].rolling(63, min_periods=45).var()
        betas.append(covariance / variance.replace(0, np.nan))
    frame["beta_63"] = pd.concat(betas).sort_index()

    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    frame["fwd_return_1d"] = frame.groupby("symbol", sort=False)["ret_1"].shift(-1)
    frame["fwd_excess_return_1d"] = frame["fwd_return_1d"] - frame.groupby("date")[
        "fwd_return_1d"
    ].transform("mean")

    winsor = float(config["features"].get("winsorize_zscore", 5.0))
    for column in RAW_FEATURE_COLUMNS:
        mean = frame.groupby("date")[column].transform("mean")
        std = frame.groupby("date")[column].transform("std").replace(0, np.nan)
        frame[f"z_{column}"] = ((frame[column] - mean) / std).clip(-winsor, winsor)

    required = MODEL_FEATURE_COLUMNS + ["fwd_return_1d", "fwd_excess_return_1d"]
    frame = frame.dropna(subset=required).reset_index(drop=True)
    return frame
