"""Portfolio construction for a market-neutral equity stat-arb book."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _date_zscore(series: pd.Series) -> pd.Series:
    standard_deviation = series.std()
    if standard_deviation == 0 or np.isnan(standard_deviation):
        return series * 0.0
    return (series - series.mean()) / standard_deviation


def _normalize_side(raw_weights: pd.Series, target_gross: float) -> pd.Series:
    total = raw_weights.abs().sum()
    if total == 0 or np.isnan(total):
        return raw_weights * 0.0
    return raw_weights / total * target_gross


def construct_portfolio(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Convert forecasts into capped long/short portfolio weights."""

    portfolio_config = config["portfolio"]
    selection_quantile = float(portfolio_config["selection_quantile"])
    gross_leverage = float(portfolio_config["gross_leverage"])
    max_abs_weight = float(portfolio_config["max_abs_weight"])
    volatility_floor = float(portfolio_config["volatility_floor"])
    sector_neutralize = bool(portfolio_config["sector_neutralize"])

    frame = predictions.dropna(subset=["prediction", "fwd_return_1d"]).copy()
    if sector_neutralize:
        frame["score"] = frame["prediction"] - frame.groupby(["date", "sector"])[
            "prediction"
        ].transform("mean")
    else:
        frame["score"] = frame["prediction"]

    frame["score"] = frame.groupby("date")["score"].transform(_date_zscore)
    frame["risk_adjusted_score"] = frame["score"] / (
        frame["vol_21"].clip(lower=volatility_floor)
    )

    position_frames: list[pd.DataFrame] = []
    for date, date_frame in frame.groupby("date", sort=True):
        if len(date_frame) < 10:
            continue

        long_cutoff = date_frame["score"].quantile(1.0 - selection_quantile)
        short_cutoff = date_frame["score"].quantile(selection_quantile)
        selected = date_frame.loc[
            (date_frame["score"] >= long_cutoff)
            | (date_frame["score"] <= short_cutoff)
        ].copy()
        if selected.empty:
            continue

        selected["weight"] = 0.0
        long_mask = selected["score"] > 0
        short_mask = selected["score"] < 0

        long_raw = selected.loc[long_mask, "risk_adjusted_score"].clip(lower=0.0)
        short_raw = selected.loc[short_mask, "risk_adjusted_score"].clip(upper=0.0)
        selected.loc[long_mask, "weight"] = _normalize_side(
            long_raw,
            gross_leverage / 2.0,
        )
        selected.loc[short_mask, "weight"] = _normalize_side(
            short_raw,
            gross_leverage / 2.0,
        )

        selected["weight"] = selected["weight"].clip(-max_abs_weight, max_abs_weight)
        long_total = selected.loc[selected["weight"] > 0, "weight"].sum()
        short_total = selected.loc[selected["weight"] < 0, "weight"].abs().sum()
        if long_total > 0:
            selected.loc[selected["weight"] > 0, "weight"] *= (
                gross_leverage / 2.0 / long_total
            )
        if short_total > 0:
            selected.loc[selected["weight"] < 0, "weight"] *= (
                gross_leverage / 2.0 / short_total
            )
        selected["weight"] = selected["weight"].clip(-max_abs_weight, max_abs_weight)

        selected["date"] = date
        position_frames.append(
            selected[
                [
                    "date",
                    "symbol",
                    "sector",
                    "prediction",
                    "score",
                    "weight",
                    "fwd_return_1d",
                    "fwd_excess_return_1d",
                    "vol_21",
                ]
            ]
        )

    if not position_frames:
        raise ValueError("Portfolio construction produced no positions.")

    return pd.concat(position_frames, ignore_index=True)
