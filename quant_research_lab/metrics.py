"""Performance and risk analytics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def evaluate_performance(backtest_frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate headline risk and performance metrics."""

    returns = backtest_frame["net_return"].astype(float)
    n_days = len(returns)
    if n_days == 0:
        raise ValueError("Cannot evaluate an empty backtest.")

    cumulative_return = float((1.0 + returns).prod() - 1.0)
    annualized_return = float((1.0 + cumulative_return) ** (TRADING_DAYS / n_days) - 1.0)
    annualized_volatility = float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
        if returns.std(ddof=1) > 0
        else 0.0
    )

    downside = returns.loc[returns < 0]
    downside_volatility = float(downside.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sortino = (
        float(returns.mean() / downside.std(ddof=1) * math.sqrt(TRADING_DAYS))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else 0.0
    )

    drawdown = max_drawdown(returns)
    calmar = (
        float(annualized_return / abs(drawdown))
        if drawdown < 0
        else float("inf")
    )
    value_at_risk_95 = float(returns.quantile(0.05))
    cvar_95 = float(returns.loc[returns <= value_at_risk_95].mean())

    metrics = {
        "days": int(n_days),
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "downside_volatility": downside_volatility,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "hit_rate": float((returns > 0).mean()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "value_at_risk_95": value_at_risk_95,
        "conditional_var_95": cvar_95,
        "average_turnover": float(backtest_frame["turnover"].mean()),
        "average_gross_exposure": float(backtest_frame["gross_exposure"].mean()),
        "average_net_exposure": float(backtest_frame["net_exposure"].mean()),
        "average_daily_cost": float(backtest_frame["transaction_cost"].mean()),
    }
    return metrics


def summarize_by_regime(backtest_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate risk and return by simulated market regime."""

    if "regime" not in backtest_frame:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for regime, group in backtest_frame.groupby("regime"):
        returns = group["net_return"]
        rows.append(
            {
                "regime": regime,
                "days": len(group),
                "mean_daily_return": returns.mean(),
                "annualized_return": (1 + returns.mean()) ** TRADING_DAYS - 1,
                "annualized_volatility": returns.std(ddof=1) * math.sqrt(TRADING_DAYS),
                "sharpe": (
                    returns.mean() / returns.std(ddof=1) * math.sqrt(TRADING_DAYS)
                    if returns.std(ddof=1) > 0
                    else 0.0
                ),
                "hit_rate": (returns > 0).mean(),
                "max_drawdown": max_drawdown(returns),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def prediction_diagnostics(predictions: pd.DataFrame) -> dict[str, float]:
    """Measure out-of-sample forecast quality."""

    clean = predictions.dropna(subset=["prediction", "fwd_excess_return_1d"])
    if clean.empty:
        return {"information_coefficient": 0.0, "rank_ic": 0.0}

    pearson_values = []
    spearman_values = []
    for _, group in clean.groupby("date"):
        if group["prediction"].nunique() < 2 or group["fwd_excess_return_1d"].nunique() < 2:
            continue
        pearson_values.append(group["prediction"].corr(group["fwd_excess_return_1d"]))
        spearman_values.append(
            group["prediction"].rank().corr(group["fwd_excess_return_1d"].rank())
        )

    return {
        "information_coefficient": float(np.nanmean(pearson_values)),
        "rank_ic": float(np.nanmean(spearman_values)),
        "ic_observations": float(len(pearson_values)),
    }
