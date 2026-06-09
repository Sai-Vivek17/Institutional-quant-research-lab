"""Backtesting engine with turnover, cost, and exposure accounting."""

from __future__ import annotations

from typing import Any

import pandas as pd


def run_backtest(positions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Backtest point-in-time weights against next-day realized returns."""

    portfolio_config = config["portfolio"]
    cost_rate = (
        float(portfolio_config["cost_bps"]) + float(portfolio_config["slippage_bps"])
    ) / 10_000.0

    weights = positions.pivot_table(
        index="date",
        columns="symbol",
        values="weight",
        aggfunc="sum",
    ).fillna(0.0)
    forward_returns = positions.pivot_table(
        index="date",
        columns="symbol",
        values="fwd_return_1d",
        aggfunc="first",
    ).reindex_like(weights).fillna(0.0)

    gross_return = (weights * forward_returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = weights.iloc[0].abs().sum()
    transaction_cost = turnover * cost_rate
    net_return = gross_return - transaction_cost
    equity = (1.0 + net_return).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    result = pd.DataFrame(
        {
            "date": weights.index,
            "gross_return": gross_return.to_numpy(),
            "transaction_cost": transaction_cost.to_numpy(),
            "net_return": net_return.to_numpy(),
            "turnover": turnover.to_numpy(),
            "gross_exposure": weights.abs().sum(axis=1).to_numpy(),
            "net_exposure": weights.sum(axis=1).to_numpy(),
            "long_count": (weights > 0).sum(axis=1).to_numpy(),
            "short_count": (weights < 0).sum(axis=1).to_numpy(),
            "equity": equity.to_numpy(),
            "drawdown": drawdown.to_numpy(),
        }
    )
    return result.reset_index(drop=True)


def attach_regimes(
    backtest_frame: pd.DataFrame,
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the market regime observed on each signal date."""

    regimes = market_data.groupby("date", as_index=False)["regime"].first()
    return backtest_frame.merge(regimes, on="date", how="left")
