"""Synthetic institutional-style market data generator.

The project uses synthetic data on purpose: recruiters can run the complete
pipeline without paid data, stale API keys, or licensing ambiguity. The
generator still contains realistic structure: market regimes, sector shocks,
asset betas, liquidity differences, fat-tail stress days, and latent signals
that the model must discover out of sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REGIME_PARAMS = {
    "bull": {"mu": 0.00055, "sigma": 0.0085, "sector_sigma": 0.0040},
    "sideways": {"mu": 0.00005, "sigma": 0.0100, "sector_sigma": 0.0048},
    "bear": {"mu": -0.00045, "sigma": 0.0140, "sector_sigma": 0.0060},
    "stress": {"mu": -0.00120, "sigma": 0.0240, "sector_sigma": 0.0110},
}


@dataclass(frozen=True)
class AssetUniverse:
    symbols: np.ndarray
    sectors: np.ndarray
    sector_ids: np.ndarray
    market_beta: np.ndarray
    sector_beta: np.ndarray
    idiosyncratic_vol: np.ndarray
    quality: np.ndarray
    value: np.ndarray
    base_volume: np.ndarray


def make_asset_universe(
    n_assets: int,
    sectors: list[str],
    rng: np.random.Generator,
) -> AssetUniverse:
    """Create a stable synthetic asset universe with realistic heterogeneity."""

    symbols = np.array([f"EQ{i:03d}" for i in range(1, n_assets + 1)])
    sector_ids = np.arange(n_assets) % len(sectors)
    rng.shuffle(sector_ids)
    sector_names = np.array([sectors[index] for index in sector_ids])

    quality = rng.normal(0.0, 1.0, n_assets)
    value = rng.normal(0.0, 1.0, n_assets)
    size = rng.normal(0.0, 1.0, n_assets)

    return AssetUniverse(
        symbols=symbols,
        sectors=sector_names,
        sector_ids=sector_ids,
        market_beta=np.clip(rng.normal(1.0, 0.22, n_assets), 0.45, 1.75),
        sector_beta=np.clip(rng.normal(0.75, 0.18, n_assets), 0.25, 1.35),
        idiosyncratic_vol=np.clip(rng.lognormal(-4.55, 0.28, n_assets), 0.006, 0.025),
        quality=quality,
        value=value,
        base_volume=np.exp(13.5 + 0.65 * size + rng.normal(0, 0.20, n_assets)),
    )


def _next_regime(
    current: str,
    rng: np.random.Generator,
    transition_probability: float = 0.025,
) -> str:
    if rng.random() > transition_probability:
        return current

    choices = {
        "bull": ["bull", "sideways", "bear"],
        "sideways": ["bull", "sideways", "bear", "stress"],
        "bear": ["sideways", "bear", "stress"],
        "stress": ["sideways", "bear", "stress"],
    }
    probabilities = {
        "bull": [0.45, 0.40, 0.15],
        "sideways": [0.25, 0.45, 0.20, 0.10],
        "bear": [0.35, 0.45, 0.20],
        "stress": [0.35, 0.45, 0.20],
    }
    return str(rng.choice(choices[current], p=probabilities[current]))


def simulate_market_data(config: dict[str, Any]) -> pd.DataFrame:
    """Simulate OHLCV data with cross-sectional alpha structure."""

    data_config = config["data"]
    rng = np.random.default_rng(int(data_config["seed"]))
    n_assets = int(data_config["assets"])
    dates = pd.bdate_range(data_config["start"], periods=int(data_config["periods"]))
    sectors = list(data_config["sectors"])
    universe = make_asset_universe(n_assets, sectors, rng)

    prices = np.exp(rng.normal(np.log(75), 0.35, n_assets))
    returns_history = np.zeros((len(dates), n_assets), dtype=float)
    records: list[dict[str, object]] = []
    regime = "bull"

    for date_index, date in enumerate(dates):
        regime = _next_regime(regime, rng)
        params = REGIME_PARAMS[regime]

        market_return = rng.normal(params["mu"], params["sigma"])
        sector_shocks = (
            0.30 * market_return
            + rng.normal(0.0, params["sector_sigma"], len(sectors))
        )

        if date_index >= 21:
            trailing_21 = returns_history[date_index - 21 : date_index].sum(axis=0)
        else:
            trailing_21 = np.zeros(n_assets)

        if date_index >= 5:
            trailing_5 = returns_history[date_index - 5 : date_index].sum(axis=0)
        else:
            trailing_5 = np.zeros(n_assets)

        latent_momentum = 0.00150 * np.tanh(9.0 * trailing_21)
        latent_reversal = -0.00115 * np.tanh(15.0 * trailing_5)
        fundamental_alpha = 0.00032 * universe.quality + 0.00026 * universe.value
        crowding_noise = rng.normal(0.0, 0.00018, n_assets)
        latent_alpha = latent_momentum + latent_reversal + fundamental_alpha
        latent_alpha += crowding_noise

        asset_returns = (
            universe.market_beta * market_return
            + universe.sector_beta * sector_shocks[universe.sector_ids]
            + latent_alpha
            + rng.normal(0.0, universe.idiosyncratic_vol, n_assets)
        )

        if regime == "stress":
            jump_mask = rng.random(n_assets) < 0.035
            asset_returns[jump_mask] += rng.normal(-0.025, 0.030, jump_mask.sum())

        asset_returns = np.clip(asset_returns, -0.22, 0.22)
        overnight_noise = rng.normal(0.0, universe.idiosyncratic_vol * 0.25)
        open_prices = prices * (1.0 + 0.20 * asset_returns + overnight_noise)
        close_prices = prices * (1.0 + asset_returns)
        close_prices = np.maximum(close_prices, 1.0)

        intraday_range = (
            np.abs(asset_returns)
            + rng.lognormal(-5.0, 0.45, n_assets)
            + 0.003
        )
        high_prices = np.maximum(open_prices, close_prices) * (1.0 + intraday_range)
        low_prices = np.minimum(open_prices, close_prices) * (1.0 - intraday_range)
        low_prices = np.maximum(low_prices, 0.50)

        volume_multiplier = np.exp(
            rng.normal(0.0, 0.22, n_assets)
            + 7.5 * np.abs(asset_returns)
            + (0.28 if regime == "stress" else 0.0)
        )
        volumes = np.maximum(10_000, universe.base_volume * volume_multiplier)

        for asset_index, symbol in enumerate(universe.symbols):
            records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "sector": universe.sectors[asset_index],
                    "open": open_prices[asset_index],
                    "high": high_prices[asset_index],
                    "low": low_prices[asset_index],
                    "close": close_prices[asset_index],
                    "volume": int(volumes[asset_index]),
                    "quality_score": universe.quality[asset_index]
                    + rng.normal(0.0, 0.035),
                    "value_score": universe.value[asset_index]
                    + rng.normal(0.0, 0.035),
                    "raw_return": asset_returns[asset_index],
                    "market_return": market_return,
                    "regime": regime,
                }
            )

        returns_history[date_index] = asset_returns
        prices = close_prices

    data = pd.DataFrame.from_records(records)
    return data.sort_values(["date", "symbol"]).reset_index(drop=True)
