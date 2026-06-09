"""Configuration helpers for the quant research pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "Institutional Quant Research Lab",
        "description": "Cross-sectional equity alpha research and backtesting.",
    },
    "data": {
        "start": "2018-01-01",
        "periods": 1250,
        "assets": 80,
        "seed": 42,
        "sectors": [
            "Technology",
            "Financials",
            "Healthcare",
            "Industrials",
            "Consumer",
            "Energy",
            "Utilities",
            "Materials",
        ],
    },
    "features": {
        "winsorize_zscore": 5.0,
    },
    "model": {
        "target": "fwd_excess_return_1d",
        "train_window": 252,
        "purge_days": 2,
        "retrain_every": 5,
        "ridge_lambda": 25.0,
        "half_life_days": 126,
        "min_train_observations": 2500,
    },
    "portfolio": {
        "selection_quantile": 0.20,
        "gross_leverage": 1.0,
        "max_abs_weight": 0.05,
        "volatility_floor": 0.008,
        "cost_bps": 1.5,
        "slippage_bps": 1.0,
        "sector_neutralize": True,
    },
    "report": {
        "enabled": True,
        "output_dir": "reports/generated",
        "title": "Equity Statistical Arbitrage Tear Sheet",
    },
    "output": {
        "save_intermediate": True,
        "data_dir": "data/processed",
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating the original input."""

    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and merge it into the project defaults."""

    if config_path is None:
        return deepcopy(DEFAULT_CONFIG)

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file) or {}
    return deep_update(DEFAULT_CONFIG, user_config)


def resolve_output_path(path: str | Path) -> Path:
    """Resolve a path relative to the current working directory."""

    output_path = Path(path)
    if output_path.is_absolute():
        return output_path
    return Path.cwd() / output_path


def ensure_output_directories(config: dict[str, Any]) -> None:
    """Create output directories used by the pipeline."""

    data_dir = resolve_output_path(config["output"]["data_dir"])
    report_dir = resolve_output_path(config["report"]["output_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
