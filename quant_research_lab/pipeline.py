"""End-to-end quant research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_lab.backtest import attach_regimes, run_backtest
from quant_research_lab.config import (
    deep_update,
    ensure_output_directories,
    load_config,
    resolve_output_path,
)
from quant_research_lab.data import simulate_market_data
from quant_research_lab.features import build_features
from quant_research_lab.metrics import (
    evaluate_performance,
    prediction_diagnostics,
    summarize_by_regime,
)
from quant_research_lab.models import walk_forward_predictions
from quant_research_lab.portfolio import construct_portfolio
from quant_research_lab.reports import generate_tearsheet


@dataclass
class PipelineResult:
    config: dict[str, Any]
    market_data: pd.DataFrame
    features: pd.DataFrame
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    positions: pd.DataFrame
    backtest: pd.DataFrame
    metrics: dict[str, Any]
    regime_summary: pd.DataFrame
    artifact_paths: dict[str, Path]


def _save_intermediate_outputs(result: PipelineResult) -> None:
    data_dir = resolve_output_path(result.config["output"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    result.market_data.to_csv(data_dir / "market_data.csv", index=False)
    result.features.to_csv(data_dir / "features.csv", index=False)
    result.positions.to_csv(data_dir / "positions.csv", index=False)


def run_pipeline(
    config_path: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> PipelineResult:
    """Run the complete research, forecast, portfolio, and reporting flow."""

    config = load_config(config_path)
    if config_overrides:
        config = deep_update(config, config_overrides)
    ensure_output_directories(config)

    market_data = simulate_market_data(config)
    features = build_features(market_data, config)
    predictions, coefficients = walk_forward_predictions(features, config)
    positions = construct_portfolio(predictions, config)
    backtest = attach_regimes(run_backtest(positions, config), market_data)

    metrics = evaluate_performance(backtest)
    metrics.update(prediction_diagnostics(predictions))
    regime_summary = summarize_by_regime(backtest)

    artifact_paths: dict[str, Path] = {}
    if config["report"].get("enabled", True):
        artifact_paths = generate_tearsheet(
            backtest_frame=backtest,
            predictions=predictions,
            coefficients=coefficients,
            metrics=metrics,
            regime_summary=regime_summary,
            config=config,
        )

    result = PipelineResult(
        config=config,
        market_data=market_data,
        features=features,
        predictions=predictions,
        coefficients=coefficients,
        positions=positions,
        backtest=backtest,
        metrics=metrics,
        regime_summary=regime_summary,
        artifact_paths=artifact_paths,
    )

    if config["output"].get("save_intermediate", True):
        _save_intermediate_outputs(result)

    return result
