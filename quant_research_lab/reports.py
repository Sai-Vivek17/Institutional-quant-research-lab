"""HTML tear sheet generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from quant_research_lab.config import resolve_output_path


METRIC_FORMATS = {
    "cumulative_return": "{:.2%}",
    "annualized_return": "{:.2%}",
    "annualized_volatility": "{:.2%}",
    "sharpe": "{:.2f}",
    "sortino": "{:.2f}",
    "max_drawdown": "{:.2%}",
    "calmar": "{:.2f}",
    "hit_rate": "{:.2%}",
    "value_at_risk_95": "{:.2%}",
    "conditional_var_95": "{:.2%}",
    "average_turnover": "{:.2f}",
    "average_gross_exposure": "{:.2f}",
    "average_net_exposure": "{:.2f}",
    "average_daily_cost": "{:.3%}",
    "information_coefficient": "{:.4f}",
    "rank_ic": "{:.4f}",
}


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _format_metric(name: str, value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        formatter = METRIC_FORMATS.get(name, "{:.4f}")
        return formatter.format(value)
    return str(value)


def _plot_equity_curve(backtest_frame: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "equity_curve.png"
    plt.figure(figsize=(10.5, 4.8))
    plt.plot(backtest_frame["date"], backtest_frame["equity"], color="#0F766E", lw=2.2)
    plt.title("Equity Curve")
    plt.xlabel("")
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.25)
    _save_plot(path)
    return path


def _plot_drawdown(backtest_frame: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "drawdown.png"
    plt.figure(figsize=(10.5, 3.6))
    plt.fill_between(
        backtest_frame["date"],
        backtest_frame["drawdown"],
        0,
        color="#B91C1C",
        alpha=0.70,
    )
    plt.title("Drawdown")
    plt.xlabel("")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.25)
    _save_plot(path)
    return path


def _plot_rolling_sharpe(backtest_frame: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "rolling_sharpe.png"
    returns = backtest_frame.set_index("date")["net_return"]
    rolling_sharpe = (
        returns.rolling(63).mean() / returns.rolling(63).std() * (252**0.5)
    )
    plt.figure(figsize=(10.5, 3.6))
    plt.plot(rolling_sharpe.index, rolling_sharpe, color="#2563EB", lw=1.8)
    plt.axhline(0, color="#111827", lw=1, alpha=0.45)
    plt.title("Rolling 63-Day Sharpe")
    plt.xlabel("")
    plt.ylabel("Sharpe")
    plt.grid(alpha=0.25)
    _save_plot(path)
    return path


def _plot_regime_returns(regime_summary: pd.DataFrame, output_dir: Path) -> Path | None:
    if regime_summary.empty:
        return None

    path = output_dir / "regime_returns.png"
    plt.figure(figsize=(8.5, 4.0))
    sns.barplot(
        data=regime_summary,
        x="regime",
        y="annualized_return",
        color="#7C3AED",
    )
    plt.axhline(0, color="#111827", lw=1, alpha=0.45)
    plt.title("Annualized Return by Regime")
    plt.xlabel("")
    plt.ylabel("Annualized Return")
    plt.grid(axis="y", alpha=0.25)
    _save_plot(path)
    return path


def _plot_feature_importance(coefficients: pd.DataFrame, output_dir: Path) -> Path | None:
    if coefficients.empty:
        return None

    feature_columns = [
        column
        for column in coefficients.columns
        if column not in {"train_start", "train_end", "intercept"}
    ]
    if not feature_columns:
        return None

    importance = (
        coefficients[feature_columns]
        .abs()
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .reset_index()
    )
    importance.columns = ["feature", "mean_abs_coefficient"]

    path = output_dir / "feature_importance.png"
    plt.figure(figsize=(9.0, 4.8))
    sns.barplot(
        data=importance,
        y="feature",
        x="mean_abs_coefficient",
        color="#F59E0B",
    )
    plt.title("Average Absolute Ridge Coefficient")
    plt.xlabel("Mean absolute coefficient")
    plt.ylabel("")
    plt.grid(axis="x", alpha=0.25)
    _save_plot(path)
    return path


def _metrics_table(metrics: dict[str, Any]) -> str:
    rows = []
    preferred_order = [
        "days",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "value_at_risk_95",
        "conditional_var_95",
        "average_turnover",
        "average_gross_exposure",
        "average_net_exposure",
        "information_coefficient",
        "rank_ic",
    ]
    for key in preferred_order:
        if key not in metrics:
            continue
        label = key.replace("_", " ").title()
        value = _format_metric(key, metrics[key])
        rows.append(f"<tr><th>{label}</th><td>{value}</td></tr>")
    return "\n".join(rows)


def generate_tearsheet(
    backtest_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    metrics: dict[str, Any],
    regime_summary: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Path]:
    """Save plots, metrics, CSV artifacts, and a self-contained HTML report."""

    output_dir = resolve_output_path(config["report"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    backtest_path = output_dir / "daily_backtest.csv"
    predictions_path = output_dir / "oos_predictions.csv"
    coefficients_path = output_dir / "coefficient_history.csv"
    metrics_path = output_dir / "metrics.json"
    regime_path = output_dir / "regime_summary.csv"
    html_path = output_dir / "tear_sheet.html"

    backtest_frame.to_csv(backtest_path, index=False)
    predictions[
        [
            "date",
            "symbol",
            "sector",
            "prediction",
            "fwd_return_1d",
            "fwd_excess_return_1d",
        ]
    ].to_csv(predictions_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    regime_summary.to_csv(regime_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    chart_paths = {
        "equity_curve": _plot_equity_curve(backtest_frame, output_dir),
        "drawdown": _plot_drawdown(backtest_frame, output_dir),
        "rolling_sharpe": _plot_rolling_sharpe(backtest_frame, output_dir),
        "regime_returns": _plot_regime_returns(regime_summary, output_dir),
        "feature_importance": _plot_feature_importance(coefficients, output_dir),
    }

    title = config["report"].get("title", "Quant Research Tear Sheet")
    chart_sections = []
    for name, path in chart_paths.items():
        if path is None:
            continue
        label = name.replace("_", " ")
        chart_sections.append(
            f"""
        <section>
          <h2>{label.title()}</h2>
          <img src="{path.name}" alt="{label}">
        </section>
        """
        )
    chart_markup = "\n".join(chart_sections)
    regime_table = regime_summary.to_html(index=False, float_format=lambda x: f"{x:.4f}")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: #111827;
      background: #F8FAFC;
    }}
    header {{
      padding: 40px 7vw 28px;
      background: #0F172A;
      color: #F8FAFC;
    }}
    header p {{
      max-width: 920px;
      color: #CBD5E1;
      line-height: 1.55;
    }}
    main {{
      padding: 28px 7vw 48px;
    }}
    h1, h2 {{
      margin: 0 0 14px;
    }}
    section {{
      margin: 0 0 28px;
      padding: 22px;
      background: white;
      border: 1px solid #E5E7EB;
      border-radius: 8px;
    }}
    img {{
      width: 100%;
      max-width: 1120px;
      display: block;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #E5E7EB;
      text-align: left;
    }}
    th {{
      color: #475569;
      font-weight: 650;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>
      Transaction-cost-aware, market-neutral cross-sectional equity strategy.
      Signals are trained with purged walk-forward validation and evaluated
      across simulated market regimes to avoid the common notebook-only
      backtesting mistakes.
    </p>
  </header>
  <main>
    <div class="grid">
      <section>
        <h2>Headline Metrics</h2>
        <table>{_metrics_table(metrics)}</table>
      </section>
      <section>
        <h2>Regime Stress Test</h2>
        {regime_table}
      </section>
    </div>
    {chart_markup}
  </main>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")

    paths = {
        "html": html_path,
        "metrics": metrics_path,
        "backtest": backtest_path,
        "predictions": predictions_path,
        "coefficients": coefficients_path,
        "regime_summary": regime_path,
    }
    paths.update({key: value for key, value in chart_paths.items() if value is not None})
    return paths
