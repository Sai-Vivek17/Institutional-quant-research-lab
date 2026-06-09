"""Command-line interface for the quant research lab."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from quant_research_lab.pipeline import run_pipeline


def _add_run_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full data, feature, model, portfolio, and report pipeline.",
    )
    run_parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to a YAML config file.",
    )
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the report output directory.",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the synthetic data seed.",
    )
    run_parser.add_argument(
        "--no-report",
        action="store_true",
        help="Run the pipeline without generating the HTML tear sheet.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quant-lab",
        description="Institutional-style quant research and backtesting pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    args = parser.parse_args(argv)

    if args.command == "run":
        overrides: dict[str, object] = {}
        if args.output_dir:
            overrides.setdefault("report", {})["output_dir"] = args.output_dir
        if args.seed is not None:
            overrides.setdefault("data", {})["seed"] = args.seed
        if args.no_report:
            overrides.setdefault("report", {})["enabled"] = False

        result = run_pipeline(Path(args.config), overrides)
        metrics = result.metrics
        print("Pipeline complete")
        print(f"Days: {metrics['days']:,}")
        print(f"Sharpe: {metrics['sharpe']:.2f}")
        print(f"Annualized return: {metrics['annualized_return']:.2%}")
        print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Rank IC: {metrics['rank_ic']:.4f}")
        if result.artifact_paths.get("html"):
            print(f"Tear sheet: {result.artifact_paths['html']}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
