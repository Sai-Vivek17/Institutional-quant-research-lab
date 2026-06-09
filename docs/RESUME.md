# Resume And GitHub Positioning

## Resume Bullets

- Built an institutional-style equity statistical arbitrage research platform
  in Python, including synthetic market simulation, point-in-time factor
  engineering, purged walk-forward validation, market-neutral portfolio
  construction, and transaction-cost-aware backtesting.
- Implemented a cross-sectional ridge forecasting model with rolling retraining,
  recency-weighted samples, sector-neutral signal ranking, volatility-adjusted
  position sizing, exposure caps, and regime stress testing.
- Developed a reproducible quant research CLI that generates an HTML tear sheet
  with Sharpe, Sortino, drawdown, VaR, CVaR, turnover, rolling Sharpe, feature
  importance, and information coefficient diagnostics.
- Packaged the project with modular source code, config-driven experiments,
  unit tests, CI workflow, documentation, and one-command reproducibility.

## GitHub Description

Production-style quant research lab for cross-sectional equity alpha: synthetic
market data, factor engineering, purged walk-forward modeling, market-neutral
backtesting, costs, risk analytics, and HTML tear sheets.

## Interview Talking Points

- Why synthetic data was chosen: reproducibility, licensing safety, and complete
  control over known regimes and latent alpha structure.
- How lookahead bias is prevented: lagged features, next-day targets, rolling
  training windows, and purge gaps before test dates.
- Why the portfolio is market neutral: the goal is to evaluate stock-selection
  skill rather than market beta.
- Why transaction costs matter: strategies with attractive gross returns can
  fail after turnover and slippage.
- What you would improve next: real data adapter, borrow cost modeling, sector
  exposure constraints, risk model covariance optimizer, experiment tracking,
  and live paper-trading integration.

## Suggested Repository Tags

```text
quantitative-finance
algorithmic-trading
statistical-arbitrage
portfolio-optimization
backtesting
risk-management
python
```
