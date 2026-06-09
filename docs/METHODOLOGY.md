# Methodology

## Objective

Build a realistic quant research workflow that demonstrates three skills at
once: alpha research, portfolio/risk thinking, and production-quality Python.

The strategy is a cross-sectional equity statistical arbitrage prototype. Each
day, it predicts next-day excess returns for a universe of synthetic equities,
then holds a market-neutral long/short portfolio.

## Data Design

The data generator produces OHLCV bars for a configurable equity universe. It
includes:

- Market regimes: bull, sideways, bear, and stress.
- Sector-specific shocks and asset-specific market/sector betas.
- Idiosyncratic volatility and liquidity differences.
- Fat-tail stress jumps.
- Latent momentum, short-term reversal, quality, and value effects.

Synthetic data is used to make the project fully runnable without paid datasets
or vendor licensing constraints. The point is not to claim live-trading alpha;
the point is to show a clean research process.

## Feature Engineering

Features are generated from lagged price and volume information:

- Short and medium-term returns.
- Momentum excluding the most recent week.
- Short-term reversal.
- Realized volatility.
- Intraday range.
- Liquidity and volume shock.
- Fundamental-style quality and value scores.
- Distance from 63-day high.
- Rolling market beta.

Features are cross-sectionally z-scored by date. The target is next-day
cross-sectional excess return, which reduces market direction dependence.

## Validation

The model uses purged walk-forward validation:

- Train only on historical dates.
- Leave a configurable purge gap before each test date.
- Retrain on a rolling window.
- Save only out-of-sample predictions.

This avoids the most common resume-project error: training and evaluating on
overlapping or leaky data.

## Model

The baseline model is a ridge regression implemented directly with NumPy. That
keeps the project dependency-light while showing the math clearly:

```text
beta = inv(X'WX + lambda I) X'Wy
```

Recent observations receive higher sample weights using a half-life decay.

## Portfolio Construction

For each date:

- Sector-neutralize forecast scores.
- Rank stocks cross-sectionally.
- Buy the top quantile and short the bottom quantile.
- Risk-adjust raw scores by recent volatility.
- Normalize to equal long and short gross exposure.
- Apply position caps.

The backtester then applies turnover-driven transaction costs and slippage.

## Risk Reporting

The tear sheet includes:

- Equity curve and drawdown.
- Sharpe, Sortino, Calmar, hit rate, max drawdown.
- VaR and conditional VaR.
- Average turnover, gross exposure, net exposure, and daily costs.
- Rolling Sharpe.
- Feature coefficient importance.
- Return breakdown by market regime.
- Information coefficient and rank IC.
