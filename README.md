# Regime Classifier

A research & backtesting framework for the futures/options market regime classification strategy described in the accompanying research paper.

## Overview

This framework implements the quantifiable components of a discretionary trading strategy for NQ (Nasdaq futures) using:
- **Options-derived metrics**: GEX, DEX, CHEX, gamma flip, call/put walls
- **Volume profile analysis**: POC, VAH/VAL, HVN/LVN
- **VWAP with standard deviation bands**
- **Volatility modeling**: Realized volatility (RV), IV expected moves, Z-scores
- **Order flow**: CVD, absorption, exhaustion, delta divergence
- **Black-Scholes Greeks**: Delta, Gamma, Theta, Vega, Charm

## Project Structure

```
regime-classifier/
├── config/           # YAML configuration (all thresholds, no magic numbers)
├── data/             # Data loaders (NQ, QQQ, options chain, footprint)
├── features/         # Feature engineering (volume profile, VWAP, RV/IV, Greeks, conversion)
├── regime/           # Pluggable regime classifiers
├── signals/          # Entry/exit signal generation
├── backtest/         # Vectorized + event-driven backtester, walk-forward, bootstrap
├── reports/          # Tearsheets: equity curve, drawdowns, win rate, Monte Carlo
└── tests/            # Unit tests
```

## Installation

```bash
pip install -e .
```

## Configuration

All parameters live in `config/strategy.yaml`. No hardcoded thresholds in code.

## Usage

```bash
# Run backtest
python -m regime_classifier.backtest --config config/strategy.yaml

# Generate tearsheet
python -m regime_classifier.reports --results results/backtest.json
```

## Data Requirements

You must provide:
- NQ futures OHLCV (1m or tick)
- QQQ options chain (OI, IV, Greeks per strike per expiry)
- Footprint / order flow data (bid/ask volume per price per bar)

Recommended sources: Databento, Cboe, ORATS, Bookmap, Sierra Chart, Rithmic.

## License

MIT