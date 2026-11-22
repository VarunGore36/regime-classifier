# Regime Classifier

A research & backtesting framework for the futures/options market regime classification strategy.

## Features

### Data Layer
- **Multi-source data providers**: Databento, IBKR, Polygon.io, CBOE, Local files
- **Synthetic data generator** for testing
- **Data validation** with comprehensive checks
- **Timezone-aware** timestamp handling

### Feature Engineering
- **Volume Profile**: POC, VAH/VAL, HVN/LVN, Ledges, TPO profiles
- **VWAP**: Session-anchored & rolling, standard deviation bands (1σ, 2σ, 3σ)
- **Realized Volatility**: Log returns, RV Z-scores, volatility acceleration, Parkinson/Garman-Klass estimators
- **Implied Volatility**: Expected moves, IV percentile/rank, IV/RV ratio (VRP)
- **Black-Scholes Greeks**: Delta, Gamma, Theta, Vega, Rho, Charm, Vanna, Vomma
- **Black-76**: Futures options Greeks
- **NQ/QQQ Conversion**: Dynamic ratio, key level mapping
- **CVD/Delta**: Cumulative volume delta, divergence detection
- **Order Flow**: Absorption, Exhaustion, Trapped Traders, Delta Divergence
- **Entropy Models**: Shannon, Permutation, Sample, Approximate, Multiscale entropy
- **IV Skew**: 25Δ/10Δ risk reversals, butterflies, term structure slope/curvature
- **GARCH**: GARCH(1,1), EGARCH, GJR-GARCH conditional volatility
- **Shadow Gamma**: Hidden dealer gamma estimation

### Regime Classification
- **RV Z-Score**: Volatility environment (low/normal/elevated/very elevated)
- **GEX Sign**: Dealer gamma positioning (positive/negative/neutral)
- **DEX Sign**: Dealer delta positioning (long/short/neutral)
- **Entropy**: Market organization (low/high/normal)
- **IV Skew**: Options market sentiment
- **GARCH**: Volatility dynamics
- **Shadow Gamma**: Hidden dealer positioning
- **Composite**: Weighted combination with configurable weights

### Signal Generation
- **POC Reversion**: Mean reversion to Point of Control (81% win rate per paper)
- **VWAP Bands**: Reversion & breakout at 1σ/2σ/3σ (91% win rate at session open per paper)
- **Break/Fade**: Volume/delta confirmed breakouts vs fades
- **Entropy Signals**: Trend following in low entropy, reversion in high entropy
- **IV Skew Signals**: Contrarian from steep put skew, complacency from flat skew
- **GARCH Signals**: Vol contraction/expansion, extreme residuals, low vol breakouts
- **Shadow Gamma Signals**: Dealer hedging pressure, gamma concentration breakouts

### Exits
- Stop Loss (fixed ticks)
- Take Profit (fixed ticks)
- Trailing Stop
- Time-based (session end)
- Regime Change

### Backtesting & Validation
- **Vectorized Engine**: Fast pandas-based backtest
- **Walk-Forward**: Train/test splits with configurable windows
- **Bootstrap**: Stationary bootstrap (Politis & Romano) for profit probability
- **Monte Carlo**: Trade resampling for equity curve distributions
- **Ablation Testing**: Component importance analysis
- **Metrics**: Sharpe, Sortino, Calmar, Profit Factor, Expectancy, Max DD

### Reporting
- **Tearsheets**: Equity curve, drawdown, underwater, monthly heatmap
- **Trade Analysis**: P&L by hour/day/signal, streaks, MAE/MFE
- **PDF Reports**: Auto-generated with matplotlib

## Installation

```bash
git clone https://github.com/VarunGore36/regime-classifier.git
cd regime-classifier
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run backtest with synthetic data
python main.py backtest --synthetic --output reports/backtest.pdf

# Run walk-forward analysis
python main.py walkforward --synthetic --output reports/walkforward.pdf

# Run bootstrap validation
python main.py bootstrap --synthetic --n-paths 1000
```

## Python API

```python
from regime_classifier import (
    run_pipeline, RegimePipeline,
    BacktestEngine, SignalGenerator,
    SyntheticDataLoader, load_config
)

# Quick pipeline run
result = run_pipeline(
    provider="synthetic",
    start="2024-01-01",
    end="2024-12-31",
    output_dir="reports"
)

# Or with full control
pipeline = RegimePipeline("config/strategy.yaml")
result = pipeline.run_synthetic(
    start=pd.Timestamp("2024-01-01", tz="America/New_York"),
    end=pd.Timestamp("2024-12-31", tz="America/New_York")
)

# Access results
print(result.backtest_result.metrics)
print(result.regime_result.primary_regime.iloc[-1])
```

## Configuration

All parameters in `config/strategy.yaml`:

```yaml
# Volume Profile
volume_profile:
  va_percentile: 0.70
  hvn_threshold: 1.5
  lvn_threshold: 0.3

# VWAP
vwap:
  bands: [1.0, 2.0, 3.0]

# Realized Volatility
realized_vol:
  window_minutes: 30
  zscore_windows: [20, 60, 240]

# Signals
signals:
  poc_reversion:
    enabled: true
    stop_ticks: 20
    target_ticks: 40
  vwap_bands:
    enabled: true
    band: 3.0
  break_trades:
    enabled: true
  fade_trades:
    enabled: false

# Backtest
backtest:
  initial_capital: 100000
  commission_per_contract: 2.50
  slippage_ticks: 1
```

## Data Requirements

| Data | Frequency | Columns |
|------|-----------|---------|
| NQ Futures | 1-min | timestamp, open, high, low, close, volume |
| QQQ ETF | 1-min | timestamp, open, high, low, close, volume |
| Options Chain | Per snapshot | timestamp, expiry, strike, right, oi, volume, iv, delta, gamma, theta, vega, underlying_price |
| Footprint | Per bar | timestamp, price, bid_volume, ask_volume |

## Project Structure

```
regime-classifier/
├── config/           # YAML configuration
├── data/             # Data loaders & providers
├── features/         # Feature engineering
├── regime/           # Regime classifiers
├── signals/          # Entry/exit signals
├── backtest/         # Backtesting engine
├── reports/          # Tearsheet generation
├── utils/            # Utility functions
├── validation/       # Data validation
├── pipeline/         # End-to-end pipeline
├── tests/            # Unit & integration tests
├── examples/         # Jupyter notebooks
└── docs/             # Documentation
```

## Documentation

- [Architecture](docs/architecture.md)
- [Usage Guide](docs/usage.md)

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=regime_classifier --cov-report=html
```

## License

MIT