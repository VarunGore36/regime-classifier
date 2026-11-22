# Usage Guide

## Installation

```bash
# Clone repository
git clone https://github.com/VarunGore36/regime-classifier.git
cd regime-classifier

# Install in development mode
pip install -e ".[dev]"

# Or install with notebook support
pip install -e ".[notebook]"
```

## Quick Start

### 1. Generate Synthetic Data & Run Backtest

```bash
# Run backtest with synthetic data
python main.py backtest --synthetic --output reports/backtest.pdf

# Run walk-forward validation
python main.py walkforward --synthetic --output reports/walkforward.pdf

# Run bootstrap validation
python main.py bootstrap --synthetic --n-paths 1000
```

### 2. Use with Real Data

Prepare your data as Parquet files:

```
data/
├── nq_1min.parquet      # NQ futures OHLCV
├── qqq_1min.parquet     # QQQ ETF OHLCV
├── options_chain.parquet # Options chain snapshots
└── footprint.parquet    # Footprint/order flow
```

Required columns:

```python
# NQ/QQQ bars
nq_bars = pd.DataFrame({
    "timestamp": pd.DatetimeIndex(...),  # tz-aware
    "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...
})

# Options chain
options = pd.DataFrame({
    "timestamp": ...,      # snapshot time
    "expiry": ...,         # expiration date
    "strike": ...,         # strike price
    "right": "C" or "P",   # call/put
    "oi": ...,             # open interest
    "volume": ...,         # volume
    "iv": ...,             # implied volatility
    "delta": ..., "gamma": ..., "theta": ..., "vega": ...,
    "underlying_price": ... # spot at snapshot
})

# Footprint
footprint = pd.DataFrame({
    "timestamp": ...,      # bar timestamp
    "price": ...,          # price level
    "bid_volume": ...,     # aggressive sell volume
    "ask_volume": ...      # aggressive buy volume
})
```

Run with custom config:

```bash
python main.py backtest --config config/strategy.yaml --output reports/backtest.pdf
```

## Python API

### Basic Backtest

```python
from regime_classifier import (
    BacktestEngine, SignalGenerator,
    NQDataLoader, QQQDataLoader, OptionsChainLoader, FootprintLoader,
    load_config
)

# Load config
config = load_config("config/strategy.yaml")

# Load data
nq_loader = NQDataLoader("data/nq_1min.parquet")
qqq_loader = QQQDataLoader("data/qqq_1min.parquet")
opt_loader = OptionsChainLoader("data/options_chain.parquet")
fp_loader = FootprintLoader("data/footprint.parquet")

start = pd.Timestamp("2024-01-01", tz="America/New_York")
end = pd.Timestamp("2024-12-31", tz="America/New_York")

nq_bars = nq_loader.load_bars(start, end, "1min")
qqq_bars = qqq_loader.load_bars(start, end, "1min")
options = opt_loader.load_chain(start, end)
footprint = fp_loader.load_footprint(start, end)

# Run backtest
engine = BacktestEngine(config.to_dict())
signal_gen = SignalGenerator(config.to_dict())

result = engine.run(nq_bars, qqq_bars, options, footprint, signal_gen)

# View metrics
print(result.metrics)

# Generate report
from regime_classifier import generate_report
generate_report(result, "reports/backtest.pdf")
```

### Feature Engineering Only

```python
from regime_classifier import (
    calculate_volume_profile, calculate_vwap,
    calculate_realized_volatility, calculate_cvd,
    detect_absorption, detect_exhaustion,
    calculate_conversion_ratio, calculate_greeks_for_chain
)

# Volume Profile
vol_profile = calculate_volume_profile(nq_bars)
print(f"POC: {vol_profile.poc}, VAH: {vol_profile.vah}, VAL: {vol_profile.val}")

# VWAP
vwap = calculate_vwap(nq_bars)
print(f"VWAP: {vwap.vwap.iloc[-1]}")

# Realized Volatility
rv = calculate_realized_volatility(nq_bars)
print(f"RV Z-Score: {rv.rv_zscore.iloc[-1]}, Regime: {rv.regime.iloc[-1]}")

# CVD from footprint
cvd = calculate_cvd(footprint, nq_bars)
print(f"CVD: {cvd.cvd.iloc[-1]}")

# Absorption/Exhaustion
absorp = detect_absorption(footprint)
exh = detect_exhaustion(footprint)
print(f"Absorption bullish: {absorp.absorption_bullish.sum()}")
print(f"Exhaustion bullish: {exh.exhaustion_bullish.sum()}")
```

### Regime Classification

```python
from regime_classifier import CombinedRegimeClassifier

classifier = CombinedRegimeClassifier()

# Prepare data with GEX/DEX
regime_data = nq_bars[["timestamp", "close"]].copy()
regime_data["gex_net"] = ...  # From options
regime_data["dex_net"] = ...  # From options
regime_data["gamma_flip"] = ...  # From options
regime_data["underlying_price"] = nq_bars["close"]
regime_data["iv"] = ...  # From IV features
regime_data["rv"] = ...  # From RV features

result = classifier.classify(regime_data)

print(f"Primary Regime: {result.primary_regime.iloc[-1]}")
print(f"Bias: {result.bias.iloc[-1]}")
print(f"Score: {result.regime_score.iloc[-1]}")
print(f"Sub-regimes: { {k: v.iloc[-1] for k, v in result.sub_regimes.items()} }")
```

### Signal Generation

```python
from regime_classifier import SignalGenerator

generator = SignalGenerator(config.to_dict())
signals = generator.generate(nq_bars, qqq_bars, options, footprint)

# Check entries
entries = signals.entry
longs = entries.long_entry.sum()
shorts = entries.short_entry.sum()
print(f"Long entries: {longs}, Short entries: {shorts}")

# Check exits
exits = signals.exit
print(f"Stop loss exits: {(exits.exit_reason == 'stop_loss_long').sum() + (exits.exit_reason == 'stop_loss_short').sum()}")

# Position & Equity
position = signals.position
equity = signals.equity_curve
print(f"Final equity: ${equity.iloc[-1]:.2f}")
```

## Configuration

All parameters in `config/strategy.yaml`. Key sections:

```yaml
# Data
data:
  nq_symbol: "NQ"
  qqq_symbol: "QQQ"
  session_open: "09:30"
  session_close: "16:00"

# Volume Profile
volume_profile:
  va_percentile: 0.70
  hvn_threshold: 1.5

# VWAP
vwap:
  bands: [1.0, 2.0, 3.0]

# Realized Vol
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
    volume_surge_mult: 2.0
  fade_trades:
    enabled: false

# Backtest
backtest:
  initial_capital: 100000
  commission_per_contract: 2.50
  slippage_ticks: 1
  contract_multiplier: 20
```

## Walk-Forward Analysis

```python
from regime_classifier import run_walkforward_backtest, BacktestEngine

engine = BacktestEngine(config.to_dict())

results = run_walkforward_backtest(
    engine, nq_bars, qqq_bars, options, footprint,
    train_months=6,
    test_months=1,
    step_months=1
)

for i, r in enumerate(results):
    print(f"Period {i+1}: Return={r.metrics['total_return_pct']:.2f}%, "
          f"Trades={r.metrics['total_trades']}, "
          f"Sharpe={r.metrics['sharpe_ratio']:.2f}")
```

## Bootstrap Validation

```python
from regime_classifier import bootstrap_backtest

result = bootstrap_backtest(
    engine, nq_bars, qqq_bars, options, footprint,
    n_paths=1000,
    block_size=20,
    confidence=0.95
)

print(f"Profit Probability: {result.profit_probability:.2%}")
print(f"P-value: {result.p_value:.4f}")
print(f"95% CI: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]")
```

## Ablation Testing

```python
from regime_classifier import ablation_test

def disable_poc(config):
    config = config.copy()
    config["poc_reversion"] = {"enabled": False}
    return config

def disable_vwap(config):
    config = config.copy()
    config["vwap_bands"] = {"enabled": False}
    return config

results = ablation_test(engine, nq_bars, qqq_bars, options, footprint, {
    "poc_reversion": disable_poc,
    "vwap_bands": disable_vwap
})

for name, metrics in results.items():
    print(f"{name}: Return={metrics['total_return_pct']:.2f}%, "
          f"WinRate={metrics['win_rate_pct']:.1f}%")
```

## Reports

```python
from regime_classifier import create_tearsheet, generate_report

# Generate full PDF report
generate_report(result, "reports/full_report.pdf")

# Or just create tearsheet figure
fig = create_tearsheet(result, TearsheetConfig(save_path="reports/tearsheet.pdf"))
```

## CLI Commands

```bash
# Generate synthetic data
regime-classifier synthetic --symbol NQ --start-date 2024-01-01 --end-date 2024-12-31 --output data/nq_synthetic.parquet

# Run backtest
regime-classifier backtest --synthetic --config config/strategy.yaml --output reports/backtest.pdf

# Walk-forward
regime-classifier walkforward --synthetic --train-months 6 --test-months 1 --step-months 1

# Bootstrap
regime-classifier bootstrap --synthetic --n-paths 1000 --block-size 20
```

## Data Providers

Recommended data sources:

| Data Type | Providers |
|-----------|-----------|
| Futures OHLCV | Databento, CME, IQFeed, Rithmic, Sierra Chart |
| Options Chain | Databento, ORATS, Cboe, IVolatility, Barchart |
| Footprint/Order Flow | Bookmap, Sierra Chart, Rithmic, Quantower, Exocharts |

## Common Issues

### Timezone Handling
All timestamps must be timezone-aware (preferably `America/New_York`):

```python
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("America/New_York")
# Or if already UTC:
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
```

### Memory Optimization
For large datasets:
- Use `pd.read_parquet(..., columns=[...])` to load only needed columns
- Process in chunks: `for chunk in pd.read_parquet(..., chunksize=100000)`
- Use `dtype` parameter to reduce memory

### Missing Data
The framework expects complete 1-minute bars. Fill gaps:

```python
# Resample to 1min and forward fill
df = df.set_index("timestamp").resample("1min").ffill().reset_index()
```