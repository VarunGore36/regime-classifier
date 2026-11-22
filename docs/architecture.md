# Architecture Overview

## System Components

```
regime-classifier/
├── config/           # Configuration management (YAML-based)
├── data/             # Data loaders (NQ, QQQ, Options, Footprint)
├── features/         # Feature engineering modules
├── regime/           # Regime classification engines
├── signals/          # Entry/exit signal generation
├── backtest/         # Backtesting engine & validation
├── reports/          # Tearsheet & report generation
└── tests/            # Unit & integration tests
```

## Data Flow

```
Raw Data (NQ, QQQ, Options, Footprint)
         │
         ▼
Data Loaders (data/)
         │
         ▼
Features (features/)
  ├── Volume Profile
  ├── VWAP + Bands
  ├── Realized Volatility
  ├── Implied Volatility
  ├── Greeks (BS/Black76)
  ├── NQ/QQQ Conversion
  ├── CVD/Delta
  └── Absorption/Exhaustion
         │
         ▼
Regime Classification (regime/)
  ├── RV Z-Score
  ├── GEX Sign
  ├── DEX Sign
  ├── Entropy
  └── Combined (Weighted)
         │
         ▼
Signal Generation (signals/)
  ├── POC Reversion
  ├── VWAP Bands
  ├── Break/Fade
  ├── Stop Loss
  ├── Take Profit
  ├── Trailing Stop
  └── Time/Regime Exit
         │
         ▼
Backtest Engine (backtest/)
  ├── Vectorized Backtest
  ├── Walk-Forward
  ├── Bootstrap
  ├── Monte Carlo
  └── Ablation
         │
         ▼
Reports (reports/)
  ├── Equity Curve
  ├── Drawdown
  ├── Monthly Returns
  ├── Trade Analysis
  └── Metrics Table
```

## Key Design Decisions

### 1. Configuration-Driven
All parameters in `config/strategy.yaml` - no magic numbers in code.

### 2. Modular Features
Each feature is independent and testable. Can be used standalone.

### 3. Pluggable Regime Classifiers
Composite pattern allows mixing different regime signals with weights.

### 4. Vectorized Backtest
Uses pandas for speed. Event-driven fallback for complex logic.

### 5. Data Abstraction
Loaders implement common interface. Easy to swap data sources.

## Configuration

All strategy parameters in `config/strategy.yaml`:

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
    lookback_bars: 20
    stop_ticks: 20
    target_ticks: 40
```

## Extending the Framework

### Add New Feature
1. Create `features/new_feature.py`
2. Export in `features/__init__.py`
3. Add config section in `config/strategy.yaml`
4. Write tests in `tests/test_features.py`

### Add New Regime Classifier
1. Subclass `RegimeClassifier` in `regime/`
2. Implement `classify(data)` method
3. Add to `CombinedRegimeClassifier` weights

### Add New Signal
1. Create signal class in `signals/entries.py` or `exits.py`
2. Follow `EntrySignal`/`ExitSignal` pattern
3. Integrate in `SignalGenerator.generate()`

## Data Requirements

Minimum required data for full backtest:

| Data | Frequency | Columns |
|------|-----------|---------|
| NQ Futures | 1-min | timestamp, open, high, low, close, volume |
| QQQ ETF | 1-min | timestamp, open, high, low, close, volume |
| Options Chain | Per snapshot | timestamp, expiry, strike, right, oi, volume, iv, delta, gamma, theta, vega, underlying_price |
| Footprint | Per bar | timestamp, price, bid_volume, ask_volume |

## Performance Considerations

- Vectorized operations preferred over loops
- Use `pd.DataFrame.eval()` for complex expressions
- Cache expensive computations (greeks, volume profile)
- Profile with `python -m cProfile` before optimizing

## Testing Strategy

- Unit tests for each feature (`tests/test_features.py`)
- Regime classifier tests (`tests/test_regime.py`)
- Signal tests (`tests/test_signals.py`)
- Backtest integration tests (`tests/test_backtest.py`)
- Run with: `pytest tests/ -v --cov=regime_classifier`