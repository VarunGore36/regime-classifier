"""Pytest configuration."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_bars():
    """Create sample OHLCV bars."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01 09:30", periods=500, freq="1min", tz="America/New_York")
    returns = np.random.normal(0, 0.001, 500)
    prices = 15000 * np.exp(np.cumsum(returns))

    noise = np.random.normal(0, 0.0005, (500, 3))
    opens = prices * (1 + noise[:, 0])
    highs = np.maximum(opens, prices) * (1 + np.abs(noise[:, 1]))
    lows = np.minimum(opens, prices) * (1 - np.abs(noise[:, 2]))
    volumes = np.random.randint(100, 5000, 500)

    return pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes
    })


@pytest.fixture
def sample_footprint(sample_bars):
    """Create sample footprint from bars."""
    rows = []
    for _, bar in sample_bars.iterrows():
        n_levels = 15
        prices = np.linspace(bar["low"], bar["high"], n_levels)
        for p in prices:
            bid_vol = np.random.randint(0, 50)
            ask_vol = np.random.randint(0, 50)
            rows.append({
                "timestamp": bar["timestamp"],
                "price": p,
                "bid_volume": bid_vol,
                "ask_volume": ask_vol
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_options_chain():
    """Create sample options chain."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01 09:30", periods=10, freq="30min", tz="America/New_York")
    expiries = pd.date_range("2024-01-01", periods=4, freq="W-THU")

    rows = []
    for ts in dates:
        for exp in expiries:
            for strike in np.arange(14800, 15200, 50):
                for right in ["C", "P"]:
                    rows.append({
                        "timestamp": ts,
                        "expiry": exp,
                        "strike": strike,
                        "right": right,
                        "oi": np.random.randint(0, 5000),
                        "volume": np.random.randint(0, 1000),
                        "iv": 0.15 + np.random.normal(0, 0.02),
                        "delta": np.random.normal(0, 0.3),
                        "gamma": np.random.uniform(0, 0.02),
                        "theta": np.random.normal(-5, 2),
                        "vega": np.random.uniform(5, 20),
                        "underlying_price": 15000 + np.random.normal(0, 20)
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def regime_data():
    """Create sample data for regime classification."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2024-01-01 09:30", periods=n, freq="1min", tz="America/New_York")

    returns = np.random.normal(0, 0.001, n)
    vol_regime = np.random.choice([0.5, 1.0, 2.0], n, p=[0.3, 0.5, 0.2])
    returns = returns * np.repeat(vol_regime, n // 3 + 1)[:n]

    prices = 15000 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        "timestamp": dates,
        "close": prices,
        "gex_net": np.random.normal(0, 1e6, n),
        "dex_net": np.random.normal(0, 1e6, n),
        "gamma_flip": 15000 + np.random.normal(0, 50, n),
        "underlying_price": prices,
        "iv": 0.15 + np.random.normal(0, 0.02, n),
        "rv": 0.15 + np.random.normal(0, 0.02, n)
    })