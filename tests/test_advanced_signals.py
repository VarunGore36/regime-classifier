"""Tests for advanced signals."""

import pytest
import pandas as pd
import numpy as np

from signals import (
    EntropySignalGenerator, IVSkewSignalGenerator,
    GARCHSignalGenerator, ShadowGammaSignalGenerator
)
from features import get_entropy_features, calculate_iv_skew, calculate_term_structure
from features import calculate_garch_volatility, get_garch_signals
from features import calculate_shadow_gamma, get_shadow_gamma_signals
from regime import CompositeAdvancedRegimeClassifier


def create_sample_bars(n=300):
    """Create sample OHLCV bars."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01 09:30", periods=n, freq="1min", tz="America/New_York")
    returns = np.random.normal(0, 0.001, n)
    prices = 15000 * np.exp(np.cumsum(returns))

    noise = np.random.normal(0, 0.0005, (n, 3))
    opens = prices * (1 + noise[:, 0])
    highs = np.maximum(opens, prices) * (1 + np.abs(noise[:, 1]))
    lows = np.minimum(opens, prices) * (1 - np.abs(noise[:, 2]))
    volumes = np.random.randint(100, 5000, n)

    return pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes
    })


def create_sample_returns(bars):
    """Create returns from bars."""
    return np.log(bars["close"] / bars["close"].shift(1)).dropna()


def create_sample_options_chain():
    """Create sample options chain."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01 09:30", periods=5, freq="30min", tz="America/New_York")
    expiries = pd.date_range("2024-01-01", periods=3, freq="W-THU")
    spot = 15000

    rows = []
    for ts in timestamps:
        for exp in expiries:
            for strike in np.arange(spot - 300, spot + 300, 50):
                for right in ["C", "P"]:
                    dte = (exp - ts).days
                    delta = 0.5 if right == "C" else -0.5
                    if strike > spot:
                        delta = 0.5 - 0.3 * (strike - spot) / 300 if right == "C" else -0.5 + 0.3 * (strike - spot) / 300
                    else:
                        delta = 0.5 + 0.3 * (spot - strike) / 300 if right == "C" else -0.5 - 0.3 * (spot - strike) / 300
                    delta = np.clip(delta, -0.99, 0.99)

                    rows.append({
                        "timestamp": ts,
                        "expiry": exp,
                        "strike": strike,
                        "right": right,
                        "oi": np.random.randint(0, 5000),
                        "volume": np.random.randint(0, 1000),
                        "iv": 0.15 + np.random.normal(0, 0.02),
                        "delta": delta,
                        "gamma": np.random.uniform(0, 0.02),
                        "theta": np.random.normal(-5, 2),
                        "vega": np.random.uniform(5, 20),
                        "underlying_price": spot + np.random.normal(0, 10),
                        "dte": dte
                    })
    return pd.DataFrame(rows)


def create_advanced_regime_result(bars):
    """Create mock advanced regime result."""
    n = len(bars)
    return type('RegimeResult', (), {
        'regime': pd.Series(['neutral'] * n, index=bars.index),
        'confidence': pd.Series([0.7] * n, index=bars.index),
        'components': {}
    })()


class TestEntropySignalGenerator:
    def test_generate(self):
        bars = create_sample_bars(200)
        returns = create_sample_returns(bars)
        entropy_features = get_entropy_features(returns)
        regime = create_advanced_regime_result(bars)

        signal_gen = EntropySignalGenerator()
        result = signal_gen.generate(bars, entropy_features, regime)

        assert hasattr(result, 'long_entry')
        assert hasattr(result, 'short_entry')
        assert hasattr(result, 'signal_type')
        assert result.long_entry.dtype == bool
        assert result.short_entry.dtype == bool

    def test_disabled(self):
        bars = create_sample_bars(100)
        returns = create_sample_returns(bars)
        entropy_features = get_entropy_features(returns)
        regime = create_advanced_regime_result(bars)

        signal_gen = EntropySignalGenerator(config={"enabled": False})
        result = signal_gen.generate(bars, entropy_features, regime)

        assert not result.long_entry.any()
        assert not result.short_entry.any()


class TestIVSkewSignalGenerator:
    def test_generate(self):
        bars = create_sample_bars(200)
        options = create_sample_options_chain()
        underlying = bars.set_index("timestamp")["close"]

        skew_result = calculate_iv_skew(options, underlying)
        term_result = calculate_term_structure(options)
        skew_signals = get_skew_signals(skew_result, term_result)
        regime = create_advanced_regime_result(bars)

        signal_gen = IVSkewSignalGenerator()
        result = signal_gen.generate(bars, skew_signals, regime)

        assert hasattr(result, 'long_entry')
        assert hasattr(result, 'short_entry')
        assert result.long_entry.dtype == bool
        assert result.short_entry.dtype == bool


class TestGARCHSignalGenerator:
    def test_generate(self):
        bars = create_sample_bars(200)
        returns = create_sample_returns(bars)
        garch_result = calculate_garch_volatility(returns)
        garch_signals = get_garch_signals(garch_result, returns)
        regime = create_advanced_regime_result(bars)

        signal_gen = GARCHSignalGenerator()
        result = signal_gen.generate(bars, garch_signals, regime)

        assert hasattr(result, 'long_entry')
        assert hasattr(result, 'short_entry')
        assert result.long_entry.dtype == bool
        assert result.short_entry.dtype == bool


class TestShadowGammaSignalGenerator:
    def test_generate(self):
        bars = create_sample_bars(200)
        options = create_sample_options_chain()
        underlying = bars.set_index("timestamp")["close"]

        shadow_result = calculate_shadow_gamma(options, underlying)
        shadow_signals = get_shadow_gamma_signals(shadow_result)
        regime = create_advanced_regime_result(bars)

        signal_gen = ShadowGammaSignalGenerator()
        result = signal_gen.generate(bars, shadow_signals, regime)

        assert hasattr(result, 'long_entry')
        assert hasattr(result, 'short_entry')
        assert result.long_entry.dtype == bool
        assert result.short_entry.dtype == bool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])