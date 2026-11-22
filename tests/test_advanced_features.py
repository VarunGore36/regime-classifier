"""Tests for advanced features."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from features import (
    get_entropy_features, calculate_shannon_entropy, calculate_permutation_entropy,
    calculate_sample_entropy, calculate_approximate_entropy, classify_entropy_regime,
    calculate_iv_skew, calculate_term_structure, get_skew_signals,
    estimate_garch11, calculate_garch_volatility, calculate_shadow_gamma,
    get_garch_signals, get_shadow_gamma_signals
)
from regime import (
    AdvancedEntropyRegimeClassifier, GARCHRegimeClassifier,
    CompositeAdvancedRegimeClassifier
)


def create_sample_returns(n=1000):
    """Create sample returns."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01 09:30", periods=n, freq="1min", tz="America/New_York")
    # Regime-switching volatility
    vol_regime = np.random.choice([0.5, 1.0, 2.0], n, p=[0.3, 0.5, 0.2])
    returns = np.random.normal(0, 0.001, n) * np.repeat(vol_regime, n // 3 + 1)[:n]
    return pd.Series(returns, index=dates)


def create_sample_options_chain(n_snapshots=10, n_strikes=20):
    """Create sample options chain."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01 09:30", periods=n_snapshots, freq="30min", tz="America/New_York")
    expiries = pd.date_range("2024-01-01", periods=4, freq="W-THU")
    spot = 15000

    rows = []
    for ts in timestamps:
        for exp in expiries:
            for strike in np.arange(spot - 500, spot + 500, 50):
                for right in ["C", "P"]:
                    dte = (exp - ts).days
                    delta = 0.5 if right == "C" else -0.5
                    if strike > spot:
                        delta = 0.5 - 0.3 * (strike - spot) / 500 if right == "C" else -0.5 + 0.3 * (strike - spot) / 500
                    else:
                        delta = 0.5 + 0.3 * (spot - strike) / 500 if right == "C" else -0.5 - 0.3 * (spot - strike) / 500
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


def create_sample_bars(n=1000):
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


class TestEntropy:
    def test_shannon_entropy(self):
        returns = create_sample_returns(500)
        ent = calculate_shannon_entropy(returns, window=100)
        assert len(ent) == len(returns)
        assert (ent.dropna() >= 0).all()

    def test_permutation_entropy(self):
        returns = create_sample_returns(500)
        ent = calculate_permutation_entropy(returns, window=100)
        assert len(ent) == len(returns)
        assert (ent.dropna() >= 0).all()
        assert (ent.dropna() <= np.log(6)).all()  # Normalized to max log(3!)

    def test_sample_entropy(self):
        returns = create_sample_returns(500)
        ent = calculate_sample_entropy(returns, window=100)
        assert len(ent) == len(returns)
        assert (ent.dropna() >= 0).all()

    def test_approximate_entropy(self):
        returns = create_sample_returns(500)
        ent = calculate_approximate_entropy(returns, window=100)
        assert len(ent) == len(returns)
        assert (ent.dropna() >= 0).all()

    def test_get_entropy_features(self):
        returns = create_sample_returns(500)
        features = get_entropy_features(returns, window=100)
        assert hasattr(features, 'shannon_entropy')
        assert hasattr(features, 'permutation_entropy')
        assert hasattr(features, 'sample_entropy')
        assert hasattr(features, 'approximate_entropy')
        assert hasattr(features, 'entropy_regime')

    def test_classify_entropy_regime(self):
        returns = create_sample_returns(500)
        perm_ent = calculate_permutation_entropy(returns, window=100)
        regime = classify_entropy_regime(perm_ent, lookback=200)
        valid = ["low_entropy", "high_entropy", "normal"]
        assert regime.dropna().isin(valid).all()


class TestIVSkew:
    def test_calculate_iv_skew(self):
        options = create_sample_options_chain()
        bars = create_sample_bars(200)
        underlying = bars.set_index("timestamp")["close"]

        result = calculate_iv_skew(options, underlying)
        assert hasattr(result, 'skew_25d')
        assert hasattr(result, 'skew_10d')
        assert hasattr(result, 'atm_iv')
        assert hasattr(result, 'skew_slope')
        assert hasattr(result, 'kurtosis')

    def test_calculate_term_structure(self):
        options = create_sample_options_chain()
        result = calculate_term_structure(options)
        assert hasattr(result, 'iv_by_dte')
        assert hasattr(result, 'slope')
        assert hasattr(result, 'curvature')
        assert hasattr(result, 'atm_iv_30d')
        assert hasattr(result, 'iv_cone')

    def test_get_skew_signals(self):
        options = create_sample_options_chain()
        bars = create_sample_bars(200)
        underlying = bars.set_index("timestamp")["close"]

        skew_result = calculate_iv_skew(options, underlying)
        term_result = calculate_term_structure(options)
        signals = get_skew_signals(skew_result, term_result)

        assert "skew_steep" in signals.columns
        assert "skew_flat" in signals.columns
        assert "term_steep" in signals.columns


class TestGARCH:
    def test_estimate_garch11(self):
        returns = create_sample_returns(2000)
        omega, alpha, beta = estimate_garch11(returns)
        assert omega > 0
        assert alpha > 0
        assert beta > 0
        assert alpha + beta < 1

    def test_calculate_garch_volatility(self):
        returns = create_sample_returns(2000)
        result = calculate_garch_volatility(returns)
        assert hasattr(result, 'conditional_vol')
        assert hasattr(result, 'forecast')
        assert hasattr(result, 'vol_regime')
        assert hasattr(result, 'params')
        assert hasattr(result, 'log_likelihood')

        valid_regimes = ["low_vol", "high_vol", "normal"]
        assert result.vol_regime.dropna().isin(valid_regimes).all()

    def test_get_garch_signals(self):
        returns = create_sample_returns(2000)
        garch_result = calculate_garch_volatility(returns)
        signals = get_garch_signals(garch_result, returns)

        assert "garch_vol" in signals.columns
        assert "vol_regime" in signals.columns
        assert "vol_expanding" in signals.columns
        assert "extreme_residual" in signals.columns


class TestShadowGamma:
    def test_calculate_shadow_gamma(self):
        options = create_sample_options_chain()
        bars = create_sample_bars(200)
        underlying = bars.set_index("timestamp")["close"]

        result = calculate_shadow_gamma(options, underlying)
        assert hasattr(result, 'shadow_gamma')
        assert hasattr(result, 'gamma_regime')
        assert hasattr(result, 'hedging_pressure')

        valid_regimes = ["concentrated", "dispersed", "normal"]
        assert result.gamma_regime.dropna().isin(valid_regimes).all()

    def test_get_shadow_gamma_signals(self):
        options = create_sample_options_chain()
        bars = create_sample_bars(200)
        underlying = bars.set_index("timestamp")["close"]

        shadow_result = calculate_shadow_gamma(options, underlying)
        signals = get_shadow_gamma_signals(shadow_result)

        assert "shadow_gamma" in signals.columns
        assert "gamma_regime" in signals.columns
        assert "hedging_pressure" in signals.columns
        assert "dealers_buy" in signals.columns
        assert "dealers_sell" in signals.columns


class TestAdvancedRegimeClassifiers:
    def test_advanced_entropy_classifier(self):
        bars = create_sample_bars(500)
        classifier = AdvancedEntropyRegimeClassifier(window=50)
        result = classifier.classify(bars)

        valid = ["low_entropy", "high_entropy", "normal"]
        assert result.regime.dropna().isin(valid).all()
        assert (result.confidence >= 0).all()
        assert (result.confidence <= 1).all()

    def test_garch_classifier(self):
        bars = create_sample_bars(1000)
        classifier = GARCHRegimeClassifier()
        result = classifier.classify(bars)

        valid = ["low_vol", "high_vol", "normal"]
        assert result.regime.dropna().isin(valid).all()

    def test_composite_advanced_classifier(self):
        bars = create_sample_bars(1000)
        classifier = CompositeAdvancedRegimeClassifier()
        result = classifier.classify(bars)

        valid = ["favorable", "unfavorable", "neutral"]
        assert result.regime.dropna().isin(valid).all()
        assert "entropy" in result.components
        assert "garch" in result.components
        assert "composite_score" in result.components


if __name__ == "__main__":
    pytest.main([__file__, "-v"])