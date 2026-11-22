"""Tests for regime module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from regime import (
    RVZScoreRegimeClassifier,
    GEXRegimeClassifier,
    DEXRegimeClassifier,
    EntropyRegimeClassifier,
    CombinedRegimeClassifier,
    get_regime_summary
)


def create_sample_data(n=2000):
    """Create sample data for regime testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01 09:30", periods=n, freq="1min", tz="America/New_York")

    # Create price series with varying volatility
    returns = np.random.normal(0, 0.001, n)
    # Add volatility clusters
    vol_regime = np.random.choice([0.5, 1.0, 2.0], n, p=[0.3, 0.5, 0.2])
    returns = returns * np.repeat(vol_regime, n // 3 + 1)[:n]

    prices = 15000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "timestamp": dates,
        "close": prices,
        "gex_net": np.random.normal(0, 1e6, n),
        "dex_net": np.random.normal(0, 1e6, n),
        "gamma_flip": 15000 + np.random.normal(0, 50, n),
        "underlying_price": prices,
        "iv": 0.15 + np.random.normal(0, 0.02, n),
        "rv": 0.15 + np.random.normal(0, 0.02, n)
    })

    return df


class TestRVZScoreRegimeClassifier:
    def test_classify(self):
        data = create_sample_data(1000)
        classifier = RVZScoreRegimeClassifier(window_minutes=30)
        result = classifier.classify(data)

        assert hasattr(result, 'regime')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'components')

        valid_regimes = ["low_vol", "normal_vol", "elevated_vol", "very_elevated_vol", "unknown"]
        assert result.regime.dropna().isin(valid_regimes).all()
        assert (result.confidence >= 0).all()
        assert (result.confidence <= 1).all()

    def test_components(self):
        data = create_sample_data(1000)
        classifier = RVZScoreRegimeClassifier()
        result = classifier.classify(data)

        assert "rv" in result.components
        assert "rv_zscore" in result.components
        assert "vol_acceleration" in result.components
        assert "vol_regime" in result.components


class TestGEXRegimeClassifier:
    def test_classify(self):
        data = create_sample_data(1000)
        classifier = GEXRegimeClassifier()
        result = classifier.classify(data)

        assert hasattr(result, 'regime')
        assert hasattr(result, 'confidence')

        valid_regimes = ["positive_gex", "negative_gex", "neutral_gex"]
        assert result.regime.dropna().isin(valid_regimes).all()

    def test_gamma_flip_distance(self):
        data = create_sample_data(1000)
        classifier = GEXRegimeClassifier(flip_buffer=0.2)
        result = classifier.classify(data)

        assert "gex_net" in result.components
        assert "gamma_flip" in result.components
        assert "dist_from_flip" in result.components


class TestDEXRegimeClassifier:
    def test_classify(self):
        data = create_sample_data(1000)
        classifier = DEXRegimeClassifier()
        result = classifier.classify(data)

        valid_regimes = ["long_dex", "short_dex", "neutral_dex"]
        assert result.regime.dropna().isin(valid_regimes).all()


class TestEntropyRegimeClassifier:
    def test_classify(self):
        data = create_sample_data(500)
        classifier = EntropyRegimeClassifier(window=50)
        result = classifier.classify(data)

        valid_regimes = ["low_entropy", "high_entropy", "normal"]
        assert result.regime.dropna().isin(valid_regimes).all()

    def test_calculate_entropy(self):
        classifier = EntropyRegimeClassifier(window=50)
        returns = pd.Series(np.random.normal(0, 0.01, 200))
        entropy = classifier.calculate_entropy(returns)

        assert len(entropy) == len(returns)
        assert (entropy.dropna() >= 0).all()


class TestCombinedRegimeClassifier:
    def test_classify(self):
        data = create_sample_data(1000)
        classifier = CombinedRegimeClassifier()
        result = classifier.classify(data)

        assert hasattr(result, 'primary_regime')
        assert hasattr(result, 'regime_score')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'sub_regimes')
        assert hasattr(result, 'bias')
        assert hasattr(result, 'components')

        # Check sub-regimes
        assert "rv" in result.sub_regimes
        assert "gex" in result.sub_regimes
        assert "dex" in result.sub_regimes
        assert "entropy" in result.sub_regimes
        assert "iv_rv" in result.sub_regimes

        # Bias should be valid
        valid_bias = ["bullish", "bearish", "neutral"]
        assert result.bias.dropna().isin(valid_bias).all()

        # Regime score should be between -1 and 1
        assert (result.regime_score >= -1).all()
        assert (result.regime_score <= 1).all()

    def test_get_regime_summary(self):
        data = create_sample_data(1000)
        classifier = CombinedRegimeClassifier()
        result = classifier.classify(data)

        timestamp = data["timestamp"].iloc[500]
        summary = get_regime_summary(result, timestamp)

        assert "primary_regime" in summary
        assert "regime_score" in summary
        assert "confidence" in summary
        assert "bias" in summary
        assert "sub_regimes" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])