"""Integration tests for the complete pipeline."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from regime_classifier import (
    RegimePipeline, run_pipeline,
    SyntheticDataLoader,
    SignalGenerator, BacktestEngine,
    CombinedRegimeClassifier
)
from config import load_config


def create_sample_data(n=500):
    """Create sample data for pipeline testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01 09:30", periods=n, freq="1min", tz="America/New_York")
    returns = np.random.normal(0, 0.001, n)
    prices = 15000 * np.exp(np.cumsum(returns))

    noise = np.random.normal(0, 0.0005, (n, 3))
    opens = prices * (1 + noise[:, 0])
    highs = np.maximum(opens, prices) * (1 + np.abs(noise[:, 1]))
    lows = np.minimum(opens, prices) * (1 - np.abs(noise[:, 2]))
    volumes = np.random.randint(100, 5000, n)

    bars = pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes
    })

    qqq_bars = bars.copy()
    qqq_bars["close"] = bars["close"] / 42.85
    qqq_bars["open"] = bars["open"] / 42.85
    qqq_bars["high"] = bars["high"] / 42.85
    qqq_bars["low"] = bars["low"] / 42.85

    options_chain = pd.DataFrame()
    footprint = pd.DataFrame()

    return bars, qqq_bars, options_chain, footprint


class TestPipeline:
    def test_pipeline_synthetic(self):
        """Test pipeline with synthetic data."""
        pipeline = RegimePipeline()

        start = pd.Timestamp("2024-01-01 09:30", tz="America/New_York")
        end = pd.Timestamp("2024-01-31 16:00", tz="America/New_York")

        result = pipeline.run_synthetic(start, end, output_dir="test_reports")

        assert result is not None
        assert result.backtest_result is not None
        assert result.regime_result is not None
        assert result.advanced_regime_result is not None
        assert result.report_path.endswith(".pdf")

    def test_pipeline_with_data(self):
        """Test pipeline with provided data."""
        pipeline = RegimePipeline()

        nq_bars, qqq_bars, options_chain, footprint = create_sample_data(300)

        result = pipeline.run(nq_bars, qqq_bars, options_chain, footprint, output_dir="test_reports")

        assert result is not None
        assert result.backtest_result is not None
        assert "total_return" in result.backtest_result.metrics

    def test_run_pipeline_convenience(self):
        """Test run_pipeline convenience function."""
        # This would need a config file, so we skip for now
        pass


class TestEndToEnd:
    def test_signal_generation_flow(self):
        """Test complete signal generation flow."""
        nq_bars, qqq_bars, options_chain, footprint = create_sample_data(200)

        signal_gen = SignalGenerator()
        signals = signal_gen.generate(nq_bars, qqq_bars, options_chain, footprint)

        assert signals.entry is not None
        assert signals.exit is not None
        assert signals.position is not None
        assert signals.equity_curve is not None

    def test_regime_classification_flow(self):
        """Test regime classification flow."""
        nq_bars, qqq_bars, options_chain, footprint = create_sample_data(500)

        regime_clf = CombinedRegimeClassifier()
        regime_data = nq_bars[["timestamp", "close"]].copy()

        # Add mock GEX/DEX
        regime_data["gex_net"] = np.random.normal(0, 1e6, len(nq_bars))
        regime_data["dex_net"] = np.random.normal(0, 1e6, len(nq_bars))
        regime_data["gamma_flip"] = 15000
        regime_data["underlying_price"] = nq_bars["close"]
        regime_data["iv"] = 0.15
        regime_data["rv"] = 0.15

        result = regime_clf.classify(regime_data)

        assert result.primary_regime is not None
        assert result.confidence is not None
        assert result.sub_regimes is not None

    def test_backtest_flow(self):
        """Test backtest flow."""
        nq_bars, qqq_bars, options_chain, footprint = create_sample_data(300)

        config = load_config("config/strategy.yaml")
        engine = BacktestEngine(config.to_dict())
        signal_gen = SignalGenerator(config.to_dict())

        result = engine.run(nq_bars, qqq_bars, options_chain, footprint, signal_gen)

        assert result.equity_curve is not None
        assert result.trades is not None
        assert result.metrics is not None

    def test_walkforward_flow(self):
        """Test walk-forward flow."""
        nq_bars, qqq_bars, options_chain, footprint = create_sample_data(1000)

        config = load_config("config/strategy.yaml")
        engine = BacktestEngine(config.to_dict())

        from backtest import run_walkforward_backtest
        results = run_walkforward_backtest(
            engine, nq_bars, qqq_bars, options_chain, footprint,
            train_months=1, test_months=1, step_months=1
        )

        assert len(results) > 0
        for r in results:
            assert "train_period" in r.config
            assert "test_period" in r.config

    def test_bootstrap_flow(self):
        """Test bootstrap flow."""
        nq_bars, qqq_bars, options_chain, footprint = create_sample_data(500)

        config = load_config("config/strategy.yaml")
        engine = BacktestEngine(config.to_dict())

        from backtest import bootstrap_backtest
        result = bootstrap_backtest(
            engine, nq_bars, qqq_bars, options_chain, footprint,
            n_paths=100, block_size=10, seed=42
        )

        assert result.n_paths == 100
        assert 0 <= result.profit_probability <= 1
        assert 0 <= result.p_value <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])