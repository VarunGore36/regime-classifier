"""Tests for backtest module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from backtest import (
    BacktestEngine,
    BacktestResult,
    run_walkforward_backtest,
    bootstrap_backtest,
    monte_carlo_simulation,
    ablation_test
)
from signals import SignalGenerator
from regime import CombinedRegimeClassifier


def create_sample_data(n=1000):
    """Create sample data for backtesting."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01 09:30", periods=n, freq="1min", tz="America/New_York")

    # Trending market with some noise
    trend = np.linspace(0, 0.02, n)  # 2% trend over period
    noise = np.random.normal(0, 0.001, n)
    returns = trend + noise
    prices = 15000 * np.exp(np.cumsum(returns))

    noise_ohlc = np.random.normal(0, 0.0005, (n, 3))
    opens = prices * (1 + noise_ohlc[:, 0])
    highs = np.maximum(opens, prices) * (1 + np.abs(noise_ohlc[:, 1]))
    lows = np.minimum(opens, prices) * (1 - np.abs(noise_ohlc[:, 2]))
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


class TestBacktestEngine:
    def test_run(self):
        bars, qqq_bars, options, fp = create_sample_data(500)

        engine = BacktestEngine()
        signal_gen = SignalGenerator()

        result = engine.run(bars, qqq_bars, options, fp, signal_gen)

        assert isinstance(result, BacktestResult)
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'trades')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'signals')

        # Check metrics
        metrics = result.metrics
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "total_trades" in metrics

    def test_extract_trades(self):
        bars, qqq_bars, options, fp = create_sample_data(500)

        engine = BacktestEngine()
        signal_gen = SignalGenerator()

        result = engine.run(bars, qqq_bars, options, fp, signal_gen)
        trades = result.trades

        if len(trades) > 0:
            assert "entry_time" in trades.columns
            assert "exit_time" in trades.columns
            assert "side" in trades.columns
            assert "entry_price" in trades.columns
            assert "exit_price" in trades.columns
            assert "pnl" in trades.columns
            assert "duration_min" in trades.columns
            assert "signal_type" in trades.columns
            assert "exit_reason" in trades.columns


class TestWalkForwardBacktest:
    def test_run_walkforward(self):
        bars, qqq_bars, options, fp = create_sample_data(2000)

        engine = BacktestEngine()

        results = run_walkforward_backtest(
            engine, bars, qqq_bars, options, fp,
            train_months=2,
            test_months=1,
            step_months=1
        )

        assert isinstance(results, list)
        assert len(results) > 0

        for r in results:
            assert isinstance(r, BacktestResult)
            assert "train_period" in r.config
            assert "test_period" in r.config


class TestBootstrap:
    def test_bootstrap_backtest(self):
        bars, qqq_bars, options, fp = create_sample_data(800)

        engine = BacktestEngine()

        result = bootstrap_backtest(
            engine, bars, qqq_bars, options, fp,
            n_paths=100,  # Small for testing
            block_size=10,
            seed=42
        )

        assert hasattr(result, 'n_paths')
        assert hasattr(result, 'profit_probability')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'mean_return')
        assert hasattr(result, 'std_return')
        assert hasattr(result, 'confidence_interval')

        assert 0 <= result.profit_probability <= 1
        assert 0 <= result.p_value <= 1
        assert result.n_paths == 100


class TestMonteCarlo:
    def test_monte_carlo_simulation(self):
        # Create sample trades
        trades = pd.DataFrame({
            "pnl": np.random.normal(10, 50, 100),
            "entry_time": pd.date_range("2024-01-01", periods=100, freq="D"),
            "exit_time": pd.date_range("2024-01-01", periods=100, freq="D")
        })

        result = monte_carlo_simulation(trades, n_simulations=200, seed=42)

        assert "median_final_equity" in result
        assert "mean_final_equity" in result
        assert "profit_probability" in result
        assert "percentile_5" in result
        assert "percentile_95" in result
        assert "median_max_drawdown" in result
        assert "all_equity_curves" in result

        assert result.all_equity_curves.shape == (200, 101)


class TestAblationTest:
    def test_ablation_test(self):
        bars, qqq_bars, options, fp = create_sample_data(500)

        engine = BacktestEngine()

        # Define component modifiers
        def no_poc(config):
            config = config.copy()
            config["poc_reversion"] = {"enabled": False}
            return config

        def no_vwap(config):
            config = config.copy()
            config["vwap_bands"] = {"enabled": False}
            return config

        components = {
            "poc_reversion": no_poc,
            "vwap_bands": no_vwap
        }

        results = ablation_test(engine, bars, qqq_bars, options, fp, components)

        assert "base" in results
        assert "without_poc_reversion" in results
        assert "without_vwap_bands" in results

        for key in ["base", "without_poc_reversion", "without_vwap_bands"]:
            assert "total_return" in results[key]
            assert "total_trades" in results[key]
            assert "win_rate" in results[key]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])