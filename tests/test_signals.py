"""Tests for signals module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from signals import (
    POCReversionSignal,
    VWAPBandSignal,
    BreakFadeSignal,
    StopLossExit,
    TakeProfitExit,
    TrailingStopExit,
    TimeExit,
    SignalGenerator,
    combine_exits
)
from features import calculate_volume_profile, calculate_vwap, get_profile_levels
from regime import CombinedRegimeClassifier


def create_sample_bars(n=500):
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


def create_sample_footprint(bars):
    """Create sample footprint from bars."""
    rows = []
    for _, bar in bars.iterrows():
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


def create_regime_result(bars):
    """Create mock regime result."""
    n = len(bars)
    return type('RegimeResult', (), {
        'regime': pd.Series(['normal'] * n, index=bars.index),
        'confidence': pd.Series([0.7] * n, index=bars.index),
        'sub_regimes': {
            'gex': pd.Series(['positive_gex'] * n, index=bars.index),
            'rv': pd.Series(['normal_vol'] * n, index=bars.index),
            'dex': pd.Series(['neutral_dex'] * n, index=bars.index),
            'entropy': pd.Series(['normal'] * n, index=bars.index),
            'iv_rv': pd.Series(['neutral'] * n, index=bars.index),
        },
        'bias': pd.Series(['neutral'] * n, index=bars.index),
        'components': {}
    })()


class TestPOCReversionSignal:
    def test_generate(self):
        bars = create_sample_bars(300)
        vol_profile = calculate_volume_profile(bars)
        regime = create_regime_result(bars)

        signal = POCReversionSignal()
        result = signal.generate(bars, vol_profile, regime)

        assert hasattr(result, 'long_entry')
        assert hasattr(result, 'short_entry')
        assert hasattr(result, 'long_price')
        assert hasattr(result, 'short_price')
        assert hasattr(result, 'stop_loss')
        assert hasattr(result, 'take_profit')
        assert hasattr(result, 'signal_type')

        # Should be boolean series
        assert result.long_entry.dtype == bool
        assert result.short_entry.dtype == bool

    def test_disabled(self):
        bars = create_sample_bars(100)
        vol_profile = calculate_volume_profile(bars)
        regime = create_regime_result(bars)

        signal = POCReversionSignal(config={"enabled": False})
        result = signal.generate(bars, vol_profile, regime)

        assert not result.long_entry.any()
        assert not result.short_entry.any()


class TestVWAPBandSignal:
    def test_generate(self):
        bars = create_sample_bars(300)
        vwap_result = calculate_vwap(bars)
        regime = create_regime_result(bars)

        signal = VWAPBandSignal()
        result = signal.generate(bars, vwap_result, regime)

        assert hasattr(result, 'long_entry')
        assert hasattr(result, 'short_entry')
        assert result.long_entry.dtype == bool
        assert result.short_entry.dtype == bool


class TestBreakFadeSignal:
    def test_generate(self):
        bars = create_sample_bars(300)
        footprint = create_sample_footprint(bars)
        vol_profile = calculate_volume_profile(bars)
        levels = get_profile_levels(vol_profile)
        regime = create_regime_result(bars)

        signal = BreakFadeSignal()
        result = signal.generate(bars, footprint, levels, regime)

        assert hasattr(result, 'long_entry')
        assert hasattr(result, 'short_entry')
        assert result.long_entry.dtype == bool
        assert result.short_entry.dtype == bool


class TestStopLossExit:
    def test_generate(self):
        bars = create_sample_bars(100)
        entry_price = pd.Series([15000] * 100, index=bars.index)
        position = pd.Series([1] * 50 + [0] * 50, index=bars.index)

        exit_signal = StopLossExit(stop_ticks=20)
        result = exit_signal.generate(bars, entry_price, position)

        assert hasattr(result, 'exit_long')
        assert hasattr(result, 'exit_short')
        assert hasattr(result, 'exit_price')
        assert hasattr(result, 'exit_reason')

        # Should exit when low hits stop
        stop_price = 15000 - 20 * 0.25
        # Create bar that hits stop
        bars.loc[10, "low"] = stop_price - 1
        result2 = exit_signal.generate(bars, entry_price, position)
        assert result2.exit_long.iloc[10]


class TestTakeProfitExit:
    def test_generate(self):
        bars = create_sample_bars(100)
        entry_price = pd.Series([15000] * 100, index=bars.index)
        position = pd.Series([1] * 50 + [0] * 50, index=bars.index)

        exit_signal = TakeProfitExit(target_ticks=40)
        result = exit_signal.generate(bars, entry_price, position)

        # Should exit when high hits target
        target_price = 15000 + 40 * 0.25
        bars.loc[10, "high"] = target_price + 1
        result2 = exit_signal.generate(bars, entry_price, position)
        assert result2.exit_long.iloc[10]


class TestTrailingStopExit:
    def test_generate(self):
        bars = create_sample_bars(50)
        entry_price = pd.Series([15000] * 50, index=bars.index)
        position = pd.Series([1] * 50, index=bars.index)

        # Create uptrend
        bars["high"] = 15000 + np.arange(50) * 0.5
        bars["low"] = bars["high"] - 5

        exit_signal = TrailingStopExit(trail_ticks=10)
        result = exit_signal.generate(bars, entry_price, position)

        # Should trail and eventually exit
        assert hasattr(result, 'exit_long')
        assert result.exit_long.dtype == bool


class TestTimeExit:
    def test_generate(self):
        bars = create_sample_bars(100)
        # Set last bar to exit time
        bars.loc[99, "timestamp"] = pd.Timestamp("2024-01-01 15:55:00", tz="America/New_York")
        position = pd.Series([1] * 100, index=bars.index)

        exit_signal = TimeExit(exit_hour=15, exit_minute=55)
        result = exit_signal.generate(bars, position)

        assert result.exit_long.iloc[99]
        assert result.exit_reason.iloc[99] == "time_exit_long"


class TestCombineExits:
    def test_combine_exits(self):
        bars = create_sample_bars(50)
        entry_price = pd.Series([15000] * 50, index=bars.index)
        position = pd.Series([1] * 50, index=bars.index)

        # Stop loss at bar 10
        bars.loc[10, "low"] = 15000 - 20 * 0.25 - 1
        sl = StopLossExit(stop_ticks=20)
        sl_result = sl.generate(bars, entry_price, position)

        # Take profit at bar 20
        bars.loc[20, "high"] = 15000 + 40 * 0.25 + 1
        tp = TakeProfitExit(target_ticks=40)
        tp_result = tp.generate(bars, entry_price, position)

        combined = combine_exits(sl_result, tp_result)

        # First exit (stop loss at bar 10) should win
        assert combined.exit_long.iloc[10]
        assert combined.exit_reason.iloc[10] == "stop_loss_long"
        # Should not exit at bar 20 (already flat)
        assert not combined.exit_long.iloc[20]


class TestSignalGenerator:
    def test_generate(self):
        bars = create_sample_bars(200)
        qqq_bars = bars.copy()
        qqq_bars["close"] = bars["close"] / 42.85  # Approximate ratio
        options_chain = pd.DataFrame()
        footprint = create_sample_footprint(bars)

        generator = SignalGenerator()
        result = generator.generate(bars, qqq_bars, options_chain, footprint)

        assert hasattr(result, 'entry')
        assert hasattr(result, 'exit')
        assert hasattr(result, 'position')
        assert hasattr(result, 'equity_curve')

        # Position should be -1, 0, or 1
        assert result.position.isin([-1, 0, 1]).all()

        # Equity curve should start at initial capital
        from config import get_config
        cfg = get_config()
        assert abs(result.equity_curve.iloc[0] - cfg.backtest.initial_capital) < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])