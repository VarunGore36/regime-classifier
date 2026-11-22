"""Tests for features module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from features import (
    calculate_volume_profile,
    calculate_vwap,
    calculate_realized_volatility,
    calculate_expected_move,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
    calculate_conversion_ratio,
    convert_qqq_to_nq,
    calculate_cvd,
    detect_absorption,
    detect_exhaustion,
)


def create_sample_bars(n=1000, start_price=15000, freq="1min"):
    """Create sample OHLCV bars."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01 09:30", periods=n, freq=freq, tz="America/New_York")
    returns = np.random.normal(0, 0.001, n)
    prices = start_price * np.exp(np.cumsum(returns))

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
        n_levels = 20
        prices = np.linspace(bar["low"], bar["high"], n_levels)
        for p in prices:
            bid_vol = np.random.randint(0, 100)
            ask_vol = np.random.randint(0, 100)
            rows.append({
                "timestamp": bar["timestamp"],
                "price": p,
                "bid_volume": bid_vol,
                "ask_volume": ask_vol
            })
    return pd.DataFrame(rows)


class TestVolumeProfile:
    def test_calculate_volume_profile(self):
        bars = create_sample_bars(500)
        profile = calculate_volume_profile(bars)

        assert hasattr(profile, 'poc')
        assert hasattr(profile, 'vah')
        assert hasattr(profile, 'val')
        assert hasattr(profile, 'hvns')
        assert hasattr(profile, 'lvns')
        assert hasattr(profile, 'profile')

        # POC should be within price range
        assert bars["low"].min() <= profile.poc <= bars["high"].max()
        # VAH > VAL
        assert profile.vah > profile.val

    def test_poc_is_max_volume(self):
        bars = create_sample_bars(500)
        profile = calculate_volume_profile(bars)

        # POC price should have max volume in profile
        poc_volume = profile.profile[profile.profile["price"] == profile.poc]["volume"].values[0]
        max_volume = profile.profile["volume"].max()
        assert abs(poc_volume - max_volume) < 0.01


class TestVWAP:
    def test_calculate_vwap(self):
        bars = create_sample_bars(500)
        vwap_result = calculate_vwap(bars)

        assert hasattr(vwap_result, 'vwap')
        assert hasattr(vwap_result, 'upper_bands')
        assert hasattr(vwap_result, 'lower_bands')
        assert hasattr(vwap_result, 'std_dev')

        # VWAP should be within price range
        vwap_series = vwap_result.vwap.dropna()
        assert (vwap_series >= bars["low"].min()).all()
        assert (vwap_series <= bars["high"].max()).all()

    def test_vwap_bands(self):
        bars = create_sample_bars(500)
        vwap_result = calculate_vwap(bars, bands=[1.0, 2.0, 3.0])

        for band in [1.0, 2.0, 3.0]:
            assert band in vwap_result.upper_bands
            assert band in vwap_result.lower_bands
            # Upper > VWAP > Lower
            upper = vwap_result.upper_bands[band].dropna()
            lower = vwap_result.lower_bands[band].dropna()
            vwap = vwap_result.vwap.dropna()
            assert (upper > vwap).all()
            assert (lower < vwap).all()


class TestRealizedVolatility:
    def test_calculate_realized_volatility(self):
        bars = create_sample_bars(2000)
        rv_result = calculate_realized_volatility(bars, window_minutes=30)

        assert hasattr(rv_result, 'rv')
        assert hasattr(rv_result, 'rv_zscore')
        assert hasattr(rv_result, 'log_returns')
        assert hasattr(rv_result, 'volatility_acceleration')
        assert hasattr(rv_result, 'regime')

        # RV should be positive
        assert (rv_result.rv.dropna() >= 0).all()

        # Regime should be valid
        valid_regimes = ["low_vol", "normal_vol", "elevated_vol", "very_elevated_vol", "unknown"]
        assert rv_result.regime.dropna().isin(valid_regimes).all()

    def test_log_returns(self):
        bars = create_sample_bars(100)
        rv_result = calculate_realized_volatility(bars, window_minutes=30)

        # Log returns = ln(Pt/Pt-1)
        expected = np.log(bars["close"] / bars["close"].shift(1))
        pd.testing.assert_series_equal(
            rv_result.log_returns.dropna(),
            expected.dropna(),
            check_names=False
        )


class TestBlackScholes:
    def test_black_scholes_greeks_call(self):
        greeks = black_scholes_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call"
        )

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks
        assert "rho" in greeks
        assert "charm" in greeks
        assert "vanna" in greeks
        assert "vomma" in greeks

        # Call delta should be positive
        assert greeks["delta"] > 0
        # Gamma should be positive
        assert greeks["gamma"] > 0
        # Vega should be positive
        assert greeks["vega"] > 0

    def test_black_scholes_greeks_put(self):
        greeks = black_scholes_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="put"
        )

        # Put delta should be negative
        assert greeks["delta"] < 0
        # Gamma positive
        assert greeks["gamma"] > 0

    def test_black_scholes_price(self):
        price = black_scholes_price(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
        assert price > 0

        # Put-call parity
        call = black_scholes_price(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")
        put = black_scholes_price(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="put")
        parity = call - put
        expected = 100 - 100 * np.exp(-0.05 * 0.25)
        assert abs(parity - expected) < 0.01

    def test_implied_volatility(self):
        # Price with known IV
        true_iv = 0.2
        price = black_scholes_price(S=100, K=100, T=0.25, r=0.05, sigma=true_iv, option_type="call")

        # Recover IV
        iv = implied_volatility(price, S=100, K=100, T=0.25, r=0.05, option_type="call")
        assert abs(iv - true_iv) < 0.001


class TestConversion:
    def test_calculate_conversion_ratio(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01 09:30", periods=100, freq="1min", tz="America/New_York")
        nq = pd.DataFrame({"timestamp": dates, "close": 15000 + np.cumsum(np.random.normal(0, 5, 100))})
        qqq = pd.DataFrame({"timestamp": dates, "close": 350 + np.cumsum(np.random.normal(0, 0.1, 100))})

        ratio = calculate_conversion_ratio(nq, qqq)
        assert len(ratio) == 100
        assert (ratio > 0).all()

    def test_convert_qqq_to_nq(self):
        nq_price = 15000
        qqq_price = 350
        ratio = nq_price / qqq_price

        qqq_level = 355
        nq_level = convert_qqq_to_nq(qqq_level, ratio)
        expected = qqq_level * ratio
        assert abs(nq_level - expected) < 0.01


class TestCVD:
    def test_calculate_cvd(self):
        bars = create_sample_bars(100)
        footprint = create_sample_footprint(bars)

        cvd_result = calculate_cvd(footprint, bars)

        assert hasattr(cvd_result, 'cvd')
        assert hasattr(cvd_result, 'delta')
        assert hasattr(cvd_result, 'buy_volume')
        assert hasattr(cvd_result, 'sell_volume')
        assert hasattr(cvd_result, 'delta_ma')
        assert hasattr(cvd_result, 'cvd_divergence')

        # CVD should be cumulative sum of delta
        expected_cvd = cvd_result.delta.cumsum()
        pd.testing.assert_series_equal(
            cvd_result.cvd,
            expected_cvd,
            check_names=False
        )


class TestAbsorption:
    def test_detect_absorption(self):
        bars = create_sample_bars(100)
        footprint = create_sample_footprint(bars)

        result = detect_absorption(footprint)

        assert hasattr(result, 'absorption_bullish')
        assert hasattr(result, 'absorption_bearish')
        assert hasattr(result, 'absorption_strength')

        # Should be boolean series
        assert result.absorption_bullish.dtype == bool
        assert result.absorption_bearish.dtype == bool


class TestExhaustion:
    def test_detect_exhaustion(self):
        bars = create_sample_bars(100)
        footprint = create_sample_footprint(bars)

        result = detect_exhaustion(footprint)

        assert hasattr(result, 'exhaustion_bullish')
        assert hasattr(result, 'exhaustion_bearish')
        assert hasattr(result, 'exhaustion_strength')

        assert result.exhaustion_bullish.dtype == bool
        assert result.exhaustion_bearish.dtype == bool


class TestExpectedMove:
    def test_calculate_expected_move(self):
        move = calculate_expected_move(price=15000, iv=0.15, dte=7, confidence=0.68)
        # 15000 * 0.15 * sqrt(7/365) * 1.0
        expected = 15000 * 0.15 * np.sqrt(7/365)
        assert abs(move - expected) < 0.01

    def test_confidence_levels(self):
        move_68 = calculate_expected_move(100, 0.2, 30, 0.68)
        move_95 = calculate_expected_move(100, 0.2, 30, 0.95)
        # 95% should be wider than 68%
        assert move_95 > move_68


if __name__ == "__main__":
    pytest.main([__file__, "-v"])