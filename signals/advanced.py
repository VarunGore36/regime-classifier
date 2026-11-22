"""Advanced signals using entropy, IV skew, GARCH, shadow gamma."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np

from config import get_config
from features import (
    get_entropy_features, classify_entropy_regime,
    get_skew_signals, get_garch_signals, get_shadow_gamma_signals,
    calculate_shadow_gamma
)
from regime import CompositeAdvancedRegimeClassifier


@dataclass
class EntropySignal:
    """Entropy-based signal."""
    long_entry: pd.Series
    short_entry: pd.Series
    long_price: pd.Series
    short_price: pd.Series
    stop_loss: pd.Series
    take_profit: pd.Series
    signal_type: pd.Series
    metadata: dict


@dataclass
class IVSkewSignal:
    """IV Skew-based signal."""
    long_entry: pd.Series
    short_entry: pd.Series
    long_price: pd.Series
    short_price: pd.Series
    stop_loss: pd.Series
    take_profit: pd.Series
    signal_type: pd.Series
    metadata: dict


@dataclass
class GARCHSignal:
    """GARCH volatility signal."""
    long_entry: pd.Series
    short_entry: pd.Series
    long_price: pd.Series
    short_price: pd.Series
    stop_loss: pd.Series
    take_profit: pd.Series
    signal_type: pd.Series
    metadata: dict


@dataclass
class ShadowGammaSignal:
    """Shadow Gamma signal."""
    long_entry: pd.Series
    short_entry: pd.Series
    long_price: pd.Series
    short_price: pd.Series
    stop_loss: pd.Series
    take_profit: pd.Series
    signal_type: pd.Series
    metadata: dict


class EntropySignalGenerator:
    """
    Generate signals based on entropy regime.

    Low entropy (organized) = trend following
    High entropy (disorganized) = mean reversion / avoid
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = get_config()
        self.enabled = config.get("enabled", True) if config else True
        self.stop_ticks = config.get("stop_ticks", 20) if config else 20
        self.target_ticks = config.get("target_ticks", 40) if config else 40
        self.tick_size = 0.25

    def generate(
        self,
        bars: pd.DataFrame,
        entropy_features,
        regime_result
    ) -> EntropySignal:
        if not self.enabled:
            return self._empty_signal(bars)

        df = bars.copy()
        df["entropy_regime"] = entropy_features.entropy_regime.reindex(df.index, method="ffill")
        df["perm_entropy"] = entropy_features.permutation_entropy.reindex(df.index, method="ffill")

        # Low entropy = trend following
        # High entropy = avoid / mean reversion
        long_trend = (
            (df["entropy_regime"] == "low_entropy") &
            (df["close"] > df["open"]) &  # Bullish candle
            (df["close"] > df["close"].shift(1))  # Higher close
        )

        short_trend = (
            (df["entropy_regime"] == "low_entropy") &
            (df["close"] < df["open"]) &  # Bearish candle
            (df["close"] < df["close"].shift(1))  # Lower close
        )

        # High entropy = mean reversion (contrarian)
        long_reversion = (
            (df["entropy_regime"] == "high_entropy") &
            (df["close"] < df["open"]) &  # Oversold bounce
            (df["low"] < df["low"].rolling(20).min())  # New low
        )

        short_reversion = (
            (df["entropy_regime"] == "high_entropy") &
            (df["close"] > df["open"]) &  # Overbought fade
            (df["high"] > df["high"].rolling(20).max())  # New high
        )

        long_condition = long_trend | long_reversion
        short_condition = short_trend | short_reversion

        entry_price = df["close"]
        stop_long = entry_price - self.stop_ticks * self.tick_size
        stop_short = entry_price + self.stop_ticks * self.tick_size
        target_long = entry_price + self.target_ticks * self.tick_size
        target_short = entry_price - self.target_ticks * self.tick_size

        signal_type = pd.Series("none", index=df.index)
        signal_type[long_trend & long_condition] = "entropy_trend_long"
        signal_type[short_trend & short_condition] = "entropy_trend_short"
        signal_type[long_reversion & long_condition] = "entropy_reversion_long"
        signal_type[short_reversion & short_condition] = "entropy_reversion_short"

        return EntropySignal(
            long_entry=long_condition,
            short_entry=short_condition,
            long_price=entry_price,
            short_price=entry_price,
            stop_loss=pd.Series(np.where(long_condition, stop_long, np.where(short_condition, stop_short, np.nan)), index=df.index),
            take_profit=pd.Series(np.where(long_condition, target_long, np.where(short_condition, target_short, np.nan)), index=df.index),
            signal_type=signal_type,
            metadata={"type": "entropy"}
        )

    def _empty_signal(self, bars: pd.DataFrame) -> EntropySignal:
        return EntropySignal(
            long_entry=pd.Series(False, index=bars.index),
            short_entry=pd.Series(False, index=bars.index),
            long_price=pd.Series(np.nan, index=bars.index),
            short_price=pd.Series(np.nan, index=bars.index),
            stop_loss=pd.Series(np.nan, index=bars.index),
            take_profit=pd.Series(np.nan, index=bars.index),
            signal_type=pd.Series("none", index=bars.index),
            metadata={}
        )


class IVSkewSignalGenerator:
    """
    Generate signals from IV skew and term structure.

    Steep put skew = fear = potential bounce (contrarian long)
    Flat skew = complacency = potential selloff (short)
    Term structure steep = forward vol rising = cautious
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = get_config()
        self.enabled = config.get("enabled", True) if config else True
        self.stop_ticks = config.get("stop_ticks", 15) if config else 15
        self.target_ticks = config.get("target_ticks", 30) if config else 30
        self.tick_size = 0.25

    def generate(
        self,
        bars: pd.DataFrame,
        skew_signals: pd.DataFrame,
        regime_result
    ) -> IVSkewSignal:
        if not self.enabled:
            return self._empty_signal(bars)

        df = bars.copy()

        # Align skew signals
        for col in skew_signals.columns:
            df[col] = skew_signals[col].reindex(df.index, method="ffill")

        # Contrarian signals from skew
        # Steep put skew (fear) = potential long
        long_skew = (
            df["skew_steep"] &
            df["kurtosis_high"] &
            (df["close"] > df["open"])  # Reversal candle
        )

        # Flat skew (complacency) = potential short
        short_skew = (
            df["skew_flat"] &
            df["kurtosis_high"] &
            (df["close"] < df["open"])  # Reversal candle
        )

        # Term structure signals
        long_term = (
            df["term_steep"] &
            (df["close"] < df["open"])  # Buy dip in rising vol env
        )

        short_term = (
            df["term_inverted"] &
            (df["close"] > df["open"])  # Sell rip in falling vol env
        )

        long_condition = long_skew | long_term
        short_condition = short_skew | short_term

        entry_price = df["close"]
        stop_long = entry_price - self.stop_ticks * self.tick_size
        stop_short = entry_price + self.stop_ticks * self.tick_size
        target_long = entry_price + self.target_ticks * self.tick_size
        target_short = entry_price - self.target_ticks * self.tick_size

        signal_type = pd.Series("none", index=df.index)
        signal_type[long_skew & long_condition] = "skew_contrarian_long"
        signal_type[short_skew & short_condition] = "skew_contrarian_short"
        signal_type[long_term & long_condition] = "term_structure_long"
        signal_type[short_term & short_condition] = "term_structure_short"

        return IVSkewSignal(
            long_entry=long_condition,
            short_entry=short_condition,
            long_price=entry_price,
            short_price=entry_price,
            stop_loss=pd.Series(np.where(long_condition, stop_long, np.where(short_condition, stop_short, np.nan)), index=df.index),
            take_profit=pd.Series(np.where(long_condition, target_long, np.where(short_condition, target_short, np.nan)), index=df.index),
            signal_type=signal_type,
            metadata={"type": "iv_skew"}
        )

    def _empty_signal(self, bars: pd.DataFrame) -> IVSkewSignal:
        return IVSkewSignal(
            long_entry=pd.Series(False, index=bars.index),
            short_entry=pd.Series(False, index=bars.index),
            long_price=pd.Series(np.nan, index=bars.index),
            short_price=pd.Series(np.nan, index=bars.index),
            stop_loss=pd.Series(np.nan, index=bars.index),
            take_profit=pd.Series(np.nan, index=bars.index),
            signal_type=pd.Series("none", index=bars.index),
            metadata={}
        )


class GARCHSignalGenerator:
    """
    Generate signals from GARCH volatility forecasts.

    Vol expanding = reduce size / avoid
    Vol contracting = increase size / trend follow
    Extreme residuals = mean reversion opportunity
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = get_config()
        self.enabled = config.get("enabled", True) if config else True
        self.stop_ticks = config.get("stop_ticks", 20) if config else 20
        self.target_ticks = config.get("target_ticks", 40) if config else 40
        self.tick_size = 0.25
        self.residual_threshold = config.get("residual_threshold", 2.0) if config else 2.0

    def generate(
        self,
        bars: pd.DataFrame,
        garch_signals: pd.DataFrame,
        regime_result
    ) -> GARCHSignal:
        if not self.enabled:
            return self._empty_signal(bars)

        df = bars.copy()

        for col in garch_signals.columns:
            df[col] = garch_signals[col].reindex(df.index, method="ffill")

        # Vol contracting = trend friendly
        long_vol_contract = (
            df["vol_contracting"] &
            (df["close"] > df["open"]) &
            (df["close"] > df["close"].shift(1))
        )

        short_vol_contract = (
            df["vol_contracting"] &
            (df["close"] < df["open"]) &
            (df["close"] < df["close"].shift(1))
        )

        # Extreme residuals = mean reversion
        long_residual = (
            df["extreme_residual"] &
            (df["residuals"] < -self.residual_threshold) &  # Oversold
            (df["close"] > df["open"])  # Reversal
        )

        short_residual = (
            df["extreme_residual"] &
            (df["residuals"] > self.residual_threshold) &  # Overbought
            (df["close"] < df["open"])  # Reversal
        )

        # Low vol regime = breakout friendly
        long_low_vol = (
            df["low_vol_regime"] &
            (df["close"] > df["high"].shift(1)) &  # Breakout
            (df["volume"] > df["volume"].rolling(20).mean() * 1.5)
        )

        short_low_vol = (
            df["low_vol_regime"] &
            (df["close"] < df["low"].shift(1)) &  # Breakdown
            (df["volume"] > df["volume"].rolling(20).mean() * 1.5)
        )

        long_condition = long_vol_contract | long_residual | long_low_vol
        short_condition = short_vol_contract | short_residual | short_low_vol

        entry_price = df["close"]
        stop_long = entry_price - self.stop_ticks * self.tick_size
        stop_short = entry_price + self.stop_ticks * self.tick_size
        target_long = entry_price + self.target_ticks * self.tick_size
        target_short = entry_price - self.target_ticks * self.tick_size

        signal_type = pd.Series("none", index=df.index)
        signal_type[long_vol_contract & long_condition] = "garch_trend_long"
        signal_type[short_vol_contract & short_condition] = "garch_trend_short"
        signal_type[long_residual & long_condition] = "garch_reversion_long"
        signal_type[short_residual & short_condition] = "garch_reversion_short"
        signal_type[long_low_vol & long_condition] = "garch_breakout_long"
        signal_type[short_low_vol & short_condition] = "garch_breakout_short"

        return GARCHSignal(
            long_entry=long_condition,
            short_entry=short_condition,
            long_price=entry_price,
            short_price=entry_price,
            stop_loss=pd.Series(np.where(long_condition, stop_long, np.where(short_condition, stop_short, np.nan)), index=df.index),
            take_profit=pd.Series(np.where(long_condition, target_long, np.where(short_condition, target_short, np.nan)), index=df.index),
            signal_type=signal_type,
            metadata={"type": "garch"}
        )

    def _empty_signal(self, bars: pd.DataFrame) -> GARCHSignal:
        return GARCHSignal(
            long_entry=pd.Series(False, index=bars.index),
            short_entry=pd.Series(False, index=bars.index),
            long_price=pd.Series(np.nan, index=bars.index),
            short_price=pd.Series(np.nan, index=bars.index),
            stop_loss=pd.Series(np.nan, index=bars.index),
            take_profit=pd.Series(np.nan, index=bars.index),
            signal_type=pd.Series("none", index=bars.index),
            metadata={}
        )


class ShadowGammaSignalGenerator:
    """
    Generate signals from Shadow Gamma (hidden dealer positioning).

    Concentrated gamma = fragile = avoid / trade breakouts
    Dispersed gamma = stable = normal trading
    Dealers buying = bullish pressure
    Dealers selling = bearish pressure
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = get_config()
        self.enabled = config.get("enabled", True) if config else True
        self.stop_ticks = config.get("stop_ticks", 15) if config else 15
        self.target_ticks = config.get("target_ticks", 30) if config else 30
        self.tick_size = 0.25

    def generate(
        self,
        bars: pd.DataFrame,
        shadow_signals: pd.DataFrame,
        regime_result
    ) -> ShadowGammaSignal:
        if not self.enabled:
            return self._empty_signal(bars)

        df = bars.copy()

        for col in shadow_signals.columns:
            df[col] = shadow_signals[col].reindex(df.index, method="ffill")

        # Dealers buying = support
        long_dealers = (
            df["dealers_buy"] &
            (df["close"] > df["open"]) &
            (df["low"] > df["low"].shift(1))  # Higher low
        )

        # Dealers selling = resistance
        short_dealers = (
            df["dealers_sell"] &
            (df["close"] < df["open"]) &
            (df["high"] < df["high"].shift(1))  # Lower high
        )

        # Gamma concentrated = fragile = breakout trades
        long_concentrated = (
            df["gamma_concentrated"] &
            (df["close"] > df["high"].shift(1)) &  # Breakout
            (df["volume"] > df["volume"].rolling(20).mean() * 2.0)
        )

        short_concentrated = (
            df["gamma_concentrated"] &
            (df["close"] < df["low"].shift(1)) &  # Breakdown
            (df["volume"] > df["volume"].rolling(20).mean() * 2.0)
        )

        # Gamma dispersed = normal = mean reversion
        long_dispersed = (
            df["gamma_dispersed"] &
            (df["close"] < df["open"]) &
            (df["low"] < df["low"].rolling(10).min())
        )

        short_dispersed = (
            df["gamma_dispersed"] &
            (df["close"] > df["open"]) &
            (df["high"] > df["high"].rolling(10).max())
        )

        long_condition = long_dealers | long_concentrated | long_dispersed
        short_condition = short_dealers | short_concentrated | short_dispersed

        entry_price = df["close"]
        stop_long = entry_price - self.stop_ticks * self.tick_size
        stop_short = entry_price + self.stop_ticks * self.tick_size
        target_long = entry_price + self.target_ticks * self.tick_size
        target_short = entry_price - self.target_ticks * self.tick_size

        signal_type = pd.Series("none", index=df.index)
        signal_type[long_dealers & long_condition] = "shadow_gamma_dealers_long"
        signal_type[short_dealers & short_condition] = "shadow_gamma_dealers_short"
        signal_type[long_concentrated & long_condition] = "shadow_gamma_breakout_long"
        signal_type[short_concentrated & short_condition] = "shadow_gamma_breakout_short"
        signal_type[long_dispersed & long_condition] = "shadow_gamma_reversion_long"
        signal_type[short_dispersed & short_condition] = "shadow_gamma_reversion_short"

        return ShadowGammaSignal(
            long_entry=long_condition,
            short_entry=short_condition,
            long_price=entry_price,
            short_price=entry_price,
            stop_loss=pd.Series(np.where(long_condition, stop_long, np.where(short_condition, stop_short, np.nan)), index=df.index),
            take_profit=pd.Series(np.where(long_condition, target_long, np.where(short_condition, target_short, np.nan)), index=df.index),
            signal_type=signal_type,
            metadata={"type": "shadow_gamma"}
        )

    def _empty_signal(self, bars: pd.DataFrame) -> ShadowGammaSignal:
        return ShadowGammaSignal(
            long_entry=pd.Series(False, index=bars.index),
            short_entry=pd.Series(False, index=bars.index),
            long_price=pd.Series(np.nan, index=bars.index),
            short_price=pd.Series(np.nan, index=bars.index),
            stop_loss=pd.Series(np.nan, index=bars.index),
            take_profit=pd.Series(np.nan, index=bars.index),
            signal_type=pd.Series("none", index=bars.index),
            metadata={}
        )