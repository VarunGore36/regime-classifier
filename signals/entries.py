"""Entry signal generation."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np

from config import get_config
from features import get_profile_levels, get_vwap_signals, get_delta_signals, get_orderflow_signals
from regime import CombinedRegimeClassifier


@dataclass
class EntrySignal:
    """Entry signal result."""
    long_entry: pd.Series
    short_entry: pd.Series
    long_price: pd.Series
    short_price: pd.Series
    stop_loss: pd.Series
    take_profit: pd.Series
    signal_type: pd.Series
    metadata: dict


class POCReversionSignal:
    """POC mean reversion entry signal."""

    def __init__(self, config: Optional[dict] = None):
        cfg = get_config()
        self.lookback_bars = config.get("lookback_bars", cfg.signals.poc_reversion.lookback_bars) if config else cfg.signals.poc_reversion.lookback_bars
        self.entry_offset_ticks = config.get("entry_offset_ticks", cfg.signals.poc_reversion.entry_offset_ticks) if config else cfg.signals.poc_reversion.entry_offset_ticks
        self.stop_ticks = config.get("stop_ticks", cfg.signals.poc_reversion.stop_ticks) if config else cfg.signals.poc_reversion.stop_ticks
        self.target_ticks = config.get("target_ticks", cfg.signals.poc_reversion.target_ticks) if config else cfg.signals.poc_reversion.target_ticks
        self.enabled = config.get("enabled", cfg.signals.poc_reversion.enabled) if config else cfg.signals.poc_reversion.enabled

    def generate(self, bars: pd.DataFrame, volume_profile, regime_result) -> EntrySignal:
        if not self.enabled:
            return self._empty_signal(bars)

        levels = get_profile_levels(volume_profile)
        poc = levels["poc"]
        tick_size = 0.25

        df = bars.copy()
        df["poc"] = poc

        # Distance from POC
        df["dist_from_poc"] = (df["close"] - poc) / tick_size

        # Long: price below POC, bullish reversal signs
        long_condition = (
            (df["dist_from_poc"] < -self.entry_offset_ticks) &
            (df["close"] > df["open"]) &  # Bullish candle
            (df["volume"] > df["volume"].rolling(20).mean())  # Volume confirmation
        )

        # Short: price above POC, bearish reversal signs
        short_condition = (
            (df["dist_from_poc"] > self.entry_offset_ticks) &
            (df["close"] < df["open"]) &  # Bearish candle
            (df["volume"] > df["volume"].rolling(20).mean())
        )

        # Regime filter: only trade in positive GEX / low vol
        if hasattr(regime_result, "sub_regimes"):
            gex_regime = regime_result.sub_regimes.get("gex", pd.Series("neutral", index=df.index))
            rv_regime = regime_result.sub_regimes.get("rv", pd.Series("normal", index=df.index))

            long_condition = long_condition & (gex_regime == "positive_gex") & (rv_regime != "very_elevated_vol")
            short_condition = short_condition & (gex_regime == "positive_gex") & (rv_regime != "very_elevated_vol")

        entry_price_long = df["close"]
        entry_price_short = df["close"]
        stop_long = entry_price_long - self.stop_ticks * tick_size
        stop_short = entry_price_short + self.stop_ticks * tick_size
        target_long = entry_price_long + self.target_ticks * tick_size
        target_short = entry_price_short - self.target_ticks * tick_size

        return EntrySignal(
            long_entry=long_condition,
            short_entry=short_condition,
            long_price=entry_price_long,
            short_price=entry_price_short,
            stop_loss=pd.Series(np.where(long_condition, stop_long, np.where(short_condition, stop_short, np.nan)), index=df.index),
            take_profit=pd.Series(np.where(long_condition, target_long, np.where(short_condition, target_short, np.nan)), index=df.index),
            signal_type=pd.Series(np.where(long_condition, "poc_reversion_long", np.where(short_condition, "poc_reversion_short", "none")), index=df.index),
            metadata={"poc": poc, "type": "poc_reversion"}
        )

    def _empty_signal(self, bars: pd.DataFrame) -> EntrySignal:
        n = len(bars)
        return EntrySignal(
            long_entry=pd.Series(False, index=bars.index),
            short_entry=pd.Series(False, index=bars.index),
            long_price=pd.Series(np.nan, index=bars.index),
            short_price=pd.Series(np.nan, index=bars.index),
            stop_loss=pd.Series(np.nan, index=bars.index),
            take_profit=pd.Series(np.nan, index=bars.index),
            signal_type=pd.Series("none", index=bars.index),
            metadata={}
        )


class VWAPBandSignal:
    """VWAP band entry signal (reversion or breakout)."""

    def __init__(self, config: Optional[dict] = None):
        cfg = get_config()
        self.band = config.get("band", cfg.signals.vwap_bands.band) if config else cfg.signals.vwap_bands.band
        self.session_open_only = config.get("session_open_only", cfg.signals.vwap_bands.session_open_only) if config else cfg.signals.vwap_bands.session_open_only
        self.entry_offset_ticks = config.get("entry_offset_ticks", cfg.signals.vwap_bands.entry_offset_ticks) if config else cfg.signals.vwap_bands.entry_offset_ticks
        self.stop_ticks = config.get("stop_ticks", cfg.signals.vwap_bands.stop_ticks) if config else cfg.signals.vwap_bands.stop_ticks
        self.target_ticks = config.get("target_ticks", cfg.signals.vwap_bands.target_ticks) if config else cfg.signals.vwap_bands.target_ticks
        self.enabled = config.get("enabled", cfg.signals.vwap_bands.enabled) if config else cfg.signals.vwap_bands.enabled

    def generate(self, bars: pd.DataFrame, vwap_result, regime_result) -> EntrySignal:
        if not self.enabled:
            return self._empty_signal(bars)

        tick_size = 0.25
        signals_df = get_vwap_signals(bars, vwap_result, band=self.band, session_open_only=self.session_open_only)

        df = bars.copy()
        df["vwap"] = vwap_result.vwap.values
        df[f"upper_{self.band}"] = vwap_result.upper_bands[self.band].values
        df[f"lower_{self.band}"] = vwap_result.lower_bands[self.band].values

        # Reversion signals (mean reversion)
        long_reversion = signals_df[f"reversion_long_{self.band}"]
        short_reversion = signals_df[f"reversion_short_{self.band}"]

        # Breakout signals
        long_breakout = signals_df[f"breakout_long_{self.band}"]
        short_breakout = signals_df[f"breakout_short_{self.band}"]

        # Combine (prefer reversion in pos GEX, breakout in neg GEX)
        gex_regime = pd.Series("neutral", index=df.index)
        if hasattr(regime_result, "sub_regimes"):
            gex_regime = regime_result.sub_regimes.get("gex", gex_regime)

        long_condition = (long_reversion & (gex_regime == "positive_gex")) | (long_breakout & (gex_regime == "negative_gex"))
        short_condition = (short_reversion & (gex_regime == "positive_gex")) | (short_breakout & (gex_regime == "negative_gex"))

        entry_price_long = df["close"]
        entry_price_short = df["close"]
        stop_long = entry_price_long - self.stop_ticks * tick_size
        stop_short = entry_price_short + self.stop_ticks * tick_size
        target_long = entry_price_long + self.target_ticks * tick_size
        target_short = entry_price_short - self.target_ticks * tick_size

        signal_type = pd.Series("none", index=df.index)
        signal_type[long_reversion & long_condition] = "vwap_reversion_long"
        signal_type[short_reversion & short_condition] = "vwap_reversion_short"
        signal_type[long_breakout & long_condition] = "vwap_breakout_long"
        signal_type[short_breakout & short_condition] = "vwap_breakout_short"

        return EntrySignal(
            long_entry=long_condition,
            short_entry=short_condition,
            long_price=entry_price_long,
            short_price=entry_price_short,
            stop_loss=pd.Series(np.where(long_condition, stop_long, np.where(short_condition, stop_short, np.nan)), index=df.index),
            take_profit=pd.Series(np.where(long_condition, target_long, np.where(short_condition, target_short, np.nan)), index=df.index),
            signal_type=signal_type,
            metadata={"band": self.band, "type": "vwap_bands"}
        )

    def _empty_signal(self, bars: pd.DataFrame) -> EntrySignal:
        n = len(bars)
        return EntrySignal(
            long_entry=pd.Series(False, index=bars.index),
            short_entry=pd.Series(False, index=bars.index),
            long_price=pd.Series(np.nan, index=bars.index),
            short_price=pd.Series(np.nan, index=bars.index),
            stop_loss=pd.Series(np.nan, index=bars.index),
            take_profit=pd.Series(np.nan, index=bars.index),
            signal_type=pd.Series("none", index=bars.index),
            metadata={}
        )


class BreakFadeSignal:
    """Break vs Fade entry signal."""

    def __init__(self, config: Optional[dict] = None):
        cfg = get_config()
        self.break_enabled = config.get("break_enabled", cfg.signals.break_trades.enabled) if config else cfg.signals.break_trades.enabled
        self.fade_enabled = config.get("fade_enabled", cfg.signals.fade_trades.enabled) if config else cfg.signals.fade_trades.enabled
        self.break_volume_mult = config.get("break_volume_mult", cfg.signals.break_trades.volume_surge_mult) if config else cfg.signals.break_trades.volume_surge_mult
        self.fade_volume_mult = config.get("fade_volume_mult", cfg.signals.fade_trades.volume_surge_mult) if config else cfg.signals.fade_trades.volume_surge_mult
        self.break_delta_thresh = config.get("break_delta_thresh", cfg.signals.break_trades.delta_threshold) if config else cfg.signals.break_trades.delta_threshold
        self.fade_delta_thresh = config.get("fade_delta_thresh", cfg.signals.fade_trades.delta_threshold) if config else cfg.signals.fade_trades.delta_threshold
        self.break_stop_ticks = config.get("break_stop_ticks", cfg.signals.break_trades.stop_ticks) if config else cfg.signals.break_trades.stop_ticks
        self.break_target_ticks = config.get("break_target_ticks", cfg.signals.break_trades.target_ticks) if config else cfg.signals.break_trades.target_ticks
        self.fade_stop_ticks = config.get("fade_stop_ticks", cfg.signals.fade_trades.stop_ticks) if config else cfg.signals.fade_trades.stop_ticks
        self.fade_target_ticks = config.get("fade_target_ticks", cfg.signals.fade_trades.target_ticks) if config else cfg.signals.fade_trades.target_ticks

    def generate(self, bars: pd.DataFrame, footprint: pd.DataFrame, levels: dict, regime_result) -> EntrySignal:
        tick_size = 0.25
        df = bars.copy()

        # Calculate delta from footprint
        if footprint is not None:
            delta = footprint.groupby("timestamp")["delta"].sum()
            df["delta"] = delta.reindex(df["timestamp"]).values
        else:
            df["delta"] = 0

        # Key levels
        vah = levels.get("vah", df["high"].rolling(20).max())
        val = levels.get("val", df["low"].rolling(20).min())
        hvn_prices = levels.get("hvn_prices", [])
        lvn_prices = levels.get("lvn_prices", [])

        # Break conditions: price breaks level with volume/delta surge
        avg_volume = df["volume"].rolling(20).mean()

        long_break = (
            (df["close"] > df["high"].shift(1)) &  # Break above prior high
            (df["volume"] > avg_volume * self.break_volume_mult) &
            (df["delta"] > self.break_delta_thresh)
        )

        short_break = (
            (df["close"] < df["low"].shift(1)) &  # Break below prior low
            (df["volume"] > avg_volume * self.break_volume_mult) &
            (df["delta"] < -self.break_delta_thresh)
        )

        # Fade conditions: price touches level and reverses
        long_fade = (
            (df["low"] <= val + 2 * tick_size) &
            (df["close"] > df["open"]) &
            (df["volume"] > avg_volume * self.fade_volume_mult) &
            (df["delta"] > self.fade_delta_thresh)
        )

        short_fade = (
            (df["high"] >= vah - 2 * tick_size) &
            (df["close"] < df["open"]) &
            (df["volume"] > avg_volume * self.fade_volume_mult) &
            (df["delta"] < -self.fade_delta_thresh)
        )

        # Regime filter
        gex_regime = pd.Series("neutral", index=df.index)
        if hasattr(regime_result, "sub_regimes"):
            gex_regime = regime_result.sub_regimes.get("gex", gex_regime)

        # In positive GEX: prefer fades (mean reversion)
        # In negative GEX: prefer breaks (trend following)
        long_condition = (long_fade & self.fade_enabled & (gex_regime == "positive_gex")) | \
                         (long_break & self.break_enabled & (gex_regime == "negative_gex"))
        short_condition = (short_fade & self.fade_enabled & (gex_regime == "positive_gex")) | \
                          (short_break & self.break_enabled & (gex_regime == "negative_gex"))

        entry_price_long = df["close"]
        entry_price_short = df["close"]
        stop_long = pd.Series(np.where(
            long_fade & long_condition, entry_price_long - self.fade_stop_ticks * tick_size,
            np.where(long_break & long_condition, entry_price_long - self.break_stop_ticks * tick_size, np.nan)
        ), index=df.index)
        stop_short = pd.Series(np.where(
            short_fade & short_condition, entry_price_short + self.fade_stop_ticks * tick_size,
            np.where(short_break & short_condition, entry_price_short + self.break_stop_ticks * tick_size, np.nan)
        ), index=df.index)
        target_long = pd.Series(np.where(
            long_fade & long_condition, entry_price_long + self.fade_target_ticks * tick_size,
            np.where(long_break & long_condition, entry_price_long + self.break_target_ticks * tick_size, np.nan)
        ), index=df.index)
        target_short = pd.Series(np.where(
            short_fade & short_condition, entry_price_short - self.fade_target_ticks * tick_size,
            np.where(short_break & short_condition, entry_price_short - self.break_target_ticks * tick_size, np.nan)
        ), index=df.index)

        # Combine stop/target for EntrySignal (single series each)
        stop_loss = pd.Series(np.nan, index=df.index)
        take_profit = pd.Series(np.nan, index=df.index)
        stop_loss[long_condition] = stop_long[long_condition]
        stop_loss[short_condition] = stop_short[short_condition]
        take_profit[long_condition] = target_long[long_condition]
        take_profit[short_condition] = target_short[short_condition]

        signal_type = pd.Series("none", index=df.index)
        signal_type[long_fade & long_condition] = "fade_long"
        signal_type[short_fade & short_condition] = "fade_short"
        signal_type[long_break & long_condition] = "break_long"
        signal_type[short_break & short_condition] = "break_short"

        return EntrySignal(
            long_entry=long_condition,
            short_entry=short_condition,
            long_price=entry_price_long,
            short_price=entry_price_short,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_type=signal_type,
            metadata={"type": "break_fade"}
        )

    def _empty_signal(self, bars: pd.DataFrame) -> EntrySignal:
        n = len(bars)
        return EntrySignal(
            long_entry=pd.Series(False, index=bars.index),
            short_entry=pd.Series(False, index=bars.index),
            long_price=pd.Series(np.nan, index=bars.index),
            short_price=pd.Series(np.nan, index=bars.index),
            stop_loss=pd.Series(np.nan, index=bars.index),
            take_profit=pd.Series(np.nan, index=bars.index),
            signal_type=pd.Series("none", index=bars.index),
            metadata={}
        )