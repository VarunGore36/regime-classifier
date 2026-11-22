"""Absorption, Exhaustion, and Trapped Traders detection."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np

from config import get_config


@dataclass
class AbsorptionResult:
    """Absorption detection results."""
    absorption_bullish: pd.Series
    absorption_bearish: pd.Series
    absorption_strength: pd.Series


@dataclass
class ExhaustionResult:
    """Exhaustion detection results."""
    exhaustion_bullish: pd.Series
    exhaustion_bearish: pd.Series
    exhaustion_strength: pd.Series


@dataclass
class TrappedTradersResult:
    """Trapped traders detection results."""
    trapped_long: pd.Series
    trapped_short: pd.Series


def detect_absorption(
    footprint: pd.DataFrame,
    min_delta_divergence: int = 50,
    max_price_move_ticks: int = 5,
    min_volume: int = 100,
    tick_size: float = 0.25
) -> AbsorptionResult:
    """
    Detect absorption: aggressive delta but price doesn't move.

    Bullish absorption: aggressive buying (positive delta) but price doesn't go down
    Bearish absorption: aggressive selling (negative delta) but price doesn't go up
    """
    bars = footprint.groupby("timestamp").agg(
        total_delta=("delta", "sum"),
        total_volume=("total_volume", "sum"),
        price_change=("price", lambda x: x.iloc[-1] - x.iloc[0]),
        high=("price", "max"),
        low=("price", "min"),
        open=("price", "first"),
        close=("price", "last")
    ).reset_index()

    bars["price_move_ticks"] = (bars["high"] - bars["low"]) / tick_size

    # Bullish absorption: buyers aggressive, price holds or rises
    bullish = (
        (bars["total_delta"] > min_delta_divergence) &
        (bars["price_move_ticks"] <= max_price_move_ticks) &
        (bars["total_volume"] >= min_volume) &
        (bars["price_change"] >= 0)
    )

    # Bearish absorption: sellers aggressive, price holds or falls
    bearish = (
        (bars["total_delta"] < -min_delta_divergence) &
        (bars["price_move_ticks"] <= max_price_move_ticks) &
        (bars["total_volume"] >= min_volume) &
        (bars["price_change"] <= 0)
    )

    # Strength = delta / price_move (higher = stronger absorption)
    bars["absorption_strength"] = np.where(
        bars["price_move_ticks"] > 0,
        bars["total_delta"].abs() / bars["price_move_ticks"],
        bars["total_delta"].abs() * 10
    )

    return AbsorptionResult(
        absorption_bullish=bullish,
        absorption_bearish=bearish,
        absorption_strength=bars["absorption_strength"]
    )


def detect_exhaustion(
    footprint: pd.DataFrame,
    delta_reversal_threshold: int = 100,
    volume_dryup_mult: float = 0.3,
    max_bars: int = 3,
    tick_size: float = 0.25
) -> ExhaustionResult:
    """
    Detect exhaustion: delta reverses, volume dries up, price stalls.
    """
    bars = footprint.groupby("timestamp").agg(
        total_delta=("delta", "sum"),
        total_volume=("total_volume", "sum"),
        price_change=("price", lambda x: x.iloc[-1] - x.iloc[0]),
        high=("price", "max"),
        low=("price", "min"),
        open=("price", "first"),
        close=("price", "last")
    ).reset_index().sort_values("timestamp")

    bars["prev_delta"] = bars["total_delta"].shift(1)
    bars["prev_volume"] = bars["total_volume"].shift(1)
    bars["delta_reversal"] = bars["total_delta"] * bars["prev_delta"] < 0
    bars["volume_dryup"] = bars["total_volume"] < (bars["prev_volume"] * volume_dryup_mult)

    # Bullish exhaustion: sellers exhausted, buyers step in
    bullish = (
        (bars["prev_delta"] < -delta_reversal_threshold) &
        (bars["total_delta"] > 0) &
        bars["volume_dryup"] &
        (bars["price_change"] >= 0)
    )

    # Bearish exhaustion: buyers exhausted, sellers step in
    bearish = (
        (bars["prev_delta"] > delta_reversal_threshold) &
        (bars["total_delta"] < 0) &
        bars["volume_dryup"] &
        (bars["price_change"] <= 0)
    )

    # Strength based on delta reversal magnitude and volume dryup
    bars["exhaustion_strength"] = np.where(
        bars["delta_reversal"] & bars["volume_dryup"],
        bars["prev_delta"].abs() / bars["prev_volume"] * (1 - volume_dryup_mult),
        0
    )

    return ExhaustionResult(
        exhaustion_bullish=bullish,
        exhaustion_bearish=bearish,
        exhaustion_strength=bars["exhaustion_strength"]
    )


def detect_trapped_traders(
    footprint: pd.DataFrame,
    imbalance_ratio: float = 3.0,
    confirmation_bars: int = 2,
    tick_size: float = 0.25
) -> TrappedTradersResult:
    """
    Detect trapped traders: large imbalance, price moves against them.

    Trapped longs: heavy buying, then price drops
    Trapped shorts: heavy selling, then price rises
    """
    bars = footprint.groupby("timestamp").agg(
        total_delta=("delta", "sum"),
        total_volume=("total_volume", "sum"),
        buy_volume=("ask_volume", "sum"),
        sell_volume=("bid_volume", "sum"),
        open=("price", "first"),
        close=("price", "last"),
        high=("price", "max"),
        low=("price", "min")
    ).reset_index().sort_values("timestamp")

    bars["imbalance"] = bars["buy_volume"] / bars["sell_volume"].replace(0, 1)
    bars["large_buy_imbalance"] = bars["imbalance"] > imbalance_ratio
    bars["large_sell_imbalance"] = bars["imbalance"] < (1 / imbalance_ratio)

    bars["trapped_long"] = False
    bars["trapped_short"] = False

    for i in range(len(bars) - confirmation_bars):
        if bars.iloc[i]["large_buy_imbalance"]:
            future_closes = bars.iloc[i+1:i+1+confirmation_bars]["close"]
            if all(future_closes < bars.iloc[i]["close"]):
                bars.iloc[i, bars.columns.get_loc("trapped_long")] = True

        if bars.iloc[i]["large_sell_imbalance"]:
            future_closes = bars.iloc[i+1:i+1+confirmation_bars]["close"]
            if all(future_closes > bars.iloc[i]["close"]):
                bars.iloc[i, bars.columns.get_loc("trapped_short")] = True

    return TrappedTradersResult(
        trapped_long=bars["trapped_long"],
        trapped_short=bars["trapped_short"]
    )


def detect_delta_divergence(
    footprint: pd.DataFrame,
    bars: pd.DataFrame,
    lookback: int = 10
) -> pd.DataFrame:
    """
    Detect delta divergence: price makes new extreme but delta doesn't confirm.
    """
    df = bars.copy()
    df["cvd"] = footprint.groupby("timestamp")["delta"].sum().cumsum().values

    df["price_high"] = df["high"].rolling(lookback).max()
    df["price_low"] = df["low"].rolling(lookback).min()
    df["cvd_high"] = df["cvd"].rolling(lookback).max()
    df["cvd_low"] = df["cvd"].rolling(lookback).min()

    df["bearish_divergence"] = (
        (df["high"] > df["price_high"].shift(1)) &
        (df["cvd"] < df["cvd_high"].shift(1))
    )
    df["bullish_divergence"] = (
        (df["low"] < df["price_low"].shift(1)) &
        (df["cvd"] > df["cvd_low"].shift(1))
    )

    return df


def get_orderflow_signals(
    absorption: AbsorptionResult,
    exhaustion: ExhaustionResult,
    trapped: TrappedTradersResult,
    bars: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine all order flow signals into single DataFrame.
    """
    df = bars.copy()
    df["absorption_bullish"] = absorption.absorption_bullish.values
    df["absorption_bearish"] = absorption.absorption_bearish.values
    df["absorption_strength"] = absorption.absorption_strength.values

    df["exhaustion_bullish"] = exhaustion.exhaustion_bullish.values
    df["exhaustion_bearish"] = exhaustion.exhaustion_bearish.values
    df["exhaustion_strength"] = exhaustion.exhaustion_strength.values

    df["trapped_long"] = trapped.trapped_long.values
    df["trapped_short"] = trapped.trapped_short.values

    # Combined signals
    df["of_long_signal"] = (
        df["absorption_bullish"] |
        df["exhaustion_bullish"] |
        df["trapped_short"]  # Trapped shorts = bullish
    )
    df["of_short_signal"] = (
        df["absorption_bearish"] |
        df["exhaustion_bearish"] |
        df["trapped_long"]  # Trapped longs = bearish
    )

    return df