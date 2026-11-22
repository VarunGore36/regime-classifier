"""CVD (Cumulative Volume Delta) and Delta features."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np

from config import get_config


@dataclass
class CVDResult:
    """CVD calculation results."""
    cvd: pd.Series                      # Cumulative Volume Delta
    delta: pd.Series                    # Per-bar delta (buy_vol - sell_vol)
    buy_volume: pd.Series               # Aggressive buy volume
    sell_volume: pd.Series              # Aggressive sell volume
    delta_ma: pd.Series                 # Moving average of delta
    cvd_divergence: pd.Series           # CVD vs price divergence


def calculate_cvd(
    footprint: pd.DataFrame,
    bars: Optional[pd.DataFrame] = None
) -> CVDResult:
    """
    Calculate CVD from footprint or bar data.

    Args:
        footprint: DataFrame with [timestamp, price, bid_volume, ask_volume, delta]
        bars: Optional pre-aggregated bars with buy_volume, sell_volume

    Returns:
        CVDResult
    """
    if bars is not None:
        df = bars.copy()
    else:
        # Aggregate footprint to bars
        df = footprint.groupby("timestamp").agg(
            buy_volume=("ask_volume", "sum"),
            sell_volume=("bid_volume", "sum"),
            delta=("delta", "sum"),
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("total_volume", "sum")
        ).reset_index()

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Per-bar delta
    df["delta"] = df["buy_volume"] - df["sell_volume"]

    # CVD (cumulative)
    df["cvd"] = df["delta"].cumsum()

    # Delta moving average
    config = get_config()
    ma_window = 20
    df["delta_ma"] = df["delta"].rolling(ma_window).mean()

    # CVD divergence with price
    df["price_high"] = df["high"].rolling(20).max()
    df["price_low"] = df["low"].rolling(20).min()
    df["cvd_high"] = df["cvd"].rolling(20).max()
    df["cvd_low"] = df["cvd"].rolling(20).min()

    # Bearish divergence: price higher high, CVD lower high
    bearish_div = (
        (df["high"] > df["price_high"].shift(1)) &
        (df["cvd"] < df["cvd_high"].shift(1))
    )

    # Bullish divergence: price lower low, CVD higher low
    bullish_div = (
        (df["low"] < df["price_low"].shift(1)) &
        (df["cvd"] > df["cvd_low"].shift(1))
    )

    df["cvd_divergence"] = 0
    df.loc[bearish_div, "cvd_divergence"] = -1
    df.loc[bullish_div, "cvd_divergence"] = 1

    return CVDResult(
        cvd=df["cvd"],
        delta=df["delta"],
        buy_volume=df["buy_volume"],
        sell_volume=df["sell_volume"],
        delta_ma=df["delta_ma"],
        cvd_divergence=df["cvd_divergence"]
    )


def calculate_delta_profile(
    footprint: pd.DataFrame,
    price_column: str = "price",
    delta_column: str = "delta"
) -> pd.DataFrame:
    """
    Calculate delta profile: delta per price level.

    Returns DataFrame with price levels and cumulative delta.
    """
    profile = footprint.groupby(price_column)[delta_column].sum().reset_index()
    profile = profile.sort_values(price_column).reset_index(drop=True)
    profile["cumulative_delta"] = profile[delta_column].cumsum()
    return profile


def get_delta_signals(
    cvd_result: CVDResult,
    bars: pd.DataFrame,
    delta_threshold: int = 100,
    divergence_lookback: int = 10
) -> pd.DataFrame:
    """
    Generate delta/CVD based signals.
    """
    df = bars.copy()
    df["cvd"] = cvd_result.cvd.values
    df["delta"] = cvd_result.delta.values
    df["delta_ma"] = cvd_result.delta_ma.values
    df["cvd_divergence"] = cvd_result.cvd_divergence.values

    # Delta signals
    df["delta_bullish"] = df["delta"] > delta_threshold
    df["delta_bearish"] = df["delta"] < -delta_threshold
    df["delta_neutral"] = df["delta"].abs() <= delta_threshold

    # CVD trend
    df["cvd_uptrend"] = df["cvd"] > df["cvd"].rolling(20).mean()
    df["cvd_downtrend"] = df["cvd"] < df["cvd"].rolling(20).mean()

    # CVD divergence signals (already calculated)
    df["bullish_divergence"] = df["cvd_divergence"] == 1
    df["bearish_divergence"] = df["cvd_divergence"] == -1

    # Delta exhaustion: delta extreme but price stalls
    df["delta_exhaustion_long"] = (
        (df["delta"] > delta_threshold * 2) &
        (df["close"] <= df["open"]) &
        (df["high"] - df["low"] < (df["high"].rolling(20).mean() - df["low"].rolling(20).mean()) * 0.5)
    )
    df["delta_exhaustion_short"] = (
        (df["delta"] < -delta_threshold * 2) &
        (df["close"] >= df["open"]) &
        (df["high"] - df["low"] < (df["high"].rolling(20).mean() - df["low"].rolling(20).mean()) * 0.5)
    )

    return df