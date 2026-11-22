"""VWAP (Volume Weighted Average Price) feature calculation."""

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np

from config import get_config


@dataclass
class VWAPResult:
    """VWAP calculation results."""
    vwap: pd.Series                    # VWAP value per bar
    upper_bands: dict                  # Dict of band_mult -> pd.Series
    lower_bands: dict                  # Dict of band_mult -> pd.Series
    std_dev: pd.Series                 # Rolling standard deviation
    typical_price: pd.Series           # (H+L+C)/3 per bar


def calculate_vwap(
    bars: pd.DataFrame,
    bands: Optional[List[float]] = None,
    anchor_session_open: bool = True,
    session_open_hour: int = 9,
    session_open_minute: int = 30,
    timezone: str = "America/New_York"
) -> VWAPResult:
    """
    Calculate VWAP with standard deviation bands.

    Args:
        bars: DataFrame with [timestamp, open, high, low, close, volume]
        bands: List of standard deviation multipliers (default from config)
        anchor_session_open: Reset VWAP at session open
        session_open_hour: Session open hour (24h)
        session_open_minute: Session open minute
        timezone: Timezone for session times

    Returns:
        VWAPResult with vwap, bands, std_dev
    """
    config = get_config()

    if bands is None:
        bands = config.vwap.bands

    df = bars.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Typical price
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["pv"] = df["typical_price"] * df["volume"]

    # Determine session anchors
    if anchor_session_open:
        df["session_date"] = df["timestamp"].dt.tz_convert(timezone).dt.date
        # Check if timestamp is at or after session open
        session_open = pd.Timestamp(f"{session_open_hour}:{session_open_minute}:00").time()
        df["is_session_start"] = (
            (df["timestamp"].dt.tz_convert(timezone).dt.time >= session_open) &
            (df["timestamp"].dt.tz_convert(timezone).dt.time < pd.Timestamp(f"{session_open_hour}:{session_open_minute+1}:00").time())
        )
        # Mark first bar of each session
        df["session_id"] = df.groupby("session_date").cumcount()
        df["new_session"] = df["session_id"] == 0
    else:
        df["new_session"] = False

    # Calculate cumulative PV and volume per session
    df["cum_pv"] = df.groupby((df["new_session"]).cumsum())["pv"].cumsum()
    df["cum_vol"] = df.groupby((df["new_session"]).cumsum())["volume"].cumsum()

    # VWAP
    df["vwap"] = df["cum_pv"] / df["cum_vol"]

    # Standard deviation of typical price from VWAP
    df["price_dev"] = df["typical_price"] - df["vwap"]
    df["price_dev_sq"] = df["price_dev"] ** 2
    df["cum_price_dev_sq"] = df.groupby((df["new_session"]).cumsum())["price_dev_sq"].cumsum()
    df["std_dev"] = np.sqrt(df["cum_price_dev_sq"] / df["cum_vol"])

    # Bands
    upper_bands = {}
    lower_bands = {}
    for band in bands:
        upper_bands[band] = df["vwap"] + band * df["std_dev"]
        lower_bands[band] = df["vwap"] - band * df["std_dev"]

    return VWAPResult(
        vwap=df["vwap"],
        upper_bands=upper_bands,
        lower_bands=lower_bands,
        std_dev=df["std_dev"],
        typical_price=df["typical_price"]
    )


def calculate_rolling_vwap(
    bars: pd.DataFrame,
    window: int = 20,
    bands: Optional[List[float]] = None
) -> VWAPResult:
    """
    Calculate rolling VWAP (not session-anchored).
    Useful for shorter-term analysis.
    """
    config = get_config()
    if bands is None:
        bands = config.vwap.bands

    df = bars.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["pv"] = df["typical_price"] * df["volume"]

    df["cum_pv"] = df["pv"].rolling(window, min_periods=1).sum()
    df["cum_vol"] = df["volume"].rolling(window, min_periods=1).sum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"]

    df["price_dev"] = df["typical_price"] - df["vwap"]
    df["price_dev_sq"] = df["price_dev"] ** 2
    df["cum_price_dev_sq"] = df["price_dev_sq"].rolling(window, min_periods=1).sum()
    df["std_dev"] = np.sqrt(df["cum_price_dev_sq"] / df["cum_vol"])

    upper_bands = {}
    lower_bands = {}
    for band in bands:
        upper_bands[band] = df["vwap"] + band * df["std_dev"]
        lower_bands[band] = df["vwap"] - band * df["std_dev"]

    return VWAPResult(
        vwap=df["vwap"],
        upper_bands=upper_bands,
        lower_bands=lower_bands,
        std_dev=df["std_dev"],
        typical_price=df["typical_price"]
    )


def get_vwap_signals(
    bars: pd.DataFrame,
    vwap_result: VWAPResult,
    band: float = 3.0,
    session_open_only: bool = False
) -> pd.DataFrame:
    """
    Generate VWAP-based signals.

    Returns DataFrame with signal columns.
    """
    df = bars.copy()
    df["vwap"] = vwap_result.vwap.values
    df[f"vwap_upper_{band}"] = vwap_result.upper_bands[band].values
    df[f"vwap_lower_{band}"] = vwap_result.lower_bands[band].values

    # Price relative to VWAP
    df["above_vwap"] = df["close"] > df["vwap"]
    df["below_vwap"] = df["close"] < df["vwap"]

    # Touch/cross bands
    df[f"touch_upper_{band}"] = (df["high"] >= df[f"vwap_upper_{band}"]) & (df["low"] <= df[f"vwap_upper_{band}"])
    df[f"touch_lower_{band}"] = (df["low"] <= df[f"vwap_lower_{band}"]) & (df["high"] >= df[f"vwap_lower_{band}"])

    # Cross bands
    df[f"cross_above_upper_{band}"] = (df["close"] > df[f"vwap_upper_{band}"]) & (df["close"].shift(1) <= df[f"vwap_upper_{band}"].shift(1))
    df[f"cross_below_lower_{band}"] = (df["close"] < df[f"vwap_lower_{band}"]) & (df["close"].shift(1) >= df[f"vwap_lower_{band}"].shift(1))

    # Mean reversion signals
    df[f"reversion_long_{band}"] = df[f"touch_lower_{band}"] & (df["close"] > df["open"])  # Touch lower, bullish candle
    df[f"reversion_short_{band}"] = df[f"touch_upper_{band}"] & (df["close"] < df["open"])  # Touch upper, bearish candle

    # Breakout signals
    df[f"breakout_long_{band}"] = df[f"cross_above_upper_{band}"] & (df["volume"] > df["volume"].rolling(20).mean() * 1.5)
    df[f"breakout_short_{band}"] = df[f"cross_below_lower_{band}"] & (df["volume"] > df["volume"].rolling(20).mean() * 1.5)

    if session_open_only:
        # Only signal at session open (first 30 min)
        df["session_minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - (9 * 60 + 30)
        session_mask = (df["session_minute"] >= 0) & (df["session_minute"] <= 30)
        signal_cols = [c for c in df.columns if c.startswith(("reversion_", "breakout_"))]
        df.loc[~session_mask, signal_cols] = False

    return df