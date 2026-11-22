"""Realized Volatility feature calculation."""

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np

from config import get_config


@dataclass
class RealizedVolResult:
    """Realized Volatility results."""
    rv: pd.Series                      # Realized volatility per bar
    rv_zscore: pd.Series               # Z-score of RV
    log_returns: pd.Series             # Log returns
    volatility_acceleration: pd.Series # Change in RV (ΔRV)
    regime: pd.Series                  # Volatility regime classification


def calculate_realized_volatility(
    bars: pd.DataFrame,
    window_minutes: Optional[int] = None,
    zscore_windows: Optional[List[int]] = None,
    elevated_threshold: Optional[float] = None,
    very_elevated_threshold: Optional[float] = None
) -> RealizedVolResult:
    """
    Calculate Realized Volatility (RV) and Z-scores.

    Formula:
    - Log return: rt = ln(Pt / Pt-1)
    - RV = sqrt(sum(rt^2)) over window
    - Z-score = (RV - mean(RV)) / std(RV)

    Args:
        bars: DataFrame with [timestamp, close]
        window_minutes: RV calculation window (default from config)
        zscore_windows: List of windows for Z-score calculation
        elevated_threshold: Z-score threshold for elevated vol
        very_elevated_threshold: Z-score threshold for very elevated vol

    Returns:
        RealizedVolResult
    """
    config = get_config()

    if window_minutes is None:
        window_minutes = config.realized_vol.window_minutes
    if zscore_windows is None:
        zscore_windows = config.realized_vol.zscore_windows
    if elevated_threshold is None:
        elevated_threshold = config.realized_vol.elevated_threshold
    if very_elevated_threshold is None:
        very_elevated_threshold = config.realized_vol.very_elevated_threshold

    df = bars.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Realized Volatility (sum of squared log returns over window)
    df["rv"] = np.sqrt(df["log_return"] ** 2 * window_minutes)  # Per bar RV scaled
    df["rv"] = df["log_return"].rolling(window_minutes).apply(
        lambda x: np.sqrt(np.sum(x ** 2)) * np.sqrt(390 / window_minutes)  # Annualized
    )

    # Z-scores for multiple windows
    zscores = {}
    for w in zscore_windows:
        rv_mean = df["rv"].rolling(w, min_periods=w//2).mean()
        rv_std = df["rv"].rolling(w, min_periods=w//2).std()
        zscores[w] = (df["rv"] - rv_mean) / rv_std

    # Use the longest window as primary Z-score
    primary_z = zscores[max(zscore_windows)]

    # Volatility acceleration (change in RV)
    df["vol_acceleration"] = df["rv"].diff()

    # Regime classification
    def classify_regime(z):
        if pd.isna(z):
            return "unknown"
        elif z < 0:
            return "low_vol"
        elif z < 1:
            return "normal_vol"
        elif z < 2:
            return "elevated_vol"
        else:
            return "very_elevated_vol"

    df["rv_regime"] = primary_z.apply(classify_regime)

    return RealizedVolResult(
        rv=df["rv"],
        rv_zscore=primary_z,
        log_returns=df["log_return"],
        volatility_acceleration=df["vol_acceleration"],
        regime=df["rv_regime"]
    )


def calculate_rolling_volatility(
    bars: pd.DataFrame,
    window: int = 30,
    annualize: bool = True
) -> pd.Series:
    """
    Simple rolling volatility (std of returns).
    """
    df = bars.copy()
    df["returns"] = df["close"].pct_change()
    vol = df["returns"].rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(390 * 252)  # Annualize from 1-min bars
    return vol


def calculate_parkinson_volatility(
    bars: pd.DataFrame,
    window: int = 30
) -> pd.Series:
    """
    Parkinson volatility using high/low range.
    More efficient than close-to-close.
    """
    df = bars.copy()
    df["hl_range"] = np.log(df["high"] / df["low"])
    # Parkinson estimator: sqrt(1/(4*ln(2)) * mean(ln(H/L)^2))
    factor = 1.0 / (4.0 * np.log(2.0))
    park_vol = np.sqrt(factor * df["hl_range"].rolling(window).apply(lambda x: np.mean(x ** 2)))
    return park_vol * np.sqrt(390 * 252)  # Annualize


def calculate_garman_klass_volatility(
    bars: pd.DataFrame,
    window: int = 30
) -> pd.Series:
    """
    Garman-Klass volatility using OHLC.
    Even more efficient.
    """
    df = bars.copy()
    df["log_hl"] = np.log(df["high"] / df["low"]) ** 2
    df["log_co"] = np.log(df["close"] / df["open"]) ** 2
    gk = 0.5 * df["log_hl"] - (2 * np.log(2) - 1) * df["log_co"]
    gk_vol = np.sqrt(gk.rolling(window).mean())
    return gk_vol * np.sqrt(390 * 252)  # Annualize


def get_volatility_signals(
    rv_result: RealizedVolResult,
    zscore_threshold: float = 1.0
) -> pd.DataFrame:
    """
    Generate volatility regime signals.
    """
    df = pd.DataFrame({
        "rv": rv_result.rv,
        "rv_zscore": rv_result.rv_zscore,
        "vol_acceleration": rv_result.volatility_acceleration,
        "regime": rv_result.regime
    })

    df["vol_expanding"] = df["vol_acceleration"] > 0
    df["vol_contracting"] = df["vol_acceleration"] < 0
    df["elevated_vol"] = df["rv_zscore"] > zscore_threshold
    df["very_elevated_vol"] = df["rv_zscore"] > 2.0
    df["low_vol"] = df["rv_zscore"] < 0

    # Vol regime change signals
    df["regime_change"] = df["regime"] != df["regime"].shift(1)
    df["vol_spike"] = (df["rv_zscore"] > 2.0) & (df["rv_zscore"].shift(1) <= 2.0)
    df["vol_crush"] = (df["rv_zscore"] < 0) & (df["rv_zscore"].shift(1) >= 0)

    return df