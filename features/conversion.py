"""NQ/QQQ price conversion utilities."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np

from config import get_config


@dataclass
class ConversionResult:
    """NQ/QQQ conversion results."""
    ratio: pd.Series                    # NQ / QQQ ratio over time
    nq_levels: dict                     # Key QQQ levels converted to NQ
    qqq_levels: dict                    # Key NQ levels converted to QQQ


def calculate_conversion_ratio(
    nq_bars: pd.DataFrame,
    qqq_bars: pd.DataFrame,
    on: str = "timestamp"
) -> pd.Series:
    """
    Calculate NQ/QQQ conversion ratio.

    Aligns both series on timestamp and computes ratio.
    """
    nq = nq_bars.set_index(on)["close"].rename("nq_close")
    qqq = qqq_bars.set_index(on)["close"].rename("qqq_close")

    aligned = pd.concat([nq, qqq], axis=1).dropna()
    ratio = aligned["nq_close"] / aligned["qqq_close"]
    ratio.name = "nq_qqq_ratio"
    return ratio


def convert_qqq_to_nq(
    qqq_level: float,
    ratio: float
) -> float:
    """
    Convert QQQ price level to NQ equivalent.

    Formula: NQ_level = QQQ_level * (NQ_price / QQQ_price)
    """
    return qqq_level * ratio


def convert_nq_to_qqq(
    nq_level: float,
    ratio: float
) -> float:
    """
    Convert NQ price level to QQQ equivalent.

    Formula: QQQ_level = NQ_level / (NQ_price / QQQ_price)
    """
    return nq_level / ratio


def convert_key_levels(
    qqq_levels: dict,
    nq_price: float,
    qqq_price: float
) -> dict:
    """
    Convert dictionary of QQQ key levels to NQ.

    Args:
        qqq_levels: Dict of {level_name: qqq_price}
        nq_price: Current NQ price
        qqq_price: Current QQQ price

    Returns:
        Dict of {level_name: nq_price}
    """
    ratio = nq_price / qqq_price
    return {name: convert_qqq_to_nq(price, ratio) for name, price in qqq_levels.items()}


def get_conversion_signals(
    nq_bars: pd.DataFrame,
    qqq_bars: pd.DataFrame,
    qqq_key_levels: dict,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Generate NQ levels from QQQ key levels and check for reactions.

    Args:
        nq_bars: NQ OHLCV bars
        qqq_bars: QQQ OHLCV bars
        qqq_key_levels: Dict of {name: price} for QQQ (gamma_wall, poc, etc.)
        lookback: Bars to check for reaction

    Returns:
        DataFrame with NQ levels and reaction signals
    """
    # Calculate rolling ratio
    ratio_series = calculate_conversion_ratio(nq_bars, qqq_bars)

    # Align to NQ bars
    nq = nq_bars.copy().set_index("timestamp")
    nq["ratio"] = ratio_series.reindex(nq.index, method="ffill")

    # Convert QQQ levels to NQ for each bar
    for level_name, qqq_price in qqq_key_levels.items():
        nq[f"nq_{level_name}"] = nq["ratio"] * qqq_price

    # Check price reactions at converted levels
    tick_size = 0.25
    tolerance_ticks = 4  # 1 point tolerance

    for level_name in qqq_key_levels.keys():
        col = f"nq_{level_name}"
        nq[f"at_{level_name}"] = (
            (nq["high"] >= nq[col] - tolerance_ticks * tick_size) &
            (nq["low"] <= nq[col] + tolerance_ticks * tick_size)
        )

        # Reaction: price touches level and reverses
        nq[f"reaction_{level_name}_long"] = (
            nq[f"at_{level_name}"] &
            (nq["close"] > nq["open"]) &
            (nq["low"] <= nq[col])
        )
        nq[f"reaction_{level_name}_short"] = (
            nq[f"at_{level_name}"] &
            (nq["close"] < nq["open"]) &
            (nq["high"] >= nq[col])
        )

    return nq.reset_index()


def dynamic_conversion_ratio(
    nq_bars: pd.DataFrame,
    qqq_bars: pd.DataFrame,
    window: int = 390,  # 1 day of 1-min bars
    min_periods: int = 100
) -> pd.Series:
    """
    Calculate dynamic conversion ratio using rolling regression.

    More robust than simple ratio for handling drift.
    """
    nq = nq_bars.set_index("timestamp")["close"]
    qqq = qqq_bars.set_index("timestamp")["close"]

    aligned = pd.concat([nq, qqq], axis=1).dropna()
    aligned.columns = ["nq", "qqq"]

    # Rolling linear regression: NQ = alpha + beta * QQQ
    # Beta is the conversion ratio
    def rolling_beta(x, y):
        if len(x) < min_periods:
            return np.nan
        x = x.values
        y = y.values
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov = np.sum((x - x_mean) * (y - y_mean))
        var = np.sum((x - x_mean) ** 2)
        return cov / var if var > 0 else np.nan

    beta = aligned["qqq"].rolling(window).apply(
        lambda q: rolling_beta(aligned.loc[q.index, "qqq"], aligned.loc[q.index, "nq"]),
        raw=False
    )
    beta.name = "dynamic_ratio"
    return beta