"""Exit signal generation."""

from dataclasses import dataclass
from typing: Optional
import pandas as pd
import numpy as np

from config import get_config


@dataclass
class ExitSignal:
    """Exit signal result."""
    exit_long: pd.Series
    exit_short: pd.Series
    exit_price: pd.Series
    exit_reason: pd.Series


class StopLossExit:
    """Stop loss exit."""

    def __init__(self, stop_ticks: int = 20, tick_size: float = 0.25):
        self.stop_ticks = stop_ticks
        self.tick_size = tick_size

    def generate(self, bars: pd.DataFrame, entry_price: pd.Series, position: pd.Series) -> ExitSignal:
        """
        Generate stop loss exits.

        Args:
            bars: OHLCV bars
            entry_price: Entry price per bar (NaN when not in position)
            position: 1 for long, -1 for short, 0 for flat
        """
        df = bars.copy()
        df["entry_price"] = entry_price
        df["position"] = position

        # Long stop loss
        long_stop = df["entry_price"] - self.stop_ticks * self.tick_size
        long_exit = (df["position"] == 1) & (df["low"] <= long_stop)

        # Short stop loss
        short_stop = df["entry_price"] + self.stop_ticks * self.tick_size
        short_exit = (df["position"] == -1) & (df["high"] >= short_stop)

        exit_price = pd.Series(np.nan, index=df.index)
        exit_price[long_exit] = long_stop[long_exit]
        exit_price[short_exit] = short_stop[short_exit]

        exit_reason = pd.Series("none", index=df.index)
        exit_reason[long_exit] = "stop_loss_long"
        exit_reason[short_exit] = "stop_loss_short"

        return ExitSignal(
            exit_long=long_exit,
            exit_short=short_exit,
            exit_price=exit_price,
            exit_reason=exit_reason
        )


class TakeProfitExit:
    """Take profit exit."""

    def __init__(self, target_ticks: int = 40, tick_size: float = 0.25):
        self.target_ticks = target_ticks
        self.tick_size = tick_size

    def generate(self, bars: pd.DataFrame, entry_price: pd.Series, position: pd.Series) -> ExitSignal:
        df = bars.copy()
        df["entry_price"] = entry_price
        df["position"] = position

        # Long take profit
        long_target = df["entry_price"] + self.target_ticks * self.tick_size
        long_exit = (df["position"] == 1) & (df["high"] >= long_target)

        # Short take profit
        short_target = df["entry_price"] - self.target_ticks * self.tick_size
        short_exit = (df["position"] == -1) & (df["low"] <= short_target)

        exit_price = pd.Series(np.nan, index=df.index)
        exit_price[long_exit] = long_target[long_exit]
        exit_price[short_exit] = short_target[short_exit]

        exit_reason = pd.Series("none", index=df.index)
        exit_reason[long_exit] = "take_profit_long"
        exit_reason[short_exit] = "take_profit_short"

        return ExitSignal(
            exit_long=long_exit,
            exit_short=short_exit,
            exit_price=exit_price,
            exit_reason=exit_reason
        )


class TrailingStopExit:
    """Trailing stop exit."""

    def __init__(self, trail_ticks: int = 10, tick_size: float = 0.25):
        self.trail_ticks = trail_ticks
        self.tick_size = tick_size

    def generate(self, bars: pd.DataFrame, entry_price: pd.Series, position: pd.Series) -> ExitSignal:
        df = bars.copy()
        df["entry_price"] = entry_price
        df["position"] = position

        # Calculate trailing stop levels
        trail_distance = self.trail_ticks * self.tick_size

        # For long: trail below highest high since entry
        # For short: trail above lowest low since entry
        long_trail = pd.Series(np.nan, index=df.index)
        short_trail = pd.Series(np.nan, index=df.index)

        in_long = False
        in_short = False
        highest_high = 0
        lowest_low = np.inf

        for i in range(len(df)):
            if df["position"].iloc[i] == 1 and not in_long:
                in_long = True
                in_short = False
                highest_high = df["high"].iloc[i]
            elif df["position"].iloc[i] == -1 and not in_short:
                in_short = True
                in_long = False
                lowest_low = df["low"].iloc[i]
            elif df["position"].iloc[i] == 0:
                in_long = False
                in_short = False

            if in_long:
                highest_high = max(highest_high, df["high"].iloc[i])
                long_trail.iloc[i] = highest_high - trail_distance
            if in_short:
                lowest_low = min(lowest_low, df["low"].iloc[i])
                short_trail.iloc[i] = lowest_low + trail_distance

        # Exit when price hits trail
        long_exit = (df["position"] == 1) & (df["low"] <= long_trail)
        short_exit = (df["position"] == -1) & (df["high"] >= short_trail)

        exit_price = pd.Series(np.nan, index=df.index)
        exit_price[long_exit] = long_trail[long_exit]
        exit_price[short_exit] = short_trail[short_exit]

        exit_reason = pd.Series("none", index=df.index)
        exit_reason[long_exit] = "trailing_stop_long"
        exit_reason[short_exit] = "trailing_stop_short"

        return ExitSignal(
            exit_long=long_exit,
            exit_short=short_exit,
            exit_price=exit_price,
            exit_reason=exit_reason
        )


class TimeExit:
    """Time-based exit (e.g., end of session)."""

    def __init__(self, exit_hour: int = 15, exit_minute: int = 55, timezone: str = "America/New_York"):
        self.exit_hour = exit_hour
        self.exit_minute = exit_minute
        self.timezone = timezone

    def generate(self, bars: pd.DataFrame, position: pd.Series) -> ExitSignal:
        df = bars.copy()
        df["position"] = position

        # Session end time
        df["is_exit_time"] = (
            (df["timestamp"].dt.tz_convert(self.timezone).dt.hour == self.exit_hour) &
            (df["timestamp"].dt.tz_convert(self.timezone).dt.minute >= self.exit_minute)
        )

        long_exit = (df["position"] == 1) & df["is_exit_time"]
        short_exit = (df["position"] == -1) & df["is_exit_time"]

        exit_price = pd.Series(np.nan, index=df.index)
        exit_price[long_exit] = df.loc[long_exit, "close"]
        exit_price[short_exit] = df.loc[short_exit, "close"]

        exit_reason = pd.Series("none", index=df.index)
        exit_reason[long_exit] = "time_exit_long"
        exit_reason[short_exit] = "time_exit_short"

        return ExitSignal(
            exit_long=long_exit,
            exit_short=short_exit,
            exit_price=exit_price,
            exit_reason=exit_reason
        )


class RegimeChangeExit:
    """Exit on regime change."""

    def __init__(self, regime_classifier):
        self.regime_classifier = regime_classifier

    def generate(self, bars: pd.DataFrame, position: pd.Series, regime_result) -> ExitSignal:
        df = bars.copy()
        df["position"] = position

        # Exit long if regime turns bearish
        bias = regime_result.bias if hasattr(regime_result, "bias") else pd.Series("neutral", index=df.index)

        long_exit = (df["position"] == 1) & (bias == "bearish")
        short_exit = (df["position"] == -1) & (bias == "bullish")

        exit_price = pd.Series(np.nan, index=df.index)
        exit_price[long_exit] = df.loc[long_exit, "close"]
        exit_price[short_exit] = df.loc[short_exit, "close"]

        exit_reason = pd.Series("none", index=df.index)
        exit_reason[long_exit] = "regime_change_long"
        exit_reason[short_exit] = "regime_change_short"

        return ExitSignal(
            exit_long=long_exit,
            exit_short=short_exit,
            exit_price=exit_price,
            exit_reason=exit_reason
        )


def combine_exits(*exits: ExitSignal) -> ExitSignal:
    """Combine multiple exit signals (first exit wins)."""
    if not exits:
        return ExitSignal(
            exit_long=pd.Series(dtype=bool),
            exit_short=pd.Series(dtype=bool),
            exit_price=pd.Series(dtype=float),
            exit_reason=pd.Series(dtype=str)
        )

    combined_long = exits[0].exit_long.copy()
    combined_short = exits[0].exit_short.copy()
    combined_price = exits[0].exit_price.copy()
    combined_reason = exits[0].exit_reason.copy()

    for ext in exits[1:]:
        # Only fill where not already exited
        long_mask = ext.exit_long & ~combined_long
        short_mask = ext.exit_short & ~combined_short

        combined_long[long_mask] = True
        combined_short[short_mask] = True
        combined_price[long_mask] = ext.exit_price[long_mask]
        combined_price[short_mask] = ext.exit_price[short_mask]
        combined_reason[long_mask] = ext.exit_reason[long_mask]
        combined_reason[short_mask] = ext.exit_reason[short_mask]

    return ExitSignal(
        exit_long=combined_long,
        exit_short=combined_short,
        exit_price=combined_price,
        exit_reason=combined_reason
    )