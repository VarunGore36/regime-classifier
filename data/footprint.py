"""Footprint / Order Flow data loader."""

from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from .base import FileDataLoader


class FootprintLoader(FileDataLoader):
    """
    Footprint chart data loader.

    Expected columns (per price level per bar):
    - timestamp: datetime (bar open time)
    - price: float (price level)
    - bid_volume: int (volume traded at bid / aggressive sell)
    - ask_volume: int (volume traded at ask / aggressive buy)
    - delta: int (ask_volume - bid_volume)
    - symbol: str
    """

    REQUIRED_COLUMNS = ["timestamp", "price", "bid_volume", "ask_volume"]

    def __init__(
        self,
        data_path: str,
        timezone: str = "America/New_York",
        file_format: str = "parquet"
    ):
        super().__init__("FOOTPRINT", data_path, timezone, file_format)

    def load_footprint(
        self,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Load footprint data for date range."""
        df = self._load_file(self.data_path)
        df = self._standardize_footprint(df)

        df = df[(df["timestamp"] >= pd.Timestamp(start, tz=self.timezone)) &
                (df["timestamp"] <= pd.Timestamp(end, tz=self.timezone))]

        return df.sort_values(["timestamp", "price"]).reset_index(drop=True)

    def load_bar_footprint(
        self,
        timestamp: datetime
    ) -> pd.DataFrame:
        """Load footprint for a single bar."""
        df = self._load_file(self.data_path)
        df = self._standardize_footprint(df)

        ts = pd.Timestamp(timestamp, tz=self.timezone)
        df = df[df["timestamp"] == ts]

        return df.sort_values("price").reset_index(drop=True)

    def _standardize_footprint(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(self.timezone)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(self.timezone)

        # Calculate delta if not present
        if "delta" not in df.columns:
            df["delta"] = df["ask_volume"] - df["bid_volume"]

        df["total_volume"] = df["bid_volume"] + df["ask_volume"]
        return df

    def aggregate_to_bars(self, footprint: pd.DataFrame) -> pd.DataFrame:
        """Aggregate footprint to OHLCV bars."""
        bars = footprint.groupby("timestamp").agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("total_volume", "sum"),
            cvd=("delta", "sum"),  # Cumulative Volume Delta
            total_delta=("delta", "sum"),
            buy_volume=("ask_volume", "sum"),
            sell_volume=("bid_volume", "sum")
        ).reset_index()
        bars["symbol"] = self.symbol
        return bars

    def calculate_cvd(self, footprint: pd.DataFrame) -> pd.DataFrame:
        """Calculate Cumulative Volume Delta per bar."""
        cvd = footprint.groupby("timestamp")["delta"].sum().cumsum().reset_index()
        cvd.columns = ["timestamp", "cvd"]
        return cvd

    def detect_absorption(
        self,
        footprint: pd.DataFrame,
        min_delta_divergence: int = 50,
        max_price_move_ticks: int = 5,
        min_volume: int = 100,
        tick_size: float = 0.25
    ) -> pd.DataFrame:
        """
        Detect absorption: aggressive delta but price doesn't move.

        Returns DataFrame with absorption signals per bar.
        """
        bars = footprint.groupby("timestamp").agg(
            total_delta=("delta", "sum"),
            total_volume=("total_volume", "sum"),
            price_change=("price", lambda x: x.iloc[-1] - x.iloc[0]),
            high=("price", "max"),
            low=("price", "min")
        ).reset_index()

        bars["price_move_ticks"] = (bars["high"] - bars["low"]) / tick_size
        bars["absorption_bullish"] = (
            (bars["total_delta"] > min_delta_divergence) &
            (bars["price_move_ticks"] <= max_price_move_ticks) &
            (bars["total_volume"] >= min_volume) &
            (bars["price_change"] >= 0)
        )
        bars["absorption_bearish"] = (
            (bars["total_delta"] < -min_delta_divergence) &
            (bars["price_move_ticks"] <= max_price_move_ticks) &
            (bars["total_volume"] >= min_volume) &
            (bars["price_change"] <= 0)
        )

        return bars

    def detect_exhaustion(
        self,
        footprint: pd.DataFrame,
        delta_reversal_threshold: int = 100,
        volume_dryup_mult: float = 0.3,
        max_bars: int = 3,
        tick_size: float = 0.25
    ) -> pd.DataFrame:
        """
        Detect exhaustion: delta reverses, volume dries up.

        Returns DataFrame with exhaustion signals per bar.
        """
        bars = footprint.groupby("timestamp").agg(
            total_delta=("delta", "sum"),
            total_volume=("total_volume", "sum"),
            price_change=("price", lambda x: x.iloc[-1] - x.iloc[0]),
            high=("price", "max"),
            low=("price", "min")
        ).reset_index().sort_values("timestamp")

        bars["prev_delta"] = bars["total_delta"].shift(1)
        bars["prev_volume"] = bars["total_volume"].shift(1)
        bars["delta_reversal"] = bars["total_delta"] * bars["prev_delta"] < 0
        bars["volume_dryup"] = bars["total_volume"] < (bars["prev_volume"] * volume_dryup_mult)

        bars["exhaustion_bullish"] = (
            (bars["prev_delta"] < -delta_reversal_threshold) &
            (bars["total_delta"] > 0) &
            bars["volume_dryup"] &
            (bars["price_change"] >= 0)
        )
        bars["exhaustion_bearish"] = (
            (bars["prev_delta"] > delta_reversal_threshold) &
            (bars["total_delta"] < 0) &
            bars["volume_dryup"] &
            (bars["price_change"] <= 0)
        )

        return bars

    def detect_trapped_traders(
        self,
        footprint: pd.DataFrame,
        imbalance_ratio: float = 3.0,
        confirmation_bars: int = 2,
        tick_size: float = 0.25
    ) -> pd.DataFrame:
        """
        Detect trapped traders: large imbalance, price moves against them.

        Returns DataFrame with trapped trader signals per bar.
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

        # Trapped longs: large buy imbalance, next bars close lower
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

        return bars

    def calculate_delta_divergence(
        self,
        footprint: pd.DataFrame,
        lookback: int = 10
    ) -> pd.DataFrame:
        """
        Calculate delta divergence: price makes new high/low but delta doesn't confirm.
        """
        bars = self.aggregate_to_bars(footprint).sort_values("timestamp")

        bars["price_high"] = bars["high"].rolling(lookback).max()
        bars["price_low"] = bars["low"].rolling(lookback).min()
        bars["delta_high"] = bars["cvd"].rolling(lookback).max()
        bars["delta_low"] = bars["cvd"].rolling(lookback).min()

        # Bearish divergence: price higher high, delta lower high
        bars["bearish_divergence"] = (
            (bars["high"] > bars["price_high"].shift(1)) &
            (bars["cvd"] < bars["delta_high"].shift(1))
        )

        # Bullish divergence: price lower low, delta higher low
        bars["bullish_divergence"] = (
            (bars["low"] < bars["price_low"].shift(1)) &
            (bars["cvd"] > bars["delta_low"].shift(1))
        )

        return bars